"""
parallel_runner.py
Queue-based parallel execution for ProcEx pipeline stages 3-5.

Stage A — TTSAgent ∥ VisualDirector:
  Each agent gets a deep copy of state to work on independently.
  Results merged back onto the shared state by scene ID after both finish.
  No data race possible — agents never share the same objects.

Stage B — ManimCoder + ImageGenAgent + RendererAgent (queue pipeline):

  Queue architecture:

    MANIM / TEXT_ANIMATION scenes:
      code_queue → [coder workers] → render_queue

    IMAGE_GEN scenes:
      image_queue → [image workers] ──────────────────→ render_queue
                         └── (on failure, retry N times)
                         └── (all retries fail) → degrade to TEXT_ANIMATION → code_queue

    IMAGE_MANIM_HYBRID scenes (two-phase):
      image_queue → [image workers] → code_queue → [coder workers] → render_queue

    render_queue → [renderer workers]:
      success          → done
      fail + retry < max → code_queue (with error, Manim only)
      fail + retry < max → image retry (IMAGE_GEN only)
      fail, retries gone → emergency fallback → done

  Each scene is owned by exactly one queue at a time.
  No locks needed — ownership transfers atomically via queue.put().
  Renderer workers exit only via sentinel, sent after all scenes complete.
"""
from __future__ import annotations
import copy
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from queue import Queue, Empty

from state import ProcExState, Scene, VisualStrategy
from config import ProcExConfig
from utils.llm_client import LLMClient


_DONE          = object()   # sentinel — distinct from None
_QUEUE_TIMEOUT = 0.25       # seconds to block on empty queue before re-checking


@dataclass
class SceneTask:
    """Unit of work carried through the code/render queue pipeline."""
    scene:         Scene
    attempt:       int = 0   # render failures so far
    render_error:  str = ""  # traceback from last Manim crash (for regen prompt)
    image_attempt: int = 0   # image generation retries so far


# ══════════════════════════════════════════════════════════════════════════════
# Stage A — TTSAgent ∥ VisualDirector
# ══════════════════════════════════════════════════════════════════════════════

def run_tts_and_director_parallel(
    state: ProcExState,
    cfg:   ProcExConfig,
    llm:   LLMClient,
) -> ProcExState:
    """
    Run TTSAgent and VisualDirector simultaneously on independent deep copies
    of state. Merges both result sets back onto the original by scene ID.

    Fix for Problem 1: deep copy eliminates the data race — the two agents
    never share Python objects, so concurrent attribute writes cannot corrupt
    each other regardless of GIL scheduling.
    """
    from agents.tts_agent       import TTSAgent
    from agents.visual_director import VisualDirector

    # Give each agent its own isolated copy of state to mutate freely
    state_tts = copy.deepcopy(state)
    state_dir = copy.deepcopy(state)

    tts_result = [None]
    dir_result = [None]
    errors     = []

    def run_tts():
        try:
            tts_result[0] = TTSAgent(cfg, llm).run(state_tts)
        except Exception as e:
            errors.append(("TTSAgent", e))

    def run_dir():
        try:
            dir_result[0] = VisualDirector(cfg, llm).run(state_dir)
        except Exception as e:
            errors.append(("VisualDirector", e))

    print("[Parallel] ▶ Stage A: TTSAgent ∥ VisualDirector...")
    t1 = threading.Thread(target=run_tts, name="TTSAgent")
    t2 = threading.Thread(target=run_dir, name="VisualDirector")
    t1.start(); t2.start()
    try:
        t1.join(); t2.join()
    except KeyboardInterrupt:
        print("[Parallel] ⚠ Stage A interrupted — waiting for threads...")
        t1.join(timeout=5); t2.join(timeout=5)
        raise

    # Report any errors
    for agent_name, exc in errors:
        print(f"[Parallel] ERROR {agent_name}: {exc}")
        import traceback
        traceback.print_exc()
    tts_errors = [e for a, e in errors if a == "TTSAgent"]
    dir_errors = [e for a, e in errors if a == "VisualDirector"]
    if tts_errors:
        raise RuntimeError(f"TTSAgent failed: {tts_errors[0]}")
    if dir_errors:
        print(f"[Parallel] ⚠ VisualDirector failed: {dir_errors[0]} — using default MANIM for all scenes")

    # ── Merge TTS results onto original state ─────────────────────────────
    if tts_result[0]:
        r = tts_result[0]
        state.audio_path           = r.audio_path
        state.all_timestamps       = r.all_timestamps
        state.total_audio_duration = r.total_audio_duration
        _merge_scene_fields(state, r.scenes, [
            "duration_seconds", "tts_audio_path", "tts_duration", "timestamps",
        ])

    # ── Merge VisualDirector results onto original state ──────────────────
    if dir_result[0]:
        dir_scenes = dir_result[0].scenes
        print(f"[Parallel] Director returned {len(dir_scenes)} scenes "
              f"(original: {len(state.scenes)})")

        # Check if subscene expansion happened (more scenes returned than started)
        if len(dir_scenes) > len(state.scenes):
            # Replace state.scenes with the expanded list entirely.
            # Then back-fill TTS fields from the original scenes by matching
            # parent_scene_id (for subscene beats) or scene.id (for unchanged scenes).
            tts_by_id = {s.id: s for s in tts_result[0].scenes} if tts_result[0] else {}

            for scene in dir_scenes:
                # Use parent_scene_id to find TTS data for subscene beats
                tts_source_id = scene.parent_scene_id if scene.parent_scene_id else scene.id
                tts_src = tts_by_id.get(tts_source_id)
                if tts_src:
                    scene.tts_audio_path = tts_src.tts_audio_path
                    scene.tts_duration   = scene.duration_seconds  # proportional slice
                    # Timestamps already sliced by _expand_subscenes

            state.scenes = dir_scenes
            print(f"[Parallel] VisualDirector expanded {len(tts_by_id)} → "
                  f"{len(dir_scenes)} scenes (subscene beats)")
        else:
            # No expansion — simple field merge
            _merge_scene_fields(state, dir_scenes, [
                "visual_strategy", "visual_prompt", "visual_reasoning",
                "needs_labels", "label_list", "element_count", "zone_allocation",
                "parent_scene_id", "subscene_index",
            ])

    print("[Parallel] ✓ Stage A complete")
    return state


def _merge_scene_fields(
    state: ProcExState,
    source_scenes: list,
    fields: list[str],
) -> None:
    """Copy named fields from source_scenes onto state.scenes, matched by id."""
    by_id = {s.id: s for s in source_scenes}
    for scene in state.scenes:
        src = by_id.get(scene.id)
        if src:
            for f in fields:
                if hasattr(src, f):
                    setattr(scene, f, getattr(src, f))


# ══════════════════════════════════════════════════════════════════════════════
# Stage B — ManimCoder + ImageGenAgent + RendererAgent (queue pipeline)
# ══════════════════════════════════════════════════════════════════════════════

def run_generation_render_pipeline(
    state: ProcExState,
    cfg:   ProcExConfig,
    llm:   LLMClient,
) -> ProcExState:
    from agents.manim_coder     import ManimCoder, _fallback_scene
    from agents.image_gen_agent import ImageGenAgent
    from agents.renderer        import RendererAgent, REGEN_RETRIES
    from agents.vlm_critic      import VLMCritic, MAX_REROUTE_ATTEMPTS
    from agents.visual_director import VisualDirector
    from config                 import RESOLUTIONS

    manim_dir  = cfg.dirs["manim"]
    scenes_dir = cfg.dirs["scenes"]
    images_dir = cfg.dirs["images"]
    for d in (manim_dir, scenes_dir, images_dir):
        os.makedirs(d, exist_ok=True)

    coder    = ManimCoder(cfg, llm)
    imager   = ImageGenAgent(cfg, llm)
    renderer = RendererAgent(cfg, llm)
    critic   = VLMCritic(cfg, llm)
    director = VisualDirector(cfg, llm)

    # ── VIDEO_GEN budget enforcement ─────────────────────────────────────
    from agents.video_gen_agent import VideoGenAgent, DEFAULT_CLIP_SECS
    videogen = VideoGenAgent(cfg, llm)
    state    = VideoGenAgent.enforce_budget(state)

    # Classify scenes by their initial strategy
    manim_scenes  = [s for s in state.scenes if s.visual_strategy in
                     (VisualStrategy.MANIM, VisualStrategy.TEXT_ANIMATION)]
    image_scenes  = [s for s in state.scenes if s.visual_strategy in
                     (VisualStrategy.IMAGE_GEN,)]
    video_scenes  = [s for s in state.scenes if s.visual_strategy ==
                     VisualStrategy.VIDEO_GEN]

    total_scenes  = [len(state.scenes)]
    print(
        f"[Parallel] Stage B: {len(manim_scenes)} Manim, "
        f"{len(image_scenes)} image, {len(video_scenes)} video scenes"
    )

    # ── Queues ────────────────────────────────────────────────────────────
    code_queue   = Queue()   # SceneTask → coder workers
    render_queue = Queue()   # SceneTask → renderer workers
    results      = {}        # scene.id → clip_path  (written once per scene)
    done_count   = [0]
    done_scenes  = {}       # id → Scene, tracks ALL scenes including reroute subscenes
    done_lock    = threading.Lock()

    def log(msg: str):
        print(f"[Parallel] {msg}")

    def mark_done(scene: Scene, clip_path: str | None):
        """Record a scene as finished (success or fallback). Thread-safe."""
        if clip_path:
            scene.clip_path    = clip_path
            results[scene.id]  = clip_path
            done_scenes[scene.id] = scene   # ← store object for state rebuild
        else:
            state.log_error("RendererAgent", f"Scene {scene.id}: all fallbacks failed")
        with done_lock:
            done_count[0] += 1

    res    = RESOLUTIONS.get(state.resolution, RESOLUTIONS["1080p"])
    aspect = res.aspect_ratio

    # ── Shutdown event — must be created BEFORE workers are defined ───────
    _shutdown = threading.Event()

    # ── Coder worker ──────────────────────────────────────────────────────

    def coder_worker(wid: int):
        while not _shutdown.is_set():
            try:
                task = code_queue.get(timeout=_QUEUE_TIMEOUT)
            except Empty:
                continue
            if task is _DONE:
                code_queue.put(_DONE)   # fan-out to sibling workers
                break

            scene     = task.scene
            log(f"Coder-{wid}: Scene {scene.id} "
                f"(render_attempt={task.attempt} regen={bool(task.render_error)}"
                f")")

            # Set file paths if not yet assigned
            if not scene.manim_class_name:
                scene.manim_class_name = f"Scene{scene.id:02d}"
            if not scene.manim_file_path:
                scene.manim_file_path  = os.path.join(manim_dir, f"scene_{scene.id:02d}.py")

            try:
                code = coder._generate_scene_code(
                    scene, state.skill_pack,
                    initial_error = task.render_error,
                    res           = res,
                    aspect        = aspect,
                )
                coder._write_scene_file(scene.manim_file_path, code, res=res)
            except Exception as e:
                log(f"Coder-{wid}: Scene {scene.id} generation exception: {e} — using fallback")
                coder._write_scene_file(
                    scene.manim_file_path,
                    _fallback_scene(scene.manim_class_name, scene),
                    res=res,
                )

            log(f"Coder-{wid}: Scene {scene.id} code ready → render queue")
            render_queue.put(SceneTask(
                scene        = scene,
                attempt      = task.attempt,
                render_error = task.render_error,
            ))

    # ── Image worker ──────────────────────────────────────────────────────

    def image_worker(wid: int, scene: Scene):
        """
        Generate image for one IMAGE_GEN or HYBRID scene.

        Fix for Problems 2 & 3:
          - Retries up to cfg.max_llm_retries times on failure
          - On all retries exhausted: degrades IMAGE_GEN → TEXT_ANIMATION (routes
            to code_queue) rather than sending empty scene to render_queue
          - HYBRID scenes always route to code_queue after image gen (Problem 4)
        """
        log(f"Image-{wid}: Scene {scene.id} ({scene.visual_strategy.value}) generating...")

        succeeded = False
        last_err  = ""

        for attempt in range(1, cfg.max_llm_retries + 1):
            try:
                imager._generate_for_scene(scene, state.resolution, images_dir)
                succeeded = True
                log(f"Image-{wid}: Scene {scene.id} image OK (attempt {attempt})")
                break
            except Exception as e:
                last_err = str(e)
                log(f"Image-{wid}: Scene {scene.id} image attempt {attempt} failed: {e}")
                if attempt < cfg.max_llm_retries:
                    time.sleep(1.5 * attempt)   # brief back-off before retry

        if not succeeded:
            # All image generation attempts failed
            log(f"Image-{wid}: Scene {scene.id} — all image attempts failed. "
                f"Degrading to TEXT_ANIMATION.")
            scene.visual_strategy = VisualStrategy.TEXT_ANIMATION
            state.log_error(
                "ImageGenAgent",
                f"Scene {scene.id}: image gen failed after {cfg.max_llm_retries} "
                f"attempts ({last_err}) — degraded to TEXT_ANIMATION"
            )
            # Route to coder for a text animation fallback
            code_queue.put(SceneTask(scene=scene))
            return

        # IMAGE_GEN success → render directly
        if scene.visual_strategy == VisualStrategy.IMAGE_GEN:
            log(f"Image-{wid}: Scene {scene.id} IMAGE_GEN → render queue")
            render_queue.put(SceneTask(scene=scene))

        # IMAGE_MANIM_HYBRID retired — IMAGE_GEN handles all annotations

    # ── Video gen worker ──────────────────────────────────────────────────
    # Runs in a daemon thread per scene — polls Novita until done or timeout.
    # On failure degrades to IMAGE_GEN so the scene always produces a frame.

    def video_worker(scene: Scene):
        """Generate one VIDEO_GEN scene via Novita Seedance 1.5 Pro."""
        log(f"Video: Scene {scene.id} VIDEO_GEN generating...")
        videos_dir = cfg.dirs["videos"]
        try:
            clip_path = videogen.generate_for_scene(
                scene      = scene,
                resolution = state.resolution,
                videos_dir = videos_dir,
            )
        except Exception as e:
            clip_path = None
            log(f"Video: Scene {scene.id} generation error: {e}")

        if clip_path and os.path.exists(clip_path):
            scene.clip_path  = clip_path
            scene.video_path = clip_path
            log(f"Video: Scene {scene.id} ✓ → {os.path.basename(clip_path)}")
            mark_done(scene, clip_path)
        else:
            # Degrade to IMAGE_GEN — dispatch to image_worker
            log(f"Video: Scene {scene.id} failed — degrading to IMAGE_GEN")
            scene.visual_strategy = VisualStrategy.IMAGE_GEN
            with done_lock:
                # image_worker will call mark_done when it completes
                pass
            image_worker(1, scene)

    # ── Renderer worker ───────────────────────────────────────────────────

    def renderer_worker(wid: int):
        while not _shutdown.is_set():
            try:
                task = render_queue.get(timeout=_QUEUE_TIMEOUT)
            except Empty:
                continue
            if task is _DONE:
                render_queue.put(_DONE)
                break

            scene = task.scene
            log(f"Renderer-{wid}: Scene {scene.id} rendering (attempt {task.attempt+1})...")

            clip_path, error, critic_result = renderer.render_with_critic(
                scene, state.resolution, scenes_dir, critic=critic
            )

            # ── Critic reroute ────────────────────────────────────────────────
            if (critic_result and critic_result.status == "reroute"
                    and critic_result.reroute_frame is not None):
                reroute_attempts = getattr(scene, "critic_reroute_attempts", 0) or 0
                log(
                    f"Renderer-{wid}: Scene {scene.id} — Critic rerouting to "
                    f"VisualDirector (attempt {reroute_attempts+1}/{MAX_REROUTE_ATTEMPTS})"
                )
                try:
                    import dataclasses
                    updated = director.reroute_scene(
                        scene, critic_result.reroute_frame, aspect=getattr(
                            RESOLUTIONS.get(state.resolution,
                            RESOLUTIONS["1080p"]), "aspect_ratio", "16:9"
                        )
                    )
                    # Grab beats BEFORE dataclasses.replace() — it only copies
                    # declared dataclass fields, so _reroute_beats (set via
                    # object.__setattr__) would be silently dropped otherwise.
                    reroute_beats = getattr(updated, "_reroute_beats", None)

                    # Increment reroute counter before re-queuing
                    updated = dataclasses.replace(
                        updated,
                        critic_reroute_attempts = reroute_attempts + 1,
                        clip_path = "",   # clear stale clip
                    )
                    # If VisualDirector chose PATH B, expand subscenes first
                    if reroute_beats:
                        log(f"Renderer-{wid}: Scene {scene.id} — reroute PATH B: "
                            f"expanding into {len(reroute_beats)} subscenes")
                        expanded = director._expand_subscenes(
                            [updated], [(updated, reroute_beats)]
                        )
                        for sub in expanded:
                            code_queue.put(SceneTask(scene=sub, attempt=0))
                        # Remove original from done tracking, add subscenes
                        with done_lock:
                            total_scenes[0] += len(expanded) - 1
                    else:
                        # PATH A — re-queue updated scene for fresh render
                        code_queue.put(SceneTask(scene=updated, attempt=0))
                    continue   # don't mark_done — new render is pending
                except Exception as e:
                    log(f"Renderer-{wid}: Scene {scene.id} — reroute failed ({e}), "
                        f"keeping original clip")
                    mark_done(scene, clip_path)
                    continue

            # ── Critic split_needed ───────────────────────────────────────────
            # Reroute budget exhausted — force PATH B split via VisualDirector.
            # VisualDirector.reroute_scene(force_split=True) only decides HOW
            # to split; the Critic already decided THAT it must split.
            if (critic_result and critic_result.status == "split_needed"
                    and critic_result.reroute_frame is not None):
                log(
                    f"Renderer-{wid}: Scene {scene.id} — split_needed: "
                    f"forcing PATH B via VisualDirector"
                )
                try:
                    import dataclasses
                    aspect = getattr(
                        RESOLUTIONS.get(state.resolution, RESOLUTIONS["1080p"]),
                        "aspect_ratio", "16:9"
                    )
                    updated = director.reroute_scene(
                        scene, critic_result.reroute_frame,
                        aspect=aspect, force_split=True
                    )
                    reroute_beats = getattr(updated, "_reroute_beats", None)
                    if reroute_beats:
                        log(f"Renderer-{wid}: Scene {scene.id} — forced split: "
                            f"expanding into {len(reroute_beats)} subscenes")
                        expanded = director._expand_subscenes(
                            [updated], [(updated, reroute_beats)]
                        )
                        for sub in expanded:
                            code_queue.put(SceneTask(scene=sub, attempt=0))
                        with done_lock:
                            total_scenes[0] += len(expanded) - 1
                        continue   # don't mark_done — subscenes pending
                    else:
                        # VisualDirector failed to produce beats — keep original
                        log(f"Renderer-{wid}: Scene {scene.id} — forced split "
                            f"produced no beats, keeping original clip")
                        mark_done(scene, clip_path)
                        continue
                except Exception as e:
                    log(f"Renderer-{wid}: Scene {scene.id} — forced split failed ({e}), "
                        f"keeping original clip")
                    mark_done(scene, clip_path)
                    continue

            # ── Critic imagegen_fallback ──────────────────────────────────────
            # Scene is at split_depth>=1 — further splitting would violate N≤K≤2N.
            # Convert to IMAGE_GEN and enrich the prompt with chain context so
            # the image generator knows where this scene fits in the video.
            if critic_result and critic_result.status == "imagegen_fallback":
                log(
                    f"Renderer-{wid}: Scene {scene.id} — depth limit reached, "
                    f"converting to ImageGen fallback"
                )
                try:
                    from agents.image_reprompter import ImageReprompter

                    # Bug fix 1: clear stale Manim clip_path so the old
                    # failed render is never assembled into the final video.
                    scene.clip_path = ""

                    # Bug fix 2: use tts_duration as the authoritative clip
                    # length for ken-burns so ImageGen clips match audio exactly.
                    if scene.tts_duration and scene.tts_duration > 0:
                        scene.duration_seconds = scene.tts_duration

                    # Build deeply enriched prompt via ImageReprompter
                    reprompter = ImageReprompter(cfg, llm)
                    scene = reprompter.reprompt(scene, state)

                    scene.visual_strategy = VisualStrategy.IMAGE_GEN
                    scene.needs_labels    = len(scene.label_list) > 0

                    # Dispatch to image_worker (runs in its own thread)
                    import threading
                    t = threading.Thread(
                        target = image_worker,
                        args   = (wid, scene),
                        name   = f"ImageFallback-{scene.id}",
                    )
                    t.start()
                    continue   # renderer_worker moves on; image_worker handles done
                except Exception as e:
                    log(f"Renderer-{wid}: Scene {scene.id} — imagegen fallback setup failed ({e}), "
                        f"keeping original clip")
                    mark_done(scene, clip_path)
                    continue

            if clip_path:
                log(f"Renderer-{wid}: Scene {scene.id} ✓")
                mark_done(scene, clip_path)

            elif task.attempt < REGEN_RETRIES:
                next_attempt     = task.attempt + 1
                next_img_attempt = getattr(scene, "_img_attempt", 0) + 1
                log(f"Renderer-{wid}: Scene {scene.id} failed "
                    f"({next_attempt}/{REGEN_RETRIES+1}) — requeueing")

                if scene.visual_strategy in (VisualStrategy.MANIM,
                                             VisualStrategy.TEXT_ANIMATION):
                    code_queue.put(SceneTask(
                        scene        = scene,
                        attempt      = next_attempt,
                        render_error = coder._summarise_render_error(error or ""),
                    ))
                else:
                    # IMAGE_GEN render fail — cap image retries to avoid infinite loop
                    if next_img_attempt <= cfg.max_llm_retries:
                        log(f"Renderer-{wid}: Scene {scene.id} IMAGE_GEN render fail "
                            f"(error: {str(error)[:200]}) "
                            f"— re-generating image (img_attempt={next_img_attempt})")
                        scene._img_attempt = next_img_attempt  # track on scene object
                        t = threading.Thread(
                            target=image_worker,
                            args=(wid, scene),
                            name=f"ImageRetry-{scene.id}",
                        )
                        t.start()
                    else:
                        log(f"Renderer-{wid}: Scene {scene.id} — image retries exhausted, "
                            f"degrading to TEXT_ANIMATION")
                        scene.visual_strategy = VisualStrategy.TEXT_ANIMATION
                        code_queue.put(SceneTask(scene=scene, attempt=next_attempt))

            else:
                log(f"Renderer-{wid}: Scene {scene.id} — emergency fallback")
                fb = renderer.render_emergency(scene, state.resolution, scenes_dir)
                mark_done(scene, fb)


    # ── Seed queues ───────────────────────────────────────────────────────

    for scene in manim_scenes:
        code_queue.put(SceneTask(scene=scene))

    # ── Dispatch VIDEO_GEN scenes in background threads ───────────────────
    # Each video generation is a long async poll — run in parallel with coders
    from agents.video_gen_agent import DEFAULT_CLIP_SECS
    for s in video_scenes:
        t = threading.Thread(
            target=video_worker, args=(s,), name=f"Video-{s.id}", daemon=True
        )
        t.start()

    # ── Launch coder workers ──────────────────────────────────────────────

    n_coders    = max(1, min(len(state.scenes), cfg.coder_workers))
    n_renderers = max(1, min(total_scenes[0], cfg.render_workers))
    threads     = []

    for i in range(n_coders):
        t = threading.Thread(target=coder_worker, args=(i+1,), name=f"Coder-{i+1}")
        t.start()
        threads.append(t)

    # ── Launch image workers (parallel via ThreadPoolExecutor) ────────────

    n_images = max(1, min(len(image_scenes), cfg.image_workers))
    if image_scenes:
        with ThreadPoolExecutor(max_workers=n_images, thread_name_prefix="Image") as pool:
            futs = [pool.submit(image_worker, i+1, s) for i, s in enumerate(image_scenes)]
            for f in as_completed(futs):
                exc = f.exception()
                if exc:
                    log(f"Image worker raised uncaught exception: {exc}")

    # ── Launch renderer workers ───────────────────────────────────────────

    for i in range(n_renderers):
        t = threading.Thread(target=renderer_worker, args=(i+1,), name=f"Renderer-{i+1}")
        t.start()
        threads.append(t)

    # ── Monitor progress — main thread waits here ─────────────────────────

    log(f"Workers: {n_coders} coders, {n_images} image, {n_renderers} renderers")
    start    = time.time()
    last_log = 0

    try:
        while True:
            with done_lock:
                n_done = done_count[0]
            if n_done >= total_scenes[0]:
                break
            now = time.time()
            if now - last_log >= 30:
                log(f"Progress: {n_done}/{total_scenes[0]} done ({(now-start)/60:.1f} min elapsed)")
                last_log = now
            time.sleep(0.5)
    except KeyboardInterrupt:
        log("⚠ Interrupted — signalling workers to stop...")
        _shutdown.set()
        code_queue.put(_DONE)
        render_queue.put(_DONE)
        for t in threads:
            t.join(timeout=5)
        raise   # propagate up to main.py handler which saves checkpoint

    # ── Shut down workers ─────────────────────────────────────────────────
    # Send sentinels AFTER all work is confirmed done — fixes Problem 5
    code_queue.put(_DONE)
    render_queue.put(_DONE)
    for t in threads:
        t.join(timeout=10)

    # ── Write results back to state ─────────────────────────────────────────
    # Rebuild state.scenes from done_scenes to include all reroute/split
    # subscenes that were created at runtime and never added to state.scenes.
    # Sort by id so scenes play in correct order. Original scenes without
    # clips (rerouted parents) are excluded — their subscenes replace them.
    if done_scenes:
        def _ancestry_key(scene_id: int):
            """
            Decode a scene id into its ancestral path tuple so scenes sort in
            narrative order regardless of numeric magnitude.
            id=4     → (4,)       id=302   → (3, 2)
            id=30101 → (3, 1, 1)  id=30102 → (3, 1, 2)
            Plain numeric sort puts 4 before 302 before 30101 — wrong.
            Ancestry sort puts them as (3,1,1), (3,1,2), (3,2), (4,) — correct.
            """
            path = []
            while scene_id >= 100:
                path.append(scene_id % 100)
                scene_id = scene_id // 100
            path.append(scene_id)
            return tuple(reversed(path))

        all_done = sorted(done_scenes.values(), key=lambda s: _ancestry_key(s.id))
        # Only keep scenes that have a clip — discard rerouted parents
        state.scenes = [s for s in all_done if s.clip_path]
        # Also include original scenes that weren't rerouted but somehow
        # missed done_scenes (edge case: fallback with no clip_path)
        done_ids = {s.id for s in state.scenes}
        for s in sorted(
            (orig for orig in state.scenes if orig.id not in done_ids),
            key=lambda s: s.id
        ):
            pass  # already filtered above

    state.rendered_clips = [s.clip_path for s in state.scenes if s.clip_path]

    elapsed = time.time() - start
    log(
        f"Stage B complete: {len(state.rendered_clips)}/{total_scenes[0]} clips "
        f"in {elapsed/60:.1f} min"
    )
    return state