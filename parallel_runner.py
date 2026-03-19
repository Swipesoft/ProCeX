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

    # Classify scenes by their initial strategy
    manim_scenes  = [s for s in state.scenes if s.visual_strategy in
                     (VisualStrategy.MANIM, VisualStrategy.TEXT_ANIMATION)]
    image_scenes  = [s for s in state.scenes if s.visual_strategy in
                     (VisualStrategy.IMAGE_GEN,)]

    total_scenes  = [len(state.scenes)]
    print(f"[Parallel] Stage B: {len(manim_scenes)} Manim, {len(image_scenes)} image scenes")

    # ── Queues ────────────────────────────────────────────────────────────
    code_queue   = Queue()   # SceneTask → coder workers
    render_queue = Queue()   # SceneTask → renderer workers
    results      = {}        # scene.id → clip_path  (written once per scene)
    done_count   = [0]
    done_lock    = threading.Lock()

    def log(msg: str):
        print(f"[Parallel] {msg}")

    def mark_done(scene: Scene, clip_path: str | None):
        """Record a scene as finished (success or fallback). Thread-safe."""
        if clip_path:
            scene.clip_path    = clip_path
            results[scene.id]  = clip_path
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
            # Reroute: Critic saw the peak-density frame and decided the scene
            # needs structural re-planning, not just positional patching.
            # We call VisualDirector.reroute_scene() with the frame — it decides
            # PATH A (revise layout) or PATH B (split into beats) freely.
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
                    # Increment reroute counter before re-queuing
                    updated = dataclasses.replace(
                        updated,
                        critic_reroute_attempts = reroute_attempts + 1,
                        clip_path = "",   # clear stale clip
                    )
                    # If VisualDirector chose PATH B, expand subscenes first
                    reroute_beats = getattr(updated, "_reroute_beats", None)
                    if reroute_beats:
                        log(f"Renderer-{wid}: Scene {scene.id} — reroute PATH B: "
                            f"expanding into {len(reroute_beats)} subscenes")
                        expanded = director._expand_subscenes(
                            [updated], [(updated.id, reroute_beats)]
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

    # ── Write results back to state ───────────────────────────────────────
    state.rendered_clips = []
    for scene in state.scenes:
        if scene.id in results:
            scene.clip_path = results[scene.id]
            state.rendered_clips.append(results[scene.id])

    elapsed = time.time() - start
    log(
        f"Stage B complete: {len(state.rendered_clips)}/{total_scenes[0]} clips "
        f"in {elapsed/60:.1f} min"
    )
    return state