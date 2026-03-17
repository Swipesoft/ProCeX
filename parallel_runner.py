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

    # Report any errors — TTS failure is fatal (no timestamps = no anchors)
    for agent_name, exc in errors:
        print(f"[Parallel] ERROR {agent_name}: {exc}")
    tts_errors = [e for a, e in errors if a == "TTSAgent"]
    if tts_errors:
        raise RuntimeError(f"TTSAgent failed: {tts_errors[0]}")

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
        _merge_scene_fields(state, dir_result[0].scenes, [
            "visual_strategy", "visual_prompt", "visual_reasoning",
            "needs_labels", "label_list",
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
    from agents.vlm_critic      import VLMCritic

    manim_dir  = cfg.dirs["manim"]
    scenes_dir = cfg.dirs["scenes"]
    images_dir = cfg.dirs["images"]
    for d in (manim_dir, scenes_dir, images_dir):
        os.makedirs(d, exist_ok=True)

    coder    = ManimCoder(cfg, llm)
    imager   = ImageGenAgent(cfg, llm)
    renderer = RendererAgent(cfg, llm)
    critic   = VLMCritic(cfg, llm)

    # Classify scenes by their initial strategy
    manim_scenes  = [s for s in state.scenes if s.visual_strategy in
                     (VisualStrategy.MANIM, VisualStrategy.TEXT_ANIMATION)]
    image_scenes  = [s for s in state.scenes if s.visual_strategy in
                     (VisualStrategy.IMAGE_GEN,)]

    total_scenes  = len(state.scenes)
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

    from config import RESOLUTIONS
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
        """
        Fix for Problem 5: renderer workers exit ONLY via sentinel, never by
        checking done_count. The main thread sends the sentinel after all scenes
        are confirmed done — so workers can't exit early while image scenes are
        still being generated.
        """
        while True:
            try:
                task = render_queue.get(timeout=_QUEUE_TIMEOUT)
            except Empty:
                continue
            if task is _DONE:
                render_queue.put(_DONE)   # fan-out to sibling workers
                break

            scene = task.scene
            log(f"Renderer-{wid}: Scene {scene.id} rendering (attempt {task.attempt+1})...")

            clip_path, error = renderer.render_with_critic(
                scene, state.resolution, scenes_dir, critic=critic
            )

            if clip_path:
                log(f"Renderer-{wid}: Scene {scene.id} ✓")
                mark_done(scene, clip_path)

            elif task.attempt < REGEN_RETRIES:
                # ── Retry with error context ──────────────────────────────
                next_attempt = task.attempt + 1
                log(f"Renderer-{wid}: Scene {scene.id} failed "
                    f"({next_attempt}/{REGEN_RETRIES+1}) — requeueing")

                if scene.visual_strategy in (VisualStrategy.MANIM,
                                             VisualStrategy.TEXT_ANIMATION,
                                             ):
                    # Re-generate Manim code with the error traceback
                    code_queue.put(SceneTask(
                        scene        = scene,
                        attempt      = next_attempt,
                        render_error = coder._summarise_render_error(error or ""),
                    ))
                else:
                    # IMAGE_GEN render fail — retry image generation itself
                    log(f"Renderer-{wid}: Scene {scene.id} IMAGE_GEN render fail "
                        f"— re-generating image (img_attempt={task.image_attempt+1})")
                    if task.image_attempt < cfg.max_llm_retries:
                        # Re-queue as image work
                        t = threading.Thread(
                            target=image_worker,
                            args=(wid, scene),
                            name=f"ImageRetry-{scene.id}",
                        )
                        t.start()
                    else:
                        # Image retries also exhausted
                        log(f"Renderer-{wid}: Scene {scene.id} — degrading to TEXT_ANIMATION")
                        scene.visual_strategy = VisualStrategy.TEXT_ANIMATION
                        code_queue.put(SceneTask(scene=scene, attempt=next_attempt))

            else:
                # ── Emergency fallback ────────────────────────────────────
                log(f"Renderer-{wid}: Scene {scene.id} — emergency fallback")
                fb = renderer.render_emergency(scene, state.resolution, scenes_dir)
                mark_done(scene, fb)

    # ── Seed queues ───────────────────────────────────────────────────────

    for scene in manim_scenes:
        code_queue.put(SceneTask(scene=scene))

    # ── Launch coder workers ──────────────────────────────────────────────

    n_coders    = max(1, min(len(state.scenes), cfg.coder_workers))
    n_renderers = max(1, min(total_scenes, cfg.render_workers))
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
            if n_done >= total_scenes:
                break
            now = time.time()
            if now - last_log >= 30:
                log(f"Progress: {n_done}/{total_scenes} done ({(now-start)/60:.1f} min elapsed)")
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
        f"Stage B complete: {len(state.rendered_clips)}/{total_scenes} clips "
        f"in {elapsed/60:.1f} min"
    )
    return state