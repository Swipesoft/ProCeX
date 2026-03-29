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

# Threshold for automatic I2V upgrade of IMAGE_GEN scenes (seconds).
# IMAGE_GEN scenes with tts_duration >= this get upgraded to:
#   video(first 8s via Novita I2V) + ken-burns(remaining duration)
# Set to 0 to disable. Needs NOVITA_API_KEY in .env.
I2V_UPGRADE_THRESHOLD = 8.0

# Maximum number of image/video scenes processed concurrently.
# Each image scene fires up to 3 parallel Novita I2V calls.
# 3 scenes × 3 I2V calls = 9 concurrent connections → SSL pool exhaustion.
# Capping at 3 scenes in flight limits peak Novita concurrency to a safe level.
SCENE_CONCURRENCY_LIMIT = 3


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
            "duration_seconds", "tts_audio_path", "tts_duration",
            "tts_audio_start", "timestamps",
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
                    if scene.parent_scene_id:
                        # Subscene beat: tts_duration is the proportional slice of
                        # the parent's ACTUAL recorded audio (not LLM estimate).
                        # scene.duration_seconds here is the beat's fraction of the
                        # parent duration_seconds (LLM estimate) — NOT the actual audio.
                        # Use tts_duration already set by _expand_subscenes if available,
                        # else fall back to scene.duration_seconds.
                        if not scene.tts_duration or scene.tts_duration <= 0:
                            scene.tts_duration = scene.duration_seconds
                    else:
                        # Original non-split scene: preserve the actual TTS duration
                        # recorded by TTSAgent. NEVER overwrite with LLM estimate.
                        scene.tts_duration    = tts_src.tts_duration
                        scene.tts_audio_start = getattr(tts_src, "tts_audio_start", 0.0)
                    # Timestamps already sliced by _expand_subscenes

            # ── Fix subscene beat durations AND tts_audio_start offsets ─────
            #
            # ROOT CAUSE OF SCENE-4 BUG:
            # _expand_subscenes uses parent.tts_duration to size beats, but
            # VisualDirector ran in PARALLEL with TTSAgent — so tts_duration
            # was 0.0 at that moment and it fell back to duration_seconds
            # (the LLM estimate). Example from logs:
            #   Director estimate: 41.0s  → beats sum to 40.6s
            #   Actual TTS:        48.3s  → last 7.7s has NO beat covering it
            #   Result: last ~2 sentences play in the audio but video has ended.
            #
            # FIX — two passes:
            # Pass 1: Proportionally rescale each group of sibling beats so
            #         their durations sum to the actual parent tts_duration.
            # Pass 2: Recompute tts_audio_start for every beat from scratch
            #         using the rescaled durations + actual parent abs offset.
            #
            # This is purely additive — no existing feature is removed.

            # Group subscene beats by parent id
            from collections import defaultdict as _dd
            beats_by_parent = _dd(list)
            for scene in dir_scenes:
                if scene.parent_scene_id:
                    beats_by_parent[scene.parent_scene_id].append(scene)

            for parent_id, beats in beats_by_parent.items():
                parent_src = tts_by_id.get(parent_id)
                if not parent_src:
                    continue

                actual_dur = getattr(parent_src, "tts_duration", 0.0) or 0.0
                parent_abs = getattr(parent_src, "tts_audio_start", 0.0)

                if actual_dur <= 0:
                    continue

                # ── Pass 1: rescale beat durations ────────────────────────────
                beat_sum = sum(
                    getattr(b, "tts_duration", 0.0) or b.duration_seconds
                    for b in beats
                )

                if beat_sum > 0 and abs(beat_sum - actual_dur) > 0.05:
                    # Proportional rescale
                    scale = actual_dur / beat_sum
                    scaled = [
                        round((getattr(b, "tts_duration", 0.0) or b.duration_seconds) * scale, 4)
                        for b in beats
                    ]
                    # Absorb rounding error into last beat so sum is exact
                    rounding_err = round(actual_dur - sum(scaled[:-1]), 4)
                    scaled[-1] = rounding_err
                    for b, new_dur in zip(beats, scaled):
                        b.tts_duration     = new_dur
                        b.duration_seconds = new_dur
                    print(
                        f"[Parallel] Scene {parent_id} beats rescaled: "
                        f"{beat_sum:.2f}s → {actual_dur:.2f}s "
                        f"(scale={scale:.4f})"
                    )

                # ── Pass 2: recompute tts_audio_start from actual parent offset ─
                # Sort beats by their original subscene_index to ensure correct order
                beats_sorted = sorted(beats, key=lambda b: getattr(b, "subscene_index", b.id))
                cursor = parent_abs
                for b in beats_sorted:
                    b.tts_audio_start = round(cursor, 4)
                    cursor += b.tts_duration

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

    # ── Post-planning VIDEO_GEN promotion ─────────────────────────────────
    # If VisualDirector assigned zero VIDEO_GEN scenes (common when running
    # non-documentary or when director prompt didn't trigger it), promote a
    # subset of IMAGE_GEN scenes to VIDEO_GEN as a fallback to ensure the
    # video has some live motion. Also handles MANIM fallback scenes.
    #
    # Rules:
    #  - Only promote when NOVITA_API_KEY is set
    #  - Target ~10-20% of total scenes for VIDEO_GEN (budget = 16s/60s)
    #  - Only promote IMAGE_GEN scenes with tts_duration <= 8s (Seedance max)
    #  - For IMAGE_GEN scenes > 8s: skip (don't split here, director handles splits)
    #  - When promoting from MANIM fallback (critic rerouted to IMAGE_GEN),
    #    same 8s constraint applies
    #  - VIDEO_GEN from this fallback path ignores the 16s/60s budget constraint
    #    (budget enforcement is for director-planned VIDEO_GEN only)
    import os as _os
    novita_key = bool(_os.environ.get("NOVITA_API_KEY", ""))
    video_scenes_planned = [s for s in state.scenes if s.visual_strategy ==
                             VisualStrategy.VIDEO_GEN]

    if novita_key and not video_scenes_planned:
        # How many VIDEO_GEN scenes to target: ~15% of total, min 1 max 6
        n_total   = len(state.scenes)
        n_target  = max(1, min(6, int(n_total * 0.15)))
        promoted  = 0
        budget_s  = 0.0
        max_budget = state.target_duration_minutes * 60.0 * (16 / 60)

        # Prefer IMAGE_GEN scenes that are short enough for the video model
        # and are visually interesting (persons/places — non-TECHNICAL)
        candidates = [
            s for s in state.scenes
            if s.visual_strategy == VisualStrategy.IMAGE_GEN
            and (getattr(s, "tts_duration", 0.0) or s.duration_seconds) <= 8.0
            and getattr(s, "paragraph_type", "") not in ("TECHNICAL",)
        ]
        # Sort by scene position — promote middle scenes for pacing
        mid = n_total // 2
        candidates.sort(key=lambda s: abs(s.id - mid))

        for scene in candidates:
            dur = getattr(scene, "tts_duration", 0.0) or scene.duration_seconds
            if promoted >= n_target or budget_s + dur > max_budget * 1.5:
                break
            scene.visual_strategy = VisualStrategy.VIDEO_GEN
            budget_s += dur
            promoted += 1

        if promoted:
            print(
                f"[Parallel] Post-planning VIDEO_GEN promotion: "
                f"{promoted} IMAGE_GEN scenes → VIDEO_GEN "
                f"(budget used: {budget_s:.0f}s)"
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


    # ── Silent VIDEO_GEN → IMAGE_GEN downgrade for long scenes ─────────────
    # A VIDEO_GEN scene only has 8s of live video; the rest is a frozen frame.
    # For scenes > 12s that is an unacceptable ratio (e.g. 30s = 73% frozen).
    # Downgrade them to IMAGE_GEN so _try_i2v_upgrade handles them with
    # multiple 8s beats — same convention as IMAGE_GEN → video upgrade.
    VIDEO_GEN_MAX_DURATION = 12.0
    downgraded_count = 0
    for _s in state.scenes:
        if _s.visual_strategy == VisualStrategy.VIDEO_GEN:
            _dur = getattr(_s, "tts_duration", 0.0) or _s.duration_seconds
            if _dur > VIDEO_GEN_MAX_DURATION:
                _s.visual_strategy = VisualStrategy.IMAGE_GEN
                downgraded_count += 1
                log(f"Scene {_s.id} VIDEO_GEN → IMAGE_GEN "
                    f"({_dur:.1f}s > {VIDEO_GEN_MAX_DURATION}s, multi-beat I2V takes over)")
    if downgraded_count:
        print(f"[Parallel] {downgraded_count} VIDEO_GEN scene(s) silently converted "
              f"to IMAGE_GEN (tts_duration > {VIDEO_GEN_MAX_DURATION}s)")

    # Classify scenes by their final strategy (AFTER downgrade so strategies are final)
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

    # Semaphore limiting concurrent API-heavy scenes (image + video).
    # Prevents Novita SSL pool exhaustion when many scenes fire I2V in parallel.
    _api_scene_sem = threading.Semaphore(SCENE_CONCURRENCY_LIMIT)

    def image_worker(wid: int, scene: Scene):
        """
        Generate image for one IMAGE_GEN or HYBRID scene.

        Fix for Problems 2 & 3:
          - Retries up to cfg.max_llm_retries times on failure
          - On all retries exhausted: degrades IMAGE_GEN → TEXT_ANIMATION (routes
            to code_queue) rather than sending empty scene to render_queue
          - HYBRID scenes always route to code_queue after image gen (Problem 4)
        """
        _api_scene_sem.acquire()
        log(f"Image-{wid}: Scene {scene.id} ({scene.visual_strategy.value}) generating..."
            f" [slot acquired, {SCENE_CONCURRENCY_LIMIT} max concurrent]")

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
            _api_scene_sem.release()
            log(f"Image-{wid}: Scene {scene.id} slot released (TEXT_ANIMATION degradation)")
            code_queue.put(SceneTask(scene=scene))
            return

        # IMAGE_GEN success → try silent I2V upgrade, else normal render
        if scene.visual_strategy == VisualStrategy.IMAGE_GEN:
            combined = _try_i2v_upgrade(scene)
            if combined:
                scene.clip_path = combined
                _api_scene_sem.release()
                log(f"Image-{wid}: Scene {scene.id} slot released (I2V upgrade done)")
                mark_done(scene, combined)
            else:
                _api_scene_sem.release()
                log(f"Image-{wid}: Scene {scene.id} slot released (→ render queue)")
                render_queue.put(SceneTask(scene=scene))

        # IMAGE_MANIM_HYBRID retired — IMAGE_GEN handles all annotations

    # ── I2V upgrade helper ────────────────────────────────────────────────
    # Silently upgrades long IMAGE_GEN scenes: generates an I2V video for the
    # first 8s then appends ken-burns for the remainder. VisualDirector never
    # knows — it still plans IMAGE_GEN. Assembler sees one clip per scene.

    # ── Beat prompt helper ────────────────────────────────────────────────
    # Generates N Ghibli-style image+motion prompts from a scene's narration
    # split at sentence/clause boundaries into N temporal buckets.

    def _generate_beat_prompts(scene: Scene, n_beats: int) -> "list[dict]":
        """
        Returns a list of dicts, one per beat:
          {
            "text":         str,   # sentence(s) for this beat
            "duration":     float, # seconds this beat covers
            "image_prompt": str,   # Ghibli-style image prompt
            "motion_hint":  str,   # I2V motion directive
            "start_word_i": int,   # first word index
            "end_word_i":   int,   # last word index (exclusive)
          }
        Falls back gracefully to [None] * n_beats on any LLM failure.
        """
        import re as _re

        narration = scene.narration_text.strip()
        timestamps = list(getattr(scene, "timestamps", []) or [])
        tts_dur    = getattr(scene, "tts_duration", 0.0) or scene.duration_seconds

        # ── Step 1: split narration at sentence/clause boundaries ────────
        # Split on: .  !  ?  ;  — (em-dash)  but never mid-word.
        # Keep delimiters so we can reconstruct full sentences.
        raw_segments = _re.split(r'(?<=[.!?;])\s+|(?<=—)\s*', narration)
        segments = [s.strip() for s in raw_segments if s.strip()]
        if not segments:
            segments = [narration]

        # ── Step 2: group segments into n_beats temporal buckets ─────────
        # Map each word to its timestamp, then assign segments to buckets
        # so each bucket covers roughly tts_dur / n_beats seconds.
        target_dur = tts_dur / n_beats

        # Build word→timestamp lookup from scene.timestamps
        word_times = {}  # word_index → end_time
        for i, ts in enumerate(timestamps):
            word_times[i] = getattr(ts, "end", 0.0)

        # Assign each word an index within the narration
        narration_words = narration.split()
        n_words = len(narration_words)

        # Map segment start/end to word indices
        seg_word_ranges = []
        word_cursor = 0
        for seg in segments:
            seg_words = seg.split()
            seg_word_ranges.append((word_cursor, word_cursor + len(seg_words)))
            word_cursor += len(seg_words)

        # Group segments into n_beats buckets by cumulative duration
        beats_raw = []
        bucket_segs, bucket_dur = [], 0.0
        bucket_start_wi = 0

        for idx, (seg, (wi_start, wi_end)) in enumerate(
                zip(segments, seg_word_ranges)):
            # Estimate segment duration from timestamps
            t_start = word_times.get(wi_start, wi_start / max(n_words, 1) * tts_dur)
            t_end   = word_times.get(min(wi_end - 1, n_words - 1),
                                     wi_end / max(n_words, 1) * tts_dur)
            seg_dur = max(t_end - t_start, 0.5)

            bucket_segs.append(seg)
            bucket_dur += seg_dur

            # Flush bucket when: full enough, OR last segment
            is_last = (idx == len(segments) - 1)
            should_flush = (bucket_dur >= target_dur and
                            len(beats_raw) < n_beats - 1) or is_last

            if should_flush:
                wi_end_bucket = wi_end
                beats_raw.append({
                    "text":         " ".join(bucket_segs),
                    "duration":     round(bucket_dur, 2),
                    "start_word_i": bucket_start_wi,
                    "end_word_i":   wi_end_bucket,
                })
                bucket_start_wi = wi_end
                bucket_segs, bucket_dur = [], 0.0

        # Ensure exactly n_beats (merge last two if over, pad if under)
        while len(beats_raw) > n_beats:
            last = beats_raw.pop()
            beats_raw[-1]["text"]     += " " + last["text"]
            beats_raw[-1]["duration"] += last["duration"]
            beats_raw[-1]["end_word_i"] = last["end_word_i"]
        while len(beats_raw) < n_beats and beats_raw:
            beats_raw.append(dict(beats_raw[-1]))  # pad with copy of last

        # ── Step 3: LLM generates image+motion prompt for each beat ──────
        # Detect if this scene features historical figures — VOICE and STORY
        # paragraph types typically describe real people (Einstein, Bohr, etc.)
        # For these, enforce Ghibli aesthetic: avoids AI face hallucination,
        # produces captivating painterly portraits instead of photorealistic slop.
        _para_type    = getattr(scene, "paragraph_type", "") or ""
        _is_figure_scene = _para_type in ("VOICE", "STORY")

        if _is_figure_scene:
            _style_rule = (
                "CRITICAL STYLE: Studio Ghibli aesthetic is MANDATORY for this scene. "
                "The scene features a historical figure. Do NOT attempt photorealism — "
                "paint them in Ghibli's signature style: soft warm outlines, atmospheric "
                "watercolour backgrounds, expressive eyes, era-appropriate clothing with "
                "hand-painted texture. The figure should feel alive and emotionally present "
                "without needing to be a photographic likeness. Think Miyazaki's 'The Wind "
                "Rises' — painterly, cinematic, deeply human. No text. No watermarks. "
                "No modern elements."
            )
        else:
            _style_rule = (
                "Style: Studio Ghibli aesthetic — painterly, atmospheric depth, "
                "soft diffused light, emotionally resonant framing, rich background "
                "detail. No text or watermarks."
            )

        BEAT_SYSTEM = (
            "You are a Ghibli-style visual director. Given a narration beat and its "
            "parent visual context, produce a JSON object with exactly two keys:\n"
            "  \"image_prompt\": a 60-120 word painterly image prompt. "
            + _style_rule + "\n"
            "  \"motion_hint\": a 10-20 word I2V camera/motion directive that matches "
            "the emotional beat (e.g. \"slow push-in on the figure\", "
            "\"gentle rack focus from background to foreground\", "
            "\"soft aerial drift across the landscape\").\n"
            "Maintain visual continuity with the parent context. Return ONLY valid JSON."
        )

        results = []
        MAX_PROMPT_RETRIES = 3

        for i, beat in enumerate(beats_raw):
            beat_user = (
                f"Parent visual context: {scene.visual_prompt[:200]}\n\n"
                f"Beat {i+1}/{n_beats} narration ({beat['duration']:.1f}s):\n"
                f"{beat['text']}\n\n"
                f"Paragraph type: {getattr(scene, 'paragraph_type', '') or 'general'}\n"
                f"Generate the image_prompt and motion_hint for this beat."
            )

            prompt_result = None
            for attempt in range(1, MAX_PROMPT_RETRIES + 1):
                try:
                    prompt_result = llm.complete_json(
                        BEAT_SYSTEM, beat_user,
                        max_tokens=512,
                        temperature=0.85,
                        primary_provider="gemini",
                    )
                    if "image_prompt" in prompt_result and "motion_hint" in prompt_result:
                        break
                    prompt_result = None
                except Exception as e:
                    log(f"Image: Scene {scene.id} beat {i+1} prompt attempt "
                        f"{attempt}/{MAX_PROMPT_RETRIES} failed: {e}")
                    if attempt < MAX_PROMPT_RETRIES:
                        time.sleep(2.0 * attempt)

            if prompt_result:
                beat["image_prompt"] = prompt_result.get("image_prompt", beat["text"])
                beat["motion_hint"]  = prompt_result.get("motion_hint", "slow cinematic drift")
            else:
                # Fallback: use parent visual_prompt with beat text appended
                beat["image_prompt"] = (
                    f"{scene.visual_prompt[:150]}. Scene moment: {beat['text'][:100]}"
                )
                beat["motion_hint"] = "slow cinematic camera drift"
                log(f"Image: Scene {scene.id} beat {i+1} — LLM failed, "
                    f"using parent prompt as fallback")

            results.append(beat)

        return results

    # ── I2V upgrade — multi-beat pipeline ─────────────────────────────────
    # Replaces single-image I2V with N beat-specific images animated in sequence.
    # External interface is identical: returns combined clip path or None.
    # On any failure, falls back gracefully to the original single-image path.

    def _try_i2v_upgrade(scene: Scene):
        import subprocess as _sp, tempfile as _tf, shutil as _sh, os as _os
        from concurrent.futures import ThreadPoolExecutor as _TPE, as_completed as _ac
        from utils.ken_burns import image_to_video_clip, cycle_effect
        import copy as _copy

        tts_dur = getattr(scene, "tts_duration", 0.0) or scene.duration_seconds
        if tts_dur < I2V_UPGRADE_THRESHOLD:
            return None
        novita_key = _os.environ.get("NOVITA_API_KEY", "")
        if not novita_key:
            return None
        if not scene.image_paths or not _os.path.exists(scene.image_paths[0]):
            return None

        # ── Determine number of beats ─────────────────────────────────────
        # New formula: keep adding 8s beats until the whole duration is covered.
        # Each beat is 8s (API max) except the last which takes the remainder.
        # If remainder < 4s (API minimum) → that last beat uses ken-burns instead.
        # This eliminates freeze-extend entirely for long scenes.
        #
        # Examples:
        #   30s → ceil(30/8)=4 beats: [8s, 8s, 8s, 6s]  → 0s frozen
        #   33s → ceil(33/8)=5 beats: [8s, 8s, 8s, 8s, 1s<4→ken-burns]
        #   15s → ceil(15/8)=2 beats: [8s, 7s]           → 0s frozen
        #    9s → ceil(9/8) =2 beats: [8s, 1s<4→ken-burns]
        BEAT_CLIP_SECS = 8    # max I2V clip length
        MIN_I2V_SECS   = 4    # I2V API minimum duration

        import math as _math
        n_beats   = max(1, _math.ceil(tts_dur / BEAT_CLIP_SECS))
        remainder = tts_dur - (n_beats - 1) * BEAT_CLIP_SECS
        # If remainder is too short for I2V, the last beat uses ken-burns
        last_beat_is_kb = (n_beats > 1 and remainder < MIN_I2V_SECS)
        # How many beats are actual I2V (remainder may be ken-burns)
        n_i2v_beats = n_beats - 1 if last_beat_is_kb else n_beats

        log(f"Image: Scene {scene.id} multi-beat I2V: {n_beats} beats "
            f"({n_i2v_beats} I2V + "
            f"{'1 ken-burns' if last_beat_is_kb else '0 ken-burns'}) "
            f"covering {tts_dur:.1f}s total")

        combined_path = _os.path.join(
            cfg.dirs["scenes"], f"scene_{scene.id:02d}.mp4"
        )
        images_dir = cfg.dirs["images"]
        videos_dir = cfg.dirs["videos"]

        # ── Step 1: Compute per-beat durations ────────────────────────────
        # Build the duration list before generating prompts so each beat's
        # narration slice is proportional to its actual clip duration.
        beat_durations = []
        for _bi in range(n_beats):
            if _bi < n_beats - 1:
                beat_durations.append(float(BEAT_CLIP_SECS))
            else:
                beat_durations.append(round(remainder, 3))

        # ── Step 1b: Generate beat prompts ────────────────────────────────
        MAX_BEATS_RETRIES = 3
        try:
            beats = _generate_beat_prompts(scene, n_beats)
        except Exception as e:
            log(f"Image: Scene {scene.id} beat prompt generation failed ({e}) "
                f"— falling back to single-image I2V")
            beats = None

        if not beats:
            # Graceful fallback: single I2V of original image (original behaviour)
            return _single_i2v_fallback(scene, tts_dur, combined_path,
                                        videos_dir, images_dir, _sp, _tf, _sh, _os)

        # Override durations from prompt generator with our precise durations
        for _bi, _beat in enumerate(beats):
            _beat["duration"] = beat_durations[_bi]

        # ── Step 2: Generate one image per beat (parallel) ────────────────
        # Create a lightweight scene clone per beat to avoid mutating the original.
        beat_image_paths = [None] * n_beats
        IMAGE_MAX_RETRIES = 3

        def _gen_beat_image(beat_idx: int, beat: dict) -> "str | None":
            """Generate image for one beat. Returns image path or None."""
            beat_scene_id = f"{scene.id}b{beat_idx+1}"
            img_path = _os.path.join(images_dir,
                                     f"scene_{scene.id}_beat_{beat_idx+1}_raw.png")

            # Build a minimal scene-like object for _generate_for_scene
            beat_scene = _copy.copy(scene)
            beat_scene.visual_prompt = beat["image_prompt"]
            beat_scene.image_paths   = []
            # Use a unique numeric id for file naming inside _generate_for_scene
            # by temporarily overriding image save path via direct API call
            for attempt in range(1, IMAGE_MAX_RETRIES + 1):
                try:
                    import google.generativeai  # verify import path
                except ImportError:
                    pass
                try:
                    from google import genai as _genai_mod
                    from google.genai import types as _types

                    prompt = (
                        beat["image_prompt"] + "\n\n"
                        "Style: Studio Ghibli. Painterly. Atmospheric. "
                        "No text. No watermarks. No UI elements."
                    )
                    if state.resolution.endswith("_v"):
                        prompt += " Vertical 9:16 portrait composition."

                    response = imager._genai.models.generate_content(
                        model=imager.cfg.nano_pro_model,
                        contents=prompt,
                        config=_types.GenerateContentConfig(
                            response_modalities=["image", "text"],
                        ),
                    )
                    parts = response.parts or []
                    saved = False
                    for part in parts:
                        if hasattr(part, "inline_data") and part.inline_data:
                            import base64 as _b64
                            img_bytes = part.inline_data.data
                            if isinstance(img_bytes, str):
                                img_bytes = _b64.b64decode(img_bytes)
                            with open(img_path, "wb") as f:
                                f.write(img_bytes)
                            saved = True
                            break
                    if saved:
                        log(f"Image: Scene {scene.id} beat {beat_idx+1} image OK "
                            f"(attempt {attempt})")
                        return img_path
                    raise RuntimeError("No image data in response")
                except Exception as e:
                    log(f"Image: Scene {scene.id} beat {beat_idx+1} image attempt "
                        f"{attempt}/{IMAGE_MAX_RETRIES} failed: {e}")
                    if attempt < IMAGE_MAX_RETRIES:
                        time.sleep(2.0 * attempt)
            return None

        with _TPE(max_workers=min(n_beats, 3)) as pool:
            futs = {pool.submit(_gen_beat_image, i, b): i
                    for i, b in enumerate(beats)}
            for fut in _ac(futs):
                idx = futs[fut]
                try:
                    beat_image_paths[idx] = fut.result()
                except Exception as e:
                    log(f"Image: Scene {scene.id} beat {idx+1} image future "
                        f"failed: {e}")

        # Check how many beats have valid images
        valid_beats = [(i, beats[i], p) for i, p in enumerate(beat_image_paths)
                       if p and _os.path.exists(p)]

        if not valid_beats:
            log(f"Image: Scene {scene.id} all beat images failed — "
                f"falling back to single-image I2V")
            return _single_i2v_fallback(scene, tts_dur, combined_path,
                                        videos_dir, images_dir, _sp, _tf, _sh, _os)

        # For partially failed beats, fill gaps with original image
        for i in range(n_beats):
            if beat_image_paths[i] is None:
                beat_image_paths[i] = scene.image_paths[0]
                log(f"Image: Scene {scene.id} beat {i+1} using original image "
                    f"as fallback")

        # ── Step 3: Submit I2V for each beat (parallel) ───────────────────
        from config import RESOLUTIONS as _RES
        res        = _RES.get(state.resolution, _RES["1080p"])
        ratio_str  = "9:16" if res.is_portrait else "16:9"
        I2V_MAX_RETRIES = 3
        beat_clips = [None] * n_beats

        def _gen_beat_clip(beat_idx: int, beat: dict,
                           img_path: str) -> "str | None":
            """Submit I2V and download clip for one beat."""
            clip_secs = max(4, min(8, int(beats[beat_idx]["duration"])))

            import base64 as _b64, requests as _req, time as _time
            NOVITA_I2V = "https://api.novita.ai/v3/async/seedance-v1.5-pro-i2v"
            NOVITA_RES = "https://api.novita.ai/v3/async/task-result"
            headers    = {
                "Authorization": f"Bearer {novita_key}",
                "Content-Type":  "application/json",
            }
            # Encode image
            try:
                with open(img_path, "rb") as f: raw = f.read()
                ext  = _os.path.splitext(img_path)[1].lstrip(".").lower()
                mime = f"image/{ext}" if ext in ("jpg","jpeg","png","webp") else "image/png"
                b64  = f"data:{mime};base64,{_b64.b64encode(raw).decode()}"
            except Exception as e:
                log(f"Image: Scene {scene.id} beat {beat_idx+1} encode failed: {e}")
                return None

            motion = beat.get("motion_hint", "slow cinematic drift")
            prompt = (f"{beat['image_prompt'][:300]}. {motion}. "
                      f"No text overlays. No watermarks.")

            payload = {
                "image": b64, "prompt": prompt,
                "duration": clip_secs, "ratio": "adaptive",
                "resolution": "720p", "fps": 24,
                "watermark": False, "generate_audio": False,
                "camera_fixed": False, "seed": -1,
            }

            for attempt in range(1, I2V_MAX_RETRIES + 1):
                try:
                    r = _req.post(NOVITA_I2V, json=payload,
                                  headers=headers, timeout=30)
                    if r.status_code == 429:
                        import re as _re2
                        m = _re2.search(r'retry.{0,10}?([\d.]+)\s*s',
                                        r.text, _re2.IGNORECASE)
                        sleep = float(m.group(1)) + 2 if m else 30.0
                        log(f"Image: Scene {scene.id} beat {beat_idx+1} I2V 429 "
                            f"— retry in {sleep:.0f}s (attempt {attempt})")
                        _time.sleep(sleep)
                        continue
                    r.raise_for_status()
                    task_id = r.json().get("task_id", "")
                    if not task_id:
                        raise RuntimeError("No task_id in I2V response")
                    break
                except Exception as e:
                    log(f"Image: Scene {scene.id} beat {beat_idx+1} I2V submit "
                        f"attempt {attempt}/{I2V_MAX_RETRIES} failed: {e}")
                    if attempt < I2V_MAX_RETRIES:
                        _time.sleep(5.0 * attempt)
                    else:
                        return None

            # Poll
            deadline = _time.time() + 300
            while _time.time() < deadline:
                _time.sleep(5)
                try:
                    poll = _req.get(NOVITA_RES, headers=headers,
                                    params={"task_id": task_id}, timeout=15)
                    poll.raise_for_status()
                    data   = poll.json()
                    status = data.get("task", {}).get("status", "")
                    if status == "TASK_STATUS_SUCCEED":
                        videos    = data.get("videos", [])
                        video_url = videos[0].get("video_url", "") if videos else ""
                        if not video_url:
                            return None
                        # Download (no auth header — S3 pre-signed)
                        dl = _req.get(video_url, headers={},
                                      timeout=120, stream=True)
                        dl.raise_for_status()
                        out_path = _os.path.join(
                            videos_dir,
                            f"scene_{scene.id}_beat_{beat_idx+1}.mp4"
                        )
                        with open(out_path, "wb") as f:
                            for chunk in dl.iter_content(8192):
                                if chunk: f.write(chunk)
                        log(f"Image: Scene {scene.id} beat {beat_idx+1} clip ✓ "
                            f"({clip_secs}s)")
                        return out_path
                    elif status in ("TASK_STATUS_FAILED", "TASK_STATUS_EXPIRED"):
                        log(f"Image: Scene {scene.id} beat {beat_idx+1} task {status}")
                        return None
                except Exception as e:
                    log(f"Image: Scene {scene.id} beat {beat_idx+1} poll error: {e}")

            log(f"Image: Scene {scene.id} beat {beat_idx+1} timed out")
            return None

        # Submit I2V for all beats except the last one when it's a ken-burns beat
        i2v_beat_indices = list(range(n_i2v_beats))  # 0..n_i2v_beats-1
        if i2v_beat_indices:
            with _TPE(max_workers=min(len(i2v_beat_indices), 3)) as pool:
                futs = {pool.submit(_gen_beat_clip, i, beats[i], beat_image_paths[i]): i
                        for i in i2v_beat_indices}
                for fut in _ac(futs):
                    idx = futs[fut]
                    try:
                        beat_clips[idx] = fut.result()
                    except Exception as e:
                        log(f"Image: Scene {scene.id} beat {idx+1} clip future: {e}")
        # The last beat is already None in beat_clips — Step 4 will fill it with ken-burns

        # ── Step 4: Fill failed clips with ken-burns fallback ─────────────
        for i in range(n_beats):
            if beat_clips[i] is None or not _os.path.exists(beat_clips[i] or ""):
                img  = beat_image_paths[i] or scene.image_paths[0]
                dur  = beats[i]["duration"]
                kb   = _os.path.join(videos_dir,
                                     f"scene_{scene.id}_beat_{i+1}_kb.mp4")
                try:
                    image_to_video_clip(
                        image_path=img, duration=dur,
                        output_path=kb, resolution=state.resolution,
                        effect=cycle_effect(scene.id + i),
                    )
                    beat_clips[i] = kb
                    log(f"Image: Scene {scene.id} beat {i+1} ken-burns fallback ✓")
                except Exception as e:
                    log(f"Image: Scene {scene.id} beat {i+1} ken-burns fallback "
                        f"failed: {e}")

        # ── Step 5: Concat all beat clips ─────────────────────────────────
        valid_clips = [c for c in beat_clips if c and _os.path.exists(c)]
        if not valid_clips:
            log(f"Image: Scene {scene.id} all beats failed — "
                f"falling back to single-image I2V")
            return _single_i2v_fallback(scene, tts_dur, combined_path,
                                        videos_dir, images_dir, _sp, _tf, _sh, _os)

        if len(valid_clips) == 1:
            _sh.copy2(valid_clips[0], combined_path)
            log(f"Image: Scene {scene.id} multi-beat ✓ (1 clip)")
            return combined_path

        with _tf.NamedTemporaryFile(mode="w", suffix=".txt", delete=False,
                                    encoding="utf-8") as f:
            for c in valid_clips:
                f.write(f"file '{_os.path.abspath(c)}'\n")
            list_path = f.name
        try:
            r = _sp.run([
                "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path,
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-pix_fmt", "yuv420p", "-an", combined_path,
            ], capture_output=True, text=True, timeout=300)
            _os.unlink(list_path)
            if r.returncode != 0:
                raise RuntimeError(r.stderr[-300:])
        except Exception as e:
            try: _os.unlink(list_path)
            except Exception: pass
            log(f"Image: Scene {scene.id} concat failed ({e}) — "
                f"using first valid clip")
            _sh.copy2(valid_clips[0], combined_path)

        log(f"Image: Scene {scene.id} multi-beat I2V ✓ → {combined_path} "
            f"({len(valid_clips)}/{n_beats} clips)")
        return combined_path

    # ── Single-image I2V fallback ─────────────────────────────────────────
    # Preserved exactly from original _try_i2v_upgrade for graceful degradation.

    def _single_i2v_fallback(scene, tts_dur, combined_path,
                              videos_dir, images_dir, _sp, _tf, _sh, _os):
        """Original single-image I2V + ken-burns behaviour. Unchanged."""
        from utils.ken_burns import image_to_video_clip, cycle_effect

        video_secs = min(8, int(tts_dur))
        image_secs = round(tts_dur - video_secs, 3)
        log(f"Image: Scene {scene.id} single I2V fallback "
            f"({video_secs}s live + {image_secs:.1f}s still)")

        try:
            video_clip = videogen.generate_for_scene(
                scene=scene, resolution=state.resolution, videos_dir=videos_dir
            )
        except Exception as e:
            log(f"Image: Scene {scene.id} single I2V failed ({e}) — ken-burns only")
            return None
        if not video_clip or not _os.path.exists(video_clip):
            return None

        if image_secs < 0.5:
            _sh.copy2(video_clip, combined_path)
            return combined_path

        kb_path = _os.path.join(videos_dir, f"scene_{scene.id:02d}_kbtail.mp4")
        try:
            image_to_video_clip(
                image_path=scene.image_paths[0], duration=image_secs,
                output_path=kb_path, resolution=state.resolution,
                effect=cycle_effect(scene.id + 1),
            )
        except Exception as e:
            log(f"Image: Scene {scene.id} ken-burns tail failed ({e})")
            _sh.copy2(video_clip, combined_path)
            return combined_path

        with _tf.NamedTemporaryFile(mode="w", suffix=".txt", delete=False,
                                    encoding="utf-8") as f:
            f.write(f"file '{_os.path.abspath(video_clip)}'\n")
            f.write(f"file '{_os.path.abspath(kb_path)}'\n")
            list_path = f.name
        try:
            r = _sp.run([
                "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path,
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-pix_fmt", "yuv420p", "-an", combined_path,
            ], capture_output=True, text=True, timeout=120)
            _os.unlink(list_path)
            if r.returncode != 0:
                raise RuntimeError(r.stderr[-300:])
        except Exception as e:
            try: _os.unlink(list_path)
            except Exception: pass
            _sh.copy2(video_clip, combined_path)

        return combined_path

    # ── Video gen worker ──────────────────────────────────────────────────
    # Runs in a daemon thread per scene — polls Novita until done or timeout.
    # On failure degrades to IMAGE_GEN so the scene always produces a frame.

    def video_worker(scene: Scene):
        """Generate one VIDEO_GEN scene via Novita Seedance 1.5 Pro."""
        _api_scene_sem.acquire()
        log(f"Video: Scene {scene.id} VIDEO_GEN generating..."
            f" [slot acquired, {SCENE_CONCURRENCY_LIMIT} max concurrent]")
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
            _api_scene_sem.release()
            log(f"Video: Scene {scene.id} slot released")
            mark_done(scene, clip_path)
        else:
            # Degrade to IMAGE_GEN — dispatch to image_worker
            log(f"Video: Scene {scene.id} failed — degrading to IMAGE_GEN")
            _api_scene_sem.release()   # release before image_worker re-acquires
            log(f"Video: Scene {scene.id} slot released (before IMAGE_GEN fallback)")
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
                # Cycle-breaker: if visual_strategy is already TEXT_ANIMATION,
                # IMAGE_GEN already failed previously for this scene. Do NOT
                # dispatch to image_worker again — that creates an infinite
                # loop: TEXT_ANIMATION → VLMCritic → imagegen_fallback →
                # IMAGE_GEN fails → TEXT_ANIMATION → VLMCritic → ... forever.
                # Instead accept the TEXT_ANIMATION clip as-is.
                if scene.visual_strategy == VisualStrategy.TEXT_ANIMATION:
                    log(
                        f"Renderer-{wid}: Scene {scene.id} — IMAGE_GEN already failed, "
                        f"accepting TEXT_ANIMATION as final fallback"
                    )
                    mark_done(scene, clip_path)
                    continue

                try:
                    from agents.image_reprompter import ImageReprompter

                    # Clear stale Manim clip_path so the old failed render is
                    # never assembled into the final video.
                    scene.clip_path = ""

                    # Use tts_duration as the authoritative clip length for
                    # ken-burns so ImageGen clips match audio exactly.
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