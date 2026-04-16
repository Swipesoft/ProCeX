"""
orchestrator.py
ProcEx pipeline orchestrator — parallel edition.

Execution plan:
  [Sequential]  DomainRouter → ScriptWriter
  [Parallel A]  TTSAgent  ∥  VisualDirector        (both need only ScriptWriter output)
  [Parallel B]  ManimCoder + ImageGenAgent + RendererAgent  (queue-based pipeline)
                  code_queue → coder workers → render_queue
                  image scenes → image workers → render_queue
                  render_queue → renderer workers → done / retry → code_queue
  [Sequential]  AssemblerAgent

Checkpoint-resume still works — each stage checks _stage_done() before running.
"""
from __future__ import annotations
import os
import time
from state import ProcExState, InputType, VisualStrategy
from config import ProcExConfig
from utils.llm_client import LLMClient
# -------------------------------------------------------
# Switch from TogetherAI inference to Modal cloud self-hosted
# from utils.gemma_client import GemmaClient
from utils.modal_gemma_client import GemmaClient
# -------------------------------------------------------
from utils.slug import slugify
from utils.pdf_parser import extract_pdf_text


def _upgrade_opening_scene(state, cfg, llm):
    """
    Post-Stage B hook: converts the first scene's clip to an animated video
    if it was rendered as Manim or TEXT_ANIMATION.

    Strategy:
      1. Find the first scene by tts_audio_start (may be a subscene beat).
      2. If clip_path points to a Manim render → replace with cinematic opener.
      3. Generate a Ghibli/cinematic image via Gemini Pro.
      4. Animate it via Novita I2V (8s clip).  Fallback: ken-burns.
      5. Write new clip_path in-place — Assembler never knows anything changed.

    Does nothing if:
      - First scene is already IMAGE_GEN / VIDEO_GEN (already animated)
      - NOVITA_API_KEY not set (falls back to ken-burns on generated image)
      - Any step fails (non-critical — original Manim clip is kept)
    """
    import os, subprocess, tempfile, base64, time, requests

    from state import VisualStrategy
    from config import RESOLUTIONS
    from utils.ken_burns import image_to_video_clip

    if not state.scenes:
        return

    # ── Find the first scene ──────────────────────────────────────────────────
    # Sort all scenes by tts_audio_start so we get the true first segment
    # (could be subscene 101 if Scene 1 was split by VisualDirector).
    rendered = [s for s in state.scenes if s.clip_path and os.path.exists(s.clip_path)]
    if not rendered:
        return

    first = min(rendered, key=lambda s: getattr(s, "tts_audio_start", 0.0))

    # Only upgrade Manim or TEXT_ANIMATION scenes — IMAGE_GEN/VIDEO_GEN are
    # already animated.
    if first.visual_strategy not in (VisualStrategy.MANIM,
                                      VisualStrategy.TEXT_ANIMATION):
        print(f"[OpeningHook] Scene {first.id} is already "
              f"{first.visual_strategy.value} — no upgrade needed")
        return

    print(f"[OpeningHook] Scene {first.id} ({first.visual_strategy.value}) → "
          f"upgrading to cinematic opener...")

    res       = RESOLUTIONS.get(state.resolution, RESOLUTIONS["1080p"])
    ratio_str = "9:16" if res.is_portrait else "16:9"
    images_dir = cfg.dirs["images"]
    videos_dir = cfg.dirs["videos"]
    scenes_dir = cfg.dirs["scenes"]
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(scenes_dir, exist_ok=True)

    # ── Step 1: Generate cinematic opening image ──────────────────────────────
    # Build a prompt tuned for dramatic openers — dark, cinematic, specific to
    # the topic from the scene's narration_text.
    narration_snippet = (first.narration_text or "")[:200].replace("\n", " ")
    topic_hint        = state.topic_slug.replace("_", " ") if state.topic_slug else ""

    opening_prompt = (
        f"Studio Ghibli aesthetic, dramatic cinematic opening frame. "
        f"Dark atmospheric background with subtle light rays. "
        f"The scene evokes: {narration_snippet}. "
        f"Rich painterly detail, emotionally resonant, no text, no watermarks, "
        f"no UI elements. "
        f"{'Vertical 9:16 portrait composition.' if res.is_portrait else 'Wide 16:9 cinematic composition.'}"
    )

    img_path = os.path.join(images_dir, f"scene_{first.id:02d}_opening_hook.png")

    try:
        # Use the same google.genai client pattern as ImageGenAgent — NOT google.generativeai
        from google import genai as _genai_mod
        from google.genai import types as _types

        genai_client = _genai_mod.Client(api_key=cfg.gemini_api_key)
        # Use the same model as ImageGenAgent Pro tier (cfg.nano_fast_model)
        response = genai_client.models.generate_content(
            model=cfg.nano_fast_model,   # "gemini-3.1-flash-image-preview"
            contents=opening_prompt,
            config=_types.GenerateContentConfig(
                response_modalities=["image", "text"],
            ),
        )
        parts = response.parts or []
        saved = False
        for part in parts:
            if hasattr(part, "inline_data") and part.inline_data:
                img_bytes = part.inline_data.data
                if isinstance(img_bytes, str):
                    import base64 as _b64
                    img_bytes = _b64.b64decode(img_bytes)
                with open(img_path, "wb") as f:
                    f.write(img_bytes)
                saved = True
                break
        if not saved:
            finish = ""
            try:
                finish = str(response.candidates[0].finish_reason) if response.candidates else "no candidates"
            except Exception:
                pass
            raise RuntimeError(f"No image data in response (finish_reason={finish})")
        print(f"[OpeningHook] Image generated → {os.path.basename(img_path)}")
    except Exception as e:
        print(f"[OpeningHook] Image generation failed ({e}) — keeping Manim clip")
        return

    # ── Step 2: Animate via Novita I2V ────────────────────────────────────────
    novita_key = os.environ.get("NOVITA_API_KEY", "")
    tts_dur    = getattr(first, "tts_duration", 0.0) or first.duration_seconds
    clip_secs  = min(8, max(4, int(tts_dur)))

    new_clip_path = os.path.join(scenes_dir,
                                 f"scene_{first.id:02d}_opening_hook.mp4")

    if novita_key:
        try:
            # Encode image
            with open(img_path, "rb") as f: raw = f.read()
            ext  = os.path.splitext(img_path)[1].lstrip(".").lower()
            mime = f"image/{ext}" if ext in ("jpg","jpeg","png","webp") else "image/png"
            b64  = f"data:{mime};base64,{base64.b64encode(raw).decode()}"

            headers = {
                "Authorization": f"Bearer {novita_key}",
                "Content-Type":  "application/json",
            }
            payload = {
                "image":          b64,
                "prompt":         (
                    "Slow dramatic camera push-in. Atmospheric lighting shift. "
                    "Subtle particle effects. Cinematic. No text. No watermarks."
                ),
                "duration":       clip_secs,
                "ratio":          "adaptive",
                "resolution":     "720p",
                "fps":            24,
                "watermark":      False,
                "generate_audio": False,
                "camera_fixed":   False,
                "seed":           -1,
            }

            NOVITA_I2V  = "https://api.novita.ai/v3/async/seedance-v1.5-pro-i2v"
            NOVITA_POLL = "https://api.novita.ai/v3/async/task-result"

            r = requests.post(NOVITA_I2V, json=payload, headers=headers, timeout=30)
            r.raise_for_status()
            task_id = r.json().get("task_id", "")
            if not task_id:
                raise RuntimeError("No task_id returned")
            print(f"[OpeningHook] I2V submitted — task_id={task_id}")

            # Poll
            deadline = time.time() + 300
            video_url = None
            while time.time() < deadline:
                time.sleep(5)
                poll = requests.get(NOVITA_POLL, headers=headers,
                                    params={"task_id": task_id}, timeout=15)
                poll.raise_for_status()
                data   = poll.json()
                status = data.get("task", {}).get("status", "")
                if status == "TASK_STATUS_SUCCEED":
                    videos    = data.get("videos", [])
                    video_url = videos[0].get("video_url", "") if videos else ""
                    break
                elif status in ("TASK_STATUS_FAILED", "TASK_STATUS_EXPIRED"):
                    raise RuntimeError(f"I2V task {status}")

            if not video_url:
                raise RuntimeError("No video_url after polling")

            # Download — no auth header for S3 pre-signed URLs
            dl = requests.get(video_url, headers={}, timeout=120, stream=True)
            dl.raise_for_status()
            with open(new_clip_path, "wb") as f:
                for chunk in dl.iter_content(8192):
                    if chunk: f.write(chunk)
            print(f"[OpeningHook] I2V clip downloaded → {os.path.basename(new_clip_path)}")

        except Exception as e:
            print(f"[OpeningHook] I2V failed ({e}) — falling back to ken-burns")
            novita_key = ""   # trigger ken-burns fallback below

    if not novita_key:
        # Ken-burns fallback — still far better than a Manim slide as an opener
        try:
            from utils.ken_burns import image_to_video_clip, cycle_effect
            image_to_video_clip(
                image_path  = img_path,
                duration    = tts_dur,
                output_path = new_clip_path,
                resolution  = state.resolution,
                effect      = "drift",   # slow drift is most cinematic for opener
            )
            print(f"[OpeningHook] Ken-burns fallback → {os.path.basename(new_clip_path)}")
        except Exception as e:
            print(f"[OpeningHook] Ken-burns failed ({e}) — keeping original Manim clip")
            return

    # ── Step 3: Replace clip_path in state so Assembler uses new clip ─────────
    if os.path.exists(new_clip_path) and os.path.getsize(new_clip_path) > 1000:
        old_clip = first.clip_path
        first.clip_path      = new_clip_path
        first.visual_strategy = VisualStrategy.IMAGE_GEN   # label it correctly
        print(f"[OpeningHook] ✓ Scene {first.id} upgraded: "
              f"{os.path.basename(old_clip)} → {os.path.basename(new_clip_path)}")
    else:
        print(f"[OpeningHook] New clip empty or missing — keeping original Manim clip")


class ProcExOrchestrator:

    def __init__(self, cfg: ProcExConfig | None = None):
        self.cfg = cfg or ProcExConfig()
        self.cfg.make_dirs()
        if getattr(self.cfg, 'gemma_provider', False):
            self.llm = GemmaClient(self.cfg)
            print('[ProcEx] 🟣 Gemma 4 31B mode active — GemmaClient initialised')
        else:
            self.llm = LLMClient(self.cfg)

    def run(
        self,
        input_path: str,
        topic_hint: str = "",
        resolution: str = "1080p",
        target_minutes: float = 5.0,
        resume_checkpoint: str | None = None,
        presentation_style: str = "auto",
        context: str = "",
    ) -> str:
        start_time = time.time()

        # ── Gemma provider overrides ──────────────────────────────────────────
        """
        if getattr(self.cfg, "gemma_provider", False):
            # Force landscape 1080p — avoids portrait layout constraints
            # that cause over-rejection in VLMCritic when image_gen is off
            if resolution.endswith("_v"):
                print(f"[ProcEx] Gemma mode: {resolution} → 1080p (portrait overridden)")
                resolution = "1080p"
            # Image generation requires Gemini imagen — disabled for Gemma run
            self.cfg.enable_critic_loop = True   # critic still runs (Gemma vision)
        """
        if getattr(self.cfg, "gemma_provider", False):
            self.cfg.enable_critic_loop = True
        # ── Load or create state ──────────────────────────────────────────
        if resume_checkpoint and os.path.exists(resume_checkpoint):
            print(f"[ProcEx] Resuming from checkpoint: {resume_checkpoint}")
            state = ProcExState.load_checkpoint(resume_checkpoint)
        else:
            state = self._init_state(input_path, topic_hint, resolution, target_minutes, presentation_style)
            # Fresh run — purge any stale checkpoint for this topic so no
            # agent is ever silently skipped with outdated decisions.
            self._clear_checkpoint(state)

        # Store context on state — survives checkpointing, reaches all agents.
        # On resume: CLI --context overrides any saved context; if no CLI context
        # is given, the checkpoint's saved context is preserved automatically.
        if context:
            state.context = context
            print(f"[ProcEx] ℹ Context set ({len(context)} chars) — all agents guided by it")
        elif state.context:
            print(f"[ProcEx] ℹ Context restored from checkpoint ({len(state.context)} chars)")

        for issue in self.cfg.validate():
            print(f"[ProcEx] ⚠ {issue}")

        # ════════════════════════════════════════════════════════════════
        # Stage 1 — DomainRouter  [sequential]
        # ════════════════════════════════════════════════════════════════
        state = self._run_agent("DomainRouter", state)

        # ════════════════════════════════════════════════════════════════
        # Stage 2 — ScriptWriter  [sequential]
        # Bypassed for documentary PDFs — scenes already parsed from tags.
        # ════════════════════════════════════════════════════════════════
        if getattr(state, "_is_documentary", False):
            print(
                f"[ProcEx] ↩ Skipping ScriptWriter — documentary PDF "
                f"({len(state.scenes)} scenes already parsed)"
            )
        else:
            state = self._run_agent("ScriptWriter", state)

        # ════════════════════════════════════════════════════════════════
        # Stage 2.5 — SlopRefiner  [sequential, post-script pre-TTS]
        # Detects and corrects slop patterns in scene.narration_text.
        # Two-pass: regex pre-filter → LLM correction for flagged scenes only.
        # Recalculates scene.duration_seconds after any rewrite.
        # Skipped if TTSAgent already done (resume path).
        # ════════════════════════════════════════════════════════════════
        if not self._stage_done(state, "TTSAgent") and state.scenes:
            try:
                from utils.slop_refiner import refine_scenes
                print("\n[ProcEx] ▶ Running SlopRefiner...")
                state = refine_scenes(state, self.llm, self.cfg)
                print("[ProcEx] ✓ SlopRefiner done")
            except Exception as _slop_err:
                print(f"[ProcEx] ⚠ SlopRefiner failed ({_slop_err}) — continuing with original narration")

        # ════════════════════════════════════════════════════════════════
        # Stage 3 — TTSAgent ∥ VisualDirector  [parallel A]
        # ════════════════════════════════════════════════════════════════
        tts_done      = self._stage_done(state, "TTSAgent")
        director_done = self._stage_done(state, "VisualDirector")

        if tts_done and director_done:
            print("[ProcEx] ↩ Skipping TTSAgent + VisualDirector (already done)")
        elif tts_done:
            print("[ProcEx] ↩ TTSAgent already done — running VisualDirector only")
            state = self._run_agent("VisualDirector", state)
        elif director_done:
            print("[ProcEx] ↩ VisualDirector already done — running TTSAgent only")
            state = self._run_agent("TTSAgent", state)
        else:
            print("\n[ProcEx] ▶ Running TTSAgent ∥ VisualDirector (parallel)...")
            from parallel_runner import run_tts_and_director_parallel
            try:
                state = run_tts_and_director_parallel(state, self.cfg, self.llm)
            except Exception as e:
                print(f"[ProcEx] ✗ Parallel stage A failed: {e}")
                self._checkpoint(state)
                raise
            self._checkpoint(state)
            print("[ProcEx] ✓ TTSAgent + VisualDirector done")

        # ════════════════════════════════════════════════════════════════
        # Stage 4 — ManimCoder + ImageGenAgent + RendererAgent  [parallel B]
        # ════════════════════════════════════════════════════════════════
        if self._stage_done(state, "RendererAgent"):
            print("[ProcEx] ↩ Skipping generation+render pipeline (already done)")
        else:
            print("\n[ProcEx] ▶ Running ManimCoder + ImageGenAgent + RendererAgent (parallel queue)...")
            from parallel_runner import run_generation_render_pipeline
            try:
                state = run_generation_render_pipeline(state, self.cfg, self.llm)
            except Exception as e:
                print(f"[ProcEx] ✗ Parallel stage B failed: {e}")
                self._checkpoint(state)
                raise
            self._checkpoint(state)
            print("[ProcEx] ✓ Generation + render pipeline done")


        # ════════════════════════════════════════════════════════════════
        # Stage 4.5 — Opening Hook Upgrade  [post-Stage B, pre-Assembler]
        # If the very first rendered scene is a Manim/Text clip, convert it
        # to an animated video (I2V) or at minimum a cinematic image with
        # ken-burns. This gives every video a live-motion opening frame that
        # captures viewers before they scroll away.
        # ════════════════════════════════════════════════════════════════
        _image_gen_allowed = (
            not getattr(self.cfg, "gemma_provider", False)
            and getattr(self.cfg, "image_gen_enabled", True)
            and state.skill_pack.get("image_gen_enabled", True)
        )
        if _image_gen_allowed:
            try:
                _upgrade_opening_scene(state, self.cfg, self.llm)
            except Exception as _e:
                print(f"[ProcEx] Opening hook upgrade failed (non-critical): {_e}")
        else:
            print("[ProcEx] ↩ Skipping opening hook upgrade (image generation disabled)")

        # ════════════════════════════════════════════════════════════════
        # Stage 5 — AssemblerAgent  [sequential]
        # ════════════════════════════════════════════════════════════════
        state = self._run_agent("AssemblerAgent", state)

        # ── Summary ───────────────────────────────────────────────────────
        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"[ProcEx] Pipeline complete in {elapsed/60:.1f} minutes")
        print(f"[ProcEx] Output: {state.final_video_path}")

        if state.errors:
            print(f"[ProcEx] ⚠ {len(state.errors)} non-fatal errors:")
            for e in state.errors:
                print(f"  • [{e['agent']}] {e['error']}")

        return state.final_video_path

    # ── Agent runner ──────────────────────────────────────────────────────────

    def _run_agent(self, agent_name: str, state: ProcExState) -> ProcExState:
        """Run a single named agent sequentially with skip + checkpoint logic."""
        if self._stage_done(state, agent_name):
            print(f"[ProcEx] ↩ Skipping {agent_name} (already done)")
            return state

        agent_class = self._agent_class(agent_name)

        # TTSAgent always uses Gemini TTS regardless of provider —
        # Gemma cannot generate audio. Swap back to LLMClient for TTS.
        if getattr(self.cfg, "gemma_provider", False) and agent_name == "TTSAgent":
            tts_llm = LLMClient(self.cfg)
            agent   = agent_class(self.cfg, tts_llm)
        else:
            agent   = agent_class(self.cfg, self.llm)

        print(f"\n[ProcEx] ▶ Running {agent_name}...")
        try:
            state = agent.run(state)
        except Exception as e:
            print(f"[ProcEx] ✗ {agent_name} FAILED: {e}")
            self._checkpoint(state)
            raise

        self._checkpoint(state)
        print(f"[ProcEx] ✓ {agent_name} done")
        return state

    @staticmethod
    def _agent_class(name: str):
        from agents.domain_router   import DomainRouter
        from agents.script_writer   import ScriptWriter
        from agents.tts_agent       import TTSAgent
        from agents.visual_director import VisualDirector
        from agents.manim_coder     import ManimCoder
        from agents.image_gen_agent import ImageGenAgent
        from agents.renderer        import RendererAgent
        from agents.assembler       import AssemblerAgent

        mapping = {
            "DomainRouter":   DomainRouter,
            "ScriptWriter":   ScriptWriter,
            "TTSAgent":       TTSAgent,
            "VisualDirector": VisualDirector,
            "ManimCoder":     ManimCoder,
            "ImageGenAgent":  ImageGenAgent,
            "RendererAgent":  RendererAgent,
            "AssemblerAgent": AssemblerAgent,
        }
        if name not in mapping:
            raise ValueError(f"Unknown agent: {name}")
        return mapping[name]

    # ── State init ────────────────────────────────────────────────────────────

    def _init_state(self, input_path, topic_hint, resolution, target_minutes, presentation_style="auto"):
        state = ProcExState(
            resolution              = resolution,
            target_duration_minutes = target_minutes,
            presentation_style      = presentation_style,
        )
        ext = os.path.splitext(input_path)[-1].lower()
        if ext == ".pdf":
            print(f"[ProcEx] Loading PDF: {input_path}")
            state.raw_input  = extract_pdf_text(input_path)
            state.input_type = InputType.PDF
        elif ext in (".txt", ".md"):
            with open(input_path, encoding="utf-8", errors="replace") as f:
                state.raw_input = f.read()
            state.input_type = InputType.TEXT
        else:
            if os.path.exists(input_path):
                with open(input_path, encoding="utf-8", errors="replace") as f:
                    state.raw_input = f.read()
            else:
                state.raw_input = input_path
            state.input_type = InputType.TEXT

        hint_text        = topic_hint or state.raw_input[:200]
        state.topic_slug = slugify(hint_text)
        print(f"[ProcEx] Topic slug: {state.topic_slug}")

        # ── Documentary PDF detection ─────────────────────────────────────────
        # If the PDF was produced by DeepDocumentaryAgent it contains ▸ type
        # labels. Parse paragraphs directly into scenes — ScriptWriter LLM
        # re-generation is bypassed so the multi-voice structure is preserved.
        from utils.documentary_parser import is_documentary_pdf, parse_documentary_scenes
        if state.raw_input and is_documentary_pdf(state.raw_input):
            print("[ProcEx] ✓ Documentary PDF detected — parsing tagged paragraphs directly")
            parsed_scenes = parse_documentary_scenes(
                state.raw_input,
                target_minutes=target_minutes,
            )
            if parsed_scenes:
                state.scenes              = parsed_scenes
                state._is_documentary     = True   # flag for Stage 2 bypass
                n_voice = sum(1 for s in parsed_scenes if s.paragraph_type == "VOICE")
                n_tech  = sum(1 for s in parsed_scenes if s.paragraph_type == "TECHNICAL")
                print(
                    f"[ProcEx]   {len(parsed_scenes)} scenes parsed — "
                    f"{n_voice} VOICE, {n_tech} TECHNICAL, "
                    f"{len(parsed_scenes) - n_voice - n_tech} NARRATOR/STORY"
                )
            else:
                print("[ProcEx] ⚠ Documentary parse returned no scenes — falling back to ScriptWriter")
                state._is_documentary = False
        else:
            state._is_documentary = False

        return state

    # ── Checkpoint ────────────────────────────────────────────────────────────

    def _checkpoint(self, state: ProcExState) -> None:
        ckpt_dir  = self.cfg.dirs["checkpoints"]
        ckpt_path = os.path.join(ckpt_dir, f"{state.topic_slug}_checkpoint.json")
        try:
            state.save_checkpoint(ckpt_path)
        except Exception as e:
            print(f"[ProcEx] Checkpoint save failed (non-critical): {e}")

    def _clear_checkpoint(self, state: ProcExState) -> None:
        """Delete any existing checkpoint for this topic slug on a fresh run."""
        ckpt_dir  = self.cfg.dirs["checkpoints"]
        ckpt_path = os.path.join(ckpt_dir, f"{state.topic_slug}_checkpoint.json")
        if os.path.exists(ckpt_path):
            try:
                os.remove(ckpt_path)
                print(f"[ProcEx] ✗ Stale checkpoint deleted: {ckpt_path}")
            except Exception as e:
                print(f"[ProcEx] ⚠ Could not delete stale checkpoint: {e}")

    @staticmethod
    def _stage_done(state: ProcExState, agent_name: str) -> bool:
        if agent_name == "DomainRouter"   and state.skill_pack:              return True
        if agent_name == "ScriptWriter"   and state.scenes:                  return True
        if agent_name == "TTSAgent"       and state.audio_path:              return True
        if agent_name == "VisualDirector":
            # Done only when every scene has a visual_prompt AND either:
            #   - MANIM/TEXT_ANIMATION scene with zone_allocation populated (dict not empty)
            #   - IMAGE_GEN/HYBRID scene (zone_allocation not applicable)
            manim_types = {VisualStrategy.MANIM, VisualStrategy.TEXT_ANIMATION}
            if state.scenes and all(s.visual_prompt for s in state.scenes):
                if all(
                    bool(s.zone_allocation) if s.visual_strategy in manim_types else True
                    for s in state.scenes
                ):
                    return True
        if agent_name == "ManimCoder"     and all(
            s.manim_file_path or s.visual_strategy not in (
                VisualStrategy.MANIM, VisualStrategy.TEXT_ANIMATION)
            for s in state.scenes):                                          return True
        if agent_name == "ImageGenAgent"  and all(
            s.image_paths or s.visual_strategy not in (
                VisualStrategy.IMAGE_GEN, VisualStrategy.IMAGE_MANIM_HYBRID)
            for s in state.scenes):                                          return True
        if agent_name == "RendererAgent"  and state.rendered_clips:          return True
        if agent_name == "AssemblerAgent" and state.final_video_path:        return True
        return False