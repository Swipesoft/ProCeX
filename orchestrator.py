"""
orchestrator.py
ProcEx pipeline orchestrator.
Wires all agents in order, handles checkpointing, and saves final video to:
    output/videos/<topic_slug>.mp4
"""
from __future__ import annotations
import os
import json
import time
from state import ProcExState, InputType, VisualStrategy
from config import ProcExConfig
from utils.llm_client import LLMClient
from utils.slug import slugify
from utils.pdf_parser import extract_pdf_text


class ProcExOrchestrator:
    """
    Linear pipeline with checkpoint-resume support.

    Pipeline order:
        1. DomainRouter      → classify domain, load skill pack
        2. ScriptWriter      → generate structured scenes
        3. TTSAgent          → audio + word timestamps
        4. VisualDirector    → per-scene visual strategy + prompts
        5. ManimCoder        → generate Manim .py files for MANIM scenes
        6. ImageGenAgent     → NanoBanana for IMAGE_GEN scenes
        7. RendererAgent     → render all scenes to .mp4 clips
        8. AssemblerAgent    → concat clips + audio → final video
    """

    def __init__(self, cfg: ProcExConfig | None = None):
        self.cfg = cfg or ProcExConfig()
        self.cfg.make_dirs()
        self.llm = LLMClient(self.cfg)

    def run(
        self,
        input_path: str,
        topic_hint: str = "",
        resolution: str = "1080p",
        target_minutes: float = 5.0,
        resume_checkpoint: str | None = None,
    ) -> str:
        """
        Main entry point.

        Args:
            input_path:        path to PDF file
            topic_hint:        optional human-readable topic for slug/title
            resolution:        "720p" | "1080p" | "4K"
            target_minutes:    desired video length
            resume_checkpoint: path to a .json checkpoint to resume from

        Returns:
            path to output .mp4
        """
        start_time = time.time()

        # ── Load or create state ──────────────────────────────────────────
        if resume_checkpoint and os.path.exists(resume_checkpoint):
            print(f"[ProcEx] Resuming from checkpoint: {resume_checkpoint}")
            state = ProcExState.load_checkpoint(resume_checkpoint)
        else:
            state = self._init_state(input_path, topic_hint, resolution, target_minutes)

        # ── Validate config ───────────────────────────────────────────────
        issues = self.cfg.validate()
        for issue in issues:
            print(f"[ProcEx] ⚠ {issue}")

        # ── Run pipeline ──────────────────────────────────────────────────
        pipeline = self._build_pipeline()

        for agent_class in pipeline:
            agent = agent_class(self.cfg, self.llm)
            agent_name = agent_class.__name__

            # Skip if this stage already completed (checkpoint resume)
            if self._stage_done(state, agent_name):
                print(f"[ProcEx] ↩ Skipping {agent_name} (already done)")
                continue

            print(f"\n[ProcEx] ▶ Running {agent_name}...")
            try:
                state = agent.run(state)
            except Exception as e:
                print(f"[ProcEx] ✗ {agent_name} FAILED: {e}")
                self._checkpoint(state)
                raise

            # Checkpoint after each stage
            self._checkpoint(state)
            print(f"[ProcEx] ✓ {agent_name} done")

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

    # ── Pipeline builder ──────────────────────────────────────────────────────

    @staticmethod
    def _build_pipeline():
        # Import here to avoid circular imports
        from agents.domain_router  import DomainRouter
        from agents.script_writer  import ScriptWriter
        from agents.tts_agent      import TTSAgent
        from agents.visual_director import VisualDirector
        from agents.manim_coder    import ManimCoder
        from agents.image_gen_agent import ImageGenAgent
        from agents.renderer       import RendererAgent
        from agents.assembler      import AssemblerAgent

        return [
            DomainRouter,
            ScriptWriter,
            TTSAgent,
            VisualDirector,
            ManimCoder,
            ImageGenAgent,
            RendererAgent,
            AssemblerAgent,
        ]

    # ── State init ────────────────────────────────────────────────────────────

    def _init_state(
        self,
        input_path: str,
        topic_hint: str,
        resolution: str,
        target_minutes: float,
    ) -> ProcExState:
        """Load input document and create fresh ProcExState."""
        state = ProcExState(
            resolution              = resolution,
            target_duration_minutes = target_minutes,
        )

        # Determine input type and load content
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
            # Treat as plain text even without extension
            if os.path.exists(input_path):
                with open(input_path, encoding="utf-8", errors="replace") as f:
                    state.raw_input = f.read()
            else:
                # input_path IS the content
                state.raw_input = input_path
            state.input_type = InputType.TEXT

        # Generate topic slug
        hint_text = topic_hint or state.raw_input[:200]
        state.topic_slug = slugify(hint_text)
        print(f"[ProcEx] Topic slug: {state.topic_slug}")

        return state

    # ── Checkpoint helpers ────────────────────────────────────────────────────

    def _checkpoint(self, state: ProcExState) -> None:
        ckpt_dir  = self.cfg.dirs["checkpoints"]
        ckpt_path = os.path.join(ckpt_dir, f"{state.topic_slug}_checkpoint.json")
        try:
            state.save_checkpoint(ckpt_path)
        except Exception as e:
            print(f"[ProcEx] Checkpoint save failed (non-critical): {e}")

    @staticmethod
    def _stage_done(state: ProcExState, agent_name: str) -> bool:
        """Check if a pipeline stage has already completed (for resume)."""
        if agent_name == "DomainRouter"    and state.skill_pack:             return True
        if agent_name == "ScriptWriter"    and state.scenes:                 return True
        if agent_name == "TTSAgent"        and state.audio_path:             return True
        if agent_name == "VisualDirector"  and all(
            s.visual_prompt for s in state.scenes):                          return True
        if agent_name == "ManimCoder"      and all(
            s.manim_file_path or s.visual_strategy not in (
                VisualStrategy.MANIM, VisualStrategy.TEXT_ANIMATION)
            for s in state.scenes):                                          return True
        if agent_name == "ImageGenAgent"   and all(
            s.image_paths or s.visual_strategy not in (
                VisualStrategy.IMAGE_GEN, VisualStrategy.IMAGE_MANIM_HYBRID)
            for s in state.scenes):                                          return True
        if agent_name == "RendererAgent"   and state.rendered_clips:         return True
        if agent_name == "AssemblerAgent"  and state.final_video_path:       return True
        return False
