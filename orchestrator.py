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
from utils.slug import slugify
from utils.pdf_parser import extract_pdf_text


class ProcExOrchestrator:

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
        start_time = time.time()

        # ── Load or create state ──────────────────────────────────────────
        if resume_checkpoint and os.path.exists(resume_checkpoint):
            print(f"[ProcEx] Resuming from checkpoint: {resume_checkpoint}")
            state = ProcExState.load_checkpoint(resume_checkpoint)
        else:
            state = self._init_state(input_path, topic_hint, resolution, target_minutes)

        for issue in self.cfg.validate():
            print(f"[ProcEx] ⚠ {issue}")

        # ════════════════════════════════════════════════════════════════
        # Stage 1 — DomainRouter  [sequential]
        # ════════════════════════════════════════════════════════════════
        state = self._run_agent("DomainRouter", state)

        # ════════════════════════════════════════════════════════════════
        # Stage 2 — ScriptWriter  [sequential]
        # ════════════════════════════════════════════════════════════════
        state = self._run_agent("ScriptWriter", state)

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
        agent       = agent_class(self.cfg, self.llm)

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

    def _init_state(self, input_path, topic_hint, resolution, target_minutes):
        state = ProcExState(
            resolution              = resolution,
            target_duration_minutes = target_minutes,
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
        return state

    # ── Checkpoint ────────────────────────────────────────────────────────────

    def _checkpoint(self, state: ProcExState) -> None:
        ckpt_dir  = self.cfg.dirs["checkpoints"]
        ckpt_path = os.path.join(ckpt_dir, f"{state.topic_slug}_checkpoint.json")
        try:
            state.save_checkpoint(ckpt_path)
        except Exception as e:
            print(f"[ProcEx] Checkpoint save failed (non-critical): {e}")

    @staticmethod
    def _stage_done(state: ProcExState, agent_name: str) -> bool:
        if agent_name == "DomainRouter"   and state.skill_pack:              return True
        if agent_name == "ScriptWriter"   and state.scenes:                  return True
        if agent_name == "TTSAgent"       and state.audio_path:              return True
        if agent_name == "VisualDirector" and all(
            s.visual_prompt for s in state.scenes):                          return True
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