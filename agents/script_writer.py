"""
agents/script_writer.py
Generates a structured, domain-aware narration script broken into scenes.
Output: state.scenes populated, state.full_script_text set.
"""
from __future__ import annotations
import json
from state import ProcExState, Scene, VisualStrategy
from config import ProcExConfig
from utils.llm_client import LLMClient
from agents.base_agent import BaseAgent


SCRIPT_SYSTEM = """You are a world-class educational documentary scriptwriter.
Your scripts are used to generate cinematic explainer videos with timed animations.

CRITICAL OUTPUT FORMAT:
Return ONLY valid JSON. No preamble, no markdown fences, no commentary.

Format:
{{
  "title": "Human-readable video title",
  "total_estimated_duration_minutes": <float>,
  "scenes": [
    {{
      "id": 1,
      "title": "Scene title",
      "narration_text": "Full narration text for this scene. Natural, spoken-word style.",
      "duration_seconds": <float>,
      "visual_hint": "Brief 1-sentence hint about what should be shown visually"
    }},
    ...
  ]
}}

RULES:
- narration_text must be natural spoken English (not bullet points).
- Each scene = one cohesive idea, ~25-55 seconds of narration.
- duration_seconds = approximate TTS duration (estimate 2.8 words/second).
- First scene: compelling hook that states the core question/problem.
- Last scene: synthesis + memorable takeaway.
- visual_hint: helps the VisualDirector — be specific about what concepts need visual representation.
- DO NOT decide visual strategy here — that is the VisualDirector's job.
- Match target_scenes count as closely as possible.
"""


def _build_script_prompt(state: ProcExState, target_scenes: int) -> str:
    skill = state.skill_pack
    instructions = skill.get("script_instructions", "")
    domain = state.domain.value

    return f"""Domain: {domain}
Target video duration: {state.target_duration_minutes} minutes
Target scene count: {target_scenes} scenes

Domain-specific writing instructions:
{instructions}

---
SOURCE MATERIAL:
{state.raw_input[:50_000]}

---
Write a complete, engaging, narration-ready script for this content.
The narration should feel like a brilliant professor explaining this to a curious student.
Cinematic, precise, and genuinely illuminating.
"""


class ScriptWriter(BaseAgent):
    name = "ScriptWriter"

    def run(self, state: ProcExState) -> ProcExState:
        scenes_per_min = self.cfg.scenes_per_minute
        target_scenes  = max(5, int(state.target_duration_minutes * scenes_per_min))

        self._log(f"Writing script: {state.target_duration_minutes}min target, ~{target_scenes} scenes")

        system = SCRIPT_SYSTEM
        user   = _build_script_prompt(state, target_scenes)

        for attempt in range(self.cfg.max_llm_retries):
            try:
                result = self.llm.complete_json(system, user, max_tokens=16384, temperature=0.75)
                break
            except Exception as e:
                self._log(f"Attempt {attempt+1} failed: {e}")
                if attempt == self.cfg.max_llm_retries - 1:
                    raise

        raw_scenes = result.get("scenes", [])
        if not raw_scenes:
            raise ValueError("ScriptWriter returned no scenes")

        state.scenes = []
        full_text_parts = []

        for s in raw_scenes:
            scene = Scene(
                id               = s.get("id", len(state.scenes) + 1),
                narration_text   = s.get("narration_text", "").strip(),
                duration_seconds = float(s.get("duration_seconds", 35.0)),
                visual_prompt    = s.get("visual_hint", ""),  # temp — VisualDirector overwrites
            )
            state.scenes.append(scene)
            full_text_parts.append(scene.narration_text)

        state.full_script_text = " ".join(full_text_parts)

        total = sum(s.duration_seconds for s in state.scenes)
        self._log(
            f"Script ready: {len(state.scenes)} scenes, "
            f"~{total/60:.1f} min estimated, "
            f"~{len(state.full_script_text.split())} words"
        )
        return state
