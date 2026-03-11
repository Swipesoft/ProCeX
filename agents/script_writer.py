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

═══════════════════════════════════════════════════════════
MATHEMATICAL NARRATION RULES  (apply to ALL domains)
═══════════════════════════════════════════════════════════

These rules govern how any formula, equation, or symbolic expression
is spoken in narration_text. They apply equally to mathematics, machine
learning, physics, economics, biology, and any other quantitative domain.

RULE 1 — DEFINE BEFORE YOU USE
  The first time a symbol or formula appears in narration, introduce it
  with a plain-English "where" clause that assigns a human-readable name
  to each symbol. Never assume the viewer reads the screen.

  ✓ DO:   "The network computes an output — where W is the weight matrix,
            x is the input vector, and b the bias term."
  ✗ DON'T: "The network computes W transpose x plus b."

RULE 2 — NAME THE OPERATION, NOT THE SYMBOLS
  After defining symbols, refer to operations by their mathematical name.
  Never read out a formula's structure as if spelling it aloud.

  ✓ DO:   "This is a dot product of the weights and input, offset by a bias —
            the classic affine transformation."
  ✓ DO:   "We decompose the matrix into its eigenvectors and eigenvalues."
  ✓ DO:   "The softmax normalises these scores into a probability distribution."
  ✓ DO:   "We compute the gradient of the loss with respect to each weight."
  ✓ DO:   "The convolution slides a kernel across the input, summing overlaps."
  ✗ DON'T: "We compute W transpose x plus b."
  ✗ DON'T: "A equals Q lambda Q inverse."
  ✗ DON'T: "e to the z-i over the sum of e to the z-j."
  ✗ DON'T: "partial L partial W."

RULE 3 — GREEK LETTERS: USE THEIR MEANING, NOT THEIR NAME
  Replace Greek letter names with their contextual meaning unless the
  letter name itself IS the standard spoken form (e.g. "alpha" for
  learning rate is acceptable colloquially).

  ✓ DO:   "the sigmoid activation"         (not "sigma of z")
  ✓ DO:   "the learning rate"              (not "eta" or "the learning rate eta")
  ✓ DO:   "the mean of the distribution"   (not "mu")
  ✓ DO:   "the standard deviation"         (not "sigma" when referring to spread)
  ✓ DO:   "summing across all elements"    (not "sum over i from 1 to n")

RULE 4 — SUBSCRIPTS, SUPERSCRIPTS, AND INDICES: DESCRIBE THE ROLE
  Never read index notation aloud. Describe what it means structurally.

  ✓ DO:   "for each element in the sequence"     (not "x sub i")
  ✓ DO:   "raised to the second power"           (not "x superscript 2")
  ✓ DO:   "the weight connecting layer l to l+1" (not "W superscript l")
  ✓ DO:   "each training example"                (not "the i-th sample")

RULE 5 — RE-USE THE HANDLE, NOT THE FORMULA
  After the first definition-and-naming of a concept, subsequent
  references use only the established handle — never re-spell the formula.

  ✓ DO (second mention):  "Applying this affine transformation again..."
  ✓ DO (second mention):  "Back-propagating this gradient..."
  ✗ DON'T (second mention): "Applying W transpose x plus b again..."

RULE 6 — FRACTIONS AND RATIOS: STATE THE RELATIONSHIP
  ✓ DO:   "normalised by the total count"      (not "divided by N")
  ✓ DO:   "the ratio of signal to noise"       (not "S over N")
  ✓ DO:   "scaled by one over the square root of the dimension"

RULE 7 — LIMITS, SUMMATIONS, INTEGRALS: NAME THE SWEEP
  ✓ DO:   "summing the contributions of every training example"
  ✓ DO:   "integrating the probability density across all outcomes"
  ✓ DO:   "as the step size approaches zero"
  ✗ DON'T: "the integral from zero to infinity of f of x dx"
═══════════════════════════════════════════════════════════

GENERAL SCRIPT RULES:
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
                result = self.llm.complete_json(system, user, max_tokens=16384, temperature=0.75, primary_provider="gemini")
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