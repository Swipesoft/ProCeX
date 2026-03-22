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
- duration_seconds = the TTS duration of THIS SINGLE SCENE ONLY (not the whole video).
  Calculate it as: word_count_of_this_scene / 2.8 (2.8 words per second).
  A 40-word scene = 14.3s. A 55-word scene = 19.6s. A 80-word scene = 28.6s.
  NEVER write the total video duration as a scene's duration_seconds.
  All scene durations must SUM to approximately (target_minutes * 60) seconds.
- First scene: compelling hook that states the core question/problem.
- Last scene: synthesis + memorable takeaway.
- visual_hint: helps the VisualDirector — be specific about what concepts need visual representation.
- DO NOT decide visual strategy here — that is the VisualDirector's job.
- Match target_scenes count as closely as possible.

═══════════════════════════════════════════════════════════
INTER-SCENE COHERENCE RULES  (the most violated rules — read carefully)
═══════════════════════════════════════════════════════════

These rules govern how scenes connect to each other. A script where every
scene reads well in isolation but the whole feels disjointed is a FAILED script.

RULE C1 — NO DUPLICATE NARRATION
  Every scene must contain UNIQUE content. No sentence, phrase, or idea
  may appear verbatim or near-verbatim in more than one scene.
  If you catch yourself repeating a setup from a previous scene, you have
  written the wrong scene. Delete and rewrite from a new angle.

RULE C2 — EACH SCENE ENDS ON A HOOK FOR THE NEXT
  The last sentence of every scene (except the last) must create forward
  momentum — a question, a contradiction, a consequence that the next
  scene will resolve. The viewer should feel pulled forward, not dropped.
  ✓ DO:   "...and that proof held for exactly 2,000 years. [pauses]
           Until one man in 1830 noticed something no one else had."
  ✗ DON'T: "...Euclid's work was very influential in ancient times."
            (dead end — the viewer has no reason to keep watching)

RULE C3 — CALLBACK THREADING
  Later scenes must echo or callback to specific language, images, or
  ideas from earlier scenes. This creates the feeling of a single
  continuous story rather than a series of unrelated facts.
  ✓ DO:   Scene 1 introduces "the fifth postulate". Scene 4 opens with:
           "That fifth postulate — the one everyone accepted — was wrong."
  ✗ DON'T: Scene 4 introduces the fifth postulate as if for the first time.

RULE C4 — SENTENCE-LEVEL FLOW WITHIN A SCENE
  Every sentence within a scene must follow from the previous one.
  No sentence may introduce a new subject without a bridging clause.
  ✓ DO:   "He built a closed logical system from five axioms. [pauses]
           But the king wanted a shortcut. Euclid had an answer for that too."
  ✗ DON'T: "He built a closed logical system from five axioms. [pauses]
            King Ptolemy asked for a faster way." (subject switch, no bridge)

RULE C5 — WRITE THE SCRIPT IN ORDER, READ IT BACK AS ONE PIECE
  Before finalising, mentally read the entire script as one continuous
  narration. If any transition between scenes feels abrupt, add a bridging
  phrase at the END of the preceding scene or the START of the next.
  A bridge can be as short as: "And that changed everything." /
  "But here is what nobody expected." / "The answer arrived from an
  unlikely direction."

RULE C6 — EMOTIONAL ARC ACROSS THE WHOLE VIDEO
  The video must have a single emotional trajectory — curiosity → tension
  → revelation → awe, or setup → complication → resolution.
  Map each scene to a position on this arc BEFORE writing it.
  A scene that breaks the emotional trajectory is the wrong scene.
═══════════════════════════════════════════════════════════
"""


def _load_style_pack(style_id: str) -> dict:
    """Load a style YAML from skills/styles/. Returns empty dict if not found."""
    import os, yaml  # yaml = PyYAML, already in requirements
    styles_dir = os.path.join(os.path.dirname(__file__), "..", "skills", "styles")
    path = os.path.join(styles_dir, f"{style_id}.yaml")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


AUTO_SELECT_SYSTEM = """You are a video content strategist. Given a topic, domain, and target duration,
select the single best presentation style for the video.

Available styles:
  tiktok-scifi    — sci-fi dramatic narration, 45s-1.5min, for concepts that sound cool/futuristic
  tiktok-thriller — historical thriller story arc, 45s-2min, for topics with real discovery moments
  youtube-tutorial — warm deep explainer, 4-10min, for topics requiring progressive learning

Return ONLY a JSON object with one key:
{"style": "<style_id>"}
No preamble, no markdown fences."""


def _auto_select_style(state: "ProcExState", llm) -> str:
    """Ask the LLM to pick the best style based on topic + domain + duration."""
    from utils.llm_client import LLMClient

    # Load auto_select_signals from each style to give LLM rich context
    style_ids = ["tiktok-scifi", "tiktok-thriller", "youtube-tutorial"]
    signals_block = ""
    for sid in style_ids:
        pack = _load_style_pack(sid)
        signals = pack.get("auto_select_signals", "")
        if signals:
            signals_block += f"\n[{sid}]\n{signals.strip()}\n"

    user_prompt = f"""Topic: {state.topic_slug.replace("_", " ")}
Domain: {state.domain.value}
Target duration: {state.target_duration_minutes} minutes
Full text excerpt: {state.raw_input[:800]}

Style selection signals:
{signals_block}

Select the single best style_id for this video."""

    try:
        result = llm.complete_json(
            AUTO_SELECT_SYSTEM, user_prompt,
            max_tokens=128, temperature=0.3, primary_provider="gemini"
        )
        chosen = result.get("style", "youtube-tutorial").strip()
        if chosen not in style_ids:
            chosen = "youtube-tutorial"
        return chosen
    except Exception:
        # Fallback heuristic
        if state.target_duration_minutes <= 1.5:
            return "tiktok-scifi"
        elif state.target_duration_minutes <= 2.0:
            return "tiktok-thriller"
        return "youtube-tutorial"


def _build_script_prompt(state: "ProcExState", target_scenes: int) -> str:
    skill      = state.skill_pack
    style      = state.style_pack
    domain     = state.domain.value

    domain_instructions = skill.get("script_instructions", "")
    style_tone          = style.get("narration_tone", "")
    style_structure     = style.get("script_structure", "")
    style_pacing        = style.get("pacing", "")
    vocal_palette       = style.get("vocal_palette", "")
    narrative_arcs      = style.get("narrative_arcs", "")   # thriller only
    style_name          = style.get("display_name", "Standard")

    # Build style injection block
    style_block = ""
    if style:
        style_block = f"""
═══════════════════════════════════════════════════════════
PRESENTATION STYLE: {style_name.upper()}
═══════════════════════════════════════════════════════════

NARRATION TONE & RULES:
{style_tone}

VOCAL SOUNDS PALETTE (embed these directly in narration_text):
{vocal_palette}
{"NARRATIVE ARC FORMATS (pick one and commit to it):" + chr(10) + narrative_arcs if narrative_arcs else ""}

SCRIPT STRUCTURE FOR THIS STYLE:
{style_structure}

PACING GUIDANCE:
{style_pacing}
═══════════════════════════════════════════════════════════
"""

    return f"""Domain: {domain}
Presentation style: {style_name}
Target video duration: {state.target_duration_minutes} minutes
Target scene count: {target_scenes} scenes

Domain-specific writing instructions:
{domain_instructions}
{style_block}
---
SOURCE MATERIAL:
{state.raw_input[:50_000]}

---
Write a complete narration-ready script for this content in the presentation
style specified above. The vocal sounds in the palette go DIRECTLY into
narration_text — Gemini TTS will render them as real audio.
Match the tone and energy of the style exactly.
"""



class ScriptWriter(BaseAgent):
    name = "ScriptWriter"

    def run(self, state: ProcExState) -> ProcExState:
        # ── 1. Resolve presentation style ─────────────────────────────────────
        requested_style = getattr(state, "presentation_style", None) or                           getattr(self.cfg, "presentation_style", "auto")

        if requested_style == "auto":
            self._log("Style: auto — asking LLM to select best style...")
            resolved_style = _auto_select_style(state, self.llm)
            self._log(f"Style: auto → resolved to '{resolved_style}'")
        else:
            resolved_style = requested_style

        style_pack = _load_style_pack(resolved_style)
        if style_pack:
            self._log(f"Style loaded: {style_pack.get('display_name', resolved_style)}")
        else:
            self._log(f"Style '{resolved_style}' not found — using default narration")

        state.presentation_style = resolved_style
        state.style_pack         = style_pack

        # Override scenes_per_minute from style if specified
        style_spm      = style_pack.get("scenes_per_minute", None)
        scenes_per_min = style_spm if style_spm else self.cfg.scenes_per_minute
        target_scenes  = max(3, int(state.target_duration_minutes * scenes_per_min))

        self._log(
            f"Writing script: {state.target_duration_minutes}min target, "
            f"~{target_scenes} scenes, style={resolved_style}"
        )

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
            # Estimate duration from word count if LLM-provided value seems wrong.
            # LLMs often write durations that are multiples of the total target
            # rather than per-scene values (e.g. 60s per scene for a 1min video).
            narration = s.get("narration_text", "").strip()
            word_count = len(narration.split())
            tts_estimated = round(word_count / 2.8, 1)   # 2.8 words/sec

            llm_duration = float(s.get("duration_seconds", tts_estimated))
            scene = Scene(
                id               = s.get("id", len(state.scenes) + 1),
                narration_text   = narration,
                duration_seconds = llm_duration,
                visual_prompt    = s.get("visual_hint", ""),  # temp — VisualDirector overwrites
            )
            state.scenes.append(scene)
            full_text_parts.append(narration)

        # ── Duration sanity check ─────────────────────────────────────────────
        # If the LLM's scene durations sum to more than 1.5× the target,
        # it wrote per-video durations instead of per-scene ones.
        # Rescale proportionally so the total matches the target.
        target_secs = state.target_duration_minutes * 60.0
        raw_total   = sum(s.duration_seconds for s in state.scenes)

        if raw_total > target_secs * 1.5 or raw_total < target_secs * 0.4:
            self._log(
                f"Duration mismatch: LLM wrote {raw_total:.0f}s total, "
                f"target={target_secs:.0f}s — rescaling scene durations"
            )
            # Rescale each scene proportionally, but also cross-check against
            # word-count estimate — take the max so clips aren't too short.
            for scene in state.scenes:
                word_count    = len(scene.narration_text.split())
                tts_estimated = round(word_count / 2.8, 1)
                scaled        = round((scene.duration_seconds / raw_total) * target_secs, 1)
                scene.duration_seconds = max(scaled, tts_estimated, 10.0)

        state.full_script_text = " ".join(full_text_parts)

        # ── Duplicate scene detection ─────────────────────────────────────────
        # The LLM occasionally returns scenes with identical or near-identical
        # narration. Detect and log these so the user knows the script is bad.
        seen_hashes = {}
        duplicates  = []
        for scene in state.scenes:
            # Use first 120 chars as fingerprint — catches verbatim copies
            fingerprint = scene.narration_text[:120].strip().lower()
            if fingerprint in seen_hashes:
                duplicates.append((seen_hashes[fingerprint], scene.id))
                self._log(
                    f"WARNING: Scene {scene.id} narration is a near-duplicate "
                    f"of Scene {seen_hashes[fingerprint]} — script quality issue"
                )
            else:
                seen_hashes[fingerprint] = scene.id

        if duplicates:
            self._log(
                f"Script has {len(duplicates)} duplicate scene(s) — "
                f"consider re-running ScriptWriter. Duplicate pairs: {duplicates}"
            )

        total = sum(s.duration_seconds for s in state.scenes)
        self._log(
            f"Script ready: {len(state.scenes)} scenes, "
            f"~{total/60:.1f} min estimated, "
            f"~{len(state.full_script_text.split())} words"
        )
        return state