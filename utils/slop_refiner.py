"""
utils/slop_refiner.py
SlopRefiner — post-script, pre-TTS narration quality pass.

Runs between ScriptWriter / documentary-parser and TTSAgent.
Detects and corrects LLM slop patterns in scene.narration_text using a
two-pass strategy:

  Pass 1 — Regex pre-filter (free, instantaneous):
    Each pattern in anti_slop.yaml has a regex_hint.
    Scenes that match zero hints are skipped entirely.
    Only flagged scenes proceed to Pass 2.

  Pass 2 — LLM correction (one call per flagged scene):
    A compact prompt lists only the remedy lines (not full examples).
    The LLM rewrites the narration in-place.
    Duration is recalculated from the new word count.

Design principles:
  - Non-destructive: if LLM correction fails, original text is kept.
  - Duration-preserving: recalculates scene.duration_seconds after rewrite.
  - Context-efficient: ~180 tokens of pattern overhead per scene call.
  - Global: applies to all pipeline modes (research, documentary, PDF input).
"""
from __future__ import annotations

import os
import re
import yaml
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from state import ProcExState, Scene
    from utils.llm_client import LLMClient
    from config import ProcExConfig


# ── Constants ─────────────────────────────────────────────────────────────────
WORDS_PER_SECOND  = 2.8   # matches documentary_parser and script_writer
MIN_SCENE_SECS    = 8.0   # never shorten below this
ANTI_SLOP_YAML    = os.path.join(
    os.path.dirname(__file__), "..", "skills", "refine", "anti_slop.yaml"
)

# ── Compact correction prompt (remedies only, no examples) ────────────────────
_CORRECTION_SYSTEM = """\
You are a narration editor for an educational documentary video.
Your job is to rewrite the provided narration to eliminate specific slop patterns
while preserving ALL factual content, emotional beats, pacing, and spoken-word rhythm.

RULES:
1. Apply ONLY the remedies listed below — do not make other changes.
2. Keep the same approximate length (±15% word count is acceptable).
3. Output ONLY the corrected narration text — no preamble, no explanation,
   no quotation marks around the result.
4. Write for spoken delivery: natural rhythm, no bullet points, no headers.
5. Preserve any vocal markers already present ([pauses], oof, huh, wait —).
"""

_CORRECTION_USER = """\
NARRATION TO CORRECT:
{narration}

SLOP PATTERNS DETECTED — apply these remedies:
{remedies}

Rewrite the narration now:
"""


# ── Public interface ───────────────────────────────────────────────────────────

def refine_scenes(
    state: "ProcExState",
    llm:   "LLMClient",
    cfg:   "ProcExConfig",
) -> "ProcExState":
    """
    Run the two-pass slop refiner on all scenes in state.
    Modifies scene.narration_text and scene.duration_seconds in-place.
    Returns state (for chaining).
    """
    patterns = _load_patterns()
    if not patterns:
        print("[SlopRefiner] No patterns loaded — skipping")
        return state

    if not state.scenes:
        return state

    flagged   = 0
    corrected = 0

    for scene in state.scenes:
        text = (scene.narration_text or "").strip()
        if not text:
            continue

        # Pass 1: regex pre-filter
        triggered = _detect_patterns(text, patterns)
        if not triggered:
            continue

        flagged += 1
        pattern_names = [p["name"] for p in triggered]
        print(
            f"[SlopRefiner] Scene {scene.id}: "
            f"{len(triggered)} pattern(s) flagged — "
            f"{', '.join(pattern_names)}"
        )

        # Pass 2: LLM correction
        try:
            corrected_text = _correct_scene(text, triggered, llm)
            if corrected_text and corrected_text.strip() != text:
                scene.narration_text   = corrected_text.strip()
                scene.duration_seconds = _recalc_duration(corrected_text)
                corrected += 1
                print(
                    f"[SlopRefiner] Scene {scene.id}: corrected "
                    f"({len(text.split())}w → {len(corrected_text.split())}w)"
                )
            else:
                print(f"[SlopRefiner] Scene {scene.id}: no change from LLM")
        except Exception as e:
            print(f"[SlopRefiner] Scene {scene.id}: correction failed ({e}) — keeping original")

    print(
        f"[SlopRefiner] Done — {flagged}/{len(state.scenes)} scenes flagged, "
        f"{corrected} corrected"
    )
    return state


# ── Internal helpers ───────────────────────────────────────────────────────────

def _load_patterns() -> list[dict]:
    """Load and validate anti-slop patterns from yaml."""
    path = os.path.abspath(ANTI_SLOP_YAML)
    if not os.path.exists(path):
        print(f"[SlopRefiner] WARNING: anti_slop.yaml not found at {path}")
        return []
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    patterns = data.get("anti_slop_biotics", [])
    # Compile regex hints ahead of time
    compiled = []
    for p in patterns:
        hint = p.get("regex_hint", "")
        if not hint:
            continue
        try:
            p["_regex"] = re.compile(hint, re.MULTILINE)
            compiled.append(p)
        except re.error as e:
            print(f"[SlopRefiner] Bad regex in pattern {p.get('id')}: {e}")
    return compiled


def _detect_patterns(text: str, patterns: list[dict]) -> list[dict]:
    """Return list of patterns whose regex_hint matches the text."""
    triggered = []
    for p in patterns:
        rx = p.get("_regex")
        if rx and rx.search(text):
            triggered.append(p)
    return triggered


def _correct_scene(
    text:     str,
    patterns: list[dict],
    llm:      "LLMClient",
) -> str:
    """Call LLM with compact remedy list. Returns corrected text."""
    # Build compact remedy block — just name + remedy, no examples
    remedy_lines = "\n".join(
        f"• [{p['name']}] {p['remedy']}"
        for p in patterns
    )

    user_prompt = _CORRECTION_USER.format(
        narration = text,
        remedies  = remedy_lines,
    )

    result = llm.complete(
        system_prompt    = _CORRECTION_SYSTEM,
        user_prompt      = user_prompt,
        json_mode        = False,
        max_tokens       = max(2048, int(len(text.split()) * 2.5)),
        temperature      = 0.3,
        primary_provider = "gemma",   # Claude is strongest at precise prose editing
    )
    return result.strip()


def _recalc_duration(text: str) -> float:
    """Recalculate scene duration from word count after narration rewrite."""
    word_count = len(text.split())
    return round(max(word_count / WORDS_PER_SECOND, MIN_SCENE_SECS), 1)