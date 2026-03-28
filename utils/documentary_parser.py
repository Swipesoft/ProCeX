"""
utils/documentary_parser.py

Detects whether a PDF (already extracted to text) is a ProcEx documentary
PDF and if so, parses the tagged paragraphs directly into Scene objects.

Documentary PDFs are produced by DeepDocumentaryAgent and have a specific
structure where every paragraph is preceded by a type label:

    ▸ NARRATOR
    The narrator bridge text...

    ▸ STORY
    The biographical story text...

    ▸ TECHNICAL
    The science/math explanation...

    ▸ VOICE: Einstein
    I have always believed the universe is elegant...

This parser:
  1. Detects the documentary format (at least 3 ▸ type labels present)
  2. Parses each labelled paragraph into a Scene object
  3. Sets scene.paragraph_type and scene.tts_voice on each scene
  4. Estimates scene.duration_seconds from word count

Paragraph type → TTS voice mapping:
  NARRATOR  → tts_voice = ""          (Aoede, global default)
  STORY     → tts_voice = ""          (Aoede, global default)
  TECHNICAL → tts_voice = ""          (Aoede, global default)
  VOICE: X  → tts_voice = "Fenrir"   (male Gemini voice for critics/characters)

Why Fenrir for VOICE paragraphs? Fenrir is one of Gemini's prebuilt male
voices with a notably deeper register, providing clear contrast with Aoede.
The config.py DOCUMENTARY_VOICE_MAP can override this per character name
if different voices are wanted for different historical figures.

When is ScriptWriter bypassed?
  If is_documentary_pdf() returns True, the orchestrator skips ScriptWriter's
  LLM re-generation entirely. The scenes are already scripted — the LLM
  would only dilute the carefully crafted multi-voice structure.
  VisualDirector and TTSAgent still run normally on the parsed scenes.
"""
from __future__ import annotations

import re
from typing import Optional

# ── Voice assignments ──────────────────────────────────────────────────────────
# Default voice for [VOICE: X] paragraphs — male to contrast Aoede narrator.
# Fenrir is used because it is consistently available in gemini-2.5-flash-preview-tts
# and has a notably deeper register than Charon. Override in config.py via
# gemini_tts_voice_male if you want a different character voice.
import os
DOCUMENTARY_VOICE_DEFAULT = os.environ.get("GEMINI_TTS_VOICE_MALE", "Fenrir")

# Optional per-character overrides — extend as needed
# Any character not listed here gets DOCUMENTARY_VOICE_DEFAULT
DOCUMENTARY_VOICE_MAP: dict[str, str] = {
    # "Einstein":    "Fenrir",
    # "Bohr":        "Fenrir",
    # "Lobachevsky": "Orus",
}

# Detection threshold — min number of ▸ type labels to qualify as documentary
DETECTION_MIN_LABELS = 3

# TTS words-per-second estimate (same as ScriptWriter)
WORDS_PER_SECOND = 2.8


def is_documentary_pdf(text: str) -> bool:
    """
    Return True if the extracted PDF text looks like a ProcEx documentary PDF.
    Detects by counting ▸ LABEL lines.
    """
    labels = re.findall(
        r'(?m)^▸\s+(?:NARRATOR|STORY|TECHNICAL|VOICE)',
        text,
    )
    return len(labels) >= DETECTION_MIN_LABELS


def parse_documentary_scenes(
    text:          str,
    target_minutes: float = 6.0,
) -> list:
    """
    Parse a documentary PDF text into a list of Scene objects.

    Each ▸ LABEL block becomes one Scene with:
      - narration_text   = the paragraph text (tag stripped)
      - paragraph_type   = NARRATOR | STORY | TECHNICAL | VOICE
      - tts_voice        = "" for narration types, "Fenrir" for VOICE
      - duration_seconds = estimated from word count
      - visual_prompt    = auto-generated hint based on paragraph_type

    Returns list[Scene] — ready to assign to state.scenes.
    """
    # Late import to avoid circular dependency at module level
    from state import Scene

    # ── Split text on ▸ LABEL lines ───────────────────────────────────────────
    # Pattern matches lines like:
    #   ▸ NARRATOR
    #   ▸ STORY
    #   ▸ TECHNICAL
    #   ▸ VOICE: EINSTEIN
    split_pattern = re.compile(
        r'(?m)^▸\s+(NARRATOR|STORY|TECHNICAL|VOICE(?::\s*[A-Za-z\s]+)?)\s*$'
    )

    # Find all label positions
    matches = list(split_pattern.finditer(text))
    if not matches:
        return []

    paragraphs = []
    for i, match in enumerate(matches):
        label_raw = match.group(1).strip()
        # Text runs from end of this label to start of next label (or end of doc)
        content_start = match.end()
        content_end   = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content       = text[content_start:content_end].strip()

        # Skip empty paragraphs
        if not content:
            continue

        # Parse label into type + optional speaker
        if label_raw.upper().startswith("VOICE"):
            para_type = "VOICE"
            # Extract speaker name from "VOICE: Einstein" or just "VOICE"
            speaker_match = re.match(r'VOICE\s*:\s*(.+)', label_raw, re.IGNORECASE)
            speaker = speaker_match.group(1).strip().title() if speaker_match else "Unknown"
        else:
            para_type = label_raw.upper()
            speaker   = None

        paragraphs.append((para_type, speaker, content))

    if not paragraphs:
        return []

    # ── Build Scene objects ───────────────────────────────────────────────────
    scenes     = []
    scene_id   = 1

    for para_type, speaker, content in paragraphs:
        word_count       = len(content.split())
        duration_seconds = round(max(word_count / WORDS_PER_SECOND, 8.0), 1)

        # Resolve TTS voice
        if para_type == "VOICE":
            voice = (
                DOCUMENTARY_VOICE_MAP.get(speaker or "", "")
                or DOCUMENTARY_VOICE_DEFAULT
            )
        else:
            voice = ""   # Aoede via global cfg default

        # Auto visual_prompt hint based on paragraph_type
        visual_prompt = _auto_visual_hint(para_type, speaker, content)

        scene = Scene(
            id               = scene_id,
            narration_text   = content,
            duration_seconds = duration_seconds,
            visual_prompt    = visual_prompt,
            paragraph_type   = para_type,
            tts_voice        = voice,
        )
        scenes.append(scene)
        scene_id += 1

    return scenes


def _auto_visual_hint(
    para_type: str,
    speaker:   Optional[str],
    content:   str,
) -> str:
    """
    Generate a concise visual_prompt hint for the VisualDirector
    based on paragraph type. VisualDirector will refine this further.
    """
    # Take first 120 chars of content as context
    snippet = content[:120].replace("\n", " ").strip()

    if para_type == "NARRATOR":
        return (
            f"Minimal cinematic title card. Single evocative line of text. "
            f"Dark background, electric cyan accent. Context: {snippet}"
        )
    elif para_type == "STORY":
        return (
            f"Historical cinematic scene. Biographical context, era-appropriate "
            f"setting, atmospheric. Context: {snippet}"
        )
    elif para_type == "TECHNICAL":
        return (
            f"Mathematical/scientific diagram or equation animation. "
            f"Manim-style. Clean, precise, educational. Context: {snippet}"
        )
    elif para_type == "VOICE":
        return (
            f"Dramatic portrait or historical scene of {speaker or 'the figure'}. "
            f"Cinematic, era-appropriate, emotionally resonant. "
            f"First-person perspective. Context: {snippet}"
        )
    else:
        return snippet