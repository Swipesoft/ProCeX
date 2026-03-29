"""
utils/documentary_parser.py — parses ProcEx documentary PDFs into Scene objects.

IMPORTANT — PyMuPDF glyph mangling:
  ReportLab writes the paragraph bullet (U+25B8 ▸) via ZapfDingbats encoding.
  PyMuPDF decodes it as the letter "I" on most systems.
  _LABEL_PREFIX covers both the original glyph and all known artefacts.
"""
from __future__ import annotations
import re, os
from typing import Optional

DOCUMENTARY_VOICE_DEFAULT = os.environ.get("GEMINI_TTS_VOICE_MALE", "Fenrir")
DOCUMENTARY_VOICE_MAP: dict[str, str] = {}
DETECTION_MIN_LABELS = 3
WORDS_PER_SECOND = 2.8

# Character class covering ▸ (original) and all PDF-extractor artefacts.
# Written as a string so we can embed it in compiled patterns cleanly.
_LABEL_CHARS = r"[▸■●◆Il>]"


def is_documentary_pdf(text: str) -> bool:
    """Return True if extracted PDF text looks like a ProcEx documentary PDF."""
    pattern = re.compile(
        r"(?m)^" + _LABEL_CHARS + r"\s+(?:NARRATOR|STORY|TECHNICAL|VOICE)"
    )
    return len(pattern.findall(text)) >= DETECTION_MIN_LABELS


def _enrich_for_tts(content: str, para_type: str) -> str:
    """
    Inject tiktok-thriller vocal markers into documentary paragraph text
    before TTS. Keeps the PDF clean — markers applied at parse time only.
    """
    text = content.strip()

    # [pauses] before 4-digit years
    text = re.sub(
        r"\b(In|By|Around|After|Before|During|Since|Until|From|At)"
        r"\s+(1[0-9]{3}|20[0-2][0-9])\b",
        r"\1 [pauses] \2", text,
    )
    # [pauses] before proper names after "named / called"
    text = re.sub(
        r"\b(named|called)\s+([A-Z][a-zA-ZÀ-ÿ]+)",
        r"\1 [pauses] \2", text,
    )
    # [pauses] after first em-dash
    text = re.sub(r"\s*—\s+([a-zA-Z])", r" — [pauses] \1", text, count=1)

    # [pauses] before short dramatic fragments (≤ 4 words)
    def _pause_frag(m):
        frag = m.group(1).strip()
        return (f". [pauses] {frag}" if len(frag.split()) <= 4 else m.group(0))
    text = re.sub(r"\.\s+([A-Z][^.!?]{1,30}[.!?])", _pause_frag, text)

    # Vocal reactions — max 2, never on first sentence
    FAILURE = {"rejected","refused","denied","failed","failure","abandoned",
               "dismissed","ignored","quit","fired","lost","laughed","mocked",
               "ridiculed","burned","classified","buried","suppressed"}
    IRONY   = {"bonus","prize went","never got","died before","too late",
               "nobody believed","no one believed","worth billions","worth millions",
               "never knew","never found","never heard","never received",
               "never credited","left before","already left"}
    PIVOT   = {"except","but then","until","however","yet","instead",
               "had not","had never","would never","did not","did not know"}

    added = 0
    parts = re.split(r"(?<=[.!?])\s+", text)
    out   = []
    for i, s in enumerate(parts):
        sl = s.lower()
        if i == 0:
            out.append(s); continue
        if added < 2 and any(w in sl for w in FAILURE):
            s = s.rstrip(".!?") + ". oof"; added += 1
        elif added < 2 and any(w in sl for w in IRONY):
            s = s.rstrip(".!?") + ". huh"; added += 1
        elif added < 2 and any(w in sl for w in PIVOT):
            s = "wait — " + s[0].upper() + s[1:]; added += 1
        out.append(s)

    return re.sub(r" {2,}", " ", " ".join(out)).strip()


def parse_documentary_scenes(text: str, target_minutes: float = 6.0) -> list:
    """
    Parse a documentary PDF text into Scene objects.
    Uses _LABEL_CHARS to handle ▸ → I glyph mangling by PyMuPDF.
    Speaker name captured with .+ to include accented chars (e.g. SCHRÖDINGER).
    """
    from state import Scene

    split_pat = re.compile(
        r"(?m)^" + _LABEL_CHARS +
        r"\s+(NARRATOR|STORY|TECHNICAL|VOICE(?::\s*.+)?)\s*$"
    )
    matches = list(split_pat.finditer(text))
    if not matches:
        return []

    paragraphs = []
    for i, m in enumerate(matches):
        label_raw     = m.group(1).strip()
        content_start = m.end()
        content_end   = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content       = text[content_start:content_end].strip()
        if not content:
            continue

        if label_raw.upper().startswith("VOICE"):
            para_type = "VOICE"
            sm = re.match(r"VOICE\s*:\s*(.+)", label_raw, re.IGNORECASE)
            speaker = sm.group(1).strip().title() if sm else "Unknown"
        else:
            para_type = label_raw.upper()
            speaker   = None

        paragraphs.append((para_type, speaker, content))

    scenes   = []
    scene_id = 1
    for para_type, speaker, content in paragraphs:
        wc  = len(content.split())
        dur = round(max(wc / WORDS_PER_SECOND, 8.0), 1)

        voice = (
            (DOCUMENTARY_VOICE_MAP.get(speaker or "", "") or DOCUMENTARY_VOICE_DEFAULT)
            if para_type == "VOICE" else ""
        )

        scenes.append(Scene(
            id               = scene_id,
            narration_text   = _enrich_for_tts(content, para_type),
            duration_seconds = dur,
            visual_prompt    = _auto_visual_hint(para_type, speaker, content),
            paragraph_type   = para_type,
            tts_voice        = voice,
        ))
        scene_id += 1

    return scenes


def _auto_visual_hint(para_type: str, speaker: Optional[str], content: str) -> str:
    snippet = content[:120].replace("\n", " ").strip()
    if para_type == "NARRATOR":
        return (f"Minimal cinematic title card. Dark background, electric cyan accent. "
                f"Context: {snippet}")
    elif para_type == "STORY":
        return (f"Historical cinematic scene. Biographical, era-appropriate, atmospheric. "
                f"Context: {snippet}")
    elif para_type == "TECHNICAL":
        return (f"Mathematical/scientific diagram. Manim-style. Clean, precise, educational. "
                f"Context: {snippet}")
    elif para_type == "VOICE":
        return (f"Dramatic portrait of {speaker or 'the figure'}. Cinematic, era-appropriate, "
                f"emotionally resonant. First-person perspective. Context: {snippet}")
    return snippet