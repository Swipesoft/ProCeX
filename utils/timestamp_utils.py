"""
utils/timestamp_utils.py
Word timestamp utilities for ProcEx.

Key function: extract_animation_anchors()
  Converts raw word timestamps into a small set of named anchor points
  that ManimCoder can use to fire animations at exact narration moments.
  Rather than dumping 80 {"word","start","end"} dicts and asking the LLM
  to "sum timings to equal duration_seconds" (which it never does correctly),
  we pre-select 5-7 key moments and pre-compute the self.wait() gaps so the
  LLM just fills in the blanks.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from state import WordTimestamp


# ── Stopwords to skip when scoring anchor candidates ──────────────────────────
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "that", "this",
    "these", "those", "it", "its", "we", "our", "they", "their", "which",
    "who", "what", "when", "where", "how", "as", "so", "if", "then",
    "than", "each", "all", "both", "also", "just", "more", "most", "into",
    "about", "up", "out", "through", "such", "not", "no", "only", "very",
}


@dataclass
class AnimationAnchor:
    """A named moment in scene time where an animation should fire."""
    t:           float   # seconds from scene start (scene-relative)
    word:        str     # the word the narrator speaks at this moment
    hint:        str     # suggested visual action for the LLM
    gap_before:  float   # self.wait() seconds before this anchor
                         # (pre-computed assuming prev animation = 0.8s run_time)


def extract_animation_anchors(
    timestamps: list[WordTimestamp],
    visual_prompt: str = "",
    n_anchors: int = 6,
    default_run_time: float = 0.8,
) -> list[AnimationAnchor]:
    """
    Extract N evenly-distributed animation anchors from word timestamps.

    Algorithm:
      1. Convert timestamps to scene-relative time (subtract first word's start)
      2. Filter stopwords
      3. Score each word: technical length, capitalisation, digits, domain terms
      4. Divide the timeline into n_anchors equal buckets
      5. Pick the highest-scoring word in each bucket
      6. Pre-compute the gap_before for each anchor assuming a 0.8s animation

    Returns a list of AnimationAnchor sorted by time. Always includes t=0
    as the first anchor (scene open) and a final anchor near the end.

    The gap_before values are what the LLM writes as self.wait() calls —
    this removes all arithmetic burden from the model.
    """
    if not timestamps:
        return _uniform_anchors(n_anchors, 30.0, default_run_time)

    # Scene-relative times
    t0     = timestamps[0].start
    t_end  = timestamps[-1].end
    dur    = t_end - t0
    if dur <= 0:
        return _uniform_anchors(n_anchors, 30.0, default_run_time)

    # Score every word
    scored = []
    for ts in timestamps:
        t_rel = round(ts.start - t0, 3)
        score = _score_word(ts.word, visual_prompt)
        scored.append((t_rel, ts.word, score))

    # Divide into buckets and pick best word per bucket
    bucket_size = dur / n_anchors
    anchors_raw: list[tuple[float, str]] = []

    for i in range(n_anchors):
        bucket_start = i * bucket_size
        bucket_end   = (i + 1) * bucket_size
        candidates   = [
            (t, w, s) for (t, w, s) in scored
            if bucket_start <= t < bucket_end and w.lower() not in _STOPWORDS
        ]
        if candidates:
            best = max(candidates, key=lambda x: x[2])
            anchors_raw.append((best[0], best[1]))
        else:
            # Empty bucket — use midpoint with placeholder word
            anchors_raw.append((round(bucket_start + bucket_size / 2, 2), "..."))

    # Always anchor at t=0 — clamp first anchor regardless of bucket winner
    anchors_raw[0] = (0.0, anchors_raw[0][1])

    # Build AnimationAnchor objects with pre-computed gaps
    anchors: list[AnimationAnchor] = []
    prev_end = 0.0  # track when the previous animation ends

    for i, (t, word) in enumerate(anchors_raw):
        # Clamp t to valid range
        t = max(0.0, min(t, dur - 0.5))

        # gap_before = time from end of previous animation to this anchor
        gap = max(0.0, round(t - prev_end, 3))

        hint = _generate_hint(i, word, visual_prompt, len(anchors_raw))

        anchors.append(AnimationAnchor(
            t          = round(t, 3),
            word       = word,
            hint       = hint,
            gap_before = gap,
        ))

        # Assume this animation takes default_run_time seconds
        prev_end = t + default_run_time

    return anchors


def anchors_to_prompt_block(
    anchors: list[AnimationAnchor],
    duration: float,
    default_run_time: float = 0.8,
) -> str:
    """
    Render anchors as the structured prompt block that ManimCoder receives.
    Includes a ready-to-fill timing skeleton so the LLM only needs to
    substitute visual elements — no arithmetic required.
    """
    lines = [
        "ANIMATION ANCHORS — fire each visual reveal at the exact second shown.",
        "The narrator speaks the KEY WORD at time t. Match animations to speech.",
        "",
        f"{'t':>8}  {'KEY WORD':15}  SUGGESTED ACTION",
        "─" * 65,
    ]

    for a in anchors:
        lines.append(f"  t={a.t:>5.2f}s  {a.word[:15]:15}  {a.hint}")

    lines += [
        "─" * 65,
        "",
        "TIMING SKELETON — fill in your Manim objects below each wait():",
        "(Each animation assumed ~0.8s run_time. Adjust if needed.)",
        "",
    ]

    # Build the skeleton
    prev_end = 0.0
    for i, a in enumerate(anchors):
        gap = max(0.0, a.t - prev_end)
        if i == 0:
            if gap > 0.05:
                lines.append(f"    self.wait({gap:.2f})  # silence before first reveal")
        else:
            lines.append(f"    self.wait({gap:.2f})  # → narrator reaches '{a.word}' at t={a.t:.2f}s")

        lines.append(f"    self.play(...)  # {a.hint}")
        lines.append(f"    # run_time={default_run_time}")
        lines.append("")
        prev_end = a.t + default_run_time

    # Final wait to reach end of scene
    final_gap = max(0.5, duration - prev_end)
    lines.append(f"    self.wait({final_gap:.2f})  # hold until scene ends at t={duration:.1f}s")

    return "\n".join(lines)


# ── Scoring ────────────────────────────────────────────────────────────────────

def _score_word(word: str, context: str = "") -> float:
    """
    Score a word as an anchor candidate.
    Higher = more likely to be a conceptually meaningful moment.
    """
    w = word.strip(".,;:!?\"'()")
    if not w or w.lower() in _STOPWORDS:
        return 0.0

    score = 0.0

    # Length bonus — longer words tend to be more technical
    score += min(len(w) * 0.4, 3.0)

    # Title case or ALL CAPS — proper nouns, acronyms
    if w[0].isupper():
        score += 1.5
    if w.isupper() and len(w) > 1:
        score += 2.0

    # Contains digits — often a key technical term (e.g. "GPT-4", "d_k", "256")
    if any(c.isdigit() for c in w):
        score += 2.0

    # Looks like a technical identifier (camelCase, snake_case, hyphenated)
    if re.search(r'[A-Z][a-z]|[a-z][A-Z]|_|-', w):
        score += 1.5

    # Appears in the visual prompt (context relevance)
    if context and w.lower() in context.lower():
        score += 3.0

    return score


def _generate_hint(
    idx: int,
    word: str,
    visual_prompt: str,
    total: int,
) -> str:
    """Generate a concise visual hint for an anchor."""
    w = word.lower().strip(".,;:!?")

    if idx == 0:
        return "Establish scene — title card or intro visual"
    if idx == total - 1:
        return "Final beat — fade out all elements, hold last frame"

    # Check if the word appears in the visual prompt for context
    if visual_prompt and w in visual_prompt.lower():
        # Extract up to 6 words of context around the word in the visual prompt
        match = re.search(
            r'(?:\w+\s+){0,3}' + re.escape(w) + r'(?:\s+\w+){0,3}',
            visual_prompt, re.IGNORECASE
        )
        if match:
            return f"Reveal/highlight: {match.group(0).strip()}"

    return f"Reveal or animate concept: '{word}'"


def _uniform_anchors(
    n: int, duration: float, default_run_time: float
) -> list[AnimationAnchor]:
    """Fallback: evenly-spaced anchors when no timestamps are available."""
    step = duration / n
    anchors = []
    prev_end = 0.0
    for i in range(n):
        t   = round(i * step, 2)
        gap = max(0.0, round(t - prev_end, 3))
        anchors.append(AnimationAnchor(
            t=t, word=f"beat_{i+1}",
            hint=("Establish scene" if i == 0
                  else "Final beat — fade out" if i == n-1
                  else f"Animate concept {i+1}"),
            gap_before=gap,
        ))
        prev_end = t + default_run_time
    return anchors


# ── Existing utilities (unchanged) ────────────────────────────────────────────

def chars_to_words(alignment) -> list[WordTimestamp]:
    """Convert ElevenLabs character-level alignment to word-level timestamps."""
    words: list[WordTimestamp] = []
    current_chars: list[str]   = []
    word_start: float | None   = None
    word_end:   float           = 0.0

    for char, start, end in zip(
        alignment.characters,
        alignment.character_start_times_seconds,
        alignment.character_end_times_seconds,
    ):
        if char in (" ", "\n", "\t"):
            if current_chars:
                words.append(WordTimestamp(
                    word="".join(current_chars), start=word_start, end=word_end
                ))
                current_chars = []
                word_start    = None
        else:
            if word_start is None:
                word_start = start
            current_chars.append(char)
            word_end = end

    if current_chars:
        words.append(WordTimestamp(
            word="".join(current_chars), start=word_start, end=word_end
        ))
    return words


def assign_timestamps_to_scenes(scenes, all_timestamps: list[WordTimestamp]) -> list:
    """Distribute global word timestamps into per-scene buckets."""
    if not all_timestamps:
        return scenes

    total_chars = sum(len(s.narration_text) for s in scenes)
    ts_idx      = 0
    for scene in scenes:
        scene_ratio = len(scene.narration_text) / max(total_chars, 1)
        n_words     = max(1, round(scene_ratio * len(all_timestamps)))
        end_idx     = min(ts_idx + n_words, len(all_timestamps))
        scene.timestamps = all_timestamps[ts_idx:end_idx]
        ts_idx = end_idx
    return scenes


def timestamps_to_dict_list(timestamps: list[WordTimestamp]) -> list[dict]:
    """Serialize for injection into LLM prompts."""
    return [
        {"word": t.word, "start": round(t.start, 3), "end": round(t.end, 3)}
        for t in timestamps
    ]


def scene_time_budget(timestamps: list[WordTimestamp]) -> float:
    """Total duration covered by a scene's timestamps."""
    if not timestamps:
        return 0.0
    return timestamps[-1].end - timestamps[0].start