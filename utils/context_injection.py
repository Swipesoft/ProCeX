"""
utils/context_injection.py
Shared helper for injecting the --context string into agent prompts.

The context string captures the teaching perspective, audience level, and
scope exclusions for the video (e.g. "focus on PyTorch QKV implementation,
exclude multi-head attention, student knows RNN theory").

Injection rule (per Boochi's spec):
  - Top AND bottom of the user prompt (per-scene instruction)
  - Clearly labelled as NOT part of TTS speech so agents don't confuse
    it with narration content that should match word-level timestamps
  - System prompts are left untouched — context goes on the user side
    so every per-scene call is freshly reminded of the scope

Usage:
    from utils.context_injection import wrap_with_context
    user_prompt = wrap_with_context(user_prompt, state.context)
"""


_CONTEXT_HEADER = (
    "[TEACHING CONTEXT — NOT PART OF TTS SPEECH AND NOT TO BE READ ALOUD]\n"
    "{context}\n"
    "[END TEACHING CONTEXT]"
)


def wrap_with_context(user_prompt: str, context: str) -> str:
    """
    Inject context at the top and bottom of a user prompt.
    Returns user_prompt unchanged if context is empty.

    The double injection (top + bottom) is intentional — it emphasises
    the scope to models that may front-load or tail-weight attention.
    """
    if not context or not context.strip():
        return user_prompt

    tag = _CONTEXT_HEADER.format(context=context.strip())
    return f"{tag}\n\n{user_prompt}\n\n{tag}"

import re as _re

# Regex that matches the context block in any form it might appear —
# including cases where an LLM accidentally copies the tag into its output.
_CONTEXT_TAG_RE = _re.compile(
    r"\[TEACHING CONTEXT[^\]]*\].*?\[END TEACHING CONTEXT\]",
    _re.DOTALL | _re.IGNORECASE,
)


def strip_context_tags(text: str) -> str:
    """
    Remove any [TEACHING CONTEXT ...]...[END TEACHING CONTEXT] blocks from text.

    Called by TTSAgent on every scene's narration_text before audio generation
    so context injection never reaches the TTS voice — even if an LLM
    accidentally echoed the context tag into its output.
    """
    if not text:
        return text
    cleaned = _CONTEXT_TAG_RE.sub("", text)
    # Collapse any double blank lines left by the removal
    cleaned = _re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()