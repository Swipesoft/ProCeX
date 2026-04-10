"""
utils/gemma_client.py
GemmaClient — unified interface for Gemma 4 31B via the Google GenAI SDK.

Mirrors LLMClient's public interface so agents can swap it in transparently:
  .complete(system, user, ...)          → str
  .complete_json(system, user, ...)     → dict | list
  .complete_vision(system, user, ...)   → str   (single-frame VLMCritic)
  .complete_vision_multi(...)           → str   (multi-frame VLMCritic)
  .complete_with_tools(...)             → dict  (agentic function-calling)

Key differences from LLMClient:
  - No provider chain / fallbacks — Gemma is the sole model
  - JSON mode enforced via response_mime_type="application/json"
    + schema string in system prompt → strict, no json-repair soft parsing
  - thinking_level="HIGH" on every call for quality output
  - Vision via Part.from_bytes (native multimodal, no base64 string injection)
  - Function calling via types.Tool(function_declarations=[...])
"""
from __future__ import annotations

import json
import time
from typing import Any, Optional

from config import ProcExConfig


# ── Model constant ─────────────────────────────────────────────────────────────
GEMMA_MODEL = "gemma-4-31b-it"

# ── Thinking config (applied to every call for quality) ───────────────────────
_THINKING_LEVEL = "HIGH"

# ── Retry settings ────────────────────────────────────────────────────────────
_MAX_RETRIES   = 3
_RETRY_SLEEP   = 2.0

# ── TPM throttler ─────────────────────────────────────────────────────────────
# Gemma 4 31B via Google AI API is capped at 16K tokens/minute (input TPM).
# thinking_level=HIGH consumes ~6-8K thinking tokens per call on top of input.
# We track a rolling 60s window of estimated input tokens and pause before
# any call that would push the running total above TPM_LIMIT_SAFE.
#
# Token estimation: 1 token ≈ 4 chars (conservative for English prose).
# We only count input tokens (system + user prompt) since output/thinking
# tokens are not part of the input TPM quota.

import collections as _collections
import threading  as _threading

_TPM_LIMIT_SAFE  = 12_000   # pause threshold — 75% of 16K, leaves headroom
_TPM_WINDOW_SECS = 60       # rolling window width
_tpm_lock        = _threading.Lock()
_tpm_log: "collections.deque[tuple[float, int]]" = _collections.deque()  # (timestamp, tokens)


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: 1 token ≈ 4 characters."""
    return max(1, len(text) // 4)


def _tpm_acquire(estimated_tokens: int) -> None:
    """
    Block until adding estimated_tokens to the rolling window would not
    exceed _TPM_LIMIT_SAFE. Then register the tokens.

    Called once per generate_content invocation with the combined
    system + user prompt length.

    Bug guard 1 — empty deque after expiry:
      After evicting old entries the deque can be empty, making
      _tpm_log[0] raise IndexError. If the window is empty we always
      proceed — an empty window means 0 tokens used.

    Bug guard 2 — single call larger than the safe limit:
      If estimated_tokens > _TPM_LIMIT_SAFE the condition
      `used + estimated_tokens <= _TPM_LIMIT_SAFE` can NEVER be true,
      causing an infinite loop. We cap the registered amount at
      _TPM_LIMIT_SAFE so large prompts don't deadlock the pipeline.
    """
    # Cap so a single oversized prompt never causes an infinite wait
    charge = min(estimated_tokens, _TPM_LIMIT_SAFE)

    while True:
        with _tpm_lock:
            now = time.time()
            # Expire entries older than the rolling window
            while _tpm_log and _tpm_log[0][0] < now - _TPM_WINDOW_SECS:
                _tpm_log.popleft()
            used = sum(t for _, t in _tpm_log)

            # ── Bug guard 1: empty deque → window is clear, always proceed ──
            if not _tpm_log:
                _tpm_log.append((now, charge))
                return

            if used + charge <= _TPM_LIMIT_SAFE:
                _tpm_log.append((now, charge))
                return   # safe to proceed

            # Window is full — calculate sleep until oldest entry expires
            # _tpm_log[0] is safe here because we checked not-empty above
            oldest_ts  = _tpm_log[0][0]
            sleep_secs = max(1.0, (_TPM_WINDOW_SECS - (now - oldest_ts)) + 1.0)
            used_snap  = used   # capture before releasing lock

        # Release lock before sleeping
        print(
            f"[GemmaThrottle] TPM window at {used_snap:,}/{_TPM_LIMIT_SAFE:,} tokens — "
            f"waiting {sleep_secs:.1f}s"
        )
        time.sleep(sleep_secs)


class GemmaClient:
    """
    Thin wrapper around google.genai for Gemma 4 31B.

    Constructed once by the orchestrator when --provider gemma is set,
    then passed to every agent in place of LLMClient.

    The .cfg attribute is kept so agents that read cfg.gemini_api_key etc.
    still work without modification.
    """

    def __init__(self, cfg: ProcExConfig):
        self.cfg = cfg
        self._client = None
        self._init_client()

    def _init_client(self):
        if not self.cfg.gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is required for GemmaClient. "
                "Gemma 4 31B is accessed via the Gemini API endpoint."
            )
        try:
            from google import genai
            self._client = genai.Client(api_key=self.cfg.gemini_api_key)
            print(f"[GemmaClient] Initialised — model={GEMMA_MODEL}")
        except ImportError:
            raise RuntimeError(
                "google-genai SDK not installed. Run: pip install google-genai"
            )

    # ── Primary text interface ─────────────────────────────────────────────────

    def complete(
        self,
        system_prompt: str,
        user_prompt:   str,
        *,
        json_mode:        bool          = False,
        max_tokens:       int           = 16000,
        temperature:      float         = 0.7,
        schema:           Optional[dict] = None,   # JSON schema dict; only used when json_mode=True
        model_override:   Optional[str]  = None,   # ignored — Gemma only
        primary_provider: Optional[str]  = None,   # ignored — Gemma only
    ) -> str:
        """
        Text completion via Gemma 4 31B.

        json_mode=True  → enforces application/json response_mime_type.
                          If schema is provided it is embedded in the system
                          prompt so Gemma knows the exact shape expected.
        Returns raw string (JSON string when json_mode=True).
        """
        from google.genai import types

        # When JSON is required, append schema to system prompt
        sys_text = system_prompt
        if json_mode and schema:
            sys_text = (
                system_prompt.rstrip()
                + "\n\nYou MUST respond with valid JSON that matches this schema exactly:\n"
                + json.dumps(schema, indent=2)
                + "\nNo preamble. No markdown. No explanation. Only valid JSON."
            )
        elif json_mode:
            sys_text = (
                system_prompt.rstrip()
                + "\n\nRespond ONLY with valid JSON. No preamble, no markdown fences, "
                  "no explanation. The first character of your response must be { or [."
            )

        config_kwargs = dict(
            system_instruction = sys_text,
            max_output_tokens  = max_tokens,
            temperature        = temperature,
            thinking_config    = types.ThinkingConfig(thinking_level=_THINKING_LEVEL),
        )
        if json_mode:
            config_kwargs["response_mime_type"] = "application/json"

        # Throttle: estimate input tokens and wait if TPM window is full
        input_tokens = _estimate_tokens(sys_text + (user_prompt if isinstance(user_prompt, str) else ""))
        _tpm_acquire(input_tokens)

        last_err = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = self._client.models.generate_content(
                    model    = GEMMA_MODEL,
                    contents = user_prompt,
                    config   = types.GenerateContentConfig(**config_kwargs),
                )
                return response.text or ""
            except Exception as e:
                last_err = e
                print(f"[GemmaClient] attempt {attempt}/{_MAX_RETRIES} failed: {e}")
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_SLEEP * attempt)

        raise RuntimeError(f"[GemmaClient] All retries failed. Last error: {last_err}")

    def complete_json(
        self,
        system_prompt: str,
        user_prompt:   str,
        *,
        max_tokens:       int            = 16000,
        temperature:      float          = 0.3,
        schema:           Optional[dict] = None,
        primary_provider: Optional[str]  = None,   # ignored
    ) -> dict | list:
        """
        Strict JSON completion — validates the response before returning.
        Retries up to _MAX_RETRIES times if JSON is invalid.
        Raises RuntimeError if all retries return invalid JSON.
        """
        last_err = None
        for attempt in range(1, _MAX_RETRIES + 1):
            raw = self.complete(
                system_prompt,
                user_prompt,
                json_mode   = True,
                max_tokens  = max_tokens,
                temperature = temperature,
                schema      = schema,
            )
            try:
                return json.loads(raw)
            except json.JSONDecodeError as e:
                last_err = e
                print(
                    f"[GemmaClient] JSON parse failed (attempt {attempt}/{_MAX_RETRIES}): {e}"
                    f"\n  Raw: {raw[:200]!r}"
                )
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_SLEEP)

        raise RuntimeError(
            f"[GemmaClient] Could not parse valid JSON after {_MAX_RETRIES} attempts. "
            f"Last error: {last_err}"
        )

    # ── Vision interface ───────────────────────────────────────────────────────

    def complete_vision(
        self,
        system_prompt:    str,
        user_prompt:      str,
        image_bytes:      bytes,
        *,
        image_mime:       str           = "image/png",
        max_tokens:       int           = 16384,
        temperature:      float         = 0.1,
        primary_provider: Optional[str] = None,   # ignored
    ) -> str:
        """
        Single-image vision completion — used by VLMCritic stage 1.
        Sends image via Part.from_bytes (native multimodal, no base64 string).
        """
        return self._call_gemma_vision(
            system_prompt = system_prompt,
            user_prompt   = user_prompt,
            image_parts   = [(image_bytes, image_mime)],
            max_tokens    = max_tokens,
            temperature   = temperature,
            json_mode     = False,
        )

    def complete_vision_json(
        self,
        system_prompt: str,
        user_prompt:   str,
        image_bytes:   bytes,
        *,
        image_mime:    str            = "image/png",
        max_tokens:    int            = 16384,
        temperature:   float          = 0.1,
        schema:        Optional[dict] = None,
    ) -> dict | list:
        """
        Vision + strict JSON output — used by VLMCritic to get structured
        collision report from Gemma's visual inspection.
        """
        sys_text = system_prompt
        if schema:
            sys_text = (
                system_prompt.rstrip()
                + "\n\nYou MUST respond with valid JSON matching this schema:\n"
                + json.dumps(schema, indent=2)
                + "\nNo preamble. No markdown. Only valid JSON."
            )

        last_err = None
        for attempt in range(1, _MAX_RETRIES + 1):
            raw = self._call_gemma_vision(
                system_prompt = sys_text,
                user_prompt   = user_prompt,
                image_parts   = [(image_bytes, image_mime)],
                max_tokens    = max_tokens,
                temperature   = temperature,
                json_mode     = True,
            )
            try:
                return json.loads(raw)
            except json.JSONDecodeError as e:
                last_err = e
                print(
                    f"[GemmaClient] vision JSON parse failed "
                    f"(attempt {attempt}/{_MAX_RETRIES}): {e}"
                )
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_SLEEP)

        raise RuntimeError(
            f"[GemmaClient] Vision JSON failed after {_MAX_RETRIES} attempts: {last_err}"
        )

    def _call_gemma_vision(
        self,
        system_prompt: str,
        user_prompt:   str,
        image_parts:   list[tuple[bytes, str]],   # [(bytes, mime_type), ...]
        max_tokens:    int,
        temperature:   float,
        json_mode:     bool = False,
    ) -> str:
        """Internal: build multimodal content and call Gemma."""
        from google.genai import types

        parts = []
        for img_bytes, mime in image_parts:
            parts.append(types.Part.from_bytes(data=img_bytes, mime_type=mime))
        parts.append(types.Part.from_text(text=user_prompt))

        config_kwargs = dict(
            system_instruction = system_prompt,
            max_output_tokens  = max_tokens,
            temperature        = temperature,
            thinking_config    = types.ThinkingConfig(thinking_level=_THINKING_LEVEL),
        )
        if json_mode:
            config_kwargs["response_mime_type"] = "application/json"

        # Throttle input tokens (image bytes count too — rough: len/4)
        img_token_est = sum(len(b) // 4 for b, _ in image_parts)
        _tpm_acquire(_estimate_tokens(system_prompt + user_prompt) + img_token_est)

        last_err = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = self._client.models.generate_content(
                    model    = GEMMA_MODEL,
                    contents = types.Content(parts=parts, role="user"),
                    config   = types.GenerateContentConfig(**config_kwargs),
                )
                return response.text or ""
            except Exception as e:
                last_err = e
                print(f"[GemmaClient] vision attempt {attempt}/{_MAX_RETRIES} failed: {e}")
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_SLEEP * attempt)

        raise RuntimeError(f"[GemmaClient] Vision failed: {last_err}")

    # ── Function calling interface ─────────────────────────────────────────────

    def complete_with_tools(
        self,
        system_prompt:        str,
        user_prompt:          str,
        function_declarations: list[dict],
        *,
        max_tokens:  int   = 16000,
        temperature: float = 0.3,
    ) -> dict:
        """
        Single-turn function-calling completion.
        Returns a dict with either:
          {"type": "function_call", "name": str, "args": dict}
          {"type": "text", "text": str}

        The caller is responsible for executing the function and continuing
        the conversation loop.
        """
        from google.genai import types

        tool = types.Tool(function_declarations=function_declarations)

        config = types.GenerateContentConfig(
            system_instruction = system_prompt,
            max_output_tokens  = max_tokens,
            temperature        = temperature,
            thinking_config    = types.ThinkingConfig(thinking_level=_THINKING_LEVEL),
            tools              = [tool],
        )

        _tpm_acquire(_estimate_tokens(system_prompt + (user_prompt if isinstance(user_prompt, str) else "")))

        last_err = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = self._client.models.generate_content(
                    model    = GEMMA_MODEL,
                    contents = user_prompt,
                    config   = config,
                )
                candidate = response.candidates[0] if response.candidates else None
                if not candidate:
                    return {"type": "text", "text": ""}

                for part in (candidate.content.parts or []):
                    if hasattr(part, "function_call") and part.function_call:
                        fc = part.function_call
                        return {
                            "type": "function_call",
                            "name": fc.name,
                            "args": dict(fc.args) if fc.args else {},
                        }

                # No function call — model returned text (research complete)
                return {"type": "text", "text": response.text or ""}

            except Exception as e:
                last_err = e
                print(f"[GemmaClient] tool call attempt {attempt}/{_MAX_RETRIES} failed: {e}")
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_SLEEP * attempt)

        raise RuntimeError(f"[GemmaClient] Tool call failed: {last_err}")

    def complete_with_tool_result(
        self,
        system_prompt:        str,
        conversation:         list[dict],   # full turn history
        function_declarations: list[dict],
        *,
        max_tokens:  int   = 16000,
        temperature: float = 0.3,
    ) -> dict:
        """
        Multi-turn function-calling: sends the full conversation history
        (including tool results) back to Gemma to continue the loop.
        Returns same shape as complete_with_tools.
        """
        from google.genai import types

        tool   = types.Tool(function_declarations=function_declarations)
        config = types.GenerateContentConfig(
            system_instruction = system_prompt,
            max_output_tokens  = max_tokens,
            temperature        = temperature,
            thinking_config    = types.ThinkingConfig(thinking_level=_THINKING_LEVEL),
            tools              = [tool],
        )

        # Build Content objects from conversation history
        contents = []
        for turn in conversation:
            role  = turn["role"]   # "user" | "model" | "tool"
            parts = []
            if turn.get("text"):
                parts.append(types.Part.from_text(text=turn["text"]))
            if turn.get("function_call"):
                fc = turn["function_call"]
                parts.append(types.Part(function_call=types.FunctionCall(
                    name=fc["name"], args=fc["args"]
                )))
            if turn.get("function_response"):
                fr = turn["function_response"]
                parts.append(types.Part(function_response=types.FunctionResponse(
                    name=fr["name"], response=fr["response"]
                )))
            if parts:
                contents.append(types.Content(parts=parts, role=role))

        # Estimate tokens across entire conversation history
        conv_text = " ".join(
            t.get("text", "") + str(t.get("function_response", ""))
            for t in conversation
        )
        _tpm_acquire(_estimate_tokens(system_prompt + conv_text))

        last_err = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = self._client.models.generate_content(
                    model    = GEMMA_MODEL,
                    contents = contents,
                    config   = config,
                )
                candidate = response.candidates[0] if response.candidates else None
                if not candidate:
                    return {"type": "text", "text": ""}

                for part in (candidate.content.parts or []):
                    if hasattr(part, "function_call") and part.function_call:
                        fc = part.function_call
                        return {
                            "type": "function_call",
                            "name": fc.name,
                            "args": dict(fc.args) if fc.args else {},
                        }

                return {"type": "text", "text": response.text or ""}

            except Exception as e:
                last_err = e
                print(f"[GemmaClient] multi-turn attempt {attempt}/{_MAX_RETRIES} failed: {e}")
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_SLEEP * attempt)

        raise RuntimeError(f"[GemmaClient] Multi-turn tool call failed: {last_err}")