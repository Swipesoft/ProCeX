"""
utils/gemma_client.py
GemmaClient — Gemma 4 31B via TogetherAI serverless endpoint.

Replaces the previous google.genai implementation which hit Google AI Studio's
16K TPM limit (causing 60s throttle stalls). TogetherAI Tier 3 allows 3K RPM
with no TPM cap, eliminating all throttling needs.

Public interface mirrors LLMClient exactly so agents swap it in transparently:
  .complete(system, user, ...)              → str
  .complete_json(system, user, ...)         → dict | list
  .complete_vision(system, user, ...)       → str   (VLMCritic image frames)
  .complete_vision_json(...)                → dict  (VLMCritic structured)
  .complete_with_tools(...)                 → dict  (research agentic loop)
  .complete_with_tool_result(...)           → dict  (research multi-turn)

Model on TogetherAI: google/gemma-4-31B-it

Important — function calling:
  google/gemma-4-31B-it is NOT in TogetherAI's native function-calling
  supported models list. The agentic research loop therefore uses a
  prompt-driven JSON approach: Gemma returns structured JSON deciding
  whether to search or stop, instead of native tool_calls.
  complete_with_tools() and complete_with_tool_result() implement this
  transparently — callers (deep_research.py) are unchanged.

Vision:
  TogetherAI accepts base64 images as data URIs in image_url content blocks.
  VLMCritic sends JPEG frames; we encode them inline.

JSON mode:
  TogetherAI supports response_format={"type": "json_object"} for Gemma.
  We combine this with a schema description in the system prompt for
  strict structure enforcement.
"""
from __future__ import annotations

import base64
import json
import time
from typing import Any, Optional
import threading as _threading  #
from config import ProcExConfig


# ── Constants ─────────────────────────────────────────────────────────────────
TOGETHER_MODEL  = "google/gemma-4-31B-it" # "moonshotai/Kimi-K2.5" # "google/gemma-4-31B-it"
TOGETHER_API_KEY_ENV = "TOGETHER_API_KEY"

_MAX_RETRIES = 3
_RETRY_SLEEP = 1.0
_TOGETHER_SEM = _threading.Semaphore(1)


class GemmaClient:
    """
    Thin wrapper around TogetherAI's chat completions API for Gemma 4 31B.

    Constructed once by the orchestrator when --provider gemma is set,
    then passed to every agent in place of LLMClient.

    .cfg is kept so agents that read cfg.gemini_api_key etc. still work.
    """

    def __init__(self, cfg: ProcExConfig):
        self.cfg = cfg
        self._client = None
        self._init_client()

    def _init_client(self):
        import os
        api_key = os.environ.get(TOGETHER_API_KEY_ENV, "")
        if not api_key:
            raise RuntimeError(
                f"{TOGETHER_API_KEY_ENV} not set. "
                "Add it to your .env file to use the TogetherAI Gemma endpoint."
            )
        try:
            from together import Together
            self._client = Together(api_key=api_key)
            print(f"[GemmaClient] Initialised — model={TOGETHER_MODEL} via TogetherAI")
        except ImportError:
            raise RuntimeError(
                "together SDK not installed. Run: pip install together"
            )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _call(
        self,
        messages:        list[dict],
        max_tokens:      int   = 16000,
        temperature:     float = 0.7,
        json_mode:       bool  = False,
        schema:          Optional[dict] = None,

    ) -> str:
        """
        Single Together chat completion call with retry logic.
        Returns the assistant message content string.
        """
        kwargs: dict[str, Any] = dict(
            model       = TOGETHER_MODEL,
            messages    = messages,
            max_tokens  = max_tokens,
            temperature = temperature,
            timeout=300,
        )
        if json_mode:
            if schema:
                # Structured outputs with explicit schema
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "structured_output",
                        "schema": schema,
                    },
                }
            else:
                # Basic JSON object mode
                kwargs["response_format"] = {"type": "json_object"}

        last_err = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                with _TOGETHER_SEM:
                    response = self._client.chat.completions.create(**kwargs)
                # response = self._client.chat.completions.create(**kwargs)

                return response.choices[0].message.content or ""
            except Exception as e:
                last_err = e
                print(f"[GemmaClient] attempt {attempt}/{_MAX_RETRIES} failed: {e}")
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_SLEEP * attempt)

        raise RuntimeError(f"[GemmaClient] All retries failed. Last: {last_err}")

    def _build_messages(self, system: str, user: str) -> list[dict]:
        """Standard system+user message pair."""
        return [
            {"role": "system",  "content": system},
            {"role": "user",    "content": user},
        ]

    def _image_to_data_uri(self, image_bytes: bytes, mime: str) -> str:
        """Encode raw image bytes as a base64 data URI for TogetherAI vision."""
        b64 = base64.b64encode(image_bytes).decode()
        return f"data:{mime};base64,{b64}"

    # ── Primary text interface ─────────────────────────────────────────────────

    def complete(
        self,
        system_prompt: str,
        user_prompt:   str,
        *,
        json_mode:        bool            = False,
        max_tokens:       int             = 16000,
        temperature:      float           = 0.7,
        schema:           Optional[dict]  = None,
        model_override:   Optional[str]   = None,   # ignored — Gemma only
        primary_provider: Optional[str]   = None,   # ignored — Gemma only
    ) -> str:
        """Text completion via Gemma 4 31B on TogetherAI."""
        sys_text = system_prompt
        if json_mode and schema:
            sys_text = (
                system_prompt.rstrip()
                + "\n\nYou MUST respond with valid JSON matching this schema:\n"
                + json.dumps(schema, indent=2)
                + "\nNo preamble. No markdown. Only valid JSON."
            )
        elif json_mode:
            sys_text = (
                system_prompt.rstrip()
                + "\n\nRespond ONLY with valid JSON. "
                  "No preamble, no markdown fences, no explanation. "
                  "First character must be { or [."
            )

        messages = self._build_messages(sys_text, user_prompt)
        return self._call(
            messages,
            max_tokens  = max_tokens,
            temperature = temperature,
            json_mode   = json_mode,
            schema      = schema,
        )

    def complete_json(
        self,
        system_prompt: str,
        user_prompt:   str,
        *,
        max_tokens:       int            = 16000,
        temperature:      float          = 0.3,
        schema:           Optional[dict] = None,
        primary_provider: Optional[str]  = None,    # ignored
    ) -> dict | list:
        """
        Strict JSON completion — retries until valid JSON is parsed.
        Uses TogetherAI json_object mode + schema-in-prompt for enforcement.
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
            # Strip markdown fences just in case
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
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
            f"Last: {last_err}"
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
        primary_provider: Optional[str] = None,    # ignored
    ) -> str:
        """
        Single-image vision completion — used by VLMCritic stage 1.
        TogetherAI accepts base64 data URIs in the image_url content block.
        """
        data_uri = self._image_to_data_uri(image_bytes, image_mime)
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text",      "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            },
        ]
        return self._call(messages, max_tokens=max_tokens, temperature=temperature)

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
        Vision + strict JSON — used by VLMCritic to get structured collision report.
        """
        sys_text = system_prompt
        if schema:
            sys_text = (
                system_prompt.rstrip()
                + "\n\nYou MUST respond with valid JSON matching this schema:\n"
                + json.dumps(schema, indent=2)
                + "\nNo preamble. No markdown. Only valid JSON."
            )

        data_uri = self._image_to_data_uri(image_bytes, image_mime)
        messages = [
            {"role": "system", "content": sys_text},
            {
                "role": "user",
                "content": [
                    {"type": "text",      "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            },
        ]

        last_err = None
        for attempt in range(1, _MAX_RETRIES + 1):
            raw = self._call(
                messages,
                max_tokens  = max_tokens,
                temperature = temperature,
                json_mode   = True,
                schema      = schema,
            )
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            try:
                result = json.loads(raw)
                return result if isinstance(result, (dict, list)) else {}
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

    # ── Native function calling (TogetherAI tool_calls) ──────────────────────
    #
    # google/gemma-4-31B-it supports Tool Calling on TogetherAI (confirmed
    # in the model card Features section). We use the standard Together API
    # tool_calls format — same as the docs show for other models.
    #
    # function_declarations use the same dict schema as google.genai:
    #   {"name": str, "description": str, "parameters": {...JSON Schema...}}
    # We wrap each in {"type": "function", "function": {...}} for Together.

    def _decls_to_together_tools(self, function_declarations: list[dict]) -> list[dict]:
        """Convert google.genai-style declarations to Together tool format."""
        tools = []
        for decl in function_declarations:
            tools.append({
                "type": "function",
                "function": {
                    "name":        decl.get("name", ""),
                    "description": decl.get("description", ""),
                    "parameters":  decl.get("parameters", {}),
                },
            })
        return tools

    def _parse_tool_calls(self, response) -> dict:
        """
        Parse a Together response into the standard tool result dict.
        Returns:
          {"type": "function_call", "name": str, "args": dict}
          {"type": "text",          "text": str}
        """
        msg = response.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None) or []
        print(
            f"[GemmaClient] raw message: tool_calls={len(tool_calls)}, content={repr((msg.content or '')[:300])}, role={msg.role}"
        )
        print(f"[GemmaClient] full message dump: {msg.model_dump()}")

        if tool_calls:
            # Take the first tool call (research loop fires one at a time)
            tc   = tool_calls[0]
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, AttributeError):
                args = {}
            return {"type": "function_call", "name": name, "args": args}

        # No tool call — model returned text (research complete)
        print(
            f"[GemmaClient] text response, content length: {len(msg.content or '')}, preview: {(msg.content or '')[:200]!r}")
        return {"type": "text", "text": msg.content or ""}

    def complete_with_tools(
        self,
        system_prompt:         str,
        user_prompt:           str,
        function_declarations: list[dict],
        *,
        max_tokens:  int   = 16000,
        temperature: float = 0.3,
    ) -> dict:
        """
        Single-turn native tool-calling completion.
        Returns {"type": "function_call", "name": str, "args": dict}
                or {"type": "text", "text": str}.
        """
        messages = self._build_messages(system_prompt, user_prompt)
        tools    = self._decls_to_together_tools(function_declarations)

        last_err = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = self._client.chat.completions.create(
                    model       = TOGETHER_MODEL,
                    messages    = messages,
                    tools       = tools,
                    max_tokens  = max_tokens,
                    temperature = temperature,
                    timeout=300
                )
                return self._parse_tool_calls(response)
            except Exception as e:
                last_err = e
                print(f"[GemmaClient] tool call attempt {attempt}/{_MAX_RETRIES} failed: {e}")
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_SLEEP * attempt)

        raise RuntimeError(f"[GemmaClient] Tool call failed: {last_err}")

    def complete_with_tool_result(
        self,
        system_prompt:         str,
        conversation:          list[dict],
        function_declarations: list[dict],
        *,
        max_tokens:  int   = 16000,
        temperature: float = 0.3,
    ) -> dict:
        """
        Multi-turn native tool-calling completion.
        Rebuilds the full conversation history including tool results
        in Together's expected format, then calls the model again.
        """
        tools    = self._decls_to_together_tools(function_declarations)
        messages = [{"role": "system", "content": system_prompt}]

        for turn in conversation:
            role = turn.get("role", "user")

            if role == "user" and turn.get("text"):
                messages.append({"role": "user", "content": turn["text"]})

            elif role == "model":
                # Model's function_call turn — rebuild as assistant message
                fc = turn.get("function_call", {})
                if fc:
                    messages.append({
                        "role":    "assistant",
                        "content": "",
                        "tool_calls": [{
                            "id":   "call_0",
                            "type": "function",
                            "function": {
                                "name":      fc.get("name", ""),
                                "arguments": json.dumps(fc.get("args", {})),
                            },
                        }],
                    })
                elif turn.get("text"):
                    messages.append({"role": "assistant", "content": turn["text"]})

            elif role == "tool":
                # Tool result — inject as tool role message
                fr   = turn.get("function_response", {})
                resp = fr.get("response", {})
                result_text = resp.get("result", str(resp))
                messages.append({
                    "role":         "tool",
                    "tool_call_id": "call_0",
                    "name":         fr.get("name", "tool"),
                    "content":      result_text[:2000],
                })

        last_err = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = self._client.chat.completions.create(
                    model       = TOGETHER_MODEL,
                    messages    = messages,
                    tools       = tools,
                    max_tokens  = max_tokens,
                    temperature = temperature,
                )
                return self._parse_tool_calls(response)
            except Exception as e:
                last_err = e
                print(f"[GemmaClient] multi-turn attempt {attempt}/{_MAX_RETRIES} failed: {e}")
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_SLEEP * attempt)

        raise RuntimeError(f"[GemmaClient] Multi-turn tool call failed: {last_err}")