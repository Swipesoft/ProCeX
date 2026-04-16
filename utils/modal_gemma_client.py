"""
utils/gemma_client.py
=====================
GemmaClient — Gemma 4 31B via your Modal-hosted vLLM endpoint.

Replaces the TogetherAI backend which timed out on inputs >10K tokens.
Your Modal deployment (modal_serve.py) runs vLLM with a 32K context window
on 2× H100 SXM, so long-context agent calls complete without throttling.

Public interface is IDENTICAL to the previous version — agents swap it in
transparently with zero changes to callers:
  .complete(system, user, ...)              → str
  .complete_json(system, user, ...)         → dict | list
  .complete_vision(system, user, ...)       → str
  .complete_vision_json(...)                → dict
  .complete_with_tools(...)                 → dict
  .complete_with_tool_result(...)           → dict

Auth
----
Every request carries an X-Internal-Secret header so Modal rejects
any call that didn't originate from this client. Set both env vars:

    MODAL_GEMMA_URL       = https://<workspace>--gemma4-31b-serve.modal.run
    MODAL_INTERNAL_SECRET = <same value stored in your modal-internal-secret>
"""
from __future__ import annotations
from dotenv import load_dotenv as _load_dotenv
import os as _os
import base64
import json
import os
import time
import threading as _threading
from typing import Any, Optional, TYPE_CHECKING

# ── Constants ──────────────────────────────────────────────────────────────────
# Load .env from the project root (two levels up from utils/)
_load_dotenv(
    dotenv_path=_os.path.join(_os.path.dirname(__file__), "..", ".env"),
    override=False,   # don't override vars already set in the shell environment
)
# ───────────────────────────────────────────────────────────────────────────────

#if TYPE_CHECKING:
    # from config import ProcExConfig


# ── Constants ──────────────────────────────────────────────────────────────────

MODEL_NAME       = os.environ.get("GEMMA_MODEL_NAME",      "google/gemma-4-31B-it")
MODAL_URL_ENV    = os.environ.get("MODAL_URL_ENV_KEY",     "MODAL_GEMMA_URL")
MODAL_SECRET_ENV = os.environ.get("MODAL_SECRET_ENV_KEY",  "MODAL_INTERNAL_SECRET")
SECRET_HEADER    = os.environ.get("MODAL_SECRET_HEADER",   "X-Internal-Secret")

_MAX_RETRIES = 3
_RETRY_SLEEP = 1.0
# vLLM handles internal concurrency — raise semaphore ceiling vs. Together's 1
_MODAL_SEM   = _threading.Semaphore(16)


class GemmaClient:
    """
    Thin wrapper around your Modal-hosted vLLM endpoint for Gemma 4 31B.

    Constructed once by the orchestrator when --provider gemma is set,
    then passed to every agent in place of LLMClient.

    .cfg is kept so agents that read cfg.gemini_api_key etc. still work.
    """

    def __init__(self, cfg: "ProcExConfig"):
        self.cfg = cfg
        self._client = None
        self._init_client()

    def _init_client(self):
        modal_url = os.environ.get(MODAL_URL_ENV, "").rstrip("/")
        secret    = os.environ.get(MODAL_SECRET_ENV, "")

        if not modal_url:
            raise RuntimeError(
                f"{MODAL_URL_ENV} not set. "
                "Add MODAL_GEMMA_URL=https://<workspace>--gemma4-31b-serve.modal.run "
                "to your .env file."
            )
        if not secret:
            raise RuntimeError(
                f"{MODAL_SECRET_ENV} not set. "
                "Add the same value you stored in the modal-internal-secret Modal secret."
            )

        try:
            from openai import OpenAI
            self._client = OpenAI(
                base_url        = f"{modal_url}/v1",
                api_key         = "modal-local-key",       # vLLM accepts any non-empty string
                default_headers = {SECRET_HEADER: secret}, # auth on every request
            )
            print(f"[GemmaClient] Initialised — model={MODEL_NAME} via Modal @ {modal_url}")
        except ImportError:
            raise RuntimeError("openai SDK not installed. Run: pip install openai")

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _call(
        self,
        messages:    list[dict],
        max_tokens:  int   = 16000,
        temperature: float = 0.7,
        json_mode:   bool  = False,
        schema:      Optional[dict] = None,
        tools:       Optional[list] = None,
    ) -> Any:
        """
        Single OpenAI-compatible chat completion call to Modal/vLLM with retry.
        Returns the raw response object; callers extract .choices[0].message.
        """
        kwargs: dict[str, Any] = dict(
            model       = MODEL_NAME,
            messages    = messages,
            max_tokens  = max_tokens,
            temperature = temperature,
            timeout     = 300,
        )
        if json_mode:
            if schema:
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "structured_output", "schema": schema},
                }
            else:
                kwargs["response_format"] = {"type": "json_object"}
        if tools:
            kwargs["tools"] = tools

        last_err = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                with _MODAL_SEM:
                    return self._client.chat.completions.create(**kwargs)
            except Exception as e:
                last_err = e
                print(f"[GemmaClient] attempt {attempt}/{_MAX_RETRIES} failed: {e}")
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_SLEEP * attempt)

        raise RuntimeError(f"[GemmaClient] All retries failed. Last: {last_err}")

    def _call_text(self, messages: list[dict], **kw) -> str:
        return self._call(messages, **kw).choices[0].message.content or ""

    def _build_messages(self, system: str, user: str) -> list[dict]:
        return [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]

    def _image_to_data_uri(self, image_bytes: bytes, mime: str) -> str:
        return f"data:{mime};base64,{base64.b64encode(image_bytes).decode()}"

    def _inject_json_instructions(
        self, system_prompt: str, json_mode: bool, schema: Optional[dict]
    ) -> str:
        if not json_mode:
            return system_prompt
        if schema:
            return (
                system_prompt.rstrip()
                + "\n\nYou MUST respond with valid JSON matching this schema:\n"
                + json.dumps(schema, indent=2)
                + "\nNo preamble. No markdown. Only valid JSON."
            )
        return (
            system_prompt.rstrip()
            + "\n\nRespond ONLY with valid JSON. "
              "No preamble, no markdown fences, no explanation. "
              "First character must be { or [."
        )

    def _strip_fences(self, raw: str) -> str:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        return raw

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
        model_override:   Optional[str]   = None,   # ignored — Modal Gemma only
        primary_provider: Optional[str]   = None,   # ignored — Modal Gemma only
    ) -> str:
        sys_text = self._inject_json_instructions(system_prompt, json_mode, schema)
        messages = self._build_messages(sys_text, user_prompt)
        return self._call_text(
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
        primary_provider: Optional[str]  = None,
    ) -> dict | list:
        last_err = None
        for attempt in range(1, _MAX_RETRIES + 1):
            raw = self.complete(
                system_prompt, user_prompt,
                json_mode   = True,
                max_tokens  = max_tokens,
                temperature = temperature,
                schema      = schema,
            )
            try:
                return json.loads(self._strip_fences(raw))
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
        system_prompt: str,
        user_prompt:   str,
        image_bytes:   bytes,
        *,
        image_mime:       str           = "image/png",
        max_tokens:       int           = 16384,
        temperature:      float         = 0.1,
        primary_provider: Optional[str] = None,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text",      "text": user_prompt},
                    {"type": "image_url", "image_url": {
                        "url": self._image_to_data_uri(image_bytes, image_mime)
                    }},
                ],
            },
        ]
        return self._call_text(messages, max_tokens=max_tokens, temperature=temperature)

    def complete_vision_json(
        self,
        system_prompt: str,
        user_prompt:   str,
        image_bytes:   bytes,
        *,
        image_mime:  str            = "image/png",
        max_tokens:  int            = 16384,
        temperature: float          = 0.1,
        schema:      Optional[dict] = None,
    ) -> dict | list:
        sys_text = self._inject_json_instructions(system_prompt, True, schema)
        messages = [
            {"role": "system", "content": sys_text},
            {
                "role": "user",
                "content": [
                    {"type": "text",      "text": user_prompt},
                    {"type": "image_url", "image_url": {
                        "url": self._image_to_data_uri(image_bytes, image_mime)
                    }},
                ],
            },
        ]
        last_err = None
        for attempt in range(1, _MAX_RETRIES + 1):
            raw = self._call_text(
                messages, max_tokens=max_tokens, temperature=temperature,
                json_mode=True, schema=schema,
            )
            try:
                result = json.loads(self._strip_fences(raw))
                return result if isinstance(result, (dict, list)) else {}
            except json.JSONDecodeError as e:
                last_err = e
                print(f"[GemmaClient] vision JSON parse failed (attempt {attempt}/{_MAX_RETRIES}): {e}")
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_SLEEP)

        raise RuntimeError(
            f"[GemmaClient] Vision JSON failed after {_MAX_RETRIES} attempts: {last_err}"
        )

    # ── Native function calling ────────────────────────────────────────────────

    def _decls_to_tools(self, function_declarations: list[dict]) -> list[dict]:
        """Convert google.genai-style declarations to OpenAI tool format."""
        return [
            {
                "type": "function",
                "function": {
                    "name":        d.get("name", ""),
                    "description": d.get("description", ""),
                    "parameters":  d.get("parameters", {}),
                },
            }
            for d in function_declarations
        ]

    def _parse_tool_calls(self, response) -> dict:
        """
        Returns:
          {"type": "function_call", "name": str, "args": dict}  — tool chosen
          {"type": "text",          "text": str}                 — no tool call
        """
        msg        = response.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None) or []

        if tool_calls:
            tc = tool_calls[0]
            try:
                args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, AttributeError):
                args = {}
            return {"type": "function_call", "name": tc.function.name, "args": args}

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
        messages = self._build_messages(system_prompt, user_prompt)
        tools    = self._decls_to_tools(function_declarations)

        last_err = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                return self._parse_tool_calls(
                    self._call(messages, max_tokens=max_tokens,
                               temperature=temperature, tools=tools)
                )
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
        tools    = self._decls_to_tools(function_declarations)
        messages = [{"role": "system", "content": system_prompt}]

        for turn in conversation:
            role = turn.get("role", "user")

            if role == "user" and turn.get("text"):
                messages.append({"role": "user", "content": turn["text"]})

            elif role == "model":
                fc = turn.get("function_call", {})
                if fc:
                    messages.append({
                        "role":       "assistant",
                        "content":    None,
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
                fr          = turn.get("function_response", {})
                resp        = fr.get("response", {})
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
                return self._parse_tool_calls(
                    self._call(messages, max_tokens=max_tokens,
                               temperature=temperature, tools=tools)
                )
            except Exception as e:
                last_err = e
                print(f"[GemmaClient] multi-turn attempt {attempt}/{_MAX_RETRIES} failed: {e}")
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_SLEEP * attempt)

        raise RuntimeError(f"[GemmaClient] Multi-turn tool call failed: {last_err}")