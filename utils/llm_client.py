"""
utils/llm_client.py
Unified text-completion client with per-call primary provider routing.

Default chain: Claude → Gemini → OpenAI
Each agent can specify a primary_provider ("claude"|"gemini"|"openai") to
move its preferred LLM to the front of the chain. The other two remain as
fallbacks in default order.

Per-agent routing is configured in ProcExConfig.agent_primary_llm.
Agents pass primary_provider= on each self.llm.complete() call.
"""
from __future__ import annotations
import json
import time
import re
from typing import Optional
from config import ProcExConfig


# Default provider order when no primary is specified
_DEFAULT_ORDER = ["claude", "gemini", "openai"]


class LLMClient:

    def __init__(self, cfg: ProcExConfig):
        self.cfg         = cfg
        self._anthropic  = None
        self._genai      = None
        self._openai     = None
        self._init_clients()

    def _init_clients(self):
        if self.cfg.anthropic_api_key:
            try:
                import anthropic
                self._anthropic = anthropic.Anthropic(api_key=self.cfg.anthropic_api_key)
            except ImportError:
                print("[LLMClient] anthropic SDK not installed — skipping Claude")

        if self.cfg.gemini_api_key:
            try:
                from google import genai
                self._genai = genai.Client(api_key=self.cfg.gemini_api_key)
            except ImportError:
                print("[LLMClient] google-genai SDK not installed — skipping Gemini")

        if self.cfg.openai_api_key:
            try:
                from openai import OpenAI
                self._openai = OpenAI(api_key=self.cfg.openai_api_key)
            except ImportError:
                print("[LLMClient] openai SDK not installed — skipping OpenAI")

    # ── Primary interface ─────────────────────────────────────────────────────

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        json_mode:        bool            = False,
        max_tokens:       int             = 16000,
        temperature:      float           = 0.7,
        model_override:   Optional[str]   = None,   # force a specific model string
        primary_provider: Optional[str]   = None,   # "claude" | "gemini" | "openai"
    ) -> str:
        """
        Returns the text completion. Raises RuntimeError only if all providers fail.

        primary_provider moves the named provider to position 0 of the chain.
        The other two providers remain as ordered fallbacks.
        model_override forces a specific model string on all providers that accept it.
        """
        chain = self._build_provider_chain(
            primary_provider=primary_provider,
            model_override=model_override,
            json_mode=json_mode,
        )

        last_error = None
        for provider_name, provider_fn in chain:
            try:
                result = provider_fn(system_prompt, user_prompt, max_tokens, temperature)
                if json_mode:
                    result = self._extract_json(result)
                return result
            except Exception as e:
                last_error = e
                print(f"[LLMClient] {provider_name} failed: {e} — trying next...")
                time.sleep(1)

        raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens:       int           = 16000,
        temperature:      float         = 0.3,
        primary_provider: Optional[str] = None,
    ) -> dict | list:
        """Convenience: complete + parse JSON. Passes primary_provider through."""
        raw = self.complete(
            system_prompt, user_prompt,
            json_mode=True,
            max_tokens=max_tokens,
            temperature=temperature,
            primary_provider=primary_provider,
        )
        return json.loads(raw)

    def complete_vision(
        self,
        system_prompt:    str,
        user_prompt:      str,
        image_bytes:      bytes,
        *,
        image_mime:       str           = "image/png",
        max_tokens:       int           = 16384,
        temperature:      float         = 0.1,
        primary_provider: Optional[str] = None,
    ) -> str:
        """
        Vision-capable completion: sends an image + text to a multimodal model.
        Provider preference: Gemini first (best spatial VLM), then Claude fallback.
        OpenAI is skipped unless it's the only option — gpt-5 vision costs are high.

        primary_provider: override to "claude" or "gemini" explicitly.
        image_bytes: raw PNG/JPEG bytes of the frame.
        """
        # Vision-capable providers in preferred order
        preferred = primary_provider or "gemini"
        vision_chain = self._build_provider_chain(
            primary_provider=preferred,
            model_override=None,
        )
        # Remove openai from vision chain (no vision support in current client)
        vision_chain = [(n, fn) for n, fn in vision_chain if n != "openai"]
        if not vision_chain:
            raise RuntimeError("No vision-capable providers available.")

        # Build provider-specific vision callers
        vision_callers = {
            "gemini": lambda s, u, mt, temp: self._call_gemini_vision(
                s, u, image_bytes, image_mime, mt, temp
            ),
            "claude": lambda s, u, mt, temp: self._call_claude_vision(
                s, u, image_bytes, image_mime, mt, temp
            ),
        }

        last_error = None
        for provider_name, _ in vision_chain:
            caller = vision_callers.get(provider_name)
            if caller is None:
                continue
            try:
                return caller(system_prompt, user_prompt, max_tokens, temperature)
            except Exception as e:
                last_error = e
                print(f"[LLMClient] {provider_name} vision failed: {e} — trying next...")
                time.sleep(1)

        raise RuntimeError(f"All vision providers failed. Last error: {last_error}")

    # ── Vision: Gemini ────────────────────────────────────────────────────────

    def _call_gemini_vision(self, system, user, image_bytes, mime, max_tokens, temperature):
        from google.genai import types
        import base64 as _b64

        model = self.cfg.gemini_text_model   # e.g. gemini-3-flash-preview

        # Build multimodal contents: image part + text part
        image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime)
        text_part  = types.Part.from_text(text=user)

        # Timeout: vision calls can be slow for large frames but should never
        # hang indefinitely. 90s is generous for a 3-frame VLMCritic call.
        timeout_secs = 90 + (max_tokens // 1000) * 2
        http_opts    = types.HttpOptions(timeout=timeout_secs * 1000)  # ms

        response = self._genai.models.generate_content(
            model    = model,
            contents = types.Content(parts=[image_part, text_part], role="user"),
            config   = types.GenerateContentConfig(
                system_instruction = system,
                max_output_tokens  = max_tokens,
                temperature        = temperature,
                http_options       = http_opts,
            ),
        )
        return response.text

    # ── Vision: Claude ────────────────────────────────────────────────────────

    def _call_claude_vision(self, system, user, image_bytes, mime, max_tokens, temperature):
        import base64 as _b64
        b64_data = _b64.b64encode(image_bytes).decode("utf-8")

        response = self._anthropic.messages.create(
            model      = self.cfg.claude_model,
            max_tokens = max_tokens,
            temperature= temperature,
            system     = system,
            messages   = [{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type":       "base64",
                            "media_type": mime,
                            "data":       b64_data,
                        },
                    },
                    {"type": "text", "text": user},
                ],
            }],
        )
        return response.content[0].text

    # ── Provider chain builder ────────────────────────────────────────────────

    def _build_provider_chain(
        self,
        primary_provider: Optional[str],
        model_override:   Optional[str],
        json_mode:        bool = False,
    ) -> list[tuple[str, callable]]:
        """
        Build the ordered list of (name, callable) provider functions.

        If primary_provider is given, that provider is placed first.
        Available providers are ordered by _DEFAULT_ORDER for the remainder.
        Unavailable providers (no client / no key) are silently skipped.
        json_mode is threaded into _call_gemini so response_mime_type is set.
        """
        # All available providers in default order
        all_providers = {
            "claude": (
                self._anthropic,
                lambda s, u, mt, temp: self._call_claude(s, u, mt, temp, model_override)
            ),
            "gemini": (
                self._genai,
                lambda s, u, mt, temp: self._call_gemini(s, u, mt, temp, model_override,
                                                            json_mode=json_mode)
            ),
            "openai": (
                self._openai,
                lambda s, u, mt, temp: self._call_openai(s, u, mt, temp, model_override)
            ),
        }

        # Build ordered list of available providers
        order = list(_DEFAULT_ORDER)   # ["claude", "gemini", "openai"]

        if primary_provider and primary_provider in order:
            # Move primary to front, keep others in default order
            order.remove(primary_provider)
            order.insert(0, primary_provider)

        chain = []
        for name in order:
            client, fn = all_providers[name]
            if client is not None:
                chain.append((name, fn))

        if not chain:
            raise RuntimeError("No LLM providers configured. Set at least one API key.")

        return chain

    # ── Claude ────────────────────────────────────────────────────────────────

    def _call_claude(self, system, user, max_tokens, temperature, model_override):
        model = model_override or self.cfg.claude_model
        if not model.startswith("claude"):
            model = self.cfg.claude_model

        # Timeout scales with max_tokens — large documentary writes need ~120s.
        # Without a timeout, dropped connections hang forever (KeyboardInterrupt
        # was the only escape). Formula: 60s base + 3s per 1000 output tokens.
        timeout_secs = 60 + (max_tokens // 1000) * 3

        import httpx
        response = self._anthropic.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
            timeout=httpx.Timeout(
                connect = 15.0,
                read    = float(timeout_secs),
                write   = 30.0,
                pool    = 10.0,
            ),
        )
        return response.content[0].text

    # ── Gemini ────────────────────────────────────────────────────────────────

    def _call_gemini(self, system, user, max_tokens, temperature, model_override,
                     json_mode: bool = False):
        from google.genai import types
        model = model_override or self.cfg.gemini_text_model
        if model.startswith("claude") or model.startswith("gpt"):
            model = self.cfg.gemini_text_model

        timeout_secs = 60 + (max_tokens // 1000) * 3
        http_opts    = types.HttpOptions(timeout=timeout_secs * 1000)  # ms

        # Use response_mime_type to force JSON output from Gemini.
        # Prompt-only JSON instructions are unreliable — Gemini frequently
        # ignores them and returns prose, causing empty query fields.
        config_kwargs = dict(
            system_instruction = system,
            max_output_tokens  = max_tokens,
            temperature        = temperature,
            http_options       = http_opts,
        )
        if json_mode:
            config_kwargs["response_mime_type"] = "application/json"

        response = self._genai.models.generate_content(
            model=model,
            contents=user,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        return response.text

    # ── OpenAI ────────────────────────────────────────────────────────────────

    # Models that use max_completion_tokens instead of max_tokens,
    # and do not accept a temperature parameter.
    _OPENAI_REASONING_PREFIXES = ("o1", "o3", "o4", "gpt-5")

    def _call_openai(self, system, user, max_tokens, temperature, model_override):
        model = model_override or self.cfg.openai_model
        if model.startswith("claude") or model.startswith("gemini"):
            model = self.cfg.openai_model

        is_reasoning = any(model.startswith(p) for p in self._OPENAI_REASONING_PREFIXES)

        kwargs: dict = dict(
            model    = model,
            messages = [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )

        if is_reasoning:
            # Reasoning models: max_completion_tokens, no temperature
            kwargs["max_completion_tokens"] = max_tokens
        else:
            # Classic chat models: max_tokens + temperature
            kwargs["max_tokens"]  = max_tokens
            kwargs["temperature"] = temperature

        response = self._openai.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    # ── JSON extraction ───────────────────────────────────────────────────────

    @staticmethod
    def _extract_json(text: str) -> str:
        """
        Strip markdown fences and repair malformed LLM JSON output.
        Uses json-repair to handle truncation, missing commas, single quotes,
        trailing commas, comments, and extra text — all in one call.
        """
        from json_repair import repair_json
        text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
        text = re.sub(r"```\s*$",           "", text.strip(), flags=re.MULTILINE)
        return repair_json(text.strip(), ensure_ascii=False)