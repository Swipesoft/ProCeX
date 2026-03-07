"""
utils/llm_client.py
Unified text-completion client: Claude → Gemini → OpenAI fallback chain.
Single interface: client.complete(system, user, json_mode) → str
"""

from __future__ import annotations
import json
import time
import re
from typing import Optional
from config import ProcExConfig


class LLMClient:
    """
    Tries Claude first. On failure/unavailability → Gemini. On failure → OpenAI.
    All three share the same call signature.
    """

    def __init__(self, cfg: ProcExConfig):
        self.cfg = cfg
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

    # ── Primary interface ─────────────────────────────────────────────────

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        json_mode: bool = False,
        max_tokens: int = 8192,
        temperature: float = 0.7,
        model_override: Optional[str] = None,   # force a specific model
    ) -> str:
        """
        Returns the text completion. Raises RuntimeError only if all providers fail.
        """
        providers = self._build_provider_chain(model_override)

        last_error = None
        for provider_fn in providers:
            try:
                result = provider_fn(system_prompt, user_prompt, max_tokens, temperature)
                if json_mode:
                    result = self._extract_json(result)
                return result
            except Exception as e:
                last_error = e
                print(f"[LLMClient] Provider failed: {e} — trying next...")
                time.sleep(1)

        raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int = 8192,
        temperature: float = 0.3,
    ) -> dict | list:
        """Convenience: complete + parse JSON."""
        raw = self.complete(
            system_prompt, user_prompt,
            json_mode=True, max_tokens=max_tokens, temperature=temperature
        )
        return json.loads(raw)

    # ── Provider chain ───────────────────────────────────────────────────

    def _build_provider_chain(self, model_override: Optional[str]):
        chain = []

        if self._anthropic:
            chain.append(
                lambda s, u, mt, temp: self._call_claude(s, u, mt, temp, model_override)
            )
        if self._genai:
            chain.append(
                lambda s, u, mt, temp: self._call_gemini(s, u, mt, temp, model_override)
            )
        if self._openai:
            chain.append(
                lambda s, u, mt, temp: self._call_openai(s, u, mt, temp, model_override)
            )

        if not chain:
            raise RuntimeError("No LLM providers configured. Set at least one API key.")
        return chain

    # ── Claude ────────────────────────────────────────────────────────────

    def _call_claude(self, system, user, max_tokens, temperature, model_override):
        model = model_override or self.cfg.claude_model
        # Clamp model to Claude models only
        if not model.startswith("claude"):
            model = self.cfg.claude_model

        response = self._anthropic.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text

    # ── Gemini ───────────────────────────────────────────────────────────

    def _call_gemini(self, system, user, max_tokens, temperature, model_override):
        from google.genai import types
        model = model_override or self.cfg.gemini_text_model
        if model.startswith("claude") or model.startswith("gpt"):
            model = self.cfg.gemini_text_model

        response = self._genai.models.generate_content(
            model=model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=max_tokens,
                temperature=temperature,
            ),
        )
        return response.text

    # ── OpenAI ───────────────────────────────────────────────────────────

    def _call_openai(self, system, user, max_tokens, temperature, model_override):
        model = model_override or self.cfg.openai_model
        if model.startswith("claude") or model.startswith("gemini"):
            model = self.cfg.openai_model

        response = self._openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content

    # ── JSON extraction ──────────────────────────────────────────────────

    @staticmethod
    def _extract_json(text: str) -> str:
        """
        Strip markdown code fences and extract the first valid JSON object/array.
        Handles: ```json ... ```, ``` ... ```, and raw JSON.
        """
        # Remove fences
        text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
        text = re.sub(r"```\s*$", "", text.strip(), flags=re.MULTILINE)
        text = text.strip()

        # Find first { or [ — handle models that add prose before JSON
        for start_char, end_char in [('{', '}'), ('[', ']')]:
            idx = text.find(start_char)
            if idx != -1:
                # Find matching close bracket
                depth = 0
                in_str = False
                escape = False
                for i, ch in enumerate(text[idx:], idx):
                    if escape:
                        escape = False
                        continue
                    if ch == '\\' and in_str:
                        escape = True
                        continue
                    if ch == '"' and not escape:
                        in_str = not in_str
                    if not in_str:
                        if ch == start_char:
                            depth += 1
                        elif ch == end_char:
                            depth -= 1
                            if depth == 0:
                                return text[idx:i+1]

        return text  # return as-is, let json.loads raise the error
