"""LLM API client supporting Anthropic, OpenAI, and DeepSeek, with structured JSON output."""

import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Auto-load .env from project root
_env_path = Path(__file__).parent.parent / ".env"
load_dotenv(_env_path)


def _resolve_env(value: str) -> str:
    """Resolve ${ENV_VAR} placeholders in config values."""
    pattern = re.compile(r"\$\{(\w+)\}")
    match = pattern.fullmatch(value)
    if match:
        return os.environ.get(match.group(1), "")
    return pattern.sub(lambda m: os.environ.get(m.group(1), ""), value)


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _resolve_config_env(raw)


def _resolve_config_env(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _resolve_config_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_config_env(v) for v in obj]
    if isinstance(obj, str):
        return _resolve_env(obj)
    return obj


class LLMClient:
    def __init__(self, config_path: str = "config.yaml"):
        config = load_config(config_path)
        llm_cfg = config["llm"]
        self.provider = llm_cfg["provider"]
        self.model = llm_cfg["model"]
        self.extract_model = llm_cfg.get("extract_model", llm_cfg["model"])
        self.generate_model = llm_cfg.get("generate_model", llm_cfg["model"])
        self.max_tokens = llm_cfg.get("max_tokens", 8192)
        self.temperature = llm_cfg.get("temperature", 0.2)

        api_key = llm_cfg.get("api_key", "")
        if not api_key:
            raise ValueError(
                "LLM API key not configured. "
                "Set DEEPSEEK_API_KEY, ANTHROPIC_API_KEY, or OPENAI_API_KEY."
            )

        self._api_key = api_key
        self._base_url = llm_cfg.get("base_url", None)
        self._client = None  # lazy init

    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        stage: str = "generate",
        max_retries: int = 4,
    ) -> str:
        """
        Generate a response and return the raw text.

        Args:
            stage: "extract" → expects JSON, applies _extract_json.
                   "generate" (default) → expects code, returns raw.
        """
        import time
        stage_model_map = {
            "extract": self.extract_model,
            "generate": self.generate_model,
        }
        model = stage_model_map.get(stage, self.model)

        last_error = None
        for attempt in range(1 + max_retries):
            try:
                if self.provider == "anthropic":
                    raw = self._call_anthropic_raw(system_prompt, user_prompt, model)
                else:
                    raw = self._call_openai_raw(system_prompt, user_prompt, model)

                if stage == "extract":
                    return self._extract_json(raw)
                return raw
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    wait = 2 ** attempt
                    time.sleep(wait)
                    continue
        raise RuntimeError(
            f"LLM API call failed after {max_retries + 1} attempts: {last_error}"
        )

    def _get_client(self):
        if self._client is not None:
            return self._client
        if self.provider == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(api_key=self._api_key)
        elif self.provider == "deepseek":
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self._api_key,
                base_url="https://api.deepseek.com",
            )
        else:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
            )
        return self._client

    def _call_anthropic_raw(self, system_prompt: str, user_prompt: str, model: str) -> str:
        client = self._get_client()
        response = client.messages.create(
            model=model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text

    def _call_openai_raw(self, system_prompt: str, user_prompt: str, model: str) -> str:
        client = self._get_client()
        response = client.chat.completions.create(
            model=model,
            max_completion_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content or ""

    @staticmethod
    def _extract_json(text: str) -> str:
        """
        Extract JSON object from LLM response.

        Handles:
        - Markdown code fences anywhere in the response (re.search)
        - LLM preamble like "Here is the JSON:"
        - Nested braces in JSON string values (uses outermost { })
        """
        text = text.strip()
        # Remove markdown code fences if present (search anywhere, not just start)
        fence = re.search(r"```(?:json)?\s*\n(.*?)\n?```", text, re.DOTALL)
        if fence:
            text = fence.group(1).strip()
        # Find the outermost { ... }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
        return text
