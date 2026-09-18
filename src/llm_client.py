"""LLM API client supporting Anthropic, OpenAI, and DeepSeek, with structured JSON output."""

import re
import time

from src.config import get_llm_config


class LLMClient:
    TIMING_OPERATIONS = (
        "classify_scheme",
        "extract_innovations",
        "generate_modmul",
        "fix_syntax",
        "fix_function",
    )

    def __init__(self):
        llm_cfg = get_llm_config()
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
        self._timings = self._empty_timing_data()

    @classmethod
    def _empty_timing_data(cls) -> dict:
        return {
            operation: {"calls": 0, "seconds": 0.0}
            for operation in cls.TIMING_OPERATIONS
        }

    def timing_snapshot(self) -> dict:
        """Return rounded LLM timing totals for the current paper."""
        by_operation = {
            operation: {
                "calls": values["calls"],
                "seconds": round(values["seconds"], 3),
            }
            for operation, values in self._timings.items()
        }
        return {
            "llm_seconds": round(sum(v["seconds"] for v in self._timings.values()), 3),
            "llm_calls": sum(v["calls"] for v in self._timings.values()),
            "llm_by_operation": by_operation,
        }

    def _record_timing(self, operation: str, calls: int, seconds: float) -> None:
        if operation not in self._timings:
            self._timings[operation] = {"calls": 0, "seconds": 0.0}
        self._timings[operation]["calls"] += calls
        self._timings[operation]["seconds"] += seconds

    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        stage: str = "generate",
        max_retries: int = 4,
        operation: str | None = None,
    ) -> str:
        """
        Generate a response and return the raw text.

        Args:
            stage: "extract" → expects JSON, applies _extract_json.
                   "generate" (default) → expects code, returns raw.
            operation: Optional timing label. When omitted, ``stage`` is used.
        """
        operation_name = operation or stage
        started = time.perf_counter()
        attempts = 0
        try:
            stage_model_map = {
                "extract": self.extract_model,
                "generate": self.generate_model,
            }
            model = stage_model_map.get(stage, self.model)

            last_error = None
            for attempt in range(1 + max_retries):
                attempts += 1
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
        finally:
            self._record_timing(operation_name, attempts, time.perf_counter() - started)

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
