"""LLM API client supporting Anthropic, OpenAI, and DeepSeek, with structured JSON output."""

import re
import time
from pathlib import Path

from src.config import get_llm_config


class LLMClient:
    SUPPORTED_PROVIDERS = frozenset({"anthropic", "deepseek", "openai"})
    TIMING_OPERATIONS = (
        "classify_scheme",
        "extract_innovations",
        "classify_figures",
        "analyze_figure_page",
        "generate_modmul",
        "fix_syntax",
        "fix_function",
    )
    EXTRACT_OPERATIONS = frozenset(
        {"classify_scheme", "extract_innovations", "classify_figures"}
    )

    def __init__(self, config: dict | None = None, *, vision: bool = False):
        llm_cfg = dict(config) if config is not None else get_llm_config()
        self.vision = vision
        self.provider = str(llm_cfg.get("provider", "")).strip().lower()
        if self.provider not in self.SUPPORTED_PROVIDERS:
            supported = ", ".join(sorted(self.SUPPORTED_PROVIDERS))
            raise ValueError(
                f"Unsupported LLM provider {self.provider!r}. "
                f"Choose one of: {supported}."
            )
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
        configured_base_url = str(llm_cfg.get("base_url") or "").strip()
        self._base_url = configured_base_url.rstrip("/") or None
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

    def metadata_snapshot(self) -> dict:
        """Return the provider and effective model for each pipeline operation."""
        by_operation = {}
        for operation in self.TIMING_OPERATIONS:
            model = self._model_for_operation(operation)
            by_operation[operation] = {
                "provider": self.provider,
                "model": model,
            }
        return {
            "provider": self.provider,
            "model": self.model,
            "extract_model": self.extract_model,
            "generate_model": self.generate_model,
            "by_operation": by_operation,
        }

    def _model_for_operation(self, operation: str) -> str:
        if self.vision and operation == "analyze_figure_page":
            return self.model
        if operation in self.EXTRACT_OPERATIONS:
            return self.extract_model
        return self.generate_model

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
                    elif self.provider in ("deepseek", "openai"):
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

    def generate_multimodal_structured(
        self,
        system_prompt: str,
        user_text: str,
        image_path: str | Path,
        *,
        max_retries: int = 2,
        operation: str = "analyze_figure_page",
    ) -> str:
        """Send text plus a local image to an OpenAI-compatible vision model."""
        if self.provider not in {"openai", "deepseek"}:
            raise ValueError(
                f"Vision requests require an OpenAI-compatible provider, got {self.provider!r}"
            )
        image_bytes = Path(image_path).read_bytes()
        import base64

        image_data = base64.b64encode(image_bytes).decode("ascii")
        user_content = [
            {"type": "text", "text": user_text},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_data}"},
            },
        ]
        started = time.perf_counter()
        attempts = 0
        try:
            last_error = None
            for attempt in range(1 + max_retries):
                attempts += 1
                try:
                    raw = self._call_openai_raw(
                        system_prompt,
                        user_content,
                        self._model_for_operation(operation),
                    )
                    return self._extract_json(raw)
                except Exception as exc:
                    last_error = exc
                    if attempt < max_retries:
                        time.sleep(2 ** attempt)
            raise RuntimeError(
                f"Vision API call failed after {max_retries + 1} attempts: {last_error}"
            )
        finally:
            self._record_timing(operation, attempts, time.perf_counter() - started)

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
                base_url=self._base_url or "https://api.deepseek.com",
            )
        elif self.provider == "openai":
            from openai import OpenAI
            kwargs = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = OpenAI(**kwargs)
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

    def _call_openai_raw(
        self,
        system_prompt: str,
        user_prompt: str | list[dict],
        model: str,
    ) -> str:
        client = self._get_client()
        request = {
            "model": model,
            "max_completion_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if self.temperature is not None:
            request["temperature"] = self.temperature
        try:
            response = client.chat.completions.create(**request)
        except Exception as exc:
            if not self._is_legacy_token_error(exc):
                raise
            request["max_tokens"] = request.pop("max_completion_tokens")
            response = client.chat.completions.create(**request)
        return response.choices[0].message.content or ""

    @staticmethod
    def _is_legacy_token_error(error: Exception) -> bool:
        """Return whether a relay rejected the modern token-limit parameter."""
        message = str(error).lower()
        if "max_completion_tokens" not in message:
            return False
        return any(
            marker in message
            for marker in (
                "unsupported",
                "unknown",
                "unrecognized",
                "unexpected",
                "invalid",
                "not support",
                "does not support",
                "not allowed",
                "not accepted",
                "extra",
            )
        )

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
        # Find the outermost JSON object or array.
        object_start = text.find("{")
        object_end = text.rfind("}")
        array_start = text.find("[")
        array_end = text.rfind("]")
        if array_start != -1 and array_end > array_start and (
            object_start == -1 or array_start < object_start
        ):
            text = text[array_start : array_end + 1]
        elif object_start != -1 and object_end > object_start:
            text = text[object_start : object_end + 1]
        return text
