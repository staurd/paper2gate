"""Unified configuration loaded once from config.yaml. All modules import from here."""

import functools
import os
import re
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def _resolve_env(value: str) -> str:
    """Resolve ${ENV_VAR} references in values."""
    m = re.fullmatch(r"\$\{(\w+)\}", value)
    if m:
        return os.environ.get(m.group(1), "")
    return re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), ""), value)


def _resolve_env_recursive(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _resolve_env_recursive(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env_recursive(v) for v in obj]
    if isinstance(obj, str):
        return _resolve_env(obj)
    return obj


@functools.lru_cache(maxsize=1)
def load_config() -> dict:
    """Load and cache config.yaml with env var resolution. Returns the full dict."""
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    return _resolve_env_recursive(raw)


def get_llm_config() -> dict:
    return load_config().get("llm", {})


def get_iverilog_config() -> dict:
    return load_config().get("iverilog", {})


def get_yosys_config() -> dict:
    return load_config().get("yosys", {})


def get_vivado_config() -> dict:
    return load_config().get("vivado", {})


def get_output_config() -> dict:
    return load_config().get("output", {})
