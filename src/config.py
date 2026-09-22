"""Project configuration loaded from the repository's config.yaml."""

import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

# Load local development secrets before resolving ${ENV_VAR} references.
load_dotenv(PROJECT_ROOT / ".env")


def _resolve_tool_path(value: str) -> str:
    """Resolve a relative configured tool path against the project root."""
    if not value:
        return value
    path = Path(value)
    if path.is_absolute():
        return str(path)
    project_path = PROJECT_ROOT / path
    return str(project_path) if project_path.exists() else value


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


def load_config() -> dict:
    """Load the repository config with environment-variable expansion."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Configuration file not found: {CONFIG_PATH}")
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    return _resolve_env_recursive(raw)


def get_llm_config() -> dict:
    raw = dict(load_config().get("llm", {}))
    provider = str(raw.get("provider", "")).strip().lower()

    provider_configs = raw.get("providers")
    if isinstance(provider_configs, dict):
        selected = provider_configs.get(provider)
        if not isinstance(selected, dict):
            available = ", ".join(sorted(str(name) for name in provider_configs))
            raise ValueError(
                f"No LLM configuration found for provider {provider!r}. "
                f"Available providers: {available or 'none'}."
            )
        common = raw.get("common", {})
        cfg = dict(common) if isinstance(common, dict) else {}
        cfg.update(selected)
        cfg["provider"] = provider
    else:
        # Keep accepting the original flat llm configuration.
        cfg = raw

    env_names = {
        "deepseek": "DEEPSEEK_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
    }
    env_name = env_names.get(provider)
    if env_name and os.environ.get(env_name):
        cfg["api_key"] = os.environ[env_name]
    return cfg


def get_iverilog_config() -> dict:
    cfg = dict(load_config().get("iverilog", {}))
    cfg["binary"] = _resolve_tool_path(
        os.environ.get("PAPER2GATE_IVERILOG_BINARY")
        or str(cfg.get("binary") or "")
        or "iverilog"
    )
    cfg.setdefault("flags", "-g2012")
    return cfg


def get_vivado_config() -> dict:
    cfg = dict(load_config().get("vivado", {}))
    cfg["binary"] = _resolve_tool_path(
        os.environ.get("PAPER2GATE_VIVADO_BINARY")
        or str(cfg.get("binary") or "")
        or "vivado"
    )
    return cfg


def get_output_config() -> dict:
    return load_config().get("output", {})
