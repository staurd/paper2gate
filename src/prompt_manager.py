"""Unified prompt management: loading, caching, and Jinja2 rendering.

Usage:
    from src.prompt_manager import load_prompt

    # Render with Jinja2 variables
    prompt = load_prompt("generate_modmul.jinja", module_name="modmul", ...)

    # Load raw text without rendering
    text = load_prompt("fix_verilog.jinja")
"""

import functools
from pathlib import Path

from jinja2 import Template

PROMPT_DIR = Path(__file__).parent.parent / "prompts"


def load_prompt(name: str, **kwargs) -> str:
    """
    Load a prompt file from prompts/ and optionally render with Jinja2.

    Args:
        name: Prompt filename (e.g., "generate_modmul.jinja")
        **kwargs: Jinja2 template variables (optional)

    Returns:
        Rendered prompt string. If no kwargs given, returns raw source.
    """
    source = _read_prompt(name)
    if kwargs:
        return Template(source).render(**kwargs)
    return source


@functools.lru_cache(maxsize=16)
def _read_prompt(name: str) -> str:
    """Read a prompt file from disk. Results are cached."""
    return (PROMPT_DIR / name).read_text(encoding="utf-8")
