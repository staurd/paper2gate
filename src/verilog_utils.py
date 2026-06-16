"""Shared Verilog utility functions used across pipeline stages."""

import re


def extract_verilog(text: str) -> str:
    """Extract Verilog source from LLM response, stripping markdown fences."""
    text = text.strip()
    fence = re.search(r"```(?:verilog|systemverilog)?\s*\n(.*?)```", text, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    return text


def is_valid_module(verilog: str) -> bool:
    """Check that output has both a module declaration and endmodule."""
    return bool(re.search(r'\bmodule\s+\w+', verilog)) and \
           bool(re.search(r'\bendmodule\b', verilog))
