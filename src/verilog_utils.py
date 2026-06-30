"""Shared Verilog utility functions used across pipeline stages."""

import re
from pathlib import Path


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


def cleanup(*paths: Path):
    """Delete temp files, ignoring errors."""
    for p in paths:
        try:
            p.unlink()
        except OSError:
            pass


def scan_ports(verilog_files: list[str], module_name: str) -> dict:
    """Scan module port list. Returns {has_clk, has_rst, inputs, outputs}."""
    for fpath in verilog_files:
        src = Path(fpath).read_text(encoding="utf-8", errors="ignore")
        m = re.search(
            rf'module\s+{module_name}\s*(?:#\([^)]*\))?\s*\(([^;]+)\)',
            src, re.DOTALL,
        )
        if not m:
            continue
        ports_block = m.group(1)
        inputs, outputs = [], []
        has_clk, has_rst = False, False
        for line in ports_block.split("\n"):
            m2 = re.match(
                r'(input|output)\s+(?:wire|reg)?\s*(?:\[[\w:*-]+\])?\s*(\w+)',
                line.strip(),
            )
            if not m2:
                continue
            direction, name = m2.group(1), m2.group(2)
            if "clk" in name.lower():
                has_clk = True
            elif "rst" in name.lower():
                has_rst = True
            if direction == "input":
                inputs.append(name)
            else:
                outputs.append(name)
        return {"has_clk": has_clk, "has_rst": has_rst, "inputs": inputs, "outputs": outputs}
    return {"has_clk": True, "has_rst": True, "inputs": ["a", "b"], "outputs": ["r"]}
