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


def validate_modmul_interface(
    verilog: str,
    data_width: int,
    module_name: str = "modmul",
) -> list[str]:
    """Validate the fixed modmul interface and synthesizable RTL constraints."""
    errors: list[str] = []
    if not is_valid_module(verilog):
        return ["missing module declaration or endmodule"]
    code = re.sub(r"//.*?$|/\*.*?\*/", "", verilog, flags=re.MULTILINE | re.DOTALL)
    module_names = re.findall(r"\bmodule\s+([A-Za-z_][A-Za-z0-9_$]*)", code)
    if module_names != [module_name]:
        errors.append(f"expected exactly one module named '{module_name}', found {module_names}")

    header = re.search(
        rf"\bmodule\s+{re.escape(module_name)}\s*(?:#\s*\([^)]*\))?\s*\((.*?)\)\s*;",
        verilog,
        re.DOTALL,
    )
    if not header:
        return [f"module '{module_name}' with an ANSI port list was not found"]

    block = header.group(1)
    expected_width = f"[{data_width - 1}:0]"
    if not re.search(r"\binput\b[^;]*\bclk\b", block):
        errors.append("missing input port 'clk'")
    if not re.search(r"\binput\b[^;]*\brst\b", block):
        errors.append("missing input port 'rst'")
    if not re.search(rf"\binput\s+{re.escape(expected_width)}\s+", block):
        errors.append(f"missing {expected_width} input declaration for A/B")
    for port in ("A", "B"):
        if not re.search(rf"\b{port}\b", block):
            errors.append(f"missing input port '{port}'")
    if not re.search(rf"\boutput\s+{re.escape(expected_width)}\s+\bR\b", block):
        errors.append(f"output port 'R' must be {expected_width}")

    forbidden = {
        r"%": "modulo operator",
        r"(?<!/)/(?!/)": "division operator",
        r"\binitial\b": "initial block",
        r"\$finish\b": "$finish",
        r"\$display\b": "$display",
    }
    for token, description in forbidden.items():
        found = bool(re.search(token, code))
        if found:
            errors.append(f"contains forbidden {description}")

    return errors


def cleanup(*paths: Path):
    """Delete temp files, ignoring errors."""
    for p in paths:
        try:
            p.unlink()
        except OSError:
            pass


def scan_ports(verilog_files: list[str], module_name: str) -> dict:
    """Scan module port list. Returns {has_clk, has_rst, inputs, outputs}.

    Handles both single-port-per-line and comma-separated multi-port styles:
        input [11:0] A, B, C;
        input clk, rst;
    """
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

        # Split into individual port fragments (by comma, then strip whitespace/comments)
        current_direction = None
        for line in ports_block.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Remove inline comments
            line = re.sub(r'//.*$', '', line).strip()
            if not line:
                continue

            # Split line by comma to handle "input [11:0] A, B, C"
            fragments = [f.strip() for f in line.split(",") if f.strip()]
            for frag in fragments:
                # Check if this fragment declares a direction
                dir_match = re.match(r'(input|output)\s+(.*)', frag)
                if dir_match:
                    current_direction = dir_match.group(1)
                    rest = dir_match.group(2)
                else:
                    rest = frag

                # Extract port name (skip wire/reg, optional range, then name)
                name_match = re.search(
                    r'(?:wire|reg)?\s*(?:\[[\w:*-]+\])?\s*(\w+)',
                    rest,
                )
                if not name_match:
                    continue
                name = name_match.group(1)
                if "clk" in name.lower():
                    has_clk = True
                elif "rst" in name.lower():
                    has_rst = True
                if current_direction == "input":
                    inputs.append(name)
                elif current_direction == "output":
                    outputs.append(name)

        return {"has_clk": has_clk, "has_rst": has_rst, "inputs": inputs, "outputs": outputs}
    return {"has_clk": True, "has_rst": True, "inputs": ["a", "b"], "outputs": ["r"]}
