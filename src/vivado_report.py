"""Run Vivado synthesis on generated butterfly and extract resource usage."""

import re
import shutil
import subprocess
from pathlib import Path

from src.config import get_vivado_config
from src.console import console

DEFAULT_PART = "xc7a35tcsg324-1"

# Vivado's -part wants device + package + speed grade, but papers usually print
# only the device name ("XC7A100T"). These entries supply the package/speed
# grade most common in the PQC literature for that device — same die, but NOT
# necessarily the paper's exact board. resolve_part() reports when it guessed,
# so the user can override with --part.
DEVICE_PARTS = {
    "xc7a35t": "xc7a35tcsg324-1",
    "xc7a50t": "xc7a50tcsg324-1",
    "xc7a100t": "xc7a100tcsg324-1",
    "xc7a200t": "xc7a200tffg1156-3",
    "xc7z020": "xc7z020clg400-1",
    "xc7z045": "xc7z045ffg900-2",
    "xc7k325t": "xc7k325tffg900-2",
    "xc7k410t": "xc7k410tffg900-2",
    "xc7vx485t": "xc7vx485tffg1157-1",
    "xc7v2000t": "xc7v2000tflg1925-1",
    "xc6slx45": "xc6slx45csg324-3",
    "xc6slx150": "xc6slx150fgg484-3",
}

# A legal Xilinx part carries device + package + speed grade, e.g.
# "xc7a200tffg1156-3". The package (2+ letters then 3+ digits: csg324, ffg1156)
# is required — "xc7a100t-3" names a device and a speed grade but no package,
# and Vivado rejects it. Papers do print that form, so it must fall through to
# the lookup table rather than being passed through as-is.
_COMPLETE_PART = re.compile(r"^xc\d[a-z0-9]*[a-z]{2,}\d{3,}-\d$")


def resolve_part(raw: str | None) -> tuple[str, str]:
    """Map a paper-reported FPGA string to a legal Vivado part.

    Returns (part, provenance), provenance being one of:
      "paper"     - the paper's string is already a complete part
      "completed" - device name only; package/speed grade filled from DEVICE_PARTS
      "unknown"   - unrecognised string; fell back to DEFAULT_PART
      "empty"     - nothing reported; fell back to DEFAULT_PART
    """
    tokens = re.findall(r"[a-z0-9][a-z0-9-]*", (raw or "").lower())
    if not tokens:
        return DEFAULT_PART, "empty"

    # Prefer a complete part anywhere in the string ("Xilinx Artix-7
    # xc7a200tffg1156-3") over a bare device name.
    for tok in tokens:
        if _COMPLETE_PART.match(tok):
            return tok, "paper"

    for tok in tokens:
        if tok in DEVICE_PARTS:
            return DEVICE_PARTS[tok], "completed"
        # Device name plus package but no speed grade, e.g. "xc7a100tcsg324".
        for device, part in DEVICE_PARTS.items():
            if tok.startswith(device):
                return part, "completed"

    return DEFAULT_PART, "unknown"


def _find_vivado() -> str:
    """Find vivado binary. Checks config, then PATH."""
    try:
        path = get_vivado_config().get("binary", "")
        if path and Path(path).exists():
            return path
    except Exception:
        pass
    # Check PATH
    found = shutil.which("vivado")
    if found:
        return found
    # Common locations
    for base in ["C:/Xilinx", "D:/Xilinx", "D:/Vivado2018.3", "C:/Vivado"]:
        for root, dirs, _ in Path(base).glob("**/bin"):
            vivado_exe = Path(str(root)) / "vivado.bat"
            if vivado_exe.exists():
                return str(vivado_exe)
    return "vivado"


def run_synthesis(
    verilog_files: list[str],
    top: str,
    output_dir: Path,
    part: str = DEFAULT_PART,
) -> dict | None:
    """
    Run Vivado synthesis and return resource counts.

    Reports go to <output_dir>/synthesis/<top>/ so the several synthesis runs
    of one invocation (modmul, then butterfly) do not overwrite each other's
    utilization.rpt / _synth.tcl.

    Returns dict with LUT/FF/DSP/BRAM keys, or None on failure.
    """
    vivado = _find_vivado()
    # Everything Vivado is handed (the tcl, the sources it reads, the reports
    # it writes) is resolved first: Vivado resolves them against its own CWD,
    # not ours.
    verilog_files = [str(Path(f).resolve()) for f in verilog_files]
    output_dir = (Path(output_dir) / "synthesis" / top).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write TCL script using absolute paths for robustness
    tcl = _generate_tcl(verilog_files, top, part, output_dir)
    tcl_path = output_dir / "_synth.tcl"
    tcl_path.write_text(tcl, encoding="utf-8")

    console.print(f"    Part: [cyan]{part}[/cyan]   (top: {top})")
    console.print("    Running Vivado synthesis (~20s)...")
    try:
        result = subprocess.run(
            [vivado, "-mode", "batch", "-source", str(tcl_path)],
            capture_output=True, text=True, timeout=120,
            cwd=str(output_dir),
        )
    except FileNotFoundError:
        console.print("    [yellow]Vivado not found — skipping synthesis[/yellow]")
        return None
    except subprocess.TimeoutExpired:
        console.print("    [yellow]Vivado timed out[/yellow]")
        return None

    if result.returncode != 0:
        console.print("    [red]Vivado synthesis failed[/red]")
        for line in result.stderr.strip().split("\n")[:3]:
            console.print(f"      [dim]{line.strip()}[/dim]")
        return None

    # Parse utilization report
    util_path = output_dir / "utilization.rpt"
    if util_path.exists():
        stats = _parse_utilization(util_path.read_text(encoding="utf-8"))
        _print_stats(stats)
        return stats

    return None


def _generate_tcl(verilog_files: list[str], top: str, part: str, out_dir: Path) -> str:
    reads = "\n".join(f"read_verilog {{{f}}}" for f in verilog_files)
    util_path = str(out_dir / "utilization.rpt").replace("\\", "/")
    time_path = str(out_dir / "timing.rpt").replace("\\", "/")
    return f"""
{reads}
synth_design -top {top} -part {part}
report_utilization -file {{{util_path}}}
report_timing -file {{{time_path}}}
puts "=== SYNTH DONE ==="
"""


def _parse_utilization(text: str) -> dict:
    """Extract LUT/FF/DSP/BRAM from Vivado utilization report."""
    stats = {}
    for line in text.split("\n"):
        line = line.strip()
        if "Slice LUTs" in line and "|" in line:
            parts = [x.strip() for x in line.split("|") if x.strip()]
            if len(parts) >= 2:
                try: stats["LUT"] = int(parts[1])
                except ValueError: pass
        if "Slice Registers" in line and "|" in line:
            parts = [x.strip() for x in line.split("|") if x.strip()]
            if len(parts) >= 2:
                try: stats["FF"] = int(parts[1])
                except ValueError: pass
        if line.startswith("| DSPs") and "|" in line:
            parts = [x.strip() for x in line.split("|") if x.strip()]
            if len(parts) >= 2:
                try: stats["DSP"] = int(parts[1])
                except ValueError: pass
        if "Block RAM Tile" in line and "|" in line:
            parts = [x.strip() for x in line.split("|") if x.strip()]
            if len(parts) >= 2:
                try: stats["BRAM"] = int(parts[1])
                except ValueError: pass
    return stats


def _print_stats(stats: dict):
    console.print("    ┌──────────┬────────┐")
    console.print(f"    │ Slice LUTs │ {stats.get('LUT', '?'):>6} │")
    console.print(f"    │ Slice Regs │ {stats.get('FF', '?'):>6} │")
    console.print(f"    │ DSP48E1    │ {stats.get('DSP', '?'):>6} │")
    console.print(f"    │ Block RAM  │ {stats.get('BRAM', '?'):>6} │")
    console.print("    └──────────┴────────┘")
