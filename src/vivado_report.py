"""Run Vivado synthesis on generated butterfly and extract resource usage."""

import re
import shutil
import subprocess
from pathlib import Path

from src.config import get_vivado_config
from src.console import console


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
    part: str = "xc7a35tcsg324-1",
) -> dict | None:
    """
    Run Vivado synthesis and return resource counts.

    Returns dict with LUT/FF/DSP/BRAM keys, or None on failure.
    """
    vivado = _find_vivado()
    project_root = Path(__file__).parent.parent

    # Write TCL script using absolute paths for robustness
    tcl = _generate_tcl(verilog_files, top, part, output_dir)
    tcl_path = output_dir / "_synth.tcl"
    tcl_path.write_text(tcl, encoding="utf-8")

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
