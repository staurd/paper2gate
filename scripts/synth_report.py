"""FPGA resource reporter: Yosys (default), Vivado, or static estimate.
Usage: python scripts/synth_report.py outputs/xxx_run/ [--yosys|--vivado|--estimate]
"""

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).parent.parent


def _get_yosys_env() -> tuple[str, dict]:
    cfg = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))
    yosys_cfg = cfg.get("yosys", {})
    binary = yosys_cfg.get("binary", "yosys")
    lib_path = yosys_cfg.get("lib_path", "")
    env = os.environ.copy()
    if lib_path:
        env["PATH"] = f"{lib_path};" + env.get("PATH", "")
    bin_dir = str(Path(binary).parent)
    if bin_dir not in env["PATH"]:
        env["PATH"] = f"{bin_dir};" + env["PATH"]
    return binary, env


def _run_yosys(verilog_files: list[str], top: str, part: str = "xc7a35tcsg324-1") -> dict:
    read_cmds = "\n".join(f"read_verilog {f}" for f in verilog_files)
    script = f"""
{read_cmds}
hierarchy -top {top}
proc; opt; fsm; opt; memory; opt
techmap; opt
synth_xilinx -family xc7
stat
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".ys", delete=False) as f:
        f.write(script)
        script_path = f.name

    yosys_bin, env = _get_yosys_env()
    result = subprocess.run([yosys_bin, "-s", script_path],
                            capture_output=True, text=True, timeout=120, env=env)
    Path(script_path).unlink()
    return _parse_yosys(result.stdout + result.stderr)


def _parse_yosys(text: str) -> dict:
    # Only count from the final "=== design hierarchy ===" section (total)
    stats = {"LUT": 0, "FF": 0, "DSP": 0, "BRAM": 0}
    # Find last "=== design hierarchy ===" block and extract only lines after it
    parts = text.split("=== design hierarchy ===")
    section = parts[-1] if len(parts) > 1 else text
    for line in section.split("\n"):
        line = line.lstrip()
        if "End of script" in line:
            break
        m = re.match(r'(\d+)\s+LUT(\d)', line)
        if m: stats["LUT"] += int(m.group(1))
        m = re.match(r'(\d+)\s+DSP48E1', line)
        if m: stats["DSP"] += int(m.group(1))
        m = re.match(r'(\d+)\s+FD[CRS]E', line)
        if m: stats["FF"] += int(m.group(1))
        m = re.match(r'(\d+)\s+RAMB\d+', line)
        if m: stats["BRAM"] += int(m.group(1))
    return stats


def collect_verilog_files(run_dir: str) -> list[str]:
    vfiles = []
    for sub in ["verilog", "butterfly", "ntt"]:
        d = Path(run_dir) / sub
        if d.exists():
            vfiles.extend(str(f) for f in d.glob("*.v"))
    return vfiles


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else None
    if not run_dir:
        print("Usage: python scripts/synth_report.py <run_dir> [--vivado]")
        sys.exit(1)

    vfiles = collect_verilog_files(run_dir)
    if not vfiles:
        print(f"No .v files in {run_dir}")
        sys.exit(1)

    top = "butterfly" if any("butterfly" in f for f in vfiles) else "ntt_core"
    print(f"Files: {len(vfiles)}, Top: {top}")
    for f in vfiles:
        print(f"  {f}")

    stats = _run_yosys(vfiles, top)
    CAL = 1.7  # Yosys-to-Vivado LUT calibration factor
    print(f"\n  Yosys raw      Vivado est")
    print(f"LUTs : {stats['LUT']:>6}      ~{int(stats['LUT']/CAL):>6}")
    print(f"FFs  : {stats['FF']:>6}       {stats['FF']:>6}")
    print(f"DSPs : {stats['DSP']:>6}       {stats['DSP']:>6}")
    print(f"BRAMs: {stats['BRAM']:>6}       {stats['BRAM']:>6}")
    print(f"\n(Vivado estimate uses /{CAL} calibration. Run with --vivado for exact numbers.)")


if __name__ == "__main__":
    main()
