"""Stage 3: Integrate paper's innovative modules into a unified CT/GS butterfly,
compatible with KyberHPM1PE (acmert/kyber-polmul-hw) interface."""

import subprocess
import tempfile
from pathlib import Path

from src.config import get_iverilog_config
from src.console import console
from src.ir_models import GeneratedModule, PaperAnalysis
from src.llm_client import LLMClient
from src.prompt_manager import load_prompt
from src.verilog_utils import cleanup, extract_verilog, is_valid_module


def generate_butterfly(
    analysis: PaperAnalysis,
    modmul_path: Path | None,
    client: LLMClient,
    check_with_iverilog: bool = True,
    max_fix_rounds: int = 2,
) -> GeneratedModule:
    """
    Generate a unified CT/GS butterfly that integrates the paper's modmul.

    Output interface matches KyberHPM1PE:
      module butterfly(input clk, rst, CT, PWM, input [11:0] A,B,W,
                       output [11:0] E,O, MUL, ADD,SUB);
    """
    user_prompt = load_prompt("integrate_butterfly.jinja", innovations=analysis.innovations)

    system_prompt = (
        "You are a senior RTL design engineer specializing in PQC hardware. "
        "Generate a unified CT/GS butterfly that integrates the paper's innovative "
        "modular multiplier. Match the EXACT port interface specified. "
        "Respond ONLY with the complete Verilog inside a markdown code fence. "
        "No explanations, no extra text."
    )

    console.print("    Generating unified CT/GS butterfly...")
    raw = client.generate_structured(
        system_prompt=system_prompt, user_prompt=user_prompt, stage="generate",
    )

    verilog = extract_verilog(raw)
    module = GeneratedModule(module_name="butterfly", verilog_code=verilog)

    if not is_valid_module(verilog):
        console.print("    [red]Butterfly generation produced incomplete output[/red]")

    # --- Iverilog check (butterfly + modmul together) ---
    if check_with_iverilog:
        iv_files: list[str] = []
        if modmul_path and modmul_path.exists():
            iv_files.append(str(modmul_path))

        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".v", prefix="butterfly_",
            delete=False, encoding="utf-8",
        )
        tmp.write(module.verilog_code)
        tmp.close()
        iv_files.append(str(tmp.name))
        temp_paths = [Path(tmp.name)]

        iv_cfg = get_iverilog_config(); iv_bin = iv_cfg.get("binary", "iverilog"); iv_flags = iv_cfg.get("flags", "-g2012")
        for round_num in range(1, max_fix_rounds + 1):
            console.print(f"    Iverilog butterfly check "
                          f"({len(iv_files)} files, round {round_num}/{max_fix_rounds})...")
            exe = Path(tempfile.gettempdir()) / "paper2gate_bfly_check"
            cmd = [iv_bin] + iv_flags.split() + ["-o", str(exe)] + iv_files
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                console.print("    [green]Butterfly: iverilog PASSED[/green]")
                cleanup(*temp_paths, exe)
                break

            errors = result.stderr.strip() or result.stdout.strip()
            console.print("    [red]Butterfly: iverilog FAILED[/red]")
            for line in errors.split("\n")[:3]:
                console.print(f"      [dim]{line.strip()}[/dim]")

            if round_num < max_fix_rounds:
                console.print(f"    Fixing ({round_num}/{max_fix_rounds})...")
                fixed = _fix_butterfly(module.verilog_code, errors, client)
                module = fixed
                temp_paths[0].write_text(module.verilog_code, encoding="utf-8")
            else:
                console.print(f"    [red]Max fix rounds reached[/red]")
                cleanup(*temp_paths, exe)

        cleanup(*temp_paths, exe)

    return module


def _fix_butterfly(
    verilog_code: str, iverilog_errors: str, client: LLMClient,
) -> GeneratedModule:
    """Ask LLM to fix the butterfly based on iverilog errors, keeping the correct port interface."""
    user_prompt = load_prompt("fix_verilog.jinja",
        iverilog_errors=iverilog_errors, verilog_code=verilog_code)
    user_prompt += (
        "\n\nIMPORTANT: The module name MUST be 'butterfly' with ports: "
        "input clk, rst, CT, PWM, input [11:0] A, B, W, "
        "output [11:0] E, O, MUL, ADD, SUB. "
        "Reset is active-HIGH. Do NOT rename or reorder any ports."
    )

    system_prompt = (
        "You are a senior RTL design engineer. "
        "Fix all iverilog errors while preserving the exact module interface. "
        "Respond ONLY with the complete fixed Verilog inside a markdown code fence."
    )

    raw = client.generate_structured(system_prompt=system_prompt, user_prompt=user_prompt, stage="generate")
    fixed = extract_verilog(raw)

    if not is_valid_module(fixed):
        console.print("    [red]Butterfly fix produced invalid module[/red]")
        return GeneratedModule(module_name="butterfly", verilog_code=verilog_code)

    return GeneratedModule(module_name="butterfly", verilog_code=fixed)


def save_butterfly(module: GeneratedModule, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{module.module_name}.v"
    out_path.write_text(module.verilog_code, encoding="utf-8")
    return out_path
