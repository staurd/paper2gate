"""Stage 2: Generate Verilog code from hardware module specifications,
with iverilog syntax checking and auto-fix loop."""

import subprocess
import tempfile
from pathlib import Path
from typing import Any

from src.config import get_iverilog_config
from src.console import console
from src.ir_models import GeneratedModule, ModuleSpec
from src.llm_client import LLMClient
from src.prompt_manager import load_prompt
from src.schemes import KYBER, SchemeProfile, render_vars
from src.verilog_utils import cleanup, extract_verilog, is_valid_module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _spec_context(module_spec: ModuleSpec) -> dict[str, Any]:
    hw = module_spec.hardware_spec
    return {
        "module_name": module_spec.module_name,
        "category": module_spec.category,
        "summary": module_spec.summary,
        "parameters": hw.parameters,
        "ports": {
            "input": [p.model_dump() for p in hw.ports.get("input", [])],
            "output": [p.model_dump() for p in hw.ports.get("output", [])],
        },
        "behavior": hw.behavior,
        "timing": hw.timing,
        "constraints": hw.constraints,
        "correction_factor": hw.correction_factor,
        "latency_cycles": hw.latency_cycles,
    }


# ---------------------------------------------------------------------------
# Iverilog Integration
# ---------------------------------------------------------------------------
def _load_iverilog_config() -> tuple[str, str]:
    iv_cfg = get_iverilog_config()
    return iv_cfg.get("binary", "iverilog"), iv_cfg.get("flags", "-g2012")


def _run_iverilog(verilog_code: str) -> tuple[bool, str]:
    """
    Check Verilog syntax with iverilog.

    Returns (passed, error_output).
    """
    iverilog_bin, iverilog_flags = _load_iverilog_config()

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".v", prefix="paper2gate_",
        delete=False, encoding="utf-8",
    )
    try:
        tmp.write(verilog_code)
        tmp.close()

        cmd = [iverilog_bin] + iverilog_flags.split() + [tmp.name]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            return True, ""
        errors = result.stderr.strip()
        if not errors and result.stdout.strip():
            errors = result.stdout.strip()
        return False, errors
    except FileNotFoundError:
        return False, f"WARNING: iverilog not found at '{iverilog_bin}'. " \
                       "Install Icarus Verilog or set 'iverilog.binary' in config.yaml."
    except subprocess.TimeoutExpired:
        return False, "WARNING: iverilog timed out."
    finally:
        try:
            cleanup(Path(tmp.name))
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Fix from iverilog errors
# ---------------------------------------------------------------------------

def _fix_from_iverilog(
    module_spec: ModuleSpec,
    verilog_code: str,
    iverilog_errors: str,
    client: LLMClient,
) -> GeneratedModule:
    """Ask LLM to fix the Verilog code based on iverilog error output."""
    user_prompt = load_prompt("fix_verilog.jinja",
        iverilog_errors=iverilog_errors, verilog_code=verilog_code)

    system_prompt = (
        "You are a senior RTL design engineer. "
        "Fix all iverilog errors in the Verilog code. "
        "Respond ONLY with the complete fixed Verilog inside a markdown code fence."
    )

    raw = client.generate_structured(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        stage="generate",
    )
    fixed = extract_verilog(raw)

    if not is_valid_module(fixed):
        console.print("    [red]Fix produced invalid module — keeping original[/red]")
        return GeneratedModule(
            module_name=module_spec.module_name,
            verilog_code=verilog_code,
        )

    return GeneratedModule(
        module_name=module_spec.module_name,
        verilog_code=fixed,
    )


# ---------------------------------------------------------------------------
# Generation (no iverilog check)
# ---------------------------------------------------------------------------

def generate_verilog(
    module_spec: ModuleSpec,
    client: LLMClient,
    max_retries: int = 2,
    target_interface: str | None = None,
    scheme: SchemeProfile | None = None,
) -> GeneratedModule:
    """Generate a Verilog module from a hardware spec. No auto-fix loop.

    Args:
        target_interface: If "modred", use the modred-specialized prompt
                          that enforces the fixed modred interface.
        scheme: PQC scheme profile. Renders the fixed-interface constants
                (operand widths, modulus) into the modmul prompt. Defaults
                to KYBER for backward compatibility with direct callers.
    """
    ctx = _spec_context(module_spec)
    if target_interface == "modmul":
        prompt_name = "generate_modmul.jinja"
        ctx.update(render_vars(scheme or KYBER))
    else:
        prompt_name = "generate_verilog.jinja"
    user_prompt = load_prompt(prompt_name, **ctx)

    base_system = (
        "You are a senior RTL design engineer. "
        "Generate a COMPLETE Verilog module (from 'module' to 'endmodule'). "
        "Respond ONLY with the Verilog code inside a markdown code fence. "
        "No explanations, no extra text."
    )

    system_prompt = base_system
    verilog = ""
    for attempt in range(1 + max_retries):
        raw = client.generate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            stage="generate",
        )
        verilog = extract_verilog(raw)

        if is_valid_module(verilog):
            return GeneratedModule(
                module_name=module_spec.module_name,
                verilog_code=verilog,
            )

        if attempt < max_retries:
            console.print(
                f"    [yellow]Missing 'module'/'endmodule' — retrying "
                f"({attempt + 1}/{max_retries})...[/yellow]"
            )
            system_prompt = (
                base_system +
                " CRITICAL: Your previous output lacked 'module'/'endmodule'. "
                "The first non-comment line MUST be 'module {name}'. "
                "The last line MUST be 'endmodule'.".format(
                    name=module_spec.module_name
                )
            )
        else:
            console.print(
                f"    [red]Still invalid after {max_retries} retries "
                f"— saving as-is[/red]"
            )

    return GeneratedModule(
        module_name=module_spec.module_name,
        verilog_code=verilog,
    )


# ---------------------------------------------------------------------------
# Iverilog-based auto-fix loop
# ---------------------------------------------------------------------------

def generate_with_check(
    module_spec: ModuleSpec,
    client: LLMClient,
    max_rounds: int = 3,
    target_interface: str | None = None,
    scheme: SchemeProfile | None = None,
) -> tuple[GeneratedModule, list[str]]:
    """
    Generate Verilog, then run an iverilog-check + LLM-fix loop.

    Returns (final_module, error_logs).
    """
    console.print(f"    Generating [cyan]{module_spec.module_name}[/cyan]...")
    module = generate_verilog(module_spec, client, target_interface=target_interface, scheme=scheme)
    error_logs: list[str] = []

    for round_num in range(1, max_rounds + 1):
        console.print(f"    Iverilog check (round {round_num}/{max_rounds})...")
        passed, errors = _run_iverilog(module.verilog_code)

        if passed and not errors:
            console.print("    [green]Iverilog: PASSED[/green]")
            break

        if not passed and errors.startswith("WARNING:"):
            console.print(f"    [yellow]{errors}[/yellow]")
            break

        console.print("    [red]Iverilog: FAILED[/red]")
        error_logs.append(errors)

        for line in errors.strip().split("\n")[:3]:
            console.print(f"      [dim]{line.strip()}[/dim]")

        if round_num < max_rounds:
            console.print(f"    Fixing ({round_num}/{max_rounds})...")
            module = _fix_from_iverilog(
                module_spec, module.verilog_code, errors, client,
            )
        else:
            console.print(
                f"    [red]Max rounds ({max_rounds}) reached "
                f"— keeping current version[/red]"
            )

    return module, error_logs


# ---------------------------------------------------------------------------
# File output
# ---------------------------------------------------------------------------

def save_verilog(module: GeneratedModule, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{module.module_name}.v"
    out_path.write_text(module.verilog_code, encoding="utf-8")
    return out_path


def fix_from_verification(
    module_spec: ModuleSpec,
    verilog_code: str,
    failure_details: str,
    client: LLMClient,
) -> GeneratedModule:
    """Ask LLM to fix Verilog code after golden model verification failures."""
    user_prompt = load_prompt("fix_from_verification.jinja",
        failure_details=failure_details, verilog_code=verilog_code,
        summary=module_spec.summary,
        behavior=module_spec.hardware_spec.behavior)

    system_prompt = (
        "You are a senior RTL design engineer. "
        "Fix the functional bugs in the Verilog code based on test failures. "
        "Respond ONLY with the complete fixed Verilog inside a markdown code fence."
    )

    raw = client.generate_structured(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        stage="generate",
    )

    fixed = extract_verilog(raw)

    if not is_valid_module(fixed):
        console.print("    [red]Verification fix produced invalid module[/red]")
        return GeneratedModule(
            module_name=module_spec.module_name,
            verilog_code=verilog_code,
        )

    return GeneratedModule(
        module_name=module_spec.module_name,
        verilog_code=fixed,
    )


def save_error_logs(error_logs: list[str], module_name: str, output_dir: Path) -> Path:
    """Save iverilog error logs to file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{module_name}_iverilog_log.txt"
    content = "\n\n".join(
        f"=== Round {i+1} ===\n{err}" for i, err in enumerate(error_logs)
    ) if error_logs else "All rounds passed."
    out_path.write_text(content, encoding="utf-8")
    return out_path
