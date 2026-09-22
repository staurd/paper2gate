"""Stage 2: Generate Verilog code from hardware module specifications,
with iverilog syntax checking and auto-fix loop."""

import subprocess
import tempfile
import re
import shutil
from pathlib import Path
from typing import Any

from src.config import get_iverilog_config
from src.console import console
from src.ir_models import GeneratedModule, ModuleSpec
from src.llm_client import LLMClient
from src.prompt_manager import load_prompt
from src.schemes import KYBER, SchemeProfile, render_vars
from src.verilog_utils import cleanup, extract_verilog, is_valid_module, validate_modmul_interface


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


def _validate_generated_code(
    verilog: str,
    scheme: SchemeProfile | None,
) -> list[str]:
    if not is_valid_module(verilog):
        return ["missing 'module'/'endmodule'"]
    return validate_modmul_interface(verilog, (scheme or KYBER).data_width)


# ---------------------------------------------------------------------------
# Iverilog checks
# ---------------------------------------------------------------------------
def _load_iverilog_config() -> tuple[str, str]:
    iv_cfg = get_iverilog_config()
    return iv_cfg.get("binary", "iverilog"), iv_cfg.get("flags", "-g2012")


def iverilog_available() -> bool:
    """Return whether the configured Icarus executable can be launched."""
    binary, _ = _load_iverilog_config()
    return Path(binary).exists() or shutil.which(binary) is not None


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


def check_verilog(verilog_code: str) -> tuple[bool, str]:
    """Public syntax-check wrapper used by the pipeline and smoke checks."""
    return _run_iverilog(verilog_code)


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
        operation="fix_syntax",
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

def generate_modmul(
    module_spec: ModuleSpec,
    client: LLMClient,
    max_retries: int = 2,
    scheme: SchemeProfile | None = None,
    target_part: str | None = None,
) -> GeneratedModule:
    """Generate the fixed-interface modmul from a hardware spec.

    Args:
        scheme: PQC scheme profile. Renders the fixed-interface constants
                (operand widths, modulus) into the modmul prompt. Defaults
                to KYBER for backward compatibility with direct callers.
        target_part: Legal Vivado part to target during RTL generation.
    """
    ctx = _spec_context(module_spec)
    prompt_name = "generate_modmul.jinja"
    ctx.update(render_vars(scheme or KYBER))
    ctx["target_part"] = target_part or ""
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
            operation="generate_modmul",
        )
        verilog = extract_verilog(raw)

        validation_errors = _validate_generated_code(verilog, scheme)
        if not validation_errors:
            return GeneratedModule(
                module_name=module_spec.module_name,
                verilog_code=verilog,
            )

        if attempt < max_retries:
            console.print(
                f"    [yellow]Generated Verilog failed validation "
                f"({attempt + 1}/{max_retries})...[/yellow]"
            )
            system_prompt = (
                base_system +
                " CRITICAL: your previous output failed validation: "
                f"{'; '.join(validation_errors)}. Return a complete valid module."
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
    scheme: SchemeProfile | None = None,
    target_part: str | None = None,
) -> tuple[GeneratedModule, list[str]]:
    """
    Generate Verilog, then run an iverilog-check + LLM-fix loop.

    Returns (final_module, error_logs).
    """
    console.print(f"    Generating [cyan]{module_spec.module_name}[/cyan]...")
    module = generate_modmul(
        module_spec,
        client,
        scheme=scheme,
        target_part=target_part,
    )
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
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]*", module.module_name):
        raise ValueError(f"Unsafe or invalid Verilog module name: {module.module_name!r}")
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
        operation="fix_function",
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
