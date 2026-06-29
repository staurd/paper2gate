"""Stage 2: Generate Verilog code from hardware module specifications,
with iverilog syntax checking and auto-fix loop."""

import functools
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Template
from rich.console import Console

from src.ir_models import GeneratedModule, ModuleSpec
from src.llm_client import LLMClient
from src.verilog_utils import extract_verilog, is_valid_module

PROMPT_DIR = Path(__file__).parent.parent / "prompts"

console = Console()


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
    }


def _render_prompt(name: str, ctx: dict[str, Any]) -> str:
    source = (PROMPT_DIR / name).read_text(encoding="utf-8")
    return Template(source).render(**ctx)


# ---------------------------------------------------------------------------
# Iverilog Integration
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _load_iverilog_config() -> tuple[str, str]:
    """Read iverilog settings from config.yaml. Cached — reads once."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    try:
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        iv_cfg = cfg.get("iverilog", {})
        return iv_cfg.get("binary", "iverilog"), iv_cfg.get("flags", "-g2012")
    except Exception:
        return "iverilog", "-g2012"


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
            Path(tmp.name).unlink()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Fix from iverilog errors
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _fix_prompt_template() -> str:
    return (PROMPT_DIR / "fix_verilog.txt").read_text(encoding="utf-8")


def _fix_from_iverilog(
    module_spec: ModuleSpec,
    verilog_code: str,
    iverilog_errors: str,
    client: LLMClient,
) -> GeneratedModule:
    """Ask LLM to fix the Verilog code based on iverilog error output."""
    template = _fix_prompt_template()
    user_prompt = template \
        .replace("{{iverilog_errors}}", iverilog_errors) \
        .replace("{{verilog_code}}", verilog_code)

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
) -> GeneratedModule:
    """Generate a Verilog module from a hardware spec. No auto-fix loop."""
    ctx = _spec_context(module_spec)
    user_prompt = _render_prompt("generate_verilog.txt", ctx)

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
) -> tuple[GeneratedModule, list[str]]:
    """
    Generate Verilog, then run an iverilog-check + LLM-fix loop.

    Returns (final_module, error_logs).
    """
    console.print(f"    Generating [cyan]{module_spec.module_name}[/cyan]...")
    module = generate_verilog(module_spec, client)
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
    import functools

    @functools.lru_cache(maxsize=1)
    def _load_fix_prompt():
        return (PROMPT_DIR / "fix_from_verification.txt").read_text(encoding="utf-8")

    user_prompt = _load_fix_prompt() \
        .replace("{{failure_details}}", failure_details) \
        .replace("{{verilog_code}}", verilog_code)

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
