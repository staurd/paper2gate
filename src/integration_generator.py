"""Stage 3: Integrate paper's innovative modules into a complete NTT butterfly."""

from pathlib import Path

from jinja2 import Template
from rich.console import Console

from src.ir_models import GeneratedModule, PaperAnalysis
from src.llm_client import LLMClient
from src.code_generator import _run_iverilog, _fix_from_iverilog
from src.verilog_utils import extract_verilog, is_valid_module

PROMPT_DIR = Path(__file__).parent.parent / "prompts"
console = Console()


def generate_butterfly(
    analysis: PaperAnalysis,
    client: LLMClient,
    butterfly_type: str = "auto",
    check_with_iverilog: bool = True,
) -> GeneratedModule:
    """
    Generate a complete NTT butterfly top-level module that integrates
    the paper's innovative sub-module(s) with standard components.

    Optionally runs iverilog syntax check + fix loop.
    """
    source = (PROMPT_DIR / "integrate_butterfly.txt").read_text(encoding="utf-8")
    template = Template(source)

    type_hint = ""
    if butterfly_type == "CT":
        type_hint = "Use Cooley-Tukey architecture (multiply first, then add/subtract)."
    elif butterfly_type == "GS":
        type_hint = "Use Gentleman-Sande architecture (add/subtract first, then multiply)."

    user_prompt = template.render(
        innovations=analysis.innovations,
        type_hint=type_hint,
    )

    system_prompt = (
        "You are a senior RTL design engineer specializing in PQC hardware. "
        "Generate a complete NTT butterfly that integrates innovative sub-modules "
        "with standard components. "
        "Respond ONLY with the complete Verilog inside a markdown code fence. "
        "No explanations, no extra text."
    )

    raw = client.generate_structured(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        stage="generate",
    )

    verilog = extract_verilog(raw)

    if not is_valid_module(verilog):
        console.print("    [red]Integration produced incomplete output[/red]")

    if check_with_iverilog:
        passed, errors = _run_iverilog(verilog)
        if not passed:
            console.print("    [red]Butterfly failed iverilog check[/red]")
            for line in errors.strip().split("\n")[:3]:
                console.print(f"      [dim]{line.strip()}[/dim]")
            # Use LLM to fix — create a dummy module spec for the fix function
            butterfly_spec = _dummy_module_spec(analysis)
            fixed_module = _fix_from_iverilog(
                butterfly_spec, verilog, errors, client,
            )
            passed2, _ = _run_iverilog(fixed_module.verilog_code)
            if passed2:
                console.print("    [green]Butterfly fixed and passed iverilog[/green]")
                return fixed_module

    return GeneratedModule(
        module_name="butterfly_top",
        verilog_code=verilog,
    )


def _dummy_module_spec(analysis: PaperAnalysis):
    """Create a minimal ModuleSpec from the paper analysis for fix prompts."""
    from src.ir_models import ModuleSpec, HardwareSpec
    return ModuleSpec(
        module_name="butterfly_top",
        category="butterfly",
        summary=f"Butterfly for {analysis.paper_title}",
        hardware_spec=HardwareSpec(),
    )


def save_butterfly(module: GeneratedModule, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{module.module_name}.v"
    out_path.write_text(module.verilog_code, encoding="utf-8")
    return out_path
