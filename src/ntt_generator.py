"""Stage 4: Generate complete NTT core with standard components,
integrating the paper's innovative butterfly."""

import math
import re
import tempfile
from pathlib import Path

from rich.console import Console

from src.ir_models import GeneratedModule, PaperAnalysis
from src.llm_client import LLMClient
from src.code_generator import _load_iverilog_config, _run_iverilog, _fix_from_iverilog
from src.verification_runner import _scan_ports
from src.verilog_utils import extract_verilog, is_valid_module

PROMPT_DIR = Path(__file__).parent.parent / "prompts"
console = Console()

# Kyber defaults (used when paper doesn't specify)
DEFAULT_Q = 3329
DEFAULT_OMEGA = 17
DEFAULT_N = 256


def _compute_twiddle_values(omega: int, q: int, n: int = 256) -> list[int]:
    """Compute omega^i mod q for i=0..(n/2 - 1)."""
    values = []
    val = 1
    for i in range(n // 2):
        values.append(val)
        val = (val * omega) % q
    return values


def _format_twiddle_case(values: list[int], dw: int) -> str:
    lines = []
    for i, v in enumerate(values):
        lines.append(f"        7'd{i}: data = {dw}'d{v};")
    lines.append("        default: data = {dw}'d0;")
    return "\n".join(lines)


def generate_ntt(
    analysis: PaperAnalysis,
    butterfly_path: Path | None,
    client: LLMClient,
    check_with_iverilog: bool = True,
    max_fix_rounds: int = 2,
) -> GeneratedModule:
    """
    Generate a complete NTT core that integrates the paper's butterfly.

    Uses parameters from the paper's hardware_spec (Q, DATA_WIDTH, etc.)
    and scans the actual butterfly .v file for port names.
    """
    # --- Extract NTT parameters from the paper's innovations ---
    q = DEFAULT_Q
    omega = DEFAULT_OMEGA
    n = DEFAULT_N
    dw = 12
    log_n = 8  # log2(N)

    if analysis.innovations:
        hw = analysis.innovations[0].hardware_spec
        for key, val in hw.parameters.items():
            key_upper = key.upper()
            # Extract numeric value from description string
            m = re.search(r'\d+', str(val))
            v = int(m.group(0)) if m else None
            if v is None:
                continue
            if key_upper in ("Q", "MODULUS", "MODULUS_Q", "MOD"):
                q = v
                # Infer omega: for Kyber N=256, omega=17. For Dilithium N=256, omega=1753.
            elif key_upper in ("DATA_WIDTH", "WIDTH", "DW"):
                dw = v

    log_n = int(math.log2(n))

    # --- Compute twiddle values ---
    twiddle = _compute_twiddle_values(omega, q, n)
    twiddle_case = _format_twiddle_case(twiddle, dw)

    # --- Scan butterfly ports from actual generated file ---
    butterfly_ports = None
    if butterfly_path and butterfly_path.exists():
        butterfly_ports = _scan_ports([str(butterfly_path)], "butterfly_top")
        console.print(f"    Scanned butterfly ports: {butterfly_ports}")

    # --- Build butterfly port mapping for prompt ---
    bfly_port_text = _format_butterfly_port_text(butterfly_ports)

    # --- Extract butterfly latency from hardware spec ---
    bfly_timing = "Unknown latency. Use a shift register with at least 5 pipeline stages."
    if analysis.innovations:
        timing = analysis.innovations[0].hardware_spec.timing
        if timing:
            bfly_timing = f"From paper spec: {timing}"

    # --- Render prompt ---
    source = (PROMPT_DIR / "generate_ntt.txt").read_text(encoding="utf-8")
    user_prompt = source \
        .replace("{{twiddle_values}}", twiddle_case) \
        .replace("{{q}}", str(q)) \
        .replace("{{omega}}", str(omega)) \
        .replace("{{n}}", str(n)) \
        .replace("{{log_n}}", str(log_n)) \
        .replace("{{dw}}", str(dw)) \
        .replace("{{dw_minus_1}}", str(dw - 1)) \
        .replace("{{butterfly_ports}}", bfly_port_text) \
        .replace("{{butterfly_timing}}", bfly_timing)

    system_prompt = (
        "You are a senior RTL design engineer specializing in PQC hardware. "
        "Generate a complete NTT core with all sub-modules. "
        "Respond ONLY with the complete Verilog inside a markdown code fence. "
        "No explanations, no extra text."
    )

    console.print(f"    Generating NTT core (q={q}, N={n})...")
    raw = client.generate_structured(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        stage="generate",
    )

    verilog = extract_verilog(raw)
    module = GeneratedModule(module_name="ntt_core", verilog_code=verilog)

    if not is_valid_module(verilog):
        console.print("    [red]NTT generation produced incomplete output[/red]")

    # --- Iverilog check (includes butterfly .v file) ---
    if check_with_iverilog:
        # Collect files for iverilog: butterfly (if exists) + ntt_core temp file
        iverilog_files: list[str] = []
        if butterfly_path and butterfly_path.exists():
            iverilog_files.append(str(butterfly_path))
        # Write ntt_core to temp file
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".v", prefix="ntt_core_",
            delete=False, encoding="utf-8",
        )
        tmp.write(module.verilog_code)
        tmp.close()
        iverilog_files.append(str(tmp.name))
        # Track temp files only (never delete butterfly_path!)
        temp_files = [Path(tmp.name)]

        for round_num in range(1, max_fix_rounds + 1):
            console.print(
                f"    Iverilog NTT check ({len(iverilog_files)} files, "
                f"round {round_num}/{max_fix_rounds})..."
            )
            import subprocess
            iv_bin, iv_flags = _load_iverilog_config()
            exe = Path(tempfile.gettempdir()) / "paper2gate_ntt_check"
            cmd = [iv_bin] + iv_flags.split() + ["-o", str(exe)] + iverilog_files
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                console.print("    [green]NTT: iverilog PASSED[/green]")
                _cleanup(*temp_files, exe)
                break
            errors = result.stderr.strip()
            if not errors:
                errors = result.stdout.strip()
            console.print("    [red]NTT: iverilog FAILED[/red]")
            for line in errors.split("\n")[:3]:
                console.print(f"      [dim]{line.strip()}[/dim]")

            if round_num < max_fix_rounds:
                console.print(f"    Fixing NTT ({round_num}/{max_fix_rounds})...")
                fixed = _fix_ntt_from_errors(
                    module.verilog_code, errors, bfly_port_text, bfly_timing, client
                )
                module = fixed
                # Update temp file with fixed code
                tmp_path = temp_files[0]
                tmp_path.write_text(module.verilog_code, encoding="utf-8")
            else:
                console.print(f"    [red]Max NTT fix rounds reached[/red]")
                _cleanup(*temp_files, exe)

        # Clean up temp files on exit
        try:
            _cleanup(*temp_files, exe)
        except Exception:
            pass

    return module


def _format_butterfly_port_text(ports: dict | None) -> str:
    """Format butterfly port info for the prompt. If ports unknown, use default."""
    if not ports:
        return """\
butterfly_top u_bfly(
    .clk(clk), .rst_n(rst_n),
    .a_in(a), .b_in(b), .w(omega),
    .a_out(a_out), .b_out(b_out),
    .valid_out(bfly_valid)
);"""

    inputs = [p for p in ports["inputs"] if "clk" not in p.lower() and "rst" not in p.lower()]
    outputs = ports.get("outputs", [])
    clk_name = next((p for p in ports["inputs"] if "clk" in p.lower()), "clk")
    rst_name = next((p for p in ports["inputs"] if "rst" in p.lower()), "rst_n")

    lines = ["butterfly_top u_bfly("]
    lines.append(f"    .{clk_name}(clk),")
    if ports.get("has_rst"):
        lines.append(f"    .{rst_name}(rst_n),")
    # Map first 3 data inputs to a_in, b_in, w
    labels = ["a", "b", "omega"]
    for i, port in enumerate(inputs[:3]):
        lines.append(f"    .{port}({labels[i]}),")
    for port in outputs[:2]:
        lines.append(f"    .{port}({port}),")
    if len(outputs) > 2:
        lines.append(f"    .{outputs[2]}(bfly_valid)");
    else:
        lines.append(f"    .valid_out(bfly_valid)");  # will be connected if port exists

    # Remove trailing comma from last line
    lines[-1] = lines[-1].rstrip(",")
    lines.append(");")
    return "\n".join(lines)


def _fix_ntt_from_errors(
    verilog_code: str,
    iverilog_errors: str,
    butterfly_port_text: str,
    butterfly_timing: str,
    client: LLMClient,
) -> GeneratedModule:
    """Fix NTT core iverilog errors, with butterfly port info in the prompt."""
    import functools

    @functools.lru_cache(maxsize=1)
    def _load_fix_prompt():
        return (PROMPT_DIR / "fix_verilog.txt").read_text(encoding="utf-8")

    prompt = _load_fix_prompt()
    user_prompt = prompt \
        .replace("{{iverilog_errors}}", iverilog_errors) \
        .replace("{{verilog_code}}", verilog_code)
    # Append butterfly port info so LLM knows the correct port names
    user_prompt += (
        f"\n\n## IMPORTANT: Butterfly Port Names\n\n"
        f"The butterfly_top module has these exact ports:\n```verilog\n"
        f"{butterfly_port_text}\n```\n"
        f"Timing info: {butterfly_timing}\n"
        f"Use ONLY these port names when instantiating butterfly_top. "
        f"Do NOT declare a new butterfly_top module — it is provided externally."
    )

    system_prompt = (
        "You are a senior RTL design engineer. "
        "Fix all iverilog errors in the NTT Verilog code. "
        "Respond ONLY with the complete fixed Verilog inside a markdown code fence."
    )

    raw = client.generate_structured(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        stage="generate",
    )
    fixed = extract_verilog(raw)

    if not is_valid_module(fixed):
        console.print("    [red]NTT fix produced invalid module[/red]")
        return GeneratedModule(module_name="ntt_core", verilog_code=verilog_code)

    return GeneratedModule(module_name="ntt_core", verilog_code=fixed)


def _cleanup(*paths: Path | str):
    for p in paths:
        try:
            Path(p).unlink()
        except OSError:
            pass


def save_ntt(module: GeneratedModule, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "ntt_core.v"
    out_path.write_text(module.verilog_code, encoding="utf-8")
    return out_path
