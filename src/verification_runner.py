"""Automated functional verification using Python golden model + iverilog simulation."""

import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path

from src.config import get_iverilog_config
from src.console import console
from src.golden_model import generate_test_vectors, mod_mul
from src.ir_models import parse_module_params
from src.verilog_utils import cleanup, scan_ports


def _get_iv() -> tuple[str, str]:
    iv_cfg = get_iverilog_config()
    return iv_cfg.get("binary", "iverilog"), iv_cfg.get("flags", "-g2012")


# ---------------------------------------------------------------------------
# Testbench generation
# ---------------------------------------------------------------------------

def _gen_modmul_testbench(
    module_name: str,
    vectors: list[dict],
    ports: dict,
    latency: int,
    dw: int,
) -> str:
    data_in = [p for p in ports["inputs"] if "clk" not in p.lower() and "rst" not in p.lower()]
    if len(data_in) < 2:
        raise ValueError(f"Module {module_name} has only {len(data_in)} data inputs, need 2 for modmul verification")
    in_a, in_b = data_in[0], data_in[1]
    if not ports["outputs"]:
        raise ValueError(f"Module {module_name} has no output port")
    out_r = ports["outputs"][0]

    # Find actual clock/reset port names (handle both "clk"/"rst" and "clk"/"rst_n")
    clk_port = next((p for p in ports["inputs"] if "clk" in p.lower()), "clk")
    rst_port = next((p for p in ports["inputs"] if "rst" in p.lower()), "rst_n")
    rst_active_high = not rst_port.endswith("_n")  # "rst" = active-high, "rst_n" = active-low

    clk_line = f".{clk_port}(clk)," if ports["has_clk"] else ""
    rst_line = f".{rst_port}(rst_n)," if ports["has_rst"] else ""
    clk_wait = f"repeat({max(1, latency)}) @(posedge clk); #1;" if ports["has_clk"] else "#1;"

    # Reset polarity: drive rst_n low for active-low, high for active-high
    rst_init = "1'b1" if rst_active_high else "1'b0"
    rst_release = "1'b0" if rst_active_high else "1'b1"

    tests = []
    for i, v in enumerate(vectors):
        expected = v["expected"]
        tests.append(f"        // Test {i}: a={v['a']}, b={v['b']} -> {expected}")
        tests.append(f"        {in_a} = {dw}'d{v['a']}; {in_b} = {dw}'d{v['b']};")
        tests.append(f"        {clk_wait}")
        tests.append(f"        if ({out_r} !== {dw}'d{expected}) begin")
        tests.append(f'            $display("FAIL[{i}]: a=%d b=%d got=%d expected={expected}", '
                     f'{in_a}, {in_b}, {out_r});')
        tests.append(f"            errors = errors + 1;")
        tests.append(f"        end else begin")
        tests.append(f'            $display("PASS[{i}]");')
        tests.append(f"        end")

    test_body = "\n".join(tests)
    dwm1 = dw - 1

    return textwrap.dedent(f"""\
    `timescale 1ns / 1ps
    module tb_verify;
        reg clk, rst_n;
        reg [{dwm1}:0] {in_a}, {in_b};
        wire [{dwm1}:0] {out_r};
        integer errors;

        {module_name} dut (
            {clk_line}
            {rst_line}
            .{in_a}({in_a}),
            .{in_b}({in_b}),
            .{out_r}({out_r})
        );

        always #5 clk = ~clk;

        initial begin
            clk = 0; rst_n = {rst_init};
            {in_a} = 0; {in_b} = 0;
            errors = 0;
            #15 rst_n = {rst_release};
            @(posedge clk);

    {test_body}

            if (errors == 0)
                $display("\\n=== ALL %0d TESTS PASSED ===", {len(vectors)});
            else
                $display("\\n=== %0d/%0d TESTS FAILED ===", errors, {len(vectors)});
            $finish;
        end
    endmodule
    """)



# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------

def _run_simulation(
    verilog_files: list[str],
    testbench_code: str,
    label: str,
) -> tuple[bool, str]:
    """Compile and run a testbench. Returns (passed, failure_output)."""
    iv_bin, iv_flags = _get_iv()
    tmp_dir = Path(tempfile.gettempdir())

    tb_path = tmp_dir / "paper2gate_tb.v"
    tb_path.write_text(testbench_code, encoding="utf-8")
    exe_path = tmp_dir / "paper2gate_sim"

    cmd = [iv_bin] + iv_flags.split() + ["-o", str(exe_path)] + verilog_files + [str(tb_path)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except FileNotFoundError as exc:
        cleanup(tb_path, exe_path)
        raise OSError(f"iverilog not found: {iv_bin}") from exc
    except subprocess.TimeoutExpired as exc:
        cleanup(tb_path, exe_path)
        raise OSError("iverilog compilation timed out") from exc

    if result.returncode != 0:
        cleanup(tb_path, exe_path)
        return False, result.stderr.strip()

    vvp_bin = shutil.which("vvp")
    if not vvp_bin:
        candidates = [Path(iv_bin).parent / "vvp", Path(iv_bin).parent / "vvp.exe"]
        vvp_bin = next((str(path) for path in candidates if path.exists()), str(candidates[0]))
    try:
        result = subprocess.run([vvp_bin, str(exe_path)], capture_output=True, text=True, timeout=30)
    except FileNotFoundError as exc:
        cleanup(tb_path, exe_path)
        raise OSError(f"vvp not found: {vvp_bin}") from exc
    except subprocess.TimeoutExpired as exc:
        cleanup(tb_path, exe_path)
        raise OSError("vvp simulation timed out") from exc
    cleanup(tb_path, exe_path)

    output = result.stdout + result.stderr
    if "ALL" in output and "TESTS PASSED" in output:
        return True, ""
    fail_lines = [l.strip() for l in output.split("\n") if "FAIL" in l]
    return False, "\n".join(fail_lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def verify_module(
    verilog_files: list[str],
    module_name: str,
    hardware_spec_params: dict | None = None,
    num_vectors: int = 4,
    latency: int = 3,
    k_factor: int | None = None,
) -> tuple[bool, str]:
    """
    Verify a generated module against the golden model.

    Args:
        k_factor: Correction factor the module's output carries relative to a
                  true a*b mod q. For K-reduction designs the module computes
                  k*a*b mod q (Kyber: k=13). When given, this overrides any K
                  parsed from hardware_spec_params. Pass 1 for plain a*b mod q.

    Returns (passed, failure_details).
    """
    params = parse_module_params(hardware_spec_params or {})
    dw = params["DATA_WIDTH"]
    q = params["Q"]
    k = k_factor if k_factor is not None else params["K"]

    ports = scan_ports(verilog_files, module_name)

    vectors = generate_test_vectors(num_vectors, q)
    for v in vectors:
        v["expected"] = mod_mul(v["a"], v["b"], q, k)
    tb = _gen_modmul_testbench(module_name, vectors, ports, latency, dw)

    passed, fail_output = _run_simulation(verilog_files, tb, module_name)
    if passed:
        console.print(f"    [green]{module_name}: PASSED[/green]")
    else:
        console.print(f"    [red]{module_name}: FAILED[/red]")
        for line in fail_output.split("\n")[:5]:
            console.print(f"      [dim]{line.strip()}[/dim]")
    return passed, fail_output
