"""Automated functional verification using Python golden model + iverilog simulation."""

import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path

from rich.console import Console

from src.golden_model import (
    ct_butterfly,
    generate_test_vectors,
    mod_mul,
    parse_module_params,
)
from src.code_generator import _load_iverilog_config

console = Console()
IVERILOG_BIN, IVERILOG_FLAGS = None, None


def _get_iv() -> tuple[str, str]:
    global IVERILOG_BIN, IVERILOG_FLAGS
    if IVERILOG_BIN is None:
        IVERILOG_BIN, IVERILOG_FLAGS = _load_iverilog_config()
    return IVERILOG_BIN, IVERILOG_FLAGS


# ---------------------------------------------------------------------------
# Port scanner
# ---------------------------------------------------------------------------

def _scan_ports(verilog_files: list[str], module_name: str) -> dict:
    """Scan module port list. Returns {has_clk, has_rst, inputs, outputs}."""
    import re
    for fpath in verilog_files:
        src = Path(fpath).read_text(encoding="utf-8", errors="ignore")
        m = re.search(
            rf'module\s+{module_name}\s*(?:#\([^)]*\))?\s*\(([^;]+)\)',
            src, re.DOTALL,
        )
        if not m:
            continue
        ports_block = m.group(1)
        inputs, outputs = [], []
        has_clk, has_rst = False, False
        for line in ports_block.split("\n"):
            m2 = re.match(
                r'(input|output)\s+(?:wire|reg)?\s*(?:\[[\w:*-]+\])?\s*(\w+)',
                line.strip(),
            )
            if not m2:
                continue
            direction, name = m2.group(1), m2.group(2)
            if "clk" in name.lower():
                has_clk = True
            elif "rst" in name.lower():
                has_rst = True
            if direction == "input":
                inputs.append(name)
            else:
                outputs.append(name)
        return {"has_clk": has_clk, "has_rst": has_rst, "inputs": inputs, "outputs": outputs}
    return {"has_clk": True, "has_rst": True, "inputs": ["a", "b"], "outputs": ["r"]}


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
    out_r = ports["outputs"][0]
    clk_line = ".clk(clk)," if ports["has_clk"] else ""
    rst_line = ".rst_n(rst_n)," if ports["has_rst"] else ""
    clk_wait = f"repeat({latency + 1}) @(posedge clk);" if ports["has_clk"] else "#1;"

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
            clk = 0; rst_n = 1'b0;
            {in_a} = 0; {in_b} = 0;
            errors = 0;
            #15 rst_n = 1'b1;
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
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

    if result.returncode != 0:
        _cleanup(tb_path, exe_path)
        return False, result.stderr.strip()

    vvp_bin = shutil.which("vvp") or str(Path(iv_bin).parent / "vvp")
    result = subprocess.run([vvp_bin, str(exe_path)], capture_output=True, text=True, timeout=30)
    _cleanup(tb_path, exe_path)

    output = result.stdout + result.stderr
    if "ALL" in output and "TESTS PASSED" in output:
        return True, ""
    # Collect FAIL lines
    fail_lines = [l.strip() for l in output.split("\n") if "FAIL" in l]
    return False, "\n".join(fail_lines)


def _cleanup(*paths: Path):
    for p in paths:
        try:
            p.unlink()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def verify_module(
    verilog_files: list[str],
    module_name: str,
    module_type: str = "modmul",
    hardware_spec_params: dict | None = None,
    num_vectors: int = 4,
    latency: int = 3,
) -> tuple[bool, str]:
    """
    Verify a generated module against the golden model.

    Returns (passed, failure_details).
    """
    import random

    params = parse_module_params(hardware_spec_params or {})
    dw = params["DATA_WIDTH"]
    q = params["Q"]
    k = params["K"]

    ports = _scan_ports(verilog_files, module_name)

    if module_type == "modmul":
        vectors = generate_test_vectors(num_vectors, q)
        for v in vectors:
            v["expected"] = mod_mul(v["a"], v["b"], q, k)
        tb = _gen_modmul_testbench(module_name, vectors, ports, latency, dw)
    elif module_type == "butterfly":
        rng = random.Random(42)
        vectors = []
        k_inv = pow(k, -1, q) if k is not None else 1
        for _ in range(num_vectors):
            a = rng.randint(0, q - 1)
            b = rng.randint(0, q - 1)
            w_raw = rng.randint(1, q - 1)
            w = (w_raw * k_inv) % q
            ea, eb = ct_butterfly(a, b, w, q, k)
            vectors.append({"a": a, "b": b, "w": w, "expected_a": ea, "expected_b": eb})
        tb = _gen_butterfly_testbench(module_name, vectors, ports, latency, dw)
    else:
        return False, f"Unknown module type: {module_type}"

    passed, fail_output = _run_simulation(verilog_files, tb, module_name)
    if passed:
        console.print(f"    [green]{module_name}: PASSED[/green]")
    else:
        console.print(f"    [red]{module_name}: FAILED[/red]")
        for line in fail_output.split("\n")[:5]:
            console.print(f"      [dim]{line.strip()}[/dim]")
    return passed, fail_output
