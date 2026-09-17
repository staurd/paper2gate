"""Tests for scheme profiles, detection, prompt rendering, latency parsing."""

import sys

sys.path.insert(0, ".")

from src.schemes import (
    DILITHIUM,
    KYBER,
    detect_scheme,
    render_vars,
    resolve_scheme,
)
from src.innovation_extractor import _parse_analysis
from src.golden_model import mod_mul
from src.prompt_manager import load_prompt

# --- scheme detection -------------------------------------------------------

assert detect_scheme("q = 8380417 mod") is DILITHIUM
assert detect_scheme("q = 3329") is KYBER
assert detect_scheme("modulus 8,380,417") is DILITHIUM          # comma format
# 8380417 wins even when the paper also cites Kyber's q (verified real case:
# "A Hard Crystal" contains "3329" once, right after "8380417")
assert detect_scheme("8380417 appears... kyber q=3329 too") is DILITHIUM
assert detect_scheme("Dilithium implementation paper") is DILITHIUM
assert detect_scheme("A lattice-based crypto paper") is KYBER    # fallback
assert resolve_scheme("dilithium", "x") is DILITHIUM
assert resolve_scheme(None, "kyber text") is KYBER
assert resolve_scheme("auto", "kyber text") is KYBER

# --- render_vars ------------------------------------------------------------

v = render_vars(KYBER)
assert v["dw"] == 12 and v["dw_minus_1"] == 11 and v["dw_plus_1"] == 13
assert v["pw"] == 24 and v["pw_minus_1"] == 23
assert v["q"] == 3329 and v["q_minus_1"] == 3328 and v["q_minus_1_sq"] == 11075584

v = render_vars(DILITHIUM)
assert v["dw"] == 23 and v["dw_minus_1"] == 22 and v["dw_plus_1"] == 24
assert v["pw"] == 46 and v["pw_minus_1"] == 45
assert v["q"] == 8380417 and v["q_minus_1"] == 8380416

# --- prompt rendering -------------------------------------------------------

def _render_modmul(profile):
    return load_prompt(
        "generate_modmul.jinja",
        module_name="modmul",
        category="modular_arithmetic",
        summary="s",
        parameters={},
        ports={"input": [], "output": []},
        behavior="b",
        timing="t",
        constraints="c",
        latency_cycles=0,
        correction_factor=1,
        **render_vars(profile),
    )

ky = _render_modmul(KYBER)
assert "[11:0]" in ky and "3329" in ky and "13'd3329" in ky and "11075584" in ky

di = _render_modmul(DILITHIUM)
assert "[11:0]" not in di and "3329" not in di
assert "[22:0]" in di and "[45:0]" in di and "8380417" in di
assert "23'd8380417" in di and "8380416" in di

ex_ky = load_prompt(
    "extract_innovation.jinja",
    paper_text="",
    target_interface="modmul",
    **render_vars(KYBER),
)
assert "[11:0]" in ex_ky and "3329" in ex_ky

ex_di = load_prompt(
    "extract_innovation.jinja",
    paper_text="",
    target_interface="modmul",
    **render_vars(DILITHIUM),
)
assert "[11:0]" not in ex_di and "3329" not in ex_di
assert "[22:0]" in ex_di and "8380417" in ex_di and "latency_cycles" in ex_di

# --- latency parsing ---------------------------------------------------------

a = _parse_analysis({"paper_title": "t", "innovations": [
    {"module_name": "modmul", "category": "modular_arithmetic",
     "hardware_spec": {"latency_cycles": 5}}]})
assert a.innovations[0].hardware_spec.latency_cycles == 5

b = _parse_analysis({"paper_title": "t", "innovations": [
    {"module_name": "m", "hardware_spec": {}}]})
assert b.innovations[0].hardware_spec.latency_cycles == 0

# --- golden model with q = 8380417 -------------------------------------------

assert mod_mul(8380416, 8380416, 8380417) == 1            # (-1)(-1) mod q
assert mod_mul(1234567, 7654321, 8380417) == (1234567 * 7654321) % 8380417

# --- reference trimming ------------------------------------------------------

from src.innovation_extractor import _trim_references

# A trailing bibliography (12-21% of a real paper) is dropped.
body = "A" * 8000 + "\nReferences\n" + "B" * 1500
out, dropped = _trim_references(body)
assert "References" not in out and "B" not in out
assert out == "A" * 8000 + "\n" and dropped == 1511

# A "References" heading early in the document is not the bibliography
# (survey papers cite per section) — never trim on it.
early = "References\n" + "A" * 20000
assert _trim_references(early) == (early, 0)

# A heading near the end with almost nothing after it isn't worth trimming.
stub = "A" * 8000 + "\nReferences\n" + "B" * 100
assert _trim_references(stub) == (stub, 0)

# No heading at all (short papers, or PDF extraction that lost it).
plain = "A" * 5000
assert _trim_references(plain) == (plain, 0)

# Real papers: every one in test_papers has a trimmable bibliography.
import glob as _glob
import os as _os

_seen = set()
for f in _glob.glob("outputs/*/extracted_text/full_text.txt"):
    key = _os.path.basename(_os.path.dirname(_os.path.dirname(f))).split("_20")[0]
    if key in _seen:
        continue
    _seen.add(key)
    text = open(f, encoding="utf-8", errors="replace").read()
    trimmed, n = _trim_references(text)
    assert n == 0 or n > 1000
    assert len(trimmed) <= len(text)

print("All scheme tests passed.")
