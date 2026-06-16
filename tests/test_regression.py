"""Quick regression tests for critical functions."""
import re
import sys

# --- Test _extract_json ---
from src.llm_client import LLMClient

# Test 1: markdown fence anywhere in response
r1 = LLMClient._extract_json('Here is JSON:\n```json\n{"a": 1}\n```')
assert r1 == '{"a": 1}', f"FAIL test1: {r1!r}"

# Test 2: no fence, just JSON
r2 = LLMClient._extract_json('{"a": 1}')
assert r2 == '{"a": 1}', f"FAIL test2: {r2!r}"

# Test 3: nested braces in strings
r3 = LLMClient._extract_json('{"x": "{not json}", "y": 2}')
assert r3 == '{"x": "{not json}", "y": 2}', f"FAIL test3: {r3!r}"

# Test 4: preamble before fence (the real bug case)
r4 = LLMClient._extract_json('Some text before\n```json\n{"a": 1}\n```\nSome text after')
assert r4 == '{"a": 1}', f"FAIL test4: {r4!r}"

# Test 5: no fence, no JSON — should return original
r5 = LLMClient._extract_json('just plain text')
assert r5 == 'just plain text', f"FAIL test5: {r5!r}"

print("_extract_json: ALL OK")

# --- Test _is_valid_module ---
from src.verilog_utils import is_valid_module

assert is_valid_module('module foo(); endmodule') == True
assert is_valid_module('wire x = 1;') == False
assert is_valid_module('module bar(); wire x;') == False
assert is_valid_module('module my_mod #(param W=8)(input clk, output out);\nassign out=1;\nendmodule') == True

print("_is_valid_module: ALL OK")

# --- Test _extract_verilog ---
from src.verilog_utils import extract_verilog

v1 = extract_verilog('```verilog\nmodule foo();\nendmodule\n```')
assert v1 == 'module foo();\nendmodule', f"FAIL v1: {v1!r}"

v2 = extract_verilog('module foo();\nendmodule')
assert v2 == 'module foo();\nendmodule', f"FAIL v2: {v2!r}"

print("_extract_verilog: ALL OK")

print("\nAll regression tests passed.")
