"""Test iverilog integration."""
import sys
sys.path.insert(0, ".")  # noqa

from src.code_generator import _run_iverilog
from src.verilog_utils import is_valid_module

# Test 1: valid module
valid_code = """module test(input clk, output reg out);
  always @(posedge clk) out <= ~out;
endmodule"""
passed, errors = _run_iverilog(valid_code)
print(f"Valid module - passed: {passed}")
if errors:
    print(f"  Errors: {errors.strip()[:150]}")

# Test 2: bad module (undeclared signal)
bad_code = """module bad(input clk, output out);
  assign out = x;
endmodule"""
passed, errors = _run_iverilog(bad_code)
print(f"\nBad module - passed: {passed}")
if errors:
    print(f"  Errors: {errors.strip()[:300]}")

# Test 3: structural check
assert is_valid_module(valid_code), "valid_module should pass"
assert is_valid_module(bad_code), "bad_code still has module structure"
assert not is_valid_module("wire x = 1;"), "no module should fail"

print("\nAll tests completed.")
