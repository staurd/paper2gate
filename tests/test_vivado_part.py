"""Tests for FPGA device extraction and Vivado part resolution."""

import sys

sys.path.insert(0, ".")

import src.main as main_mod
from src.innovation_extractor import _parse_analysis
from src.ir_models import HardwareSpec, ModuleSpec, PaperAnalysis
from src.prompt_manager import load_prompt
from src.schemes import KYBER, render_vars
from src.vivado_report import DEFAULT_PART, resolve_part

# --- a complete part string is used verbatim --------------------------------

# The Kyber butterfly paper prints the full part.
assert resolve_part("xc7a200tffg1156-3") == ("xc7a200tffg1156-3", "paper")
assert resolve_part("XC7A200TFFG1156-3") == ("xc7a200tffg1156-3", "paper")

# --- device + speed grade but no package is NOT a legal part -----------------

# Real case: the High-Speed NTT paper prints "Xilinx Artix XC7A100T-3".
# Vivado rejects a part with no package, so this must go through the table.
assert resolve_part("xc7a100t-3") == ("xc7a100tcsg324-1", "completed")
assert resolve_part("Xilinx Artix XC7A100T-3") == ("xc7a100tcsg324-1", "completed")

# --- bare device name gets package/speed grade filled in ---------------------

# The Dilithium paper prints only "XC7A100T"; Vivado would reject it as-is.
assert resolve_part("XC7A100T") == ("xc7a100tcsg324-1", "completed")
# Device + package but no speed grade.
assert resolve_part("xc7a100tcsg324") == ("xc7a100tcsg324-1", "completed")
# Prose around the device name.
assert resolve_part("Xilinx Artix-7 XC7A100T") == ("xc7a100tcsg324-1", "completed")
assert resolve_part("  XC7Z020  ") == ("xc7z020clg400-1", "completed")
# A complete part embedded in prose beats the bare name.
assert resolve_part("Artix-7 (xc7a200tffg1156-3)") == ("xc7a200tffg1156-3", "paper")

# --- fallbacks ---------------------------------------------------------------

assert resolve_part("") == (DEFAULT_PART, "empty")
assert resolve_part(None) == (DEFAULT_PART, "empty")
assert resolve_part("   ") == (DEFAULT_PART, "empty")
# Family only, or a device we have no package for => default + a warning.
assert resolve_part("Artix-7") == (DEFAULT_PART, "unknown")
assert resolve_part("Spartan-6") == (DEFAULT_PART, "unknown")

# --- priority: --part > config.yaml > paper > default ------------------------

main_mod.get_vivado_config = lambda: {}  # config.yaml ships with part: ""
assert main_mod.resolve_synth_part(None, "xc7a200tffg1156-3") == "xc7a200tffg1156-3"
assert main_mod.resolve_synth_part(None, "XC7A100T") == "xc7a100tcsg324-1"
assert main_mod.resolve_synth_part(None, "") == DEFAULT_PART
# --part beats everything...
assert main_mod.resolve_synth_part("xc7a35tcsg324-1", "XC7A100T") == "xc7a35tcsg324-1"

main_mod.get_vivado_config = lambda: {"part": "xc7a200tffg1156-3"}
# ...and an explicit config.yaml part overrides the paper's device.
assert main_mod.resolve_synth_part(None, "XC7A100T") == "xc7a200tffg1156-3"
assert main_mod.resolve_synth_part("xc7a35tcsg324-1", "XC7A100T") == "xc7a35tcsg324-1"

# --- extraction: prompt asks for the device ---------------------------------

prompt = load_prompt(
    "extract_innovation.jinja",
    paper_text="",
    target_interface="modmul",
    **render_vars(KYBER),
)
assert "fpga_device" in prompt


# --- extraction: parsing -----------------------------------------------------

a = _parse_analysis({
    "paper_title": "t",
    "fpga_device": "  XC7A100T ",
    "innovations": [{"module_name": "modmul", "hardware_spec": {}}],
})
assert a.fpga_device == "XC7A100T"

# Missing (older spec files, or a paper that states no device) => "".
b = _parse_analysis({"paper_title": "t", "innovations": []})
assert b.fpga_device == ""

# A non-string from the LLM must not crash the pipeline.
c = _parse_analysis({"paper_title": "t", "fpga_device": None, "innovations": []})
assert c.fpga_device == ""

# Round-trip through the saved hardware_specs.json representation.
d = PaperAnalysis.model_validate_json(
    PaperAnalysis(
        paper_title="t",
        innovations=[ModuleSpec(module_name="m", category="c", summary="s",
                               hardware_spec=HardwareSpec())],
        fpga_device="xc7a200tffg1156-3",
    ).model_dump_json()
)
assert d.fpga_device == "xc7a200tffg1156-3"
# Old spec files (no fpga_device key) still load.
e = PaperAnalysis.model_validate_json('{"paper_title": "t", "innovations": []}')
assert e.fpga_device == ""

print("All Vivado part tests passed.")
