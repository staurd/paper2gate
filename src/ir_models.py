"""Structured IR models for hardware module specifications."""

from pydantic import BaseModel, Field


class PortSpec(BaseModel):
    name: str
    width: str  # e.g. "DATA_WIDTH", "2*DATA_WIDTH", "8"
    direction: str  # "input" | "output"
    desc: str = ""


class HardwareSpec(BaseModel):
    parameters: dict[str, str] = Field(default_factory=dict)
    ports: dict[str, list[PortSpec]] = Field(default_factory=dict)
    behavior: str = ""
    timing: str = ""
    constraints: str = ""
    # Constant factor the module's output is off by, relative to a true a*b mod q.
    # K-reduction computes k*a*b mod q (k=13 for Kyber), cancelled in the NTT by
    # pre-scaling twiddles with k^-1. Barrett/Montgomery/plain reduction => 1.
    correction_factor: int = 1
    # Total clock cycles from valid A/B at the inputs to valid R at the output,
    # counting EVERY register stage including the product register.
    # 0 = unknown (the pipeline applies the scheme's default latency).
    latency_cycles: int = 0


class ModuleSpec(BaseModel):
    module_name: str
    category: str  # e.g. "modular_arithmetic", "memory", "butterfly", "control"
    summary: str
    hardware_spec: HardwareSpec = Field(default_factory=HardwareSpec)


class PaperAnalysis(BaseModel):
    paper_title: str
    innovations: list[ModuleSpec]
    # FPGA the paper's OWN experiments target, as printed (e.g. "XC7A100T",
    # "xc7a200tffg1156-3"). "" = not stated; the synthesis stage then falls
    # back to its default part. Resolved to a legal Vivado part by
    # vivado_report.resolve_part().
    fpga_device: str = ""


class GeneratedModule(BaseModel):
    module_name: str
    verilog_code: str


def parse_module_params(hardware_spec_params: dict) -> dict[str, int | None]:
    """
    Extract numeric parameters from a HardwareSpec's parameters dict.
    Handles description strings like '12 - input bit width (Kyber element size)'.
    Returns {'DATA_WIDTH': int, 'Q': int, 'K': int|None}.
    """
    import re

    def _extract_int(value: str) -> int | None:
        m = re.search(r'\d+', str(value))
        return int(m.group(0)) if m else None

    result: dict[str, int | None] = {"DATA_WIDTH": 12, "Q": 3329, "K": None}
    for key, value in hardware_spec_params.items():
        v = _extract_int(value)
        if v is None:
            continue
        key_upper = key.upper()
        if key_upper in ("Q", "MODULUS", "MODULUS_Q", "MOD"):
            result["Q"] = v
        elif key_upper in ("K", "K_PARAM"):
            result["K"] = v
        elif key_upper in ("DATA_WIDTH", "WIDTH", "N_BITS", "DW"):
            result["DATA_WIDTH"] = v
    return result
