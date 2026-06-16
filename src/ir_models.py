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


class ModuleSpec(BaseModel):
    module_name: str
    category: str  # e.g. "modular_arithmetic", "memory", "butterfly", "control"
    summary: str
    hardware_spec: HardwareSpec = Field(default_factory=HardwareSpec)


class PaperAnalysis(BaseModel):
    paper_title: str
    innovations: list[ModuleSpec]


class GeneratedModule(BaseModel):
    module_name: str
    verilog_code: str
