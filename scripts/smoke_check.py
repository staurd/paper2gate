"""Offline smoke checks for the Paper2Gate pipeline."""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402
from src.ir_models import PaperAnalysis  # noqa: E402
from src.llm_client import LLMClient  # noqa: E402
from src.main import PipelineError, run_pipeline  # noqa: E402
from src.schemes import DILITHIUM, KYBER, resolve_scheme  # noqa: E402
from src.verilog_utils import validate_modmul_interface  # noqa: E402
from src.vivado_report import DEFAULT_PART, resolve_part  # noqa: E402


def main() -> None:
    config = load_config()
    assert "yosys" not in config, "Yosys configuration must be removed"
    assert config.get("llm", {}).get("provider"), "LLM provider is missing"

    assert resolve_scheme("kyber") is KYBER
    assert resolve_scheme("dilithium") is DILITHIUM
    with tempfile.TemporaryDirectory() as temp_dir:
        for name, profile in (("kyber", KYBER), ("dilithium", DILITHIUM)):
            body = (
                f"This work implements {name} reduction; Kyber and Dilithium are compared. " * 60
            ) + "LATE_BODY_MARKER"
            raw_text = body + "\nReferences\n8380417 3329\n" + "citation " * 200
            evidence = f"implements {name} reduction"
            with patch("src.main.extract_text", return_value=raw_text), \
                 patch("src.main.LLMClient") as client_class, \
                 patch("src.main.extract_innovations", return_value=PaperAnalysis(paper_title="test", innovations=[])) as extract:
                client_class.return_value.generate_structured.return_value = json.dumps({"scheme": name, "evidence": evidence})
                output_dir = run_pipeline(f"{name}.pdf", output_base=temp_dir, dry_run=True)
            prompt = client_class.return_value.generate_structured.call_args.kwargs["user_prompt"]
            assert client_class.return_value.generate_structured.call_args.kwargs["stage"] == "extract"
            assert prompt.split("Paper excerpt:\n", 1)[1].strip() == body[:2000].strip()
            assert "LATE_BODY_MARKER" not in prompt
            assert (output_dir / "extracted_text" / "full_text.txt").read_text(encoding="utf-8") == raw_text
            assert extract.call_args.args[0].strip() == body.strip()
            assert extract.call_args.kwargs["scheme"] is profile
            summary = json.loads((output_dir / "run_summary.json").read_text(encoding="utf-8"))
            assert summary["scheme"]["name"] == name
            assert summary["scheme"]["source"] == "llm"
            assert summary["scheme"]["evidence"] == evidence
            assert summary["stages"]["classify_scheme"]["status"] == "passed"

        for label, response in (
            ("unknown", '{"scheme":"unknown","evidence":"not stated"}'),
            ("mixed", '{"scheme":"mixed","evidence":"both are targets"}'),
            ("malformed", "not json"),
            ("invalid", '{"scheme":"falcon","evidence":"unsupported"}'),
        ):
            with patch("src.main.extract_text", return_value="Kyber paper"), \
                 patch("src.main.LLMClient") as client_class, \
                 patch("src.main.extract_innovations") as extract, \
                 patch("src.main.generate_with_check") as generate:
                client_class.return_value.generate_structured.return_value = response
                try:
                    run_pipeline(f"{label}.pdf", output_base=temp_dir, dry_run=True)
                except PipelineError as exc:
                    assert "--scheme" in str(exc)
                else:
                    raise AssertionError(f"{label} classification should fail")
                extract.assert_not_called()
                generate.assert_not_called()
            summary_path = next(Path(temp_dir).glob(f"{label}_*/run_summary.json"))
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            assert summary["status"] == "failed"
            assert summary["stages"]["classify_scheme"]["status"] == "failed"
            if label in ("unknown", "mixed"):
                assert summary["scheme"]["name"] == label
                assert summary["scheme"]["source"] == "llm"
                assert summary["scheme"]["evidence"]

        with patch("src.main.extract_text", return_value="Manual Kyber paper"), \
             patch("src.main.LLMClient") as client_class, \
             patch("src.main.extract_innovations", return_value=PaperAnalysis(paper_title="test", innovations=[])) as extract:
            output_dir = run_pipeline("manual.pdf", output_base=temp_dir, dry_run=True, scheme="kyber")
        client_class.return_value.generate_structured.assert_not_called()
        assert extract.call_args.kwargs["scheme"] is KYBER
        summary = json.loads((output_dir / "run_summary.json").read_text(encoding="utf-8"))
        assert summary["scheme"]["source"] == "cli"
        assert summary["stages"]["classify_scheme"]["status"] == "skipped"
    assert resolve_part("")[0] == DEFAULT_PART
    assert LLMClient._extract_json("prefix\n```json\n{\"ok\": true}\n```") == '{"ok": true}'

    valid_modmul = """module modmul(
        input clk, rst,
        input [11:0] A, B,
        output [11:0] R
    );
        reg [23:0] P_DSP;
        reg [23:0] P_R;
        always @* P_DSP = A * B;
        always @(posedge clk or posedge rst) begin
            if (rst) P_R <= 0;
            else P_R <= P_DSP;
        end
        assign R = P_R[11:0];
    endmodule"""
    assert validate_modmul_interface(valid_modmul, 12) == []
    invalid_modmul = valid_modmul.replace("assign R = P_R[11:0];", "assign R = P_R % 3329;")
    assert validate_modmul_interface(invalid_modmul, 12)

    print("Paper2Gate smoke checks passed.")


if __name__ == "__main__":
    main()
