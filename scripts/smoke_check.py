"""Offline smoke checks for the Paper2Gate pipeline."""

import json
import os
import sys
import tempfile
from types import SimpleNamespace
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from rich.console import Console

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_llm_config, load_config  # noqa: E402
from src.code_generator import _fix_from_iverilog, fix_from_verification, generate_modmul  # noqa: E402
from src.innovation_extractor import classify_scheme, extract_innovations  # noqa: E402
from src.ir_models import GeneratedModule, HardwareSpec, ModuleSpec, PaperAnalysis  # noqa: E402
from src.llm_client import LLMClient  # noqa: E402
from src.main import (  # noqa: E402
    PipelineError,
    _batch_result_from_summary,
    _build_batch_results_table,
    main as cli_main,
    run_pipeline,
)
from src.schemes import DILITHIUM, KYBER, resolve_scheme  # noqa: E402
from src.verilog_utils import validate_modmul_interface  # noqa: E402
from src.vivado_report import DEFAULT_PART, resolve_part  # noqa: E402


VALID_MODMUL = """module modmul(
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


class RecordingClient:
    def __init__(self, response: str):
        self.response = response
        self.calls = []

    def generate_structured(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeOpenAICompletions:
    def __init__(self, legacy_token_parameter=False):
        self.calls = []
        self.legacy_token_parameter = legacy_token_parameter

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.legacy_token_parameter and "max_completion_tokens" in kwargs:
            raise RuntimeError("unsupported parameter: max_completion_tokens")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
        )


class FakeOpenAI:
    instances = []
    legacy_token_parameter = False

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.chat = SimpleNamespace(
            completions=FakeOpenAICompletions(self.legacy_token_parameter)
        )
        self.__class__.instances.append(self)


def make_timing_client() -> LLMClient:
    client = object.__new__(LLMClient)
    client.provider = "openai"
    client.model = "test"
    client.extract_model = "test"
    client.generate_model = "test"
    client.max_tokens = 32
    client.temperature = 0.0
    client._timings = LLMClient._empty_timing_data()
    return client


def main() -> None:
    config = load_config()
    assert "yosys" not in config, "Yosys configuration must be removed"
    assert config.get("llm", {}).get("provider"), "LLM provider is missing"
    assert set(config.get("llm", {}).get("providers", {})) >= {"deepseek", "openai"}

    with patch.dict(os.environ, {"OPENAI_API_KEY": "env-openai-key"}, clear=False), \
         patch(
             "src.config.load_config",
             return_value={
                 "llm": {
                     "provider": "openai",
                     "common": {"max_tokens": 4096},
                     "providers": {
                         "deepseek": {
                             "model": "deepseek-model",
                             "api_key": "deepseek-config-key",
                         },
                         "openai": {
                             "model": "relay-model",
                             "api_key": "",
                             "base_url": "https://zyrus.aitoken.credit/v1",
                         },
                     },
                 }
             },
         ):
        openai_cfg = get_llm_config()
    assert openai_cfg["api_key"] == "env-openai-key"
    assert openai_cfg["provider"] == "openai"
    assert openai_cfg["model"] == "relay-model"
    assert openai_cfg["max_tokens"] == 4096
    assert openai_cfg["base_url"] == "https://zyrus.aitoken.credit/v1"

    with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "env-deepseek-key"}, clear=False), \
         patch(
             "src.config.load_config",
             return_value={
                 "llm": {
                     "provider": "deepseek",
                     "providers": {
                         "deepseek": {
                             "model": "deepseek-model",
                             "api_key": "",
                         },
                         "openai": {"model": "relay-model"},
                     },
                 }
             },
         ):
        deepseek_cfg = get_llm_config()
    assert deepseek_cfg["provider"] == "deepseek"
    assert deepseek_cfg["model"] == "deepseek-model"
    assert deepseek_cfg["api_key"] == "env-deepseek-key"

    with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False), patch(
        "src.config.load_config",
        return_value={
            "llm": {
                "provider": "openai",
                "model": "legacy-model",
                "api_key": "legacy-key",
            }
        },
    ):
        legacy_cfg = get_llm_config()
    assert legacy_cfg["model"] == "legacy-model"
    assert legacy_cfg["api_key"] == "legacy-key"

    FakeOpenAI.instances = []
    FakeOpenAI.legacy_token_parameter = True
    with patch(
        "src.llm_client.get_llm_config",
        return_value={
            "provider": "openai",
            "model": "relay-model",
            "api_key": "config-key",
            "base_url": "https://zyrus.aitoken.credit/v1/",
            "max_tokens": 4096,
            "temperature": 0.2,
        },
    ), patch("openai.OpenAI", FakeOpenAI):
        relay_client = LLMClient()
        assert relay_client._call_openai_raw("system", "user", "relay-model") == "ok"
    assert FakeOpenAI.instances[0].kwargs == {
        "api_key": "config-key",
        "base_url": "https://zyrus.aitoken.credit/v1",
    }
    relay_calls = FakeOpenAI.instances[0].chat.completions.calls
    assert "max_completion_tokens" in relay_calls[0]
    assert relay_calls[1]["max_tokens"] == 4096
    assert relay_calls[1]["model"] == "relay-model"
    assert relay_calls[1]["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
    ]

    FakeOpenAI.instances = []
    FakeOpenAI.legacy_token_parameter = False
    with patch(
        "src.llm_client.get_llm_config",
        return_value={
            "provider": "deepseek",
            "model": "deepseek-model",
            "api_key": "deepseek-key",
        },
    ), patch("openai.OpenAI", FakeOpenAI):
        deepseek_client = LLMClient()
        deepseek_client._get_client()
    assert FakeOpenAI.instances[0].kwargs == {
        "api_key": "deepseek-key",
        "base_url": "https://api.deepseek.com",
    }

    with patch(
        "src.llm_client.get_llm_config",
        return_value={"provider": "unsupported", "model": "test", "api_key": "key"},
    ):
        try:
            LLMClient()
        except ValueError as exc:
            assert "Unsupported LLM provider" in str(exc)
        else:
            raise AssertionError("unsupported provider should fail clearly")

    assert resolve_scheme("kyber") is KYBER
    assert resolve_scheme("dilithium") is DILITHIUM

    timed = make_timing_client()
    with patch("src.llm_client.time.perf_counter", side_effect=[100.0, 102.5]), \
         patch.object(timed, "_call_openai_raw", return_value="ok"):
        assert timed.generate_structured("system", "user", operation="generate_modmul") == "ok"
    assert timed.timing_snapshot()["llm_calls"] == 1
    assert timed.timing_snapshot()["llm_seconds"] == 2.5
    assert timed.timing_snapshot()["llm_by_operation"]["generate_modmul"] == {"calls": 1, "seconds": 2.5}

    timed_retry = make_timing_client()
    with patch("src.llm_client.time.perf_counter", side_effect=[200.0, 204.0]), \
         patch("src.llm_client.time.sleep"), \
         patch.object(timed_retry, "_call_openai_raw", side_effect=[RuntimeError("temporary"), "ok"]):
        assert timed_retry.generate_structured("system", "user", max_retries=1, operation="fix_syntax") == "ok"
    assert timed_retry.timing_snapshot()["llm_by_operation"]["fix_syntax"] == {"calls": 2, "seconds": 4.0}

    timed_failure = make_timing_client()
    with patch("src.llm_client.time.perf_counter", side_effect=[300.0, 306.0]), \
         patch("src.llm_client.time.sleep"), \
         patch.object(timed_failure, "_call_openai_raw", side_effect=[RuntimeError("failed")] * 3):
        try:
            timed_failure.generate_structured("system", "user", max_retries=2, operation="fix_function")
        except RuntimeError:
            pass
        else:
            raise AssertionError("final LLM failure should be raised")
    assert timed_failure.timing_snapshot()["llm_by_operation"]["fix_function"] == {"calls": 3, "seconds": 6.0}

    classifier = RecordingClient('{"scheme":"kyber","evidence":"Kyber target"}')
    classify_scheme("Kyber modular reduction", classifier)
    assert classifier.calls[-1]["operation"] == "classify_scheme"
    extractor = RecordingClient('{"paper_title":"test","innovations":[]}')
    extract_innovations("Kyber modular reduction", extractor, scheme=KYBER)
    assert extractor.calls[-1]["operation"] == "extract_innovations"
    spec = ModuleSpec(
        module_name="modmul",
        category="modular_arithmetic",
        summary="",
        hardware_spec=HardwareSpec(),
    )
    generator = RecordingClient(VALID_MODMUL)
    generate_modmul(spec, generator, scheme=KYBER)
    assert generator.calls[-1]["operation"] == "generate_modmul"
    syntax_fixer = RecordingClient(VALID_MODMUL)
    _fix_from_iverilog(spec, VALID_MODMUL, "syntax error", syntax_fixer)
    assert syntax_fixer.calls[-1]["operation"] == "fix_syntax"
    function_fixer = RecordingClient(VALID_MODMUL)
    fix_from_verification(spec, VALID_MODMUL, "functional failure", function_fixer)
    assert function_fixer.calls[-1]["operation"] == "fix_function"

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
            assert client_class.return_value.generate_structured.call_args.kwargs["operation"] == "classify_scheme"
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
            assert "timing" in summary
            assert "llm_seconds" in summary["timing"]

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
            assert "timing" in summary
            assert summary["timing"]["llm_calls"] == 0
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
        assert summary["resources"] == {"LUT": None, "FF": None, "DSP": None, "BRAM": None}

        synth_stats = {"LUT": 124, "FF": 18, "DSP": 2, "BRAM": 0}
        synth_analysis = PaperAnalysis(
            paper_title="synthesis test",
            innovations=[
                ModuleSpec(
                    module_name="modmul",
                    category="modular_arithmetic",
                    summary="test implementation",
                    hardware_spec=HardwareSpec(),
                )
            ],
        )
        with patch("src.main.extract_text", return_value="Manual Kyber paper"), \
             patch("src.main.LLMClient"), \
             patch("src.main.extract_innovations", return_value=synth_analysis), \
             patch(
                 "src.main.generate_modmul",
                 return_value=GeneratedModule(module_name="modmul", verilog_code=VALID_MODMUL),
             ), \
             patch("src.main.vivado_available", return_value=True), \
             patch("src.main.run_synthesis", return_value=synth_stats):
            output_dir = run_pipeline(
                "synthesis.pdf",
                output_base=temp_dir,
                use_review=False,
                use_verify=False,
                scheme="kyber",
                part=DEFAULT_PART,
            )
        summary = json.loads((output_dir / "run_summary.json").read_text(encoding="utf-8"))
        assert summary["resources"] == synth_stats
        assert summary["synthesis_part"] == DEFAULT_PART

    populated = _batch_result_from_summary(
        "paper.pdf",
        {
            "status": "passed",
            "synthesis_part": "xc7a200tffg1156-3",
            "timing": {"llm_seconds": 1.25, "pipeline_seconds": 2.5},
            "resources": {"LUT": 124, "FF": 18, "DSP": 2, "BRAM": 0},
        },
    )
    unavailable = _batch_result_from_summary("unavailable.pdf", {"status": "failed"})
    batch_table = _build_batch_results_table([populated, unavailable])
    rendered_output = StringIO()
    Console(file=rendered_output, width=200, color_system=None).print(batch_table)
    rendered_table = rendered_output.getvalue()
    assert "Outputs" not in rendered_table
    assert all(
        heading in rendered_table
        for heading in ("Part", "LLM Time", "Total Time", "LUT", "FF", "DSP", "BRAM")
    )
    assert "xc7a200tffg1156-3" in rendered_table
    assert "1.250s" in rendered_table
    assert "2.500s" in rendered_table
    assert "124" in rendered_table
    assert rendered_table.count("n/a") >= 6

    with tempfile.TemporaryDirectory() as cli_temp_dir:
        output_dirs = []
        for name, part in (("single", "xc7a35tcsg324-1"), ("batch", "xc7a200tffg1156-3")):
            output_dir = Path(cli_temp_dir) / name
            output_dir.mkdir()
            (output_dir / "run_summary.json").write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "synthesis_part": part,
                        "timing": {"llm_seconds": 1.0, "pipeline_seconds": 2.0},
                        "resources": {"LUT": 1, "FF": 2, "DSP": 3, "BRAM": 4},
                    }
                ),
                encoding="utf-8",
            )
            output_dirs.append(output_dir)

        with patch("src.main.expand_pdf_paths", return_value=["single.pdf"]), \
             patch("src.main.os.path.exists", return_value=True), \
             patch("src.main.run_pipeline", return_value=output_dirs[0]), \
             patch("src.main._build_batch_results_table", wraps=_build_batch_results_table) as build_table, \
             patch.object(sys, "argv", ["paper2gate", "single.pdf"]):
            try:
                cli_main()
            except SystemExit as exc:
                assert exc.code == 0
        build_table.assert_not_called()

        with patch("src.main.expand_pdf_paths", return_value=["first.pdf", "second.pdf"]), \
             patch("src.main.os.path.exists", return_value=True), \
             patch("src.main.run_pipeline", side_effect=output_dirs), \
             patch("src.main._build_batch_results_table", wraps=_build_batch_results_table) as build_table, \
             patch.object(sys, "argv", ["paper2gate", "first.pdf", "second.pdf"]):
            try:
                cli_main()
            except SystemExit as exc:
                assert exc.code == 0
        build_table.assert_called_once()
    assert resolve_part("")[0] == DEFAULT_PART
    assert LLMClient._extract_json("prefix\n```json\n{\"ok\": true}\n```") == '{"ok": true}'

    valid_modmul = VALID_MODMUL
    assert validate_modmul_interface(valid_modmul, 12) == []
    invalid_modmul = valid_modmul.replace("assign R = P_R[11:0];", "assign R = P_R % 3329;")
    assert validate_modmul_interface(invalid_modmul, 12)

    print("Paper2Gate smoke checks passed.")


if __name__ == "__main__":
    main()
