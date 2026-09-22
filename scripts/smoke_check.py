"""Offline smoke checks for the Paper2Gate pipeline."""

import json
import os
import sys
import tempfile
import base64
from types import SimpleNamespace
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from rich.console import Console

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_llm_config, get_vision_config, load_config  # noqa: E402
from src.code_generator import _fix_from_iverilog, fix_from_verification, generate_modmul  # noqa: E402
from src.figure_analysis import analyze_figure_pages, classify_figure_captions  # noqa: E402
from src.innovation_extractor import (  # noqa: E402
    _parse_analysis,
    classify_scheme,
    extract_innovations,
)
from src.ir_models import GeneratedModule, HardwareSpec, ModuleSpec, PaperAnalysis  # noqa: E402
from src.llm_client import LLMClient  # noqa: E402
from src.main import (  # noqa: E402
    PipelineError,
    _batch_result_from_summary,
    _build_batch_results_table,
    _print_llm_metadata,
    _write_batch_summary,
    build_batch_output_dir,
    main as cli_main,
    run_pipeline,
)
from src.pdf_parser import extract_figure_captions  # noqa: E402
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
    assert isinstance(config.get("llm", {}).get("vision", {}).get("enabled"), bool)

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

    with patch.dict(os.environ, {"OPENAI_API_KEY": "env-openai-key"}, clear=False), \
         patch(
             "src.config.load_config",
             return_value={
                 "llm": {
                     "vision": {
                         "enabled": True,
                         "provider": "openai",
                         "model": "vision-model",
                         "api_key": "config-key",
                     }
                 }
             },
         ):
        vision_cfg = get_vision_config()
    assert vision_cfg["enabled"] is True
    assert vision_cfg["provider"] == "openai"
    assert vision_cfg["model"] == "vision-model"
    assert vision_cfg["api_key"] == "env-openai-key"

    multimodal_client = object.__new__(LLMClient)
    multimodal_client.provider = "openai"
    multimodal_client.vision = True
    multimodal_client.model = "vision-model"
    multimodal_client.extract_model = "vision-model"
    multimodal_client.generate_model = "vision-model"
    multimodal_client._timings = LLMClient._empty_timing_data()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as image_file:
        image_file.write(b"fake-png")
        image_path = Path(image_file.name)
    captured_multimodal = {}

    def capture_multimodal(system_prompt, user_prompt, model):
        captured_multimodal.update(
            {"system": system_prompt, "content": user_prompt, "model": model}
        )
        return '{"ok": true}'

    try:
        with patch.object(multimodal_client, "_call_openai_raw", side_effect=capture_multimodal):
            assert multimodal_client.generate_multimodal_structured(
                "system", "analyze this", image_path
            ) == '{"ok": true}'
    finally:
        image_path.unlink()
    assert captured_multimodal["model"] == "vision-model"
    assert captured_multimodal["content"][0] == {"type": "text", "text": "analyze this"}
    image_url = captured_multimodal["content"][1]["image_url"]["url"]
    assert image_url.startswith("data:image/png;base64,")
    assert base64.b64decode(image_url.split(",", 1)[1]) == b"fake-png"

    import fitz

    with tempfile.TemporaryDirectory() as figure_temp_dir:
        figure_pdf = Path(figure_temp_dir) / "figures.pdf"
        document = fitz.open()
        first_page = document.new_page()
        first_page.insert_text((72, 72), "Figure 1: Proposed modular reduction datapath\nFigure 2: NTT butterfly")
        second_page = document.new_page()
        second_page.insert_text((72, 72), "Fig. 3. Pipelined modular multiplier")
        document.save(figure_pdf)
        document.close()
        figures = extract_figure_captions(str(figure_pdf))
        assert [item["figure_id"] for item in figures] == ["fig_1", "fig_2", "fig_3"]
        assert [item["page"] for item in figures] == [1, 1, 2]

        classifier = RecordingClient(
            '[{"figure_id":"fig_1","page":1,"relevant":true,"category":"modular_reduction","confidence":0.9,"reason":"datapath"},{"figure_id":"fig_2","page":1,"relevant":false},{"figure_id":"fig_3","page":2,"relevant":true,"category":"modular_multiplication","confidence":0.8}]'
        )
        progress_messages = []
        candidates = classify_figure_captions(figures, classifier, progress=progress_messages.append)
        assert [item["figure_id"] for item in candidates if item["relevant"]] == ["fig_1", "fig_3"]
        assert classifier.calls[0]["operation"] == "classify_figures"
        assert any("Classifying figure captions" in message for message in progress_messages)

        vision_client = object.__new__(LLMClient)
        vision_client.provider = "openai"
        vision_client.model = "vision-model"
        vision_client.extract_model = "vision-model"
        vision_client.generate_model = "vision-model"
        vision_client._timings = LLMClient._empty_timing_data()
        vision_client.generate_multimodal_structured = lambda *args, **kwargs: '{"relevant":true,"datapath":["DSP","reduction"],"latency_cycles":3}'
        evidence, errors = analyze_figure_pages(
            str(figure_pdf),
            figures,
            candidates,
            Path(figure_temp_dir) / "output",
            vision_client,
            progress=progress_messages.append,
        )
        assert errors == []
        assert len(evidence) == 2
        assert any("Rendering PDF page 1" in message for message in progress_messages)
        assert any("Analyzing PDF page 2" in message for message in progress_messages)
        assert (Path(figure_temp_dir) / "output" / "figures" / "fig_1_page_1.png").exists()
        assert (Path(figure_temp_dir) / "output" / "figures" / "fig_3_page_2.png").exists()

    sample_paper = PROJECT_ROOT / "test_papers" / "A_Better_Kyber_Butterfly_for_FPGAs.pdf"
    if sample_paper.exists():
        sample_figures = extract_figure_captions(str(sample_paper))
        fig4_pages = [item["pdf_page"] for item in sample_figures if item["figure_id"] == "fig_4"]
        assert fig4_pages == [5], f"Figure 4 should map to PDF page 5, got {fig4_pages}"
        assert all(item["source"] == "caption" for item in sample_figures)

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
    relay_metadata = relay_client.metadata_snapshot()
    assert relay_metadata["provider"] == "openai"
    assert relay_metadata["model"] == "relay-model"
    assert relay_metadata["by_operation"]["classify_scheme"] == {
        "provider": "openai",
        "model": "relay-model",
    }
    assert relay_metadata["by_operation"]["generate_modmul"] == {
        "provider": "openai",
        "model": "relay-model",
    }

    with patch(
        "src.llm_client.get_llm_config",
        return_value={
            "provider": "deepseek",
            "model": "deepseek-default",
            "extract_model": "deepseek-reasoner",
            "generate_model": "deepseek-chat",
            "api_key": "deepseek-key",
        },
    ):
        staged_client = LLMClient()
    staged_metadata = staged_client.metadata_snapshot()
    assert staged_metadata["provider"] == "deepseek"
    assert staged_metadata["extract_model"] == "deepseek-reasoner"
    assert staged_metadata["generate_model"] == "deepseek-chat"
    assert staged_metadata["by_operation"]["extract_innovations"] == {
        "provider": "deepseek",
        "model": "deepseek-reasoner",
    }
    assert staged_metadata["by_operation"]["fix_function"] == {
        "provider": "deepseek",
        "model": "deepseek-chat",
    }

    with patch("src.main.console.print") as print_console:
        _print_llm_metadata(relay_client)
    printed = "\n".join(str(call.args[0]) for call in print_console.call_args_list)
    assert "LLM provider: [cyan]openai[/cyan]" in printed
    assert "LLM model: [cyan]relay-model[/cyan]" in printed
    assert "LLM extract model" not in printed

    with patch("src.main.console.print") as print_console:
        _print_llm_metadata(staged_client)
    printed = "\n".join(str(call.args[0]) for call in print_console.call_args_list)
    assert "LLM provider: [cyan]deepseek[/cyan]" in printed
    assert "LLM extract model: [cyan]deepseek-reasoner[/cyan]" in printed
    assert "LLM generate model: [cyan]deepseek-chat[/cyan]" in printed

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
    visual_extractor = RecordingClient('{"paper_title":"test","innovations":[]}')
    extract_innovations(
        "Kyber modular reduction",
        visual_extractor,
        scheme=KYBER,
        visual_evidence=[{"page": 4, "latency_cycles": 3, "evidence": ["register"]}],
    )
    assert "Visual evidence from selected hardware figures" in visual_extractor.calls[-1]["user_prompt"]
    assert '"latency_cycles": 3' in visual_extractor.calls[-1]["user_prompt"]
    assert "prior work" in visual_extractor.calls[-1]["user_prompt"]
    signed_analysis = _parse_analysis(
        {
            "paper_title": "signed reduction",
            "innovations": [
                {
                    "module_name": "modmul",
                    "category": "modular_arithmetic",
                    "summary": "A negated K-reduction.",
                    "hardware_spec": {
                        "behavior": "The algorithm outputs C' = -13*C mod 3329.",
                        "correction_factor": 13,
                    },
                }
            ],
        }
    )
    assert signed_analysis.innovations[0].hardware_spec.correction_factor == -13
    internal_negative_analysis = _parse_analysis(
        {
            "paper_title": "internal signed step",
            "innovations": [
                {
                    "module_name": "modmul",
                    "category": "modular_arithmetic",
                    "summary": "A positive K-reduction output.",
                    "hardware_spec": {
                        "behavior": (
                            "Compute the intermediate -13*Cl, then return "
                            "R = 13*(A*B) mod 3329."
                        ),
                        "correction_factor": 13,
                    },
                }
            ],
        }
    )
    assert internal_negative_analysis.innovations[0].hardware_spec.correction_factor == 13
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
                client_class.return_value.provider = "openai"
                client_class.return_value.model = "test-model"
                client_class.return_value.extract_model = "test-model"
                client_class.return_value.generate_model = "test-model"
                client_class.return_value.metadata_snapshot.return_value = {
                    "provider": "openai",
                    "model": "test-model",
                    "extract_model": "test-model",
                    "generate_model": "test-model",
                    "by_operation": {
                        "classify_scheme": {"provider": "openai", "model": "test-model"}
                    },
                }
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
            assert summary["llm"]["provider"] == "openai"
            assert summary["llm"]["model"] == "test-model"
            assert summary["llm"]["by_operation"]["classify_scheme"] == {
                "provider": "openai",
                "model": "test-model",
            }

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
        batch_root = build_batch_output_dir(Path(cli_temp_dir))
        second_batch_root = build_batch_output_dir(Path(cli_temp_dir))
        assert second_batch_root.name != batch_root.name
        first_output = batch_root / "paper_a_20260922_100000"
        second_output = batch_root / "paper_b_20260922_100100"
        failed_output = batch_root / "paper_c_20260922_100200"
        first_output.mkdir()
        second_output.mkdir()
        failed_output.mkdir()
        first_result = _batch_result_from_summary(
            "paper_a.pdf",
            {
                "paper": str(Path("paper_a.pdf").resolve()),
                "status": "passed",
                "synthesis_part": "xc7a35tcsg324-1",
                "timing": {"llm_seconds": 1.5, "pipeline_seconds": 3.0, "llm_calls": 2},
                "llm": {
                    "provider": "openai",
                    "model": "gpt-5.6-terra",
                    "extract_model": "gpt-5.6-terra",
                    "generate_model": "gpt-5.6-terra",
                    "by_operation": {},
                },
                "resources": {"LUT": 1, "FF": 2, "DSP": 3, "BRAM": 4},
            },
            first_output,
        )
        second_result = _batch_result_from_summary(
            "paper_b.pdf",
            {
                "paper": str(Path("paper_b.pdf").resolve()),
                "status": "unverified",
                "timing": {"llm_seconds": 2.5, "pipeline_seconds": 4.0, "llm_calls": 3},
                "resources": {"LUT": None, "FF": None, "DSP": None, "BRAM": None},
            },
            second_output,
        )
        failed_result = _batch_result_from_summary(
            "paper_c.pdf",
            {
                "paper": str(Path("paper_c.pdf").resolve()),
                "status": "failed",
                "error": "functional verification failed",
                "timing": {"llm_seconds": 0.5, "pipeline_seconds": 1.0, "llm_calls": 1},
                "resources": {"LUT": None, "FF": None, "DSP": None, "BRAM": None},
            },
            failed_output,
        )
        batch_summary_path = _write_batch_summary(
            batch_root,
            "2026-09-22T10:00:00+08:00",
            "2026-09-22T10:00:10+08:00",
            10.0,
            [first_result, second_result, failed_result],
        )
        batch_summary = json.loads(batch_summary_path.read_text(encoding="utf-8"))
        assert batch_summary["batch_id"] == batch_root.name
        assert batch_summary["status"] == "failed"
        assert batch_summary["paper_count"] == 3
        assert batch_summary["status_counts"] == {"passed": 1, "unverified": 1, "failed": 1}
        assert batch_summary["totals"] == {
            "pipeline_seconds": 8.0,
            "llm_seconds": 4.5,
            "llm_calls": 6,
        }
        assert batch_summary["papers"][0]["output_dir"] == first_output.name
        assert batch_summary["papers"][0]["run_summary"] == f"{first_output.name}/run_summary.json"
        assert batch_summary["papers"][0]["llm"]["provider"] == "openai"
        assert batch_summary["papers"][0]["llm"]["model"] == "gpt-5.6-terra"
        assert batch_summary["papers"][2]["error"] == "functional verification failed"

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
        assert list(Path(cli_temp_dir).glob("batch_*/batch_summary.json")) == [batch_summary_path]

        with patch("src.main.expand_pdf_paths", return_value=["first.pdf", "second.pdf"]), \
             patch("src.main.os.path.exists", return_value=True), \
             patch("src.main.run_pipeline", side_effect=output_dirs), \
             patch("src.main._build_batch_results_table", wraps=_build_batch_results_table) as build_table, \
             patch.object(
                 sys,
                 "argv",
                 ["paper2gate", "first.pdf", "second.pdf", "--output", cli_temp_dir],
             ):
            try:
                cli_main()
            except SystemExit as exc:
                assert exc.code == 0
        build_table.assert_called_once()
        generated_batches = sorted(Path(cli_temp_dir).glob("batch_*/batch_summary.json"))
        assert generated_batches
        generated_batch = json.loads(generated_batches[-1].read_text(encoding="utf-8"))
        assert generated_batch["paper_count"] == 2
        assert generated_batch["status"] == "passed"

        def failed_pipeline(**kwargs):
            failed_dir = Path(kwargs["output_base"]) / f"{Path(kwargs['pdf_path']).stem}_20260922_100300"
            failed_dir.mkdir(parents=True)
            (failed_dir / "run_summary.json").write_text(
                json.dumps(
                    {
                        "paper": str(Path(kwargs["pdf_path"]).resolve()),
                        "status": "failed",
                        "error": "mock generation failure",
                        "timing": {"llm_seconds": 1.0, "pipeline_seconds": 2.0, "llm_calls": 1},
                        "resources": {"LUT": None, "FF": None, "DSP": None, "BRAM": None},
                    }
                ),
                encoding="utf-8",
            )
            error = PipelineError("mock generation failure")
            error.output_dir = failed_dir
            raise error

        with patch("src.main.expand_pdf_paths", return_value=["failed.pdf", "failed2.pdf"]), \
             patch("src.main.os.path.exists", return_value=True), \
             patch("src.main.run_pipeline", side_effect=failed_pipeline), \
             patch.object(
                 sys,
                 "argv",
                 ["paper2gate", "failed.pdf", "failed2.pdf", "--output", cli_temp_dir],
             ):
            try:
                cli_main()
            except SystemExit as exc:
                assert exc.code == 1
        failed_batches = sorted(Path(cli_temp_dir).glob("batch_*/batch_summary.json"))
        failed_summary = json.loads(failed_batches[-1].read_text(encoding="utf-8"))
        assert failed_summary["status"] == "failed"
        assert failed_summary["status_counts"]["failed"] == 2
        assert failed_summary["papers"][0]["output_dir"] == "failed_20260922_100300"
        assert failed_summary["papers"][0]["error"] == "mock generation failure"
    assert resolve_part("")[0] == DEFAULT_PART
    assert LLMClient._extract_json("prefix\n```json\n{\"ok\": true}\n```") == '{"ok": true}'

    valid_modmul = VALID_MODMUL
    assert validate_modmul_interface(valid_modmul, 12) == []
    invalid_modmul = valid_modmul.replace("assign R = P_R[11:0];", "assign R = P_R % 3329;")
    assert validate_modmul_interface(invalid_modmul, 12)

    print("Paper2Gate smoke checks passed.")


if __name__ == "__main__":
    main()
