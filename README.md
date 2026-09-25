# Paper2Gate

Paper2Gate extracts modular-arithmetic hardware ideas from PQC papers and generates a fixed-interface Verilog `modmul` implementation.

中文项目说明见 [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md)。

## Run

```bash
paper2gate paper.pdf
```

The CLI also accepts a directory of PDFs. Useful options are:

```text
--dry-run       extract the modmul specification only
--no-review     skip Icarus syntax review
--no-verify     skip golden-model verification
--no-synth      skip Vivado synthesis
--scheme        auto, kyber, or dilithium
--part          Vivado part override
--output        output directory override
```

When processing multiple PDFs, Paper2Gate creates one `outputs/batch_<timestamp>/`
directory. Each paper gets its own child output directory, and the batch root
contains `batch_summary.json` with per-paper status, timing, LLM call counts,
resource results, LLM provider/model metadata, and batch totals. The batch total
sums processing time and LLM calls; FPGA resources remain per-paper because
summing resources from separate designs would be misleading. A single-PDF run
keeps the existing per-run output path and does not create a batch directory.

The project always reads the repository-root `config.yaml`. Machine-specific values can be supplied through environment variables:

```text
DEEPSEEK_API_KEY
OPENAI_API_KEY
ANTHROPIC_API_KEY
PAPER2GATE_IVERILOG_BINARY
PAPER2GATE_VIVADO_BINARY
```

Empty tool paths in `config.yaml` mean that the executable is searched on `PATH`.

Vivado synthesis runs from a short temporary working directory so its internal
`.Xil` files do not inherit long batch-paper paths; reports are still written
under each paper's `synthesis/modmul/` directory. On hosted shells that omit
Windows architecture environment variables, the runner automatically selects
the installed 64-bit Vivado runtime when it is available.

## LLM provider configuration

Both providers are preconfigured under `llm.providers`; change only
`llm.provider` to switch between them. DeepSeek remains the default:

```yaml
llm:
  provider: deepseek  # change to openai to use the GPT relay
  common:
    max_tokens: 8192
    temperature: 0.2
  providers:
    deepseek:
      model: deepseek-v4-pro
      api_key: ${DEEPSEEK_API_KEY}
      base_url: https://api.deepseek.com
    openai:
      model: gpt-4o
      api_key: ${OPENAI_API_KEY}
      base_url: https://zyrus.aitoken.credit/v1
  vision:
    enabled: false
    provider: openai
    model: gpt-4o
    api_key: ${OPENAI_API_KEY}
    base_url: https://zyrus.aitoken.credit/v1
    max_tokens: 4096
    temperature: 0.1
```

The selected provider's model, API key, and endpoint are merged automatically
into the client configuration. The `base_url` should normally include `/v1`.
The optional `extract_model` and `generate_model` settings can select different
models for paper extraction and Verilog generation. Store API keys in the
corresponding environment variable (or `.env`); do not commit keys to the
repository. If a relay rejects `max_completion_tokens`, the client retries that
request with the older `max_tokens` parameter.

Set `llm.vision.enabled` to `true` to enable figure analysis. The text model
first classifies figure captions, then the configured vision model receives
only pages containing captions relevant to modular multiplication or modular
reduction. Vision failures fall back to text-only extraction and mark the run
as `unverified`. The selected pages and evidence are saved as
`figure_candidates.json`, `vision_evidence.json`, and PNG files under `figures/`.
During a vision-enabled run, the console prints caption count, selected figure
IDs with their physical PDF pages, and each page's render/analysis progress.
The vision provider must expose an OpenAI-compatible image-message API; verify
that the configured model actually supports image input.

## Pipeline

`extracted_text/full_text.txt` keeps the original PDF text, and
`extracted_text/pages/page_XXXX.txt` keeps the same text split by physical PDF
page (page numbers are 1-based). When vision is enabled, figure candidates and
vision evidence record this physical page as `pdf_page`; the console reports
the same mapping before rendering each page. In auto mode, an LLM classifies
the target scheme from the first 2,000 characters after the trailing
references section is trimmed. Ambiguous classifications stop that paper; use
`--scheme` to select a profile explicitly. Innovation extraction then uses the
selected profile and the trimmed paper text.

Every run extracts exactly one modular-arithmetic innovation and generates the fixed-interface `modmul` module. Vivado, when enabled, synthesizes that generated module as the top-level design.

Each run writes `run_summary.json` with stage status, timing, LLM metadata, and
FPGA resources. The console prints the selected provider and model immediately
after the LLM client is initialized. The `llm` object records the provider,
default model, extraction/generation models, and `by_operation` mappings for
`classify_scheme`, `extract_innovations`, `classify_figures`,
`analyze_figure_page`, `generate_modmul`, `fix_syntax`, and `fix_function`.
The `llm.vision` object records the independent vision provider and model. The
`timing` object reports total pipeline time, total LLM time/calls, and LLM time
by operation. The top-level `resources` object reports
`LUT`, `FF`, `DSP`, and `BRAM`; values are `null` when synthesis is skipped or
unavailable. `passed`, `failed`, `skipped`, `unverified`, and `unavailable` are
kept distinct. Vivado is optional for generation and simulation, but resource
results are unavailable without it.

## Offline smoke check

```bash
python scripts/smoke_check.py
```

This checks configuration loading, scheme and FPGA-part resolution, JSON extraction, and the fixed modmul interface without calling an LLM.
