# Paper2Gate

Paper2Gate extracts modular-arithmetic hardware ideas from PQC papers and generates a fixed-interface Verilog `modmul` implementation.

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

The project always reads the repository-root `config.yaml`. Machine-specific values can be supplied through environment variables:

```text
DEEPSEEK_API_KEY
OPENAI_API_KEY
ANTHROPIC_API_KEY
PAPER2GATE_IVERILOG_BINARY
PAPER2GATE_VIVADO_BINARY
```

Empty tool paths in `config.yaml` mean that the executable is searched on `PATH`.

## Pipeline

`extracted_text/full_text.txt` keeps the original PDF text. In auto mode, an LLM classifies the target scheme from the first 2,000 characters after the trailing references section is trimmed. Ambiguous classifications stop that paper; use `--scheme` to select a profile explicitly. Innovation extraction then uses the selected profile and the trimmed paper text.

Every run extracts exactly one modular-arithmetic innovation and generates the fixed-interface `modmul` module. Vivado, when enabled, synthesizes that generated module as the top-level design.

Each run writes `run_summary.json` with stage status and timing. The `timing` object reports total pipeline time, total LLM time/calls, and LLM time by operation (`classify_scheme`, `extract_innovations`, `generate_modmul`, `fix_syntax`, and `fix_function`). `passed`, `failed`, `skipped`, `unverified`, and `unavailable` are kept distinct. Vivado is optional for generation and simulation, but resource results are unavailable without it.

## Offline smoke check

```bash
python scripts/smoke_check.py
```

This checks configuration loading, scheme and FPGA-part resolution, JSON extraction, and the fixed modmul interface without calling an LLM.
