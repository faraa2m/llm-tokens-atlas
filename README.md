# llm-tokens-atlas

> Open benchmark of LLM tokenization — offline vs empirical calibration deltas. v0.1.0 ships 3 providers (Anthropic, OpenAI, Mistral); Google + Cohere pending for v0.2.0.

## What this is

`llm-tokens-atlas` is a reproducible, open dataset and analysis pipeline that measures how offline tokenizers (e.g. tiktoken proxies, the published BPE vocabularies) compare against the **empirical** token counts returned by each provider's own API or OSS tokenizer. **v0.1.0 covers 3 providers** (Anthropic `claude-opus-4-7`, OpenAI `gpt-4o`, Mistral `mistral-large-latest`), 5 prompt formats (Markdown, XML, JSON, YAML, plain text), and 7,485 real-world prompt requests (499 unique prompts × 5 formats × 3 providers; n=2,495 per provider) drawn from open corpora. Google `gemini-2.5-pro` and Cohere `command-r` parity sweeps are scheduled for v0.2.0 — the schema is already validated for both providers; the only missing piece is the empirical sweep.

The output is a per-provider, per-format **calibration delta distribution** — so anyone estimating cost or context-window budgets ahead of an API call can quantify the bias of their offline counter instead of treating it as exact.

This project builds on [tokenometer](https://github.com/faraa2m/tokenometer), which surfaced the underlying methodology and a notable preliminary finding: cl100k_base underestimates `claude-opus-4-7` tokens by ~62% (median).

## Status

v0.1.0 (released 2026-05-11) — 3-provider coverage (Anthropic + OpenAI + Mistral). Schema, drivers, and data are stable for the three shipped providers; Google + Cohere rows will be added in v0.2.0 without a breaking schema change.

## Headline findings (v0.1.0)

Released 2026-05-11. n = 7,485 rows (2,495 per provider). Detailed numbers live in [`analysis/results.json`](./analysis/results.json).

| Provider | Model | Median offline-vs-empirical delta | OLS slope | R² |
|---|---|---|---|---|
| anthropic | `claude-opus-4-7` | **+41.3%** (cl100k_base underestimates) | 1.611 | 0.9956 |
| openai | `gpt-4o` | 0.0% (tiktoken-as-truth oracle, mean +3.0%) | 1.024 | 0.9986 |
| mistral | `mistral-large-latest` | −0.1% (mistral-tokenizer-js, mean +1.9%) | 1.016 | 0.9993 |

The Anthropic row is the headline: the publicly-recommended offline tokenizer
underestimates real `claude-opus-4-7` cost by ~41% across thousands of prompts,
and 100% of rows underestimate (no exact / overestimate cases). OpenAI and
Mistral are baselines confirming the offline-vs-empirical pipeline is calibrated
correctly when the provider's own tokenizer is the oracle.

## Install

TBD — see `pyproject.toml`. The recommended workflow uses [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

## Usage

TBD. See `llm_tokens_atlas/` for collection and counting drivers, and `analysis/notebooks/` for plots.

## Reproducing results

```bash
make reproduce
```

This regenerates the dataset from scratch. Tokenizer and provider API versions are pinned (see `data/lockfile.json` once published).

See [`docs/REPRODUCING.md`](./docs/REPRODUCING.md) for full instructions — required API keys per provider, expected runtime at each scale, output sizes, and a CI-friendly tiny variant (`make reproduce-tiny`).

## Tokenometer integration

Atlas reuses [`tokenometer`](https://github.com/faraa2m/tokenometer)'s
multi-provider tokenizer logic (5 providers supported upstream; Atlas v0.1.0
exercises 3 of them — Anthropic, OpenAI, Mistral — with Google + Cohere
exercises arriving in v0.2.0) instead of reimplementing it in Python. The
integration lives in a single module:

- **[`llm_tokens_atlas/tokenometer_bridge.py`](./llm_tokens_atlas/tokenometer_bridge.py)** —
  Python facade over the tokenometer CLI. Exposes `count_offline`,
  `count_empirical`, `list_providers`, `list_models`, `list_formats`,
  plus a `count_offline_batch` / `count_empirical_batch` pair for the
  high-throughput atlas pipeline.
- **[`llm_tokens_atlas/install_tokenometer.sh`](./llm_tokens_atlas/install_tokenometer.sh)** —
  idempotent installer; `make install` runs it. Finds tokenometer via (1)
  `tokenometer` on PATH, (2) a sibling `../tokenometer/` repo build, (3)
  builds the sibling if source is present, or (4) fails with an install
  hint.

Any new Python code that needs token counts should import from
`llm_tokens_atlas.tokenometer_bridge`. Do not invoke the tokenometer CLI
directly from other modules.

## Publishing the dataset

The canonical home for the dataset is
<https://huggingface.co/datasets/faraa2m/llm-tokens-atlas>. The Hugging Face
dataset card lives at [`data/README.md`](./data/README.md). The upload
script is [`llm_tokens_atlas/publish_to_hf.py`](./llm_tokens_atlas/publish_to_hf.py); set
`HF_TOKEN` in your env and run it with `--dataset llm-tokens-atlas`.

## Reproducing

- [`docs/REPRODUCING.md`](./docs/REPRODUCING.md) — `make reproduce`
  mechanics + expected runtime.

## Citation

Released 2026-05-11. Cite as **v0.1.0** (3-provider coverage). Coverage will
expand to 5 providers in v0.2.0 (Google + Cohere); cite the version you used.
Until the paper is on arxiv, cite the GitHub repo and the HuggingFace
dataset directly:

```bibtex
@misc{llm-tokens-atlas-2026,
  author       = {Faraazuddin Mohammed},
  title        = {{llm-tokens-atlas}: An Open Benchmark of LLM Tokenization Calibration},
  year         = {2026},
  version      = {v0.1.0},
  howpublished = {\url{https://github.com/faraa2m/llm-tokens-atlas}},
  note         = {3-provider coverage (Anthropic, OpenAI, Mistral); v0.2.0 adds Google + Cohere. Companion arxiv preprint forthcoming.}
}
```

## License

- Code: [Apache-2.0](./LICENSE)
- Data (everything under `data/`): [CC-BY-4.0](./LICENSE-DATA)
