# llm-tokens-atlas

> Open benchmark of LLM tokenization across 5 providers — offline vs empirical calibration deltas.

## What this is

`llm-tokens-atlas` is a reproducible, open dataset and analysis pipeline that measures how offline tokenizers (e.g. tiktoken proxies, the published BPE vocabularies) compare against the **empirical** token counts returned by each provider's own API. It covers 5 providers, 5 prompt formats (Markdown, XML, JSON, YAML, plain text), and thousands of real-world prompts drawn from open corpora.

The output is a per-provider, per-format **calibration delta distribution** — so anyone estimating cost or context-window budgets ahead of an API call can quantify the bias of their offline counter instead of treating it as exact.

This project builds on [tokenometer](https://github.com/faraa2m/tokenometer), which surfaced the underlying methodology and a notable preliminary finding: cl100k_base underestimates `claude-opus-4-7` tokens by ~62% (median).

## Status

Early / pre-release. Schema, scripts, and data are under active development. Expect breaking changes until v0.1.0.

## Install

TBD — see `pyproject.toml`. The recommended workflow uses [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

## Usage

TBD. See `scripts/` for collection and counting drivers, and `analysis/notebooks/` for plots.

## Reproducing results

```bash
make reproduce
```

This regenerates the dataset from scratch. Tokenizer and provider API versions are pinned (see `data/lockfile.json` once published).

See [`docs/REPRODUCING.md`](./docs/REPRODUCING.md) for full instructions — required API keys per provider, expected runtime at each scale, output sizes, and a CI-friendly tiny variant (`make reproduce-tiny`).

## Tokenometer integration

Atlas reuses [`tokenometer`](https://github.com/faraa2m/tokenometer)'s
5-provider tokenizer logic instead of reimplementing it in Python. The
integration lives in a single module:

- **[`scripts/tokenometer_bridge.py`](./scripts/tokenometer_bridge.py)** —
  Python facade over the tokenometer CLI. Exposes `count_offline`,
  `count_empirical`, `list_providers`, `list_models`, `list_formats`,
  plus a `count_offline_batch` / `count_empirical_batch` pair for the
  high-throughput atlas pipeline.
- **[`scripts/install_tokenometer.sh`](./scripts/install_tokenometer.sh)** —
  idempotent installer; `make install` runs it. Finds tokenometer via (1)
  `tokenometer` on PATH, (2) a sibling `../tokenometer/` repo build, (3)
  builds the sibling if source is present, or (4) fails with an install
  hint.
- **[`scripts/_tokenometer_bridge_design.md`](./scripts/_tokenometer_bridge_design.md)** —
  design note: subprocess CLI vs HTTP bridge trade-off, why we chose
  subprocess.

Any new Python code that needs token counts should import from
`scripts.tokenometer_bridge`. Do not invoke the tokenometer CLI directly
from other scripts.

## Publishing the dataset

The canonical home for the dataset is
<https://huggingface.co/datasets/faraa2m/llm-tokens-atlas>. The Hugging Face
dataset card lives at [`data/README.md`](./data/README.md). The upload
script is [`scripts/publish_to_hf.py`](./scripts/publish_to_hf.py); set
`HF_TOKEN` in your env and run it with `--dataset llm-tokens-atlas`.

## Reproducing

- [`docs/REPRODUCING.md`](./docs/REPRODUCING.md) — `make reproduce`
  mechanics + expected runtime.

## Citation

Until the paper is on arxiv, cite the GitHub repo and the HuggingFace
dataset directly:

```bibtex
@misc{llm-tokens-atlas-2026,
  author       = {Faraazuddin Mohammed},
  title        = {{llm-tokens-atlas}: An Open Benchmark of LLM Tokenization Calibration},
  year         = {2026},
  howpublished = {\url{https://github.com/faraa2m/llm-tokens-atlas}},
  note         = {Companion arxiv preprint forthcoming}
}
```

## License

- Code: [Apache-2.0](./LICENSE)
- Data (everything under `data/`): [CC-BY-4.0](./LICENSE-DATA)
