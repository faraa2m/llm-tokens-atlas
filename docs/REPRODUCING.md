# Reproducing `llm-tokens-atlas`

This guide walks through regenerating the published dataset on a clean machine.
The pipeline is deterministic given the same input corpus, tokenizer versions,
and provider API versions (snapshotted in `data/lockfile.json`).

## Prerequisites

- **Python** 3.11
- [**uv**](https://github.com/astral-sh/uv) (latest)
- **GNU make** (preinstalled on macOS / Linux)
- ~5 GB free disk for HuggingFace dataset caches
- Internet access for HuggingFace dataset downloads and provider API calls

## API keys

The full empirical-counting pass talks to five providers. Set these env vars
before `make reproduce`; missing keys cause that provider to be skipped with a
warning rather than a hard failure.

| Provider  | Env var               | Cost     | Required for full run |
|-----------|-----------------------|----------|------------------------|
| OpenAI    | _none_ (uses `tiktoken` locally) | free | no, but recommended |
| Mistral   | _none_ (uses OSS tokenizer)      | free | no, but recommended |
| Anthropic | `ANTHROPIC_API_KEY`   | free tier (`messages.countTokens`) | yes |
| Google    | `GOOGLE_API_KEY`      | free tier (`model.countTokens`)    | yes |
| Cohere    | `COHERE_API_KEY`      | free tier (`tokenize`)             | yes |

> The CI smoke job (`reproduce-tiny`) runs only the credential-free providers
> (`openai,mistral`) so the workflow stays green without secrets.

## One-shot reproduction

```bash
git clone https://github.com/faraa2m/llm-tokens-atlas.git
cd llm-tokens-atlas
export ANTHROPIC_API_KEY=...
export GOOGLE_API_KEY=...
export COHERE_API_KEY=...
make reproduce          # default N=5000 prompts
```

To regenerate at the published 10k-prompt scale:

```bash
make reproduce N=10000
```

To run only a subset of providers (e.g. you have no Cohere key):

```bash
make reproduce PROVIDERS=openai,mistral,anthropic,google
```

## What each target does

| Target          | Output                                | Notes |
|-----------------|---------------------------------------|-------|
| `install`       | `.venv/`                              | `uv sync --all-extras` |
| `corpus`        | `data/raw_prompts.jsonl`              | Samples from HumanEval, LMSYS-chat-1M (open subset), MT-bench, multilingual Wikipedia, GitHub READMEs, filtered ShareGPT |
| `offline`       | `data/offline_counts.jsonl`           | Calls `@tokenometer/core` offline counters across all 5 providers |
| `empirical`     | `data/empirical_counts.jsonl`         | Calls each provider's free tokenize endpoint (subset controlled by `PROVIDERS=`) |
| `build`         | `data/processed/atlas.parquet`        | Joins raw + offline + empirical; one row per (prompt × provider × format × model) |
| `lockfile`      | `data/lockfile.json`                  | Tokenizer + provider API versions captured for reproducibility |
| `reproduce`     | all of the above, in order            | The published artifact pipeline |
| `reproduce-tiny`| `N=5`, key-free providers only        | CI smoke variant |

## Expected runtime

| Scale            | Wall time            | Bottleneck |
|------------------|----------------------|-----------|
| `reproduce-tiny` (N=5)   | < 2 min       | uv sync + HF cache warm-up |
| `reproduce` (N=5000)     | ~30-60 min    | Provider rate limits on empirical counting |
| `reproduce N=10000`      | ~60-120 min   | Provider rate limits + parquet build |

The pipeline is checkpointed at each JSONL stage, so re-running after a
network blip resumes rather than starting over.

## Expected output size

| Artifact                     | Size (N=5000)   |
|------------------------------|-----------------|
| `data/raw_prompts.jsonl`     | ~5-8 MB         |
| `data/offline_counts.jsonl`  | ~3-5 MB         |
| `data/empirical_counts.jsonl`| ~3-5 MB         |
| `data/processed/atlas.parquet` | ~2-4 MB (compressed) |
| `data/lockfile.json`         | ~2-5 KB         |

`data/processed/atlas.parquet` is the **published dataset artifact** (mirrored
to HuggingFace; see `data/README.md` for the dataset card).

## Troubleshooting

- **`uv: command not found`** — install from <https://github.com/astral-sh/uv>.
- **Rate-limit errors during `make empirical`** — re-run; checkpoints are
  resumable. Or restrict providers via `PROVIDERS=...`.
- **`HF dataset download stalls`** — pre-warm the cache:
  `uv run python -c "from datasets import load_dataset; load_dataset('openai/humaneval')"`.
- **Dataset row count differs across runs** — confirm `ATLAS_SEED=42` is set
  (the corpus collector uses it for deterministic sampling).

## See also

- [`README.md`](../README.md) — project overview.
- [`data/README.md`](../data/README.md) — HuggingFace dataset card.
- [`data/lockfile.json`](../data/lockfile.json) — pinned tokenizer + API versions.
