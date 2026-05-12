# llm-tokens-atlas — reproducible dataset pipeline.
#
# Targets are single-purpose, idempotent where possible, and print what they
# are doing. Override the prompt count via `make reproduce N=10000`.
#
# Pipeline order (encoded in `reproduce`):
#   corpus  -> offline counts -> empirical counts -> build dataset -> lockfile
#
# Sibling scripts owned by other pipeline modules:
#   llm_tokens_atlas/collect_corpus.py      (atlas-corpus)
#   llm_tokens_atlas/count_offline.py       (atlas-offline)
#   llm_tokens_atlas/count_empirical.py     (atlas-empirical)
#   llm_tokens_atlas/build_dataset.py       (atlas-schema)
#   llm_tokens_atlas/lockfile.py            (atlas-schema)
#
# This Makefile only chains them; it does not implement them.

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

# Tunable knobs ------------------------------------------------------------
# Default prompt count for the full reproduce target. Override on the
# command line, e.g. `make reproduce N=10000`. CI uses `reproduce-tiny`
# which fixes N=5.
N ?= 5000

# Provider set for empirical counting. Comma-separated; default covers the
# providers we can call without API keys in CI (OpenAI via tiktoken,
# Mistral via the OSS tokenizer). Override locally for the full run, e.g.
# `make empirical PROVIDERS=anthropic,google,openai,mistral,cohere`.
PROVIDERS ?= openai,mistral,anthropic,google,cohere

# Output paths (kept in one place so `clean` and `reproduce` stay in sync).
DATA_DIR        := data
RAW_PROMPTS     := $(DATA_DIR)/raw_prompts.jsonl
OFFLINE_COUNTS  := $(DATA_DIR)/offline_counts.jsonl
EMPIRICAL_COUNTS:= $(DATA_DIR)/empirical_counts.jsonl
PROCESSED_DIR   := $(DATA_DIR)/processed
DATASET_PARQUET := $(PROCESSED_DIR)/atlas.parquet
LOCKFILE        := $(DATA_DIR)/lockfile.json

# Phony targets (no real file outputs at the make-target level; scripts own
# their own file outputs so make doesn't need to track them as prereqs).
.PHONY: help install lint test corpus offline empirical build lockfile \
        reproduce reproduce-tiny clean

help: ## Print available targets
	@echo "llm-tokens-atlas — make targets"
	@echo ""
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / { printf "  %-16s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@echo ""
	@echo "Knobs: N=$(N)  PROVIDERS=$(PROVIDERS)"

install: ## Install Python dependencies via uv + locate tokenometer CLI
	@echo ">> Installing dependencies (uv sync --all-extras)"
	uv sync --all-extras
	@echo ">> Locating tokenometer CLI (llm_tokens_atlas/install_tokenometer.sh)"
	bash llm_tokens_atlas/install_tokenometer.sh

lint: ## Run ruff + mypy
	@echo ">> Linting (ruff + mypy)"
	uv run ruff check .
	uv run mypy llm_tokens_atlas/

test: ## Run pytest
	@echo ">> Running tests (pytest)"
	uv run pytest -v

corpus: ## Sample prompts from open corpora into raw_prompts.jsonl
	@echo ">> Collecting corpus (N=$(N) -> $(RAW_PROMPTS))"
	@mkdir -p $(DATA_DIR)
	uv run python llm_tokens_atlas/collect_corpus.py --n $(N) --out $(RAW_PROMPTS)

offline: ## Compute offline token counts via @tokenometer/core
	@echo ">> Offline counting ($(RAW_PROMPTS) -> $(OFFLINE_COUNTS))"
	uv run python llm_tokens_atlas/count_offline.py --in $(RAW_PROMPTS) --out $(OFFLINE_COUNTS)

empirical: ## Call each provider's empirical token-count endpoint
	@echo ">> Empirical counting (providers=$(PROVIDERS), $(RAW_PROMPTS) -> $(EMPIRICAL_COUNTS))"
	uv run python llm_tokens_atlas/count_empirical.py \
		--in $(RAW_PROMPTS) \
		--out $(EMPIRICAL_COUNTS) \
		--providers $(PROVIDERS)

build: ## Merge offline+empirical counts into the published parquet dataset
	@echo ">> Building dataset ($(DATASET_PARQUET))"
	@mkdir -p $(PROCESSED_DIR)
	uv run python llm_tokens_atlas/build_dataset.py \
		--raw $(RAW_PROMPTS) \
		--offline $(OFFLINE_COUNTS) \
		--empirical $(EMPIRICAL_COUNTS) \
		--out $(DATASET_PARQUET)

lockfile: ## Snapshot tokenizer + provider API versions to data/lockfile.json
	@echo ">> Writing lockfile ($(LOCKFILE))"
	uv run python llm_tokens_atlas/lockfile.py --out $(LOCKFILE)

reproduce: install corpus offline empirical build lockfile ## Full pipeline (override with N=...)
	@echo ">> Reproduce complete. Dataset: $(DATASET_PARQUET)"

reproduce-tiny: ## CI smoke variant: N=5, key-free providers only
	@$(MAKE) reproduce N=5 PROVIDERS=openai,mistral

clean: ## Remove generated artifacts (data + caches)
	@echo ">> Cleaning generated artifacts"
	rm -f $(RAW_PROMPTS) $(OFFLINE_COUNTS) $(EMPIRICAL_COUNTS) $(LOCKFILE)
	rm -rf $(PROCESSED_DIR) $(DATA_DIR)/figures/ \
		.pytest_cache/ .ruff_cache/ .mypy_cache/
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
