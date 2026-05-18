---
language:
- en
license: cc-by-4.0
pretty_name: LLM Tokens Atlas
size_categories:
- 10K<n<100K
task_categories:
- other
tags:
- llm
- tokenization
- benchmark
- calibration
- cost-estimation
annotations_creators:
- machine-generated
language_creators:
- found
multilinguality:
- monolingual
source_datasets:
- extended|lmsys/lmsys-chat-1m
- extended|openai/openai_humaneval
- extended|wikimedia/wikipedia
configs:
- config_name: default
  data_files:
  - split: train
    path: data/processed/atlas.parquet
---

# Dataset Card for LLM Tokens Atlas

`llm-tokens-atlas` is an open, reproducible benchmark of LLM tokenization across
**3 providers in v0.1.0** (Anthropic `claude-opus-4-7`, OpenAI `gpt-4o`, Mistral
`mistral-large-latest`) across **5 prompt formats** (Markdown, XML, JSON, YAML,
Plain text), evaluated on 7,485 real-world prompt requests (n=2,495 per provider,
499 unique prompts × 5 formats × 3 providers). Google + Cohere parity sweeps are
scheduled for v0.2.0. For each `(prompt, provider, model, format)` cell we record
both an *offline* token count (from the provider's published or
community-reverse-engineered tokenizer) and an *empirical* token count (from the
provider's authoritative count-tokens API endpoint, where one exists).

The artifact is the **calibration delta distribution** between offline and
empirical counts — per provider, per model, per format. It quantifies how wrong
the cheap-to-run offline counter is, so anyone estimating cost or
context-window budgets ahead of a real API call can correct for the bias rather
than treat it as exact.

- Homepage / source code: <https://github.com/faraa2m/llm-tokens-atlas>
- License (data): [CC-BY-4.0](https://github.com/faraa2m/llm-tokens-atlas/blob/main/LICENSE-DATA)
- License (code): [Apache-2.0](https://github.com/faraa2m/llm-tokens-atlas/blob/main/LICENSE)

## Coverage / What's in this release

| Field | Value |
|---|---|
| Version | v0.1.0 (3-provider) |
| Release date | 2026-05-11 |
| Total rows | 7,485 |
| Unique prompts | 499 |
| Providers shipped | 3 (Anthropic, OpenAI, Mistral) |
| Providers pending | 2 (Google, Cohere) — schema-reserved, gated on v0.2.0 sweeps |
| Formats | 5 (plain, markdown, json, xml, yaml) |
| Domains | 3 (code, prose, chat) |

**Per-provider breakdown:**

| Provider | Model | n | Empirical source |
|---|---|---|---|
| anthropic | `claude-opus-4-7` | 2,495 | `messages.countTokens` (Anthropic API) |
| openai | `gpt-4o` | 2,495 | `tiktoken` `o200k_base` (tiktoken-as-truth, local) |
| mistral | `mistral-large-latest` | 2,495 | `mistral-tokenizer-js` (vendor OSS tokenizer) |

**Per-format breakdown** (rows are split evenly across formats; each format gets
1,497 rows total = 499 prompts × 3 providers):

| Format | n total | n per provider |
|---|---|---|
| plain | 1,497 | 499 |
| markdown | 1,497 | 499 |
| json | 1,497 | 499 |
| xml | 1,497 | 499 |
| yaml | 1,497 | 499 |

**Pending for v0.2.0:**

- Google `gemini-2.5-pro` via `model.countTokens` — gated on `GOOGLE_API_KEY` sweep.
- Cohere `command-r` via `POST /v1/tokenize` — gated on `COHERE_API_KEY` sweep.

Both providers are reserved in the schema and validated end-to-end; the only
missing piece is the empirical sweep. Rows for these providers will land in
v0.2.0 without a major-version schema bump.

**Headline empirical finding (v0.1.0):**

- Anthropic `claude-opus-4-7`: cl100k_base offline tokenizer **underestimates**
  empirical counts by **41.3% median** (p25=36.8%, p75=46.7%, p95=58.6%; n=2,495).
  OLS calibration fit: slope = 1.611, intercept = 18.20, R² = 0.9956.
  100% of rows underestimate (no exact or overestimate cases).
- OpenAI `gpt-4o`: offline `o200k_base` is treated as oracle (tiktoken-as-truth);
  median delta = 0.0%, mean = 3.01%; calibration fit slope = 1.024, R² = 0.9986;
  57.1% exact, 42.4% underestimate, 0.5% overestimate.
- Mistral `mistral-large-latest`: median delta = −0.06% (mistral-tokenizer-js
  slightly overestimates by construction), mean = 1.85%; slope = 1.016, R² = 0.9993;
  42.1% underestimate, 57.8% overestimate, ~0% exact.

Per-format × per-provider median deltas and full statistics live in
`analysis/results.json` (the single source of truth for v0.1.0 numbers).

## Dataset Summary

LLM Tokens Atlas measures the gap between what an offline tokenizer thinks a
prompt costs and what a provider's API actually reports. Each row pairs a
single prompt rendered in one of five surface formats with one provider model,
records both counts, and computes the absolute and relative delta. Aggregated
across providers and formats, the result is a calibrated joint distribution of
offline-vs-empirical bias that downstream tools (cost estimators, routers,
context budgeters, prompt optimizers) can plug in instead of guessing.

The dataset is intentionally narrow in task scope — it is a *measurement
artifact*, not a training corpus. The prompts are drawn from already-public,
redistribution-friendly corpora (LMSYS-Chat-1M sample, HumanEval, MT-Bench,
GitHub READMEs, multilingual Wikipedia snippets). No model outputs are
collected; only token counts.

## Supported Tasks and Leaderboards

This dataset does not target a downstream supervised task. It is a
**measurement benchmark** intended for:

- Calibrating client-side / offline tokenizers against provider APIs.
- Benchmarking cost-estimation libraries (e.g. `tokencost`, `tokenometer`).
- Studying tokenizer drift over time as providers update their tokenizers.
- Auditing context-window budgeting tools.
- Grounding cost-aware routing systems in calibrated empirical token counts
  rather than offline proxies multiplied by published pricing.

`task_categories: [other]` reflects that this is a measurement / calibration
artifact rather than a classical NLP supervised task.

## Languages

The dataset's prompt text is overwhelmingly **English** (the LMSYS-Chat-1M
sample and HumanEval contributions are English; multilingual Wikipedia
snippets, when included, are tagged separately in the `language` field). For
calibration purposes the underlying language matters less than the format and
provider, but multilingual coverage is limited and explicitly so. See
[Limitations](#limitations).

## Dataset Structure

### Data Files

A single Parquet file at `data/processed/atlas.parquet` is the canonical
artifact. It is produced by `llm_tokens_atlas/build_dataset.py`, which inner-joins
three intermediate JSONL streams — `data/raw_prompts.jsonl`,
`data/offline_counts.jsonl`, and `data/empirical_counts.jsonl` — on the
composite key `(prompt_id, provider, format, model)`, then attaches
prompt-level columns and computes calibration deltas. The full row-level
schema for each intermediate stream is published as JSON Schema at
`data/schema.json`; that file is the formal source of truth and should be
preferred over this narrative description when they disagree.

### Data Fields

The processed Parquet has the following columns:

**Prompt-level (attached from `raw_prompts.jsonl` via `prompt_id` join):**

| Field             | Type     | Description |
|-------------------|----------|-------------|
| `prompt_id`       | string   | Stable identifier for the underlying prompt. Same prompt → same `prompt_id` across all `(provider, model, format)` rows; this is the join key downstream consumers should split on for held-out evaluation. |
| `source`          | string   | Origin corpus: `lmsys-chat-1m`, `humaneval`, `mt-bench`, `github-readmes`, `wikipedia-multilingual`, or similar (see `data/provenance.md` for the authoritative list). |
| `text`            | string   | The prompt text exactly as sent to the tokenizer / API. UTF-8. |
| `text_len_chars`  | int64    | Length of `text` in Unicode codepoints (not bytes). |
| `text_len_words`  | int64    | Whitespace-split word count of `text`. Approximate — for filtering / grouping, not as a tokenization signal. |
| `language`        | string   | ISO 639-1 / BCP-47 code: `en`, `zh`, `es`, `fr`, `de`, `ja`, `multi`, `code`, etc. `code` indicates source code; `multi` indicates mixed-language. |
| `domain`          | string   | High-level domain tag for stratified analysis: `code`, `prose`, `chat`, `structured`, `multilingual`, `other`. |
| `collected_at`    | string (ISO-8601) | UTC timestamp at which the prompt was collected into the corpus. |

**Cell key:**

| Field      | Type   | Description |
|------------|--------|-------------|
| `provider` | string | One of: `anthropic`, `openai`, `mistral` (shipped in v0.1.0). Schema-reserved future values: `google`, `cohere` (rows land in v0.2.0). |
| `format`   | string | Surface format the prompt is rendered in. One of: `plain`, `markdown`, `xml`, `json`, `yaml`. |
| `model`    | string | Concrete model identifier evaluated. v0.1.0 ships three: `claude-opus-4-7`, `gpt-4o`, `mistral-large-latest`. Schema also reserves `gemini-2.5-pro` (Google) and `command-r` (Cohere) for v0.2.0. |

**Offline counts (from `offline_counts.jsonl`):**

| Field               | Type     | Description |
|---------------------|----------|-------------|
| `offline_count`     | int64    | Token count from the offline tokenizer (tiktoken proxy, published BPE vocab, community-reverse-engineered tokenizer, or `mistral-common`). |
| `tokenizer_version` | string   | Pinned tokenizer version identifier (e.g. `tiktoken@cl100k_base`, `@tokenometer/core@1.0.0`, `mistral-common@1.7.0`). Pinned for reproducibility in `data/lockfile.json`. |
| `offline_ts`        | string (ISO-8601) | UTC timestamp at which the offline count was produced. |

**Empirical counts (from `empirical_counts.jsonl`):**

| Field              | Type     | Description |
|--------------------|----------|-------------|
| `empirical_count`  | int64    | Token count from the provider's authoritative count-tokens endpoint (or tiktoken-as-truth for OpenAI, where it is treated as the oracle). |
| `is_oracle`        | bool     | True when this value is the ground-truth oracle for the provider (e.g. tiktoken for OpenAI, provider countTokens API for Anthropic / Google). False when it is empirical but not official (e.g. inferred from stream usage). |
| `empirical_source` | string   | How the empirical count was obtained: `api` (HTTP call), `tiktoken` (local oracle), `sdk` (vendor SDK helper), or `stream-usage` (inferred from generation metadata). |
| `endpoint`         | string   | Concrete endpoint or library identifier (e.g. `https://api.anthropic.com/v1/messages/count_tokens@2024-10-22`, `tiktoken.encoding_for_model(gpt-4o)`). |
| `empirical_ts`     | string (ISO-8601) | UTC timestamp at which the empirical count was produced. |

**Computed calibration columns (added at build time):**

| Field        | Type     | Description |
|--------------|----------|-------------|
| `delta`      | int64    | `empirical_count − offline_count`. Positive ⇒ offline underestimates the real count; negative ⇒ overestimates; zero ⇒ exact agreement. |
| `abs_delta`  | int64    | `abs(delta)`. Useful for symmetric error metrics. |
| `delta_pct`  | float64  | `delta / empirical_count × 100`. The headline per-row calibration error, in percent. NaN where `empirical_count == 0`. |
| `direction`  | string   | Categorical mapping of `sign(delta)`: `underestimate`, `overestimate`, or `exact`. |

### Data Splits

The dataset is published as a single `train` split. There is no held-out
evaluation set — this is a calibration / measurement artifact, not a supervised
task. Downstream users who need a held-out set should split by `prompt_id` so
that the same prompt rendered in different formats does not leak across splits.

## Dataset Creation

### Curation Rationale

A handful of independent observations in 2026 — most prominently two blog posts
documenting that Anthropic's updated tokenizer inflates token counts by
~40–47% on real prompts — surfaced the same underlying phenomenon: deployed
offline tokenizers consistently disagree with provider APIs, and the disagreement
is large enough to matter for cost, context-budgeting, and routing decisions.
What has been missing is a *peer-citable, openly distributed dataset* that
quantifies this drift rigorously across providers, formats, and time. LLM
Tokens Atlas fills that gap: instead of a single-point anecdote, it publishes
the joint distribution of offline-vs-empirical deltas with full reproducibility
metadata. The cited prior observations are credited in
[Citation Information / Related Work](#related-work) below.

### Source Data

#### Initial Data Collection and Normalization

Prompts are sampled from already-public, redistribution-friendly corpora and
re-rendered into each of the five surface formats. Per-row provenance — which
corpus a prompt originated from, what license it ships under, and what
normalization steps were applied — is documented in `data/provenance.md`,
maintained by the sibling `atlas-corpus` collector and treated as the
authoritative record for downstream attribution. Source corpora include:

- **LMSYS-Chat-1M** (open sample) — real chat prompts.
- **HumanEval** — programming task descriptions.
- **MT-Bench** — multi-turn evaluation prompts.
- **GitHub README snippets** — long-form, structured technical prose.
- **Multilingual Wikipedia** — for non-English coverage where applicable.
- **ShareGPT (filtered, open subset)** — additional chat coverage.

Prompts are normalized into a canonical text representation, then deterministically
re-rendered into each of the five formats so that `format` is a controlled
independent variable rather than a property of the source corpus.

#### Who are the source language producers?

The prompt text comes from public chat logs, open-source code documentation,
benchmark authors, and Wikipedia contributors. No identifying information is
collected, retained, or republished beyond what is already in the source
corpora; per-source licensing terms are preserved as required.

### Annotations

#### Annotation Process

There are no human annotations. Each row records two *machine-measured* token
counts:

1. **Offline count** — computed locally by invoking the offline tokenizer
   shipped or reverse-engineered for the target provider/model. The exact
   tokenizer (with version pin) is recorded in `tokenizer_id` and in
   `data/lockfile.json`.
2. **Empirical count** — obtained by calling the provider's authoritative
   token-counting endpoint or OSS tokenizer. The API version used is recorded
   in `api_version` and in `data/lockfile.json`.

**Empirical-source coverage in v0.1.0:**

- **Anthropic** — `messages.countTokens` (HTTP, official API). Network sweep
  executed; empirical counts are authoritative.
- **OpenAI** — `tiktoken.encoding_for_model("gpt-4o")` with `o200k_base`
  vocab, run locally. Treated as the oracle (tiktoken-as-truth) since no
  separate count-tokens HTTP endpoint exists.
- **Mistral** — `mistral-tokenizer-js` (vendor OSS tokenizer), run locally.

**Not executed in v0.1.0 (schema-validated only, populate in v0.2.0):**

- **Google** — `model.countTokens` HTTP endpoint integration is validated
  against the schema but the empirical sweep has not been run; no Google
  rows are present in `data/processed/atlas.parquet` in v0.1.0.
- **Cohere** — `POST /v1/tokenize` integration is similarly validated but
  unrun; no Cohere rows are present in v0.1.0.

Both Google and Cohere sweeps land in v0.2.0 alongside their API keys
(`GOOGLE_API_KEY`, `COHERE_API_KEY`); the row schema is unchanged.

#### Who are the annotators?

Not applicable — counts are produced by deterministic code, not annotators.

### Versioning Policy

Releases use semantic versioning (`MAJOR.MINOR.PATCH`). Each release pins the
offline tokenizer versions and provider API versions in
`data/lockfile.json`. Because providers update their tokenizers periodically
(see [Considerations](#considerations-for-using-the-data)), a row's
`tokenizer_id` and `api_version` should be treated as part of its identity for
longitudinal analysis. Major-version bumps may break row schemas; minor and
patch versions preserve the published schema.

### Personal and Sensitive Information

The dataset does not contain new personal information beyond what is already
present in the source corpora. The LMSYS-Chat-1M and ShareGPT-derived samples
inherit upstream filtering. We do not collect, store, or republish any
provider-side outputs other than integer token counts.

## Considerations for Using the Data

### Social Impact of Dataset

The intended impact is positive: more accurate cost estimation reduces wasted
spend on LLM APIs, particularly for resource-constrained teams that rely on
free or low-tier offline tokenizers. Better calibration also reduces the
incentive for vendors to obscure tokenizer behavior, since any drift is now
publicly measurable.

### Discussion of Biases

- **English-heavy.** Prompts disproportionately reflect English LMSYS-domain
  chat. Calibration deltas measured here may not generalize cleanly to
  predominantly non-Latin scripts where tokenizer behavior differs sharply.
- **LMSYS-domain skew.** LMSYS-Chat-1M is biased toward arena-style head-to-head
  prompts; readers should not assume the prompt mix mirrors production traffic
  for an arbitrary application.
- **Provider tokenizer drift over time.** Providers update their tokenizers; a
  calibration delta observed today may not hold next quarter. The
  `tokenizer_id` and `api_version` fields plus `data/lockfile.json` are the
  guard against silently comparing apples to oranges across releases.
- **Format-mediated bias.** Some formats (JSON, XML) introduce structural
  characters that some tokenizers fuse and others split; this is the central
  phenomenon the dataset measures, so we record it explicitly rather than
  smooth it away.

### Limitations

- **No generative outputs.** The dataset only measures tokenization, not model
  quality. Cost-aware routing systems built on this data must source quality
  signals separately.
- **Empirical-count availability is uneven.** Where no public count-tokens
  endpoint exists, `empirical_count` is null and the row participates in
  offline-coverage analyses only.
- **Sampling is finite.** 10k-prompt class size is enough for stable
  per-provider deltas but lighter for some `(provider × model × format ×
  language)` cells; the row counts per cell are reported in
  `data/provenance.md`.

## Additional Information

### Dataset Curators

Curated and maintained by Faraazuddin Mohammed
(<https://github.com/faraa2m>). Issues, PRs, and provenance corrections
welcome at <https://github.com/faraa2m/llm-tokens-atlas/issues>.

### Licensing Information

- **Data** (everything under `data/`): [Creative Commons Attribution 4.0
  International (CC-BY-4.0)](https://github.com/faraa2m/llm-tokens-atlas/blob/main/LICENSE-DATA).
  You may share and adapt the data for any purpose, including commercial,
  provided you give appropriate credit.
- **Code** (collection scripts, analysis code, this card itself outside the
  dataset payload): [Apache-2.0](https://github.com/faraa2m/llm-tokens-atlas/blob/main/LICENSE).

Suggested attribution string:

> "Data from llm-tokens-atlas (Faraazuddin Mohammed, 2026),
> <https://huggingface.co/datasets/faraa2m/llm-tokens-atlas>, CC-BY-4.0."

### Citation Information

```bibtex
@misc{llm-tokens-atlas-2026,
  author       = {Faraazuddin Mohammed},
  title        = {{llm-tokens-atlas}: An Open Benchmark of LLM Tokenization Calibration Across Providers and Formats},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/datasets/faraa2m/llm-tokens-atlas}}
}
```

### Related Work

The offline-vs-empirical tokenizer drift phenomenon was independently surfaced
in 2026 by community blog posts; this dataset is intended as the
peer-citable, reproducible version of that observation, not as the discovery
of it. Honest positioning credits the prior surfacers:

- *"I Measured Claude 4.7's New Tokenizer — Here's What It Costs You"* (2026,
  Claude Code Camp blog) —
  <https://www.claudecodecamp.com/p/i-measured-claude-4-7-s-new-tokenizer-here-s-what-it-costs-you>.
  First blog-post measurement of the post-2026 Anthropic tokenizer update on a
  single model.
- *"Anthropic's New Tokenizer Is Quietly Hiking Your Claude Costs by 47% — I
  Fixed It"* (2026, Medium / AI Software Engineer) —
  <https://medium.com/ai-software-engineer/anthropics-new-tokenizer-is-quietly-hiking-your-claude-costs-by-47-i-fixed-it-91c69ff0017b>.
  Independent practitioner blog confirming the drift at ~47% on a separate
  workload.
- *`tokencost`* (AgentOps, 2024–present) —
  <https://github.com/AgentOps-AI/tokencost>. The dominant production
  cost-estimator library; uses tiktoken proxies for older Claude and
  Anthropic's empirical countTokens for Claude 3+. Closest practitioner
  precedent for "estimate cost from tokens"; complementary to (not
  superseded by) this dataset.

Methodologically related academic work:

- Sachan et al., *RouterBench: A Benchmark for Multi-LLM Routing Systems*
  (2024) — `arXiv:2403.12031`. Sibling open dataset of API outcomes, but
  routing-focused rather than tokenization-focused.
- Zheng et al., *LMSYS-Chat-1M: A Large-Scale Real-World LLM Conversation
  Dataset* (2023) — `arXiv:2309.11998`. Source corpus.
- Ali et al., *Tokenizer Choice For LLM Training: Negligible or Crucial?*
  (2023) — `arXiv:2310.08754`. Training-time tokenizer study; complementary to
  this deployment-time drift study.
- Rust et al., *How Good is Your Tokenizer? On the Monolingual Performance of
  Multilingual Language Models* (2021, ACL). Methodology precedent.
- Javier Rando, *The Worst (But Only) Claude 3 Tokenizer* (2024) —
  <https://javirando.com/blog/2024/claude-tokenizer/>. Canonical source for
  the offline Claude tokenizer the community uses; cited here as the offline
  baseline for the Anthropic provider rows.

### Foundation

This dataset extends the methodology and an early finding from
[tokenometer](https://github.com/faraa2m/tokenometer) (cl100k_base
underestimates `claude-opus-4-7` tokens by ~62% median), generalizing the
single-point measurement into a multi-provider, multi-format calibration
distribution.

### Contact

- GitHub: <https://github.com/faraa2m/llm-tokens-atlas>
- Issues: <https://github.com/faraa2m/llm-tokens-atlas/issues>

### Dataset Card Authors

Faraazuddin Mohammed (<https://github.com/faraa2m>).
