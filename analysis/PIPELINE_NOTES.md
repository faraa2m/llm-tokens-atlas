# Pipeline notes — atlas-analysis observations

This file logs sibling-script issues surfaced while running the pipeline
at N=500. Per the analysis-module brief, sibling code is **not** edited; issues
are documented here so the relevant owner can patch them when convenient.

## Issue 1 — `llm_tokens_atlas/count_offline.py` aborts on prompts containing ChatML special tokens

**Symptom.** At N=500, the offline counter aborted after writing 5,425 rows
(roughly 217 prompts × 25 cells) with this `gpt-tokenizer` failure:

```
RuntimeError: tokenometer CLI failed (rc=1);
stderr="Unexpected error: Error: Disallowed special token found: <|im_start|>"
```

**Root cause.** `@tokenometer/core` uses
[`gpt-tokenizer`](https://www.npmjs.com/package/gpt-tokenizer)'s default
`encode()`, which rejects ChatML / GPT-4 special tokens (`<|im_start|>`,
`<|im_end|>`, `<|endoftext|>`, `<|fim_prefix|>`, `<|fim_middle|>`,
`<|fim_suffix|>`, `<|endofprompt|>`) unless the caller explicitly passes them
in `allowedSpecial`. Real-world corpora (GitHub READMEs, ShareGPT, model
documentation) occasionally contain these literals as text, not as control
tokens; the offline counter has no way to disambiguate.

**Surface area in our N=500 sample.** Exactly **1 prompt** (out of 500)
contains these markers:

| source | domain | markers |
|---|---|---|
| github-readmes | code | `<|im_start|>`, `<|im_end|>` |

So the impact at this scale is one bad prompt, but the failure is total —
the counter dies on first contact and emits no row for the rest of the
pipeline.

**Mitigations considered.**

1. **Edit `llm_tokens_atlas/count_offline.py` to catch + skip.** Rejected per the
   "do not edit sibling pipeline code" rule in the analysis brief.
2. **Edit `tokenometer/packages/core/src/tokenize.ts` to pass
   `allowedSpecial: 'all'`.** Rejected for the same reason; also a true API
   change that needs review.
3. **Pre-filter the input corpus.** Adopted. The analysis driver works on
   the union of (a) the original offline rows that did land before the
   crash and (b) a re-run on a corpus filtered to remove ChatML-marker
   prompts. The dataset card and provenance should call this out so HF
   users understand what was excluded.

**Recommended pipeline patches.** Either:

- `llm_tokens_atlas/count_offline.py`: wrap the `_run_tokenometer(...)` call in a
  try/except that catches `RuntimeError`, logs the offending `prompt_id`,
  and continues. The lost cells are recorded in a `skipped.jsonl` file.
- `tokenometer/packages/core/src/tokenize.ts`: thread an
  `allowedSpecial?: Set<string> | 'all'` option through `countTokens()` /
  `tokenize()` so callers (atlas, future projects) can opt into treating
  special tokens as opaque text. Recommended default for benchmark
  pipelines: `'all'`.

The atlas dataset documentation should also note the exclusion in
`data/README.md` once the downstream patch ships.

### Workaround in this session

We added `analysis/scripts/filter_prompts.py` to produce
`data/raw_prompts.filtered.jsonl` excluding prompts that contain any of
the documented ChatML markers. The pipeline is then driven on the
filtered file. The original `data/raw_prompts.jsonl` is left untouched
(it is the immutable input artifact for the public corpus).

## Issue 2 — `count_offline.py` and `count_empirical.py` ship divergent `DEFAULT_MODELS`

**Symptom.** When `llm_tokens_atlas/build_dataset.py` inner-joins offline and
empirical counts on `(prompt_id, provider, format, model)`, several
provider partitions vanish. Specifically:

| provider | offline model | empirical models | overlap |
|---|---|---|---|
| openai | `gpt-4o` | `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo` | `gpt-4o` |
| anthropic | `claude-opus-4-7` | `claude-opus-4-7`, `claude-sonnet-4-6`, `claude-haiku-4-5` | `claude-opus-4-7` |
| google | `gemini-2.5-pro` | `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-1.5-pro` | `gemini-2.5-pro` |
| mistral | `mistral-large-latest` | `mistral-large-2407`, `mistral-small-2409`, `open-mistral-nemo-2407` | **none** |
| cohere | `command-r` | `command-r-plus`, `command-r` | `command-r` |

The mistral partition is entirely lost, and the other partitions silently
drop the empirical rows for the 2 extra models per provider — those rows
are noise (no offline counterpart).

**Root cause.** No shared registry of "this provider's canonical atlas
model". Each script picks its own list. The empirical default list was
likely chosen to mirror a per-provider product family; the offline list
was chosen to mirror tokenometer's CLI naming conventions.

**Recommended pipeline patch.** Introduce a single `ATLAS_MODELS` constant
(e.g. in `llm_tokens_atlas/_atlas_models.py`) imported by both `count_offline.py`
and `count_empirical.py`. Either:

- Promote `DEFAULT_MODELS` in `count_offline.py` to a `list[str]` so
  every provider can have multiple models, and align `count_empirical.py`
  to read the same list; or
- Add a `--models` flag to `count_empirical.py` (mirrors
  `count_offline.py`'s) so callers can override the model set per run.

### Workaround in this session

`analysis/scripts/run_empirical_aligned.py` reuses
`count_empirical.py`'s `Counter` classes verbatim but constrains the
cell matrix to **exactly** the offline-driver model set (one canonical
model per provider). The output goes to `data/empirical_counts.jsonl`
and joins cleanly with `data/offline_counts.jsonl`.

