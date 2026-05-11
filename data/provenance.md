# Provenance — `data/raw_prompts.jsonl`

Generated at: 2026-05-11T00:32:37Z
Output file: `data/raw_prompts.jsonl`
Total rows: 500
Sampling seed: 42 (python `random.Random` + numpy)

This file is the authoritative record of which prompts were sampled,
from which upstream corpora, and under which licenses. It mirrors the
schema enum `promptRow.source` declared in `data/schema.json`.

## Sources

### `humaneval` — OpenAI HumanEval

- Dataset URL: <https://huggingface.co/datasets/openai/openai_humaneval>
- License: MIT (<https://github.com/openai/human-eval/blob/master/LICENSE>)
- Upstream revision (snapshot): `7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544`
- Rows sampled: **100**
- Sampling strategy: Uniform random sample of the 164 'test' split problems; we keep the `prompt` field (function signature + docstring), which is what an LLM would receive.

  BibTeX:

  ```bibtex
  @misc{chen2021humaneval,
    title  = {Evaluating Large Language Models Trained on Code},
    author = {Mark Chen and Jerry Tworek and Heewoo Jun and others},
    year   = {2021},
    eprint = {2107.03374},
    archivePrefix = {arXiv},
    primaryClass  = {cs.LG}
  }
  ```

### `wildchat-1m` — AllenAI WildChat-1M

- Dataset URL: <https://huggingface.co/datasets/allenai/WildChat-1M>
- License: ODC-BY (Open Data Commons Attribution) (<https://opendatacommons.org/licenses/by/1-0/>)
- Upstream revision (snapshot): `7d6490e462285cf85d91eabea0f9a954fbddcd1f`
- Rows sampled: **119**
- Sampling strategy: Uniform random sample of `redacted==True && toxic==False && language=='English'` conversations from shard `data/train-00000-of-00014.parquet`; we keep only the first **user** turn, drop assistant turns. The `redacted` filter is the PII-safety filter recommended by the dataset authors.

  Notes: WildChat-1M is used here as a substitute for LMSYS-chat-1M, which is **gated** on HuggingFace and therefore not redistributable in a fully-reproducible, credentials-free pipeline. WildChat-1M is the spiritual successor — same conversational-log format, same `redacted` and `toxic` flags — and is published under ODC-BY for redistribution. The atlas paper should cite both.

  BibTeX:

  ```bibtex
  @inproceedings{zhao2024wildchat,
    title     = {{WildChat}: 1M ChatGPT Interaction Logs in the Wild},
    author    = {Wenting Zhao and Xiang Ren and Jack Hessel and Claire Cardie
                 and Yejin Choi and Yuntian Deng},
    booktitle = {ICLR},
    year      = {2024}
  }
  ```

### `mt-bench` — MT-Bench (LMSYS human judgments)

- Dataset URL: <https://huggingface.co/datasets/lmsys/mt_bench_human_judgments>
- License: CC-BY-4.0 (<https://creativecommons.org/licenses/by/4.0/>)
- Upstream revision (snapshot): `f7d2896d2cc5d80f8b55c2bbc722613555233c25`
- Rows sampled: **80**
- Sampling strategy: We deduplicate the 3355-row `human` split by `question_id`, then take the first **user** turn from `conversation_a`. Result is a pool of 80-ish unique evaluation questions; we sample uniformly without replacement.

  BibTeX:

  ```bibtex
  @inproceedings{zheng2023judging,
    title     = {Judging {LLM}-as-a-Judge with {MT}-Bench and Chatbot Arena},
    author    = {Lianmin Zheng and Wei-Lin Chiang and Ying Sheng
                 and Siyuan Zhuang and Zhanghao Wu and Yonghao Zhuang
                 and Zi Lin and Zhuohan Li and Dacheng Li
                 and Eric P. Xing and Hao Zhang and Joseph E. Gonzalez
                 and Ion Stoica},
    booktitle = {NeurIPS Datasets and Benchmarks},
    year      = {2023}
  }
  ```

### `wikipedia-en` — English Wikipedia (`20231101.en` snapshot)

- Dataset URL: <https://huggingface.co/datasets/wikimedia/wikipedia>
- License: CC-BY-SA-4.0 + GFDL (<https://creativecommons.org/licenses/by-sa/4.0/>)
- Upstream revision (snapshot): `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`
- Rows sampled: **118**
- Sampling strategy: Uniform random sample of articles from a single parquet shard (`20231101.en/train-00000-of-00041.parquet`). For each article we keep the first paragraph (split on `\n\n`), truncated to 2000 chars. Articles whose first paragraph is shorter than 80 chars (navigation stubs, disambiguation pages) are skipped.

  BibTeX:

  ```bibtex
  @misc{wikipedia2023,
    title  = {{Wikimedia/Wikipedia Snapshot 20231101 (English)}},
    author = {{Wikimedia Foundation}},
    year   = {2023},
    howpublished = {\url{https://huggingface.co/datasets/wikimedia/wikipedia}}
  }
  ```

### `github-readmes` — GitHub READMEs (curated seed list)

- Dataset URL: <https://raw.githubusercontent.com/>
- License: Per-repo (Apache-2.0 / MIT / BSD; see GITHUB_README_REPOS list in scripts/collect_corpus.py) (<https://github.com/faraa2m/llm-tokens-atlas/blob/main/scripts/collect_corpus.py>)
- Upstream revision (snapshot): `n/a (no HF repo)`
- Rows sampled: **83**
- Sampling strategy: Fetched the canonical README (in `.md` / `.rst` / `.asciidoc` / plain) of a curated list of well-known repositories via `raw.githubusercontent.com`. All listed repositories use permissive OSS licenses (Apache-2.0, MIT, BSD-3-Clause, or PSF). The full list lives in `GITHUB_README_REPOS` in the collection script for auditability. Order is shuffled with the seeded rng before fetching, so the first N fetched are deterministic. The static seed list is used in lieu of `bigcode/the-stack` (which is auto-gated and therefore breaks the credentials-free path).

  BibTeX:

  ```bibtex
  @misc{githubreadmes2026,
    title  = {Curated GitHub READMEs for tokenization benchmarking},
    author = {Faraazuddin Mohammed and the llm-tokens-atlas authors},
    year   = {2026},
    howpublished = {\url{https://github.com/faraa2m/llm-tokens-atlas/blob/main/scripts/collect_corpus.py}}
  }
  ```

## Reproducibility

Determinism: all RNG state is seeded with the `--seed` CLI flag (default 42). `uuid` values are drawn from the seeded `random.Random` rather than `uuid.uuid4()` (which is OS-random) so prompt ids are stable across runs.

HF dataset snapshots are pinned implicitly by the `huggingface_hub` cache; the captured commit SHAs above are the snapshots used at generation time. To regenerate the exact bytes, set `HF_HUB_OFFLINE=1` after the cache is warm, or manually pin `revision=<sha>` in the collection calls.
