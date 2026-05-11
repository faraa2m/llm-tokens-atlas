"""Empirical-count driver aligned with the offline driver's model set.

Why this script exists
======================

The ``scripts/count_offline.py`` driver defaults to **one canonical model
per provider** (e.g. ``claude-opus-4-7`` for Anthropic), while
``scripts/count_empirical.py`` defaults to **three models per provider**.
Worse, the two scripts do not share a single source of truth for that
model list; for Mistral they disagree on the family ID itself
(``mistral-large-latest`` vs. ``mistral-large-2407``).

When ``scripts/build_dataset.py`` performs its inner-join on
``(prompt_id, provider, format, model)``, this mismatch silently drops
the Mistral rows entirely and only retains the OpenAI cells where
``gpt-4o`` is the common model.

Per the atlas-analysis brief we do not edit sibling pipeline scripts;
we work around the gap from the analysis layer. This driver reuses the
``Counter`` classes inside ``scripts/count_empirical.py`` but constrains
the model set to **exactly** what the offline driver emits, guaranteeing
a clean join.

Usage
-----

.. code-block:: bash

    uv run python analysis/scripts/run_empirical_aligned.py \\
        --in data/raw_prompts.filtered.jsonl \\
        --out data/empirical_counts.jsonl \\
        --providers openai,mistral,anthropic,google,cohere

Providers whose API keys are missing are silently dropped (with a log
message), matching the upstream script's behaviour.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

# Make scripts/ importable so we can reuse the Counter classes verbatim.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.count_empirical import (  # noqa: E402
    Cell,
    ProgressWriter,
    _count_with_retry,
    build_counters,
    finalize_output,
    read_done_keys,
    read_prompts,
)
from scripts.format_wrappers import ALL_FORMATS, wrap  # noqa: E402

# These mirror DEFAULT_MODELS in scripts/count_offline.py exactly — one model
# per provider, so the inner join with the offline output is well-defined.
ALIGNED_MODELS: dict[str, str] = {
    "anthropic": "claude-opus-4-7",
    "openai": "gpt-4o",
    "google": "gemini-2.5-pro",
    "mistral": "mistral-large-latest",
    "cohere": "command-r",
}

LOG = logging.getLogger("run_empirical_aligned")


def build_cells_aligned(
    prompts: list[dict],
    providers: list[str],
    formats: tuple[str, ...],
) -> list[Cell]:
    """One cell per (prompt × provider × format), using ALIGNED_MODELS[provider]."""
    cells: list[Cell] = []
    for prompt in prompts:
        for provider in providers:
            model = ALIGNED_MODELS.get(provider)
            if model is None:
                continue
            for fmt in formats:
                cells.append(
                    Cell(
                        prompt_id=prompt["prompt_id"],
                        prompt_text=prompt["text"],
                        provider=provider,
                        fmt=fmt,
                        model=model,
                    )
                )
    return cells


async def run_aligned(
    prompts: list[dict],
    counters: dict,
    providers: list[str],
    formats: tuple[str, ...],
    progress_writer: ProgressWriter,
    done_keys: set,
    concurrency: int,
) -> int:
    """Drive the empirical workers; return the count of newly-written rows."""
    cells = build_cells_aligned(prompts, providers, formats)
    todo = [c for c in cells if c.key not in done_keys]
    if not todo:
        LOG.info("nothing to do; %d cells already counted", len(cells))
        return 0
    LOG.info(
        "running %d cells (skipping %d already done) at concurrency=%d",
        len(todo),
        len(cells) - len(todo),
        concurrency,
    )
    sem = asyncio.Semaphore(concurrency)
    written = 0
    lock = asyncio.Lock()

    async def _worker(cell: Cell) -> None:
        nonlocal written
        async with sem:
            counter = counters[cell.provider]
            wrapped = wrap(cell.prompt_text, cell.fmt)
            try:
                row = await _count_with_retry(counter, cell, wrapped)
            except Exception as e:
                LOG.error(
                    "giving up on %s/%s %s: %s",
                    cell.provider, cell.model, cell.fmt, e,
                )
                return
            async with lock:
                progress_writer.write(row)
                written += 1

    await asyncio.gather(*(_worker(c) for c in todo))
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_path", type=Path, required=True)
    parser.add_argument("--out", dest="out_path", type=Path, required=True)
    parser.add_argument(
        "--progress",
        dest="progress_path",
        type=Path,
        default=None,
        help="Checkpoint file (append-only). Defaults to <--out parent>/empirical_progress.jsonl.",
    )
    parser.add_argument(
        "--providers",
        type=str,
        default="anthropic,google,openai,mistral,cohere",
        help="Comma-separated provider names to run (default: all five).",
    )
    parser.add_argument(
        "--formats",
        type=str,
        default=",".join(ALL_FORMATS),
        help=f"Comma-separated formats (default: {','.join(ALL_FORMATS)}).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Max concurrent counter calls (default: 4).",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=0,
        help="Cap the number of prompts to process (0 = no cap).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    formats = tuple(f.strip() for f in args.formats.split(",") if f.strip())
    for f in formats:
        if f not in ALL_FORMATS:
            raise SystemExit(f"unknown format {f!r}; valid: {ALL_FORMATS}")

    counters = build_counters(providers)
    if not counters:
        LOG.error(
            "no providers runnable; set at least one of "
            "ANTHROPIC_API_KEY / GOOGLE_API_KEY / COHERE_API_KEY, "
            "or include 'openai'/'mistral' which need no credentials."
        )
        return 2
    runnable_providers = [p for p in providers if p in counters]

    prompts = read_prompts(args.in_path, max_prompts=args.max_prompts)
    if not prompts:
        LOG.warning("no prompts in %s", args.in_path)
        return 0

    progress_path = (
        args.progress_path
        if args.progress_path is not None
        else args.out_path.parent / "empirical_progress.jsonl"
    )
    done_keys = read_done_keys(progress_path)
    LOG.info(
        "%d prompts × providers=%s formats=%s — %d cells already done in %s",
        len(prompts),
        runnable_providers,
        list(formats),
        len(done_keys),
        progress_path,
    )

    writer = ProgressWriter(progress_path)
    t0 = time.time()
    try:
        written = asyncio.run(
            run_aligned(
                prompts=prompts,
                counters=counters,
                providers=runnable_providers,
                formats=formats,
                progress_writer=writer,
                done_keys=done_keys,
                concurrency=args.concurrency,
            )
        )
    finally:
        writer.close()

    final = finalize_output(progress_path, args.out_path)
    LOG.info(
        "wrote %d new rows (final dedup'd count: %d) -> %s in %.1fs",
        written, final, args.out_path, time.time() - t0,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
