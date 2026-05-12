"""Filter prompts that the offline tokenizer cannot handle.

Workaround for the issue documented in ``analysis/PIPELINE_NOTES.md``:
``@tokenometer/core``'s offline path (via ``gpt-tokenizer``) rejects any
input that contains a recognized OpenAI special token, even when that
literal appears as plain text. This script drops such prompts from the
input JSONL so the rest of the pipeline can run.

It is intentionally **not** placed in ``llm_tokens_atlas/`` because that
package is owned by the corpus/counting pipeline. Living in
``analysis/scripts/`` keeps the ownership boundary clear: this is an
analysis-side workaround, not an amendment to the corpus or counting
stages.

Usage:

    uv run python analysis/scripts/filter_prompts.py \\
        --in data/raw_prompts.jsonl \\
        --out data/raw_prompts.filtered.jsonl \\
        --excluded data/raw_prompts.excluded.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Tokens that gpt-tokenizer treats as special and refuses to encode unless
# the caller opts in. Drawn from the cl100k_base and o200k_base specials.
SPECIAL_MARKERS: tuple[str, ...] = (
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<|endofprompt|>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
)


def has_special_marker(text: str) -> tuple[bool, list[str]]:
    """Return (flagged, markers_found)."""
    found = [m for m in SPECIAL_MARKERS if m in text]
    return (bool(found), found)


def filter_file(in_path: Path, out_path: Path, excluded_path: Path) -> tuple[int, int]:
    """Write filtered prompts to `out_path` and excluded prompts to `excluded_path`.

    Returns ``(n_kept, n_excluded)``.
    """
    n_kept = 0
    n_excluded = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    excluded_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        in_path.open("r", encoding="utf-8") as fin,
        out_path.open("w", encoding="utf-8") as fout,
        excluded_path.open("w", encoding="utf-8") as fexc,
    ):
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            flagged, markers = has_special_marker(row.get("text", ""))
            if flagged:
                row_with_reason = dict(row, _excluded_for="contains_special_tokens",
                                       _markers=markers)
                fexc.write(json.dumps(row_with_reason, ensure_ascii=False) + "\n")
                n_excluded += 1
            else:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_kept += 1
    return n_kept, n_excluded


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_path", type=Path, required=True)
    parser.add_argument("--out", dest="out_path", type=Path, required=True)
    parser.add_argument(
        "--excluded", dest="excluded_path", type=Path, required=True,
        help="Where to write rows that were filtered out (with the marker(s) "
        "that triggered exclusion).",
    )
    args = parser.parse_args(argv)
    kept, excluded = filter_file(args.in_path, args.out_path, args.excluded_path)
    print(f"kept {kept} prompts; excluded {excluded} -> {args.excluded_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
