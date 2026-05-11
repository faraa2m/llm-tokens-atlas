"""Build the processed atlas dataset Parquet file.

Reads three JSONL streams produced by sibling agents:
  - data/raw_prompts.jsonl          (prompts table)
  - data/offline_counts.jsonl       (offline_counts table)
  - data/empirical_counts.jsonl     (empirical_counts table)

Validates each row against the matching sub-schema in data/schema.json,
inner-joins the count streams on (prompt_id, provider, format, model),
attaches prompt-level columns, computes calibration deltas, and writes
a single Apache Parquet file to --out.

Computed columns:
  - delta       = empirical_count - offline_count
  - delta_pct   = delta / empirical_count * 100   (NaN where empirical_count == 0)
  - abs_delta   = abs(delta)
  - direction   = "underestimate" | "overestimate" | "exact"
                  (relative to the offline tokenizer)

Usage:
  uv run python scripts/build_dataset.py \\
      --raw data/raw_prompts.jsonl \\
      --offline data/offline_counts.jsonl \\
      --empirical data/empirical_counts.jsonl \\
      --out data/processed/atlas.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
from jsonschema import Draft202012Validator  # type: ignore[import-untyped]
from jsonschema.exceptions import ValidationError  # type: ignore[import-untyped]

# --------------------------------------------------------------------------- #
# Constants                                                                   #
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCHEMA = REPO_ROOT / "data" / "schema.json"

JOIN_KEY = ["prompt_id", "provider", "format", "model"]


@dataclass(frozen=True)
class StreamSpec:
    """Binding of an input JSONL stream to its row sub-schema."""

    name: str               # "prompts" | "offline_counts" | "empirical_counts"
    row_def: str            # JSON Schema $defs key, e.g. "promptRow"


PROMPTS_SPEC = StreamSpec("prompts", "promptRow")
OFFLINE_SPEC = StreamSpec("offline_counts", "offlineCountRow")
EMPIRICAL_SPEC = StreamSpec("empirical_counts", "empiricalCountRow")


# --------------------------------------------------------------------------- #
# Schema loading                                                              #
# --------------------------------------------------------------------------- #


def load_schema(schema_path: Path) -> dict[str, Any]:
    """Load the atlas JSON Schema document."""
    with schema_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def row_validator(schema: dict[str, Any], row_def_key: str) -> Draft202012Validator:
    """Build a Draft 2020-12 validator for a single row sub-schema in $defs.

    Uses the full document as the base so internal $refs in $defs resolve
    correctly (e.g. promptRow references prompt_id via #/$defs/prompt_id).
    """
    defs = schema.get("$defs", {})
    if row_def_key not in defs:
        raise KeyError(f"$defs.{row_def_key} not present in schema {schema.get('$id')}")
    row_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$defs": defs,
        "$ref": f"#/$defs/{row_def_key}",
    }
    Draft202012Validator.check_schema(row_schema)
    return Draft202012Validator(row_schema)


# --------------------------------------------------------------------------- #
# JSONL reading + validation                                                  #
# --------------------------------------------------------------------------- #


def read_jsonl_validated(
    path: Path,
    validator: Draft202012Validator,
    stream_name: str,
) -> list[dict[str, Any]]:
    """Read one JSONL file and validate every row against `validator`.

    Raises ValueError on first malformed line or schema violation, with a
    message that includes file path and 1-indexed line number so users can
    jump straight to the offending row.
    """
    rows: list[dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"{path}:{lineno}: malformed JSON in {stream_name}: {e}"
                ) from e
            try:
                validator.validate(obj)
            except ValidationError as e:
                path_in_obj = "/".join(str(p) for p in e.absolute_path) or "<root>"
                raise ValueError(
                    f"{path}:{lineno}: {stream_name} schema violation "
                    f"at '{path_in_obj}': {e.message}"
                ) from e
            rows.append(obj)
    return rows


# --------------------------------------------------------------------------- #
# Join + computed columns                                                     #
# --------------------------------------------------------------------------- #


def _direction(delta: int) -> str:
    """Map signed delta to a categorical direction relative to the offline tokenizer.

    - offline < empirical => offline underestimates the real count.
    - offline > empirical => offline overestimates the real count.
    - offline == empirical => exact agreement.
    """
    if delta > 0:
        return "underestimate"
    if delta < 0:
        return "overestimate"
    return "exact"


def build_dataframe(
    prompts: list[dict[str, Any]],
    offline: list[dict[str, Any]],
    empirical: list[dict[str, Any]],
) -> pd.DataFrame:
    """Inner-join the three streams and compute calibration columns.

    Join semantics:
      1. Inner-join offline_counts and empirical_counts on
         (prompt_id, provider, format, model). Only tuples present in both
         count streams survive — the calibration delta is undefined otherwise.
      2. Left-merge prompt-level columns from prompts onto the joined frame
         using prompt_id. Rows missing a matching prompt would carry NaNs and
         are dropped (with a stderr warning) so downstream analysis isn't
         polluted by orphan counts.

    Returns:
        DataFrame with columns:
            prompt_id, source, text, text_len_chars, text_len_words, language,
            domain, collected_at,
            provider, format, model,
            offline_count, tokenizer_version, offline_ts,
            empirical_count, is_oracle, empirical_source, endpoint, empirical_ts,
            delta, delta_pct, abs_delta, direction
    """
    prompts_df = pd.DataFrame(prompts)
    offline_df = pd.DataFrame(offline).rename(columns={"ts": "offline_ts"})
    empirical_df = pd.DataFrame(empirical).rename(
        columns={"ts": "empirical_ts", "source": "empirical_source"}
    )

    if offline_df.empty or empirical_df.empty:
        raise ValueError(
            "build_dataframe requires non-empty offline_counts and "
            "empirical_counts; nothing to join."
        )

    counts = offline_df.merge(
        empirical_df,
        on=JOIN_KEY,
        how="inner",
        validate="one_to_one",
    )

    # Attach prompt-level columns (left-merge keeps every joined count row).
    merged = counts.merge(
        prompts_df,
        on="prompt_id",
        how="left",
        validate="many_to_one",
    )

    orphans = merged["text"].isna().sum() if "text" in merged.columns else 0
    if orphans:
        print(
            f"warning: dropping {orphans} count row(s) with no matching prompt_id",
            file=sys.stderr,
        )
        merged = merged.dropna(subset=["text"]).reset_index(drop=True)

    # Calibration deltas — integer math, NaN for delta_pct when empirical is 0.
    merged["delta"] = (merged["empirical_count"] - merged["offline_count"]).astype(int)
    merged["abs_delta"] = merged["delta"].abs().astype(int)
    merged["direction"] = merged["delta"].map(_direction).astype("string")

    # delta_pct = delta / empirical_count * 100; NaN where empirical_count == 0
    empirical_f = merged["empirical_count"].astype(float)
    delta_f = merged["delta"].astype(float)
    merged["delta_pct"] = (delta_f / empirical_f.where(empirical_f != 0)) * 100.0

    column_order = [
        "prompt_id",
        "source",
        "text",
        "text_len_chars",
        "text_len_words",
        "language",
        "domain",
        "collected_at",
        "provider",
        "format",
        "model",
        "offline_count",
        "tokenizer_version",
        "offline_ts",
        "empirical_count",
        "is_oracle",
        "empirical_source",
        "endpoint",
        "empirical_ts",
        "delta",
        "delta_pct",
        "abs_delta",
        "direction",
    ]
    return merged[[c for c in column_order if c in merged.columns]]


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="build_dataset",
        description=(
            "Validate raw_prompts/offline_counts/empirical_counts JSONL "
            "streams, join them, and emit a single processed Parquet file."
        ),
    )
    parser.add_argument("--raw", required=True, type=Path, help="data/raw_prompts.jsonl")
    parser.add_argument("--offline", required=True, type=Path, help="data/offline_counts.jsonl")
    parser.add_argument("--empirical", required=True, type=Path, help="data/empirical_counts.jsonl")
    parser.add_argument("--out", required=True, type=Path, help="Output Parquet path")
    parser.add_argument(
        "--schema",
        type=Path,
        default=DEFAULT_SCHEMA,
        help="Path to data/schema.json (default: repo-relative).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    schema = load_schema(args.schema)

    prompts_v = row_validator(schema, PROMPTS_SPEC.row_def)
    offline_v = row_validator(schema, OFFLINE_SPEC.row_def)
    empirical_v = row_validator(schema, EMPIRICAL_SPEC.row_def)

    prompts = read_jsonl_validated(args.raw, prompts_v, PROMPTS_SPEC.name)
    offline = read_jsonl_validated(args.offline, offline_v, OFFLINE_SPEC.name)
    empirical = read_jsonl_validated(args.empirical, empirical_v, EMPIRICAL_SPEC.name)

    df = build_dataframe(prompts, offline, empirical)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, engine="pyarrow", index=False)

    print(
        f"wrote {len(df)} rows -> {args.out} "
        f"(prompts={len(prompts)}, offline={len(offline)}, empirical={len(empirical)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
