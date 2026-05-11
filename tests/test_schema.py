"""Tests for the atlas schema, dataset builder, and lockfile generator.

Covers:
- All three synthetic fixture streams validate against data/schema.json.
- scripts/build_dataset.py joins correctly and computes the expected
  calibration columns row-for-row.
- scripts/lockfile.py emits a non-empty, valid JSON document with the
  required reproducibility fields.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from jsonschema.exceptions import ValidationError

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = REPO_ROOT / "data" / "schema.json"
FIXTURES = REPO_ROOT / "tests" / "fixtures"

sys.path.insert(0, str(REPO_ROOT))

from scripts.build_dataset import (  # noqa: E402
    EMPIRICAL_SPEC,
    OFFLINE_SPEC,
    PROMPTS_SPEC,
    build_dataframe,
    load_schema,
    read_jsonl_validated,
    row_validator,
)
from scripts.lockfile import build_lockfile  # noqa: E402

# --------------------------------------------------------------------------- #
# Schema validation                                                           #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def schema() -> dict:
    return load_schema(SCHEMA_PATH)


def test_schema_has_pinned_id(schema):
    assert (
        schema["$id"] == "https://github.com/faraa2m/llm-tokens-atlas/schema/v1.json"
    )
    assert (
        schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    )


def test_fixture_prompts_validate_against_schema(schema):
    validator = row_validator(schema, PROMPTS_SPEC.row_def)
    rows = read_jsonl_validated(FIXTURES / "raw_prompts.jsonl", validator, PROMPTS_SPEC.name)
    assert len(rows) == 3
    assert {r["prompt_id"] for r in rows} == {"fix-001", "fix-002", "fix-003"}


def test_fixture_offline_counts_validate_against_schema(schema):
    validator = row_validator(schema, OFFLINE_SPEC.row_def)
    rows = read_jsonl_validated(
        FIXTURES / "offline_counts.jsonl", validator, OFFLINE_SPEC.name
    )
    assert len(rows) == 6
    for r in rows:
        assert r["offline_count"] >= 0
        assert r["format"] in {"markdown", "xml", "json", "yaml", "plain"}
        assert r["provider"] in {"openai", "anthropic", "google", "mistral", "cohere"}


def test_fixture_empirical_counts_validate_against_schema(schema):
    validator = row_validator(schema, EMPIRICAL_SPEC.row_def)
    rows = read_jsonl_validated(
        FIXTURES / "empirical_counts.jsonl", validator, EMPIRICAL_SPEC.name
    )
    assert len(rows) == 6
    for r in rows:
        assert r["empirical_count"] >= 0
        assert isinstance(r["is_oracle"], bool)
        assert r["source"] in {"api", "tiktoken", "sdk", "stream-usage"}


def test_schema_rejects_bad_count(schema):
    """A negative offline_count must fail validation (counts ≥ 0 constraint)."""
    validator = row_validator(schema, OFFLINE_SPEC.row_def)
    bad = {
        "prompt_id": "bad-001",
        "provider": "openai",
        "format": "plain",
        "model": "gpt-4o-2024-08-06",
        "offline_count": -1,
        "tokenizer_version": "tiktoken@cl100k_base",
        "ts": "2026-05-10T00:00:00Z",
    }
    with pytest.raises(ValidationError):
        validator.validate(bad)


def test_schema_rejects_bad_format(schema):
    """A format outside the five enum values must fail validation."""
    validator = row_validator(schema, OFFLINE_SPEC.row_def)
    bad = {
        "prompt_id": "bad-002",
        "provider": "openai",
        "format": "toml",
        "model": "gpt-4o-2024-08-06",
        "offline_count": 1,
        "tokenizer_version": "tiktoken@cl100k_base",
        "ts": "2026-05-10T00:00:00Z",
    }
    with pytest.raises(ValidationError):
        validator.validate(bad)


# --------------------------------------------------------------------------- #
# build_dataset                                                               #
# --------------------------------------------------------------------------- #


def _expected_deltas() -> dict[tuple[str, str], dict[str, float | int | str]]:
    """Hand-computed expected calibration values per (prompt_id, provider)."""
    return {
        ("fix-001", "openai"): {
            "delta": 0, "abs_delta": 0, "delta_pct": 0.0, "direction": "exact",
        },
        ("fix-001", "anthropic"): {
            "delta": 4, "abs_delta": 4, "delta_pct": 40.0, "direction": "underestimate",
        },
        ("fix-002", "openai"): {
            "delta": 0, "abs_delta": 0, "delta_pct": 0.0, "direction": "exact",
        },
        ("fix-002", "anthropic"): {
            "delta": 6, "abs_delta": 6, "delta_pct": 40.0, "direction": "underestimate",
        },
        ("fix-003", "openai"): {
            "delta": -2, "abs_delta": 2, "delta_pct": -100.0, "direction": "overestimate",
        },
        ("fix-003", "anthropic"): {
            "delta": 5, "abs_delta": 5, "delta_pct": 62.5, "direction": "underestimate",
        },
    }


def _load_all(schema):
    prompts = read_jsonl_validated(
        FIXTURES / "raw_prompts.jsonl",
        row_validator(schema, PROMPTS_SPEC.row_def),
        PROMPTS_SPEC.name,
    )
    offline = read_jsonl_validated(
        FIXTURES / "offline_counts.jsonl",
        row_validator(schema, OFFLINE_SPEC.row_def),
        OFFLINE_SPEC.name,
    )
    empirical = read_jsonl_validated(
        FIXTURES / "empirical_counts.jsonl",
        row_validator(schema, EMPIRICAL_SPEC.row_def),
        EMPIRICAL_SPEC.name,
    )
    return prompts, offline, empirical


def test_build_dataframe_shapes_and_columns(schema):
    df = build_dataframe(*_load_all(schema))
    assert len(df) == 6
    required = {
        "prompt_id", "source", "text", "text_len_chars", "text_len_words",
        "language", "domain", "collected_at",
        "provider", "format", "model",
        "offline_count", "tokenizer_version", "offline_ts",
        "empirical_count", "is_oracle", "empirical_source", "endpoint", "empirical_ts",
        "delta", "delta_pct", "abs_delta", "direction",
    }
    missing = required - set(df.columns)
    assert not missing, f"missing columns: {missing}"


def test_build_dataframe_delta_columns_match_hand_computed(schema):
    df = build_dataframe(*_load_all(schema))
    expected = _expected_deltas()
    for (pid, prov), want in expected.items():
        sub = df[(df["prompt_id"] == pid) & (df["provider"] == prov)]
        assert len(sub) == 1, f"expected single row for ({pid}, {prov})"
        row = sub.iloc[0]
        assert int(row["delta"]) == want["delta"]
        assert int(row["abs_delta"]) == want["abs_delta"]
        assert math.isclose(
            float(row["delta_pct"]), float(want["delta_pct"]), rel_tol=0, abs_tol=1e-9
        )
        assert row["direction"] == want["direction"]


def test_build_dataset_cli_writes_parquet(schema, tmp_path):
    """End-to-end: run scripts/build_dataset.py as a subprocess and read the Parquet."""
    out = tmp_path / "atlas.parquet"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "build_dataset.py"),
        "--raw", str(FIXTURES / "raw_prompts.jsonl"),
        "--offline", str(FIXTURES / "offline_counts.jsonl"),
        "--empirical", str(FIXTURES / "empirical_counts.jsonl"),
        "--out", str(out),
        "--schema", str(SCHEMA_PATH),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, (
        f"build_dataset.py failed (rc={result.returncode})\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert out.exists()
    df = pd.read_parquet(out)
    assert len(df) == 6
    assert set(df["direction"].unique()) <= {"underestimate", "overestimate", "exact"}


# --------------------------------------------------------------------------- #
# lockfile                                                                    #
# --------------------------------------------------------------------------- #


def test_lockfile_payload_has_required_fields(tmp_path):
    payload = build_lockfile()
    for key in (
        "schema_version",
        "generated_at",
        "dataset_commit",
        "python",
        "tokenizer_versions",
        "api_endpoints",
        "dependencies",
    ):
        assert key in payload, f"missing key in lockfile payload: {key}"
    assert (
        payload["schema_version"] == "https://github.com/faraa2m/llm-tokens-atlas/schema/v1.json"
    )
    assert set(payload["tokenizer_versions"]).issuperset(
        {"openai", "anthropic", "google", "mistral", "cohere"}
    )
    assert payload["api_endpoints"]["anthropic"]["endpoint"].startswith("https://")
    assert isinstance(payload["dependencies"], list) and len(payload["dependencies"]) > 0


def test_lockfile_generator_writes_valid_json(tmp_path):
    """End-to-end: run scripts/lockfile.py and round-trip the JSON."""
    out = tmp_path / "lockfile.json"
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "lockfile.py"), "--out", str(out)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, (
        f"lockfile.py failed (rc={result.returncode})\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert out.exists() and out.stat().st_size > 0
    with out.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["schema_version"] is not None
    assert payload["generated_at"].endswith("Z")
