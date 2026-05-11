"""Tests for `scripts/collect_corpus.py`.

This module is intentionally minimal: the corpus-collection script is
network-bound and slow, so we run a single end-to-end smoke test at small
n_total (10 rows) and verify the output JSONL conforms to the
`promptRow` schema declared in `data/schema.json`.

CI marks this test as "slow" via the `@pytest.mark.slow` marker so it can
be skipped in fast unit-test runs. The smoke test is the gate-keeper for
the reproduce target (`make reproduce`) and is required for the
corpus-collection acceptance criteria, so it should NOT be skipped by
default.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = REPO_ROOT / "data" / "schema.json"


@pytest.fixture(scope="module")
def prompt_row_validator() -> Draft202012Validator:
    """Return a JSON-schema validator for individual promptRow objects."""
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    # The schema in data/schema.json is the *top-level* dataset schema; the
    # row schema lives in $defs/promptRow. Re-root and validate against
    # that, using the same $defs as the root so internal $refs resolve.
    row_schema = {
        **schema["$defs"]["promptRow"],
        "$defs": schema["$defs"],
    }
    Draft202012Validator.check_schema(row_schema)
    return Draft202012Validator(row_schema)


def test_collect_corpus_smoke(tmp_path: Path, prompt_row_validator: Draft202012Validator) -> None:
    """End-to-end smoke test: invoke the CLI at n=10 and verify the output.

    The test mirrors the acceptance-criteria invocation but at a tiny
    sample size to keep CI runs under a minute. We:
      1. Invoke `uv run python scripts/collect_corpus.py --n 10
         --out <tmp>/tiny.jsonl --skip-provenance` as a subprocess.
      2. Assert exit code is 0.
      3. Load the JSONL and verify it has the expected number of rows
         (at least 5 — one per source minimum) and that every row
         passes `promptRow` schema validation.
      4. Verify all five expected `source` values appear.
    """
    out_path = tmp_path / "tiny.jsonl"
    cmd = [
        "uv",
        "run",
        "python",
        str(REPO_ROOT / "scripts" / "collect_corpus.py"),
        "--n",
        "10",
        "--out",
        str(out_path),
        "--skip-provenance",
        "--seed",
        "42",
    ]
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, (
        f"collect_corpus.py exited non-zero ({result.returncode}).\n"
        f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
    )
    assert out_path.is_file(), f"output {out_path} not produced"

    rows: list[dict] = []
    with out_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    # At n=10, with five sources, we may produce slightly fewer rows
    # than requested if a source rejects everything; but each source
    # must contribute at least 1 row for the smoke test to be meaningful.
    assert len(rows) >= 5, f"expected ≥5 rows, got {len(rows)}"

    seen_sources = {r["source"] for r in rows}
    expected_sources = {"humaneval", "wildchat-1m", "mt-bench", "wikipedia-en", "github-readmes"}
    assert seen_sources == expected_sources, (
        f"missing sources. expected {expected_sources}, got {seen_sources}"
    )

    # Schema-validate every row.
    for i, row in enumerate(rows):
        errors = sorted(prompt_row_validator.iter_errors(row), key=lambda e: e.path)
        assert not errors, (
            f"row {i} (source={row.get('source')}) failed promptRow validation: "
            f"{[e.message for e in errors]}"
        )


def test_per_source_targets_sum_to_n_total() -> None:
    """The target distribution must sum to n_total even after capping.

    This is a fast, in-process unit test (no network). It pins the
    redistribution policy in `_per_source_targets` so future edits
    don't accidentally short-change the total row count.
    """
    sys.path.insert(0, str(REPO_ROOT))
    from scripts.collect_corpus import (  # noqa: PLC0415
        SOURCE_GITHUB_READMES,
        SOURCE_HUMANEVAL,
        SOURCE_MTBENCH,
        SOURCE_WIKIPEDIA,
        SOURCE_WILDCHAT,
        _per_source_targets,
    )

    for n_total in (500, 1000, 5000):
        targets = _per_source_targets(n_total)
        assert set(targets.keys()) == {
            SOURCE_HUMANEVAL,
            SOURCE_WILDCHAT,
            SOURCE_MTBENCH,
            SOURCE_WIKIPEDIA,
            SOURCE_GITHUB_READMES,
        }
        assert sum(targets.values()) == n_total, (
            f"per-source targets for n_total={n_total} sum to "
            f"{sum(targets.values())}, expected {n_total}: {targets}"
        )
        # Acceptance criterion: ≥ 50 per source at n_total=500.
        if n_total >= 500:
            for src, t in targets.items():
                assert t >= 50, f"source {src} target {t} < 50 at n_total={n_total}"
