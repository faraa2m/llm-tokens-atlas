"""End-to-end tests for `llm_tokens_atlas/count_offline.py` and `llm_tokens_atlas/format_wrappers.py`.

These tests deliberately exercise the **real** tokenometer CLI (resolved via
the sibling checkout fallback in `TokenometerCli.resolve`). The tokenometer
CLI is offline-only here — no network calls are made — so the tests stay
fully deterministic and cheap.

Skip conditions: if the tokenometer CLI can't be located, every test in this
module is skipped with a single message rather than failing — that way a
fresh clone of the atlas repo without a tokenometer sibling still passes CI
on the non-CLI tests (e.g. wrapper tests in `test_format_wrappers.py`).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_PROMPTS = REPO_ROOT / "tests" / "fixtures" / "raw_prompts.jsonl"

sys.path.insert(0, str(REPO_ROOT))
from llm_tokens_atlas.count_offline import TokenometerCli  # noqa: E402
from llm_tokens_atlas.format_wrappers import ALL_FORMATS  # noqa: E402

EXPECTED_FIELDS = frozenset(
    {
        "prompt_id",
        "provider",
        "format",
        "model",
        "offline_count",
        "tokenizer_version",
        "ts",
    }
)

EXPECTED_PROVIDERS = frozenset(
    {"anthropic", "openai", "google", "mistral", "cohere"}
)


def _cli_available() -> bool:
    """True if the tokenometer CLI is locatable in this environment."""
    try:
        TokenometerCli.resolve(None)
    except FileNotFoundError:
        return False
    return True


cli_required = pytest.mark.skipif(
    not _cli_available(),
    reason="tokenometer CLI not locatable (set TOKENOMETER_CLI or add sibling checkout)",
)


def _run_offline(in_path: Path, out_path: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    """Drive `llm_tokens_atlas/count_offline.py` as a subprocess. Returns the proc."""
    argv = [
        sys.executable,
        str(REPO_ROOT / "llm_tokens_atlas" / "count_offline.py"),
        "--in",
        str(in_path),
        "--out",
        str(out_path),
        *extra,
    ]
    return subprocess.run(
        argv,
        capture_output=True,
        check=False,
        text=True,
        timeout=120,
        cwd=str(REPO_ROOT),
    )


def _read_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


@cli_required
def test_output_schema_matches_brief(tmp_path: Path) -> None:
    """Every output row has exactly the fields documented in the script's docstring."""
    out_path = tmp_path / "offline.jsonl"
    proc = _run_offline(FIXTURE_PROMPTS, out_path)
    assert proc.returncode == 0, f"stderr: {proc.stderr}"

    rows = _read_rows(out_path)
    assert rows, "expected non-empty output"

    # 3 prompts in the fixture x 5 providers x 5 formats = 75 rows.
    n_prompts = sum(1 for _ in FIXTURE_PROMPTS.read_text().splitlines() if _.strip())
    assert len(rows) == n_prompts * 5 * 5

    for row in rows:
        assert set(row.keys()) == EXPECTED_FIELDS, (
            f"row keys diverge from schema; got {set(row.keys())!r}"
        )
        assert isinstance(row["prompt_id"], str) and row["prompt_id"]
        assert row["provider"] in EXPECTED_PROVIDERS
        assert row["format"] in ALL_FORMATS
        assert isinstance(row["model"], str) and row["model"]
        assert isinstance(row["offline_count"], int) and row["offline_count"] >= 0
        # tokenizer_version captures the tokenizer kind, the CLI version, and
        # the pricing-snapshot date — all three are essential to reproduce a
        # historical count.
        assert "@tokenometer-" in row["tokenizer_version"]
        assert "+rates-" in row["tokenizer_version"]
        # ISO-8601 with trailing Z.
        assert row["ts"].endswith("Z") and "T" in row["ts"]


@cli_required
def test_run_is_deterministic(tmp_path: Path) -> None:
    """Same input twice -> byte-identical counts (modulo the `ts` timestamp)."""
    out_a = tmp_path / "offline_a.jsonl"
    out_b = tmp_path / "offline_b.jsonl"
    proc_a = _run_offline(FIXTURE_PROMPTS, out_a)
    proc_b = _run_offline(FIXTURE_PROMPTS, out_b)
    assert proc_a.returncode == 0
    assert proc_b.returncode == 0

    rows_a = _read_rows(out_a)
    rows_b = _read_rows(out_b)
    assert len(rows_a) == len(rows_b)

    def _strip_ts(rows: list[dict]) -> list[dict]:
        return [{k: v for k, v in r.items() if k != "ts"} for r in rows]

    assert _strip_ts(rows_a) == _strip_ts(rows_b), (
        "non-deterministic output across two runs"
    )


@cli_required
def test_formats_produce_distinct_counts(tmp_path: Path) -> None:
    """For a given (prompt, provider, model), at least two formats must differ.

    This is a sanity check on `format_wrappers.py`: if every format collapsed
    to the same wrapped string, the atlas would be measuring nothing
    interesting. We don't require *all* formats to differ (e.g. plain and
    markdown can produce the same count on text that happens not to trigger
    the markdown wrapper's `## prompt` header surcharge in the tokenizer's
    BPE vocabulary), but we do require non-trivial spread.
    """
    out_path = tmp_path / "offline.jsonl"
    proc = _run_offline(FIXTURE_PROMPTS, out_path)
    assert proc.returncode == 0, proc.stderr
    rows = _read_rows(out_path)

    # Group by (prompt_id, provider, model) and check the set of distinct
    # offline_counts across formats has size >= 2 for every group.
    groups: dict[tuple[str, str, str], set[int]] = {}
    for r in rows:
        key = (r["prompt_id"], r["provider"], r["model"])
        groups.setdefault(key, set()).add(r["offline_count"])

    underdiverse = {k: counts for k, counts in groups.items() if len(counts) < 2}
    assert not underdiverse, (
        f"these (prompt, provider, model) groups had identical counts across "
        f"all 5 formats — wrapper or tokenizer collapse: {underdiverse!r}"
    )


@cli_required
def test_formats_flag_subsets(tmp_path: Path) -> None:
    """`--formats markdown,json` produces only those two formats."""
    out_path = tmp_path / "offline.jsonl"
    proc = _run_offline(FIXTURE_PROMPTS, out_path, "--formats", "markdown,json")
    assert proc.returncode == 0
    rows = _read_rows(out_path)
    formats_seen = {r["format"] for r in rows}
    assert formats_seen == {"markdown", "json"}


@cli_required
def test_models_override(tmp_path: Path) -> None:
    """`--models anthropic=claude-haiku-4-5` rebinds Anthropic's model id."""
    out_path = tmp_path / "offline.jsonl"
    proc = _run_offline(
        FIXTURE_PROMPTS,
        out_path,
        "--models",
        "anthropic=claude-haiku-4-5",
    )
    assert proc.returncode == 0, proc.stderr
    rows = _read_rows(out_path)
    anthropic_models = {r["model"] for r in rows if r["provider"] == "anthropic"}
    assert anthropic_models == {"claude-haiku-4-5"}


def test_invalid_format_rejected() -> None:
    """`--formats` rejects unknown values with a clean exit code."""
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "llm_tokens_atlas" / "count_offline.py"),
            "--in",
            str(FIXTURE_PROMPTS),
            "--out",
            "/tmp/never.jsonl",
            "--formats",
            "markdown,bogus",
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=10,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode != 0
    assert "bogus" in proc.stderr or "bogus" in proc.stdout


@cli_required
def test_chatml_special_tokens_do_not_crash(tmp_path: Path) -> None:
    """Prompts containing `<|im_start|>` etc. produce counts via the local fallback.

    Regression for the gpt-tokenizer "Disallowed special token" crash
    documented in ``analysis/PIPELINE_NOTES.md`` Issue 1. The CLI rejects
    ChatML literals by default; the driver catches the failure and falls
    back to local Python encoding with ``allowed_special='all'`` so the
    pipeline still produces a complete cell matrix for that prompt.
    """
    chatml_prompt = (
        '{"prompt_id":"chatml-001","source":"synthetic","text":"hello '
        "<|im_start|>user\\nWhat's up?<|im_end|> normal text here\","
        '"text_len_chars":50,"text_len_words":8,"language":"en",'
        '"domain":"chat","collected_at":"2026-05-10T00:00:00Z"}\n'
    )
    in_path = tmp_path / "chatml.jsonl"
    in_path.write_text(chatml_prompt, encoding="utf-8")

    out_path = tmp_path / "offline.jsonl"
    proc = _run_offline(in_path, out_path)
    assert proc.returncode == 0, f"stderr: {proc.stderr}"

    rows = _read_rows(out_path)
    # 1 prompt × 5 providers × 5 formats = 25 rows even though gpt-tokenizer
    # would normally reject the input.
    assert len(rows) == 25, f"expected 25 rows from the fallback path, got {len(rows)}"
    providers = {r["provider"] for r in rows}
    assert providers == EXPECTED_PROVIDERS, (
        f"all 5 providers must be present in the fallback output; got {providers!r}"
    )
    for row in rows:
        assert row["offline_count"] >= 0
        # The stderr message confirms the fallback fired; we don't assert on
        # it here so the test stays robust if the message text changes.
    # Sanity: a prompt without ChatML markers should still work via the CLI
    # path with byte-identical counts modulo the fallback. Not a full
    # determinism check (covered by `test_run_is_deterministic`), but we
    # do want to confirm the fallback's counts are non-degenerate.
    assert sum(int(r["offline_count"]) for r in rows) > 0
