"""Tests for ``scripts/count_empirical.py``.

Coverage strategy
-----------------

We exercise three layers:

1. **OpenAI tiktoken path** — runs end-to-end with no API key. This proves
   the schema/round-trip + the CLI's resume semantics + the writer.

2. **Anthropic counter** — mocked via ``AsyncAnthropic`` patch so we never
   hit the wire. Verifies the count → row mapping and the ``is_oracle/source``
   markers.

3. **Credential gating** — provider keys absent => no rows emitted, no crash.

The tests do *not* exercise Google/Mistral/Cohere against the network.
Mistral is tested at unit level (its tokenizer is local; just verify the
counter returns a row). Google/Cohere counters require keys and are
gated; their construction-without-key path is covered.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import count_empirical as ce  # noqa: E402

FIXTURE = REPO_ROOT / "tests" / "fixtures" / "tiny_prompts.jsonl"


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _make_cell(prompt_id: str = "p1", provider: str = "openai", fmt: str = "plain",
               model: str = "gpt-4o", text: str = "hello") -> ce.Cell:
    return ce.Cell(
        prompt_id=prompt_id,
        prompt_text=text,
        provider=provider,
        fmt=fmt,
        model=model,
    )


# --------------------------------------------------------------------------- #
# OpenAI (tiktoken) — smoke / schema / round-trip                              #
# --------------------------------------------------------------------------- #


def test_openai_counter_returns_schema_aligned_row() -> None:
    counter = ce.OpenAICounter()
    cell = _make_cell(model="gpt-4o", text="hello world")
    wrapped = "hello world"
    row = asyncio.run(counter.count(cell, wrapped))
    # Schema check (matches data/schema.json $defs.empiricalCountRow)
    assert row.prompt_id == "p1"
    assert row.provider == "openai"
    assert row.format == "plain"
    assert row.model == "gpt-4o"
    assert row.is_oracle is True
    assert row.source == "tiktoken"
    assert row.empirical_count > 0
    assert "tiktoken==" in row.endpoint
    assert "encoding=" in row.endpoint
    assert row.ts.endswith("Z")
    # Round-trip via JSONL
    parsed = json.loads(row.to_jsonl())
    assert parsed["empirical_count"] == row.empirical_count
    assert parsed["is_oracle"] is True


def test_openai_counter_picks_cl100k_for_legacy_model() -> None:
    counter = ce.OpenAICounter()
    cell = _make_cell(model="gpt-4-turbo", text="abc def")
    row = asyncio.run(counter.count(cell, "abc def"))
    # gpt-4-turbo uses cl100k_base; tiktoken's encoding_for_model maps it.
    assert "cl100k_base" in row.endpoint


# --------------------------------------------------------------------------- #
# Anthropic — mocked SDK                                                       #
# --------------------------------------------------------------------------- #


def test_anthropic_counter_uses_count_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch ``AsyncAnthropic`` so we can assert call args + emit a row."""

    fake_result = MagicMock()
    fake_result.input_tokens = 142

    fake_messages = MagicMock()
    fake_messages.count_tokens = AsyncMock(return_value=fake_result)

    fake_client = MagicMock()
    fake_client.messages = fake_messages

    fake_class = MagicMock(return_value=fake_client)

    # The counter imports AsyncAnthropic lazily inside ``_client_lazy``.
    # We patch the import target at the module location it's imported from.
    monkeypatch.setattr("anthropic.AsyncAnthropic", fake_class, raising=False)

    counter = ce.AnthropicCounter(api_key="sk-test")
    assert counter.available() is True

    cell = _make_cell(provider="anthropic", model="claude-opus-4-7", text="hi")
    row = asyncio.run(counter.count(cell, "hi"))

    assert row.empirical_count == 142
    assert row.is_oracle is True
    assert row.source == "api"
    assert row.provider == "anthropic"
    assert row.model == "claude-opus-4-7"
    assert "anthropic.messages.count_tokens" in row.endpoint

    # SDK call: client built once, count_tokens hit once.
    fake_class.assert_called_once_with(api_key="sk-test")
    fake_messages.count_tokens.assert_awaited_once()
    call_kwargs = fake_messages.count_tokens.await_args.kwargs
    assert call_kwargs["model"] == "claude-opus-4-7"
    assert call_kwargs["messages"] == [{"role": "user", "content": "hi"}]


def test_anthropic_counter_skips_when_key_missing() -> None:
    counter = ce.AnthropicCounter(api_key=None)
    assert counter.available() is False


# --------------------------------------------------------------------------- #
# Mistral — local tokenizer, no key                                            #
# --------------------------------------------------------------------------- #


def test_mistral_counter_returns_row_with_sdk_source() -> None:
    counter = ce.MistralCounter()
    cell = _make_cell(
        provider="mistral",
        model="mistral-large-2407",
        text="hello world",
    )
    row = asyncio.run(counter.count(cell, "hello world"))
    assert row.empirical_count > 0
    assert row.source == "sdk"
    assert row.is_oracle is True
    assert "mistral-common==" in row.endpoint


def test_mistral_counter_resolves_alias() -> None:
    """Unknown model id falls back through MISTRAL_MODEL_ALIASES."""
    counter = ce.MistralCounter()
    cell = _make_cell(
        provider="mistral",
        model="open-mixtral-8x22b",  # alias -> open-mixtral-8x22b-2404
        text="alpha",
    )
    row = asyncio.run(counter.count(cell, "alpha"))
    assert row.empirical_count > 0
    assert "open-mixtral-8x22b-2404" in row.endpoint


# --------------------------------------------------------------------------- #
# build_counters: credential gating                                            #
# --------------------------------------------------------------------------- #


def test_build_counters_drops_providers_without_keys(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("WARNING", logger="count_empirical")
    counters = ce.build_counters(
        ["openai", "mistral", "anthropic", "google", "cohere"],
        env={},  # no keys at all
    )
    # openai + mistral need no creds
    assert set(counters.keys()) == {"openai", "mistral"}
    # And we logged a skip for the credential-gated ones.
    assert any("anthropic" in r.message for r in caplog.records)
    assert any("google" in r.message for r in caplog.records)
    assert any("cohere" in r.message for r in caplog.records)


def test_build_counters_unknown_provider_logged(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("WARNING", logger="count_empirical")
    counters = ce.build_counters(["openai", "nonsense"], env={})
    assert "nonsense" not in counters
    assert any("unknown provider" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
# IO + resume                                                                  #
# --------------------------------------------------------------------------- #


def test_read_prompts_round_trip() -> None:
    prompts = ce.read_prompts(FIXTURE)
    assert len(prompts) == 2
    assert prompts[0]["prompt_id"] == "smoke_001"


def test_read_done_keys_returns_empty_when_no_file(tmp_path: Path) -> None:
    assert ce.read_done_keys(tmp_path / "missing.jsonl") == set()


def test_read_done_keys_loads_existing(tmp_path: Path) -> None:
    p = tmp_path / "progress.jsonl"
    p.write_text(
        json.dumps({
            "prompt_id": "a", "provider": "openai", "format": "plain",
            "model": "gpt-4o", "empirical_count": 3, "is_oracle": True,
            "source": "tiktoken", "endpoint": "x", "ts": "2026-05-10T00:00:00Z",
        }) + "\n",
        encoding="utf-8",
    )
    keys = ce.read_done_keys(p)
    assert ("a", "openai", "plain", "gpt-4o") in keys


def test_finalize_output_dedupes_last_wins(tmp_path: Path) -> None:
    progress = tmp_path / "progress.jsonl"
    out = tmp_path / "empirical.jsonl"
    base = {
        "prompt_id": "x", "provider": "openai", "format": "plain",
        "model": "gpt-4o", "is_oracle": True, "source": "tiktoken",
        "endpoint": "x", "ts": "2026-05-10T00:00:00Z",
    }
    with progress.open("w", encoding="utf-8") as f:
        f.write(json.dumps({**base, "empirical_count": 1}) + "\n")
        f.write(json.dumps({**base, "empirical_count": 2}) + "\n")  # last wins
    n = ce.finalize_output(progress, out)
    assert n == 1
    rows = [json.loads(ln) for ln in out.read_text().splitlines() if ln]
    assert rows[0]["empirical_count"] == 2


# --------------------------------------------------------------------------- #
# End-to-end via run() with OpenAI counter (no network)                        #
# --------------------------------------------------------------------------- #


def test_run_with_openai_only_is_idempotent(tmp_path: Path) -> None:
    """Run the build orchestrator once, then again — second run is a no-op."""
    progress = tmp_path / "progress.jsonl"
    writer = ce.ProgressWriter(progress)

    prompts = ce.read_prompts(FIXTURE)
    counters = ce.build_counters(["openai"], env={})

    try:
        written_first = asyncio.run(
            ce.run(
                prompts=prompts,
                counters=counters,
                models={"openai": ["gpt-4o"]},
                formats=("plain", "json"),
                progress_writer=writer,
                done_keys=set(),
                concurrency=2,
            )
        )
    finally:
        writer.close()

    # 2 prompts × 1 model × 2 formats = 4 rows
    assert written_first == 4

    # Re-run with done_keys derived from the same progress file.
    done = ce.read_done_keys(progress)
    assert len(done) == 4

    writer2 = ce.ProgressWriter(progress)
    try:
        written_second = asyncio.run(
            ce.run(
                prompts=prompts,
                counters=counters,
                models={"openai": ["gpt-4o"]},
                formats=("plain", "json"),
                progress_writer=writer2,
                done_keys=done,
                concurrency=2,
            )
        )
    finally:
        writer2.close()
    assert written_second == 0

    # finalize_output produces a deduped final stream.
    out = tmp_path / "empirical.jsonl"
    final_count = ce.finalize_output(progress, out)
    assert final_count == 4
    rows = [json.loads(ln) for ln in out.read_text().splitlines() if ln]
    for r in rows:
        assert r["provider"] == "openai"
        assert r["source"] == "tiktoken"
        assert r["is_oracle"] is True
        assert r["empirical_count"] > 0


def test_cli_runs_end_to_end_with_openai(tmp_path: Path) -> None:
    """Run the CLI's ``main()`` with --providers=openai. No network needed."""
    out = tmp_path / "empirical.jsonl"
    rc = ce.main(
        [
            "--in", str(FIXTURE),
            "--out", str(out),
            "--providers", "openai",
            "--formats", "plain,json",
            "--concurrency", "2",
            "--log-level", "WARNING",
        ]
    )
    assert rc == 0
    rows = [json.loads(ln) for ln in out.read_text().splitlines() if ln]
    # 2 prompts × {gpt-4o, gpt-4o-mini, gpt-4-turbo} × 2 formats = 12 rows
    assert len(rows) == 12
    for r in rows:
        assert r["provider"] == "openai"
        assert r["is_oracle"] is True
        assert r["empirical_count"] >= 1
