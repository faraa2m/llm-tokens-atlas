"""Tests for `llm_tokens_atlas/tokenometer_bridge.py`.

These tests exercise the real tokenometer CLI by default (resolved via the
sibling-repo pinfile written by `llm_tokens_atlas/install_tokenometer.sh`, an env
override, or `tokenometer` on PATH). When no CLI can be located, every test
in this module is skipped with one message — that keeps the rest of the
test suite green on a clean clone that has not yet run `make install`.

Why we test against the real CLI instead of mocking
====================================================
The bridge's only job is to be a thin, predictable proxy onto the
tokenometer CLI. Mocking the subprocess would test the mock, not the
bridge. The real CLI is fast (~250 ms cold start) and deterministic in
offline mode (no network calls), so we just run it.

What we deliberately do NOT test here
=====================================
- Empirical counting against the real provider APIs. That requires API
  keys, costs money on some providers, and is non-deterministic across
  provider-side tokenizer updates. The bridge's empirical path is
  validated by sibling agents' empirical-count scripts under their own
  integration tests.
- Exact token counts for specific prompts. Tokenometer owns the tokenizer
  catalog and updates it independently of atlas; pinning specific counts
  here would create false failures whenever upstream bumps a tokenizer.
  We test shape (int, non-negative, deterministic) instead of exact values.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from llm_tokens_atlas.tokenometer_bridge import (  # noqa: E402
    ATLAS_FORMATS,
    DEFAULT_MODEL_BY_PROVIDER,
    PROVIDERS,
    BatchItem,
    BatchResult,
    MissingApiKeyError,
    TokenometerNotInstalledError,
    _infer_provider,
    _normalize_format,
    count_offline,
    count_offline_batch,
    list_formats,
    list_models,
    list_providers,
)

# ---------------------------------------------------------------------------
# CLI-availability gate
# ---------------------------------------------------------------------------


def _cli_available() -> bool:
    """True if the tokenometer CLI is locatable in this environment.

    Mirrors the bridge's own resolution path — we attempt one harmless
    invocation and treat any locate-time error as "unavailable".
    """
    try:
        from llm_tokens_atlas.tokenometer_bridge import _resolve_cli

        _resolve_cli()
    except TokenometerNotInstalledError:
        return False
    return True


cli_required = pytest.mark.skipif(
    not _cli_available(),
    reason=(
        "tokenometer CLI not locatable; run `bash llm_tokens_atlas/install_tokenometer.sh` "
        "or set TOKENOMETER_CLI."
    ),
)


# ---------------------------------------------------------------------------
# Pure-Python tests (no CLI required)
# ---------------------------------------------------------------------------


def test_providers_constant() -> None:
    """The provider set is the canonical five from tokenometer's types.ts."""
    assert set(PROVIDERS) == {"anthropic", "cohere", "google", "mistral", "openai"}


def test_atlas_formats_constant() -> None:
    """Atlas spells 'plain' while tokenometer spells 'text'; both must round-trip."""
    assert set(ATLAS_FORMATS) == {"plain", "markdown", "json", "xml", "yaml"}


def test_normalize_format_plain_to_text() -> None:
    assert _normalize_format("plain") == "text"


def test_normalize_format_passes_through() -> None:
    for fmt in ("json", "yaml", "xml", "markdown", "text"):
        assert _normalize_format(fmt) == fmt


def test_normalize_format_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown format"):
        _normalize_format("toml")


def test_infer_provider_classifies_known_prefixes() -> None:
    cases = {
        "claude-opus-4-7": "anthropic",
        "claude-haiku-4-5": "anthropic",
        "gpt-4o": "openai",
        "gpt-4o-mini": "openai",
        "o1-mini": "openai",
        "gemini-2.5-pro": "google",
        "command-r": "cohere",
        "command-r-plus": "cohere",
        "mistral-large-2411": "mistral",
        "codestral-2501": "mistral",
        "pixtral-12b": "mistral",
        "ministral-3b": "mistral",
        "open-mistral-7b": "mistral",
        "open-mixtral-8x22b": "mistral",
    }
    for model_id, expected in cases.items():
        assert _infer_provider(model_id) == expected, model_id


def test_default_model_table_has_every_provider() -> None:
    """Every provider needs a default model for `count_offline(..., model=None)`."""
    assert set(DEFAULT_MODEL_BY_PROVIDER) == set(PROVIDERS)


def test_count_empirical_mistral_raises_without_calling_cli() -> None:
    """Mistral has no empirical endpoint; the bridge should fail fast.

    No CLI is invoked: the precheck in `_require_empirical_key` catches it.
    """
    from llm_tokens_atlas.tokenometer_bridge import count_empirical

    with pytest.raises(MissingApiKeyError, match="mistral has no public"):
        count_empirical("hello", "mistral", "plain")


def test_count_offline_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unknown provider"):
        count_offline("hello", "deepseek", "plain")


def test_count_offline_rejects_unknown_format() -> None:
    with pytest.raises(ValueError, match="Unknown format"):
        count_offline("hello", "openai", "toml")


# ---------------------------------------------------------------------------
# CLI-backed tests (skipped without tokenometer)
# ---------------------------------------------------------------------------


@cli_required
def test_count_offline_returns_non_negative_int() -> None:
    """`count_offline("hello world", "openai", "plain", "gpt-4o")` returns int >= 0.

    This is the canonical smoke test from the brief — minimal call surface
    against the most-trusted tokenizer in the catalog (tiktoken o200k_base).
    """
    n = count_offline("hello world", "openai", "plain", "gpt-4o")
    assert isinstance(n, int)
    assert n >= 0


@cli_required
def test_count_offline_is_deterministic() -> None:
    """Same call twice returns the same value.

    Offline counting is a pure function of (text, model, format) given a
    fixed tokenometer version. If this ever fails, either the tokenizer is
    nondeterministic (a serious tokenometer bug) or the bridge is leaking
    state across calls (a serious bridge bug).
    """
    first = count_offline("hello world", "openai", "plain", "gpt-4o")
    second = count_offline("hello world", "openai", "plain", "gpt-4o")
    assert first == second


@cli_required
def test_count_offline_default_model_resolves() -> None:
    """`model=None` should pick a sensible default and still produce a count."""
    n = count_offline("hello world", "openai", "plain", model=None)
    assert n >= 0


@cli_required
def test_count_offline_anthropic_is_approximate() -> None:
    """Anthropic offline mode uses cl100k_base as a proxy — still returns an int.

    We don't pin the *value* (cl100k_base for "hello world" is small but
    tokenometer's approximation is allowed to evolve); we pin that the
    bridge does not silently choke on the approximate flag.
    """
    n = count_offline("hello world", "anthropic", "plain", "claude-opus-4-7")
    assert n >= 0


@cli_required
def test_count_offline_format_axis_varies() -> None:
    """Wrapping a prompt in JSON inflates the byte count, so tokens should not decrease.

    Tokenometer is told to wrap the prompt itself (we pass `format="json"`),
    so the input grows from ~13 chars to ~30+. We assert >= rather than >
    because tokenometer's offline tokenizer for some providers may collapse
    structural punctuation into single tokens that happen to net out — but
    it should not shrink relative to plain text.
    """
    plain = count_offline("hello world", "openai", "plain", "gpt-4o")
    wrapped = count_offline("hello world", "openai", "json", "gpt-4o")
    assert wrapped >= plain


@cli_required
def test_list_providers_returns_canonical_five() -> None:
    """list_providers() returns a non-empty list including the five canonical providers."""
    result = list_providers()
    assert isinstance(result, list)
    assert len(result) >= 5
    for p in ("anthropic", "openai", "google", "mistral", "cohere"):
        assert p in result, p


@cli_required
def test_list_formats_returns_non_empty_atlas_names() -> None:
    """list_formats() uses atlas-side names ('plain', not 'text')."""
    result = list_formats()
    assert isinstance(result, list)
    assert len(result) > 0
    assert "plain" in result
    assert "text" not in result  # atlas naming hides this internal alias


@cli_required
def test_list_models_returns_non_empty_full_catalog() -> None:
    """list_models() returns the live tokenometer catalog (non-empty, sorted)."""
    result = list_models()
    assert isinstance(result, list)
    assert len(result) > 0
    # Sorted ascending — easier for callers to diff against.
    assert result == sorted(result)


@cli_required
def test_list_models_filtered_by_provider_is_subset() -> None:
    """list_models(provider) is a non-empty subset of the unfiltered catalog."""
    full = set(list_models())
    for provider in PROVIDERS:
        scoped = list_models(provider)
        assert isinstance(scoped, list)
        assert set(scoped) <= full, f"{provider} returned models outside the full catalog"
        # Anthropic, Google, OpenAI, Cohere all have at least one known model
        # in any healthy tokenometer install; only Mistral might come up
        # empty if tokenlens hasn't yet shipped Mistral entries (it has at
        # 1.0.1, but we don't want a false negative if upstream changes).
        if provider in {"anthropic", "openai", "google", "cohere"}:
            assert len(scoped) > 0, f"{provider} has no models in the catalog"


@cli_required
def test_list_models_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unknown provider"):
        list_models("deepseek")


# ---------------------------------------------------------------------------
# Batch path
# ---------------------------------------------------------------------------


@cli_required
def test_count_offline_batch_empty_input() -> None:
    """Empty batch returns empty list without invoking the CLI."""
    assert count_offline_batch([]) == []


@cli_required
def test_count_offline_batch_preserves_order() -> None:
    """Output order matches input order even when items are bucketed by bucket key."""
    items = [
        BatchItem(text="first prompt", provider="openai", format="plain"),
        BatchItem(text="second prompt", provider="anthropic", format="plain"),
        BatchItem(text="third prompt", provider="openai", format="plain"),
    ]
    results = count_offline_batch(items)
    assert len(results) == 3
    for r in results:
        assert isinstance(r, BatchResult)
        assert r.tokens >= 0
    # Slot 0 and slot 2 are both OpenAI (same bucket) but should be in the
    # original positions in the output list.
    assert results[0].provider == "openai"
    assert results[1].provider == "anthropic"
    assert results[2].provider == "openai"


@cli_required
def test_count_offline_batch_matches_single_call() -> None:
    """A batch of 1 returns the same count as the single-call path."""
    single = count_offline("hello world", "openai", "plain", "gpt-4o")
    batch = count_offline_batch(
        [BatchItem(text="hello world", provider="openai", format="plain", model="gpt-4o")]
    )
    assert batch[0].tokens == single


@cli_required
def test_count_offline_batch_carries_tokenizer_metadata() -> None:
    """Batch results carry the tokenizer name so callers can audit each count's source."""
    results = count_offline_batch(
        [
            BatchItem(
                text="hello world", provider="openai", format="plain", model="gpt-4o"
            ),
            BatchItem(
                text="hello world",
                provider="anthropic",
                format="plain",
                model="claude-opus-4-7",
            ),
        ]
    )
    # OpenAI uses tiktoken o200k_base (exact).
    assert results[0].tokenizer == "o200k_base"
    assert results[0].approximate is False
    # Anthropic uses cl100k_base as an offline proxy (approximate).
    assert results[1].tokenizer == "cl100k_base"
    assert results[1].approximate is True
