"""Single source of truth for the per-provider model set the atlas measures.

Both ``count_offline.py`` and ``count_empirical.py`` import :data:`ATLAS_MODELS`
from this module so the offline and empirical drivers always materialise the
same (prompt_id, provider, format, model) cell matrix. Divergence here is the
root cause of the silently-dropped partitions documented in
``analysis/PIPELINE_NOTES.md`` (Issue 2).

Design notes
============

- **One canonical model per provider for the headline calibration.** The
  offline driver historically materialised exactly one model per provider so
  per-cell rows joined unambiguously with the empirical side. We preserve
  that semantic by listing the headline model **first** in each per-provider
  list — callers that want exactly one model can use ``ATLAS_MODELS[provider]
  [0]``.

- **Multi-model lists kept for empirical breadth.** The empirical driver hits
  each provider's free token-count endpoint, which is cheap, so we retain
  the wider per-family lists it shipped. Offline rows for the additional
  models can be produced by the offline driver iterating the full list.

- **Mistral canonical id.** Aligned to ``mistral-large-latest`` — the same id
  the offline driver's previous ``DEFAULT_MODELS`` used. The previous
  empirical list shipped ``mistral-large-2407``, ``mistral-small-2409``, and
  ``open-mistral-nemo-2407``, none of which overlapped with the offline side.
  ``mistral-large-latest`` is the SentencePiece-family id ``tokenometer``'s
  offline tokenizer resolves cleanly and that ``mistral-common``'s
  :data:`MISTRAL_MODEL_ALIASES` maps onto ``mistral-large-2411`` for the
  empirical run, so joining now works.
"""

from __future__ import annotations

from typing import Final

#: Per-provider canonical model list. Headline model is element ``[0]``.
#:
#: When changing this list, also re-run ``make build && make lockfile`` so
#: ``data/lockfile.json`` records the new model set with the next dataset
#: snapshot.
ATLAS_MODELS: Final[dict[str, list[str]]] = {
    "anthropic": [
        "claude-opus-4-7",
        "claude-sonnet-4-6",
        "claude-haiku-4-5",
    ],
    "openai": [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
    ],
    "google": [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-1.5-pro",
    ],
    "mistral": [
        # Aligned with tokenometer offline tokenizer + mistral-common
        # MISTRAL_MODEL_ALIASES so offline/empirical join cleanly.
        "mistral-large-latest",
    ],
    "cohere": [
        "command-r",
        "command-r-plus",
    ],
}

#: Stable iteration order across all drivers + analysis code.
PROVIDER_ORDER: Final[tuple[str, ...]] = (
    "anthropic",
    "openai",
    "google",
    "mistral",
    "cohere",
)


def headline_models() -> dict[str, str]:
    """Return the one canonical (headline) model per provider.

    Used by the offline driver's default mode, which produces exactly one
    row per (prompt, provider, format) cell — the headline calibration.
    """
    return {provider: models[0] for provider, models in ATLAS_MODELS.items()}
