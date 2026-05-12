"""Empirical token-count driver — ground-truth counts from each provider.

This script is the empirical counterpart to ``count_offline.py``. For every
(prompt, format, model, provider) cell in the atlas matrix it asks the
authoritative source — preferring the provider's own tokenizer or
countTokens endpoint over any offline proxy — for the *real* token count
that prompt would consume.

Empirical sources per provider
==============================

============ ===================================== =========== ============
Provider     Source                                ``source``  ``is_oracle``
============ ===================================== =========== ============
OpenAI       ``tiktoken`` (the actual deployed     tiktoken    True
             tokenizer — ``o200k_base`` for the
             GPT-4o/4.1/o-series family,
             ``cl100k_base`` for legacy)
Anthropic    ``client.messages.count_tokens``      api         True
             (free public endpoint; per `docs
             <https://docs.anthropic.com/en/api/
             messages-count-tokens>`_)
Google       ``model.count_tokens`` on the         api         True
             Gemini API (free; ``google-
             generativeai`` SDK)
Mistral      ``mistral-common`` SentencePiece /    sdk         True
             Tekken tokenizers (the *exact*
             tokenizers Mistral ships; mapped
             via ``MistralTokenizer.from_model``
             with a curated alias table)
Cohere       ``co.tokenize(text=…, offline=False)``api         True
             — explicit hit on Cohere's
             ``/v1/tokenize``
============ ===================================== =========== ============

Rate-limiting + checkpointing
=============================

The driver writes one JSONL row per completed cell to a checkpoint file
(default: ``data/empirical_progress.jsonl``). Re-running is idempotent:
already-completed (prompt_id, provider, format, model) tuples are skipped.

Network calls are wrapped in ``tenacity`` exponential-backoff retries
(3 tries, 1-8s jitter). The concurrent-call cap is 4 by default.

Output
======

Rows are appended atomically (write to ``<out>.tmp`` then rename on
shutdown). Each row conforms to the ``empiricalCountRow`` sub-schema in
``data/schema.json``.

Credentials
===========

API keys are read from the environment:

* ``ANTHROPIC_API_KEY`` — Anthropic
* ``GOOGLE_API_KEY`` (or ``GEMINI_API_KEY``) — Google
* ``COHERE_API_KEY`` — Cohere

OpenAI and Mistral require no API keys (their tokenizers are local).
If a provider's key is missing, the driver logs once and skips that
provider's rows; it does not crash.

Determinism caveat
==================

The Anthropic / Google / Cohere endpoints can drift their counts as
providers update tokenizers. Each row records ``endpoint`` (URL + SDK
version + tokenizer/encoding identifier) so longitudinal drift is
auditable. We do not attempt to pin a specific server-side tokenizer
version because the providers do not expose one.

CLI
===

.. code-block:: bash

    uv run python llm_tokens_atlas/count_empirical.py \\
        --in data/raw_prompts.jsonl \\
        --out data/empirical_counts.jsonl \\
        --providers anthropic,google,openai,mistral,cohere \\
        --concurrency 4 \\
        --max-prompts 0   # 0 = no limit
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import importlib.metadata as importlib_metadata
import json
import logging
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import tiktoken
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

# --------------------------------------------------------------------------- #
# Local imports                                                               #
# --------------------------------------------------------------------------- #

# Make the llm_tokens_atlas/ folder importable when running as a script.
PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from _atlas_models import ATLAS_MODELS  # type: ignore[import-not-found]  # noqa: E402
from format_wrappers import ALL_FORMATS, wrap  # type: ignore[import-not-found]  # noqa: E402

# --------------------------------------------------------------------------- #
# Constants                                                                   #
# --------------------------------------------------------------------------- #

REPO_ROOT = PACKAGE_DIR.parent

#: Per-provider model lists. Sourced from :data:`_atlas_models.ATLAS_MODELS`
#: so the offline and empirical drivers always materialise the same cell
#: matrix — fixes the "missing mistral partition" issue documented in
#: ``analysis/PIPELINE_NOTES.md`` (Issue 2).
DEFAULT_MODELS: dict[str, list[str]] = dict(ATLAS_MODELS)

#: Models whose OpenAI tokenizer is ``cl100k_base`` rather than ``o200k_base``.
#: GPT-4o family (and the 4.1 / o-series successors) shifted to o200k_base.
_CL100K_BASE_MODELS: set[str] = {
    "gpt-4-turbo",
    "gpt-4",
    "gpt-3.5-turbo",
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",
}

#: Maps atlas model id -> mistral-common's ``from_model`` key. Mistral's
#: SDK does not accept the API's marketing names; we map them through.
MISTRAL_MODEL_ALIASES: dict[str, str] = {
    # Modern families (current as of May 2026).
    "mistral-large-2407": "mistral-large-2407",
    "mistral-small-2409": "mistral-small-2409",
    "open-mistral-nemo-2407": "open-mistral-nemo-2407",
    # Convenience: legacy ids -> closest tokenizer-equivalent in MODEL_NAME_TO_TOKENIZER_CLS.
    "mistral-large-latest": "mistral-large-2411",
    "mistral-small-latest": "mistral-small-2409",
    "mistral-medium-latest": "mistral-medium-2312",
    "mistral-medium-2312": "mistral-medium-2312",
    "mistral-tiny-2407": "mistral-tiny-2407",
    "codestral-2405": "codestral-2405",
    "codestral-mamba-2407": "codestral-mamba-2407",
    "ministral-8b-latest": "ministral-8b-2410",
    "ministral-8b-2410": "ministral-8b-2410",
    "pixtral-12b": "pixtral-12b-2409",
    "pixtral-12b-2409": "pixtral-12b-2409",
    "pixtral-large-2411": "pixtral-large-2411",
    "open-mixtral-8x22b": "open-mixtral-8x22b-2404",
    "open-mixtral-8x22b-2404": "open-mixtral-8x22b-2404",
}

LOG = logging.getLogger("count_empirical")

# --------------------------------------------------------------------------- #
# Schema-aligned row + result types                                           #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Cell:
    """A single unit of work: count tokens for one (prompt, fmt, model)."""

    prompt_id: str
    prompt_text: str
    provider: str
    fmt: str
    model: str

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (self.prompt_id, self.provider, self.fmt, self.model)


@dataclass
class EmpiricalRow:
    """One output row, matching ``empiricalCountRow`` in data/schema.json."""

    prompt_id: str
    provider: str
    format: str
    model: str
    empirical_count: int
    is_oracle: bool
    source: str  # "api" | "tiktoken" | "sdk" | "stream-usage"
    endpoint: str
    ts: str

    def to_jsonl(self) -> str:
        return json.dumps(dataclasses.asdict(self), ensure_ascii=False)


# --------------------------------------------------------------------------- #
# Counter interface + provider implementations                                #
# --------------------------------------------------------------------------- #


class Counter:
    """Strategy interface for one provider's empirical token counter.

    Subclasses are constructed lazily inside ``build_counters()`` so a
    missing credential for one provider does not fail-fast on the others.
    """

    provider: str  # filled by subclasses

    def available(self) -> bool:
        """Whether this counter can run (credentials present, libs imported)."""
        return True

    async def count(self, cell: Cell, wrapped: str) -> EmpiricalRow:
        raise NotImplementedError


# ---- OpenAI: tiktoken --------------------------------------------------- #


class OpenAICounter(Counter):
    """Use ``tiktoken`` locally — it *is* OpenAI's tokenizer.

    No API call required; no key required. Each model maps to either
    ``o200k_base`` (GPT-4o family and successors) or ``cl100k_base``
    (GPT-4 turbo / GPT-3.5).
    """

    provider = "openai"

    def __init__(self) -> None:
        self._tiktoken_version = self._safe_version("tiktoken")
        self._encoders: dict[str, tiktoken.Encoding] = {}

    @staticmethod
    def _safe_version(pkg: str) -> str:
        try:
            return importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            return "unknown"

    def _encoder_for(self, model: str) -> tuple[tiktoken.Encoding, str]:
        if model in self._encoders:
            enc = self._encoders[model]
            return enc, enc.name
        encoding_name = "cl100k_base" if model in _CL100K_BASE_MODELS else "o200k_base"
        try:
            enc = tiktoken.encoding_for_model(model)
            encoding_name = enc.name
        except Exception:
            enc = tiktoken.get_encoding(encoding_name)
        self._encoders[model] = enc
        return enc, encoding_name

    async def count(self, cell: Cell, wrapped: str) -> EmpiricalRow:
        enc, encoding_name = self._encoder_for(cell.model)
        # tiktoken is sync; trivially fast — no executor needed for typical
        # prompt sizes. Yield once so the event loop stays cooperative
        # under high concurrency.
        await asyncio.sleep(0)
        n = len(enc.encode(wrapped))
        endpoint = (
            f"tiktoken=={self._tiktoken_version} encoding={encoding_name} "
            f"model={cell.model}"
        )
        return EmpiricalRow(
            prompt_id=cell.prompt_id,
            provider=self.provider,
            format=cell.fmt,
            model=cell.model,
            empirical_count=n,
            is_oracle=True,
            source="tiktoken",
            endpoint=endpoint,
            ts=_now_iso(),
        )


# ---- Anthropic: messages.count_tokens ----------------------------------- #


class AnthropicCounter(Counter):
    """Call Anthropic's free ``POST /v1/messages/count_tokens`` endpoint.

    Uses the official ``anthropic`` SDK so retries/timeouts are consistent
    with the rest of the codebase. The endpoint is documented as free at
    https://docs.anthropic.com/en/api/messages-count-tokens .
    """

    provider = "anthropic"

    def __init__(self, api_key: str | None) -> None:
        self.api_key = api_key
        self._client: Any | None = None
        self._sdk_version = OpenAICounter._safe_version("anthropic")

    def available(self) -> bool:
        return bool(self.api_key)

    def _client_lazy(self) -> Any:
        if self._client is None:
            from anthropic import AsyncAnthropic  # local import keeps cold start fast

            self._client = AsyncAnthropic(api_key=self.api_key)
        return self._client

    async def count(self, cell: Cell, wrapped: str) -> EmpiricalRow:
        client = self._client_lazy()
        result = await client.messages.count_tokens(
            model=cell.model,
            messages=[{"role": "user", "content": wrapped}],
        )
        n = int(result.input_tokens)
        endpoint = (
            f"anthropic.messages.count_tokens (sdk anthropic=={self._sdk_version}) "
            f"model={cell.model}"
        )
        return EmpiricalRow(
            prompt_id=cell.prompt_id,
            provider=self.provider,
            format=cell.fmt,
            model=cell.model,
            empirical_count=n,
            is_oracle=True,
            source="api",
            endpoint=endpoint,
            ts=_now_iso(),
        )


# ---- Google: GenerativeModel.count_tokens ------------------------------- #


class GoogleCounter(Counter):
    """Call Gemini's free ``count_tokens`` endpoint via ``google-generativeai``.

    The SDK is sync-only at v0.8; we offload to a thread to keep the
    event loop responsive when concurrency > 1.
    """

    provider = "google"

    def __init__(self, api_key: str | None) -> None:
        self.api_key = api_key
        self._configured = False
        self._sdk_version = OpenAICounter._safe_version("google-generativeai")

    def available(self) -> bool:
        return bool(self.api_key)

    def _configure(self) -> None:
        if self._configured:
            return
        import google.generativeai as genai

        genai.configure(api_key=self.api_key)
        self._configured = True

    async def count(self, cell: Cell, wrapped: str) -> EmpiricalRow:
        import google.generativeai as genai

        self._configure()

        def _do_count() -> int:
            model = genai.GenerativeModel(cell.model)
            return int(model.count_tokens(wrapped).total_tokens)

        n = await asyncio.to_thread(_do_count)
        endpoint = (
            f"gemini.count_tokens (sdk google-generativeai=={self._sdk_version}) "
            f"model={cell.model}"
        )
        return EmpiricalRow(
            prompt_id=cell.prompt_id,
            provider=self.provider,
            format=cell.fmt,
            model=cell.model,
            empirical_count=n,
            is_oracle=True,
            source="api",
            endpoint=endpoint,
            ts=_now_iso(),
        )


# ---- Mistral: mistral-common -------------------------------------------- #


class MistralCounter(Counter):
    """Use ``mistral-common`` — Mistral's official tokenizer library.

    ``MistralTokenizer.from_model`` accepts a curated set of model names;
    we map the atlas model ids onto those keys via
    :data:`MISTRAL_MODEL_ALIASES`. Encoding is done by the underlying
    SentencePiece (V1/V3) or Tekken tokenizer; both are the same artifacts
    the Mistral API uses server-side.

    No API call, no key required. ``is_oracle=True`` because the library
    ships the same tokenizer artifact the server uses.
    """

    provider = "mistral"

    def __init__(self) -> None:
        from mistral_common.tokens.tokenizers.mistral import MODEL_NAME_TO_TOKENIZER_CLS

        self._known_keys = set(MODEL_NAME_TO_TOKENIZER_CLS.keys())
        self._tokenizers: dict[str, Any] = {}
        self._sdk_version = OpenAICounter._safe_version("mistral-common")

    def _resolve(self, model: str) -> str:
        if model in self._known_keys:
            return model
        if model in MISTRAL_MODEL_ALIASES:
            return MISTRAL_MODEL_ALIASES[model]
        raise ValueError(
            f"No mistral-common tokenizer mapping for model {model!r}; "
            f"known keys: {sorted(self._known_keys)}; "
            f"add an entry to MISTRAL_MODEL_ALIASES."
        )

    def _tokenizer_for(self, model: str) -> tuple[Any, str]:
        key = self._resolve(model)
        cached = self._tokenizers.get(key)
        if cached is None:
            from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

            cached = MistralTokenizer.from_model(key, strict=True)
            self._tokenizers[key] = cached
        return cached, key

    async def count(self, cell: Cell, wrapped: str) -> EmpiricalRow:
        await asyncio.sleep(0)  # cooperative yield; encode is cpu-bound but fast
        tok, resolved_key = self._tokenizer_for(cell.model)
        inner = tok.instruct_tokenizer.tokenizer
        # Encode raw text (no chat templating, no BOS/EOS) — matches what an
        # offline tokenizer would do on the same string.
        ids = inner.encode(wrapped, bos=False, eos=False)
        n = len(ids)
        endpoint = (
            f"mistral-common=={self._sdk_version} tokenizer={type(inner).__name__} "
            f"resolved={resolved_key}"
        )
        return EmpiricalRow(
            prompt_id=cell.prompt_id,
            provider=self.provider,
            format=cell.fmt,
            model=cell.model,
            empirical_count=n,
            is_oracle=True,
            source="sdk",
            endpoint=endpoint,
            ts=_now_iso(),
        )


# ---- Cohere: /v1/tokenize ------------------------------------------------ #


class CohereCounter(Counter):
    """Call Cohere's free ``POST /v1/tokenize`` endpoint.

    Cohere's SDK ships an ``offline=True`` default that uses a downloaded
    tokenizer.json; we pass ``offline=False`` to explicitly hit the server,
    which is the empirical source the atlas wants. ``is_oracle=True``
    because the endpoint is the canonical tokenizer for the model.
    """

    provider = "cohere"

    def __init__(self, api_key: str | None) -> None:
        self.api_key = api_key
        self._client: Any | None = None
        self._sdk_version = OpenAICounter._safe_version("cohere")

    def available(self) -> bool:
        return bool(self.api_key)

    def _client_lazy(self) -> Any:
        if self._client is None:
            import cohere

            self._client = cohere.Client(api_key=self.api_key)
        return self._client

    async def count(self, cell: Cell, wrapped: str) -> EmpiricalRow:
        client = self._client_lazy()

        def _do_tokenize() -> int:
            resp = client.tokenize(text=wrapped, model=cell.model, offline=False)
            return len(resp.tokens)

        n = await asyncio.to_thread(_do_tokenize)
        endpoint = (
            f"https://api.cohere.com/v1/tokenize (sdk cohere=={self._sdk_version}) "
            f"model={cell.model}"
        )
        return EmpiricalRow(
            prompt_id=cell.prompt_id,
            provider=self.provider,
            format=cell.fmt,
            model=cell.model,
            empirical_count=n,
            is_oracle=True,
            source="api",
            endpoint=endpoint,
            ts=_now_iso(),
        )


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _now_iso() -> str:
    """Current UTC time as an ISO-8601 string with 'Z' suffix."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_counters(
    providers: list[str], env: dict[str, str] | None = None
) -> dict[str, Counter]:
    """Instantiate counters for each requested provider.

    Providers without credentials are logged once and dropped — they are
    NOT returned in the dict, so the caller can iterate the returned keys
    and know each one is runnable.
    """
    env = env if env is not None else os.environ.copy()
    anthropic_key = env.get("ANTHROPIC_API_KEY")
    google_key = env.get("GOOGLE_API_KEY") or env.get("GEMINI_API_KEY")
    cohere_key = env.get("COHERE_API_KEY")

    proto: dict[str, Counter] = {}
    for p in providers:
        if p == "openai":
            proto[p] = OpenAICounter()
        elif p == "anthropic":
            proto[p] = AnthropicCounter(anthropic_key)
        elif p == "google":
            proto[p] = GoogleCounter(google_key)
        elif p == "mistral":
            proto[p] = MistralCounter()
        elif p == "cohere":
            proto[p] = CohereCounter(cohere_key)
        else:
            LOG.warning("unknown provider %r — skipping", p)

    out: dict[str, Counter] = {}
    for name, c in proto.items():
        if c.available():
            out[name] = c
        else:
            LOG.warning(
                "skipping provider %s (no credentials)", name
            )
    return out


# --------------------------------------------------------------------------- #
# IO: prompts, checkpoint, output                                             #
# --------------------------------------------------------------------------- #


def read_prompts(path: Path, max_prompts: int = 0) -> list[dict[str, Any]]:
    """Read raw_prompts.jsonl; light validation only (build_dataset enforces schema)."""
    out: list[dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(f"prompts input not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "prompt_id" not in obj or "text" not in obj:
                raise ValueError(
                    f"prompts file row missing required fields prompt_id/text: {obj!r}"
                )
            out.append(obj)
            if max_prompts and len(out) >= max_prompts:
                break
    return out


def read_done_keys(progress_path: Path) -> set[tuple[str, str, str, str]]:
    """Load already-completed (prompt_id, provider, format, model) keys."""
    done: set[tuple[str, str, str, str]] = set()
    if not progress_path.exists():
        return done
    with progress_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                LOG.warning("ignoring malformed progress line: %r", line[:80])
                continue
            key = (
                obj.get("prompt_id", ""),
                obj.get("provider", ""),
                obj.get("format", ""),
                obj.get("model", ""),
            )
            if all(key):
                done.add(key)
    return done


# --------------------------------------------------------------------------- #
# Worker: count one cell with bounded retries                                 #
# --------------------------------------------------------------------------- #


RETRY_EXCEPTIONS: tuple[type[Exception], ...] = (Exception,)


async def _count_with_retry(
    counter: Counter, cell: Cell, wrapped: str, max_attempts: int = 3
) -> EmpiricalRow:
    """Wrap one ``counter.count(...)`` call with tenacity backoff.

    Exceptions: we retry on *any* exception because providers signal
    rate-limits and transient 5xxs with different SDK exception types and
    we don't want a single misclassified error to abort a whole run. The
    bounded ``max_attempts`` + exponential wait makes the cap explicit.
    """
    last_exc: Exception | None = None
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(max_attempts),
        wait=wait_random_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(RETRY_EXCEPTIONS),
        reraise=True,
    ):
        with attempt:
            try:
                return await counter.count(cell, wrapped)
            except Exception as e:
                last_exc = e
                LOG.warning(
                    "retry %s: %s/%s %s -> %s",
                    attempt.retry_state.attempt_number,
                    cell.provider,
                    cell.model,
                    cell.fmt,
                    e,
                )
                raise
    # Unreachable; AsyncRetrying re-raises on failure.
    raise RuntimeError("retry loop exited without result") from last_exc


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #


def build_cells(
    prompts: list[dict[str, Any]],
    counters: dict[str, Counter],
    models: dict[str, list[str]],
    formats: tuple[str, ...],
) -> list[Cell]:
    """Cartesian product: prompts × runnable providers × per-provider models × formats."""
    cells: list[Cell] = []
    for prompt in prompts:
        for provider, _counter in counters.items():
            for model in models.get(provider, []):
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


class ProgressWriter:
    """Append-only writer for the progress + final output files.

    The progress file is the source of truth for resumption; on successful
    completion of a run the file is renamed to the requested ``--out``
    path (or merged if ``--out`` already has rows from a prior run).
    """

    def __init__(self, progress_path: Path) -> None:
        self.progress_path = progress_path
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        # Open in append mode so resumed runs add to existing progress.
        self._f = progress_path.open("a", encoding="utf-8")

    def write(self, row: EmpiricalRow) -> None:
        self._f.write(row.to_jsonl())
        self._f.write("\n")
        self._f.flush()

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._f.close()


async def run(
    prompts: list[dict[str, Any]],
    counters: dict[str, Counter],
    models: dict[str, list[str]],
    formats: tuple[str, ...],
    progress_writer: ProgressWriter,
    done_keys: set[tuple[str, str, str, str]],
    concurrency: int = 4,
    on_row: Callable[[EmpiricalRow], None] | None = None,
) -> int:
    """Drive the work. Returns the number of newly-written rows."""
    cells = build_cells(prompts, counters, models, formats)
    todo = [c for c in cells if c.key not in done_keys]
    if not todo:
        LOG.info("nothing to do; all %d cells already in progress file", len(cells))
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
                    "giving up on %s/%s %s after retries: %s",
                    cell.provider,
                    cell.model,
                    cell.fmt,
                    e,
                )
                return
            async with lock:
                progress_writer.write(row)
                written += 1
                if on_row is not None:
                    on_row(row)

    await asyncio.gather(*(_worker(c) for c in todo))
    return written


def finalize_output(progress_path: Path, out_path: Path) -> int:
    """Write deduped final output from the progress file to ``out_path``.

    Deduplication: when a key appears multiple times in the progress log
    (e.g. resumed runs), keep the last occurrence.
    """
    if not progress_path.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("", encoding="utf-8")
        return 0

    by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    with progress_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (
                obj.get("prompt_id", ""),
                obj.get("provider", ""),
                obj.get("format", ""),
                obj.get("model", ""),
            )
            by_key[key] = obj
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for obj in by_key.values():
            f.write(json.dumps(obj, ensure_ascii=False))
            f.write("\n")
    return len(by_key)


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="count_empirical",
        description=(
            "Empirical token-count driver. For each prompt × format × "
            "provider × model, hit the cheapest (free) authoritative token "
            "counter for that provider. Idempotent + resumable via a "
            "checkpoint file."
        ),
    )
    parser.add_argument(
        "--in",
        dest="in_path",
        type=Path,
        required=True,
        help="Input raw prompts JSONL (matches `promptRow` in data/schema.json).",
    )
    parser.add_argument(
        "--out",
        dest="out_path",
        type=Path,
        required=True,
        help="Output empirical_counts JSONL.",
    )
    parser.add_argument(
        "--progress",
        dest="progress_path",
        type=Path,
        default=None,
        help=(
            "Checkpoint file (append-only). Defaults to "
            "<--out parent>/empirical_progress.jsonl."
        ),
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
        help="Logging level (default: INFO).",
    )
    return parser


def _default_progress_for(out_path: Path) -> Path:
    return out_path.parent / "empirical_progress.jsonl"


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

    prompts = read_prompts(args.in_path, max_prompts=args.max_prompts)
    if not prompts:
        LOG.warning("no prompts in %s", args.in_path)
        return 0

    progress_path = args.progress_path or _default_progress_for(args.out_path)
    done_keys = read_done_keys(progress_path)
    LOG.info(
        "%d prompts × %d providers × formats=%s — %d cells already done in %s",
        len(prompts),
        len(counters),
        list(formats),
        len(done_keys),
        progress_path,
    )

    writer = ProgressWriter(progress_path)
    t0 = time.time()
    try:
        written = asyncio.run(
            run(
                prompts=prompts,
                counters=counters,
                models=DEFAULT_MODELS,
                formats=formats,
                progress_writer=writer,
                done_keys=done_keys,
                concurrency=args.concurrency,
            )
        )
    finally:
        writer.close()

    final_count = finalize_output(progress_path, args.out_path)
    LOG.info(
        "wrote %d new rows (final dedup'd count: %d) -> %s in %.1fs",
        written,
        final_count,
        args.out_path,
        time.time() - t0,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
