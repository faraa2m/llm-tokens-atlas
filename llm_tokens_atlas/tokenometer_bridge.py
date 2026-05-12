"""Python bridge to the tokenometer Node CLI.

Why this module exists
======================

`tokenometer` (https://github.com/faraa2m/tokenometer) is the canonical
multi-provider token-count + cost library used across this project family. It
is written in TypeScript. Atlas's Python pipeline needs to call its token-
counting logic without duplicating it. This module is the *only* place in the
atlas codebase that should shell out to tokenometer; every other script
imports from here.

Design choice
=============
We invoke the published tokenometer CLI via `subprocess.run`, ask for JSON
output (`--output json`), and parse the result back into Python. The CLI
already supports passing multiple prompt files and multiple comma-separated
models / formats in a single invocation, so a 5000-prompt × 5-provider ×
5-format run amortizes the ~250 ms Node startup cost across hundreds of
results — not per cell.

See `llm_tokens_atlas/_tokenometer_bridge_design.md` for the full trade-off
discussion (subprocess CLI vs HTTP bridge) and the rationale for the chosen
path.

Public surface
==============

::

    count_offline(text, provider, format, model=None) -> int
    count_empirical(text, provider, format, model=None) -> int
    list_providers() -> list[str]
    list_models(provider=None) -> list[str]
    list_formats() -> list[str]

Batch helpers (preferred for production pipelines):

::

    count_offline_batch(items: list[BatchItem]) -> list[BatchResult]
    count_empirical_batch(items: list[BatchItem]) -> list[BatchResult]

Errors are surfaced as a small, named hierarchy so callers can map them to
useful user messages:

- ``TokenometerError`` — base.
- ``TokenometerNotInstalledError`` — CLI not found; install hint included.
- ``TokenometerCallError`` — subprocess failed or output was malformed.
- ``MissingApiKeyError`` — empirical mode invoked without the right env var.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Final

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Providers tokenometer can produce offline counts for. Anthropic and Google
#: use *approximations* in offline mode; OpenAI is exact (tiktoken).
PROVIDERS: Final[tuple[str, ...]] = (
    "anthropic",
    "cohere",
    "google",
    "mistral",
    "openai",
)

#: Provider → default model id used when the caller passes ``model=None``.
#: These are stable, generally-available ids in tokenometer's catalog at
#: CLI v1.0.1. Override via the ``model`` argument or via
#: ``list_models(provider)``.
DEFAULT_MODEL_BY_PROVIDER: Final[dict[str, str]] = {
    "anthropic": "claude-opus-4-7",
    "cohere": "command-r-plus",
    "google": "gemini-2.5-pro",
    "mistral": "mistral-large-2411",
    "openai": "gpt-4o",
}

#: Formats tokenometer understands natively (``convert.ts`` pipeline). We add
#: ``plain`` as an atlas-side alias for tokenometer's ``text`` because that
#: spelling appears throughout the project plan and analysis docs.
TOKENOMETER_FORMATS: Final[tuple[str, ...]] = (
    "json",
    "markdown",
    "text",
    "xml",
    "yaml",
)

#: Atlas-side superset of formats. ``plain`` maps to tokenometer's ``text``.
ATLAS_FORMATS: Final[tuple[str, ...]] = (
    "plain",
    "markdown",
    "json",
    "xml",
    "yaml",
)

#: Required env var per provider in empirical mode. Providers absent from this
#: map either don't need a key (e.g. ``openai`` uses local tiktoken) or have
#: no empirical endpoint (e.g. ``mistral`` — see tokenometer/empirical.ts).
EMPIRICAL_ENV_VAR: Final[dict[str, str]] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",  # GEMINI_API_KEY also accepted by tokenometer
    "cohere": "COHERE_API_KEY",
}

#: File the install script writes when it pins a local sibling-repo path.
_CLI_PATH_FILE: Final[Path] = Path(__file__).resolve().parent.parent / ".tokenometer-cli-path"

#: Hint for users when the CLI cannot be found.
_INSTALL_HINT: Final[str] = (
    "tokenometer CLI not found. Run `bash llm_tokens_atlas/install_tokenometer.sh` "
    "or `npm install -g tokenometer`."
)


# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------


class TokenometerError(Exception):
    """Base for all bridge-side errors."""


class TokenometerNotInstalledError(TokenometerError):
    """Raised when no usable tokenometer CLI can be located."""


class TokenometerCallError(TokenometerError):
    """Raised when the tokenometer subprocess fails or returns malformed JSON."""


class MissingApiKeyError(TokenometerError):
    """Raised when empirical mode requires a key that isn't set in the env."""


# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BatchItem:
    """One prompt-shaped count request.

    ``provider`` and ``model`` may be left unset on the item if every item in
    the batch uses the same model / provider (the call-level fallbacks will
    fill them in). The split exists so a 5000-prompt × 5-provider × 5-format
    batch can be expressed as 25 separate ``count_offline_batch`` calls
    (one per (provider, format)) rather than 125 000 individual items.
    """

    text: str
    provider: str
    format: str
    model: str | None = None


@dataclass(frozen=True)
class BatchResult:
    """One row of the result matrix returned by a batch call.

    Mirrors the relevant subset of tokenometer's per-cell JSON shape:

    - ``tokens`` — integer input token count.
    - ``approximate`` — True when the offline tokenizer is a proxy (e.g.
      cl100k_base for Anthropic) and False when it is exact (e.g.
      tiktoken o200k_base for OpenAI).
    - ``tokenizer`` — short identifier of the tokenizer that produced the
      number (``cl100k_base``, ``o200k_base``, ``mistral_v1_v3``,
      ``heuristic``).
    """

    tokens: int
    provider: str
    model: str
    format: str
    approximate: bool
    tokenizer: str


# ---------------------------------------------------------------------------
# CLI location
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _resolve_cli() -> list[str]:
    """Return the argv prefix that invokes tokenometer on this machine.

    Resolution order (first match wins):

    1. ``TOKENOMETER_CLI`` env var (CI / explicit override).
    2. ``.tokenometer-cli-path`` file at the repo root (written by
       ``llm_tokens_atlas/install_tokenometer.sh`` when it points the bridge at a
       sibling-repo build).
    3. ``tokenometer`` binary on ``PATH`` (from a global ``npm install -g``).
    4. ``./node_modules/.bin/tokenometer`` (from a local ``npm install``).

    The result is a list because option 2 may yield ``["node",
    "/abs/path/to/index.js"]``; options 3/4 yield a single-element list.

    Failure raises :class:`TokenometerNotInstalledError` with a hint.
    """
    # (1) Explicit env override.
    override = os.environ.get("TOKENOMETER_CLI")
    if override:
        return _split_cli(override)

    # (2) File pinned by the install script.
    if _CLI_PATH_FILE.exists():
        pinned = _CLI_PATH_FILE.read_text(encoding="utf-8").strip()
        if pinned:
            return _split_cli(pinned)

    # (3) `tokenometer` on PATH.
    on_path = shutil.which("tokenometer")
    if on_path:
        return [on_path]

    # (4) Local node_modules install.
    repo_root = _CLI_PATH_FILE.parent
    local_bin = repo_root / "node_modules" / ".bin" / "tokenometer"
    if local_bin.exists():
        return [str(local_bin)]

    raise TokenometerNotInstalledError(_INSTALL_HINT)


def _split_cli(value: str) -> list[str]:
    """Split a stored CLI pointer into an argv prefix.

    Accepts either an executable path (``/usr/local/bin/tokenometer``) or a
    `node <path-to-index.js>` style command. The split is intentionally
    shell-free — we don't want to evaluate metacharacters.
    """
    value = value.strip()
    if " " in value:
        head, _, tail = value.partition(" ")
        return [head, *tail.split()]
    return [value]


def _clear_cli_cache() -> None:
    """Test seam — drop the memoized CLI lookup so unit tests can swap env."""
    _resolve_cli.cache_clear()


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def _normalize_format(fmt: str) -> str:
    """Map atlas-side format names to tokenometer-side ones.

    Atlas writes "plain" everywhere; tokenometer's internal name is "text".
    Both refer to the same code path (no-op wrapper). Tokenometer's other
    format names (``json``, ``yaml``, ``xml``, ``markdown``) are accepted
    verbatim.

    Raises ``ValueError`` for anything else.
    """
    if fmt == "plain":
        return "text"
    if fmt in TOKENOMETER_FORMATS:
        return fmt
    known = sorted(set(ATLAS_FORMATS) | set(TOKENOMETER_FORMATS))
    raise ValueError(f"Unknown format {fmt!r}; expected one of {known!r}")


def _require_provider(provider: str) -> None:
    if provider not in PROVIDERS:
        raise ValueError(
            f"Unknown provider {provider!r}; expected one of {list(PROVIDERS)!r}"
        )


def _resolve_model(provider: str, model: str | None) -> str:
    """Pick a model id given an optional override.

    If ``model`` is None, use the per-provider default. We do *not* try to
    validate against tokenometer's live catalog here — the CLI will reject
    an unknown id at call time with a clear error — because catalog
    membership changes between tokenometer releases and we don't want to
    cache a stale list.
    """
    if model is not None:
        return model
    return DEFAULT_MODEL_BY_PROVIDER[provider]


def _require_empirical_key(provider: str) -> None:
    """Pre-flight check so empirical mode fails fast with a clear message.

    Some providers (``openai``) don't need a key (tokenometer uses tiktoken
    locally); ``mistral`` has no empirical endpoint at all and is rejected
    here rather than later by the subprocess.
    """
    if provider == "mistral":
        raise MissingApiKeyError(
            "mistral has no public token-count API; offline mode only. "
            "For exact counts, send a Mistral chat completion and read "
            "usage.prompt_tokens."
        )
    if provider == "openai":
        return  # tiktoken-local; no key required
    env_var = EMPIRICAL_ENV_VAR.get(provider)
    if env_var is None:
        # Defensive — every PROVIDERS entry should have an entry above.
        return
    if provider == "google":
        if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
            raise MissingApiKeyError(
                "google empirical mode requires GOOGLE_API_KEY (or GEMINI_API_KEY)"
            )
        return
    if not os.environ.get(env_var):
        raise MissingApiKeyError(
            f"{provider} empirical mode requires {env_var}"
        )


# ---------------------------------------------------------------------------
# Subprocess driver
# ---------------------------------------------------------------------------


def _run_tokenometer(
    prompt_files: list[Path],
    models: list[str],
    formats: list[str],
    *,
    empirical: bool,
) -> dict:
    """Invoke tokenometer once and return the parsed JSON payload.

    Composes the argv that the CLI expects. We always pass ``--no-config``
    because we never want a ``.tokenometer.yml`` from the caller's tree to
    silently change counts, and ``--offline`` when offline (which overrides
    ``--empirical`` if both are accidentally set).
    """
    cli = _resolve_cli()
    args: list[str] = [
        *cli,
        *(str(p) for p in prompt_files),
        "--model",
        ",".join(models),
        "--format",
        ",".join(formats),
        "--output",
        "json",
        "--no-config",
    ]
    if empirical:
        args.append("--empirical")
    else:
        args.append("--offline")

    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as e:
        # Race: CLI vanished between resolution and exec.
        _clear_cli_cache()
        raise TokenometerNotInstalledError(_INSTALL_HINT) from e

    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        # Empirical-mode missing-key errors are surfaced more usefully.
        if empirical and "requires" in stderr and "API_KEY" in stderr:
            raise MissingApiKeyError(stderr)
        raise TokenometerCallError(
            f"tokenometer exited with status {proc.returncode}: {stderr or proc.stdout}"
        )

    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        truncated = proc.stdout[:500] + ("…" if len(proc.stdout) > 500 else "")
        raise TokenometerCallError(
            f"tokenometer output was not valid JSON: {e.msg}; got: {truncated}"
        ) from e


# ---------------------------------------------------------------------------
# Public API — single-call
# ---------------------------------------------------------------------------


def count_offline(
    text: str,
    provider: str,
    format: str,  # noqa: A002 — public-API spelling per the brief.
    model: str | None = None,
) -> int:
    """Return the offline token count for ``text`` under (provider, model, format).

    "Offline" means tokenometer uses its locally-bundled tokenizer
    (``tiktoken o200k_base`` for OpenAI, ``cl100k_base`` proxy for Anthropic,
    a chars-per-token heuristic for Google, SentencePiece v1/v3 for older
    Mistral, etc.). No network call is made and no API key is required.

    Args:
        text: The prompt body. May be already-wrapped by the caller (in which
            case use ``format="text"`` so tokenometer doesn't re-wrap), or
            the raw prompt (in which case ``format`` chooses tokenometer's
            wrapper).
        provider: One of :data:`PROVIDERS`.
        format: One of :data:`ATLAS_FORMATS` (``plain`` is accepted as an
            alias of ``text``).
        model: Optional model id. Defaults to
            :data:`DEFAULT_MODEL_BY_PROVIDER[provider]` if ``None``.

    Returns:
        Integer token count. Non-negative.

    Raises:
        ValueError: Unknown provider, unknown format.
        TokenometerNotInstalledError: CLI not on the system.
        TokenometerCallError: Subprocess exited non-zero or produced
            unparseable output.
    """
    _require_provider(provider)
    fmt = _normalize_format(format)
    model_id = _resolve_model(provider, model)

    with _write_prompt_to_tempfile(text) as prompt_path:
        payload = _run_tokenometer(
            prompt_files=[prompt_path],
            models=[model_id],
            formats=[fmt],
            empirical=False,
        )
    return _extract_single_count(payload, expected_provider=provider, expected_model=model_id)


def count_empirical(
    text: str,
    provider: str,
    format: str,  # noqa: A002 — public-API spelling per the brief.
    model: str | None = None,
) -> int:
    """Return the *empirical* (real-API) token count for the same axes.

    Calls the provider's own token-count endpoint via tokenometer:

    - ``anthropic`` → ``client.messages.countTokens`` (free).
    - ``google`` → ``model.countTokens`` (free).
    - ``cohere`` → ``POST /v1/tokenize`` (free).
    - ``openai`` → local tiktoken ``o200k_base`` (tokenometer treats this as
      "exact" since OpenAI does not publish a public token-count endpoint
      but tiktoken is the source of truth for o200k models).
    - ``mistral`` → not supported (no public endpoint). Raises
      :class:`MissingApiKeyError` with the same message tokenometer uses.

    Args:
        text: Prompt body. Same wrapping caveat as :func:`count_offline`.
        provider, format, model: Same as :func:`count_offline`.

    Returns:
        Integer empirical token count.

    Raises:
        MissingApiKeyError: Required ``*_API_KEY`` env var is unset.
        ValueError: Unknown provider, unknown format.
        TokenometerNotInstalledError, TokenometerCallError: Same as
            :func:`count_offline`.
    """
    _require_provider(provider)
    _require_empirical_key(provider)
    fmt = _normalize_format(format)
    model_id = _resolve_model(provider, model)

    with _write_prompt_to_tempfile(text) as prompt_path:
        payload = _run_tokenometer(
            prompt_files=[prompt_path],
            models=[model_id],
            formats=[fmt],
            empirical=True,
        )
    return _extract_single_count(payload, expected_provider=provider, expected_model=model_id)


def list_providers() -> list[str]:
    """Return the list of providers the bridge knows about.

    Mirrors tokenometer's ``Provider`` enum verbatim — see
    ``packages/core/src/types.ts``.
    """
    return list(PROVIDERS)


def list_models(provider: str | None = None) -> list[str]:
    """Return tokenometer's known model ids, optionally filtered by provider.

    Implementation: invokes the CLI with ``--help`` and parses the
    catalog out of the help banner (the canonical place tokenometer
    advertises its catalog from). This is slower than a static table but
    means we never go stale relative to tokenometer's actual catalog.

    Args:
        provider: If given, return only models tokenometer maps to that
            provider. Otherwise return the full catalog.

    Returns:
        Sorted list of model id strings. Never empty in a healthy install.

    Raises:
        ValueError: Unknown provider.
        TokenometerNotInstalledError: CLI not present.
    """
    if provider is not None:
        _require_provider(provider)
    catalog = _fetch_catalog_via_help()
    if provider is None:
        return sorted(catalog)
    # Tokenometer's help text doesn't tag models by provider. The catalog
    # entries follow well-known provider-tagging conventions — we filter via
    # the rate-table descriptor by invoking tokenometer with one prompt per
    # candidate id is too expensive. Instead, we hard-code prefix patterns
    # that mirror tokenometer's own provider-tagging logic.
    return sorted(m for m in catalog if _infer_provider(m) == provider)


def list_formats() -> list[str]:
    """Return the format names tokenometer understands.

    Atlas-side spelling (``plain`` instead of ``text``). The actual
    tokenometer call accepts both.
    """
    return list(ATLAS_FORMATS)


# ---------------------------------------------------------------------------
# Public API — batched
# ---------------------------------------------------------------------------


def count_offline_batch(items: list[BatchItem]) -> list[BatchResult]:
    """Offline-count many prompts in as few subprocess calls as possible.

    Items are bucketed by (provider, format, model) and dispatched as one
    tokenometer call per bucket with all prompts in that bucket as multiple
    file inputs. This is the right surface for atlas's count_offline.py.
    """
    return _count_batch(items, empirical=False)


def count_empirical_batch(items: list[BatchItem]) -> list[BatchResult]:
    """Empirical-count many prompts in as few subprocess calls as possible.

    Same shape as :func:`count_offline_batch`. Inherits the per-provider
    API-key requirements of :func:`count_empirical`.
    """
    return _count_batch(items, empirical=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _count_batch(items: list[BatchItem], *, empirical: bool) -> list[BatchResult]:
    """Bucket items by (provider, format, model) and dispatch one CLI call per bucket.

    The output preserves the input order — each input ``BatchItem`` has
    exactly one corresponding ``BatchResult`` at the same list index.
    """
    if not items:
        return []

    # Validate up front so we don't write tempfiles for a batch that will
    # be rejected.
    for item in items:
        _require_provider(item.provider)
        _normalize_format(item.format)  # raises on bad value
        if empirical:
            _require_empirical_key(item.provider)

    # Bucket index → output slot.
    by_bucket: dict[tuple[str, str, str], list[int]] = {}
    resolved_models: list[str] = []
    resolved_formats: list[str] = []
    for idx, item in enumerate(items):
        model_id = _resolve_model(item.provider, item.model)
        fmt = _normalize_format(item.format)
        resolved_models.append(model_id)
        resolved_formats.append(fmt)
        by_bucket.setdefault((item.provider, model_id, fmt), []).append(idx)

    out: list[BatchResult | None] = [None] * len(items)

    with tempfile.TemporaryDirectory(prefix="tokenometer-bridge-") as tmp:
        tmp_dir = Path(tmp)
        for (provider, model_id, fmt), idxs in by_bucket.items():
            files: list[Path] = []
            for idx in idxs:
                # Use a stable filename so the JSON's `path` field is
                # predictable, even though we map back via the file-order of
                # the input array (tokenometer preserves file order).
                p = tmp_dir / f"prompt-{idx}.txt"
                p.write_text(items[idx].text, encoding="utf-8")
                files.append(p)

            payload = _run_tokenometer(
                prompt_files=files,
                models=[model_id],
                formats=[fmt],
                empirical=empirical,
            )
            files_out = payload.get("files", [])
            if len(files_out) != len(idxs):
                raise TokenometerCallError(
                    f"tokenometer returned {len(files_out)} file blocks "
                    f"for {len(idxs)} inputs in bucket {provider}/{model_id}/{fmt}"
                )
            for slot, file_block in zip(idxs, files_out, strict=True):
                cells = file_block.get("results", [])
                if not cells:
                    raise TokenometerCallError(
                        f"tokenometer returned no cells for prompt {slot} "
                        f"in bucket {provider}/{model_id}/{fmt}"
                    )
                cell = cells[0]
                out[slot] = BatchResult(
                    tokens=int(cell["inputTokens"]),
                    provider=cell["provider"],
                    model=cell["model"],
                    format=cell["format"],
                    approximate=bool(cell["approximate"]),
                    tokenizer=str(cell["tokenizer"]),
                )

    # All slots must have been filled by the bucket loop above.
    if any(r is None for r in out):
        raise TokenometerCallError("batch had ungroup-able items")
    return [r for r in out if r is not None]


def _extract_single_count(
    payload: dict, *, expected_provider: str, expected_model: str
) -> int:
    """Pull the single tokens value out of a single-file, single-cell payload.

    Tokenometer's JSON shape is::

        {
          "files": [
            { "path": "...",
              "results": [
                {"approximate": ..., "format": ..., "inputCost": ...,
                 "inputTokens": N, "model": ..., "provider": ...,
                 "tokenizer": ...}, ...
              ]
            }
          ]
        }

    For a single-call invocation we expect exactly one file and exactly one
    result cell. Anything else is a contract violation we surface as
    :class:`TokenometerCallError`.
    """
    files = payload.get("files", [])
    if len(files) != 1:
        raise TokenometerCallError(
            f"expected 1 file in tokenometer output, got {len(files)}"
        )
    cells = files[0].get("results", [])
    if len(cells) != 1:
        raise TokenometerCallError(
            f"expected 1 result cell, got {len(cells)}"
        )
    cell = cells[0]
    if cell.get("provider") != expected_provider:
        raise TokenometerCallError(
            f"provider mismatch: requested {expected_provider!r}, "
            f"got {cell.get('provider')!r}"
        )
    if cell.get("model") != expected_model:
        # Not fatal — tokenometer may normalize aliases (e.g. dropping a
        # `:vendor` prefix). Log to stderr but don't raise.
        sys.stderr.write(
            f"tokenometer_bridge: model id normalized "
            f"{expected_model!r} -> {cell.get('model')!r}\n"
        )
    return int(cell["inputTokens"])


def _write_prompt_to_tempfile(text: str):
    """Context manager — write ``text`` to a NamedTemporaryFile and yield the path.

    Tokenometer reads prompts from files, not from stdin (the stdin path is
    used only when ``-`` is the positional). We write to a file so multi-call
    code paths share a uniform shape, and so the tempfile cleanup is bullet-
    proof on early raises.
    """
    return _TempPromptFile(text)


class _TempPromptFile:
    """Internal context manager. Writes ``text`` to a tempfile that vanishes on close."""

    def __init__(self, text: str) -> None:
        self._text = text
        self._handle: tempfile._TemporaryFileWrapper | None = None
        self._path: Path | None = None

    def __enter__(self) -> Path:
        # NamedTemporaryFile with delete=False so we control closing
        # explicitly (the subprocess needs the file alive while it reads).
        fh = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".txt",
            prefix="tokenometer-bridge-",
            delete=False,
        )
        fh.write(self._text)
        fh.flush()
        fh.close()
        self._path = Path(fh.name)
        return self._path

    def __exit__(self, *_exc_info) -> None:
        if self._path is not None and self._path.exists():
            # Worst-case: leaks a small tempfile. Don't mask the caller's
            # exception over a cleanup race.
            with contextlib.suppress(OSError):
                self._path.unlink()


@lru_cache(maxsize=1)
def _fetch_catalog_via_help() -> tuple[str, ...]:
    """Parse the model catalog out of `tokenometer --help`.

    The help banner contains a line `Known: id1, id2, id3, ...` listing
    every model in tokenometer's rate table. We grep for that prefix and
    split by comma. Cached for the lifetime of the process.
    """
    cli = _resolve_cli()
    try:
        proc = subprocess.run(
            [*cli, "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as e:
        _clear_cli_cache()
        raise TokenometerNotInstalledError(_INSTALL_HINT) from e

    if proc.returncode != 0:
        raise TokenometerCallError(
            f"tokenometer --help exited {proc.returncode}: "
            f"{(proc.stderr or '').strip()}"
        )

    catalog: list[str] = []
    for line in proc.stdout.splitlines():
        stripped = line.strip()
        if not stripped.startswith("Known:"):
            continue
        rest = stripped[len("Known:") :].strip()
        # `Known:` appears once for models and once for formats; the format
        # list contains short tokens like `json, markdown, text, xml, yaml`
        # and any one of those is a substring of common model ids — we
        # disambiguate by length. Model ids are >5 chars in our catalog.
        candidates = [c.strip() for c in rest.split(",") if c.strip()]
        if all(len(c) <= 8 for c in candidates):
            # Heuristic: assume this is the format list, not the model list.
            continue
        catalog.extend(candidates)
        # First model-like Known line wins; second occurrence (if any) is
        # the format list and would already be filtered above.
        break
    if not catalog:
        raise TokenometerCallError("could not parse model catalog from tokenometer --help")
    return tuple(catalog)


def _infer_provider(model_id: str) -> str:
    """Map a tokenometer model id to a provider.

    Mirrors the conventions in tokenometer/packages/core/src/rates.ts —
    Anthropic ids start with ``claude-``, OpenAI ids start with ``gpt`` or
    ``o`` (e.g. ``o1-mini``), Google ids start with ``gemini-``, Mistral
    ids start with ``mistral-`` / ``codestral-`` / ``pixtral-`` / ``ministral`` /
    ``magistral`` / ``devstral`` / ``open-mistral`` / ``open-mixtral``,
    Cohere ids start with ``command``. Anything else falls through to
    ``"openai"`` — tokenometer's catalog is OpenAI-heavy.
    """
    m = model_id.lower()
    if m.startswith("claude"):
        return "anthropic"
    if m.startswith("gemini"):
        return "google"
    if m.startswith("command"):
        return "cohere"
    if (
        m.startswith("mistral")
        or m.startswith("codestral")
        or m.startswith("pixtral")
        or m.startswith("ministral")
        or m.startswith("magistral")
        or m.startswith("devstral")
        or m.startswith("open-mistral")
        or m.startswith("open-mixtral")
    ):
        return "mistral"
    return "openai"
