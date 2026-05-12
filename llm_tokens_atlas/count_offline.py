"""Offline tokenization driver — per-provider, per-format counts.

Design decision: subprocess the tokenometer CLI
================================================

This script reuses `@tokenometer/core`'s offline tokenizers (cl100k_base for
Anthropic, o200k_base for OpenAI, mistral-tokenizer-js SentencePiece for the
Mistral v1/v2/v3 family, the chars-per-token heuristics for Google and
Cohere) by invoking the **tokenometer CLI** as a subprocess with
`--offline --output json`.

Why this path (over alternatives):

1. **Subprocess CLI** (this script). The CLI has stable `--output json` output
   and is already published to npm (`tokenometer@1.0.1`). One subprocess per
   batch of (model x format) cells per prompt; output parsed as JSON. The
   tokenometer CLI's full source lives next to this repo (sibling
   `tokenometer/` directory) and can be invoked directly via
   `node packages/cli/dist/index.js` without an npm install — see
   `--tokenometer-cli` flag below.
2. *Tiny Node bridge in tokenometer.* Considered but rejected: would require
   modifying tokenometer (the brief says "do NOT modify tokenometer unless
   absolutely necessary"). The CLI already exposes JSON-in / JSON-out.
3. *Re-implement tokenizers in Python.* Rejected: duplicates code, drift risk
   between the JS source-of-truth and the Python copy.

One subtlety: tokenometer's `toFormat()` does its own format conversion based
on the input. To keep `count_offline.py` and the (separate)
`count_empirical.py` measuring **byte-identical wrapped strings**, this
script wraps each prompt in Python first (via `format_wrappers.py`) and then
passes the already-wrapped string to tokenometer with `--format text` so
tokenometer treats it as opaque text. The output row carries the
**atlas-level** format name (e.g. `markdown`), not tokenometer's internal
format flag.

Output schema (`data/offline_counts.jsonl`)
============================================
Each row is one JSON object with:

```
{
  "prompt_id":         str,    # foreign key to data/raw_prompts.jsonl
  "provider":          str,    # one of anthropic|openai|google|mistral|cohere
  "format":            str,    # one of plain|markdown|json|xml|yaml
  "model":             str,    # provider-canonical model id (e.g. claude-opus-4-7)
  "offline_count":     int,    # token count from the offline tokenizer
  "tokenizer_version": str,    # "{tokenizer_kind}@{cli_version}+rates-{rates_version}"
  "ts":                str     # ISO-8601 UTC timestamp of when the row was emitted
}
```

Determinism guarantee
=====================
For a fixed input file, fixed model list, fixed format list, and a fixed
tokenometer CLI version, the (prompt_id, provider, format, model,
offline_count) tuples are byte-identical across runs. The `ts` field is
the only nondeterministic part — strip it for diff-based reproducibility
checks.

Usage
=====
    uv run python llm_tokens_atlas/count_offline.py \\
        --in  data/raw_prompts.jsonl \\
        --out data/offline_counts.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

# Allow running both as `python -m llm_tokens_atlas.count_offline` and as a
# top-level script. The package was renamed `scripts` -> `llm_tokens_atlas`
# during packaging; both forms route to the same imports below.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from llm_tokens_atlas._atlas_models import (  # noqa: E402
        PROVIDER_ORDER,
        headline_models,
    )
    from llm_tokens_atlas.format_wrappers import ALL_FORMATS, wrap  # noqa: E402
else:
    from ._atlas_models import PROVIDER_ORDER, headline_models
    from .format_wrappers import ALL_FORMATS, wrap

# One canonical model per provider — the offline driver's headline matrix.
# Sourced from `_atlas_models.ATLAS_MODELS` so it stays aligned with the
# empirical driver. Override via `--models provider=model,provider=model,...`.
DEFAULT_MODELS: Final[dict[str, str]] = headline_models()

# Re-export for backwards compatibility with any external imports of this
# constant. Kept here (rather than at the import site) so changes to
# _atlas_models only propagate through one alias.
__all__ = ["DEFAULT_MODELS", "PROVIDER_ORDER"]


@dataclass(frozen=True)
class TokenometerCli:
    """Locator + version pin for the tokenometer CLI.

    Resolution order:
      1. Explicit `--tokenometer-cli` path argument (must point to either an
         executable or `dist/index.js`).
      2. Environment variable `TOKENOMETER_CLI` (same semantics).
      3. Sibling-checkout fallback: `../tokenometer/packages/cli/dist/index.js`
         relative to this repo root.
      4. `tokenometer` on `$PATH` (e.g. from a global npm install).
    """

    invocation: tuple[str, ...]
    version: str

    @classmethod
    def resolve(cls, override: str | None) -> TokenometerCli:
        # Explicit override beats everything else.
        candidates: list[str | None] = [override, os.environ.get("TOKENOMETER_CLI")]
        # Sibling checkout fallback.
        repo_root = Path(__file__).resolve().parent.parent
        sibling = repo_root.parent / "tokenometer" / "packages" / "cli" / "dist" / "index.js"
        if sibling.exists():
            candidates.append(str(sibling))

        for cand in candidates:
            if not cand:
                continue
            inv = _candidate_invocation(cand)
            if inv is None:
                continue
            version = _probe_version(inv)
            if version is not None:
                return cls(invocation=inv, version=version)

        on_path = shutil.which("tokenometer")
        if on_path:
            inv = (on_path,)
            version = _probe_version(inv)
            if version is not None:
                return cls(invocation=inv, version=version)

        raise FileNotFoundError(
            "Could not locate the tokenometer CLI. Set --tokenometer-cli, "
            "TOKENOMETER_CLI, place a sibling tokenometer checkout, or "
            "`npm install -g tokenometer`."
        )


def _candidate_invocation(path_or_cmd: str) -> tuple[str, ...] | None:
    """Build a subprocess argv from a CLI locator string.

    A `.js` path is invoked via `node`; anything else is treated as an
    executable.
    """
    if path_or_cmd.endswith(".js"):
        p = Path(path_or_cmd)
        if not p.exists():
            return None
        node = shutil.which("node")
        if node is None:
            return None
        return (node, str(p))
    p = Path(path_or_cmd)
    if p.exists():
        return (str(p),)
    return None


def _probe_version(invocation: tuple[str, ...]) -> str | None:
    """Invoke `<cli> --version` and parse the trailing token."""
    try:
        proc = subprocess.run(
            [*invocation, "--version"],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    # Expected output: "tokenometer X.Y.Z\n"
    parts = proc.stdout.strip().split()
    if len(parts) < 2:
        return None
    return parts[-1]


@dataclass(frozen=True)
class OfflineCount:
    prompt_id: str
    provider: str
    fmt: str
    model: str
    count: int
    tokenizer_kind: str


class SpecialTokenDisallowedError(RuntimeError):
    """gpt-tokenizer rejected a ChatML / GPT-4 special-token literal.

    Raised by ``_run_tokenometer`` when the CLI's stderr contains the
    upstream ``gpt-tokenizer`` "Disallowed special token" sentinel. Real-world
    corpora (e.g. ShareGPT, GitHub READMEs, model documentation) occasionally
    contain ``<|im_start|>`` / ``<|im_end|>`` / ``<|endoftext|>`` etc. as
    plain text, which causes ``gpt-tokenizer``'s default ``encode()`` to
    refuse the input — taking down the whole batch (all 5 providers in the
    matrix, not just the gpt-tokenizer-backed ones).

    The driver catches this and re-issues the count locally via tiktoken
    with ``allowed_special='all'`` for the gpt-tokenizer-backed providers
    (anthropic via ``cl100k_base``, openai via ``o200k_base``). See
    ``_count_locally_for_special_tokens`` for the fallback path.
    """


_SPECIAL_TOKEN_SIGNATURE = "Disallowed special token"


def _run_tokenometer(
    cli: TokenometerCli,
    wrapped_text: str,
    model_ids: Iterable[str],
) -> list[dict]:
    """Invoke the tokenometer CLI on a wrapped prompt; return the parsed cells.

    Uses `--format text` so tokenometer treats the input as opaque (no
    re-wrapping). Passes `--no-config` to ignore any local .tokenometer.yml.
    The wrapped string is written to stdin.
    """
    model_arg = ",".join(model_ids)
    argv = [
        *cli.invocation,
        "-",  # read from stdin
        "--offline",
        "--no-config",
        "--output",
        "json",
        "--format",
        "text",
        "--model",
        model_arg,
    ]
    proc = subprocess.run(
        argv,
        input=wrapped_text,
        capture_output=True,
        check=False,
        text=True,
        timeout=60,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        if _SPECIAL_TOKEN_SIGNATURE in stderr:
            raise SpecialTokenDisallowedError(stderr)
        raise RuntimeError(
            f"tokenometer CLI failed (rc={proc.returncode}); "
            f"stderr={stderr!r}"
        )
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"tokenometer CLI did not return valid JSON: {e}; "
            f"stdout (first 200 chars): {proc.stdout[:200]!r}"
        ) from e
    files = payload.get("files", [])
    if not files:
        raise RuntimeError("tokenometer CLI returned no files in payload")
    # We always send exactly one prompt over stdin.
    return list(files[0]["results"])


# Models whose offline tokenizer in tokenometer is `o200k_base` (the GPT-4o
# family + successors). All other OpenAI-backed ids fall back to
# `cl100k_base`. We mirror the same table to keep counts byte-identical with
# the CLI path when the fallback fires.
_O200K_BASE_MODELS: Final[frozenset[str]] = frozenset(
    {
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4o-2024-08-06",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "o1",
        "o1-mini",
        "o3",
        "o3-mini",
        "o4-mini",
    }
)

# Mirrors tokenometer's per-provider HEURISTIC_CHARS_PER_TOKEN in
# tokenize.ts. Used only by the local fallback for special-token-bearing
# prompts; the normal path always reads counts from the CLI.
_HEURISTIC_CHARS_PER_TOKEN: Final[dict[str, float]] = {
    "google": 4.0,
    "cohere": 4.0,
    "mistral": 4.0,  # only used when the SentencePiece path is unavailable
}


def _count_locally_for_special_tokens(
    wrapped_text: str,
    provider_to_model: dict[str, str],
) -> list[dict]:
    """Local Python fallback when the CLI rejects ChatML special tokens.

    Strategy: re-implement tokenometer's offline contract for the providers
    whose tokenizers would otherwise reject the input.

    - **anthropic** + **openai**: encode locally with ``tiktoken`` and
      ``allowed_special='all'`` so ``<|im_start|>`` etc. are treated as opaque
      text. This matches the underlying ``gpt-tokenizer`` byte-exactly because
      ``tiktoken`` and ``gpt-tokenizer`` ship the same BPE tables for
      ``cl100k_base`` and ``o200k_base``.
    - **google** + **cohere**: ``chars / 4`` heuristic — identical to the
      formula in ``tokenometer/packages/core/src/tokenize.ts``.
    - **mistral**: use ``mistral-common``'s SentencePiece tokenizer when the
      model has a known alias, otherwise fall back to the same ``chars / 4``
      heuristic. ``mistral-common`` is already a hard dep of this repo
      (empirical driver), so no new dependency is incurred.

    Returns rows in the same shape ``_run_tokenometer`` would have returned:
    ``[{"model", "provider", "format", "inputTokens", "tokenizer", ...}]``.
    """
    import tiktoken  # local import keeps the hot path cold-start free

    out: list[dict] = []
    for provider, model in provider_to_model.items():
        if provider == "openai":
            encoding_name = "o200k_base" if model in _O200K_BASE_MODELS else "cl100k_base"
            enc = tiktoken.get_encoding(encoding_name)
            count = len(enc.encode(wrapped_text, allowed_special="all"))
            tokenizer_kind = encoding_name
        elif provider == "anthropic":
            # tokenometer uses cl100k_base for every Anthropic id.
            enc = tiktoken.get_encoding("cl100k_base")
            count = len(enc.encode(wrapped_text, allowed_special="all"))
            tokenizer_kind = "cl100k_base"
        elif provider in ("google", "cohere"):
            chars_per = _HEURISTIC_CHARS_PER_TOKEN[provider]
            count = math.ceil(len(wrapped_text) / chars_per)
            tokenizer_kind = "heuristic"
        elif provider == "mistral":
            count, tokenizer_kind = _mistral_local_count(wrapped_text, model)
        else:
            # Unknown provider — fall back to the same chars/4 heuristic
            # tokenometer uses for "other".
            count = math.ceil(len(wrapped_text) / 4.0)
            tokenizer_kind = "heuristic"

        out.append(
            {
                "model": model,
                "provider": provider,
                "format": "text",
                "inputTokens": int(count),
                "tokenizer": tokenizer_kind,
            }
        )
    return out


def _mistral_local_count(text: str, model: str) -> tuple[int, str]:
    """Best-effort offline Mistral count without going through the CLI.

    Uses ``mistral-common``'s tokenizer when the model has a known alias
    (the same path the empirical driver already takes). Falls back to a
    ``chars / 4`` heuristic if the model is unknown or ``mistral-common``
    cannot resolve it — matching tokenometer's behaviour for Tekken-family
    models.
    """
    try:
        from mistral_common.tokens.tokenizers.mistral import (
            MODEL_NAME_TO_TOKENIZER_CLS,
            MistralTokenizer,
        )
    except ImportError:  # pragma: no cover — atlas always installs it
        return math.ceil(len(text) / 4.0), "heuristic"

    # Map atlas model ids onto mistral-common's accepted keys. Reuses the
    # same alias table as the empirical driver so behaviour is consistent.
    alias = {
        "mistral-large-latest": "mistral-large-2411",
        "mistral-small-latest": "mistral-small-2409",
        "mistral-medium-latest": "mistral-medium-2312",
    }
    resolved = alias.get(model, model)
    if resolved not in MODEL_NAME_TO_TOKENIZER_CLS:
        return math.ceil(len(text) / 4.0), "heuristic"
    tok = MistralTokenizer.from_model(resolved, strict=True)
    ids = tok.instruct_tokenizer.tokenizer.encode(text, bos=False, eos=False)
    # mistral-tokenizer-js (which tokenometer uses) defaults to BOS=true; we
    # leave BOS off here because the wrapped text is plain content, and the
    # difference is one token — within the noise floor for the fallback
    # path. Documented in PIPELINE_NOTES.md.
    return len(ids), "mistral_v1_v3"


def count_one_prompt(
    cli: TokenometerCli,
    prompt_id: str,
    text: str,
    provider_to_model: dict[str, str],
    formats: Iterable[str],
) -> Iterator[OfflineCount]:
    """Yield one OfflineCount row per (provider, format) cell for one prompt.

    Strategy: one CLI invocation per **format** (so all models for that format
    share a single subprocess startup cost). Each invocation wraps the prompt
    once in Python, passes `--format text` to tokenometer, and parses out the
    per-model rows from the JSON output.

    Special-token tolerance
    -----------------------
    If the CLI rejects the prompt because ``gpt-tokenizer`` saw a ChatML
    literal (``<|im_start|>`` etc.), we re-issue the count locally via
    :func:`_count_locally_for_special_tokens`. The output rows are
    indistinguishable from the CLI path's rows except for one tag in the
    ``tokenizer_kind`` field, which downstream readers can use to attribute
    the fallback. See ``analysis/PIPELINE_NOTES.md`` Issue 1.
    """
    model_to_provider = {m: p for p, m in provider_to_model.items()}
    model_ids = list(provider_to_model.values())
    for fmt in formats:
        wrapped = wrap(text, fmt)
        try:
            cells = _run_tokenometer(cli, wrapped, model_ids)
        except SpecialTokenDisallowedError as e:
            sys.stderr.write(
                f"count_offline: prompt {prompt_id!r} format={fmt!r} contains "
                f"ChatML special tokens; falling back to local "
                f"`allowed_special='all'` encoding ({e}).\n"
            )
            cells = _count_locally_for_special_tokens(wrapped, provider_to_model)
        # tokenometer emits one cell per (model x format) but we constrained
        # to a single format ('text'), so there's exactly one cell per model.
        seen_models: set[str] = set()
        for cell in cells:
            model = cell["model"]
            provider = model_to_provider.get(model)
            if provider is None:
                # tokenometer auto-resolves provider via getModel; should not
                # disagree, but if it does, fall back to the cell's provider.
                provider = cell["provider"]
            if model in seen_models:
                continue  # defensive: skip duplicates if tokenometer ever emits them
            seen_models.add(model)
            yield OfflineCount(
                prompt_id=prompt_id,
                provider=provider,
                fmt=fmt,
                model=model,
                count=int(cell["inputTokens"]),
                tokenizer_kind=cell["tokenizer"],
            )


# ---- I/O helpers ----

def _read_prompts(in_path: Path) -> Iterator[tuple[str, str]]:
    """Stream (prompt_id, text) pairs from a JSONL prompt file.

    Required fields per row: `prompt_id` and `text`. Extra fields are ignored
    so this script composes with whatever the corpus-collection agent emits.
    """
    with in_path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as e:
                raise ValueError(f"{in_path}:{line_no}: invalid JSON ({e})") from e
            try:
                prompt_id = str(row["prompt_id"])
                text = str(row["text"])
            except KeyError as e:
                raise ValueError(
                    f"{in_path}:{line_no}: row missing required field {e.args[0]!r}; "
                    f"need 'prompt_id' and 'text'"
                ) from e
            yield prompt_id, text


def _format_tokenizer_version(kind: str, cli: TokenometerCli) -> str:
    """Compose the tokenizer_version string captured per row.

    Format: `{tokenizer_kind}@{cli_version}+rates-{rates_version}`. The
    rates-version pin is critical because the model->provider mapping (and
    therefore which tokenizer applies to a given model id) can change when
    upstream tokenlens ships new pricing data.
    """
    return f"{kind}@tokenometer-{cli.version}+rates-{_RATES_VERSION}"


# Pin: must match `RATES_VERSION` in tokenometer/packages/core/src/rates.ts.
# Updated when tokenometer rolls a new pricing snapshot — verify with
# `grep "RATES_VERSION =" tokenometer/packages/core/src/rates.ts` and bump
# this constant accordingly. (Hard-coded rather than parsed at runtime to
# keep the offline counter free of any dynamic loading of tokenometer
# internals beyond the CLI subprocess.)
_RATES_VERSION = "2026-05-09"


def _parse_model_overrides(spec: str | None) -> dict[str, str]:
    """Parse `--models` overrides like `anthropic=claude-haiku-4-5,openai=gpt-4o-mini`."""
    overrides = dict(DEFAULT_MODELS)
    if not spec:
        return overrides
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(
                f"--models entry {chunk!r} must be 'provider=model_id'"
            )
        provider, model = chunk.split("=", 1)
        provider = provider.strip()
        model = model.strip()
        if provider not in DEFAULT_MODELS:
            raise ValueError(
                f"--models: unknown provider {provider!r}; "
                f"known: {tuple(DEFAULT_MODELS)!r}"
            )
        overrides[provider] = model
    return overrides


def _parse_formats(spec: str | None) -> tuple[str, ...]:
    """Parse `--formats` like `markdown,json,xml`."""
    if not spec:
        return ALL_FORMATS
    fmts = tuple(f.strip() for f in spec.split(",") if f.strip())
    unknown = [f for f in fmts if f not in ALL_FORMATS]
    if unknown:
        raise ValueError(
            f"--formats: unknown format(s) {unknown!r}; known: {ALL_FORMATS!r}"
        )
    return fmts


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="count_offline.py",
        description=(
            "Drive @tokenometer/core's offline tokenizers across the "
            "5-provider x 5-format atlas for every prompt in a JSONL file."
        ),
    )
    parser.add_argument("--in", dest="in_path", required=True, type=Path)
    parser.add_argument("--out", dest="out_path", required=True, type=Path)
    parser.add_argument(
        "--models",
        default=None,
        help=(
            "Override provider->model mapping. "
            "Format: 'anthropic=claude-haiku-4-5,openai=gpt-4o-mini,...'"
        ),
    )
    parser.add_argument(
        "--formats",
        default=None,
        help=f"Comma-separated subset of {ALL_FORMATS}. Default: all.",
    )
    parser.add_argument(
        "--tokenometer-cli",
        default=None,
        help=(
            "Explicit path to the tokenometer CLI (an executable or "
            "dist/index.js). Default: resolved from $TOKENOMETER_CLI, a "
            "sibling tokenometer checkout, or $PATH."
        ),
    )
    args = parser.parse_args(argv)

    provider_to_model = _parse_model_overrides(args.models)
    formats = _parse_formats(args.formats)
    cli = TokenometerCli.resolve(args.tokenometer_cli)

    in_path: Path = args.in_path
    out_path: Path = args.out_path
    if not in_path.exists():
        raise FileNotFoundError(f"--in {in_path} does not exist")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Generated wall-clock timestamp captured once per row at write time. Same
    # timestamp for all rows from one prompt to keep rows from a single batch
    # tightly grouped, which makes downstream "rows produced in this run"
    # filtering trivial.
    now_iso = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    rows_written = 0
    with out_path.open("w", encoding="utf-8") as out_fh:
        for prompt_id, text in _read_prompts(in_path):
            for row in count_one_prompt(
                cli=cli,
                prompt_id=prompt_id,
                text=text,
                provider_to_model=provider_to_model,
                formats=formats,
            ):
                payload = {
                    "prompt_id": row.prompt_id,
                    "provider": row.provider,
                    "format": row.fmt,
                    "model": row.model,
                    "offline_count": row.count,
                    "tokenizer_version": _format_tokenizer_version(
                        row.tokenizer_kind, cli
                    ),
                    "ts": now_iso,
                }
                out_fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
                rows_written += 1

    sys.stderr.write(
        f"count_offline: wrote {rows_written} rows to {out_path} "
        f"(tokenometer-{cli.version}, rates-{_RATES_VERSION})\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
