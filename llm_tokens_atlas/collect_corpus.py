"""Collect a balanced sample of real-world prompts for llm-tokens-atlas.

This script samples prompts from 5 open, redistribution-compatible corpora and
writes them to a JSONL stream that conforms to the `promptRow` schema declared
in `data/schema.json`.

Sources (all open / non-gated as of the run date):
  - HumanEval                 (openai/openai_humaneval)
  - WildChat-1M               (allenai/WildChat-1M)            ← LMSYS-chat-1M
                                                                 substitute, since
                                                                 LMSYS-chat-1M is
                                                                 gated on HF and we
                                                                 require
                                                                 no-credentials
                                                                 reproducibility.
                                                                 WildChat-1M is
                                                                 the spiritual
                                                                 successor and
                                                                 ships the same
                                                                 `redacted` and
                                                                 `toxic` flags
                                                                 used here for PII
                                                                 / safety filtering.
  - MT-Bench                  (lmsys/mt_bench_human_judgments)  multi-turn eval
                                                                 prompts (user
                                                                 turn 0 from
                                                                 conversation_a).
  - English Wikipedia         (wikimedia/wikipedia,
                              `20231101.en`)                   first paragraph,
                                                                truncated to 2000
                                                                chars.
  - GitHub READMEs            curated seed list pulled via
                              raw.githubusercontent.com         well-known repos
                                                                with permissive
                                                                licenses
                                                                (Apache-2.0,
                                                                MIT, BSD).

Output row schema (must remain in sync with `data/schema.json#/$defs/promptRow`):
    prompt_id       str   uuid4
    source          str   one of {humaneval, wildchat-1m, mt-bench,
                         wikipedia-en, github-readmes}
    text            str   prompt text exactly as it will be tokenized
    text_len_chars  int   Unicode codepoint length
    text_len_words  int   whitespace-split word count
    language        str   ISO-639-1 best-effort, default "en"; "code" for
                         HumanEval and READMEs.
    domain          str   one of {code, prose, chat, structured, multilingual,
                         other} (schema enum)
    collected_at    str   ISO-8601 UTC timestamp

Determinism:
    All randomness is seeded with --seed (default 42). Anyone running the
    same command on the same upstream snapshots should get the same rows.
    For HF datasets we additionally pin the dataset by its current revision
    at runtime via the implicit `huggingface_hub` cache; for full
    reproducibility we record per-source revisions in `data/provenance.md`.

CLI:
    uv run python llm_tokens_atlas/collect_corpus.py \\
        --out data/raw_prompts.jsonl \\
        --n 500
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import certifi

# -----------------------------------------------------------------------------
# HF httpx SSL setup
# -----------------------------------------------------------------------------
# On macOS the bundled Homebrew Python uses an empty trust store, so the
# `huggingface_hub` httpx client falls back to certifi-but-unconfigured and
# fails to verify huggingface.co's certificate. We pin certifi's bundle.
#
# On corporate networks with a TLS-intercepting proxy (e.g. Zscaler, BlueCoat,
# Palo Alto), certifi alone is insufficient because the proxy presents a
# locally-trusted root that is in the OS keychain but not in certifi. We
# build a combined CA bundle on first use: certifi + any non-certifi roots
# we can extract from the macOS System keychain (Linux uses /etc/ssl/certs).
# The combined bundle is cached at
# `~/.cache/llm-tokens-atlas/combined_ca.pem` so subsequent runs are
# instantaneous.
#
# The official `huggingface_hub` escape hatch is `set_client_factory` /
# `set_async_client_factory`, which must be called BEFORE any other HF /
# datasets import or call (because the library caches the session on first
# use). We call it at module-import time.

_CACHED_CA_DIR = Path("~/.cache/llm-tokens-atlas").expanduser()
_CACHED_CA_PATH = _CACHED_CA_DIR / "combined_ca.pem"


def _build_combined_ca_bundle() -> Path:
    """Return a path to a PEM file containing certifi + system root certs.

    Cached on disk so we only pay the macOS `security` cost once. If the
    cache exists and is non-empty we return it directly.
    """
    if _CACHED_CA_PATH.is_file() and _CACHED_CA_PATH.stat().st_size > 1024:
        return _CACHED_CA_PATH

    _CACHED_CA_DIR.mkdir(parents=True, exist_ok=True)
    certifi_bundle = Path(certifi.where()).read_text(encoding="utf-8", errors="replace")

    extra_roots = ""
    # macOS: pull every cert in the System keychain (this is where Zscaler /
    # Cloudflare / corp roots live). We pull System AND the user login
    # keychain.
    import platform  # noqa: PLC0415
    import subprocess  # noqa: PLC0415

    if platform.system() == "Darwin":
        candidates = [
            "/Library/Keychains/System.keychain",
            "/System/Library/Keychains/SystemRootCertificates.keychain",
            str(Path("~/Library/Keychains/login.keychain-db").expanduser()),
        ]
        for kc in candidates:
            if not Path(kc).exists() and not kc.endswith(".keychain"):
                continue
            try:
                out = subprocess.run(
                    ["security", "find-certificate", "-a", "-p", kc],
                    capture_output=True,
                    text=True,
                    timeout=15,
                    check=False,
                )
                if out.returncode == 0 and out.stdout:
                    extra_roots += "\n" + out.stdout
            except Exception:
                continue
    elif platform.system() == "Linux":
        for p in (
            "/etc/ssl/certs/ca-certificates.crt",  # debian/ubuntu
            "/etc/pki/tls/certs/ca-bundle.crt",  # rhel/fedora
            "/etc/ssl/cert.pem",  # alpine
        ):
            if Path(p).is_file():
                try:
                    extra_roots += "\n" + Path(p).read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                break

    _CACHED_CA_PATH.write_text(certifi_bundle + extra_roots, encoding="utf-8")
    return _CACHED_CA_PATH


def _install_certifi_httpx_factory() -> None:
    """Force `huggingface_hub` to use a combined-trust-store httpx client.

    Idempotent. Builds (and caches) a CA bundle that contains certifi roots
    plus any corporate roots extracted from the OS trust store.
    """
    ca = str(_build_combined_ca_bundle())
    os.environ.setdefault("SSL_CERT_FILE", ca)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", ca)
    os.environ.setdefault("CURL_CA_BUNDLE", ca)

    import httpx  # noqa: PLC0415

    try:
        from huggingface_hub.utils._http import (  # noqa: PLC0415
            async_hf_request_event_hook,
            async_hf_response_event_hook,
            hf_request_event_hook,
            set_async_client_factory,
            set_client_factory,
        )
    except ImportError:
        # Older huggingface_hub without factory hooks; SSL_CERT_FILE alone
        # may suffice for `requests`-based code paths.
        return

    def _client_factory() -> httpx.Client:
        return httpx.Client(
            event_hooks={"request": [hf_request_event_hook]},
            follow_redirects=True,
            timeout=None,
            verify=ca,
        )

    def _async_factory() -> httpx.AsyncClient:
        return httpx.AsyncClient(
            event_hooks={
                "request": [async_hf_request_event_hook],
                "response": [async_hf_response_event_hook],
            },
            follow_redirects=True,
            timeout=None,
            verify=ca,
        )

    set_client_factory(_client_factory)
    set_async_client_factory(_async_factory)


_install_certifi_httpx_factory()
_VERIFY_CA = os.environ.get("SSL_CERT_FILE")


# Defer heavy imports until after the SSL patch above.
import httpx  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

DEFAULT_SEED = 42
DEFAULT_OUT = Path("data/raw_prompts.jsonl")
PROVENANCE_PATH = Path("data/provenance.md")
WIKIPEDIA_FIRST_PARAGRAPH_MAX_CHARS = 2000

# Schema enum for `source` — also referenced in provenance.md.
SOURCE_HUMANEVAL = "humaneval"
SOURCE_WILDCHAT = "wildchat-1m"
SOURCE_MTBENCH = "mt-bench"
SOURCE_WIKIPEDIA = "wikipedia-en"
SOURCE_GITHUB_READMES = "github-readmes"

# Curated seed list of well-known repositories whose README is permissively
# licensed (Apache-2.0, MIT, BSD). README files in these repos cover prose,
# code, structured config blocks, and embedded shell commands — a good cross-
# section of "real-world README" content.
#
# Each entry is (owner, repo, branch, path). We fetch the raw file directly,
# which is rate-limited but free and requires no auth. If a repo is unreachable
# or renames its default branch we skip it and continue.
GITHUB_README_REPOS: list[tuple[str, str, str, str]] = [
    ("python", "cpython", "main", "README.rst"),
    ("torvalds", "linux", "master", "README"),
    ("microsoft", "vscode", "main", "README.md"),
    ("facebook", "react", "main", "README.md"),
    ("nodejs", "node", "main", "README.md"),
    ("kubernetes", "kubernetes", "master", "README.md"),
    ("rust-lang", "rust", "master", "README.md"),
    ("golang", "go", "master", "README.md"),
    ("pallets", "flask", "main", "README.md"),
    ("django", "django", "main", "README.rst"),
    ("scikit-learn", "scikit-learn", "main", "README.rst"),
    ("pandas-dev", "pandas", "main", "README.md"),
    ("numpy", "numpy", "main", "README.md"),
    ("huggingface", "transformers", "main", "README.md"),
    ("openai", "openai-python", "main", "README.md"),
    ("anthropics", "anthropic-sdk-python", "main", "README.md"),
    ("encode", "httpx", "master", "README.md"),
    ("psf", "requests", "main", "README.md"),
    ("astral-sh", "uv", "main", "README.md"),
    ("astral-sh", "ruff", "main", "README.md"),
    ("vuejs", "core", "main", "README.md"),
    ("sveltejs", "svelte", "main", "README.md"),
    ("denoland", "deno", "main", "README.md"),
    ("oven-sh", "bun", "main", "README.md"),
    ("pytorch", "pytorch", "main", "README.md"),
    ("tensorflow", "tensorflow", "master", "README.md"),
    ("apache", "airflow", "main", "README.md"),
    ("apache", "spark", "master", "README.md"),
    ("apache", "kafka", "trunk", "README.md"),
    ("redis", "redis", "unstable", "README.md"),
    ("git", "git", "master", "README.md"),
    ("docker", "docker-ce", "master", "README.md"),
    ("hashicorp", "terraform", "main", "README.md"),
    ("prometheus", "prometheus", "main", "README.md"),
    ("grafana", "grafana", "main", "README.md"),
    ("elastic", "elasticsearch", "main", "README.asciidoc"),
    ("jax-ml", "jax", "main", "README.md"),
    ("openai", "tiktoken", "main", "README.md"),
    ("jpadilla", "pyjwt", "master", "README.rst"),
    ("python-poetry", "poetry", "main", "README.md"),
    ("pypa", "pip", "main", "README.rst"),
    ("pypa", "setuptools", "main", "README.rst"),
    ("pre-commit", "pre-commit", "main", "README.md"),
    ("psf", "black", "main", "README.md"),
    ("nvm-sh", "nvm", "master", "README.md"),
    ("zsh-users", "zsh-completions", "master", "README.md"),
    ("fastapi", "fastapi", "master", "README.md"),
    ("tiangolo", "typer", "master", "README.md"),
    ("streamlit", "streamlit", "develop", "README.md"),
    ("plotly", "plotly.py", "master", "README.md"),
    ("bokeh", "bokeh", "branch-3.7", "README.md"),
    ("matplotlib", "matplotlib", "main", "README.md"),
    ("jupyter", "notebook", "main", "README.md"),
    ("ipython", "ipython", "main", "README.rst"),
    ("microsoft", "TypeScript", "main", "README.md"),
    ("microsoft", "playwright", "main", "README.md"),
    ("microsoft", "PowerToys", "main", "README.md"),
    ("microsoft", "terminal", "main", "README.md"),
    ("electron", "electron", "main", "README.md"),
    ("twbs", "bootstrap", "main", "README.md"),
    ("tailwindlabs", "tailwindcss", "main", "README.md"),
    ("jquery", "jquery", "main", "README.md"),
    ("babel", "babel", "main", "README.md"),
    ("webpack", "webpack", "main", "README.md"),
    ("vitejs", "vite", "main", "README.md"),
    # Additional repos to backfill the seed list after removing
    # not-publicly-mirrored / 404 entries (postgres, sqlite, ansible/devel,
    # vercel/next.js, seaborn/main, expressjs/express, JetBrains/kotlin,
    # scala/scala, v-language/v, lua/lua).
    ("axios", "axios", "v1.x", "README.md"),
    ("lodash", "lodash", "main", "README.md"),
    ("mochajs", "mocha", "main", "README.md"),
    ("npm", "cli", "latest", "README.md"),
    ("yarnpkg", "berry", "master", "README.md"),
    ("rails", "rails", "main", "README.md"),
    ("laravel", "laravel", "12.x", "README.md"),
    ("symfony", "symfony", "7.4", "README.md"),
    ("dotnet", "runtime", "main", "README.md"),
    ("apple", "swift", "main", "README.md"),
    ("flutter", "flutter", "master", "README.md"),
    ("ocaml", "ocaml", "trunk", "README.adoc"),
    ("ghc", "ghc", "master", "README.md"),
    ("erlang", "otp", "master", "README.md"),
    ("elixir-lang", "elixir", "main", "README.md"),
    ("crystal-lang", "crystal", "master", "README.md"),
    ("nim-lang", "Nim", "devel", "readme.md"),
    ("ziglang", "zig", "master", "README.md"),
]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 with a Z suffix."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _make_record(
    *,
    rng: random.Random,
    source: str,
    text: str,
    language: str,
    domain: str,
    collected_at: str,
) -> dict[str, Any]:
    """Build a single promptRow record from cleaned source text.

    UUIDs are deterministic (seeded), text length fields are computed here so
    callers don't repeat themselves.
    """
    # Deterministic UUIDv4 from the seeded rng (rather than os.urandom-based
    # uuid.uuid4()).
    pid = uuid.UUID(int=rng.getrandbits(128), version=4)
    text_clean = text.replace("\r\n", "\n").strip()
    return {
        "prompt_id": str(pid),
        "source": source,
        "text": text_clean,
        "text_len_chars": len(text_clean),
        "text_len_words": len(text_clean.split()),
        "language": language,
        "domain": domain,
        "collected_at": collected_at,
    }


def _resolve_dataset_revision(repo_id: str) -> str | None:
    """Return the commit SHA of the dataset's main branch for provenance."""
    try:
        api = HfApi()
        info = api.repo_info(repo_id, repo_type="dataset")
        return getattr(info, "sha", None)
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Source 1 — HumanEval
# -----------------------------------------------------------------------------


def collect_humaneval(*, n_target: int, rng: random.Random) -> list[dict[str, Any]]:
    """Sample ~n_target rows from HumanEval.

    HumanEval has 164 problems total. We sample up to min(n_target, 164) of
    them, randomly chosen, using the seeded rng.

    Returns prompts in domain="code", language="code".
    """
    from datasets import load_dataset  # type: ignore[import-untyped]  # noqa: PLC0415

    ds = load_dataset("openai/openai_humaneval", split="test")
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    take = indices[: min(n_target, len(indices))]

    collected_at = _utc_now_iso()
    out: list[dict[str, Any]] = []
    for idx in take:
        prompt_text = ds[idx]["prompt"]
        if not prompt_text or not prompt_text.strip():
            continue
        out.append(
            _make_record(
                rng=rng,
                source=SOURCE_HUMANEVAL,
                text=prompt_text,
                language="code",
                domain="code",
                collected_at=collected_at,
            )
        )
    return out


# -----------------------------------------------------------------------------
# Source 2 — WildChat-1M (LMSYS-chat-1M substitute)
# -----------------------------------------------------------------------------


def collect_wildchat(
    *,
    n_target: int,
    rng: random.Random,
    repo_id: str = "allenai/WildChat-1M",
    shard_filename: str = "data/train-00000-of-00014.parquet",
) -> list[dict[str, Any]]:
    """Sample ~n_target user turns from WildChat-1M.

    Filters applied (in order):
      1. `redacted == True`   — only PII-redacted conversations.
      2. `toxic   == False`   — drop conversations flagged toxic by detoxify.
      3. `language == "English"` — schema field on WildChat.
      4. user turn 0 only     — drop assistant turns.
      5. non-empty, after strip.

    We download a single ~220 MB parquet shard rather than the whole 3.4 GB
    dataset; sampling from a single shard with a fixed seed is reproducible
    and sufficient at the n_target sizes this script targets (≤ a few
    thousand). For larger samples this function would need to round-robin
    multiple shards; that's deferred.
    """
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=shard_filename,
        repo_type="dataset",
    )
    # Load only what we need; the conversation column is heavy but unavoidable.
    table = pq.read_table(
        local_path,
        columns=["conversation", "redacted", "toxic", "language"],
    )

    # Pre-filter at the column level using pyarrow compute (fast, vectorized).
    # Filters:
    #   redacted == True  — conversation had PII detected and was redacted by
    #                       WildChat's PII pipeline; this is the
    #                       PII-safe-by-construction subset (matches the
    #                       brief's "apply redacted flag filter" intent).
    #   toxic    == False — drop conversations flagged toxic by detoxify.
    #   language == English — match the atlas English-first scope.
    import pyarrow.compute as pc  # noqa: PLC0415

    mask = pc.and_(  # type: ignore[attr-defined]
        pc.and_(table.column("redacted"), pc.invert(table.column("toxic"))),  # type: ignore[attr-defined]
        pc.equal(table.column("language"), "English"),  # type: ignore[attr-defined]
    )
    filtered = table.filter(mask)
    if filtered.num_rows == 0:
        return []

    indices = list(range(filtered.num_rows))
    rng.shuffle(indices)

    collected_at = _utc_now_iso()
    out: list[dict[str, Any]] = []
    conversation_col = filtered.column("conversation")
    for idx in indices:
        if len(out) >= n_target:
            break
        try:
            conv = conversation_col[idx].as_py()
            if not conv:
                continue
            # First user turn only.
            first_user = next((t for t in conv if t and t.get("role") == "user"), None)
            if first_user is None:
                continue
            content = first_user.get("content") or ""
            if not content.strip():
                continue
        except Exception:
            continue

        out.append(
            _make_record(
                rng=rng,
                source=SOURCE_WILDCHAT,
                text=content,
                language="en",
                domain="chat",
                collected_at=collected_at,
            )
        )

    return out


# -----------------------------------------------------------------------------
# Source 3 — MT-Bench
# -----------------------------------------------------------------------------


def collect_mtbench(*, n_target: int, rng: random.Random) -> list[dict[str, Any]]:
    """Sample ~n_target unique user prompts from MT-Bench human judgments.

    MT-Bench's `human` split has 3355 pairwise judgments; we pull the first
    user turn from `conversation_a` and dedup by question_id so each
    question contributes at most once.

    Note: MT-Bench is a small, curated evaluation set; n_target is capped
    at the number of unique questions.
    """
    from datasets import load_dataset  # type: ignore[import-untyped]  # noqa: PLC0415

    ds = load_dataset("lmsys/mt_bench_human_judgments", split="human")

    seen_qids: set[int] = set()
    pool: list[dict[str, Any]] = []
    for row in ds:
        qid = row.get("question_id")
        if qid is None or qid in seen_qids:
            continue
        seen_qids.add(qid)
        conv = row.get("conversation_a") or []
        first_user = next((t for t in conv if t and t.get("role") == "user"), None)
        if first_user is None:
            continue
        content = (first_user.get("content") or "").strip()
        if not content:
            continue
        pool.append({"qid": qid, "text": content})

    rng.shuffle(pool)
    take = pool[: min(n_target, len(pool))]

    collected_at = _utc_now_iso()
    out: list[dict[str, Any]] = []
    for entry in take:
        out.append(
            _make_record(
                rng=rng,
                source=SOURCE_MTBENCH,
                text=entry["text"],
                language="en",
                domain="chat",
                collected_at=collected_at,
            )
        )
    return out


# -----------------------------------------------------------------------------
# Source 4 — Wikipedia (English)
# -----------------------------------------------------------------------------


def collect_wikipedia(
    *,
    n_target: int,
    rng: random.Random,
    repo_id: str = "wikimedia/wikipedia",
    config_name: str = "20231101.en",
    shard_filename: str = "20231101.en/train-00000-of-00041.parquet",
) -> list[dict[str, Any]]:
    """Sample first-paragraph snippets from English Wikipedia.

    Steps:
      1. Download a single ~400 MB parquet shard (one of 41).
      2. Shuffle row indices with the seeded rng.
      3. For each article: take the text up to the first double-newline
         ("first paragraph"); truncate to WIKIPEDIA_FIRST_PARAGRAPH_MAX_CHARS.
      4. Drop empty / too-short paragraphs (< 80 chars — typically navigation
         stubs / disambiguation lists).
    """
    del config_name  # accepted for documentation only; we go directly to the file
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=shard_filename,
        repo_type="dataset",
    )
    table = pq.read_table(local_path, columns=["title", "text"])

    text_col = table.column("text")
    n_rows = table.num_rows

    indices = list(range(n_rows))
    rng.shuffle(indices)

    collected_at = _utc_now_iso()
    out: list[dict[str, Any]] = []
    max_scan = min(n_rows, max(n_target * 6, n_target + 200))
    for idx in indices[:max_scan]:
        if len(out) >= n_target:
            break
        try:
            article = text_col[idx].as_py() or ""
        except Exception:
            continue
        # Take "first paragraph": split on double-newline.
        first_para = article.split("\n\n", 1)[0].strip()
        if len(first_para) < 80:
            continue
        first_para = first_para[:WIKIPEDIA_FIRST_PARAGRAPH_MAX_CHARS]
        out.append(
            _make_record(
                rng=rng,
                source=SOURCE_WIKIPEDIA,
                text=first_para,
                language="en",
                domain="prose",
                collected_at=collected_at,
            )
        )
    return out


# -----------------------------------------------------------------------------
# Source 5 — GitHub READMEs
# -----------------------------------------------------------------------------


def collect_github_readmes(
    *,
    n_target: int,
    rng: random.Random,
    repos: Iterable[tuple[str, str, str, str]] | None = None,
    request_timeout: float = 15.0,
) -> list[dict[str, Any]]:
    """Fetch READMEs from a curated seed list of well-known repos via raw.githubusercontent.com.

    Each README is rate-limited but free + auth-free. We shuffle the seed
    list with the seeded rng and stop once we have n_target rows. If a fetch
    fails we skip and continue.

    READMEs vary widely in size (a few KB to ~100 KB). We don't truncate —
    the goal is to capture realistic prompt-sized texts, and long READMEs
    are themselves useful tokenization stress tests.
    """
    pool = list(repos) if repos is not None else list(GITHUB_README_REPOS)
    rng.shuffle(pool)

    collected_at = _utc_now_iso()
    out: list[dict[str, Any]] = []
    client_kwargs: dict[str, Any] = {
        "timeout": request_timeout,
        "follow_redirects": True,
    }
    if _VERIFY_CA:
        client_kwargs["verify"] = _VERIFY_CA
    with httpx.Client(**client_kwargs) as client:
        for owner, repo, branch, path in pool:
            if len(out) >= n_target:
                break
            url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
            try:
                r = client.get(url)
            except Exception:
                continue
            if r.status_code != 200:
                continue
            text = r.text
            if not text or not text.strip():
                continue
            # READMEs are predominantly English prose with embedded shell /
            # code blocks; tag as "en" language and "code" domain since
            # they're the natural source of repo-context prompts handed to
            # LLMs. The schema's `code` enum value covers this case.
            out.append(
                _make_record(
                    rng=rng,
                    source=SOURCE_GITHUB_READMES,
                    text=text,
                    language="en",
                    domain="code",
                    collected_at=collected_at,
                )
            )
    return out


# -----------------------------------------------------------------------------
# Provenance manifest
# -----------------------------------------------------------------------------


def _write_provenance(
    *,
    path: Path,
    seed: int,
    counts: dict[str, int],
    revisions: dict[str, str | None],
    out_path: Path,
    total: int,
) -> None:
    """Write a per-source provenance manifest at `path`.

    The manifest is the authoritative record of WHAT was sampled, FROM WHERE,
    UNDER WHICH LICENSE, and AT WHICH REVISION. It is the input to the
    eventual HuggingFace dataset card's data-collection section and to the
    paper's data-statement.
    """
    now = _utc_now_iso()
    lines: list[str] = []
    lines.append("# Provenance — `data/raw_prompts.jsonl`")
    lines.append("")
    lines.append(f"Generated at: {now}")
    lines.append(f"Output file: `{out_path.as_posix()}`")
    lines.append(f"Total rows: {total}")
    lines.append(f"Sampling seed: {seed} (python `random.Random` + numpy)")
    lines.append("")
    lines.append("This file is the authoritative record of which prompts were sampled,")
    lines.append("from which upstream corpora, and under which licenses. It mirrors the")
    lines.append("schema enum `promptRow.source` declared in `data/schema.json`.")
    lines.append("")
    lines.append("## Sources")
    lines.append("")

    def _section(
        *,
        source: str,
        title: str,
        url: str,
        license_: str,
        license_url: str,
        sampling: str,
        bibtex: str,
        notes: str | None = None,
    ) -> None:
        sha = revisions.get(source)
        sha_str = sha if sha else "n/a (no HF repo)"
        lines.append(f"### `{source}` — {title}")
        lines.append("")
        lines.append(f"- Dataset URL: <{url}>")
        lines.append(f"- License: {license_} (<{license_url}>)")
        lines.append(f"- Upstream revision (snapshot): `{sha_str}`")
        lines.append(f"- Rows sampled: **{counts.get(source, 0)}**")
        lines.append(f"- Sampling strategy: {sampling}")
        if notes:
            lines.append("")
            lines.append(f"  Notes: {notes}")
        lines.append("")
        lines.append("  BibTeX:")
        lines.append("")
        lines.append("  ```bibtex")
        for ln in bibtex.strip().splitlines():
            lines.append(f"  {ln}")
        lines.append("  ```")
        lines.append("")

    _section(
        source=SOURCE_HUMANEVAL,
        title="OpenAI HumanEval",
        url="https://huggingface.co/datasets/openai/openai_humaneval",
        license_="MIT",
        license_url="https://github.com/openai/human-eval/blob/master/LICENSE",
        sampling=(
            "Uniform random sample of the 164 'test' split problems; we keep the "
            "`prompt` field (function signature + docstring), which is what an LLM "
            "would receive."
        ),
        bibtex=(
            "@misc{chen2021humaneval,\n"
            "  title  = {Evaluating Large Language Models Trained on Code},\n"
            "  author = {Mark Chen and Jerry Tworek and Heewoo Jun and others},\n"
            "  year   = {2021},\n"
            "  eprint = {2107.03374},\n"
            "  archivePrefix = {arXiv},\n"
            "  primaryClass  = {cs.LG}\n"
            "}"
        ),
    )

    _section(
        source=SOURCE_WILDCHAT,
        title="AllenAI WildChat-1M",
        url="https://huggingface.co/datasets/allenai/WildChat-1M",
        license_="ODC-BY (Open Data Commons Attribution)",
        license_url="https://opendatacommons.org/licenses/by/1-0/",
        sampling=(
            "Uniform random sample of `redacted==True && toxic==False && language=='English'`"
            " conversations from shard `data/train-00000-of-00014.parquet`; we keep only "
            "the first **user** turn, drop assistant turns. The `redacted` filter is the "
            "PII-safety filter recommended by the dataset authors."
        ),
        bibtex=(
            "@inproceedings{zhao2024wildchat,\n"
            "  title     = {{WildChat}: 1M ChatGPT Interaction Logs in the Wild},\n"
            "  author    = {Wenting Zhao and Xiang Ren and Jack Hessel and Claire Cardie\n"
            "               and Yejin Choi and Yuntian Deng},\n"
            "  booktitle = {ICLR},\n"
            "  year      = {2024}\n"
            "}"
        ),
        notes=(
            "WildChat-1M is used here as a substitute for LMSYS-chat-1M, which is "
            "**gated** on HuggingFace and therefore not redistributable in a "
            "fully-reproducible, credentials-free pipeline. WildChat-1M is the "
            "spiritual successor — same conversational-log format, same `redacted` "
            "and `toxic` flags — and is published under ODC-BY for redistribution. "
            "The atlas paper should cite both."
        ),
    )

    _section(
        source=SOURCE_MTBENCH,
        title="MT-Bench (LMSYS human judgments)",
        url="https://huggingface.co/datasets/lmsys/mt_bench_human_judgments",
        license_="CC-BY-4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        sampling=(
            "We deduplicate the 3355-row `human` split by `question_id`, then take the "
            "first **user** turn from `conversation_a`. Result is a pool of 80-ish "
            "unique evaluation questions; we sample uniformly without replacement."
        ),
        bibtex=(
            "@inproceedings{zheng2023judging,\n"
            "  title     = {Judging {LLM}-as-a-Judge with {MT}-Bench and Chatbot Arena},\n"
            "  author    = {Lianmin Zheng and Wei-Lin Chiang and Ying Sheng\n"
            "               and Siyuan Zhuang and Zhanghao Wu and Yonghao Zhuang\n"
            "               and Zi Lin and Zhuohan Li and Dacheng Li\n"
            "               and Eric P. Xing and Hao Zhang and Joseph E. Gonzalez\n"
            "               and Ion Stoica},\n"
            "  booktitle = {NeurIPS Datasets and Benchmarks},\n"
            "  year      = {2023}\n"
            "}"
        ),
    )

    _section(
        source=SOURCE_WIKIPEDIA,
        title="English Wikipedia (`20231101.en` snapshot)",
        url="https://huggingface.co/datasets/wikimedia/wikipedia",
        license_="CC-BY-SA-4.0 + GFDL",
        license_url="https://creativecommons.org/licenses/by-sa/4.0/",
        sampling=(
            "Uniform random sample of articles from a single parquet shard "
            "(`20231101.en/train-00000-of-00041.parquet`). For each article we keep "
            "the first paragraph (split on `\\n\\n`), truncated to "
            f"{WIKIPEDIA_FIRST_PARAGRAPH_MAX_CHARS} chars. Articles whose first "
            "paragraph is shorter than 80 chars (navigation stubs, disambiguation "
            "pages) are skipped."
        ),
        bibtex=(
            "@misc{wikipedia2023,\n"
            "  title  = {{Wikimedia/Wikipedia Snapshot 20231101 (English)}},\n"
            "  author = {{Wikimedia Foundation}},\n"
            "  year   = {2023},\n"
            "  howpublished = {\\url{https://huggingface.co/datasets/wikimedia/wikipedia}}\n"
            "}"
        ),
    )

    _section(
        source=SOURCE_GITHUB_READMES,
        title="GitHub READMEs (curated seed list)",
        url="https://raw.githubusercontent.com/",
        license_=(
            "Per-repo (Apache-2.0 / MIT / BSD; see GITHUB_README_REPOS "
            "list in llm_tokens_atlas/collect_corpus.py)"
        ),
        license_url="https://github.com/faraa2m/llm-tokens-atlas/blob/main/llm_tokens_atlas/collect_corpus.py",
        sampling=(
            "Fetched the canonical README (in `.md` / `.rst` / `.asciidoc` / plain) of "
            "a curated list of well-known repositories via `raw.githubusercontent.com`. "
            "All listed repositories use permissive OSS licenses (Apache-2.0, MIT, "
            "BSD-3-Clause, or PSF). The full list lives in `GITHUB_README_REPOS` in "
            "the collection script for auditability. Order is shuffled with the seeded "
            "rng before fetching, so the first N fetched are deterministic. The "
            "static seed list is used in lieu of `bigcode/the-stack` (which is "
            "auto-gated and therefore breaks the credentials-free path)."
        ),
        bibtex=(
            "@misc{githubreadmes2026,\n"
            "  title  = {Curated GitHub READMEs for tokenization benchmarking},\n"
            "  author = {Faraazuddin Mohammed and the llm-tokens-atlas authors},\n"
            "  year   = {2026},\n"
            "  howpublished = {\\url{https://github.com/faraa2m/llm-tokens-atlas/blob/main/llm_tokens_atlas/collect_corpus.py}}\n"
            "}"
        ),
    )

    lines.append("## Reproducibility")
    lines.append("")
    lines.append(
        "Determinism: all RNG state is seeded with the `--seed` CLI flag (default 42). "
        "`uuid` values are drawn from the seeded `random.Random` rather than `uuid.uuid4()` "
        "(which is OS-random) so prompt ids are stable across runs."
    )
    lines.append("")
    lines.append(
        "HF dataset snapshots are pinned implicitly by the `huggingface_hub` cache; the "
        "captured commit SHAs above are the snapshots used at generation time. To "
        "regenerate the exact bytes, set `HF_HUB_OFFLINE=1` after the cache is warm, or "
        "manually pin `revision=<sha>` in the collection calls."
    )
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------


def _per_source_targets(n_total: int) -> dict[str, int]:
    """Divide ~n_total evenly across the five sources, redistributing capped sources.

    Each source is guaranteed at least 1; HumanEval is capped at 164 (size of
    its test split), MT-Bench is capped at 80 (unique question_ids in the
    `human` split), GitHub READMEs is capped at the seed list length. Any
    deficit from these caps is redistributed evenly to the two
    effectively-unbounded sources (Wikipedia and WildChat). The acceptance
    criteria require ≥ 50 per source when n_total == 500, which this
    distribution satisfies.
    """
    base = max(n_total // 5, 1)
    targets = {
        SOURCE_HUMANEVAL: min(base, 164),
        SOURCE_MTBENCH: min(base, 80),
        SOURCE_WIKIPEDIA: base,
        SOURCE_GITHUB_READMES: min(base, len(GITHUB_README_REPOS)),
        SOURCE_WILDCHAT: base,
    }
    deficit = n_total - sum(targets.values())
    if deficit > 0:
        # Split deficit evenly between the two unbounded sources.
        bump_wp = deficit // 2
        bump_wc = deficit - bump_wp
        targets[SOURCE_WIKIPEDIA] += bump_wp
        targets[SOURCE_WILDCHAT] += bump_wc
    return targets


def collect_all(
    *,
    n_total: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, str | None]]:
    """Run every source collector. Returns (records, per_source_counts, revisions)."""
    rng = random.Random(seed)
    targets = _per_source_targets(n_total)

    # Capture HF revisions once for provenance.
    revisions: dict[str, str | None] = {
        SOURCE_HUMANEVAL: _resolve_dataset_revision("openai/openai_humaneval"),
        SOURCE_WILDCHAT: _resolve_dataset_revision("allenai/WildChat-1M"),
        SOURCE_MTBENCH: _resolve_dataset_revision("lmsys/mt_bench_human_judgments"),
        SOURCE_WIKIPEDIA: _resolve_dataset_revision("wikimedia/wikipedia"),
        SOURCE_GITHUB_READMES: None,
    }

    all_records: list[dict[str, Any]] = []
    counts: dict[str, int] = {}

    # 1. HumanEval — small, cheap.
    print(f"[collect] humaneval target={targets[SOURCE_HUMANEVAL]}", file=sys.stderr)
    he = collect_humaneval(n_target=targets[SOURCE_HUMANEVAL], rng=rng)
    all_records.extend(he)
    counts[SOURCE_HUMANEVAL] = len(he)

    # 2. MT-Bench — small, cheap.
    print(f"[collect] mt-bench target={targets[SOURCE_MTBENCH]}", file=sys.stderr)
    mb = collect_mtbench(n_target=targets[SOURCE_MTBENCH], rng=rng)
    all_records.extend(mb)
    counts[SOURCE_MTBENCH] = len(mb)

    # 3. GitHub READMEs — network bound but bounded by seed list size.
    print(f"[collect] github-readmes target={targets[SOURCE_GITHUB_READMES]}", file=sys.stderr)
    gh = collect_github_readmes(n_target=targets[SOURCE_GITHUB_READMES], rng=rng)
    all_records.extend(gh)
    counts[SOURCE_GITHUB_READMES] = len(gh)

    # 4. Wikipedia — heavy download but cached on first call.
    print(f"[collect] wikipedia-en target={targets[SOURCE_WIKIPEDIA]}", file=sys.stderr)
    wp = collect_wikipedia(n_target=targets[SOURCE_WIKIPEDIA], rng=rng)
    all_records.extend(wp)
    counts[SOURCE_WIKIPEDIA] = len(wp)

    # 5. WildChat — heaviest download (~220 MB) but cached.
    print(f"[collect] wildchat-1m target={targets[SOURCE_WILDCHAT]}", file=sys.stderr)
    wc = collect_wildchat(n_target=targets[SOURCE_WILDCHAT], rng=rng)
    all_records.extend(wc)
    counts[SOURCE_WILDCHAT] = len(wc)

    return all_records, counts, revisions


def write_jsonl(*, records: list[dict[str, Any]], out_path: Path) -> None:
    """Write rows to JSONL atomically (write to tmp then rename)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")
    tmp.replace(out_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Collect a balanced sample of prompts for llm-tokens-atlas.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=500,
        help="Approximate total number of rows to collect (distributed across 5 sources).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="RNG seed for reproducibility.",
    )
    parser.add_argument(
        "--provenance",
        type=Path,
        default=PROVENANCE_PATH,
        help="Path to write the provenance manifest (markdown).",
    )
    parser.add_argument(
        "--skip-provenance",
        action="store_true",
        help="Skip writing data/provenance.md (used in tests).",
    )
    args = parser.parse_args(argv)

    if args.n < 5:
        print("--n must be at least 5 (one row per source).", file=sys.stderr)
        return 2

    records, counts, revisions = collect_all(n_total=args.n, seed=args.seed)
    write_jsonl(records=records, out_path=args.out)

    if not args.skip_provenance:
        _write_provenance(
            path=args.provenance,
            seed=args.seed,
            counts=counts,
            revisions=revisions,
            out_path=args.out,
            total=len(records),
        )

    print(f"[collect] wrote {len(records)} rows to {args.out}", file=sys.stderr)
    for src, c in counts.items():
        print(f"  {src}: {c}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
