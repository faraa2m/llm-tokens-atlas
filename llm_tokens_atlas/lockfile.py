"""Generate data/lockfile.json — the reproducibility manifest for the atlas dataset.

Captures, at build time:
  - schema_version             : `$id` of data/schema.json (pinned URL).
  - generated_at               : ISO-8601 UTC timestamp.
  - dataset_commit             : git SHA of the dataset commit (HEAD) when known.
  - python                     : python version + implementation.
  - tokenizer_versions         : per-provider tokenizer/library identifiers
                                 (resolved against the active environment).
  - api_endpoints              : provider countTokens endpoint URLs + the
                                 specific API version we treat as canonical.
  - dependencies               : pinned versions of every Python dependency
                                 declared in pyproject.toml whose package is
                                 importable in the current environment.

Pinned for reproducibility per the project plan — `make reproduce` should be
able to recreate the dataset from the lockfile alone.

Usage:
    uv run python llm_tokens_atlas/lockfile.py --out data/lockfile.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata as md
import json
import platform
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCHEMA = REPO_ROOT / "data" / "schema.json"
DEFAULT_PYPROJECT = REPO_ROOT / "pyproject.toml"


# Per-provider tokenizer libraries we expect to drive offline counts.
# Maps provider -> (distribution name on PyPI, importable module name).
TOKENIZER_PACKAGES: dict[str, list[tuple[str, str]]] = {
    "openai": [("tiktoken", "tiktoken")],
    "anthropic": [("anthropic", "anthropic")],
    "google": [("google-generativeai", "google.generativeai")],
    "mistral": [("mistral-common", "mistral_common")],
    "cohere": [("cohere", "cohere")],
}


# Stable countTokens endpoint identifiers per provider. The exact URL
# evolves over time; pinning the path + API version here makes reproducer
# behavior auditable. The empirical-counts agent reads from these.
API_ENDPOINTS: dict[str, dict[str, str]] = {
    "anthropic": {
        "endpoint": "https://api.anthropic.com/v1/messages/count_tokens",
        "api_version": "2023-06-01",
    },
    "google": {
        "endpoint": "https://generativelanguage.googleapis.com/v1beta/models/{model}:countTokens",
        "api_version": "v1beta",
    },
    "openai": {
        "endpoint": "local:tiktoken.encoding_for_model",
        "api_version": "tiktoken",
    },
    "mistral": {
        "endpoint": "local:mistral_common.Tokenizer",
        "api_version": "mistral-common",
    },
    "cohere": {
        "endpoint": "https://api.cohere.com/v1/tokenize",
        "api_version": "v1",
    },
}


def _git_sha(repo_root: Path) -> str | None:
    """Return the current git HEAD SHA, or None if git is unavailable."""
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8").strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _safe_version(distribution_name: str) -> str | None:
    """Return the installed version of a distribution, or None if not present."""
    try:
        return md.version(distribution_name)
    except md.PackageNotFoundError:
        return None


def _tokenizer_versions() -> dict[str, list[dict[str, str | None]]]:
    """Resolve per-provider tokenizer library versions from the active env."""
    out: dict[str, list[dict[str, str | None]]] = {}
    for provider, packages in TOKENIZER_PACKAGES.items():
        entries: list[dict[str, str | None]] = []
        for dist_name, module_name in packages:
            entries.append(
                {
                    "package": dist_name,
                    "module": module_name,
                    "version": _safe_version(dist_name),
                }
            )
        out[provider] = entries
    return out


def _pyproject_dependencies(pyproject_path: Path) -> list[dict[str, str | None]]:
    """Read declared dependencies from pyproject.toml and resolve installed versions."""
    if not pyproject_path.exists():
        return []
    with pyproject_path.open("rb") as f:
        data = tomllib.load(f)
    raw = (data.get("project") or {}).get("dependencies", [])
    deps: list[dict[str, str | None]] = []
    for spec in raw:
        # `pkg>=1.2` -> "pkg"; ignore extras + version markers.
        name = (
            spec.split(";")[0].split("[")[0]
            .replace(">=", " ").replace("==", " ").replace("<=", " ")
            .replace("<", " ").replace(">", " ").replace("~=", " ")
            .split()[0]
            .strip()
        )
        deps.append({"name": name, "spec": spec, "installed": _safe_version(name)})
    return deps


def _schema_id(schema_path: Path) -> str | None:
    """Return the $id of the atlas schema, if readable."""
    if not schema_path.exists():
        return None
    try:
        with schema_path.open("r", encoding="utf-8") as f:
            doc = json.load(f)
        return doc.get("$id")
    except (OSError, json.JSONDecodeError):
        return None


def build_lockfile(
    repo_root: Path = REPO_ROOT,
    schema_path: Path = DEFAULT_SCHEMA,
    pyproject_path: Path = DEFAULT_PYPROJECT,
) -> dict[str, Any]:
    """Assemble the lockfile payload."""
    now = dt.datetime.now(dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    return {
        "schema_version": _schema_id(schema_path),
        "generated_at": now,
        "dataset_commit": _git_sha(repo_root),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": _relative_executable(repo_root),
        },
        "tokenizer_versions": _tokenizer_versions(),
        "api_endpoints": API_ENDPOINTS,
        "dependencies": _pyproject_dependencies(pyproject_path),
    }


def _relative_executable(repo_root: Path) -> str:
    """Redact absolute paths from sys.executable to keep the lockfile portable
    and free of host-specific identifiers (macOS usernames, home paths)."""
    exe = Path(sys.executable)
    try:
        return f"<repo-root>/{exe.relative_to(repo_root)}"
    except ValueError:
        return f"<system>/{exe.name}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lockfile",
        description="Generate data/lockfile.json capturing the reproducibility pins.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "data" / "lockfile.json",
        help="Output JSON path (default: data/lockfile.json).",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=DEFAULT_SCHEMA,
        help="Path to data/schema.json (default: repo-relative).",
    )
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=DEFAULT_PYPROJECT,
        help="Path to pyproject.toml (default: repo-relative).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_lockfile(
        repo_root=REPO_ROOT,
        schema_path=args.schema,
        pyproject_path=args.pyproject,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
        f.write("\n")
    print(f"wrote lockfile -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
