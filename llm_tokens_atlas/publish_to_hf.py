"""Publish the LLM Tokens Atlas dataset to the Hugging Face Hub.

Usage
-----
    uv run python llm_tokens_atlas/publish_to_hf.py \
        --parquet data/processed/atlas.parquet \
        --repo faraa2m/llm-tokens-atlas

Pass ``--dry-run`` to print the planned upload manifest without touching the
Hub. Pass ``--private`` to create or treat the dataset repo as private (the
default is public). The script reads the Hugging Face access token from the
``HF_TOKEN`` environment variable and exits with a non-zero status code if it
is missing.

The artifact set uploaded mirrors what the dataset card promises: the canonical
parquet under ``data/processed/atlas.parquet``, the dataset card itself as the
repo's ``README.md`` (HF serves the card from the root README), the data
license, the formal JSON schema, the per-row provenance log, and the
tokenizer/API lockfile. Missing optional files (schema, provenance, lockfile)
are reported as warnings rather than fatal so that incremental publication is
possible while the sibling agents finish their deliverables; the parquet and
README are mandatory.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# Files in (repo-relative source path, path-in-HF-repo).
# Mandatory entries must exist on disk; optional entries are uploaded only when
# present and produce a warning otherwise.
MANDATORY_FILES: tuple[tuple[str, str], ...] = (
    # The dataset card MUST be uploaded as README.md at the repo root — that's
    # where huggingface.co/datasets/* serves it from. The on-disk copy lives at
    # data/README.md so it sits next to the rest of the dataset payload.
    ("data/README.md", "README.md"),
)

# Mandatory IF the user actually passed --parquet (we add it dynamically).
# Other mandatory data-payload files:
MANDATORY_DATA_FILES: tuple[tuple[str, str], ...] = (
    ("LICENSE-DATA", "LICENSE-DATA"),
)

OPTIONAL_FILES: tuple[tuple[str, str], ...] = (
    ("data/schema.json", "data/schema.json"),
    ("data/provenance.md", "data/provenance.md"),
    ("data/lockfile.json", "data/lockfile.json"),
)


@dataclass(frozen=True)
class UploadItem:
    """One planned upload: local source path + path-in-HF-repo."""

    local: Path
    remote: str
    mandatory: bool


def repo_root() -> Path:
    """Project root (the directory containing pyproject.toml)."""
    return Path(__file__).resolve().parent.parent


def build_manifest(
    parquet_path: Path,
    *,
    root: Path | None = None,
) -> list[UploadItem]:
    """Build the ordered list of files we intend to upload."""
    root = root or repo_root()
    items: list[UploadItem] = []

    for src, dst in MANDATORY_FILES:
        items.append(UploadItem(local=root / src, remote=dst, mandatory=True))

    # Parquet is mandatory once supplied. We upload it under the exact path the
    # dataset card's `configs` block points to so the HF dataset viewer renders
    # it by default.
    items.append(
        UploadItem(
            local=parquet_path if parquet_path.is_absolute() else root / parquet_path,
            remote="data/processed/atlas.parquet",
            mandatory=True,
        )
    )

    for src, dst in MANDATORY_DATA_FILES:
        items.append(UploadItem(local=root / src, remote=dst, mandatory=True))

    for src, dst in OPTIONAL_FILES:
        items.append(UploadItem(local=root / src, remote=dst, mandatory=False))

    return items


def print_manifest(items: list[UploadItem], repo_id: str, *, dry_run: bool) -> None:
    header = "PLANNED UPLOAD" if dry_run else "UPLOAD MANIFEST"
    print(f"=== {header} → datasets/{repo_id} ===")
    for item in items:
        exists = item.local.exists()
        flag = "REQ" if item.mandatory else "opt"
        status = "ok " if exists else ("MISS" if item.mandatory else "skip")
        size = ""
        if exists:
            try:
                size = f" ({item.local.stat().st_size:,} bytes)"
            except OSError:
                size = ""
        print(f"  [{flag}] [{status}] {item.local} -> {item.remote}{size}")
    print("=== END MANIFEST ===")


def check_token() -> str:
    """Read HF_TOKEN, exit non-zero with clear instructions if absent."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.stderr.write(
            "ERROR: HF_TOKEN is not set in the environment.\n"
            "\n"
            "To publish to the Hugging Face Hub:\n"
            "  1. Create a write token at https://huggingface.co/settings/tokens\n"
            "     (scope: 'write', or fine-grained with 'Write to a specific dataset').\n"
            "  2. Export it for this shell session:\n"
            "       export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx\n"
            "  3. Re-run this script.\n"
            "\n"
            "If you only want to preview what would be uploaded, pass --dry-run.\n"
        )
        raise SystemExit(2)
    return token


def verify_mandatory(items: list[UploadItem]) -> list[UploadItem]:
    """Filter out missing optional files; raise if any mandatory file is missing."""
    final: list[UploadItem] = []
    missing_required: list[Path] = []
    missing_optional: list[Path] = []
    for item in items:
        if item.local.exists():
            final.append(item)
        elif item.mandatory:
            missing_required.append(item.local)
        else:
            missing_optional.append(item.local)

    if missing_required:
        sys.stderr.write("ERROR: mandatory files are missing on disk:\n")
        for path in missing_required:
            sys.stderr.write(f"  - {path}\n")
        sys.stderr.write(
            "\nThese files must exist before publishing. The parquet is built by\n"
            "the dataset pipeline (`make reproduce`); the dataset card / license\n"
            "live in the repo. Run the pipeline or check the paths and retry.\n"
        )
        raise SystemExit(3)

    for path in missing_optional:
        sys.stderr.write(
            f"WARN: optional file not on disk; skipping upload: {path}\n"
        )
    return final


def push(items: list[UploadItem], repo_id: str, *, private: bool, token: str) -> str:
    """Upload to HF. Returns the dataset URL."""
    # Lazy import so --dry-run works without the SDK installed.
    from huggingface_hub import HfApi  # noqa: PLC0415

    api = HfApi(token=token)

    # Idempotent repo creation. exist_ok=True turns "already exists" into a noop.
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )

    for item in items:
        print(f"  uploading {item.local} -> {item.remote} ...")
        api.upload_file(
            path_or_fileobj=str(item.local),
            path_in_repo=item.remote,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"chore(publish): upload {item.remote}",
        )

    return f"https://huggingface.co/datasets/{repo_id}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Publish the LLM Tokens Atlas dataset to Hugging Face.",
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        default=Path("data/processed/atlas.parquet"),
        help="Path to the parquet dataset file (default: data/processed/atlas.parquet).",
    )
    parser.add_argument(
        "--repo",
        default="faraa2m/llm-tokens-atlas",
        help="Target HF dataset repo id (default: faraa2m/llm-tokens-atlas).",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create/treat the dataset repo as private (default: public).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned upload manifest and exit without uploading.",
    )
    args = parser.parse_args(argv)

    items = build_manifest(args.parquet)
    print_manifest(items, args.repo, dry_run=args.dry_run)

    if args.dry_run:
        # On dry-run we still warn about missing mandatory files so the user
        # can see them, but we exit zero to indicate the dry-run itself
        # succeeded.
        missing = [i.local for i in items if i.mandatory and not i.local.exists()]
        if missing:
            sys.stderr.write(
                "NOTE: dry-run completed but mandatory files were missing — see\n"
                "      manifest. A real run will fail until they exist.\n"
            )
        return 0

    token = check_token()
    items = verify_mandatory(items)
    url = push(items, args.repo, private=args.private, token=token)
    print(f"\nPublished: {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
