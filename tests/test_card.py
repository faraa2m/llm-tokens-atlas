"""Smoke tests for the Hugging Face dataset card at data/README.md.

These tests guard the three properties that make the card publishable without
manual edits:

  1. The card file exists and is non-trivial markdown.
  2. The YAML frontmatter parses cleanly.
  3. The whole document passes Hugging Face's ``RepoCard`` validator (i.e. it
     would be accepted by the Hub on push).
  4. The BibTeX citation block is syntactically balanced.

The tests intentionally avoid importing the upload script or hitting the
network; the card itself is the publication-grade artifact this suite
defends.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml
from huggingface_hub.repocard import DatasetCard, RepoCard

REPO_ROOT = Path(__file__).resolve().parent.parent
CARD_PATH = REPO_ROOT / "data" / "README.md"


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _read_card_text() -> str:
    """Return the full text of the dataset card."""
    return CARD_PATH.read_text(encoding="utf-8")


def _split_frontmatter(text: str) -> tuple[str, str]:
    """Return ``(yaml_block, body)`` for a markdown file with YAML frontmatter.

    Raises if the file does not start with the ``---`` fence.
    """
    if not text.startswith("---\n"):
        pytest.fail("dataset card must begin with '---' YAML frontmatter fence")
    parts = text.split("\n---\n", 1)
    if len(parts) != 2:
        pytest.fail("dataset card YAML frontmatter is not properly closed with '---'")
    yaml_block = parts[0][len("---\n") :]
    body = parts[1]
    return yaml_block, body


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------


class TestDatasetCardExists:
    def test_card_file_exists(self) -> None:
        assert CARD_PATH.is_file(), f"missing dataset card at {CARD_PATH}"

    def test_card_is_non_trivial(self) -> None:
        text = _read_card_text()
        # 2 KB is a reasonable floor — full card is ~16 KB. Anything tiny
        # means a regression that gutted the card.
        assert len(text) > 2048, (
            f"dataset card is suspiciously short ({len(text)} bytes); "
            "expected a fully populated HF dataset card"
        )


class TestYamlFrontmatter:
    def test_frontmatter_parses(self) -> None:
        text = _read_card_text()
        yaml_block, _ = _split_frontmatter(text)
        data = yaml.safe_load(yaml_block)
        assert isinstance(data, dict), "YAML frontmatter must parse to a dict"

    def test_required_keys_present(self) -> None:
        """Brief specified these keys must appear; the validator alone
        doesn't enforce them, so we pin them here."""
        text = _read_card_text()
        yaml_block, _ = _split_frontmatter(text)
        data = yaml.safe_load(yaml_block)

        assert data.get("license") == "cc-by-4.0"
        assert data.get("pretty_name") == "LLM Tokens Atlas"

        languages = data.get("language")
        assert languages == ["en"] or languages == "en"

        size_categories = data.get("size_categories")
        assert size_categories in (["10K<n<100K"], "10K<n<100K")

        task_categories = data.get("task_categories")
        assert task_categories in (["other"], "other")

        tags = data.get("tags") or []
        for needed in ("llm", "tokenization", "benchmark", "calibration", "cost-estimation"):
            assert needed in tags, f"tag '{needed}' missing from frontmatter"

        # configs block must point at the parquet path the publish script uploads to.
        configs = data.get("configs") or []
        assert configs, "configs: block missing from frontmatter"
        first = configs[0]
        assert first.get("config_name") == "default"
        data_files = first.get("data_files") or []
        assert data_files, "configs[0].data_files missing"
        assert any(
            df.get("path") == "data/processed/atlas.parquet" for df in data_files
        ), "no data_files entry points at data/processed/atlas.parquet"


class TestHuggingFaceValidator:
    """The Hub's RepoCard / DatasetCard validators are what gate publication."""

    def test_repocard_load(self) -> None:
        # RepoCard.load is the lowest-level entry point: it parses frontmatter
        # plus body and rejects on either side. Passing here means the file
        # can be served as a Hub card without 4xx-ing on push.
        card = RepoCard.load(CARD_PATH)
        # validate() runs the Hub-side schema check.
        card.validate()

    def test_dataset_card_load(self) -> None:
        # DatasetCard layers DatasetCardData on top of RepoCard. It enforces
        # dataset-specific fields and catches typos in task_categories,
        # license, etc.
        card = DatasetCard.load(CARD_PATH)
        card.validate()

    def test_dataset_card_data_roundtrip(self) -> None:
        # Round-tripping through DatasetCardData proves the frontmatter
        # serialises back into valid YAML without lossy transformations.
        card = DatasetCard.load(CARD_PATH)
        serialised = str(card.data)
        reparsed = yaml.safe_load(serialised)
        assert isinstance(reparsed, dict)
        assert reparsed.get("license") == "cc-by-4.0"


class TestBibtexCitation:
    """Cheap syntactic checks; we don't need pybtex for this."""

    _BIB_FENCE_RE = re.compile(
        r"```bibtex\s*\n(?P<body>.*?)```",
        flags=re.DOTALL | re.IGNORECASE,
    )
    _ENTRY_RE = re.compile(
        r"@(?P<kind>[A-Za-z]+)\s*\{\s*(?P<key>[^,]+)\s*,(?P<fields>.*)\}",
        flags=re.DOTALL,
    )

    def _extract_bibtex(self) -> str:
        text = _read_card_text()
        match = self._BIB_FENCE_RE.search(text)
        assert match, "no ```bibtex fenced block found in dataset card"
        return match.group("body")

    def test_bibtex_block_present(self) -> None:
        body = self._extract_bibtex()
        assert "@misc" in body or "@article" in body or "@inproceedings" in body, (
            "BibTeX block missing a recognised entry type (@misc / @article / "
            "@inproceedings)"
        )

    def test_braces_balanced(self) -> None:
        body = self._extract_bibtex()
        open_braces = body.count("{")
        close_braces = body.count("}")
        assert open_braces == close_braces, (
            f"BibTeX braces are unbalanced: {{={open_braces} vs }}={close_braces}"
        )

    def test_entry_parses(self) -> None:
        body = self._extract_bibtex()
        match = self._ENTRY_RE.search(body)
        assert match, "BibTeX entry does not match expected '@kind{key, ...}' shape"
        assert match.group("kind"), "BibTeX entry has no kind"
        assert match.group("key").strip(), "BibTeX entry has no citation key"
        # Field shape sanity: must contain at least one 'name = {...}' pair.
        fields = match.group("fields")
        field_re = re.compile(r"[A-Za-z_]+\s*=\s*[\{\"]", flags=re.MULTILINE)
        assert field_re.search(fields), "BibTeX entry has no name = {...} fields"


class TestProviderCitations:
    """Per the brief, the card must honestly credit the prior surfacers
    (the two 2026 blog posts + tokencost) for the offline-vs-empirical
    drift observation. We pin those URLs here so a future edit cannot
    silently drop them."""

    REQUIRED_CITATIONS = (
        "claudecodecamp.com",
        "ai-software-engineer/anthropics-new-tokenizer",
        "github.com/AgentOps-AI/tokencost",
    )

    def test_required_citations_present(self) -> None:
        text = _read_card_text()
        missing = [needle for needle in self.REQUIRED_CITATIONS if needle not in text]
        assert not missing, (
            "dataset card is missing required prior-art citations: "
            + ", ".join(missing)
        )
