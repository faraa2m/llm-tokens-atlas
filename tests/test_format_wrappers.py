"""Byte-level pin tests for `llm_tokens_atlas/format_wrappers.py`.

These wrappers are load-bearing: both the offline counter (this repo) and
the empirical counter (sibling agent) must produce *byte-identical* wrapped
strings for offline-vs-empirical drift measurement to be meaningful. The
tests below pin the exact output of each wrapper so cross-agent drift is
detectable by CI rather than discovered weeks later in the analysis
notebook.

If a test here fails because the wrapper output changed, the right response
is almost always:
  1. update the docstring of `format_wrappers.py` to reflect the new shape;
  2. update the empirical-counter agent's expectations in lockstep;
  3. update the corresponding pin string here.
Never silently relax these assertions.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from llm_tokens_atlas.format_wrappers import (  # noqa: E402
    ALL_FORMATS,
    is_format,
    wrap,
    wrap_as_json,
    wrap_as_markdown,
    wrap_as_plain,
    wrap_as_xml,
    wrap_as_yaml,
)

SAMPLE = "What is the capital of France?"


def test_all_formats_constant() -> None:
    assert ALL_FORMATS == ("plain", "markdown", "json", "xml", "yaml")


def test_plain_is_identity() -> None:
    assert wrap_as_plain(SAMPLE) == SAMPLE


def test_markdown_wraps_with_h2_prompt_header() -> None:
    assert wrap_as_markdown(SAMPLE) == "## prompt\n\nWhat is the capital of France?\n"


def test_json_wraps_in_compact_prompt_field() -> None:
    """Compact JSON; matches `JSON.stringify({prompt: SAMPLE})` byte-for-byte."""
    assert (
        wrap_as_json(SAMPLE)
        == '{"prompt":"What is the capital of France?"}'
    )


def test_json_handles_quotes_and_unicode() -> None:
    text = 'He said "hello" — and waved 👋.'
    wrapped = wrap_as_json(text)
    # Round-trip equality.
    assert json.loads(wrapped) == {"prompt": text}
    # No whitespace surplus from separators.
    assert '", "' not in wrapped


def test_yaml_wraps_in_literal_block_scalar() -> None:
    """Single-line input becomes a single indented line under a block scalar."""
    assert (
        wrap_as_yaml(SAMPLE)
        == "prompt: |-\n  What is the capital of France?\n"
    )


def test_yaml_preserves_internal_newlines_under_indent() -> None:
    """Multi-line input keeps every original line, indented by two spaces."""
    text = "line one\nline two\nline three"
    expected = "prompt: |-\n  line one\n  line two\n  line three\n"
    assert wrap_as_yaml(text) == expected


def test_xml_wraps_in_prompt_element_with_escaping() -> None:
    assert wrap_as_xml(SAMPLE) == "<prompt>What is the capital of France?</prompt>"


def test_xml_escapes_special_chars() -> None:
    """Inside the <prompt> envelope, all five XML special chars are escaped.

    We check the content slice between `<prompt>` and `</prompt>` rather than
    the whole wrapped string, since the envelope tags themselves contain
    `<`, `>` (and the closing tag also has `/`).
    """
    text = 'a < b & c > d "quoted" \'apos\''
    wrapped = wrap_as_xml(text)
    # Verify ordered escaping: ampersand expansion did not double-escape.
    assert "&amp;amp;" not in wrapped
    # Pin exact byte output for the content slice between the envelope tags.
    prefix, suffix = "<prompt>", "</prompt>"
    assert wrapped.startswith(prefix) and wrapped.endswith(suffix)
    content = wrapped[len(prefix) : -len(suffix)]
    assert content == "a &lt; b &amp; c &gt; d &quot;quoted&quot; &apos;apos&apos;"


def test_wrap_dispatch_matches_individual_helpers() -> None:
    for fmt in ALL_FORMATS:
        helper = {
            "plain": wrap_as_plain,
            "markdown": wrap_as_markdown,
            "json": wrap_as_json,
            "xml": wrap_as_xml,
            "yaml": wrap_as_yaml,
        }[fmt]
        assert wrap(SAMPLE, fmt) == helper(SAMPLE)


def test_wrap_accepts_text_as_plain_alias() -> None:
    assert wrap(SAMPLE, "text") == wrap_as_plain(SAMPLE)


def test_wrap_rejects_unknown_format() -> None:
    import pytest

    with pytest.raises(ValueError):
        wrap(SAMPLE, "rst")


def test_is_format_recognizes_canonical_and_alias() -> None:
    for fmt in ALL_FORMATS:
        assert is_format(fmt)
    assert is_format("text")
    assert not is_format("rst")


def test_wrappers_produce_distinct_strings() -> None:
    """The five canonical formats must produce 5 distinct outputs on any nontrivial input.

    This is the prerequisite for the format-drift study: if any two wrappers
    collapsed to the same byte string, those two formats would tokenize
    identically and the cell would carry no information.
    """
    outputs = {fmt: wrap(SAMPLE, fmt) for fmt in ALL_FORMATS}
    assert len(set(outputs.values())) == len(ALL_FORMATS), (
        f"wrappers collapsed for input {SAMPLE!r}: {outputs!r}"
    )


def test_wrappers_are_pure_functions() -> None:
    """Calling a wrapper twice yields the same string."""
    for fmt in ALL_FORMATS:
        a = wrap(SAMPLE, fmt)
        b = wrap(SAMPLE, fmt)
        assert a == b
