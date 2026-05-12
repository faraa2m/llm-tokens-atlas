"""Deterministic format wrappers — `plain | markdown | json | xml | yaml`.

Why this module exists
======================

The atlas measures token-count drift *as a function of how a prompt is wrapped*
(JSON vs YAML vs XML vs Markdown vs plain). For that measurement to be
meaningful, the same input string must produce the **same wrapped byte string**
everywhere — both when the offline tokenizer sees it and when the provider's
empirical countTokens API sees it.

Two design choices feed into this module:

1. We define **our own canonical wrappers** rather than mirror tokenometer's
   `toFormat()` in `packages/core/src/convert.ts`. Reason: tokenometer's YAML
   path uses the npm `yaml` package; PyYAML produces subtly different output
   for the same string (document-end markers, quoting style). If offline and
   empirical were computed against different byte streams the resulting
   "delta" would conflate tokenizer drift with wrapper drift — useless.

2. Because of (1), `llm_tokens_atlas/count_offline.py` wraps in Python first, then
   invokes the tokenometer CLI with `--format text` so tokenometer treats the
   wrapped string as opaque text and does not re-wrap. `count_empirical.py`
   (other agent) does the same and sends the wrapped string directly to the
   provider's countTokens endpoint.

Canonical wrapper rules
=======================
For an input prompt text `P`:

- `plain`     -> `P` (unchanged)
- `markdown`  -> `## prompt\n\n{P}\n`
- `json`      -> `json.dumps({"prompt": P}, ensure_ascii=False)`
  (compact, no whitespace; mirrors `JSON.stringify({prompt: P})` in JS)
- `yaml`      -> `prompt: |-\n  {P indented by 2 spaces, newline-preserved}\n`
  (block scalar — keeps newlines explicit, no quoting ambiguity)
- `xml`       -> `<prompt>{xml-escaped P}</prompt>`

Replicating these in another language (e.g. JS for an in-browser tool) is
deliberately trivial: each rule is a one-liner. Tests pin the exact byte
output of each wrapper so cross-implementation drift is detectable.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Final

Format = str  # one of: "plain" | "markdown" | "json" | "xml" | "yaml"

ALL_FORMATS: Final[tuple[str, ...]] = ("plain", "markdown", "json", "xml", "yaml")

# Atlas uses `plain`; tokenometer's internal name is `text`. They are
# synonymous when used here — the wrapper accepts either.
_PLAIN_ALIASES: Final[frozenset[str]] = frozenset({"plain", "text"})


def _xml_escape(raw: str) -> str:
    """Escape the five XML special characters in well-defined order.

    Order matters: `&` must be first (otherwise `&amp;` becomes `&amp;amp;`),
    then `<`, `>`, `"`, `'`. We do not preserve XML processing instructions
    or CDATA — inputs are treated as flat text.
    """
    return (
        raw.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _indent_block(text: str, prefix: str = "  ") -> str:
    """Indent every line of `text` with `prefix`. Empty lines get only the prefix.

    This matches the YAML block-scalar convention used by `wrap_as_yaml`.
    """
    return "\n".join(f"{prefix}{line}" for line in text.split("\n"))


def wrap_as_plain(text: str) -> str:
    """Return the input unchanged."""
    return text


def wrap_as_markdown(text: str) -> str:
    """Wrap as a single-section markdown document.

    Format: `## prompt\\n\\n{text}\\n`. Adds a trailing newline so a downstream
    concatenation never accidentally joins two prompts on the same line.
    """
    return f"## prompt\n\n{text}\n"


def wrap_as_json(text: str) -> str:
    """Wrap as a compact JSON object: `{"prompt": text}`.

    `ensure_ascii=False` keeps Unicode codepoints intact (matches JS
    `JSON.stringify` default behavior). `separators=(",", ":")` produces the
    canonical compact form with no surplus whitespace, which is what a
    cost-sensitive caller would use over the wire.
    """
    return json.dumps({"prompt": text}, ensure_ascii=False, separators=(",", ":"))


def wrap_as_yaml(text: str) -> str:
    """Wrap as a YAML block-scalar field: `prompt: |-\\n  {indented text}\\n`.

    Uses the `|-` (literal, strip-final-newline) block scalar so the wrapped
    string is unambiguous regardless of internal newlines or quoting. The
    chomping indicator `-` strips the trailing newline of the scalar; we
    add exactly one outer newline at end so the document is well-formed.

    For an empty input string we emit `prompt: |-\\n  \\n` so the structure
    is uniform across all inputs.
    """
    indented = _indent_block(text, "  ")
    return f"prompt: |-\n{indented}\n"


def wrap_as_xml(text: str) -> str:
    """Wrap as a single-element XML document: `<prompt>{escaped text}</prompt>`.

    No XML declaration; we treat the wrapper as a string-level fragment to
    keep tokenization comparable across providers (declarations would just
    add a constant fixed token count and bias all 5 providers identically).
    """
    return f"<prompt>{_xml_escape(text)}</prompt>"


_WRAPPERS: Final[dict[str, Callable[[str], str]]] = {
    "plain": wrap_as_plain,
    "markdown": wrap_as_markdown,
    "json": wrap_as_json,
    "yaml": wrap_as_yaml,
    "xml": wrap_as_xml,
}


def wrap(text: str, fmt: Format) -> str:
    """Dispatch by format name. Accepts `text` as a synonym for `plain`."""
    if fmt in _PLAIN_ALIASES:
        return wrap_as_plain(text)
    try:
        return _WRAPPERS[fmt](text)
    except KeyError as e:
        raise ValueError(
            f"Unknown format {fmt!r}; expected one of {ALL_FORMATS!r}"
        ) from e


def is_format(value: str) -> bool:
    """Predicate for valid format names. Accepts the `text` alias of `plain`."""
    return value in ALL_FORMATS or value in _PLAIN_ALIASES
