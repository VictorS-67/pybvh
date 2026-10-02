"""Guard: a paragraph of markdown prose is written on one line.

The house rule is that markdown prose is never hard-wrapped: the editor
soft-wraps, and a paragraph broken across lines makes every later edit
rewrap the lines around it. The test walks every markdown file git tracks
and names the file and line of each paragraph continued on a second line.

The check is a line heuristic, not a markdown parser. A line is reported
when it is prose and the line above it is prose or a list item, so a
wrapped list item is reported like a wrapped paragraph, and a blockquote
is checked like the text it quotes. These are not prose: everything inside
a fenced code block, a ``$$`` math block, an HTML block (opened as
CommonMark opens one), a table (recognised by its delimiter row), front
matter, and a ``:::`` mkdocstrings directive with its options; and
headings, list items, admonition openers, link reference definitions,
thematic breaks, attr_list block attributes (``{: #id .class }`` alone on
a line), badge rows (a line of linked images only), and the two
label lines of a GLOSSARY.md entry (``**Term**:`` above its definition,
``_Avoid_:`` below it).

Leaving a container follows Python-Markdown, which renders the mkdocs site:
a line indented less than an admonition's body starts a new paragraph, but
a line indented less than a list item's text is a lazy continuation of the
item, rendered as one paragraph with it, and is reported. Code goes in
fenced blocks: an indented code block of two or more lines reads as a
wrapped paragraph. A hard line break (two trailing spaces or a backslash)
is reported as well; separate lines go in a list or separate paragraphs.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import pytest

REPO = Path(__file__).resolve().parent.parent

# A backtick fence's info string cannot contain a backtick (CommonMark), so
# a line opening with ```inline code``` is prose, not a fence.
_FENCE = re.compile(r"\s*(`{3,}(?=[^`]*$)|~{3,})")
_TABLE_DELIMITER = re.compile(r"\s*\|?\s*:?-+:?\s*(?:\|\s*:?-+:?\s*)*\|?\s*$")
_ADMONITION = re.compile(r"\s*(?:!!!|\?\?\?)")
_LIST_ITEM = re.compile(r"\s*(?:[-*+]|\d+[.)])(?:\s|$)")
_QUOTE_MARKERS = re.compile(r"(?:\s*>\s?)*")
_NOT_PROSE = re.compile(
    r"""\s*(?:
        \#{1,6}(?:\s|$)                     # heading, not an issue number
      | \$\$.*\$\$\s*$                      # one-line display math
      | \[[^\]]+\]:\s                       # link reference definition
      | (?:-{3,}|\*{3,}|_{3,}|={3,})\s*$    # thematic break, setext underline
      | \*\*[^*]+\*\*:\s*$                  # GLOSSARY.md term label
      | _Avoid_:                            # GLOSSARY.md avoid line
      | \{:?[^{}]*\}\s*$                    # attr_list block attributes
      | (?:\[?!\[[^\]]*\]\([^)]*\)(?:\]\([^)]*\))?\s*)+$   # badge row
    )""",
    re.VERBOSE,
)

# CommonMark's HTML blocks: type 1 runs to its closing tag, types 2 to 5 to
# their end marker, types 6 and 7 to a blank line.
_HTML_RAW_TAG = re.compile(r"<(pre|script|style|textarea)(?:[\s>]|$)", re.IGNORECASE)
_HTML_END_MARKERS = (("<!--", "-->"), ("<?", "?>"), ("<![CDATA[", "]]>"), ("<!", ">"))
_HTML_TAG = re.compile(r"</?([A-Za-z][A-Za-z0-9-]*)(?:\s|/?>|$)")
_HTML_BLOCK_TAGS = frozenset(
    """address article aside base basefont blockquote body caption center col
    colgroup dd details dialog dir div dl dt fieldset figcaption figure footer
    form frame frameset h1 h2 h3 h4 h5 h6 head header hr html iframe legend li
    link main menu menuitem nav noframes ol optgroup option p param search
    section summary table tbody td tfoot th thead title tr track ul""".split()
)
_HTML_LONE_TAG = re.compile(
    r"""</?[A-Za-z][A-Za-z0-9-]*
    (?:\s+[A-Za-z_:][\w.:-]*(?:\s*=\s*(?:[^\s"'=<>`]+|'[^']*'|"[^"]*"))?)*
    \s*/?>\s*$""",
    re.VERBOSE,
)


def _is_blank(line: str) -> bool:
    return not line.strip()


class _Block(NamedTuple):
    """A non-prose block a line opens.

    ``closer`` tests each following line for the one that ends the block,
    which belongs to it; ``None`` when the block is the opening line alone.
    """

    closer: Callable[[str], bool] | None


def _html_block(stripped: str, interrupts_paragraph: bool) -> _Block | None:
    raw_tag = _HTML_RAW_TAG.match(stripped)
    if raw_tag:
        end = f"</{raw_tag.group(1).lower()}>"
        if end in stripped.lower():
            return _Block(None)
        return _Block(lambda closing: end in closing.lower())
    for start, end in _HTML_END_MARKERS:
        if stripped.startswith(start):
            if end in stripped[len(start) :]:
                return _Block(None)
            return _Block(lambda closing, end=end: end in closing)
    tag = _HTML_TAG.match(stripped)
    if tag and tag.group(1).lower() in _HTML_BLOCK_TAGS:
        return _Block(_is_blank)
    # A lone tag of any other name opens a block only outside a paragraph.
    if not interrupts_paragraph and _HTML_LONE_TAG.match(stripped):
        return _Block(_is_blank)
    return None


def _opened_block(
    line: str, next_line: str, number: int, interrupts_paragraph: bool
) -> _Block | None:
    """The non-prose block ``line`` opens, or ``None`` when it opens none."""
    stripped = line.strip()
    if number == 1 and stripped == "---":
        return _Block(lambda closing: closing.strip() == "---")
    fence = _FENCE.match(line)
    if fence:
        marker = fence.group(1)
        return _Block(
            lambda closing: (
                closing.strip().startswith(marker) and not closing.strip().strip(marker[0])
            )
        )
    if stripped == "$$" or (stripped.startswith("$$") and not stripped.endswith("$$")):
        return _Block(lambda closing: closing.rstrip().endswith("$$"))
    if stripped.startswith(":::"):
        return _Block(_is_blank)
    if "|" in next_line and _TABLE_DELIMITER.match(next_line):
        return _Block(_is_blank)
    return _html_block(stripped, interrupts_paragraph)


def _unquote(line: str) -> tuple[str, int]:
    """``line`` without its blockquote markers, and how deep it is quoted."""
    markers = _QUOTE_MARKERS.match(line)
    return line[markers.end() :], markers.group(0).count(">")


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def hard_wrapped_lines(text: str) -> list[int]:
    """The 1-based numbers of the lines that continue the line above them."""
    lines = [_unquote(quoted_line) for quoted_line in text.splitlines()]
    wrapped = []
    previous_can_continue = False
    previous_depth = 0
    admonition_body_indents: list[int] = []
    closer = None
    for index, (line, depth) in enumerate(lines):
        number = index + 1
        if depth != previous_depth:
            previous_can_continue = False
        previous_depth = depth
        if closer is not None:
            if closer(line):
                closer = None
            previous_can_continue = False
            continue
        if _is_blank(line):
            previous_can_continue = False
            continue
        while admonition_body_indents and _indent(line) < admonition_body_indents[-1]:
            admonition_body_indents.pop()
            previous_can_continue = False
        next_line = lines[index + 1][0] if index + 1 < len(lines) else ""
        block = _opened_block(line, next_line, number, previous_can_continue)
        if block is not None:
            closer = block.closer
            previous_can_continue = False
        elif _LIST_ITEM.match(line):
            previous_can_continue = True
        elif _ADMONITION.match(line):
            admonition_body_indents.append(_indent(line) + 4)
            previous_can_continue = False
        elif _NOT_PROSE.match(line):
            previous_can_continue = False
        else:
            if previous_can_continue:
                wrapped.append(number)
            previous_can_continue = True
    return wrapped


def _hard_wraps_in_tracked_markdown(repo: Path) -> list[str]:
    """``path:line`` of each hard wrap in the markdown files git tracks in ``repo``.

    A tracked file deleted from the working tree is skipped: there is no
    text left to check.
    """
    try:
        listing = subprocess.run(
            ["git", "-C", str(repo), "ls-files", "*.md"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout: there is no list of tracked markdown files")
    present = [path for path in listing.stdout.splitlines() if (repo / path).is_file()]
    return [
        f"{path}:{number}"
        for path in present
        for number in hard_wrapped_lines((repo / path).read_text(encoding="utf-8"))
    ]


def test_tracked_markdown_has_no_hard_wrapped_prose():
    wraps = _hard_wraps_in_tracked_markdown(REPO)
    assert not wraps, (
        "Markdown prose is one line per paragraph; join each of these lines "
        "to the line above it:\n" + "\n".join(wraps)
    )


def test_a_tracked_file_deleted_from_the_working_tree_is_skipped(tmp_path):
    (tmp_path / "kept.md").write_text("A paragraph broken\nacross two lines.\n")
    (tmp_path / "deleted.md").write_text("One line.\n")
    try:
        subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
        subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("git is not available to build a repository")
    (tmp_path / "deleted.md").unlink()
    assert _hard_wraps_in_tracked_markdown(tmp_path) == ["kept.md:2"]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("A paragraph broken\nacross two lines.\n", [2]),
        ("- A list item broken\n  across two lines.\n", [2]),
        ("- A list item continued\nwithout an indent.\n", [2]),
        ('!!! note "Title"\n    An admonition body broken\n    across two lines.\n', [3]),
        ("Fixed in a later release,\n#62 tracks it.\n", [2]),
        ("> A blockquote broken\n> across two lines.\n", [2]),
        ("<!-- A comment -->\nA paragraph broken\nacross two lines.\n", [3]),
        ("<kbd>Ctrl</kbd> toggles the labels,\nand Space pauses.\n", [2]),
        ("```literal``` opens a paragraph\nbroken across two lines.\n", [2]),
        ("<pre>\n\nkept\nas is\n</pre>\nA paragraph broken\nacross two lines.\n", [7]),
        ("A paragraph.\n> A quote under it.\n", []),
        ('!!! note "Title"\n    The body.\nA paragraph after the admonition.\n', []),
        ("Name | Value\n---- | -----\na | b\n", []),
        ("A paragraph with attributes.\n{: #usage .lead }\n", []),
        ("A paragraph with attributes.\n{ #usage .lead }\n", []),
        ("A paragraph above a table.\n| A | B |\n|---|---|\n", []),
        ("One paragraph.\n\nAnother paragraph.\n", []),
    ],
)
def test_a_paragraph_continued_on_the_next_line_is_reported(text, expected):
    assert hard_wrapped_lines(text) == expected


TOLERATED = """\
---
title: Front matter
tags: [one, two]
---

# A heading
Text right under a heading.

[![PyPI](https://img.shields.io/pypi/v/pybvh)](https://pypi.org/project/pybvh/)
[![Docs](https://img.shields.io/badge/docs-online-blue)](https://example.org/)

- An item
- Another item
  - A nested item
    1. A nested numbered item

| A | table |
|---|---|
| with | rows |

> A blockquote.
>
> Its second paragraph.

```python
code = "fenced"
more = "code"
```

$$
x = y
$$

<!--
An HTML comment
on two lines.
-->

<details>
<summary>An HTML block</summary>
</details>

::: pybvh.bvh.Bvh
    options:
      members: false

**Term**:
The term's definition.
_Avoid_: another term
"""


def test_the_tolerated_constructs_are_not_reported():
    assert hard_wrapped_lines(TOLERATED) == []
