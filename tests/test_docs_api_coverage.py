"""Guard: the curated docs/api member lists stay in sync with the code.

The restructured API pages (docs/api/bvh.md, analysis.md, rotations.md) list
every public member explicitly in per-member ``:::`` blocks. Explicit lists
can rot: a new public method silently never appears in the docs, or a rename
leaves a dead entry behind. These tests assert two-way set equality between
what the markdown documents and what the code actually exposes, and fail
with the exact member names so the fix is obvious.
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

from pybvh.bvh import Bvh
from pybvh import analysis, rotations

DOCS_API = Path(__file__).resolve().parent.parent / "docs" / "api"

_BLOCK_RE = re.compile(r"^::: ([\w.]+)\s*$", re.MULTILINE)
_MEMBER_ITEM_RE = re.compile(r"^\s+- (\w+)\s*$", re.MULTILINE)


def _documented(page: str, prefix: str) -> set[str]:
    """Member names documented on a page under ``prefix``.

    Collects per-member ``::: prefix.<name>`` blocks and, for pages that use
    a single block with a ``members:`` allowlist (e.g. api/index.md style),
    the listed names following a ``::: prefix`` root block.
    """
    text = (DOCS_API / page).read_text()
    names: set[str] = set()
    for match in _BLOCK_RE.finditer(text):
        target = match.group(1)
        if target.startswith(prefix + ".") and "." not in target[len(prefix) + 1:]:
            names.add(target[len(prefix) + 1:])
        elif target == prefix:
            # a root block: pick up an explicit `members:` list if present
            tail = text[match.end():]
            members_at = tail.find("members:")
            if members_at != -1:
                # stop at the first line that is not an indented list item
                block = tail[members_at:].split("\n\n", 1)[0]
                names.update(_MEMBER_ITEM_RE.findall(block))
    return names


def _module_public(mod) -> set[str]:
    return {
        n for n, o in vars(mod).items()
        if not n.startswith("_")
        and (inspect.isfunction(o) or inspect.isclass(o))
        and getattr(o, "__module__", "") == mod.__name__
    }


def _assert_two_way(documented: set[str], actual: set[str], page: str) -> None:
    undocumented = sorted(actual - documented)
    stale = sorted(documented - actual)
    problems = []
    if undocumented:
        problems.append(
            f"public members missing from docs/api/{page}: {undocumented}")
    if stale:
        problems.append(
            f"docs/api/{page} documents members that no longer exist: {stale}")
    assert not problems, "; ".join(problems)


def test_bvh_page_covers_every_public_member():
    documented = _documented("bvh.md", "pybvh.bvh.Bvh")
    actual = {n for n in vars(Bvh) if not n.startswith("_")}
    _assert_two_way(documented, actual, "bvh.md")


@pytest.mark.parametrize("page, mod, prefix", [
    ("analysis.md", analysis, "pybvh.analysis"),
    ("rotations.md", rotations, "pybvh.rotations"),
])
def test_module_page_covers_every_public_member(page, mod, prefix):
    documented = _documented(page, prefix)
    _assert_two_way(documented, _module_public(mod), page)
