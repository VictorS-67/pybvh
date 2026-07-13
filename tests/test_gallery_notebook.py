"""Freshness guard for the committed feature-gallery notebook.

The gallery's docs page is generated from the *committed outputs* of
``gallery/feature_gallery.ipynb`` (no execution at docs-build time), so a stale
notebook publishes stale figures. These tests make the two failure modes
loud:

- the jupytext pair drifting (``.py`` edited, ``.ipynb`` not re-synced),
- outputs not regenerated after an edit (non-sequential / missing
  execution counts, exactly the state a partial re-run leaves behind),

plus basic hygiene: no error outputs and no stderr in what gets published.
CI executes the notebook itself in ``tutorials.yml`` (nbmake), which
catches cells that no longer run; these checks catch cells that were
never re-run.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
IPYNB = REPO / "gallery" / "feature_gallery.ipynb"
PY = REPO / "gallery" / "feature_gallery.py"


def _normalized_cells(nb_dict):
    """(cell_type, normalized source) pairs — whitespace-insensitive."""
    out = []
    for cell in nb_dict["cells"]:
        source = "".join(cell["source"]) if isinstance(cell["source"], list) \
            else cell["source"]
        lines = [ln.rstrip() for ln in source.splitlines()]
        out.append((cell["cell_type"], "\n".join(lines).strip()))
    return out


def test_jupytext_pair_in_sync():
    jupytext = pytest.importorskip("jupytext")
    from_py = jupytext.read(PY)
    from_ipynb = json.loads(IPYNB.read_text())
    py_cells = _normalized_cells(from_py)
    nb_cells = _normalized_cells(from_ipynb)
    assert len(py_cells) == len(nb_cells), (
        f"cell count differs: {len(py_cells)} in .py vs {len(nb_cells)} in "
        f".ipynb — run `jupytext --sync gallery/feature_gallery.ipynb`")
    for i, (pc, nc) in enumerate(zip(py_cells, nb_cells)):
        assert pc == nc, (
            f"cell {i} differs between feature_gallery.py and .ipynb — "
            f"run `jupytext --sync gallery/feature_gallery.ipynb`")


def test_notebook_was_fully_executed_in_order():
    nb = json.loads(IPYNB.read_text())
    counts = [c.get("execution_count") for c in nb["cells"]
              if c["cell_type"] == "code"]
    expected = list(range(1, len(counts) + 1))
    assert counts == expected, (
        "execution counts are not sequential 1..N — the committed outputs "
        "are stale (a cell was edited without a full re-run). Re-execute: "
        "`jupyter nbconvert --to notebook --execute --inplace "
        "gallery/feature_gallery.ipynb`")


def test_notebook_outputs_are_clean():
    nb = json.loads(IPYNB.read_text())
    problems = []
    for i, cell in enumerate(nb["cells"]):
        for out in cell.get("outputs", []):
            if out.get("output_type") == "error":
                problems.append(f"cell {i}: error output ({out.get('ename')})")
            if out.get("output_type") == "stream" and out.get("name") == "stderr":
                text = "".join(out.get("text", []))[:80]
                problems.append(f"cell {i}: stderr output ({text!r})")
    assert not problems, (
        "committed notebook outputs would publish errors/warnings on the "
        "docs gallery page: " + "; ".join(problems))
