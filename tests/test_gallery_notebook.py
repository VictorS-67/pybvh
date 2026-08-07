"""Freshness guard for the committed feature-gallery notebook.

The gallery's docs page is generated from the *committed outputs* of
``gallery/feature_gallery.ipynb`` (no execution at docs-build time), so a stale
notebook publishes stale figures. These tests make the two failure modes
loud:

- the jupytext pair drifting (``.py`` edited, ``.ipynb`` not re-synced),
- outputs not regenerated after an edit (non-sequential / missing
  execution counts, exactly the state a partial re-run leaves behind),
- the figures going missing wholesale (see ``test_figures_are_committed``),

plus basic hygiene: no error outputs and no stderr in what gets published.
CI executes the notebook itself in ``tutorials.yml`` (nbmake), which
catches cells that no longer run; these checks catch cells that were
never re-run.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
IPYNB = REPO / "gallery" / "feature_gallery.ipynb"
PY = REPO / "gallery" / "feature_gallery.py"

RAW_PREFIX = "https://raw.githubusercontent.com/VictorS-67/pybvh/main/"


def _servable_from_github(repo_rel: str) -> bool:
    """The file exists and is not gitignored — ``exists()`` alone would accept a local byproduct that raw.githubusercontent.com will 404 on."""
    if not (REPO / repo_rel).exists():
        return False
    if not (REPO / ".git").exists():
        return True                     # sdist/tarball: existence is all we have
    ignored = subprocess.run(
        ["git", "-C", str(REPO), "check-ignore", "-q", repo_rel],
        capture_output=True)
    return ignored.returncode != 0


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


def test_notebook_pins_the_inline_backend():
    """The gallery's figures are never explicitly shown or saved.

    Every one of them reaches the committed outputs through the inline backend's end-of-cell flush of open figures. That flush is not installed when ``MPLBACKEND`` names another backend — CI sets ``Agg`` for the headless runner — and the loss is *silent*: no warning, no error, execution counts still sequential, so every other check in this file passes on a gallery with no pictures in it. ``%matplotlib inline`` in the setup cell pins the backend regardless of the environment.
    """
    nb = json.loads(IPYNB.read_text())
    sources = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    assert any("%matplotlib inline" in s for s in sources), (
        "feature_gallery.ipynb no longer runs `%matplotlib inline`. Without "
        "it, executing under MPLBACKEND=Agg drops every figure without "
        "warning and the docs gallery page publishes empty. Restore the "
        "magic in feature_gallery.py's setup cell as `# %matplotlib inline` "
        "— jupytext uncomments it into the notebook.")


def test_figures_are_committed():
    """A wipeout detector, not an inventory.

    The floor is deliberately far below the real figure count so that adding or retiring a capability never touches this test — it only fires when the notebook has lost its figures en masse, which is what a backend misconfiguration does.
    """
    nb = json.loads(IPYNB.read_text())
    figures = sum(1 for c in nb["cells"] for out in c.get("outputs", [])
                  for key in out.get("data", {}) if key.startswith("image/"))
    assert figures >= 40, (
        f"only {figures} figures in the committed gallery — the docs page is "
        f"generated from these outputs, so it would publish near-empty. "
        f"Re-execute: `jupyter nbconvert --to notebook --execute --inplace "
        f"gallery/feature_gallery.ipynb`")


def test_clips_are_linked_files_not_embedded_outputs():
    """The animated clips must be markdown images, never ``image/gif`` outputs.

    GitHub's notebook renderer displays ``image/png`` outputs but silently drops ``image/gif`` ones (the reader sees the ``text/plain`` fallback, ``<IPython.core.display.Image object>``), and it does not resolve *relative* paths in markdown cells either. The one form it renders is a markdown image with an absolute ``raw.githubusercontent.com`` URL — which only works because this repo is public. Note ``test_figures_are_committed`` cannot catch this: its floor counts PNG outputs, and passes happily while every clip is invisible.

    Each URL is also resolved against the working tree — and checked against .gitignore, since raw.githubusercontent.com serves only committed files — so a renamed, ignored, or uncommitted GIF fails here instead of as a broken image on github.com.
    """
    nb = json.loads(IPYNB.read_text())
    gif_outputs = [i for i, c in enumerate(nb["cells"])
                   for out in c.get("outputs", [])
                   if "image/gif" in out.get("data", {})]
    assert not gif_outputs, (
        f"cells {gif_outputs} embed image/gif outputs, which GitHub's "
        f"notebook renderer silently drops. Return the path from the "
        f"gallery_plots helper and display the committed GIF from a "
        f"markdown cell with an absolute raw.githubusercontent.com URL.")

    markdown = "\n".join("".join(c["source"]) for c in nb["cells"]
                         if c["cell_type"] == "markdown")
    srcs = re.findall(r"!\[[^\]]*\]\((\S+?)\)", markdown)
    gif_srcs = [s for s in srcs if s.endswith(".gif")]
    for src in gif_srcs:
        assert src.startswith(RAW_PREFIX), (
            f"clip image src {src!r} is not an absolute {RAW_PREFIX} URL — "
            f"GitHub's notebook renderer would show only the alt text")
        assert _servable_from_github(src[len(RAW_PREFIX):]), (
            f"clip image src {src!r} does not resolve to a committed, "
            f"non-gitignored file — GitHub would show a broken-image icon")
    linked = {s[len(RAW_PREFIX):] for s in gif_srcs}
    expected = {"gallery/feature_gallery_seq.gif",
                "gallery/feature_gallery_walk.gif",
                "gallery/feature_gallery_hand_traj.gif"}
    assert linked == expected, (
        f"the notebook's markdown clips are {sorted(linked)}, expected "
        f"{sorted(expected)} — a clip lost (or gained) its display cell")


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
