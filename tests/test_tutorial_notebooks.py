"""Freshness guard for the committed tutorial notebooks.

GitHub renders `tutorials/*.ipynb` from the committed outputs, and that rendering is how most readers meet the library — nothing re-executes them at view time. A notebook whose figures are missing therefore teaches nothing, and does so silently: the code cells still look right, only the pictures are gone.

That is not hypothetical. Executing the notebooks under `MPLBACKEND=Agg` (the value `tutorials.yml` sets for the headless runner) replaced all 41 figures with a `FigureCanvasAgg is non-interactive` warning on stderr, and the pair stayed perfectly in sync while doing it. The fix is the `%matplotlib inline` magic in each plotting tutorial's setup cell, which pins the inline backend regardless of the environment; these tests keep both the fix and its effect from eroding:

- the magic disappearing from a setup cell (`test_plotting_notebooks_pin_the_inline_backend`),
- a `plt.show()` cell committed without its figure (`test_every_plot_cell_has_a_figure`), which catches the same breakage arriving by any other route,
- rendering warnings or errors reaching the published page,
- outputs not regenerated after an edit (non-sequential execution counts — the state a partial re-run leaves behind).

CI executes the notebooks in `tutorials.yml` (nbmake), which catches cells that no longer *run*; these checks catch cells that were never re-run, or that ran without producing what the reader is supposed to see.

Cells tagged `skip-execution` are excluded throughout: they are never executed, so they carry stale execution counts and no outputs by design.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
TUTORIALS = sorted((REPO / "tutorials").glob("*.ipynb"))

# Emitted by pyplot when the active backend cannot display a figure — the
# signature of a notebook executed with a non-interactive backend forced on.
BACKEND_WARNING = "is non-interactive, and thus cannot be shown"


def _id(path):
    return path.name


def _load(path):
    return json.loads(path.read_text())


def _code_cells(nb_dict):
    """(index, cell) for code cells that are actually executed."""
    return [(i, c) for i, c in enumerate(nb_dict["cells"])
            if c["cell_type"] == "code"
            and "skip-execution" not in c.get("metadata", {}).get("tags", [])]


def _source(cell):
    source = cell["source"]
    return "".join(source) if isinstance(source, list) else source


def _has_image(cell):
    return any(key.startswith("image/")
               for out in cell.get("outputs", [])
               for key in out.get("data", {}))


def _normalized_cells(nb_dict):
    """(cell_type, normalized source) pairs — whitespace-insensitive."""
    out = []
    for cell in nb_dict["cells"]:
        source = _source(cell)
        lines = [ln.rstrip() for ln in source.splitlines()]
        out.append((cell["cell_type"], "\n".join(lines).strip()))
    return out


def test_tutorials_are_discovered():
    assert TUTORIALS, "no tutorial notebooks found — the glob is wrong"


@pytest.mark.parametrize("ipynb", TUTORIALS, ids=_id)
def test_jupytext_pair_in_sync(ipynb):
    jupytext = pytest.importorskip("jupytext")
    py = ipynb.with_suffix(".py")
    assert py.exists(), f"{ipynb.name} has no paired .py"
    py_cells = _normalized_cells(jupytext.read(py))
    nb_cells = _normalized_cells(_load(ipynb))
    assert len(py_cells) == len(nb_cells), (
        f"cell count differs: {len(py_cells)} in {py.name} vs "
        f"{len(nb_cells)} in {ipynb.name} — run `jupytext --sync "
        f"tutorials/*.ipynb`")
    for i, (pc, nc) in enumerate(zip(py_cells, nb_cells)):
        assert pc == nc, (
            f"cell {i} differs between {py.name} and {ipynb.name} — run "
            f"`jupytext --sync tutorials/*.ipynb`")


@pytest.mark.parametrize("ipynb", TUTORIALS, ids=_id)
def test_plotting_notebooks_pin_the_inline_backend(ipynb):
    nb = _load(ipynb)
    sources = [_source(c) for _, c in _code_cells(nb)]
    if not any("import matplotlib.pyplot" in s for s in sources):
        pytest.skip(f"{ipynb.name} does not plot")
    assert any("%matplotlib inline" in s for s in sources), (
        f"{ipynb.name} imports pyplot but never runs `%matplotlib inline`. "
        f"Without it, executing the notebook in an environment that sets "
        f"MPLBACKEND (CI sets Agg) silently drops every figure. Add the "
        f"magic to the setup cell of the paired .py as `# %matplotlib "
        f"inline` — jupytext uncomments it into the notebook.")


@pytest.mark.parametrize("ipynb", TUTORIALS, ids=_id)
def test_every_plot_cell_has_a_figure(ipynb):
    nb = _load(ipynb)
    missing = [i for i, cell in _code_cells(nb)
               if "plt.show()" in _source(cell) and not _has_image(cell)]
    assert not missing, (
        f"{ipynb.name} cells {missing} call plt.show() but committed no "
        f"figure — readers on GitHub see the code and no plot. Re-execute "
        f"with the inline backend: `jupyter nbconvert --to notebook "
        f"--execute --inplace tutorials/{ipynb.name}`")


@pytest.mark.parametrize("ipynb", TUTORIALS, ids=_id)
def test_no_rendering_warnings_or_errors(ipynb):
    """Backend warnings and tracebacks — not *all* stderr.

    Some tutorials deliberately trigger a pybvh ``UserWarning`` (the rest-pose / animation world-up disagreement on ``bvh_test3.bvh``) and show it to the reader on purpose, so a blanket stderr ban like the gallery's would fight the teaching material here.
    """
    nb = _load(ipynb)
    problems = []
    for i, cell in enumerate(nb["cells"]):
        for out in cell.get("outputs", []):
            if out.get("output_type") == "error":
                problems.append(f"cell {i}: error output ({out.get('ename')})")
            if out.get("output_type") == "stream" and out.get("name") == "stderr":
                text = "".join(out.get("text", []))
                if BACKEND_WARNING in text:
                    problems.append(
                        f"cell {i}: matplotlib backend warning — the notebook "
                        f"was executed with a non-interactive backend and its "
                        f"figures were dropped")
    assert not problems, f"{ipynb.name}: " + "; ".join(problems)


@pytest.mark.parametrize("ipynb", TUTORIALS, ids=_id)
def test_notebook_was_fully_executed_in_order(ipynb):
    nb = _load(ipynb)
    counts = [c.get("execution_count") for _, c in _code_cells(nb)]
    expected = list(range(1, len(counts) + 1))
    assert counts == expected, (
        f"{ipynb.name}: execution counts are not sequential 1..N — the "
        f"committed outputs are stale (a cell was edited without a full "
        f"re-run). Re-execute: `jupyter nbconvert --to notebook --execute "
        f"--inplace tutorials/{ipynb.name}`")
