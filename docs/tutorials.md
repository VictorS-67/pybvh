# Tutorials

Interactive Jupyter notebooks with detailed walkthroughs, progressing from basics to advanced workflows.

## Available tutorials

1. **[Introduction to pybvh](https://github.com/VictorS-67/pybvh/blob/main/tutorials/1.Introduction.ipynb)** — the BVH file format, reading/writing, the `Bvh` object, basic inspection
2. **[Spatial coordinates and skeleton operations](https://github.com/VictorS-67/pybvh/blob/main/tutorials/2.Spatial_coordinates.ipynb)** — forward kinematics, centering modes, skeleton operations (`retarget`, `scale`, `extract_joints`)
3. **[Rotations](https://github.com/VictorS-67/pybvh/blob/main/tutorials/3.Rotations.ipynb)** — Euler angles, rotation matrices, quaternions, SLERP, 6D representation, axis-angle, Euler order changes, gimbal lock and discontinuity illustrations
4. **[Visualization with bvhplot](https://github.com/VictorS-67/pybvh/blob/main/tutorials/4.Visualization.ipynb)** — static snapshots, video export, interactive playback, side-by-side comparison, camera control, `follow=True` tracking, backend options
5. **[Transforms and augmentation](https://github.com/VictorS-67/pybvh/blob/main/tutorials/5.Transforms.ipynb)** — `mirror`, `rotate_vertical`, `translate_root`, `add_rotation_noise`, `add_position_noise`, `perturb_speed`, `drop_frames`, composing transforms, reproducibility
6. **[Motion features and analysis](https://github.com/VictorS-67/pybvh/blob/main/tutorials/6.Features.ipynb)** — joint velocities and accelerations, angular velocities, root-relative positions, root trajectory, foot contacts, `to_feature_array()`
7. **[Batch processing](https://github.com/VictorS-67/pybvh/blob/main/tutorials/7.Batch_processing.ipynb)** — `read_bvh_directory`, `harmonize`, `batch_to_numpy`, save/load pattern
8. **[Motion descriptors](https://github.com/VictorS-67/pybvh/blob/main/tutorials/8.Motion_descriptors.ipynb)** — geometry (`curvature`, `bounding_box`, `center_of_mass`), dynamics (`node_jerk`, `smoothness`, `kinetic_energy`), gait (`gait_parameters` and its `contacts=`-sharing projections), SE(3) (`relative_transform`, `se3_log`, `rotation_geodesic_distance`), and reusing FK output via `coords=` — with closed-form sanity checks

A reader who finishes all eight has a solid working understanding of BVH data and the complete pybvh API.

## Running locally

```bash
pip install "pybvh[pandas]" jupyter
cd tutorials/
jupyter notebook
```

## Editing the tutorials (for contributors)

Each tutorial is a [Jupytext](https://jupytext.readthedocs.io/)-paired pair: a `.ipynb` file (the canonical rendered artifact, with outputs and plots) and a `.py` file in the [Percent format](https://jupytext.readthedocs.io/en/latest/formats-scripts.html#the-percent-format) (the plain-text source, git-friendly, reviewable as Python).

Both files are committed. Every cell — code, markdown, and cell-level tags like `skip-execution` or `slow-on-pr` — is mirrored in both files.

### How to edit

Install the dev extras (includes `jupytext` and `nbmake`):

```bash
pip install -e ".[dev,all-viz,pandas]"
```

Then edit either side:

- **In Jupyter Lab** — edit the `.ipynb` as usual. On save, Jupytext rewrites the paired `.py` automatically.
- **In VS Code** — open the `.py`; the Jupyter extension recognizes Percent-format cells and lets you run them with output inline. Save syncs back to the `.ipynb`.
- **In any text editor** — edit the `.py`, then run:
  ```bash
  jupytext --sync tutorials/*.ipynb
  ```
  Jupytext picks the newer file by mtime and updates the other side. Outputs on unchanged cells are preserved; outputs on modified cells are cleared (re-run the notebook to regenerate).

### Re-executing a tutorial

The `.ipynb` is the artifact GitHub renders, straight from its committed outputs — nothing re-executes it at view time, so a notebook committed without its figures teaches nothing. After changing code cells, regenerate the outputs:

```bash
jupyter nbconvert --to notebook --execute --inplace tutorials/3.Rotations.ipynb
```

Run it from the repository root; cells tagged `skip-execution` are honoured automatically.

Each plotting tutorial runs `%matplotlib inline` in its setup cell (written as `# %matplotlib inline` in the `.py`, which jupytext uncomments into the notebook). **Leave it there.** Matplotlib picks its backend from the `MPLBACKEND` environment variable, and if that names a non-interactive backend — CI sets `MPLBACKEND=Agg` for the headless runner — then executing the notebook drops every figure and replaces it with a `FigureCanvasAgg is non-interactive` warning on stderr. The magic pins the inline backend regardless of the environment, so the same command produces the same figures on any machine.

`tests/test_tutorial_notebooks.py` enforces all of this: the jupytext pair must match, execution counts must be sequential 1..N, the magic must be present in every tutorial that imports pyplot, every `plt.show()` cell must carry a figure, and no backend warning or traceback may reach the committed outputs.

### Cell-level execution control on CI

The tutorial CI ([.github/workflows/tutorials.yml](https://github.com/VictorS-67/pybvh/blob/main/.github/workflows/tutorials.yml)) executes every tutorial under [nbmake](https://github.com/treebeardtech/nbmake) with two tag conventions:

- `skip-execution` — cell is always skipped (used for interactive `bvh.play()` calls that open a window or widget).
- `slow-on-pr` — cell is skipped on pull-request builds, executed on pushes to `main` / `dev` and on manual dispatch. Used for `bvh.render(...)` calls that produce videos.

Add a tag in the `.py` by writing it into the cell header:

```python
# %% tags=["slow-on-pr"]
bvh.render("demo.mp4")
```

Then `jupytext --sync` propagates the tag into the `.ipynb` metadata.

### Keeping the pair in sync

If you ever suspect the two files have drifted, run:

```bash
jupytext --sync tutorials/*.ipynb
```

This is also safe to run as a pre-commit step. A future commit may add a pre-commit hook to enforce sync automatically.

### The Feature Gallery page

The [Gallery](gallery/index.md) docs page is **generated** from `gallery/feature_gallery.ipynb` (a Jupytext pair like the tutorials, executed and committed with outputs). CI regenerates it on every deploy; `docs/gallery/` is gitignored — never edit it by hand. To preview locally:

```bash
python scripts/export_gallery.py
mkdocs serve
```

After editing gallery code cells, re-execute the notebook (`jupyter nbconvert --to notebook --execute --inplace gallery/feature_gallery.ipynb`) before exporting, so the committed outputs stay in sync with the source. CI enforces this: the gallery notebook executes under nbmake in `tutorials.yml` (GIF cells are tagged `slow-on-pr`, like the tutorials), and `tests/test_gallery_notebook.py` fails if the jupytext pair drifts, the committed outputs are stale (non-sequential execution counts, error/stderr outputs), or the figures have gone missing.

The gallery's setup cell pins `%matplotlib inline` for the same reason the tutorials do, and it matters more here: no gallery cell calls `plt.show()`, so every figure arrives via the inline backend's end-of-cell flush of open figures. Under a different `MPLBACKEND` they vanish without even a warning — sequential execution counts, clean stderr, and an empty docs page.

A handful of figures are also embedded inline in the guide pages via stable names (`docs/gallery/img/centered-modes.png`, …) declared in `STABLE_FIGURES` inside [`scripts/export_gallery.py`](https://github.com/VictorS-67/pybvh/blob/main/scripts/export_gallery.py); the exporter fails loudly if a gallery refactor breaks one of those matches.
