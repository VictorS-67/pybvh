# Tutorials

Interactive Jupyter notebooks with detailed walkthroughs, progressing from basics to advanced workflows.

## Available tutorials

1. **[Introduction to pybvh](https://github.com/VictorS-67/pybvh/blob/main/tutorials/1.Introduction.ipynb)** — the BVH file format, reading/writing, the `Bvh` object, basic inspection
2. **[Spatial coordinates and skeleton operations](https://github.com/VictorS-67/pybvh/blob/main/tutorials/2.Spatial_coordinates.ipynb)** — forward kinematics, centering modes, skeleton operations (`retarget`, `scale`, `extract_joints`)
3. **[Rotations](https://github.com/VictorS-67/pybvh/blob/main/tutorials/3.Rotations.ipynb)** — Euler angles, rotation matrices, quaternions, 6D representation, axis-angle, Euler order changes, gimbal lock and discontinuity illustrations
4. **[Visualization with bvhplot](https://github.com/VictorS-67/pybvh/blob/main/tutorials/4.Visualization.ipynb)** — static snapshots, video export, interactive playback, side-by-side comparison, camera control, `follow=True` tracking, backend options
5. **[Transforms and augmentation](https://github.com/VictorS-67/pybvh/blob/main/tutorials/5.Transforms.ipynb)** — `mirror`, `rotate_vertical`, `translate_root`, `add_noise`, `perturb_speed`, `drop_frames`, composing transforms, reproducibility
6. **[Motion features and analysis](https://github.com/VictorS-67/pybvh/blob/main/tutorials/6.Features.ipynb)** — joint velocities and accelerations, angular velocities, root-relative positions, root trajectory, foot contacts, `to_feature_array()`
7. **[Batch processing](https://github.com/VictorS-67/pybvh/blob/main/tutorials/7.Batch_processing.ipynb)** — `read_bvh_directory`, `harmonize`, `batch_to_numpy`, save/load pattern
8. **[Motion descriptors](https://github.com/VictorS-67/pybvh/blob/main/tutorials/8.Motion_descriptors.ipynb)** — geometry (`curvature`, `bounding_box`, `center_of_mass`), dynamics (`node_jerk`, `smoothness`, `kinetic_energy`, gait), and SE(3) (`relative_transform`, `se3_log`, `rotation_geodesic_distance`), with closed-form sanity checks

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
