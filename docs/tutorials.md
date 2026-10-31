# Tutorials

These interactive Jupyter notebooks give detailed walkthroughs. They progress from the basics to advanced workflows.

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

Each tutorial is a [Jupytext](https://jupytext.readthedocs.io/)-paired pair of files:

- a `.ipynb` file: the canonical rendered artifact, with outputs and plots.
- a `.py` file in the [Percent format](https://jupytext.readthedocs.io/en/latest/formats-scripts.html#the-percent-format): the plain-text source. It is git-friendly. You can review it as Python.

Both files are committed. Both files mirror every cell: code, markdown, and cell-level tags such as `skip-execution` or `slow-on-pr`.

### How to edit

Install the `dev` dependency group. It includes `jupytext` and `nbmake`. It needs pip 25.1 or later:

```bash
pip install -e ".[all-viz,pandas]" --group dev
```

Then edit either side:

- **In Jupyter Lab** — edit the `.ipynb` as usual. On save, Jupytext rewrites the paired `.py` automatically.
- **In VS Code** — open the `.py`. The Jupyter extension recognizes Percent-format cells. It lets you run them with output inline. A save syncs back to the `.ipynb`.
- **In any text editor** — edit the `.py`, then run:
  ```bash
  jupytext --sync tutorials/*.ipynb
  ```
  Jupytext picks the newer file by mtime. It updates the other side. Unchanged cells keep their outputs. Modified cells lose their outputs. To regenerate them, re-run the notebook.

CI checks that the code cells are formatted. `ruff format --check .` reads each `.ipynb`. It skips the `.py` twin, so that every cell is checked once.

Do not let ruff write a notebook itself, for two reasons:

- it would not update the twin.
- its JSON writer rounds some floats of the saved widget state in their last digit.

Once the pair is in sync, let ruff format a copy. Then restore only the sources, through the twin:

```bash
for nb in tutorials/*.ipynb gallery/feature_gallery.ipynb; do
  ruff format --stdin-filename "$nb" < "$nb" > formatted.ipynb
  jupytext --to py:percent -o "${nb%.ipynb}.py" formatted.ipynb
  jupytext --to ipynb --update "${nb%.ipynb}.py"
done
rm formatted.ipynb
```

### Re-executing a tutorial

The `.ipynb` is the artifact that GitHub renders, straight from its committed outputs. Nothing re-executes it at view time. So a notebook committed without its figures teaches nothing. After you change code cells, regenerate the outputs:

```bash
jupyter nbconvert --to notebook --execute --inplace tutorials/3.Rotations.ipynb
```

Run it from the repository root. Cells tagged `skip-execution` are honoured automatically.

Each plotting tutorial runs `%matplotlib inline` in its setup cell. The `.py` has it as `# %matplotlib inline`. Jupytext uncomments it into the notebook. **Leave it there.** Matplotlib picks its backend from the `MPLBACKEND` environment variable. CI sets `MPLBACKEND=Agg`, a non-interactive backend, for the headless runner. If that variable names a non-interactive backend, executing the notebook drops every figure and replaces it with a `FigureCanvasAgg is non-interactive` warning on stderr. The magic pins the inline backend regardless of the environment. So the same command produces the same figures on any machine.

`tests/test_tutorial_notebooks.py` enforces all of this:

- the jupytext pair must match.
- execution counts must be sequential 1..N.
- the magic must be present in every tutorial that imports pyplot.
- every `plt.show()` cell must carry a figure.
- no backend warning or traceback may reach the committed outputs.

### Animated clips: markdown images, never cell outputs

Static figures reach GitHub as `image/png` cell outputs. That route does not exist for animations: GitHub's notebook renderer **silently drops `image/gif` outputs**. The reader sees `<IPython.core.display.Image object>` where the clip should play. The renderer does not resolve *relative* image paths in markdown cells either. There the reader sees only the alt text. The one form that the renderer displays is a markdown image with an absolute URL. That form works because this repo is public:

```markdown
![the clip](https://raw.githubusercontent.com/VictorS-67/pybvh/main/tutorials/assets/my_clip.gif)
```

The URL pins the repository and the `main` branch. So a reader of a fork or of a feature branch sees `main`'s clip. That is accepted. And a clip that is renamed or moved needs its URL updated.

So for any animated clip:

1. Write the GIF to a committed location: `tutorials/assets/` for tutorials, `gallery/` for the gallery. The rendering cell just returns the path.
2. Commit the file.
3. Display it from a markdown cell, as above.

Never return `IPython.display.Image` from a cell, for two reasons. It is invisible on GitHub. And it embeds the GIF a second time, as base64, inside the `.ipynb`.

`tests/test_tutorial_notebooks.py` and `tests/test_gallery_notebook.py` enforce both halves:

- no `image/gif` cell outputs anywhere.
- every markdown image uses an absolute `raw.githubusercontent.com` URL that resolves to a committed, non-gitignored file.

### Cell-level execution control on CI

The tutorial CI ([.github/workflows/tutorials.yml](https://github.com/VictorS-67/pybvh/blob/main/.github/workflows/tutorials.yml)) executes every tutorial under [nbmake](https://github.com/treebeardtech/nbmake) with two tag conventions:

- `skip-execution` — the cell is always skipped. It is used for interactive `bvh.play()` calls that open a window or widget.
- `slow-on-pr` — the cell is skipped on pull-request builds. It is executed on pushes to `main` / `dev` and on manual dispatch. It is used for `bvh.render(...)` calls that produce videos.

To add a tag in the `.py`, write it into the cell header:

```python
# %% tags=["slow-on-pr"]
bvh.render("demo.mp4")
```

Then `jupytext --sync` propagates the tag into the `.ipynb` metadata.

### Keeping the pair in sync

If you ever suspect that the two files have drifted, run:

```bash
jupytext --sync tutorials/*.ipynb
```

This command is also safe to run as a pre-commit step. A future commit may add a pre-commit hook that enforces sync automatically.

### The Feature Gallery page

The [Gallery](gallery/index.md) docs page is **generated** from `gallery/feature_gallery.ipynb`. That notebook is a Jupytext pair, like the tutorials. It is executed and committed with outputs. CI regenerates the page in every docs build: the strict build that each pull request runs, and the deploy on a push to `main`. `docs/gallery/` is gitignored, so never edit it by hand. The exporter rewrites each clip's absolute `raw.githubusercontent.com` URL to a local copy of the committed GIF. So the docs site serves its own clips. To preview locally:

```bash
python scripts/export_gallery.py
mkdocs serve
```

After you edit gallery code cells, re-execute the notebook before you export, so that the committed outputs stay in sync with the source. The command is `jupyter nbconvert --to notebook --execute --inplace gallery/feature_gallery.ipynb`. CI enforces this:

- The gallery notebook executes under nbmake in `tutorials.yml`. Its GIF cells are tagged `slow-on-pr`, like the tutorials.
- `tests/test_gallery_notebook.py` fails if the jupytext pair drifts, if the committed outputs are stale (non-sequential execution counts, error/stderr outputs), or if the figures have gone missing.

The gallery's setup cell pins `%matplotlib inline` for the same reason the tutorials do. It matters more here, because no gallery cell calls `plt.show()`. So every figure arrives through the inline backend's end-of-cell flush of open figures. Under a different `MPLBACKEND`, the figures vanish without even a warning: sequential execution counts, clean stderr, and an empty docs page.

A handful of figures are also embedded inline in the guide pages, through stable names (`docs/gallery/img/centered-modes.png`, …). The names are declared in `STABLE_FIGURES` inside [`scripts/export_gallery.py`](https://github.com/VictorS-67/pybvh/blob/main/scripts/export_gallery.py). The exporter fails loudly if a gallery refactor breaks one of those matches.

The animation at the top of the README and of the docs home, `docs/assets/hand-trajectory.gif`, is not a gallery figure. Nothing regenerates it. It draws `bvhplot`'s default look. So when that look changes, rerun `python scripts/make_hero_gif.py` by hand and commit the new file.
