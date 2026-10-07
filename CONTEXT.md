# CONTEXT.md — pybvh

The architecture: which module owns what, and why each boundary sits where it does. What a module computes is in its docstrings and the API reference (`docs/api/`); the terms are in `GLOSSARY.md`; the rules a change is held to are in `CODING_STANDARDS.md`; the project's scope, principles and ecosystem are in `CHARTER.md`; a decision whose argument must be kept has its record in `docs/adr/`.

## 1. The layers

A `.bvh` file declares a skeleton and, per frame, the root's position and every joint's Euler angles (`docs/guide/core-concepts.md`). pybvh reads it into a `Bvh`, poses it by forward kinematics, and derives everything else from the angles and the positions. The modules stack in that order:

```
bvhplot                          pictures, through a Scene (section 3)
batch                            many clips
features    dataframe            a clip as one array · a DataFrame as a clip
analysis    io    transforms     descriptors · the file · a clip in, a clip out
bvh                              the clip: skeleton and motion
spatial_coord                    forward kinematics
node_tree   tools   geometry     the tree · axes and orientation · position kernels
bvhnode   rotations   signal   _warnings
```

Every run-time import at module level points down this picture: a layer depends on nothing above it. Imports for type annotations only, under `TYPE_CHECKING`, are exempt: `tools` and `spatial_coord` name `Bvh` that way. `rotations`, `signal`, `geometry` and `FkTopology` know no clip: they take arrays, which lets a consumer pose and measure motion with no `Bvh` in hand, a data loader at train time for instance. Two kinds of import point up, and each sits inside the function that needs it, never at module level: a `Bvh` method that wraps another module's function, and the one function in `spatial_coord` that tells a `Bvh` from the other skeleton forms.

## 2. The module map

`docs/api/index.md` ("Modules at a glance") says what each public module offers a user. This map adds the boundaries, and the modules that page does not list.

| Module | Owns | Does not |
|---|---|---|
| `io` | the `.bvh` text: parsing and writing, and the conversion from and to the degrees a file stores | build a node tree itself: `node_tree` does |
| `dataframe` | a `Bvh` built from a pandas DataFrame, the inverse of `Bvh.to_df_dict` | import pandas at run time |
| `bvhnode` | the node classes, each node's own fields and checks | know the tree as a whole, or any motion |
| `node_tree` | the tree as a whole: the one builder, and the check `Bvh` runs on a finished tree | import anything but `bvhnode` |
| `bvh` | one clip's state, validated and cached; the skeleton and frame edits; its angles in the other rotation representations | compute a descriptor, a transform or a picture, or read or write a file: those methods wrap their module's function |
| `rotations` | every rotation representation and conversion, interpolation, SE(3), and the one Euler-to-matrix conversion | know a node or a clip |
| `spatial_coord` | forward kinematics, and `FkTopology`, the arrays it reads | keep rotation math of its own |
| `tools` | signed axes, the orientation a skeleton implies (world up, rest up and forward, L/R pairs, facing), and the input validators the package shares | import a `Bvh` at run time |
| `_warnings` | where a warning points: the line of the user's code | |
| `signal` | array-pure signal kernels | know a clip |
| `geometry` | array-pure position descriptors | know a clip, or differentiate other than through `signal` |
| `analysis` | the descriptors of a clip, as functions taking the `Bvh` first, and the array kernels behind the dynamics descriptors | assemble an export array |
| `features` | the flat feature array and its column layout | compute a descriptor: it calls `analysis` |
| `transforms` | spatial transforms and augmentation, on a clip and on raw arrays, and the reorientations | |
| `batch` | many clips: loading a directory, harmonizing, stacking into one array | own a per-clip layout: it calls `features` |

bvhplot (`pybvh/bvhplot/`) splits at its Scene: the router and `_from_bvh` read a `Bvh`, and no module behind them does. "The core" below means the package modules that know a clip or its nodes.

| Module | Owns | Does not |
|---|---|---|
| `__init__` (the router) | the public functions; preparing the user's `Bvh` (timing, resampling, world-up checks, rest pose); choosing a backend; arranging a comparison | draw |
| `_from_bvh` | turning a `Bvh` into a Scene: views, bones, camera presets, bone chains | get imported by any module but the router |
| `_scene` | the Scene and its views, checked read-only data, and the operations that change a Scene | import a plotting library or the core |
| `_viewport` | the geometry of one 3D picture: framing, floor, camera, schedule, projection | import a plotting library, or take more than array kernels from the core |
| `_style` | `Style`, its presets and palettes, the ghost and trace conventions | import a plotting library |
| `_colors` | the color resolution that needs matplotlib's color parser | |
| `_playback` | the viewer's playback state machine | import a rendering library |
| `_vedo_capsules` | the capsule geometry both vedo backends draw | |
| `_matplotlib`, `_opencv`, `_k3d`, `_vedo`, `_vedo_offscreen` | drawing a Scene with a Style in one toolkit | compute a box, a floor or a camera of their own for a 3D picture (one exception: the floor of matplotlib's overlay `sequence`), or read the core; the 2D `trajectory` plot has no viewport, and `_matplotlib` frames it |

## 3. Why the boundaries sit where they do

- **A `Bvh` method computes nothing another module owns.** A descriptor, a transform or a picture is computed by its module, and the method only adapts the clip to it: it hands the `Bvh` itself to a function of `analysis`, `features`, `transforms`, `io` or bvhplot, or arrays read from it (positions, a speed, the frame time) to an array kernel of `geometry` or `analysis`. The computation has one home, documented there, and the `Bvh` stays the discoverable surface.
- **One implementation per concern.** A node tree is built only by `node_tree` (the parser, `df_to_bvh` and `extract_joints` all go through it), so the checks are made in one place. The Euler-to-matrix conversion is only in `rotations`, and the orientation a skeleton implies is inferred only in `tools`, which `transforms`, `analysis` and bvhplot's follow camera call rather than keep their own. Skeleton topology has one derivation, `FkTopology.from_nodes`: forward kinematics runs on its result, `Bvh.fk_topology` returns it, and the edges and the bones bvhplot draws are views of that, so they cannot disagree with the geometry forward kinematics produces.
- **Units change at the boundary.** The angles a `Bvh` holds are radians; the conversion from and to the degrees of a file or a DataFrame happens where those are read or written (`io`, `dataframe`, `Bvh.to_df_dict`).
- **Descriptors, assembly and datasets are three modules.** `analysis` computes descriptors, `features` packs one clip's into an array, `batch` stacks clips and delegates each clip to `features`, so the column layout has one owner whether one clip or a dataset is exported.
- **The scene ground and the contact reference are two quantities**, both in `analysis` and never substituted for one another: ADR 0002.
- **bvhplot draws from a Scene, never from a `Bvh`.** Only the router and `_from_bvh` read a `Bvh`; `_viewport` computes each 3D picture's geometry from the Scene's views once (the 2D `trajectory` plot, matplotlib only, is framed by `_matplotlib`); a backend translates Scene, Style and viewport into its toolkit's calls. So every backend computes floor, framing and camera by the same rules: once per panel where it draws each skeleton in its own panel (matplotlib, OpenCV), once over all the views where it draws them in one scene (k3d, vedo), with one exception, the overlay `sequence`, which re-centers its sampled poses and takes its floor from their lowest point. Each backend can be tested from a Scene built from arrays (`tests/synthetic_scene.py`). The import guards in `tests/test_scene.py` pin the boundary and state what they cannot see. bvhplot's scope against pybvh-blender is `pybvh/bvhplot/CHARTER.md`; the vedo shadows are ADR 0001.

## 4. Where the rest lives

- The BVH format: `docs/guide/core-concepts.md`. The arrays, the index spaces and the node table: `GLOSSARY.md`. The forward-kinematics recurrence: `frames_to_node_positions`.
- The bundled sample clips and what each is for: `bvh_data/README.md`. A quick tour of the API: `README.md` and `docs/api/index.md`.
- The tests: what a test pins and how it asserts is in `CODING_STANDARDS.md` ("Tests"); clips and Scenes built from arrays with known properties come from `tests/synthetic_bvh.py` and `tests/synthetic_scene.py`; the frozen references, and when they may be regenerated, are in `tests/fixtures/README.md`. An invariant that holds across the package is pinned by one guard test, and every test file's docstring says what it covers.
- The docs site, the gallery and the notebooks: `docs/tutorials.md` ("Editing the tutorials").
- Dependencies, versions and tool configuration: `pyproject.toml`; the checks a pull request runs: `.github/workflows/`; how a change reaches `main`: `CONTRIBUTING.md`.
- What changed between releases, old names included: `CHANGELOG.md` and `pybvh/API_RENAME.md`; how an entry is written: `CONTRIBUTING.md` ("The CHANGELOG").
