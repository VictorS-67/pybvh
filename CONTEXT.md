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
spatial_coord   _orientation     forward kinematics · what a clip implies about up, left and forward
_contacts                        contact detection on plain arrays
node_tree   tools   geometry     the tree · signed axes and validators · position kernels
bvhnode   rotations   signal   _warnings   _derived
```

Every run-time import at module level points down this picture: a layer depends on nothing above it. Imports for type annotations only, under `TYPE_CHECKING`, are exempt: `spatial_coord` names `Bvh` that way, and `_derived` the report types of `_orientation`. `rotations`, `signal`, `geometry`, `tools`, `_orientation`, `_contacts` and `FkTopology` know no clip: they take arrays, which lets a consumer pose and measure motion with no `Bvh` in hand, a data loader at train time for instance. Two kinds of import point up, and each sits inside the function that needs it, never at module level: a `Bvh` method that wraps another module's function, and the one function in `spatial_coord` that tells a `Bvh` from the other skeleton forms.

## 2. The module map

`docs/api/index.md` ("Modules at a glance") says what each public module offers a user. This map adds the boundaries, and the modules that page does not list.

| Module | Owns | Does not |
|---|---|---|
| `io` | the `.bvh` text: parsing and writing, and the conversion from and to the degrees a file stores | build a node tree itself: `node_tree` does |
| `dataframe` | a `Bvh` built from a pandas DataFrame, the inverse of `Bvh.to_df_dict` | import pandas at run time |
| `bvhnode` | the node classes, each node's own fields and checks | know the tree as a whole, or any motion |
| `node_tree` | the tree as a whole: the one builder, the check `Bvh` runs on a finished tree, and the one comparison of two trees | import anything but `bvhnode` |
| `bvh` | one clip's state, validated, and the computation of its derived state, whose lifecycle `_derived` holds; the skeleton and frame edits, each skeleton edit a new frozen skeleton swapped in; its angles in the other rotation representations | compute a descriptor, a transform, a picture or an orientation fact, or read or write a file: those methods wrap their module's function, and the orientation members hand `_orientation` the clip's data |
| `rotations` | every rotation representation and conversion, interpolation, SE(3), and the one Euler-to-matrix conversion | know a node or a clip |
| `spatial_coord` | forward kinematics, and `FkTopology`, the arrays it reads | keep rotation math of its own |
| `tools` | the vocabulary of signed axes (`Axis`, `parse_axis`, `get_main_direction`, `extract_sign`) and the input validators the package shares | know a clip, or infer an orientation: `_orientation` does |
| `_orientation` | what a clip implies about up, left and forward, from plain data (joint names, the L/R mapping, rest positions, coordinates, world up): the world-up guess, the skeleton's own frame (rest up, rest left, rest forward), the L/R mapping's detection from names and its resolution into pairs, the facing chain, the heading's derivation and the orientation report (ADR 0004) | import a `Bvh`, or hold a value: `bvh` gathers its inputs and fills the derived state |
| `_warnings` | where a warning points: the line of the user's code | |
| `_derived` | a clip's derived state: the values themselves, and the table of which change to the skeleton, the motion or the world up clears which value (ADR 0003) | compute a value: `bvh` computes each one and fills it on read |
| `signal` | array-pure signal kernels | know a clip |
| `geometry` | array-pure position descriptors | know a clip, or differentiate other than through `signal` |
| `_contacts` | contact detection from plain data (the tested joints' positions, their and the root's rest positions, the frame time, the up axis, the settings): the signals, the thresholds, the labels, the confidence and the diagnostics | import a `Bvh`: `analysis.foot_contacts` and `ground_contacts` read the clip and run the checks that name it |
| `analysis` | the descriptors of a clip, as functions taking the `Bvh` first, and the array kernels behind the dynamics descriptors | assemble an export array |
| `features` | the flat feature array and its column layout | compute a descriptor: it calls `analysis` |
| `transforms` | spatial transforms and augmentation, on a clip and on raw arrays, and the reorientations | |
| `batch` | many clips: loading a directory, harmonizing, stacking into one array | own a per-clip layout: it calls `features` |

bvhplot (`pybvh/bvhplot/`) splits at its Scene: the router and `_from_bvh` read a `Bvh`, and no module behind them does. "The core" below means the package modules that know a clip or its nodes.

| Module | Owns | Does not |
|---|---|---|
| `__init__` (the router) | the public functions; preparing the user's `Bvh` (timing, resampling, world-up checks, rest pose); arranging a comparison; calling the backend `_backends` chooses | draw, or decide which backend runs |
| `_backends` | which backend runs a call: the environment value and the one function that reads the machine, the choice for `frame`, `render` (by the video table's preference per extension and the installed packages) and `play`, the rule for an option the chosen backend does not read (ADR 0006) and for a `Style` field the caller set at a value it does not draw, applied to the declarations it collects from the backends, and the `Style` fields every backend accepts, the install hints and the import checks, `play`'s 30 fps cap | read a `Bvh` or a Scene, or import an optional package but to ask whether it imports |
| `_from_bvh` | turning a `Bvh` into a Scene: views, bones, camera presets, bone chains | get imported by any module but the router |
| `_scene` | the Scene and its views, checked read-only data, and the operations that change a Scene | import a plotting library or the core |
| `_viewport` | the geometry of one 3D picture: framing, floor, camera, schedule, projection | import a plotting library or the core |
| `_animation_options` | what an animation draws besides the Scene's skeletons, the `AnimationOptions` the router builds from the options of `render` and `play` (`play`'s with a fixed camera and no ghost or trail): the camera's schedule, ghosts, the trail, the frame counter | import a plotting library or the core |
| `_style` | `Style`, its presets and palettes, which fields its caller set, the form a backend declares the `Style` values it draws in (`DrawnStyle`, `STYLE_VALUES`), the ghost, trace and frame-counter conventions | import a plotting library other than matplotlib's color parser, with which a `Style` checks its colors where it is made |
| `_colors` | the color resolution that needs matplotlib's color parser | |
| `_playback` | the viewer's playback state machine | import a rendering library |
| `_video` | writing an animation's frames to a file: the video target and the rate each file is written at, the table of every extension `render` writes, its writer (`.html` named as matplotlib's own player page) and whether auto gives it to matplotlib, the ffmpeg pipe (H.264), OpenCV's MPEG-4 Part 2 writer, Pillow's GIF, WebP and APNG | draw, or decide which backend runs (it keeps auto's preference per extension, which `_backends` applies) |
| `_vedo_capsules` | the capsule geometry and the floor both vedo backends draw | |
| `_matplotlib`, `_opencv`, `_k3d`, `_vedo`, `_vedo_offscreen` | drawing a Scene with a Style in one toolkit; the declarations, beside each entry function, of the options it reads and the `Style` values it draws | compute a box, a floor or a camera of their own for a 3D picture (one exception: the floor of matplotlib's overlay `sequence`), or read the core; the 2D `trajectory` plot has no viewport, and `_matplotlib` frames it |

## 3. Why the boundaries sit where they do

- **A `Bvh` method computes nothing another module owns.** A descriptor, a transform or a picture is computed by its module, and the method only adapts the clip to it: it hands the `Bvh` itself to a function of `analysis`, `features`, `transforms`, `io` or bvhplot, or arrays read from it (positions, a speed, the frame time) to an array kernel of `geometry` or `analysis`. The computation has one home, documented there, and the `Bvh` stays the discoverable surface.
- **One implementation per concern.** A node tree is built only by `node_tree` (the parser, `df_to_bvh` and `extract_joints` all go through it), so the checks are made in one place. The Euler-to-matrix conversion is only in `rotations`, and what a clip implies about orientation is computed only in `_orientation`, which `transforms`, `analysis`, `batch` and bvhplot call, directly or through `Bvh`, rather than keep their own. Skeleton topology has one derivation, `FkTopology.from_nodes`: forward kinematics runs on its result, `Bvh.fk_topology` returns it, and the edges and the bones bvhplot draws are views of that, so they cannot disagree with the geometry forward kinematics produces.
- **Units change at the boundary.** The angles a `Bvh` holds are radians; the conversion from and to the degrees of a file or a DataFrame happens where those are read or written (`io`, `dataframe`, `Bvh.to_df_dict`).
- **Descriptors, assembly and datasets are three modules.** `analysis` computes descriptors, `features` packs one clip's into an array, `batch` stacks clips and delegates each clip to `features`, so the column layout has one owner whether one clip or a dataset is exported.
- **The scene ground and the contact reference are two quantities**, never substituted for one another: ADR 0002. One kernel of `geometry` estimates both, over the points it is given: `analysis` passes it every node for the scene ground, `_contacts` the joints it tests.
- **The skeleton's frame and the world's are two frames.** The rest pose's facts are never read against world up, only a frame's root rotation carries them into the world, and `mirror` works in the skeleton's frame: ADR 0004.
- **bvhplot draws from a Scene, never from a `Bvh`.** Only the router and `_from_bvh` read a `Bvh`; `_viewport` computes each 3D picture's geometry from the Scene's views once (the 2D `trajectory` plot, matplotlib only, is framed by `_matplotlib`); a backend translates Scene, Style and viewport into its toolkit's calls. So every backend computes floor, framing and camera by the same rules: once per panel where it draws each skeleton in its own panel (matplotlib, OpenCV), once over all the views where it draws them in one scene (k3d, vedo), with one exception, the overlay `sequence`, which re-centers its sampled poses and takes its floor from their lowest point. Each backend can be tested from a Scene built from arrays (`tests/synthetic_scene.py`). The import guards in `tests/test_scene.py` pin the boundary and state what they cannot see. bvhplot's scope against pybvh-blender is `pybvh/bvhplot/CHARTER.md`; the vedo shadows are ADR 0001.

## 4. Where the rest lives

- The BVH format: `docs/guide/core-concepts.md`. The arrays, the index spaces and the node table: `GLOSSARY.md`. The forward-kinematics recurrence: `frames_to_node_positions`.
- The bundled sample clips and what each is for: `bvh_data/README.md`. A quick tour of the API: `README.md` and `docs/api/index.md`.
- The tests: what a test pins and how it asserts is in `CODING_STANDARDS.md` ("Tests"); clips and Scenes built from arrays with known properties come from `tests/synthetic_bvh.py` and `tests/synthetic_scene.py`; the frozen references, and when they may be regenerated, are in `tests/fixtures/README.md`. An invariant that holds across the package is pinned by one guard test, and every test file's docstring says what it covers.
- The docs site, the gallery and the notebooks: `docs/tutorials.md` ("Editing the tutorials").
- Dependencies, versions and tool configuration: `pyproject.toml`; the checks a pull request runs: `.github/workflows/`; how a change reaches `main`: `CONTRIBUTING.md`.
- What changed between releases, old names included: `CHANGELOG.md` and `pybvh/API_RENAME.md`; how an entry is written: `CONTRIBUTING.md` ("The CHANGELOG").
