# pybvh

[![PyPI version](https://img.shields.io/pypi/v/pybvh)](https://pypi.org/project/pybvh/)
[![Python](https://img.shields.io/pypi/pyversions/pybvh)](https://pypi.org/project/pybvh/)
[![Docs](https://img.shields.io/badge/docs-online-4051b5)](https://victors-67.github.io/pybvh/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/VictorS-67/pybvh/blob/main/LICENSE)

pybvh is a lightweight Python library to read, write and manipulate BVH motion capture files. It is for researchers and developers who work with skeletal animation and motion data.

![A skeleton moves across the frame, and the path of one hand is traced behind it. pybvh renders motion, and it extracts trajectories from the motion that you can analyze](https://raw.githubusercontent.com/VictorS-67/pybvh/main/docs/assets/hand-trajectory.gif)

**[Documentation](https://victors-67.github.io/pybvh/)** · [Quick Start](https://victors-67.github.io/pybvh/getting-started/quickstart/) · [Feature Gallery](https://victors-67.github.io/pybvh/gallery/) — every feature, one picture each · [Find a function](https://victors-67.github.io/pybvh/api/) · [Tutorials](https://victors-67.github.io/pybvh/tutorials/)

## Features

- **Read & write** BVH files, and keep the full hierarchy and all the motion data — [I/O](https://victors-67.github.io/pybvh/api/io/)
- **Rotation conversions** between Euler angles, rotation matrices, quaternions, 6D (Zhou et al.) and axis-angle. All conversions are vectorized with NumPy — [guide](https://victors-67.github.io/pybvh/guide/rotations/)
- **Forward kinematics** to compute 3D joint positions from angles — [core concepts](https://victors-67.github.io/pybvh/guide/core-concepts/)
- **Skeleton operations**: retargeting, scaling, joint extraction, Euler order changes — [guide](https://victors-67.github.io/pybvh/guide/skeleton-ops/)
- **Frame operations**: slicing, concatenation, resampling to different frame rates — [guide](https://victors-67.github.io/pybvh/guide/skeleton-ops/)
- **Spatial transforms**: mirroring, vertical rotation, speed perturbation, joint noise, root translation and frame dropout. All of them use seeded randomization — [guide](https://victors-67.github.io/pybvh/guide/augmentation/)
- **Motion analysis**: joint velocities and accelerations, root trajectory, foot contact detection and gait parameters. `to_feature_array()` exports the features in one call — [guide](https://victors-67.github.io/pybvh/guide/feature-export/)
- **Motion descriptors**: trajectory geometry (curvature, torsion, path length, bounding volumes, centre of mass), dynamics (jerk, smoothness/SPARC, kinetic energy, gait), and SE(3) rigid-transform math (twists, screw interpolation, geodesic distance). All of it is pure NumPy — [guide](https://victors-67.github.io/pybvh/guide/motion-descriptors/) · [gallery](https://victors-67.github.io/pybvh/gallery/)
- **Signal utilities** (`pybvh.signal`): finite differences, temporal statistics, smoothing, FFT/dominant frequency, polyline simplification — [API](https://victors-67.github.io/pybvh/api/signal/)
- **Batch loading** of a whole directory, with optional parallel I/O — [guide](https://victors-67.github.io/pybvh/guide/feature-export/)
- **Pandas ready**: `pd.DataFrame` takes the output of `to_df_dict()` as it is — [guide](https://victors-67.github.io/pybvh/guide/skeleton-ops/)
- **3D visualization**: publication-grade stills, the sampled-poses sequence figure, ghost trails, turntable orbits, video export and shadowed capsule renders. The backends are matplotlib, OpenCV, k3d and vedo — [guide](https://victors-67.github.io/pybvh/guide/publication-figures/) · [API](https://victors-67.github.io/pybvh/api/bvhplot/)

## Philosophy

pybvh is framework-agnostic, and it outputs pure NumPy arrays. It understands motion capture data, but it does not assume what you will do with the data. The same library serves ML researchers, biomechanics scientists and game developers. For features specific to ML (tensor packing, PyTorch Datasets, augmentation pipelines), see the companion library [pybvh-ml](https://github.com/VictorS-67/pybvh-ml) ([documentation](https://victors-67.github.io/pybvh-ml/)).

## Installation

```bash
pip install pybvh
```

## Quick Start

```python
import pybvh

# Load a BVH file (pybvh.Bvh.from_file("walk.bvh") is the classmethod spelling)
bvh = pybvh.read_bvh_file("walk.bvh")
print(bvh)  # "24 joints, 75 frames at 30.0 fps (frame_time=0.033333s, from walk.bvh)"

# Access motion data as NumPy arrays
bvh.root_pos          # (F, 3) root translation per frame
bvh.joint_angles      # (F, J, 3) Euler angles in radians
bvh.joint_names       # ['Hips', 'Spine', ...] (excludes end sites)

# Get 3D joint positions via forward kinematics
coords = bvh.node_positions()  # (F, N, 3)

# Convert to other rotation representations
root_pos, quats = bvh.to_quat()          # (F, 3), (F, J, 4)
root_pos, rot6d = bvh.to_6d()            # (F, 3), (F, J, 6)

# Write back to file
bvh.write("output.bvh")
```

## Motion Analysis

```python
vel = bvh.joint_velocities()    # (F, J, 3) in units/second
contacts = bvh.foot_contacts()  # (F, num_feet) binary labels, feet auto-detected

# One-stop export — flat feature array for ML pipelines
features = bvh.to_feature_array(
    representation="6d",
    include_velocities=True,
    include_foot_contacts=True,
)  # (F, D)
```

Beyond the basics, pybvh has a full layer of descriptors: curvature, smoothness (SPARC), kinetic energy, gait parameters and SE(3) twists. The [Feature Gallery](https://victors-67.github.io/pybvh/gallery/) draws each one with its exact call, and the [Motion Descriptors guide](https://victors-67.github.io/pybvh/guide/motion-descriptors/) explains them.

## Visualization

```python
bvh.plot_rest_pose()                              # T-pose
bvh.plot_frame(frame=0, camera="front")           # also "side", "top", (azim, elev)
bvh.plot_sequence(n_poses=8)                      # the motion-paper still: time as lightness
bvh.plot_trajectory()                             # 2D top-down root path
bvh.render("walk.mp4", codec="h264")              # video/GIF/HTML export, browser-playable
bvh.render("walk.mp4", ghost=3, trajectory=True)  # faded trail + root path on the floor
bvh.render("orbit.mp4", camera="turntable")       # a full 360-degree orbit over the clip
bvh.plot_frame(frame=0, backend="vedo")           # shadowed 3D capsules, offscreen, headless-safe
bvh.play()                                        # interactive playback (auto-detects backend)
```

![A walking skeleton rendered with three faded ghost poses trailing behind it and its root path traced on the floor](https://raw.githubusercontent.com/VictorS-67/pybvh/main/gallery/feature_gallery_ghost.gif)

Every function takes `style=`, with one of these values:

- `"paper"`, the default: a ground plane, bone colors per chain, joint markers, and the axes off.
- `"dark"`, for slides.
- `"debug"`, to read coordinates off the axes.
- A `Style` instance, which overrides any field.

```bash
pip install pybvh[opencv]       # Fast video rendering
pip install pybvh[interactive]  # k3d for Jupyter notebooks
pip install pybvh[viewer]       # vedo for desktop interactive viewer
pip install pybvh[all-viz]      # All of the above
```

To get figures from Python into a paper (vector stills, supplementary video, the capsule look), read [Publication Figures](https://victors-67.github.io/pybvh/guide/publication-figures/). For the comparison of several skeletons, camera control and the details of each backend, read the [Visualization API](https://victors-67.github.io/pybvh/api/bvhplot/).

## More

| Topic | In one line | Docs |
|---|---|---|
| Batch loading | Load a directory, harmonize heterogeneous skeletons, export one padded array | [Feature Export](https://victors-67.github.io/pybvh/guide/feature-export/) |
| Spatial transforms | `mirror`, `rotate_vertical`, `add_rotation_noise`, `perturb_speed`, `drop_frames` — seeded randomization | [Data Augmentation](https://victors-67.github.io/pybvh/guide/augmentation/) |
| Skeleton & frame ops | `retarget`, `scale`, `extract_joints`, slicing, concatenation, `resample` | [Skeleton Operations](https://victors-67.github.io/pybvh/guide/skeleton-ops/) |
| Rotation utilities | Batch-vectorized conversions between all five representations, SLERP, SE(3) | [Rotations & SE(3)](https://victors-67.github.io/pybvh/guide/rotations/) |
| World up & orientation | Up-axis detection, `forward_at`/`left_at`, reorientation | [World Up](https://victors-67.github.io/pybvh/guide/world-up/) |
| Pandas integration | `to_df_dict()` / `df_to_bvh()` round trip | [Skeleton Operations](https://victors-67.github.io/pybvh/guide/skeleton-ops/) |

## Tutorials

Eight Jupyter notebooks give detailed walkthroughs, from reading your first file to motion descriptors. See the [tutorials page](https://victors-67.github.io/pybvh/tutorials/). Each tutorial is committed as a `.ipynb` file and a `.py` file that Jupytext pairs, so you can review the source as plain Python.

## Stability and versioning

**pybvh is in 0.x. Expect breaking changes between minor versions.**

We treat 0.x as design space. When a past choice proves to be wrong, we fix it at the root, and we do not carry the old choice forward. There are no deprecation cycles and no compatibility shims. Each release ships a single clean migration path, which the [CHANGELOG](CHANGELOG.md) documents. If your production code depends on pybvh, **pin to an exact version** (`pybvh==0.8.2`). Read the upgrade notes before you move to a newer version.

This will change at **1.0**. From 1.0 on, pybvh will commit to strict semver:

- No breaking changes within a major version.
- A deprecation warning before any future removal, for at least one minor release.

Until 1.0, "make the library better" wins over "preserve the old behavior."

## Requirements

- Python >= 3.10
- NumPy >= 1.21
- Matplotlib >= 3.7

Pandas is optional (`pip install "pybvh[pandas]"`). Only the tutorials use it, and it is not part of the pybvh library.

## Citing pybvh

If pybvh is useful in your research, please cite it. On GitHub, the *Cite this repository* button in the sidebar gives an APA and a BibTeX citation, which it generates from [`CITATION.cff`](https://github.com/VictorS-67/pybvh/blob/main/CITATION.cff). The BibTeX entry is below. Set its `version` to the release you used:

```bibtex
@software{schneider_pybvh,
  author  = {Schneider, Victor},
  title   = {pybvh},
  url     = {https://github.com/VictorS-67/pybvh},
  version = {0.9.0},
  year    = {2026}
}
```

## License

[MIT](https://github.com/VictorS-67/pybvh/blob/main/LICENSE)
