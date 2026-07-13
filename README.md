# pybvh

[![PyPI version](https://img.shields.io/pypi/v/pybvh)](https://pypi.org/project/pybvh/)
[![Python](https://img.shields.io/pypi/pyversions/pybvh)](https://pypi.org/project/pybvh/)
[![Docs](https://img.shields.io/badge/docs-online-4051b5)](https://victors-67.github.io/pybvh/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A lightweight Python library for reading, writing, and manipulating BVH motion capture files.
Built for researchers and developers working with skeletal animation and motion data.

![A skeleton animates while its hand traces a blue trajectory — pybvh renders motion and extracts analyzable trajectories from it](https://raw.githubusercontent.com/VictorS-67/pybvh/main/docs/assets/hand-trajectory.gif)

**[Documentation](https://victors-67.github.io/pybvh/)** · [Quick Start](https://victors-67.github.io/pybvh/getting-started/quickstart/) · [Feature Gallery](https://victors-67.github.io/pybvh/gallery/) — every feature, one picture each · [Find a function](https://victors-67.github.io/pybvh/api/) · [Tutorials](https://victors-67.github.io/pybvh/tutorials/)

## Features

- **Read & write** BVH files with full hierarchy and motion data preservation — [I/O](https://victors-67.github.io/pybvh/api/io/)
- **Rotation conversions** between Euler angles, rotation matrices, quaternions, 6D (Zhou et al.), and axis-angle — all vectorized with NumPy — [guide](https://victors-67.github.io/pybvh/guide/rotations/)
- **Forward kinematics** to compute 3D joint positions from angles — [core concepts](https://victors-67.github.io/pybvh/guide/core-concepts/)
- **Skeleton operations**: retargeting, scaling, joint extraction, Euler order changes — [guide](https://victors-67.github.io/pybvh/guide/skeleton-ops/)
- **Frame operations**: slicing, concatenation, resampling to different frame rates — [guide](https://victors-67.github.io/pybvh/guide/skeleton-ops/)
- **Spatial transforms**: mirroring, vertical rotation, speed perturbation, joint noise, root translation, frame dropout — all with seeded randomization — [guide](https://victors-67.github.io/pybvh/guide/augmentation/)
- **Motion analysis**: joint velocities/accelerations, root trajectory, foot contact detection, gait parameters, and a one-stop `to_feature_array()` export — [guide](https://victors-67.github.io/pybvh/guide/feature-export/)
- **Motion descriptors**: trajectory geometry (curvature, torsion, path length, bounding volumes, centre of mass), dynamics (jerk, smoothness/SPARC, kinetic energy, gait), and SE(3) rigid-transform math (twists, screw interpolation, geodesic distance) — all pure NumPy — [guide](https://victors-67.github.io/pybvh/guide/motion-descriptors/) · [gallery](https://victors-67.github.io/pybvh/gallery/)
- **Signal utilities** (`pybvh.signal`): finite differences, temporal statistics, smoothing, FFT/dominant frequency, polyline simplification — [API](https://victors-67.github.io/pybvh/api/signal/)
- **Batch loading** of entire directories with optional parallel I/O — [guide](https://victors-67.github.io/pybvh/guide/feature-export/)
- **Pandas ready** — `to_df_dict()` output drops straight into `pd.DataFrame` — [guide](https://victors-67.github.io/pybvh/guide/skeleton-ops/)
- **3D visualization** with multiple backends (matplotlib, OpenCV, k3d, vedo) — [API](https://victors-67.github.io/pybvh/api/bvhplot/)

## Philosophy

pybvh is framework-agnostic and outputs pure NumPy arrays. It understands motion capture data but does not assume what you'll do with it — the same library serves ML researchers, biomechanics scientists, and game developers. For ML-specific features (tensor packing, PyTorch Datasets, augmentation pipelines), see the companion library [pybvh-ml](https://github.com/VictorS-67/pybvh-ml).

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

Beyond the basics sits a full descriptor layer — curvature, smoothness (SPARC), kinetic energy, gait parameters, SE(3) twists — each drawn with its exact call in the [Feature Gallery](https://victors-67.github.io/pybvh/gallery/) and explained in the [Motion Descriptors guide](https://victors-67.github.io/pybvh/guide/motion-descriptors/).

## Visualization

```python
bvh.plot_rest_pose()                             # T-pose
bvh.plot_frame(frame=0, camera="front")          # also "side", "top", (azim, elev)
bvh.plot_trajectory()                            # 2D top-down root path
bvh.render("walk.mp4")                           # video/GIF/HTML export
bvh.render("walk.mp4", follow=True)              # camera tracks character as it turns
bvh.play()                                       # interactive playback (auto-detects backend)
```

```bash
pip install pybvh[opencv]       # Fast video rendering
pip install pybvh[interactive]  # k3d for Jupyter notebooks
pip install pybvh[viewer]       # vedo for desktop interactive viewer
pip install pybvh[all-viz]      # All of the above
```

Multi-skeleton comparison, camera control, and backend details: [Visualization API](https://victors-67.github.io/pybvh/api/bvhplot/).

## More

| Topic | In one line | Docs |
|---|---|---|
| Batch loading | Load a directory, harmonize heterogeneous skeletons, export one padded array | [Feature Export](https://victors-67.github.io/pybvh/guide/feature-export/) |
| Spatial transforms | `mirror`, `rotate_vertical`, `add_noise`, `perturb_speed`, `drop_frames` — seeded randomization | [Data Augmentation](https://victors-67.github.io/pybvh/guide/augmentation/) |
| Skeleton & frame ops | `retarget`, `scale`, `extract_joints`, slicing, concatenation, `resample` | [Skeleton Operations](https://victors-67.github.io/pybvh/guide/skeleton-ops/) |
| Rotation utilities | Batch-vectorized conversions between all five representations, SLERP, SE(3) | [Rotations & SE(3)](https://victors-67.github.io/pybvh/guide/rotations/) |
| World up & orientation | Up-axis detection, `forward_at`/`left_at`, reorientation | [World Up](https://victors-67.github.io/pybvh/guide/world-up/) |
| Pandas integration | `to_df_dict()` / `df_to_bvh()` round trip | [Skeleton Operations](https://victors-67.github.io/pybvh/guide/skeleton-ops/) |

## Tutorials

Eight Jupyter notebooks with detailed walkthroughs, from reading your first file to motion descriptors — see the [tutorials page](https://victors-67.github.io/pybvh/tutorials/).
Each tutorial is committed as a Jupytext-paired `.ipynb` + `.py` so the source is reviewable as plain Python.

## Stability and versioning

**pybvh is in 0.x — expect breaking changes between minor versions.**

We treat 0.x as design space: when a past choice turns out to be wrong, we fix it at the root rather than carry scar tissue forward. No deprecation cycles, no compatibility shims; each release ships a single clean migration path, documented in the [CHANGELOG](CHANGELOG.md). If you depend on pybvh from production code, **pin to an exact version** (`pybvh==0.8.0`) and read the upgrade notes before bumping.

This will change at **1.0**: from then on, pybvh will commit to strict semver — no breaking changes within a major version, deprecation warnings (at least one minor release) before any future removal. Until 1.0, "make the library better" wins over "preserve the old behavior."

## Requirements

- Python >= 3.9
- NumPy >= 1.21
- Matplotlib >= 3.7

Pandas is optional (`pip install "pybvh[pandas]"`) - only used in the tutorials, not part of pybvh library.

## License

MIT
