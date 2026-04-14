# pybvh

[![PyPI version](https://img.shields.io/pypi/v/pybvh)](https://pypi.org/project/pybvh/)
[![Python](https://img.shields.io/pypi/pyversions/pybvh)](https://pypi.org/project/pybvh/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A lightweight Python library for reading, writing, and manipulating BVH motion capture files.
Built for researchers and developers working with skeletal animation and motion data.

## Features

- **Read & write** BVH files with full hierarchy and motion data preservation
- **Rotation conversions** between Euler angles, rotation matrices, quaternions, 6D (Zhou et al.), and axis-angle — all vectorized with NumPy
- **Forward kinematics** to compute 3D joint positions from angles
- **Skeleton operations**: retargeting, scaling, joint extraction, Euler order changes
- **Frame operations**: slicing, concatenation, resampling to different frame rates
- **Spatial transforms**: mirroring, vertical rotation, speed perturbation, joint noise, root translation, frame dropout — all with seeded randomization
- **Motion analysis**: joint velocities/accelerations, root-relative positions, foot contact detection, normalization utilities, and a one-stop `to_feature_array()` export
- **Batch loading** of entire directories with optional parallel I/O
- **NumPy export** in any rotation representation — ready for any downstream workflow
- **Pandas ready** via an export option ready to become a DataFrame
- **3D visualization** with multiple backends (matplotlib, OpenCV, k3d, vedo)

## Philosophy

pybvh is framework-agnostic and outputs pure NumPy arrays. It understands motion capture data but does not assume what you'll do with it — the same library serves ML researchers, biomechanics scientists, and game developers. For ML-specific features (tensor packing, PyTorch Datasets, augmentation pipelines), see the companion library [pybvh-ml](https://github.com/VictorS-67/pybvh-ml).

## Installation

```bash
pip install pybvh
```

## Quick Start

```python
import pybvh

# Load a BVH file
bvh = pybvh.read_bvh_file("walk.bvh")
print(bvh)  # "24 elements in the Hierarchy, 75 frames at 30.0 fps (frame_time=0.033333s)"

# Access motion data as NumPy arrays
bvh.root_pos          # (F, 3) root translation per frame
bvh.joint_angles      # (F, J, 3) Euler angles in degrees
bvh.joint_names       # ['Hips', 'Spine', ...] (excludes end sites)

# Get 3D joint positions via forward kinematics
coords = bvh.spatial_coords()  # (F, N, 3)

# Convert to other rotation representations
root_pos, quats, joints = bvh.to_quaternions()   # (F,3), (F,J,4), joints
root_pos, rot6d, joints = bvh.to_6d()            # (F,3), (F,J,6), joints

# Write back to file
bvh.write("output.bvh")
```

## Batch Loading

Load an entire directory of BVH files and convert to NumPy arrays in one call:

```python
from pybvh import read_bvh_directory, batch_to_numpy

# Load all BVH files from a directory
clips = read_bvh_directory("dataset/", parallel=True)

# Convert to padded NumPy array
data = batch_to_numpy(clips, representation="6d", pad=True)
# shape: (batch, max_frames, features)
```

Supported representations: `"euler"`, `"quaternion"`, `"6d"`, `"axisangle"`, `"rotmat"`.

## Motion Analysis

Compute motion derivatives, foot contacts, and export everything in a single array:

```python
# Joint velocities and accelerations (finite differences of FK positions)
vel = bvh.joint_velocities()        # (F-1, N, 3) in units/second
acc = bvh.joint_accelerations()     # (F-2, N, 3)
ang_vel = bvh.angular_velocities()  # (F-1, J, 3) in radians/second

# Root-relative positions and trajectory
rel_pos = bvh.root_relative_positions()  # (F, N, 3)
traj = bvh.root_trajectory()             # (F, 4) ground pos + heading

# Foot contact detection (auto-detects foot joints)
contacts = bvh.foot_contacts()  # (F, num_feet) binary indicators

# One-stop export — flat feature array
features = bvh.to_feature_array(
    representation="6d",
    include_velocities=True,
    include_foot_contacts=True,
)  # (F-1, D)
```

Normalize across a dataset:

```python
from pybvh import compute_normalization_stats, normalize_array, denormalize_array

stats = compute_normalization_stats(clips, representation="6d")
normalized = normalize_array(data, stats)
# stats are plain dicts — save with np.savez("stats.npz", **stats)
```

## Spatial Transforms

Standard motion transforms — all support seeded randomization for reproducibility:

```python
from pybvh import transforms

# Left-right mirroring (auto-detects joint pairs and lateral axis)
bvh_mirrored = transforms.mirror(bvh)

# Vertical rotation (auto-detects up axis)
bvh_rotated = transforms.rotate_vertical(bvh, angle_deg=90)
bvh_rotated = transforms.random_rotate_vertical(bvh, rng=np.random.default_rng(42))

# Speed perturbation (factor > 1 = faster, < 1 = slower)
bvh_fast = transforms.perturb_speed(bvh, factor=1.5)

# Joint noise injection
bvh_noisy = transforms.add_noise(bvh, sigma_deg=1.0, sigma_pos=0.5, rng=rng)

# Root translation
bvh_shifted = transforms.translate_root(bvh, offset=[100, 0, 0])

# Frame dropout with SLERP interpolation (same frame count)
bvh_dropped = transforms.drop_frames(bvh, drop_rate=0.1, rng=rng)
```

All transforms also available as `Bvh` methods: `bvh.mirror()`, `bvh.rotate_vertical(90)`, etc.

## Skeleton Operations

```python
# Change Euler rotation order for all joints (or a single joint via `joint=`)
bvh_xyz = bvh.change_euler_order("XYZ")
bvh_hips_xyz = bvh.change_euler_order("XYZ", joint="Hips")

# Scale the skeleton
bvh_scaled = bvh.scale(0.01)  # meters to centimeters

# Retarget motion to a different skeleton
bvh_retarget = bvh.retarget(reference_bvh)

# Extract a subset of joints
bvh_upper = bvh.extract_joints(["Hips", "Spine", "Neck", "Head"])

# Slice and concatenate frames
clip = bvh.slice_frames(10, 50)
combined = bvh.concat(other_bvh)

# Resample to a different frame rate
bvh_30fps = bvh.resample(30)
```

## Rotation Utilities

All functions are batch-vectorized and work on arbitrary batch dimensions:

```python
from pybvh import rotations

# Convert between any pair of representations
R = rotations.euler_to_rotmat(angles, order="ZYX", degrees=True)
q = rotations.rotmat_to_quat(R)
aa = rotations.quat_to_euler(q, order="ZYX", degrees=True)

# Quaternion SLERP interpolation
q_mid = rotations.quat_slerp(q1, q2, t=0.5)
```

## Visualization

```python
# Single-skeleton calls are methods on the Bvh object
bvh.plot_rest_pose()                             # T-pose
bvh.plot_frame(frame=0, camera="front")          # also "side", "top", (azim, elev)
bvh.plot_trajectory()                            # 2D top-down root path
bvh.render("walk.mp4")                           # video/GIF/HTML export
bvh.render("walk.mp4", follow=True)              # camera tracks character as it turns
bvh.play()                                       # interactive playback (auto-detects backend)

# Multi-skeleton comparisons go through the module functions
from pybvh import bvhplot
bvhplot.frame([bvh1, bvh2], frame=0, labels=["Original", "Generated"])
bvhplot.render([bvh1, bvh2], "compare.mp4", sync="pad")

# World orientation is exposed as a property (auto-detected, overridable)
bvh.world_up          # '+y', '+z', etc.
bvh.forward_at(0)     # character's facing direction at a specific frame
bvh.world_up = '+y'   # manual override if the auto-detect is wrong for your file
```

Install optional visualization backends for best performance:

```bash
pip install pybvh[opencv]       # Fast video rendering 
pip install pybvh[interactive]  # k3d for Jupyter notebooks
pip install pybvh[viewer]       # vedo for desktop interactive viewer
pip install pybvh[all-viz]      # All of the above
```

## Pandas Integration

```python
import pandas as pd

# BVH to DataFrame
df = pd.DataFrame(bvh.to_df_dict(mode="euler"))

# DataFrame back to BVH
from pybvh import df_to_bvh
bvh_from_df = df_to_bvh(bvh.hierarchy_info_as_dict(), df)
```

## Tutorials

The repository includes Jupyter notebooks with detailed walkthroughs:

1. [Introduction to pybvh](https://github.com/VictorS-67/pybvh/blob/main/tutorials/1.Introduction_pybvh.ipynb) — reading, writing, and basic operations
2. [Spatial coordinates](https://github.com/VictorS-67/pybvh/blob/main/tutorials/2.Spatial_coordinates.ipynb) — forward kinematics and 3D positions
3. [Rotations](https://github.com/VictorS-67/pybvh/blob/main/tutorials/3.Rotations.ipynb) — rotation representations and conversions

## Requirements

- Python >= 3.9
- NumPy >= 1.21
- Matplotlib >= 3.7

Pandas is optional (`pip install "pybvh[pandas]"`) - only used in the tutorials, not part of pybvh library.

## License

MIT
