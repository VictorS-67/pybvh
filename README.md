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
- **Batch loading** of entire directories with optional parallel I/O
- **NumPy export** in any rotation representation — ready for ML pipelines
- **Pandas ready** via an export option ready to become a Dataset
- **3D visualization** with Matplotlib (static frames and animated videos)

## Installation

```bash
pip install pybvh
```

## Quick Start

```python
import pybvh

# Load a BVH file
bvh = pybvh.read_bvh_file("walk.bvh")
print(bvh)  # 24 joints, 120 frames at 0.008333Hz

# Access motion data as NumPy arrays
bvh.root_pos          # (F, 3) root translation per frame
bvh.joint_angles      # (F, J, 3) Euler angles in degrees
bvh.joint_names       # ['Hips', 'Spine', ...] (excludes end sites)

# Get 3D joint positions via forward kinematics
coords = bvh.get_spatial_coord()  # (F, N, 3)

# Convert to other rotation representations
root_pos, quats, joints = bvh.get_frames_as_quaternion()  # (F,3), (F,J,4), joints
root_pos, rot6d, joints = bvh.get_frames_as_6d()          # (F,3), (F,J,6), joints

# Write back to file
bvh.to_bvh_file("output.bvh")
```

## Batch Loading for ML

Load an entire dataset directory and convert to NumPy arrays in one call:

```python
from pybvh import read_bvh_directory, batch_to_numpy

# Load all BVH files from a directory
clips = read_bvh_directory("dataset/", parallel=True)

# Convert to padded NumPy array — ready for training
data = batch_to_numpy(clips, representation="6d", pad=True)
# shape: (batch, max_frames, features)
```

Supported representations: `"euler"`, `"quaternion"`, `"6d"`, `"axisangle"`, `"rotmat"`.

## Skeleton Operations

```python
# Change Euler rotation order for all joints
bvh_xyz = bvh.change_all_euler_orders("XYZ")

# Scale the skeleton
bvh_scaled = bvh.scale_skeleton(0.01)  # meters to centimeters

# Retarget motion to a different skeleton
bvh_retarget = bvh.change_skeleton(reference_bvh)

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
from pybvh import plot

# Plot a single frame
plot.plot_frame(bvh, frame=0)

# Create an animation video
plot.plot_animation(bvh, output_path="walk.mp4")
```

## Pandas Integration

```python
import pandas as pd

# BVH to DataFrame
df = pd.DataFrame(bvh.get_df_constructor(mode="euler"))

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
