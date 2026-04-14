# ML Pipeline

## Batch loading

```python
from pybvh import read_bvh_directory, batch_to_numpy

clips = read_bvh_directory("dataset/", parallel=True)
data = batch_to_numpy(clips, representation="6d", pad=True)  # (B, F_max, D)
```

Supported representations: `"euler"`, `"quaternion"`, `"6d"`, `"axisangle"`, `"rotmat"`.

When `pad=False` (default), returns a list of arrays with different frame counts. When `pad=True`, zero-pads to the longest clip and returns a single `(B, F_max, D)` array.

## Motion features

Each feature captures a different aspect of the motion. Choose based on what your model needs:

```python
vel = bvh.joint_velocities()        # (F-1, N, 3) in units/second
acc = bvh.joint_accelerations()     # (F-2, N, 3)
ang_vel = bvh.angular_velocities()  # (F-1, J, 3) in radians/second

rel_pos = bvh.root_relative_positions()  # (F, N, 3)
traj = bvh.root_trajectory()             # (F, 4) ground pos + heading

contacts = bvh.foot_contacts()  # (F, num_feet) binary labels
```

| Feature | What it captures | Common use |
|---------|-----------------|------------|
| **Joint velocities** | How fast each joint moves in 3D space (finite differences of FK positions) | Motion dynamics, action recognition |
| **Joint accelerations** | Rate of velocity change per joint | Smoothness constraints, jerk detection |
| **Angular velocities** | Per-joint rotation speed via rotation matrix log map | Rotation dynamics, independent of skeleton scale |
| **Root-relative positions** | All joint positions with root subtracted each frame | Translation-invariant pose features |
| **Root trajectory** | Ground-plane position (2D) + heading as sin/cos (2D) | Locomotion conditioning, path prediction |
| **Foot contacts** | Binary per-frame indicators (height or velocity method) | Contact-aware generation, foot skating loss |

## One-stop feature export

Combines rotations, velocities, and foot contacts into a single flat array:

```python
features = bvh.to_feature_array(
    representation="6d",
    include_velocities=True,
    include_foot_contacts=True,
)  # (F-1, D) flat array
```

## Normalization

Per-channel z-score normalization across a dataset:

```python
from pybvh import compute_normalization_stats, normalize_array, denormalize_array

stats = compute_normalization_stats(clips, representation="6d")
normalized = normalize_array(data, stats)
recovered = denormalize_array(normalized, stats)

# Save/load stats
import numpy as np
np.savez("stats.npz", **stats)
loaded = dict(np.load("stats.npz"))
```

The stats dict contains `"mean"` and `"std"` arrays of shape `(D,)`. Channels with zero standard deviation are set to `std=1.0` to avoid division by zero.

## Skeleton metadata

Useful for graph-based models (GCNs, attention masks):

```python
bvh.euler_orders   # ['ZYX', 'ZYX', ...] per joint
bvh.edges          # [(1, 0), (2, 1), ...] for GCN adjacency
bvh.joint_names    # ['Hips', 'Spine', ...]
bvh.joint_count    # 24
```

!!! tip "For full ML workflows"
    [pybvh-ml](https://github.com/VictorS-67/pybvh-ml) provides tensor packing (CTV/TVC layouts), PyTorch Datasets, augmentation pipelines, and preprocessing to HDF5/npz.
