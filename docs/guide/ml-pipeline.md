# ML Pipeline

## Batch loading

```python
from pybvh import read_bvh_directory, batch_to_numpy

clips = read_bvh_directory("dataset/", parallel=True)
data = batch_to_numpy(clips, representation="6d", pad=True)  # (B, F_max, D)
```

Supported representations: `"euler"`, `"quaternion"`, `"6d"`, `"axisangle"`, `"rotmat"`.

## Motion features

```python
vel = bvh.get_joint_velocities()        # (F-1, N, 3) in units/second
acc = bvh.get_joint_accelerations()     # (F-2, N, 3)
ang_vel = bvh.get_angular_velocities()  # (F-1, J, 3) in radians/second

rel_pos = bvh.get_root_relative_positions()  # (F, N, 3)
traj = bvh.get_root_trajectory()             # (F, 4) ground pos + heading

contacts = bvh.get_foot_contacts()  # (F, num_feet) binary labels
```

## One-stop feature export

```python
features = bvh.to_feature_array(
    representation="6d",
    include_velocities=True,
    include_foot_contacts=True,
)  # (F-1, D) flat array
```

## Normalization

```python
from pybvh import compute_normalization_stats, normalize_array, denormalize_array

stats = compute_normalization_stats(clips, representation="6d")
normalized = normalize_array(data, stats)
recovered = denormalize_array(normalized, stats)

# Save/load stats (compatible with HumanML3D Mean.npy/Std.npy)
import numpy as np
np.savez("stats.npz", **stats)
```

## Skeleton metadata

```python
bvh.euler_orders   # ['ZYX', 'ZYX', ...] per joint
bvh.edges          # [(1, 0), (2, 1), ...] for GCN adjacency
bvh.joint_names    # ['Hips', 'Spine', ...]
bvh.joint_count    # 24
```

!!! tip "For full ML workflows"
    [pybvh-ml](https://github.com/VictorS-67/pybvh-ml) provides tensor packing (CTV/TVC layouts), PyTorch Datasets, augmentation pipelines, and preprocessing to HDF5/npz.
