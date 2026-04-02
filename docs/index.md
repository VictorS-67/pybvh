# pybvh

A lightweight Python library for reading, writing, and manipulating BVH motion capture files.

pybvh is framework-agnostic and outputs pure NumPy arrays. It understands motion capture data but does not assume what you'll do with it — the same library serves ML researchers, biomechanics scientists, and game developers.

## Features

- **Read & write** BVH files with full hierarchy and motion data preservation
- **Rotation conversions** between Euler angles, rotation matrices, quaternions, 6D, and axis-angle
- **Forward kinematics** to compute 3D joint positions from angles
- **Skeleton operations**: retargeting, scaling, joint extraction, Euler order changes
- **Frame operations**: slicing, concatenation, resampling
- **Data augmentation**: mirroring, rotation, speed perturbation, noise, dropout
- **ML pipeline features**: velocities, foot contacts, normalization, feature export
- **Batch loading** with optional parallel I/O
- **3D visualization** with Matplotlib

## Quick example

```python
import pybvh

bvh = pybvh.read_bvh_file("walk.bvh")
bvh.root_pos          # (F, 3) root translation
bvh.joint_angles      # (F, J, 3) Euler angles in degrees

# Convert to 6D rotation representation
root_pos, rot6d, joints = bvh.get_frames_as_6d()

# Export features for ML
features = bvh.to_feature_array(representation="6d", include_velocities=True)
```

## Companion library

For ML-specific features (tensor packing, PyTorch Datasets, augmentation pipelines), see [pybvh-ml](https://github.com/VictorS-67/pybvh-ml).
