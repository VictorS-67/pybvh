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
- **Motion features**: velocities, foot contacts, feature export
- **Batch loading** with optional parallel I/O
- **3D visualization** with multiple backends (matplotlib, OpenCV, k3d, vedo) — including a `follow=True` render mode where the camera tracks the character's rotation

## Quick example

```python
import pybvh

bvh = pybvh.read_bvh_file("walk.bvh")
bvh.root_pos          # (F, 3) root translation
bvh.joint_angles      # (F, J, 3) Euler angles in radians

# Convert to 6D rotation representation
root_pos, rot6d = bvh.to_6d()

# Export features for ML
features = bvh.to_feature_array(representation="6d", include_velocities=True)
```

## Companion library

For ML-specific features (tensor packing, PyTorch Datasets, augmentation pipelines), see [pybvh-ml](https://github.com/VictorS-67/pybvh-ml).

## Stability and versioning

**pybvh is in 0.x — expect breaking changes between minor versions.** We treat 0.x as design space: when a past choice turns out to be wrong, we fix it at the root rather than carry scar tissue forward. Each release has a clear migration path in the [CHANGELOG](https://github.com/VictorS-67/pybvh/blob/main/CHANGELOG.md), no deprecation cycles. If you depend on pybvh from production code, **pin to an exact version** (`pybvh==0.7.0`) and read the upgrade notes before bumping.

pybvh will commit to strict semver at **1.0**: no breaking changes within a major version, deprecation warnings (at least one minor release) before any removal. Until then, "make the library better" wins over "preserve the old behavior."
