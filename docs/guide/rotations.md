# Rotation Representations

pybvh supports five rotation representations, all batch-vectorized with NumPy.

## Supported formats

| Representation | Shape | Description |
|---|---|---|
| Euler angles | `(*, 3)` | BVH native format, degrees or radians |
| Rotation matrices | `(*, 3, 3)` | Full 3x3 orthogonal matrices |
| 6D (Zhou et al.) | `(*, 6)` | Continuous representation for neural networks |
| Quaternions | `(*, 4)` | `(w, x, y, z)` scalar-first, canonical `w >= 0` |
| Axis-angle | `(*, 3)` | Rotation axis scaled by angle in radians |

## Converting between representations

```python
from pybvh import rotations

# Euler -> rotation matrix
R = rotations.euler_to_rotmat(angles, order="ZYX", degrees=True)

# Rotation matrix -> quaternion
q = rotations.rotmat_to_quat(R)

# Any pair works — direct or via convenience wrappers
rot6d = rotations.euler_to_rot6d(angles, "ZYX", degrees=True)
q = rotations.euler_to_quat(angles, "ZYX", degrees=True)
aa = rotations.euler_to_axisangle(angles, "ZYX", degrees=True)
```

All functions support arbitrary batch dimensions: `(3,)`, `(N, 3)`, `(F, J, 3)` all work.

## Bvh conversion methods

```python
root_pos, rot6d, joints = bvh.get_frames_as_6d()           # (F, J, 6)
root_pos, quats, joints = bvh.get_frames_as_quaternion()    # (F, J, 4)
root_pos, aa, joints    = bvh.get_frames_as_axisangle()     # (F, J, 3)
root_pos, R, joints     = bvh.get_frames_as_rotmat()        # (F, J, 3, 3)

# Set frames back from a different representation
bvh2 = bvh.set_frames_from_6d(root_pos, rot6d)
bvh3 = bvh.set_frames_from_quaternion(root_pos, quats)
```

## Changing Euler order

```python
# Change all joints to XYZ order (preserves physical rotations)
bvh_xyz = bvh.change_all_euler_orders("XYZ")
```

## Quaternion SLERP

```python
q_mid = rotations.quat_slerp(q1, q2, t=0.5)  # Spherical linear interpolation
```
