# Rotation Representations & SE(3)

pybvh supports five rotation representations, all batch-vectorized with NumPy, plus the SE(3) rigid-transform layer (twists, screw interpolation). This page is the *machinery* — for the *which should I use* question, see [Choosing a Representation](choosing-rotations.md).

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
root_pos, rot6d = bvh.to_6d()           # (F, J, 6)
root_pos, quats = bvh.to_quat()         # (F, J, 4)
root_pos, aa = bvh.to_axisangle()    # (F, J, 3)
root_pos, R = bvh.to_rotmat()       # (F, J, 3, 3)

# Set frames back from a different representation
bvh2 = bvh.from_6d(root_pos, rot6d)
bvh3 = bvh.from_quat(root_pos, quats)
```

## Changing Euler order

```python
# Change all joints to XYZ order (preserves physical rotations)
bvh_xyz = bvh.change_euler_order("XYZ")

# Change a single joint only
bvh_hips = bvh.change_euler_order("XYZ", joint="Hips")
```

## Quaternion SLERP

```python
q_mid = rotations.quat_slerp(q1, q2, t=0.5)  # Spherical linear interpolation
```

## SE(3) rigid transforms

Beyond pure rotations, `pybvh.rotations` covers rigid transforms (rotation + translation as one 4×4 matrix) through the twist parameterization: a twist `[ω, v]` is six numbers — an axis-angle rotation vector `ω` and a linear generator `v` — that the exponential map turns into a transform.

```python
import numpy as np
from pybvh import rotations

# twist [ω, v] -> 4x4 rigid transform, and back exactly
twist = np.array([0.0, 0.0, 1.4, 1.0, 0.0, 0.6])
T = rotations.se3_exp(twist)
assert np.allclose(rotations.se3_log(T), twist)

# the SE(3) analogue of SLERP: blend two transforms along a constant screw
T_mid = rotations.screw_interpolate(T0, T1, t=0.5)   # t can be an array

# pose of one body segment in another's local frame (the geometry -> SE(3) bridge)
T_rel = rotations.relative_transform(segment_a, segment_b)   # (*, 2, 3) endpoint pairs
feats = rotations.se3_log(T_rel)                             # Lie-group features

# shortest angular distance between two orientations, in radians
angle = rotations.rotation_geodesic_distance(R1, R2)
```

Every one of these is drawn as a figure in the [Gallery](../gallery/index.md) (section 10), and the full signatures live in the [Rotations & SE(3) API](../api/rotations.md#se3-rigid-transforms).

!!! info "See also"
    [Choosing a Representation](choosing-rotations.md) — the decision guide · [Rotations & SE(3) API](../api/rotations.md) — every function · [Gallery](../gallery/index.md) — continuity and SE(3) figures
