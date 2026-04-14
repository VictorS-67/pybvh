# Choosing a Rotation Representation

pybvh supports five rotation representations. Each has trade-offs that make it better suited for different tasks.

## Decision table

| Representation | Shape | Continuous? | Best for |
|----------------|-------|-------------|----------|
| **Euler angles** | `(*, 3)` | No (gimbal lock, discontinuities) | BVH native format, human-readable inspection, storage |
| **Rotation matrices** | `(*, 3, 3)` | Yes | Forward kinematics, composing rotations, intermediate conversions |
| **6D (Zhou et al.)** | `(*, 6)` | Yes | Neural network output layers (no discontinuity penalty during training) |
| **Quaternions** | `(*, 4)` | No (antipodal equivalence) | Smooth interpolation (SLERP), compact storage, physics engines |
| **Axis-angle** | `(*, 3)` | No (wraps at 2pi) | Small rotations, angular velocity, rotation visualization |

## When to use each

### Euler angles
The native BVH format. Use when reading/writing files or when you need human-readable rotation values. Avoid for ML training (discontinuities at gimbal lock cause gradient problems) and for interpolation (linear interpolation in Euler space is not physically meaningful).

### 6D rotation (Zhou et al.)
The recommended representation for neural networks. It is **continuous** — nearby rotations map to nearby 6D vectors — so gradient-based optimization works smoothly. Use `bvh.to_6d()` for feature extraction and `bvh.from_6d()` to convert predictions back.

### Quaternions
Best for interpolation via `rotations.quat_slerp()`. Compact (4 values vs. 9 for matrices). However, quaternions have **double cover** (`q` and `-q` represent the same rotation), which can cause discontinuities during training. pybvh canonicalizes to `w >= 0`.

### Rotation matrices
Useful as an intermediate representation for composing rotations or for forward kinematics. Rarely used directly as ML features due to redundancy (9 values with 6 constraints).

### Axis-angle
Compact (3 values) and intuitive (direction = axis, magnitude = angle). Good for representing small perturbations or angular velocities. Discontinuous at `2*pi` wrapping.

## Converting between representations

```python
from pybvh import rotations

# Any pair — direct or via convenience wrappers
R = rotations.euler_to_rotmat(angles, order="ZYX", degrees=True)
q = rotations.rotmat_to_quat(R)
rot6d = rotations.euler_to_rot6d(angles, "ZYX", degrees=True)
```

All functions support arbitrary batch dimensions: `(3,)`, `(N, 3)`, `(F, J, 3)`.

## Typical ML pipeline

```python
# Extract 6D features for training
root_pos, rot6d, joints = bvh.to_6d()  # (F, J, 6)

# After model prediction, convert back
bvh_pred = bvh.from_6d(pred_root_pos, pred_rot6d)

# Evaluate with quaternion distance or rotation matrix comparison
_, q_gt, _ = bvh.to_quaternions()
_, q_pred, _ = bvh_pred.to_quaternions()
```
