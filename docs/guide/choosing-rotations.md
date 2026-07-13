# Choosing a Rotation Representation

pybvh supports five rotation representations. Each has trade-offs that make it better suited for different tasks. This page is the *decision guide* — the conversion API, shapes, and SE(3) layer live in [Rotation Representations & SE(3)](rotations.md).

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

The continuity difference in one picture — a smooth rotation crossing the ±180° boundary, read out in three representations:

![A smooth rotation crossing 180 degrees: the Euler readout jumps, the canonical quaternion flips the sign of z, 6D stays perfectly smooth](../gallery/img/rotation-continuity.png)

*Euler jumps, the canonical quaternion flips sign where `w` crosses zero, 6D stays smooth. ([Gallery](../gallery/index.md), section 2.)*

### Quaternions
Best for interpolation via `rotations.quat_slerp()`. Compact (4 values vs. 9 for matrices). However, quaternions have **double cover** (`q` and `-q` represent the same rotation), which can cause discontinuities during training. pybvh canonicalizes to `w >= 0`.

### Rotation matrices
Useful as an intermediate representation for composing rotations or for forward kinematics. Rarely used directly as ML features due to redundancy (9 values with 6 constraints).

### Axis-angle
Compact (3 values) and intuitive (direction = axis, magnitude = angle). Good for representing small perturbations or angular velocities. Discontinuous at `2*pi` wrapping.

## Converting between representations

Any pair converts directly — see [Rotation Representations & SE(3)](rotations.md) for the conversion API and shape conventions, and the [Gallery](../gallery/index.md) for the continuity figure that shows *why* 6D wins for training.

## Typical ML pipeline

```python
# Extract 6D features for training
root_pos, rot6d = bvh.to_6d()  # (F, J, 6)

# After model prediction, convert back
bvh_pred = bvh.from_6d(pred_root_pos, pred_rot6d)

# Evaluate with quaternion distance or rotation matrix comparison
_, q_gt = bvh.to_quat()
_, q_pred = bvh_pred.to_quat()
```
