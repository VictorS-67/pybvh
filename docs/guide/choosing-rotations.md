# Choosing a Rotation Representation

pybvh supports five rotation representations. Each has trade-offs that make it better suited for different tasks. This page is the *decision guide*. The conversion API, the shapes, and the SE(3) layer are in [Rotation Representations & SE(3)](rotations.md).

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
Euler angles are the native BVH format. Use them when you read or write files, or when you need human-readable rotation values. Avoid them in two cases:

- ML training: the discontinuities at gimbal lock cause gradient problems.
- Interpolation: linear interpolation in Euler space is not physically meaningful.

### 6D rotation (Zhou et al.)
6D is the recommended representation for neural networks. It is **continuous**: nearby rotations map to nearby 6D vectors. Because of this, gradient-based optimization works smoothly. Use `bvh.to_6d()` for feature extraction. Use `bvh.from_6d()` to convert predictions back.

One picture shows the difference in continuity. A smooth rotation crosses the ±180° boundary, and the picture shows its readout in three representations:

![A smooth rotation crossing 180 degrees: the Euler readout jumps, the canonical quaternion flips the sign of z, 6D stays perfectly smooth](../gallery/img/rotation-continuity.png)

*Euler jumps. The canonical quaternion flips sign where `w` crosses zero. 6D stays smooth. ([Gallery](../gallery/index.md), section 2.)*

### Quaternions
Quaternions are best for interpolation, with `rotations.quat_slerp()`. They are compact: 4 values, against 9 for matrices. However, quaternions have **double cover**: `q` and `-q` represent the same rotation. Double cover can cause discontinuities during training. pybvh canonicalizes to `w >= 0`.

### Rotation matrices
Rotation matrices are useful as an intermediate representation for composing rotations or for forward kinematics. They are rarely used directly as ML features, because they are redundant: 9 values with 6 constraints.

### Axis-angle
Axis-angle is compact (3 values) and intuitive (direction = axis, magnitude = angle). It is good for representing small perturbations or angular velocities. It is discontinuous at `2*pi` wrapping.

## Converting between representations

Any pair converts directly. For the conversion API and the shape conventions, see [Rotation Representations & SE(3)](rotations.md). For the continuity figure that shows *why* 6D wins for training, see the [Gallery](../gallery/index.md).

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
