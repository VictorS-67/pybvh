# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: pybvh
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Handling Rotations

# %% [markdown]
# BVH files store joint orientations as **Euler angles**: three rotation values applied around specific axes in a specific order. The file format uses degrees, but pybvh holds them in **radians** internally (on `bvh.joint_angles`) — the deg↔rad conversion happens at the I/O boundary. Euler angles are compact and human-readable, but they have well-known drawbacks:
#
# - **Gimbal lock** — when two axes align, one degree of freedom is lost.
# - **Discontinuities** — Euler angles can "jump" (e.g. from 179° to -179°), making them difficult for neural networks to learn.
# - **Order-dependent** — the same three numbers produce different rotations depending on the axis order.
#
# For these reasons, many applications convert Euler angles into other rotation representations before processing. The `pybvh.rotations` module provides conversions between the most commonly used representations:
#
# | Representation | Shape | Typical use case |
# |---|---|---|
# | Euler angles | `(*, 3)` | BVH file storage, human inspection |
# | Rotation matrices | `(*, 3, 3)` | Forward kinematics, composing rotations |
# | 6D rotation | `(*, 6)` | Neural network training (continuous, no singularities) |
# | Quaternions | `(*, 4)` | Smooth interpolation (SLERP), compact storage |
# | Axis-angle | `(*, 3)` | SMPL/SMPL-X body models, pose estimation |
#
# All conversions go through the **rotation matrix** as the central hub.

# %%
import numpy as np
np.set_printoptions(precision=4, suppress=True)

import pybvh
import matplotlib.pyplot as plt
from pathlib import Path

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "tutorials" else Path.cwd()
bvh_folder = REPO_ROOT / "bvh_data"
bvh = pybvh.read_bvh_file(bvh_folder / 'bvh_test1.bvh')

# %% [markdown]
# # What Euler angles actually do
#
# Euler angles decompose any 3D orientation into **three successive rotations around specific axes**. Each rotation is applied to the *already-rotated* frame — this is called an **intrinsic** rotation sequence, and it's the convention BVH files use.
#
# The animation below shows the z-x'-z" intrinsic sequence. Notice how each rotation axis belongs to the frame produced by the previous rotation, not the original fixed frame:
#
# ![Intrinsic Euler rotations: z-x'-z" sequence](assets/Euler2a.gif)
#
# *Image: [Juansempere / Xavax](https://commons.wikimedia.org/w/index.php?curid=24338647), CC BY-SA 3.0*
#
# In BVH files, the most common Euler orders are **ZYX**, **ZXY**, or **YXZ** — but each joint can have its own order. pybvh reads this from the `CHANNELS` line of each joint in the BVH hierarchy.
#
# The three drawbacks listed above — order dependence, gimbal lock, and discontinuities — are all consequences of this decomposition. Let's illustrate two of them before moving on. (Order dependence is demonstrated later when we look at [how BVH stores rotations](#How-BVH-stores-rotations:-Euler-angles).)

# %% [markdown]
# ## Pitfall: gimbal lock
#
# When the middle rotation in a three-axis Euler sequence (like ZYX) reaches ±90°, the first and third rotation axes become **parallel** — they now rotate around the same axis, and the system loses one degree of freedom. This is called **gimbal lock**. (BVH files always use three distinct axes, so the singularity is always at ±90° on the middle axis.)
#
# ![Gimbal lock: two gimbals aligning](assets/Gimbal_Lock_Plane.gif)
#
# *When the pitch (green) and yaw (magenta) gimbals align, changes to roll (blue) and yaw (magenta) produce the same rotation — the airplane can no longer distinguish between them. (Image: [Drummyfish](https://commons.wikimedia.org/w/index.php?curid=77738933), CC0)*
#
# Let's see this numerically. With a ZYX Euler order, gimbal lock occurs when Y = 90°. At that point, changing X and Z by equal-and-opposite amounts produces the **exact same rotation matrix**:

# %%
# At gimbal lock (Y = 90° in ZYX order), X and Z rotations become coupled.
# Only the *difference* (Z - X) affects the result — infinitely many (X, Z) pairs
# give the same rotation. Let's verify: all three sets below have Z - X = 30°.
gimbal_cases = np.array([
    [ 0.0, 90.0, 30.0],
    [20.0, 90.0, 50.0],
    [45.0, 90.0, 75.0],
])

Rs = pybvh.rotations.euler_to_rotmat(gimbal_cases, 'ZYX', degrees=True)

print("Three different Euler angle sets, all with Y = 90° and (Z - X) = 30°:\n")
for i in range(3):
    print(f"  (X={gimbal_cases[i, 0]:5.1f}°, Y={gimbal_cases[i, 1]:5.1f}°, Z={gimbal_cases[i, 2]:5.1f}°)")
print()
print(f"Same rotation matrix? {np.allclose(Rs[0], Rs[1]) and np.allclose(Rs[1], Rs[2])}")
print(f"\nThe rotation matrix (all three are identical):\n{Rs[0]}")

# %% [markdown]
# Three completely different Euler angle triplets, one single rotation. This is the degree of freedom that gimbal lock "eats": at the singularity, you can freely redistribute rotation between X and Z without changing the physical orientation. Rotation matrices, quaternions, and 6D representations don't have this problem — they represent every rotation uniquely (up to quaternion sign).

# %% [markdown]
# ## Pitfall: discontinuities
#
# Now let's see the third Euler drawback — the discontinuity problem. Consider a smooth rotation that crosses the 180° wraparound point:

# %%
# A smooth rotation around Z that goes from 170° to 190° — no jumps in the input
input_angles = np.linspace(170, 190, 41)
euler = np.zeros((41, 3))
euler[:, 0] = input_angles  # Z component in ZYX order

# Convert to all representations and back
rotmats = pybvh.rotations.euler_to_rotmat(euler, 'ZYX', degrees=True)
recovered_euler = pybvh.rotations.rotmat_to_euler(rotmats, 'ZYX', degrees=True)
quats = pybvh.rotations.euler_to_quat(euler, 'ZYX', degrees=True)
rot6d = pybvh.rotations.euler_to_rot6d(euler, 'ZYX', degrees=True)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].plot(input_angles, recovered_euler[:, 0], 'o-', color='C3')
axes[0].set_title('Euler Z (recovered)')
axes[0].set_xlabel('input angle (deg)')
axes[0].set_ylabel('Z component (deg)')
axes[0].axvline(180, color='gray', linestyle=':', alpha=0.5)

axes[1].plot(input_angles, quats[:, 0], 'o-', label='w')
axes[1].plot(input_angles, quats[:, 3], 'o-', label='z')
axes[1].set_title('Quaternion (canonical, w \u2265 0)')
axes[1].set_xlabel('input angle (deg)')
axes[1].axvline(180, color='gray', linestyle=':', alpha=0.5)
axes[1].legend()

axes[2].plot(input_angles, rot6d[:, 0], 'o-', label='[0]')
axes[2].plot(input_angles, rot6d[:, 3], 'o-', label='[3]')
axes[2].set_title('6D representation')
axes[2].set_xlabel('input angle (deg)')
axes[2].axvline(180, color='gray', linestyle=':', alpha=0.5)
axes[2].legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# Three things to notice:
#
# - **Euler** jumps from `+180°` to `-180°` exactly at 180° — even though the rotation itself is smooth, the *representation* breaks.
# - **Quaternion** also has a discontinuity at 180° — the `z` component flips sign. This is because pybvh enforces a canonical form (`w ≥ 0`) to resolve the double-cover ambiguity, but that canonicalization introduces its own jump when `w` passes through zero.
# - **6D** is perfectly smooth across the boundary. This is the property that makes 6D the preferred representation for neural network outputs: a small change in the rotation always corresponds to a small change in the 6D vector, which is what gradient-based optimization needs.
#
# We'll come back to 6D's other key advantage — projection of noisy outputs — later in this tutorial.

# %% [markdown]
# # How BVH stores rotations: Euler angles
#
# Let's look at how Euler angles are actually laid out in a `Bvh` object.

# %%
# Each joint has a rotation order, stored in rot_channels
for node in bvh.nodes[:3]:
    if not node.is_end_site():
        print(f"{node.name:20s} rotation order: {node.rot_channels}")

for node in bvh.nodes[9:12]:
    if not node.is_end_site():
        print(f"{node.name:20s} rotation order: {node.rot_channels}")

# %% [markdown]
# Each joint has its own **rotation order**, stored in the `rot_channels` property. Different joints in the same file can have different orders — this is a choice made by the motion capture software when exporting the BVH, typically placing the axis with the largest expected range of motion first to minimize gimbal lock risk for that joint. Motion data is stored as two arrays:
#
# - `root_pos` — `(F, 3)`: root translation per frame
# - `joint_angles` — `(F, J, 3)`: Euler angles in **radians** per joint per frame

# %%
print(f"root_pos shape:     {bvh.root_pos.shape}")
print(f"joint_angles shape: {bvh.joint_angles.shape}")
print(f"\nFirst frame, first 3 joints (Euler angles in radians):")
print(bvh.joint_angles[0, :3])
print(f"\nSame values in degrees (for display):")
print(np.rad2deg(bvh.joint_angles[0, :3]))

# %% [markdown]
# ## Why the rotation order matters
#
# The same three angle values produce a **completely different orientation** depending on the order in which the rotations are applied. This is one of the most common sources of bugs when working with BVH files.

# %%
angles = np.array([30.0, 45.0, 60.0])  # same 3 numbers, two different orders

R_zyx = pybvh.rotations.euler_to_rotmat(angles, 'ZYX', degrees=True)
R_xyz = pybvh.rotations.euler_to_rotmat(angles, 'XYZ', degrees=True)

print("R with order ZYX:")
print(R_zyx)
print("\nR with order XYZ:")
print(R_xyz)
print("\nDifferent matrices? ", not np.allclose(R_zyx, R_xyz))

# %% [markdown]
# Always use the order from `rot_channels` when working with a joint's angles, never assume.

# %% [markdown]
# # Rotation matrices: the central hub
#
# A 3×3 rotation matrix $R$ is an orthogonal matrix with $\det(R) = +1$. It is the most general and unambiguous representation: every 3D rotation corresponds to exactly one rotation matrix — no singularities, no sign ambiguity.
#
# All other representations in pybvh go **through** the rotation matrix internally: Euler ↔ Rotmat ↔ 6D, Euler ↔ Rotmat ↔ Quaternion, etc.

# %% [markdown]
# ## Euler ↔ Rotation matrix

# %%
angles_deg = np.array([30.0, 45.0, 60.0])
order = 'ZYX'

# Forward: Euler -> Rotation matrix
R = pybvh.rotations.euler_to_rotmat(angles_deg, order, degrees=True)
print("Rotation matrix:")
print(R)

# Backward: Rotation matrix -> Euler
recovered = pybvh.rotations.rotmat_to_euler(R, order, degrees=True)
print(f"\nRecovered Euler angles: {recovered}")

# %% [markdown]
# The `degrees` argument controls whether Euler angles are in degrees (`True`) or radians (`False`). It only applies to the Euler side — rotation matrices have no units.
#
# You can also extract Euler angles in a *different* order. The angles will be different but represent the same physical rotation:

# %%
# Decompose the same rotation matrix in XYZ order instead of ZYX
angles_xyz = pybvh.rotations.rotmat_to_euler(R, 'XYZ', degrees=True)
print(f"Same rotation as ZYX: {recovered}")
print(f"Same rotation as XYZ: {angles_xyz}")

# Verify both produce the same rotation matrix
R_check = pybvh.rotations.euler_to_rotmat(angles_xyz, 'XYZ', degrees=True)
print(f"Both give the same R? {np.allclose(R, R_check)}")

# %% [markdown]
# ## Batch operations
#
# All functions in `pybvh.rotations` are **fully vectorized**. They accept arrays of any shape `(*, 3)` and return arrays of shape `(*, 3, 3)`. For a 1000-frame motion with 24 joints, this is the only practical way to convert — pure-Python loops would be orders of magnitude slower.

# %%
batch_angles = np.array([
    [0, 0, 0],
    [30, 45, 60],
    [90, 0, 0],
    [-45, 90, 30],
], dtype=float)

batch_R = pybvh.rotations.euler_to_rotmat(batch_angles, 'ZYX', degrees=True)
print(f"Input shape:  {batch_angles.shape}")
print(f"Output shape: {batch_R.shape}")

# %% [markdown]
# # The other representations
#
# With rotation matrices as the central hub, the other three representations (6D, quaternion, axis-angle) follow the same pattern: there's a forward function `euler_to_X()` and a backward function `X_to_euler()`. Each goes through the rotation matrix internally.
#
# What differs between them is **what they're good for**.

# %% [markdown]
# ## 6D representation
#
# Introduced by [Zhou et al. (CVPR 2019)](https://arxiv.org/abs/1812.07035) for neural network training. The 6D vector is just the **first two columns of the rotation matrix** flattened. Its key property is **continuity**: nearby rotations always map to nearby 6D vectors (we already saw this above). This makes it the standard choice for ML model outputs that predict rotations.

# %%
angles_deg = np.array([30.0, 45.0, 60.0])

rot6d = pybvh.rotations.euler_to_rot6d(angles_deg, 'ZYX', degrees=True)
print(f"6D vector: {rot6d}")

recovered = pybvh.rotations.rot6d_to_euler(rot6d, 'ZYX', degrees=True)
print(f"Recovered: {recovered}")

# %% [markdown]
# ## Quaternions
#
# A unit quaternion $q = (w, x, y, z)$ with $\|q\| = 1$ encodes a rotation in 4 numbers. Quaternions are the default representation in many game engines and graphics libraries, and they support smooth spherical interpolation (**SLERP**) between two rotations.
#
# pybvh uses the **scalar-first** convention `(w, x, y, z)` and enforces a **canonical form** where `w ≥ 0` to resolve the double-cover ambiguity (since `q` and `-q` represent the same rotation).

# %%
q = pybvh.rotations.euler_to_quat(angles_deg, 'ZYX', degrees=True)
print(f"Quaternion (w, x, y, z): {q}")
print(f"Norm: {np.linalg.norm(q):.6f}")  # always 1.0

recovered = pybvh.rotations.quat_to_euler(q, 'ZYX', degrees=True)
print(f"Recovered: {recovered}")

# %% [markdown]
# ## Axis-angle
#
# The axis-angle representation encodes a rotation as a single 3D vector $\mathbf{v}$ where:
# - The **direction** $\hat{\mathbf{v}} = \mathbf{v} / \|\mathbf{v}\|$ is the rotation axis.
# - The **magnitude** $\|\mathbf{v}\|$ is the rotation angle in radians.
# - The **zero vector** $[0, 0, 0]$ represents no rotation.
#
# This is the representation used by **SMPL** and **SMPL-X** body models, and by many pose estimation pipelines.

# %%
aa = pybvh.rotations.euler_to_axisangle(angles_deg, 'ZYX', degrees=True)
angle_rad = np.linalg.norm(aa)
axis = aa / angle_rad

print(f"Axis-angle vector: {aa}")
print(f"Rotation axis:     {axis}")
print(f"Rotation angle:    {np.degrees(angle_rad):.2f}°")

recovered = pybvh.rotations.axisangle_to_euler(aa, 'ZYX', degrees=True)
print(f"Recovered: {recovered}")

# %% [markdown]
# # Why 6D wins for ML: the projection property
#
# We saw earlier that 6D is smooth across the 180° boundary. There's a second reason 6D is the preferred representation for neural network outputs: any 6 numbers — even noisy or arbitrary ones — can be projected onto a valid rotation matrix.
#
# When a network predicts a 6D vector, the result is rarely "clean". `rot6d_to_rotmat()` applies Gram-Schmidt orthonormalization to produce the closest valid rotation matrix. The same is **not** true for quaternions: a 4-vector with arbitrary components is not a valid rotation unless its norm is exactly 1, and you have to manually renormalize.

# %%
angles = np.array([30.0, 45.0, 60.0])

# Start with a valid rotation
R_valid = pybvh.rotations.euler_to_rotmat(angles, 'ZYX', degrees=True)
rot6d_valid = pybvh.rotations.rotmat_to_rot6d(R_valid)

# Add noise (simulating a network's imperfect output)
rng = np.random.default_rng(42)
rot6d_noisy = rot6d_valid + rng.normal(scale=0.1, size=6)

# Convert noisy 6D back to a rotation matrix — pybvh applies Gram-Schmidt
R_recovered = pybvh.rotations.rot6d_to_rotmat(rot6d_noisy)

print(f"Determinant of recovered R: {np.linalg.det(R_recovered):.6f}  (must be 1 for a rotation)")
print(f"Is orthogonal? {np.allclose(R_recovered @ R_recovered.T, np.eye(3), atol=1e-6)}")

# The recovered rotation is close but not identical to the original due to the noise
angles_recovered = pybvh.rotations.rotmat_to_euler(R_recovered, order='ZYX', degrees=True)
print(f"\nOriginal angles:  {angles}")
print(f"Recovered angles: {angles_recovered}")

# %% [markdown]
# The recovered matrix is a valid rotation (orthogonal, determinant 1) even though the input was noisy. The recovered angles are *close* to the original (30, 45, 60) but not identical — Gram-Schmidt projection finds the nearest valid rotation matrix, it doesn't magically undo the noise. This is exactly what you want during training: the network's output is always a legal rotation, and the loss function can measure how far it is from the target.
#
# Compare with quaternions:

# %%
q_valid = pybvh.rotations.euler_to_quat(angles, 'ZYX', degrees=True)
q_noisy = q_valid + rng.normal(scale=0.1, size=4)

print(f"||q_noisy|| = {np.linalg.norm(q_noisy):.4f}  (a valid unit quaternion has norm exactly 1)")

# %% [markdown]
# With a noisy quaternion, you'd have to manually renormalize, which is one extra step in your training loop and a source of subtle bugs. With 6D, projection is built into the conversion.

# %% [markdown]
# # Working with entire BVH motions
#
# So far we've used the low-level `pybvh.rotations` functions on small NumPy arrays. In practice, you'll want to convert an entire BVH animation at once. The `Bvh` class provides convenience methods that handle every joint's individual Euler order and return nicely organized arrays.
#
# All `to_*` methods return a tuple `(root_pos, joint_data)`:
# - `root_pos` — `(F, 3)`: root translation per frame
# - `joint_data` — `(F, J, *)`: rotation data per joint per frame (last dim depends on representation)

# %%
# Get the same animation in all four representations
_, rotmats = bvh.to_rotmat()
_, rot6d = bvh.to_6d()
_, quats = bvh.to_quat()
_, axisang = bvh.to_axisangle()

print(f"Rotation matrices: {rotmats.shape}")
print(f"6D:                {rot6d.shape}")
print(f"Quaternions:       {quats.shape}")
print(f"Axis-angle:        {axisang.shape}")

# %% [markdown]
# ## Round-trip verification
#
# The corresponding `from_*` methods write rotation data back into a `Bvh` object. Going **out** to any representation and **back** should preserve the motion exactly (within float precision). Let's verify this on real data by checking that the spatial coordinates of the skeleton are preserved through the round trip:

# %%
spatial_before = bvh.node_positions()

results = {}
for name, get_fn_name, set_fn_name in [
    ("6D",         'to_6d',         'from_6d'),
    ("quaternion", 'to_quat',        'from_quat'),
    ("axis-angle", 'to_axisangle',  'from_axisangle'),
]:
    test = bvh.copy()
    root_pos, joint_data = getattr(test, get_fn_name)()
    getattr(test, set_fn_name)(root_pos, joint_data)
    err = np.max(np.abs(spatial_before - test.node_positions()))
    results[name] = err

print(f"{'Representation':<15} {'Max position error':<20}")
print("-" * 35)
for name, err in results.items():
    print(f"{name:<15} {err:.2e}")

# %% [markdown]
# All round-trip errors are at machine precision (10⁻¹⁵ or zero). You can freely convert your data between representations without worrying about losing information.

# %% [markdown]
# # Changing Euler orders
#
# Different BVH files (and even different joints within the same file) may use different Euler orders. pybvh has two methods that let you change Euler orders, in both cases preserving the physical rotations exactly (only the *decomposition* changes):
#
# - `change_euler_order(order, joint=name)` — change one joint's order.
# - `change_euler_order(order)` — unify all joints to the same order.

# %%
# Change one joint's order
print(f"Hips order before: {bvh.root.rot_channels}")
print(f"Hips angles (frame 0): {bvh.joint_angles[0, 0]}")

bvh_new = bvh.change_euler_order('XYZ', joint='Hips', inplace=False)

print(f"\nHips order after:  {bvh_new.root.rot_channels}")
print(f"Hips angles (frame 0): {bvh_new.joint_angles[0, 0]}")
print("\nThe angles look completely different, but they represent the same rotation.")

# %%
# Unify all joints to a single order
bvh_unified = bvh.change_euler_order('XYZ', inplace=False)

print("Right arms joints — orders before vs after:")
for orig, new in zip(bvh.nodes[9:13], bvh_unified.nodes[9:13]):
    print(f"  {orig.name:15s}  {orig.rot_channels} -> {new.rot_channels}")

# Verify the physical motion is preserved
spatial_orig = bvh.node_positions()
spatial_unified = bvh_unified.node_positions()
print(f"\nSpatial coordinates preserved? {np.allclose(spatial_orig, spatial_unified, atol=1e-6)}")

# %% [markdown]
# # Summary of the conversion API
#
# ## Low-level functions (`pybvh.rotations`)
#
# | Function | Input → Output |
# |---|---|
# | `euler_to_rotmat(angles, order, degrees)` | `(*, 3)` → `(*, 3, 3)` |
# | `rotmat_to_euler(R, order, degrees)` | `(*, 3, 3)` → `(*, 3)` |
# | `rotmat_to_rot6d(R)` | `(*, 3, 3)` → `(*, 6)` |
# | `rot6d_to_rotmat(rot6d)` | `(*, 6)` → `(*, 3, 3)` |
# | `rotmat_to_quat(R)` | `(*, 3, 3)` → `(*, 4)` |
# | `quat_to_rotmat(q)` | `(*, 4)` → `(*, 3, 3)` |
# | `rotmat_to_axisangle(R)` | `(*, 3, 3)` → `(*, 3)` |
# | `axisangle_to_rotmat(aa)` | `(*, 3)` → `(*, 3, 3)` |
#
# **Convenience wrappers** (go through rotmat internally): `euler_to_rot6d`, `rot6d_to_euler`, `euler_to_quat`, `quat_to_euler`, `euler_to_axisangle`, `axisangle_to_euler`.
#
# ## Bvh convenience methods
#
# | Method | Description |
# |---|---|
# | `to_rotmat()` | Euler → rotation matrices `(F, J, 3, 3)` |
# | `to_6d()` | Euler → 6D `(F, J, 6)` |
# | `to_quat()` | Euler → quaternions `(F, J, 4)` |
# | `to_axisangle()` | Euler → axis-angle `(F, J, 3)` |
# | `from_rotmat(root_pos, joint_rotmats)` | Rotation matrices → Euler (writes back into frames) |
# | `from_6d(root_pos, joint_rot6d)` | 6D → Euler |
# | `from_quat(root_pos, joint_quats)` | Quaternion → Euler |
# | `from_axisangle(root_pos, joint_aa)` | Axis-angle → Euler |
# | `change_euler_order(order, joint=name, inplace)` | Change one joint's Euler order |
# | `change_euler_order(order, inplace)` | Unify all joints to one Euler order |

# %% [markdown]
# # What's next
#
# This tutorial covered the four main rotation representations, why each one is useful, and how to convert between them safely.
#
# - **Tutorial 4 — Visualization**: video export, interactive playback, camera control
# - **Tutorial 5 — Transforms**: data augmentation (mirroring, rotation, noise, speed)
# - **Tutorial 6 — Features**: velocities, foot contacts, feature arrays for ML
# - **Tutorial 7 — Batch processing**: directories, harmonization, dataset preparation
