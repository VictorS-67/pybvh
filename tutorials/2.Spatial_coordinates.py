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
# # Spatial Coordinates and Skeleton Operations

# %% [markdown]
# In Tutorial 1 we saw that BVH files store **joint rotations**, not 3D positions. But for many tasks — plotting, computing distances, feeding into ML models — we need the actual 3D position of each joint in space.
#
# The process of converting rotations into positions is called **forward kinematics**: starting from the root, we walk down the skeleton tree, applying each joint's rotation to its bone offset, accumulating the result to get every joint's world position.
#
# pybvh handles this with a single method: `node_positions()`. This tutorial covers how to use it, how to control the coordinate frame, and how to manipulate the skeleton itself.

# %%
import numpy as np
np.set_printoptions(precision=4, suppress=True)

import pybvh
import matplotlib.pyplot as plt
from pathlib import Path

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "tutorials" else Path.cwd()
bvh_folder = REPO_ROOT / "bvh_data"
bvh = pybvh.read_bvh_file(bvh_folder / 'bvh_test1.bvh')
print(bvh)

# %% [markdown]
# # Getting spatial coordinates
#
# The method `node_positions()` runs forward kinematics and returns a NumPy array of 3D positions for every node at every frame.

# %%
coords = bvh.node_positions()

print(f"joint_angles shape:  {bvh.joint_angles.shape}  — {bvh.joint_count} joints")
print(f"spatial_coord shape: {coords.shape}  — {coords.shape[1]} nodes")

# %% [markdown]
# Notice that the spatial coordinates have **more columns** than `joint_angles`. That's because they include the **end sites** — leaf nodes that have no rotation channels but do have a position (the tip of a terminal bone, like the top of the head or the end of a toe). The `joint_angles` array only has entries for joints that rotate.

# %% [markdown]
# ## Looking up joints by name
#
# To find which column index corresponds to which joint, use `bvh.node_index` — a dict mapping joint names to integer indices.
#
# There's a subtle thing to know: `Bvh` exposes **two** index dicts, because `joint_angles` and `node_positions()` have different axis-1 lengths (end sites have no rotation channels, so they appear in one array but not the other):
#
# | Array | Shape | Lookup |
# |---|---|---|
# | `bvh.node_positions()` return | `(F, N, 3)` — N = all nodes, including end sites | `bvh.node_index[name]` |
# | `bvh.joint_angles` | `(F, J, 3)` — J = joints only, no end sites | `bvh.joint_index[name]` |
#
# Pick the property whose name matches the array you're indexing. For joints that aren't end sites, both dicts contain the name — but they return **different** integers (offset by the number of preceding end sites in the depth-first node walk).

# %%
hips_idx = bvh.node_index['Hips']
head_idx = bvh.node_index['Head']

print(f"Hips 3D position (frame 0): {coords[0, hips_idx]}")
print(f"Head 3D position (frame 0): {coords[0, head_idx]}")

# %% [markdown]
# ## Single-frame extraction
#
# If you only need one frame, pass `frame` to avoid computing the entire sequence (negative indices count from the end, NumPy-style). The result is a 2D array `(num_nodes, 3)` instead of 3D.

# %%
frame_15 = bvh.node_positions(frame=15)
print(f"Single frame shape: {frame_15.shape}")

fig, ax = bvh.plot_frame(frame_15)
# note that this code does the exact same thing:
# fig, ax = bvh.plot_frame(frame=15)

plt.show()

# %% [markdown]
# # Centering modes
#
# By default, `node_positions()` returns **world coordinates** — the positions as recorded in the BVH file. But depending on your application, you may want the coordinates in a different frame of reference.
#
# The `centered` parameter controls this:
#
# - **`"world"`** (default): absolute positions from the BVH file. The skeleton is wherever it was recorded.
# - **`"first"`**: ground-plane centering — the first frame's root is shifted over the origin in the two horizontal axes, keeping its original height above the ground. The skeleton still moves from there.
# - **`"skeleton"`**: the root is at `[0, 0, 0]` in **every** frame. Only the pose changes, not the global position.
#
# The best way to understand this is visually:

# %% tags=["skip-execution"]
bvh.play(centered="world", labels=["world"])

# %% tags=["skip-execution"]
# Cell 2
bvh.play(centered="first", labels=["first"])

# %% tags=["skip-execution"]
# Cell 3
bvh.play(centered="skeleton", labels=["skeleton"])

# %%
for mode in ["world", "first", "skeleton"]:
    fig, ax = bvh.plot_frame(frame=15, centered=mode)
    ax.set_title(f'centered="{mode}"')
    plt.show()

# %% [markdown]
# We can also verify numerically: with `"skeleton"` centering, the root is always at the origin.

# %%
coords_skel = bvh.node_positions(centered="skeleton")

print("Root position with 'skeleton' centering (first 5 frames):")
print(coords_skel[:5, 0])  # all zeros

# %% [markdown]
# **When to use each mode:**
# - `"world"` — when you need the original recording positions (e.g., analyzing room-scale trajectories).
# - `"first"` — when you want the motion to start over the origin (at its natural height) but still move naturally (common for ML training data).
# - `"skeleton"` — when you only care about the body pose, not the global position (e.g., pose classification).

# %% [markdown]
# If your dataset mixes files with different up-axis conventions (some Y-up, some Z-up), you can unify them with `bvh.reorient_world_up('+y')` — this rotates the entire animation into a new coordinate system without changing how the character looks. See the [World Up guide](https://victors-67.github.io/pybvh/guide/world-up/) for details.

# %% [markdown]
# # Skeleton operations
#
# Different BVH files often come from different performers with different body proportions. If you're comparing motions across performers, or feeding data into an ML model that expects consistent bone lengths, you need to **normalize** the skeleton.
#
# pybvh provides three operations for this: scaling bone lengths, copying bone proportions from a reference skeleton, and extracting a subset of joints.

# %% [markdown]
# ## Scaling the skeleton
#
# `scale()` multiplies all bone offsets by a factor. This changes the skeleton's overall size without affecting the motion (rotations stay the same).
#
# You can pass a single number for uniform scaling, or `[sx, sy, sz]` for per-axis scaling.

# %%
small = bvh.scale(0.5)

fig, axes = pybvh.bvhplot.frame([bvh, small], frame=15,
                                 labels=['Original', 'Scaled 0.5x'])
plt.show()

# %%
# Verify: offsets are halved
print(f"Original Spine offset: {bvh.nodes[1].offset}")
print(f"Scaled Spine offset:   {small.nodes[1].offset}")

# %% [markdown]
# ## Retargeting to a reference skeleton
#
# `retarget()` copies the bone offsets from another BVH object. The motion (joint angles) stays the same, but the body proportions change to match the reference. Both skeletons must have the **same hierarchy** (same joints in the same parent-child relationships).
#
# This is how you normalize a dataset: pick one skeleton as the reference and retarget all other files to it.

# %%
# First, let's create a visibly different skeleton by scaling
tall = bvh.scale(1.5)

# Now retarget the tall skeleton's motion onto the original body proportions
retargeted = tall.retarget(bvh)

fig, axes = pybvh.bvhplot.frame([tall, retargeted], frame=15,
                                 labels=['Tall (1.5x)', 'Retargeted to original'])
plt.show()

# %%
# In practice, you'd load a reference skeleton from a file:
reference = pybvh.read_bvh_file(bvh_folder / 'standard_skeleton.bvh')
normalized = bvh.retarget(reference)

print(f"Original Spine offset:   {bvh.nodes[1].offset}")
print(f"Reference Spine offset:  {reference.nodes[1].offset}")
print(f"Normalized Spine offset: {normalized.nodes[1].offset}")

# %% [markdown]
# ## Extracting a subset of joints
#
# Sometimes you don't need all joints. Some ML models work with a reduced set of major joints (hips, knees, shoulders, etc.) and ignore spine subdivisions or toes.
#
# `extract_joints()` creates a new Bvh with only the specified joints. Removed joints' bone offsets are collapsed into their nearest kept descendant, so the rest pose geometry stays consistent. The root must always be included.

# %%
print(f"Original: {bvh.joint_count} joints — {bvh.joint_names}")

major_joints = ['Hips', 'Spine3', 'Head', 'RightArm', 'RightHand',
                'LeftArm', 'LeftHand', 'RightUpLeg', 'RightFoot',
                'LeftUpLeg', 'LeftFoot']

reduced = bvh.extract_joints(major_joints)
print(f"Reduced:  {reduced.joint_count} joints — {reduced.joint_names}")

# %%
fig, axes = pybvh.bvhplot.frame([bvh, reduced], frame=15,
                                 labels=['Original (24 joints)', 'Reduced (11 joints)'])
plt.show()

# %% [markdown]
# # What's next
#
# This tutorial covered how to get 3D joint positions from BVH rotation data, how centering modes affect the coordinate frame, and how to manipulate the skeleton itself.
#
# - **Tutorial 3 — Rotations**: convert between rotation representations (Euler, quaternion, 6D, axis-angle)
# - **Tutorial 4 — Visualization**: video export, interactive playback, camera control
# - **Tutorial 5 — Transforms**: data augmentation (mirroring, rotation, noise, speed)
