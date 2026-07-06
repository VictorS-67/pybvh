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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Introduction to pybvh

# %% [markdown]
# # What is a BVH file?
#
# **BVH** (Biovision Hierarchy) is a plain-text file format for storing motion capture data. It is one of the most widely used formats in animation, biomechanics, and human motion research. You will encounter BVH files when working with motion capture datasets (CMU MoCap, KIT, HumanAct12), animation tools (Blender, Mixamo), or pose estimation pipelines.
#
# A BVH file has two sections:
#
# 1. **HIERARCHY** — defines the skeleton as a tree of joints. Each joint has a 3D offset from its parent (the bone length and direction at rest) and a rotation order (e.g., ZYX). The root joint also has position channels.
#
# 2. **MOTION** — contains per-frame data. Each frame is one line of numbers: the root's 3D position followed by the Euler angle rotations for every joint.
#
# Careful: a BVH file stores **joint rotations**, not 3D positions. To get the actual 3D positions of each joint, you need to apply **forward kinematics** — walking down the skeleton tree and composing rotations and offsets. pybvh handles this for you.
#
# Let's look at the first few lines of an actual BVH file to see what this looks like in practice:

# %%
import numpy as np
np.set_printoptions(precision=4, suppress=True)

import pybvh
import matplotlib.pyplot as plt
from pathlib import Path

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "tutorials" else Path.cwd()
bvh_folder = REPO_ROOT / "bvh_data"
output_folder = Path('./output')
output_folder.mkdir(exist_ok=True)

# %%
# Let's peek at the raw text of a BVH file
bvh_path = bvh_folder / 'bvh_test1.bvh'
with open(bvh_path) as f:
    lines = f.readlines()

print("=== HIERARCHY section (first 15 lines) ===")
for line in lines[:15]:
    print(line, end='')

print("\n\n=== MOTION section (first 5 lines) ===")
for i, line in enumerate(lines):
    if line.strip() == 'MOTION':
        for line in lines[i:i+5]:
            print(line, end='')
        break

# %% [markdown]
# You can see the tree structure in the HIERARCHY: the root joint (Hips) contains child joints (Spine), each with their own `OFFSET` (bone vector from parent) and `CHANNELS` (which axes are animated). Notice how the OFFSET of the SPINE is `0.00000 0.00000 4.45283`? It means that at rest pose (when all joint angles are 0; usually an A-pose or a T-pose), the SPINE's distance from its parent ROOT is, in each direction:
# - x-axis: `0.00000`
# - y-axis: `0.00000`
# - z-axis: `4.45283`
#
# So we can safely assume that the Z-axis in this file is the upward direction.
#
# The MOTION section is just rows of numbers — one row per frame. 
#
# Parsing this by hand is tedious and error-prone. That's what pybvh does for you.

# %% [markdown]
# # Loading a BVH file
#
# The function `read_bvh_file` parses a `.bvh` file and returns a `Bvh` object — the central container in pybvh. A `Bvh` object holds two things: the **skeleton** (a hierarchy of joints with their bone offsets and rotation orders) and the **motion** (per-frame angles for every joint).

# %%
bvh = pybvh.read_bvh_file(bvh_folder / 'bvh_test1.bvh')
print(bvh)

# %%
print(f"Number of joints:   {bvh.joint_count}")
print(f"Number of frames:   {bvh.frame_count}")
print(f"Seconds per frame:  {bvh.frame_time}")
print(f"FPS:                {1 / bvh.frame_time:.0f}")

# %% [markdown]
# Note: the `frame_time` property stores the **time between frames** in seconds (i.e., the inverse of FPS). A value of 0.0333 means ~30 frames per second.

# %% [markdown]
# pybvh also auto-detects which axis is "up" in the file (Y-up vs Z-up, which varies across software). You can check it with `bvh.world_up`, or pass it explicitly at load time if you know your dataset's convention: `read_bvh_file("walk.bvh", world_up="+y")`. See the [World Up guide](https://victors-67.github.io/pybvh/guide/world-up/) for details.

# %% [markdown]
# Similarly, pybvh auto-detects **left/right joint pairs** from joint names (e.g., `LeftArm` ↔ `RightArm`). The result is cached in `bvh.lr_mapping` — a dict mapping left-side names to right-side names, or `None` if the heuristic can't match any pairs. On skeletons with non-standard naming, pass `lr_mapping={...}` at load time to override: `read_bvh_file("walk.bvh", lr_mapping={"LArm": "RArm", ...})`.

# %% [markdown]
# # Visualizing the skeleton
#
# Before we dive into the data, let's see what this skeleton actually looks like. pybvh includes a visualization module, `bvhplot`, for quick plotting.
#
# The **rest pose** is the skeleton's shape when all joint angles are zero — only the bone offsets define the posture. It's typically a T-pose or A-pose.

# %%
fig, ax = bvh.plot_rest_pose()
plt.show()

# %% [markdown]
# We can also visualize any frame of the animation. Here the skeleton is in an actual pose from the motion capture recording.

# %%
fig, ax = bvh.plot_frame(frame=0)
plt.show()

# %%
fig, ax = bvh.plot_frame(frame=30)
plt.show()

# %% [markdown]
# Now that we know what we're working with, let's look at how the data is actually stored.

# %% [markdown]
# # Motion data
#
# The motion data lives in two NumPy arrays inside the `Bvh` object:
#
# - **`root_pos`** — shape `(F, 3)`: the root joint's 3D position (typically the hips) at each frame.
# - **`joint_angles`** — shape `(F, J, 3)`: the Euler angle rotations (in **radians**) for each of the J joints at each frame.
#
# These are the raw values from the BVH file, converted to radians on read (the file format uses degrees; pybvh converts at the I/O boundary). Use `np.rad2deg(bvh.joint_angles)` if you want degrees for display. Remember: `joint_angles` stores **rotations**, not positions. To get 3D positions, you need forward kinematics (covered in Tutorial 2).

# %%
print(f"root_pos shape:     {bvh.root_pos.shape}   — {bvh.frame_count} frames, 3 position channels (X, Y, Z)")
print(f"joint_angles shape: {bvh.joint_angles.shape} — {bvh.frame_count} frames, {bvh.joint_count} joints, 3 Euler angles each")

# %%
print("Root position (first 5 frames):")
print(bvh.root_pos[:5])

print(f"\nEuler angles for the Hips joint (first 5 frames):")
print(bvh.joint_angles[:5, 0])

# %% [markdown]
# These arrays are standard NumPy — you can slice, index, and operate on them as you normally would.

# %% [markdown]
# ## Pandas DataFrame export
#
# If you prefer working with labeled columns, pybvh can export the motion data as a pandas DataFrame. The method `to_df_dict()` returns a dictionary ready for `pd.DataFrame()`. Each column is named `JointName_Axis_type` (e.g., `Hips_Z_rot`, `Spine_X_rot`), and a `time` column is added automatically.

# %%
import pandas as pd

df = pd.DataFrame(bvh.to_df_dict())
df.head()

# %% [markdown]
# You can also export spatial coordinates instead of Euler angles with `to_df_dict(mode='coordinates')` (see Tutorial 2). To reconstruct a `Bvh` object from a DataFrame, see `df_to_bvh()` in the API documentation.

# %% [markdown]
# # Skeleton hierarchy
#
# A BVH skeleton is a **tree**: the root joint (usually the hips) has children (spine, left leg, right leg), each of which has its own children, and so on down to the fingertips or toes.
#
# pybvh represents this tree as a flat list of node objects, stored in `bvh.nodes`. Each node knows its parent and children, so you can traverse the tree in any direction.

# %%
# The full list of nodes
for node in bvh.nodes:
    print(node.name)

# %% [markdown]
# There are three types of nodes:
#
# - **BvhRoot** — the root joint (always first in the list). Has position channels, rotation channels, and children.
# - **BvhJoint** — a regular joint. Has rotation channels, children, and a parent.
# - **BvhEndSite** — an end site (leaf node). Has only an offset and a parent. Represents the tip of a terminal bone (e.g., top of the head, tip of the toes).

# %%
# The root
print(type(bvh.root).__name__, "—", bvh.root.name)

# A regular joint
print(type(bvh.nodes[2]).__name__, "  —", bvh.nodes[2].name)

# An end site
print(type(bvh.nodes[8]).__name__, "  —", bvh.nodes[8].name)

# %% [markdown]
# ## Navigating the tree
#
# Each node has a `parent` pointer and (for joints) a `children` list. These point to the actual objects, so you can walk the tree directly.

# %%
spine = bvh.nodes[1]

print(f"Joint:    {spine.name}")
print(f"Parent:   {spine.parent.name}")
print(f"Children: {[c.name for c in spine.children]}")

# %% [markdown]
# ## Offsets and rotation channels
#
# Each node has an **offset** — a 3D vector `[x, y, z]` from its parent to itself. This is the direction and length of the bone connecting the two joints. In the rest pose, stacking these offsets from root to any joint gives you that joint's 3D position.
#
# Each joint also has **rot_channels** — the Euler rotation order for that joint (e.g., `['Z', 'Y', 'X']`). Different joints can have different orders. This matters when converting between rotation representations (see Tutorial 3).

# %%
for node in bvh.nodes[:6]:
    if hasattr(node, 'rot_channels'):
        print(f"{node.name:15s}  offset: {node.offset}  rotation order: {node.rot_channels}")
    else:
        print(f"{node.name:15s}  offset: {node.offset}  (end site — no rotation)")

# %% [markdown]
# The root also has a `pos_channels` property, indicating the order of its position channels (always `['X', 'Y', 'Z']` for standard BVH files). You can access the root directly with `bvh.root`.

# %%
print(f"Root joint:       {bvh.root.name}")
print(f"Position channels: {bvh.root.pos_channels}")
print(f"Rotation channels: {bvh.root.rot_channels}")
print(f"Children:          {[c.name for c in bvh.root.children]}")

# %% [markdown]
# # Writing and exporting
#
# To save a `Bvh` object back to a `.bvh` file, use `write()`. The round-trip is lossless — reading and writing produces an identical file.

# %%
bvh.write(bvh_folder / 'bvh_example.bvh', verbose=False)

# %% [markdown]
# You can also export the animation as a video file with `bvhplot.render()`.

# %%
path = bvh.render(output_folder / 'bvh_animation.mp4')
print(f"Animation saved to: {path}")

# %% [markdown]
# For interactive playback, camera control, side-by-side comparisons, and more export options, see Tutorial 4 (Visualization).

# %% [markdown]
# # What's next
#
# This tutorial covered the basics: loading a BVH file, visualizing the skeleton, accessing the raw motion arrays, and navigating the joint hierarchy.
#
# The following tutorials build on this:
#
# - **Tutorial 2 — Spatial Coordinates**: get 3D joint positions via forward kinematics, skeleton operations
# - **Tutorial 3 — Rotations**: convert between Euler angles, quaternions, 6D, axis-angle
# - **Tutorial 4 — Visualization**: video export, interactive playback, camera control
# - **Tutorial 5 — Transforms**: data augmentation (mirroring, rotation, noise, speed)
# - **Tutorial 6 — Features**: velocities, foot contacts, feature arrays for ML
# - **Tutorial 7 — Batch Processing**: loading directories, harmonization, dataset preparation
