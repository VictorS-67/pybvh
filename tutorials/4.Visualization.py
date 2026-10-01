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
# # Visualization with bvhplot

# %% [markdown]
# pybvh includes a built-in visualization module called `bvhplot`. It provides quick-look tools for inspecting BVH motion data without friction — one function call to see the skeleton.
#
# This tutorial covers all of `bvhplot`'s capabilities — feature by feature. If you are here to produce a figure for a paper rather than to learn the module, the [Publication Figures](https://victors-67.github.io/pybvh/guide/publication-figures/) guide is the short path: vector export, the sequence still, supplementary video, and the capsule look. **Static plots** (`rest_pose`, `frame`, `sequence`, `trajectory`) always use matplotlib, which is always available. **Video rendering** (`render`) and **interactive playback** (`play`) automatically select the fastest backend available — matplotlib is the universal fallback, but when OpenCV, vedo, or k3d are installed, pybvh uses them transparently. Every function takes a `style=` parameter (covered in its own section below) that controls the whole look — the default is a publication-grade style with a ground plane and per-chain bone colors. The last section of the tutorial details the optional backends and how to install them.

# %%
# %matplotlib inline
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import pybvh
from pybvh import bvhplot

np.set_printoptions(precision=4, suppress=True)

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "tutorials" else Path.cwd()
bvh_folder = REPO_ROOT / "bvh_data"
output_folder = Path("./output")
output_folder.mkdir(exist_ok=True)

bvh = pybvh.read_bvh_file(bvh_folder / "bvh_test1.bvh")
print(bvh)

# %% [markdown]
# ## A note on the API
#
# All visualization functions live in the `pybvh.bvhplot` module. For convenience, they're also available as **wrapper methods** on the `Bvh` class:
#
# | Use case | API style | Why |
# |---|---|---|
# | Single skeleton | `bvh.plot_frame()`, `bvh.render()`, etc. | Recommended — more intuitive |
# | Multiple skeletons (side-by-side) | `bvhplot.frame([bvh1, bvh2], ...)` | Required — designed for lists |
#
# In this tutorial, we use `bvh.method()` for single-object operations to keep the code clean and intuitive. We switch to `bvhplot.function()` when comparing multiple skeletons, since those functions are designed to handle lists.
#
# Both approaches call the same underlying code — use whichever fits your workflow.

# %% [markdown]
# # What are the visualizations for?
#
# pybvh proposes the following visualization tools, adapted to different needs:
#
# | What you want to know | Tool | Output |
# |---|---|---|
# | What does the skeleton look like | `rest_pose()` | Static rest pose (all joint angles zero) |
# | What's the pose at frame N? | `frame(frame=N)` | Static pose at a specific moment |
# | What does the whole motion look like in one image? | `sequence()` | Sampled poses in one figure, lightness encoding time |
# | What does the motion look like over time? | `play()` or `render()` | Animated sequence |
# | Does the character walk in a straight line? | `trajectory()` | 2D top-down root path |
#
# This section covers all of these. Most workflows start with `rest_pose()` or `frame()` to sanity-check the skeleton, then move to `play()` for exploration or `render()` to export results.

# %% [markdown]
# # Static snapshots

# %% [markdown]
# ## Viewing the rest pose
#
# The rest pose (also called bind pose) is the skeleton's shape when all joint angles are zero. Only the bone offsets define the posture. It's the first thing to check: does the skeleton look right?
#
# Use `bvh.plot_rest_pose()` to display it.

# %%
fig, ax = bvh.plot_rest_pose()
plt.show()

# %% [markdown]
# ## Viewing a single frame
#
# `bvh.plot_frame()` plots the skeleton at a specific frame of the animation. This is useful for inspecting the pose at a moment in time.
#
# Pass the frame index (0-based) as the `frame` parameter:

# %%
# Early frame
fig, ax = bvh.plot_frame(frame=0)
ax.set_title("Frame 0: Start of motion")
plt.show()

# %%
# Mid-motion frame
fig, ax = bvh.plot_frame(frame=30)
ax.set_title("Frame 30: Mid-motion")
plt.show()

# %% [markdown]
# ## Camera control
#
# Both `rest_pose()` and `frame()` accept a `camera` parameter to control the viewing angle. You can use preset strings or a custom `(azimuth, elevation)` tuple in degrees.
#
# **Presets** — these auto-orient to the skeleton (pybvh detects which axis is up and which is forward from the BVH hierarchy, so they work regardless of whether the file is Y-up or Z-up):
# - `'front'` — face the skeleton from the front
# - `'side'` — rotated 90° from front
# - `'top'` — bird's-eye view, looking down the up axis
#
# **Custom angles** — a tuple `(azimuth, elevation)` in degrees, passed directly to matplotlib's `view_init()`. The vertical axis is the skeleton's detected up axis. These follow matplotlib's convention, so the exact values don't correspond to an intuitive "front = 0°" — use the presets when you want a specific named view, and custom angles only for fine-tuning (e.g. a slight tilt from `'front'`).

# %%
# Compare the same frame from different camera angles
frame_num = 20

fig, axes = plt.subplots(1, 3, figsize=(15, 4), subplot_kw={"projection": "3d"})

for ax, angle in zip(axes, ["front", "side", "top"]):
    bvh.plot_frame(frame=frame_num, camera=angle, ax=ax)
    ax.set_title(f"Camera: {angle}")

plt.tight_layout()
plt.show()

# %%
# Custom angle
azimuth, elevation = -45, 30
fig, ax = bvh.plot_frame(frame=20, camera=(azimuth, elevation))
ax.set_title(f"Custom angle (azimuth={azimuth}°, elevation={elevation}°)")
plt.show()

# %% [markdown]
# ## Side-by-side comparison
#
# All `bvhplot` functions accept a list of `Bvh` objects to display multiple skeletons side by side. Use the `labels` parameter to title each subplot.
#
# This is useful for comparing different skeletons or the same motion with different transformations.

# %%
# Load a second skeleton for comparison
bvh2 = pybvh.read_bvh_file(bvh_folder / "bvh_test3.bvh")
bvh_small = bvh.scale(0.7)

fig, axes = bvhplot.rest_pose([bvh, bvh_small], labels=["Original", "Scaled 0.7x"])
plt.tight_layout()
plt.show()

# %%
# Compare the same frame from two different skeletons
fig, axes = bvhplot.frame([bvh, bvh2], frame=15, labels=["BVH 1", "BVH 2"])
plt.tight_layout()
plt.show()

# %% [markdown]
# # Visual styles
#
# Every `bvhplot` function accepts a `style=` parameter controlling the whole look of the output. It takes either a **preset name** or a **`Style` instance** for field-level control:
#
# - `'paper'` (default) — publication-grade: ground plane at the estimated floor height, per-chain bone colors, joint markers, axes hidden.
# - `'debug'` — the coordinate-inspection look: single blue skeleton, full axes and ticks, no floor. Use this when you need to read positions off the axes.
# - `'dark'` — the paper look on a near-black background, for slides and project pages.

# %%
for preset in ["paper", "debug", "dark"]:
    fig, ax = bvh.plot_frame(frame=30, style=preset)
    ax.set_title(f"style='{preset}'")
plt.show()

# %% [markdown]
# ## Overriding individual fields
#
# `Style(preset, **overrides)` starts from a preset and replaces any field. The commonly tweaked ones: `floor` (`'solid'`, `'checker'`, `'grid'`, or `None`), `axes` (`'off'` or `'full'`), `bone_width`, `joint_markers`, and `color_mode`. See the `Style` API docs for the full field list.

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 5), subplot_kw={"projection": "3d"})

styles = [
    bvhplot.Style("paper", floor="checker"),
    bvhplot.Style("paper", floor=None),
    bvhplot.Style("paper", axes="full"),
]
titles = ["floor='checker'", "floor=None", "axes='full'"]

for ax, style, title in zip(axes, styles, titles):
    bvh.plot_frame(frame=30, style=style, ax=ax)
    ax.set_title(title)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Color modes
#
# With the default `color_mode='auto'`, a **single skeleton** gets per-chain colors (left limbs warm, right limbs cool — you can tell left from right at a glance), while **side-by-side comparisons** switch to one flat color per skeleton, the standard convention for ground-truth-vs-generated figures. Force either behavior with `color_mode='chains'`, `'skeleton'`, or `'single'`.

# %%
fig, axes = bvhplot.frame([bvh, bvh.mirror()], frame=30, labels=["original", "mirror()"])
plt.suptitle("auto: side-by-side uses per-skeleton palette colors", y=0.98)
plt.tight_layout()
plt.show()

# %%
fig, axes = bvhplot.frame(
    [bvh, bvh.mirror()],
    frame=30,
    labels=["original", "mirror()"],
    style=bvhplot.Style("paper", color_mode="chains"),
)
plt.suptitle("color_mode='chains': chain colors everywhere", y=0.98)
plt.tight_layout()
plt.show()

# %% [markdown]
# # Sequence figures
#
# `bvh.plot_sequence()` draws the classic *motion-paper still*: a handful of equidistantly sampled poses in one figure, with lightness encoding time — lighter poses are earlier. It is the fastest way to convey a whole motion in a static image (papers, READMEs, slides).
#
# The `layout` parameter picks how poses are arranged:
#
# - `'offset'` (default) — poses stay at their world positions, so locomotion spreads left to right, with the root's path dashed on the floor. The right mode for travelling motion.
# - `'overlay'` — poses are superimposed (root-centered). The right mode for in-place motion, where offset poses would pile on top of each other anyway.

# %%
# A walking clip makes the offset layout shine (CMU mocap, subject 12)
walk = pybvh.read_bvh_file(bvh_folder / "cmu_12_01_walk.bvh")

fig, ax = walk.plot_sequence(n_poses=8)
plt.show()

# %%
# In-place motion: overlay layout
fig, ax = bvh.plot_sequence(n_poses=5, layout="overlay")
plt.show()

# %% [markdown]
# `n_poses` controls the sampling density, and `frames=` restricts sampling to a range — a `(start, stop)` tuple or a slice:

# %%
fig, ax = walk.plot_sequence(n_poses=6, frames=(200, 450))
ax.set_title("frames=(200, 450)")
plt.show()

# %% [markdown]
# # Centering modes

# %% [markdown]
# All visualization functions accept a `centered` parameter — the same centering modes as `node_positions()` (see Tutorial 2 for a full explanation):
#
# - `'world'` (default): absolute positions from the BVH file.
# - `'first'`: first frame's root over the origin (horizontal axes only, original height kept), motion continues from there.
# - `'skeleton'`: root at the origin in every frame (pose only, no global movement).

# %%
# Side-by-side frame plots — axis tick values reveal the centering difference
fig, axes = plt.subplots(1, 3, figsize=(16, 5), subplot_kw={"projection": "3d"})

for ax, mode in zip(axes, ["world", "first", "skeleton"]):
    bvh.plot_frame(frame=30, centered=mode, ax=ax)
    ax.set_title(f'centered="{mode}"')

plt.tight_layout()
plt.show()

# %% [markdown]
# # Root trajectory

# %% [markdown]
# The root trajectory is the path the skeleton's root joint (typically the hips) traces across the ground plane over the animation.
#
# This is useful for understanding the overall motion pattern: Is the character walking in a straight line? Turning in circles? Standing still?
#
# `bvhplot.trajectory()` shows a 2D top-down view of the root's path.
#
# > **A note on orientation.** Trajectory plots follow **map convention** — the world's forward direction (typically `+Y`) points **up on the plot**, like north on a map. This differs from the 3D `camera='front'` view earlier in this tutorial, which positions the camera in front of the character so their face points *out of the screen toward you*. Same world direction, different on-screen direction: it's the standard top-down-vs-camera split, not an inconsistency.

# %%
# Absolute trajectory
fig, ax = bvh.plot_trajectory(centered="world")
ax.set_title("World trajectory")
plt.show()

# %%
# Relative to first frame
fig, ax = bvh.plot_trajectory(centered="first")
ax.set_title("Trajectory relative to frame 0")
plt.show()

# %%
# Compare multiple trajectories
bvh2 = pybvh.read_bvh_file(bvh_folder / "bvh_test2.bvh")

fig, ax = bvhplot.trajectory([bvh, bvh2], labels=["Motion 1", "Motion 2"], centered="first")
ax.set_title("Trajectory comparison")
plt.show()

# %% [markdown]
# # Video and animation export

# %% [markdown]
# To export an animation to a file, use `bvh.render()`. This is especially useful for sharing results, including in papers or presentations.
#
# The output format is inferred from the file extension. Supported formats include `.mp4`, `.gif`, `.webp`, `.mov`, `.avi`, and `.html`.
#
# Rendered files are saved to the `output/` folder in the tutorials directory.

# %% [markdown]
# ## A note on backends
#
# By default (`backend='auto'`), `render()` uses OpenCV when installed (~100x faster) and falls back to matplotlib otherwise — all the examples below run on any install, just faster with `pip install pybvh[opencv]`. The `resolution` parameter (e.g. `(1920, 1080)`) only applies to the OpenCV backend. The *Interactive backends* section at the end of this tutorial covers the full backend matrix.

# %% [markdown]
# ## Basic rendering

# %% tags=["slow-on-pr"]
# Export to MP4
output_path = bvh.render(output_folder / "bvh_animation.mp4")
print(f"Animation saved to: {output_path}")

# %% tags=["slow-on-pr"]
# Export to GIF (smaller file, good for web/README)
output_path = bvh.render(output_folder / "bvh_animation.gif", fps=15)
print(f"GIF saved to: {output_path}")

# %% [markdown]
# ## Render options

# %% tags=["slow-on-pr"]
output_path = bvh.render(
    output_folder / "bvh_animation_custom.mp4",
    camera="side",
    fps=30,
    style=bvhplot.Style("paper", axes="full"),
)
print(f"Animation with options saved to: {output_path}")

# %% [markdown]
# ## Motion context: ghost trails and trajectory traces
#
# Two options add temporal context to a rendered clip:
#
# - `ghost=N` draws N faded copies of recent poses behind the live skeleton — older ghosts fade further toward the background. The spacing between ghosts is `Style.ghost_spacing` seconds (default 0.3).
# - `trajectory=True` draws the root's path on the floor, growing as the clip plays.
#
# Both work on the OpenCV and matplotlib backends and compose freely with each other and with any style.

# %% tags=["slow-on-pr"]
output_path = walk.render(
    output_folder / "walk_ghost_trace.mp4",
    camera="side",
    ghost=3,
    trajectory=True,
)
print(f"Ghost + trace animation saved to: {output_path}")

# %% [markdown]
# ## Turntable camera
#
# `camera='turntable'` orbits the camera a full 360° around the skeleton over the clip's duration, starting from the front view — the standard way to show a motion from all sides in one clip:

# %% tags=["slow-on-pr"]
output_path = bvh.render(output_folder / "bvh_turntable.mp4", camera="turntable")
print(f"Turntable animation saved to: {output_path}")

# %% [markdown]
# ## Frame counter
#
# Rendered clips are clean by default — publication output never stamps text on the image. For debugging or review, `frame_counter=True` stamps a `Frame f/F` counter in the corner (OpenCV backend):

# %% tags=["slow-on-pr"]
output_path = bvh.render(output_folder / "bvh_counter.mp4", frame_counter=True)
print(f"Frame-counter animation saved to: {output_path}")

# %% [markdown]
# ## Camera tracking: `follow=True`
#
# By default the camera is stable — it's pointed at the skeleton once from the first frame's orientation and stays there for the whole clip. If the character turns during the animation, you see them rotate in view (which is usually what you want for spatial awareness).
#
# For characters that rotate significantly, you can ask the camera to **track the character's facing direction**, so the view always shows them from the same (e.g. front) angle:
#
# ```python
# bvh.render('walk.mp4', follow=True)
# ```
#
# With this option, the character always faces the viewer while the world orbits around them.
#
# `follow=True` only makes sense with preset cameras (`'front'`, `'side'`, `'top'`). A custom `(azim, elev)` tuple is a fixed camera — `follow` is a silent no-op in that case.

# %% tags=["slow-on-pr"]
# Render with follow=True — camera tracks the character's facing direction.
# For this particular clip the character doesn't rotate much, so the effect
# is subtle; try it on a clip where the character turns.
output_path = bvh.render(
    output_folder / "bvh_animation_follow.mp4",
    camera="front",
    follow=True,
)
print(f"Follow-mode animation saved to: {output_path}")

# %% [markdown]
# ## Side-by-side video
#
# Just like static plots, you can render multiple skeletons side by side. The `sync` parameter controls behavior when clips have different lengths:
# - `'truncate'` (default): stop at the shortest clip
# - `'pad'`: pad shorter clips by freezing on their last frame

# %% tags=["slow-on-pr"]
bvh2 = pybvh.read_bvh_file(bvh_folder / "bvh_test2.bvh")

output_path = bvhplot.render(
    [bvh, bvh2],
    output_folder / "comparison.mp4",
    labels=["Original", "Other skeleton"],
    sync="pad",
    match_fps="highest",
)
print(f"Comparison video saved to: {output_path}")

# %% [markdown]
# # Interactive playback

# %% [markdown]
# `bvh.play()` provides interactive animation playback. It auto-detects the best backend available for your environment:
# - **k3d** (Jupyter widget) — if in Jupyter with k3d installed, interactive 3D widget
# - **vedo** (desktop window) — if installed, full 3D interactive viewer
# - **opencv** (notebook inline video) — if in Jupyter with OpenCV installed, an embedded video player
# - **matplotlib** (fallback) — always available, but less interactive
#
# Playback is clean by default; pass `frame_counter=True` to overlay the frame number on backends that support it.

# %% tags=["skip-execution"]
# This auto-detects the best backend
# falls back to matplotlib if k3d not installed
bvh.play()

# %% tags=["skip-execution"]
# Play multiple skeletons side by side
bvh2 = pybvh.read_bvh_file(bvh_folder / "bvh_test2.bvh")
bvhplot.play([bvh, bvh2], labels=["Motion 1", "Motion 2"], centered="first", sync="pad")

# %% [markdown]
# The previous cell triggers two warnings: the clips have different frame rates (30 fps vs 120 fps) and different world_up conventions (`+z` for Motion 1, `+y` for Motion 2). The cell below addresses both: `reorient_world_up()` aligns the coordinate systems, `match_fps` resamples to a common frame rate, and `sync="pad"` extends the shorter clip rather than truncating the longer one.

# %% tags=["skip-execution"]
bvh2_zup = bvh2.reorient_world_up("+z")

bvhplot.play(
    [bvh, bvh2_zup],
    labels=["Motion 1", "Motion 2"],
    centered="first",
    match_fps="highest",
    sync="pad",
)

# %% [markdown]
# # Interactive backends (optional)

# %% [markdown]
# pybvh supports optional visualization backends that provide faster rendering or richer interactive viewports. When installed, `render()` and `play()` use them automatically; this section details each one and how to install it.
#
# ## Available backends
#
# | Backend | Environment | Install | Best for |
# |---------|------------|---------|----------|
# | matplotlib | Any | *(included)* | Static plots, universal fallback |
# | OpenCV | Any | `pip install pybvh[opencv]` | Fast video rendering (~100x faster) |
# | vedo | Desktop or headless | `pip install pybvh[viewer]` | Interactive 3D desktop viewer, shadowed capsule renders |
# | k3d | Jupyter | `pip install pybvh[interactive]` | Jupyter interactive 3D widget |

# %% [markdown]
# ## Desktop viewer with vedo
#
# The vedo backend opens a full 3D window with keyboard controls (press `h` inside the viewer to toggle this list on screen):
#
# | Key | Action |
# |---|---|
# | `Space` | Play / pause |
# | `←` / `→` | Step one frame back / forward |
# | `Home` / `End` | Jump to first / last frame |
# | `+` / `-` | Speed playback up / down |
# | `f` | Cycle FPS presets |
# | `l` | Cycle loop mode |
# | `t` | Toggle the root trajectory trail |
# | `j` | Toggle joint name labels |
# | `1`–`9` | Toggle visibility of skeleton 1–9 (side-by-side mode) |
# | `s` | Save a clean screenshot (UI hidden, 2x resolution) |
# | `r` | Reset the camera |
# | `h` | Toggle the on-screen help panel |
#
# The `quality` parameter controls visual quality:
# - `'high'` (default) — 3D tubes and spheres with lighting
# - `'fast'` — flat lines and points for maximum performance

# %% tags=["skip-execution"]
# Requires vedo (pip install pybvh[viewer]) and a desktop session — opens a window,
# so this will not work on a remote/headless notebook
bvh.play(backend="vedo", quality="high")

# %% [markdown]
# ## Publication renders with vedo (offscreen)
#
# Beyond the interactive viewer, the vedo backend can produce **shadowed 3D capsule renders** — the volumetric skeleton-with-shadow look common in motion-generation papers. This runs fully offscreen (no window, headless-safe), so it works on remote servers and CI.
#
# `frame(backend='vedo')` returns an `(H, W, 3)` uint8 RGB image array instead of a matplotlib figure — display it with `plt.imshow` or save it directly via `filepath=`:

# %%
img = bvh.plot_frame(frame=30, backend="vedo", resolution=(1100, 1000))
plt.figure(figsize=(7, 6.4))
plt.imshow(img)
plt.axis("off")
plt.show()

# %% [markdown]
# `render(backend='vedo')` exports the same look as a video or GIF. The capsule renders honor `style=` like everything else (presets, floor kinds, chain colors, and the `Style.shadow` toggle); camera motions, ghosts, and traces are not supported on this backend — use OpenCV or matplotlib for those.

# %% tags=["slow-on-pr"]
output_path = bvh.render(output_folder / "bvh_capsules.mp4", backend="vedo")
print(f"Capsule render saved to: {output_path}")

# %% [markdown]
# ## Jupyter playback with k3d
#
# In Jupyter notebooks, the k3d backend renders an interactive 3D widget directly in the cell output. You can rotate, zoom, and scrub through the animation.

# %% tags=["skip-execution"]
# Requires k3d and a Jupyter session: pip install pybvh[interactive]
bvh.play(backend="k3d")

# %% [markdown]
# # Summary
#
# | Function | Purpose | Returns |
# |---|---|---|
# | `bvhplot.rest_pose(bvh)` | Static rest pose | `(Figure, Axes)` |
# | `bvhplot.frame(bvh, frame=N)` | Static frame | `(Figure, Axes)`, or an RGB array with `backend='vedo'` |
# | `bvhplot.sequence(bvh)` | Sampled poses in one figure, lightness encoding time | `(Figure, Axes)` |
# | `bvhplot.trajectory(bvh)` | 2D root path | `(Figure, Axes)` |
# | `bvhplot.render(bvh, path)` | Export animation to file | `Path` |
# | `bvhplot.play(bvh)` | Interactive playback | backend-specific |
#
# All functions accept a `style=` preset or `Style` instance; most accept `Bvh | list[Bvh]` and the `centered`, `camera`, and `labels` parameters (`sequence` is single-skeleton). The optional backends (vedo, k3d, OpenCV) provide richer playback, faster rendering, and shadowed capsule renders when installed.
#
# For the full parameter reference, see the API documentation.
