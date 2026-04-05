# bvhplot — Module Charter

## What bvhplot is

bvhplot is pybvh's built-in visualization module. It provides **quick-look** tools for inspecting BVH motion data: static snapshots, video exports, and interactive playback. It is designed for researchers and developers who want to see their data without leaving their Python environment.

## Core mission

**Show BVH motion data with zero friction.**

`from pybvh import bvhplot; bvhplot.play(bvh)` should work in any environment (script, notebook, headless server) with minimal setup. The visualization is a convenience, not the product.

## Design principles

1. **Zero-friction.** One function call to see the skeleton. No config files, no GUI setup, no separate application.
2. **Multi-environment.** Works in Jupyter notebooks (k3d, inline video), desktop scripts (vedo, matplotlib), and headless servers (video export). Backend auto-detection handles this transparently.
3. **Lightweight optional deps.** Core visualization uses only matplotlib (already a pybvh dependency). Interactive backends (vedo, k3d) and fast rendering (opencv) are optional extras.
4. **Not an editor.** bvhplot is read-only. It displays data; it does not modify it. No keyframe editing, no pose manipulation, no undo/redo.
5. **Not a professional viewer.** Features that require GUI widgets (property panels, skeleton trees, graph editors, dockable layouts) belong in a separate tool (pybvh-blender or a dedicated viewer application).

## What bvhplot owns

- **Static snapshots** (`frame`, `rest_pose`): Matplotlib 3D plots for papers and quick checks.
- **Video/GIF/HTML export** (`render`): Batch rendering of animations to files. OpenCV backend for speed, matplotlib fallback for universality.
- **Interactive playback** (`play`): In-window animation with transport controls (play/pause, speed, scrubbing, frame stepping).
- **Trajectory plots** (`trajectory`): 2D top-down root path visualization.
- **Camera math**: Consistent camera angles across all backends (front/side/top presets, azimuth/elevation).
- **Desktop viewer features**: Keyboard shortcuts, FPS selector, joint labels, trajectory trail, screenshot export, ping-pong playback. These are lightweight toggles on the 3D viewport, not GUI widgets.

## What bvhplot does NOT own

- **Property panels / joint inspectors**: Displaying per-joint rotation values, bone lengths, or channel data in a panel requires GUI widgets. Use pybvh-blender.
- **Graph editors / FCurve views**: Plotting per-channel animation curves over time requires a 2D graph widget. Use Blender's graph editor via pybvh-blender, or export data to matplotlib manually.
- **Skeleton hierarchy tree**: Navigating a 60+ joint skeleton by expanding/collapsing a tree requires a tree widget. Use pybvh-blender.
- **Multi-viewport layouts**: Showing front + side + top simultaneously requires a multi-renderer setup. Use Blender's quad view.
- **Video overlay / reference comparison**: Loading and syncing external video alongside the skeleton requires video decoding and frame alignment. Out of scope.
- **Live mocap streaming**: Real-time data input is outside bvhplot's file-based model.

## The boundary rule

**If a feature requires a GUI widget (panel, tree, graph, text input, dropdown, context menu), it does not belong in bvhplot.**

If it's a keyboard toggle on the 3D viewport (show/hide something, change a mode), it belongs in bvhplot.

## Backend architecture

| Backend | Environment | Optional dep | Purpose |
|---------|------------|-------------|---------|
| matplotlib | Any | None (core dep) | Static plots, slow animation fallback |
| OpenCV | Any | `opencv-python` | Fast video export (~1000fps rendering) |
| k3d | Jupyter | `k3d` | Interactive notebook playback |
| vedo | Desktop | `vedo` | Interactive desktop viewer with full controls |

Auto-detection priority: k3d (notebook) > vedo (desktop) > opencv (notebook fallback) > matplotlib (universal fallback).

## Ecosystem position

```
pybvh (core library)
  |
  +-- pybvh.bvhplot (built-in quick visualization)
  |
  +-- pybvh-blender (separate repo: deep inspection in Blender)
  |
  +-- pybvh-ml (separate repo: ML pipeline tools)
```

bvhplot is the only visualization component inside pybvh itself. For professional inspection, users should use pybvh-blender. For ML pipelines, users should use pybvh-ml.
