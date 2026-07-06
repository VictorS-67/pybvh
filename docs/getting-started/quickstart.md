# Quick Start

## Loading a BVH file

```python
import pybvh

bvh = pybvh.read_bvh_file("walk.bvh")
print(bvh)  # "24 joints, 75 frames at 30.0 fps (frame_time=0.033333s, from walk.bvh)"
```

## Accessing motion data

```python
bvh.root_pos          # (F, 3) root translation per frame
bvh.joint_angles      # (F, J, 3) Euler angles in radians
bvh.joint_names       # ['Hips', 'Spine', ...] (excludes end sites)
bvh.joint_count       # 24
bvh.euler_orders      # ['ZYX', 'ZYX', ...] per joint
```

## 3D joint positions

```python
coords = bvh.node_positions()  # (F, N, 3) via forward kinematics
```

## Rotation representations

```python
root_pos, quats = bvh.to_quat()         # (F, J, 4)
root_pos, rot6d = bvh.to_6d()           # (F, J, 6)
root_pos, aa = bvh.to_axisangle()    # (F, J, 3)
```

## Writing back to file

```python
bvh.write("output.bvh")
```

## Visualization

Single-skeleton calls are most natural as methods on the `Bvh` object:

```python
# Rest pose (T-pose / bind pose)
bvh.plot_rest_pose()

# Static 3D snapshot with camera control
bvh.plot_frame(frame=0, camera="front")  # also "side", "top", (azim, elev)

# Export animation to video (OpenCV if installed, else matplotlib)
bvh.render("walk.mp4")

# Camera tracks the character's rotation smoothly
bvh.render("walk_follow.mp4", follow=True)

# Interactive playback (auto-detects best backend)
bvh.play()

# 2D root trajectory
bvh.plot_trajectory()
```

Multi-skeleton comparisons use the `pybvh.bvhplot` module functions, which
accept a list of `Bvh` objects:

```python
from pybvh import bvhplot

bvhplot.frame([bvh1, bvh2], frame=0, labels=["A", "B"])
bvhplot.render([bvh1, bvh2], "compare.mp4", labels=["A", "B"], sync="pad")
bvhplot.trajectory([bvh1, bvh2], labels=["A", "B"])
```

The character's orientation is exposed as a property and a method:

```python
bvh.world_up           # e.g. '+y' or '+z' — auto-detected gravity axis
bvh.forward_at(0)      # facing direction in world space at frame 0
bvh.world_up = '+y'    # manual override if the auto-detect is wrong
```
