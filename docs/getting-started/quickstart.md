# Quick Start

## Loading a BVH file

```python
import pybvh

bvh = pybvh.read_bvh_file("walk.bvh")
print(bvh)  # 24 joints, 120 frames at 0.033333Hz
```

## Accessing motion data

```python
bvh.root_pos          # (F, 3) root translation per frame
bvh.joint_angles      # (F, J, 3) Euler angles in degrees
bvh.joint_names       # ['Hips', 'Spine', ...] (excludes end sites)
bvh.joint_count       # 24
bvh.euler_orders      # ['ZYX', 'ZYX', ...] per joint
```

## 3D joint positions

```python
coords = bvh.get_spatial_coord()  # (F, N, 3) via forward kinematics
```

## Rotation representations

```python
root_pos, quats, joints = bvh.get_frames_as_quaternion()  # (F, J, 4)
root_pos, rot6d, joints = bvh.get_frames_as_6d()          # (F, J, 6)
root_pos, aa, joints    = bvh.get_frames_as_axisangle()    # (F, J, 3)
```

## Writing back to file

```python
bvh.to_bvh_file("output.bvh")
```

## Visualization

```python
from pybvh import plot

plot.plot_frame(bvh, frame=0)
plot.plot_animation(bvh, output_path="walk.mp4")
```
