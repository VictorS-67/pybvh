# pybvh API Rename — Complete Reference

All API renames applied in this version. Old names have deprecation wrappers
that emit `DeprecationWarning` and delegate to the new name. Old names will
be removed in a future major version.

---

## Bvh class — Properties / Attributes

| Old name | New name | Notes |
|---|---|---|
| `frame_frequency` | `frame_time` | Deprecation property on old name. |
| `euler_column_names` | `_euler_column_names` | Made private. Was never used outside tests. |

## Bvh class — I/O

| Old name | New name | Notes |
|---|---|---|
| `to_bvh_file(path)` | `write(path)` | Deprecation wrapper on old name. |
| `get_df_constructor(mode)` | `to_df_dict(mode)` | Deprecation wrapper on old name. |

## Bvh class — Spatial data extraction

| Old name | New name | Notes |
|---|---|---|
| `get_spatial_coord(...)` | `spatial_coords(...)` | Deprecation wrapper on old name. |
| `get_rest_pose(...)` | `rest_pose_coords(...)` | Deprecation wrapper on old name. |

## Bvh class — Rotation conversions

| Old name | New name | Notes |
|---|---|---|
| `get_frames_as_rotmat()` | `to_rotmat()` | Deprecation wrapper on old name. |
| `get_frames_as_6d()` | `to_6d()` | Deprecation wrapper on old name. |
| `get_frames_as_quaternion()` | `to_quaternions()` | Deprecation wrapper on old name. |
| `get_frames_as_axisangle()` | `to_axisangle()` | Deprecation wrapper on old name. |
| `set_frames_from_6d(...)` | `from_6d(...)` | Deprecation wrapper on old name. |
| `set_frames_from_quaternion(...)` | `from_quaternions(...)` | Deprecation wrapper on old name. |
| `set_frames_from_axisangle(...)` | `from_axisangle(...)` | Deprecation wrapper on old name. |

## Bvh class — Skeleton operations

| Old name | New name | Notes |
|---|---|---|
| `change_skeleton(ref)` | `retarget(ref)` | Deprecation wrapper on old name. |
| `scale_skeleton(factor)` | `scale(factor)` | Deprecation wrapper on old name. |
| `single_joint_euler_angle(joint, order)` | `change_euler_order(order, joint=joint)` | Param order changed. Deprecation wrapper on old name. |
| `change_all_euler_orders(order)` | `change_euler_order(order)` | Unified with single-joint version. Deprecation wrapper on old name. |

## Bvh class — Transforms

| Old name | New name | Notes |
|---|---|---|
| `add_joint_noise(...)` | `add_noise(...)` | Deprecation wrapper on old name. |
| `speed_perturbation(factor)` | `perturb_speed(factor)` | Deprecation wrapper on old name. |
| `dropout_frames(rate)` | `drop_frames(rate)` | Deprecation wrapper on old name. |

Unchanged: `mirror()`, `rotate_vertical()`, `translate_root()`.

## Bvh class — Features

| Old name | New name | Notes |
|---|---|---|
| `get_joint_velocities(...)` | `joint_velocities(...)` | Deprecation wrapper on old name. |
| `get_joint_accelerations(...)` | `joint_accelerations(...)` | Deprecation wrapper on old name. |
| `get_angular_velocities(...)` | `angular_velocities(...)` | Deprecation wrapper on old name. |
| `get_root_relative_positions(...)` | `root_relative_positions(...)` | Deprecation wrapper on old name. |
| `get_root_trajectory(...)` | `root_trajectory(...)` | Deprecation wrapper on old name. |
| `get_foot_contacts(...)` | `foot_contacts(...)` | Deprecation wrapper on old name. |

Unchanged: `to_feature_array()`.

## Bvh class — Visualization (new, no old name)

| New name | Wraps |
|---|---|
| `plot_rest_pose(...)` | `bvhplot.rest_pose(self, ...)` |
| `plot_frame(...)` | `bvhplot.frame(self, ...)` |
| `plot_trajectory(...)` | `bvhplot.trajectory(self, ...)` |
| `render(path, ...)` | `bvhplot.render(self, path, ...)` |
| `play(...)` | `bvhplot.play(self, ...)` |

---

## Module-level function renames

### `pybvh.features`

| Old name | New name |
|---|---|
| `get_joint_velocities(bvh, ...)` | `joint_velocities(bvh, ...)` |
| `get_joint_accelerations(bvh, ...)` | `joint_accelerations(bvh, ...)` |
| `get_angular_velocities(bvh, ...)` | `angular_velocities(bvh, ...)` |
| `get_root_relative_positions(bvh, ...)` | `root_relative_positions(bvh, ...)` |
| `get_root_trajectory(bvh, ...)` | `root_trajectory(bvh, ...)` |
| `get_foot_contacts(bvh, ...)` | `foot_contacts(bvh, ...)` |

Unchanged: `to_feature_array()`.

### `pybvh.transforms`

| Old name | New name |
|---|---|
| `add_joint_noise(bvh, ...)` | `add_noise(bvh, ...)` |
| `speed_perturbation(bvh, factor)` | `perturb_speed(bvh, factor)` |
| `random_speed_perturbation(bvh, ...)` | `random_perturb_speed(bvh, ...)` |
| `dropout_frames(bvh, ...)` | `drop_frames(bvh, ...)` |

Unchanged: `translate_root()`, `random_translate_root()`, `rotate_vertical()`, `random_rotate_vertical()`, `mirror()`, `auto_detect_lr_mapping()`, `auto_detect_lr_pairs()`, `rotate_angles_vertical()`, `mirror_angles()`.

### `pybvh.spatial_coord`

| Old name | New name |
|---|---|
| `frames_to_spatial_coord(...)` | `frames_to_spatial_coords(...)` |

### Modules with no changes

- `pybvh.io` — `read_bvh_file()`, `write_bvh_file()` unchanged
- `pybvh.rotations` — all 15 functions unchanged
- `pybvh.batch` — all 5 functions unchanged
- `pybvh.df_to_bvh` — `df_to_bvh()` unchanged
- `pybvh.bvhplot` — `rest_pose()`, `frame()`, `trajectory()`, `render()`, `play()` unchanged
- `pybvh.bvhnode` — `BvhNode`, `BvhJoint`, `BvhRoot` unchanged

---

## Breaking changes

**None.** All old names coexist as deprecation wrappers. They emit
`DeprecationWarning` and delegate to the new names. They will be removed
in a future major version.
