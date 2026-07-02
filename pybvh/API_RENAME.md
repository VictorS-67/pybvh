# pybvh API Rename — Complete Reference

Most renames below were applied in v0.6.0; the **`pybvh.features` → `pybvh.analysis` + `pybvh.packing`** module split landed in v0.8.0 (documented first, immediately below). **Old names are removed outright** — there is no deprecation cycle. Because pybvh has no known production consumers beyond pybvh-ml (which has been briefed separately), shipping wrappers that would be removed one release later wasn't worth the maintenance tax. Old names will raise `AttributeError` on the `Bvh` class / `ImportError` at module level.

Migration: grep for the old name in the left column, replace with the right column.

---

## v0.8.0 — `pybvh.features` split into `pybvh.analysis` + `pybvh.packing`

`pybvh.features` was split by responsibility and **no longer exists** (`import pybvh.features` raises `ImportError`). `Bvh` method names are **unchanged** — `bvh.joint_velocities()`, `bvh.foot_contacts()`, `bvh.to_feature_array()`, … all still work. Only the module-level functions moved:

| Old (`pybvh.features.*`) | New |
|---|---|
| `node_velocities`, `joint_velocities` | `pybvh.analysis.*` |
| `node_accelerations`, `joint_accelerations` | `pybvh.analysis.*` |
| `angular_velocities`, `root_trajectory` | `pybvh.analysis.*` |
| `foot_contacts`, `auto_detect_foot_joints` | `pybvh.analysis.*` |
| `to_feature_array`, `feature_array_layout` | `pybvh.packing.*` |

Rule of thumb: **motion descriptors → `analysis`; ML feature-array assembly → `packing`.**

---

## v0.8.0 — `pybvh.tools` rotation helpers folded into `pybvh.rotations`

The duplicate Euler→rotmat implementation in `pybvh.tools` **no longer exists**; `pybvh.rotations.euler_to_rotmat` is the single batched implementation (it also powers forward kinematics).

| Old (`pybvh.tools.*`) | New |
|---|---|
| `get_premult_mat_rot(angles, order)` | `pybvh.rotations.euler_to_rotmat(angles, order)` |
| `batch_get_premult_mat_rot(angles, order)` | `pybvh.rotations.euler_to_rotmat(angles, order)` |
| `rotX(a)` / `rotY(a)` / `rotZ(a)` | *(removed)* — e.g. `euler_to_rotmat([a, 0, 0], 'XYZ')` for a single-axis matrix |
| `batch_rotX/Y/Z(angles)` | *(removed)* — same as above, batched |

---

## Bvh class — Properties / Attributes

| Old name | New name | Notes |
|---|---|---|
| `frame_frequency` | `frame_time` | Historically misnamed — stored frame *time*, not frequency. |
| `euler_column_names` | `_euler_column_names` | Made private. Was never used outside tests. |

### Preferred lookup for `joint_angles` axis 1

| Old pattern | New pattern | Notes |
|---|---|---|
| `bvh.joint_names.index(name)` | `bvh.joint_index[name]` | **Addition, not a rename.** `joint_names` still exists and still returns a plain `list[str]`; `.index()` still works. The new `joint_index` dict is the preferred lookup because it mirrors `node_index` (same dict shape) and its name matches the array it indexes. See [CONTEXT.md §5.2](CONTEXT.md). |

## Bvh class — I/O

| Old name | New name |
|---|---|
| `to_bvh_file(path)` | `write(path)` |
| `get_df_constructor(mode)` | `to_df_dict(mode)` |

## Bvh class — Spatial data extraction

| Old name | New name |
|---|---|
| `get_spatial_coord(...)` | `spatial_coords(...)` |
| `get_rest_pose(...)` | `rest_pose_coords(...)` |

## Bvh class — Rotation conversions

| Old name | New name | Notes |
|---|---|---|
| `get_frames_as_rotmat()` | `to_rotmat()` | Returns 2-tuple `(root_pos, joint_data)` — third `joints` element removed. |
| `get_frames_as_6d()` | `to_6d()` | Same 2-tuple shape change. |
| `get_frames_as_quaternion()` | `to_quaternions()` | Same 2-tuple shape change. |
| `get_frames_as_axisangle()` | `to_axisangle()` | Same 2-tuple shape change. |
| `set_frames_from_6d(...)` | `from_6d(...)` | |
| `set_frames_from_quaternion(...)` | `from_quaternions(...)` | |
| `set_frames_from_axisangle(...)` | `from_axisangle(...)` | |

## Bvh class — Skeleton operations

| Old name | New name | Notes |
|---|---|---|
| `change_skeleton(ref)` | `retarget(ref)` | |
| `scale_skeleton(factor)` | `scale(factor)` | Also scales `root_pos` now — remove any manual post-scaling. |
| `single_joint_euler_angle(joint, order)` | `change_euler_order(order, joint=joint)` | Positional args reordered. |
| `change_all_euler_orders(order)` | `change_euler_order(order)` | Unified with single-joint version. |

## Bvh class — Transforms

| Old name | New name |
|---|---|
| `add_joint_noise(...)` | `add_noise(...)` |
| `speed_perturbation(factor)` | `perturb_speed(factor)` |
| `dropout_frames(rate)` | `drop_frames(rate)` |

Unchanged: `mirror()`, `rotate_vertical()`, `translate_root()`.

## Bvh class — Features

| Old name | New name | Notes |
|---|---|---|
| `get_joint_velocities(...)` | `joint_velocities(...)` | |
| `get_joint_accelerations(...)` | `joint_accelerations(...)` | |
| `get_angular_velocities(...)` | `angular_velocities(...)` | |
| `get_root_relative_positions(...)` | *(removed)* | Use `spatial_coords(centered='skeleton')` — mathematically identical. |
| `root_relative_positions(...)` | *(removed)* | Same. Was soft-deprecated in drafts; removed before release. |
| `get_root_trajectory(...)` | `root_trajectory(...)` | |
| `get_foot_contacts(...)` | `foot_contacts(...)` | |

Unchanged: `to_feature_array()`.

### Parameter renames

| Function | Old parameter | New parameter(s) | Notes |
|---|---|---|---|
| `foot_contacts` | `threshold=` | `vel_threshold=` (for `method="velocity"`) / `height_threshold=` (for `method="height"`) | The single `threshold` was ambiguous once `method="combined"` began using both signals. |
| `batch.harmonize` | `target_up=` | `target_world_up=` | Renamed post-v0.6.0 to disambiguate once `target_rest_up=` was added to the signature. Clean break, no deprecation shim. |

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

### `pybvh.features` (v0.6.0 function renames — module since split, see top)

These bare-name renames happened in v0.6.0, while the functions still lived in `pybvh.features`. As of v0.8.0 they live in `pybvh.analysis` (`to_feature_array` / `feature_array_layout` in `pybvh.packing`).

| Old name | New name |
|---|---|
| `get_joint_velocities(bvh, ...)` | `joint_velocities(bvh, ...)` |
| `get_joint_accelerations(bvh, ...)` | `joint_accelerations(bvh, ...)` |
| `get_angular_velocities(bvh, ...)` | `angular_velocities(bvh, ...)` |
| `get_root_relative_positions(bvh, ...)` | *(removed)* — use `bvh.spatial_coords(centered='skeleton')` |
| `root_relative_positions(bvh, ...)` | *(removed)* — use `bvh.spatial_coords(centered='skeleton')` |
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
