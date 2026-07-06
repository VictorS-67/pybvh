# pybvh API Rename — Complete Reference

Most renames below were applied in v0.6.0; the **`pybvh.features` → `pybvh.analysis` + `pybvh.features`** module split landed in v0.8.0 (documented first, immediately below). **Old names are removed outright** — there is no deprecation cycle. Because pybvh has no known production consumers beyond pybvh-ml (which has been briefed separately), shipping wrappers that would be removed one release later wasn't worth the maintenance tax. Old names will raise `AttributeError` on the `Bvh` class / `ImportError` at module level.

Migration: grep for the old name in the left column, replace with the right column.

---

## v0.8.0 — `pybvh.features` split: descriptors → `pybvh.analysis`

The old mixed `pybvh.features` module was split by responsibility: motion descriptors moved to the new `pybvh.analysis`; `pybvh.features` remains but now holds **only** feature-array export. `Bvh` method names are **unchanged** — `bvh.joint_velocities()`, `bvh.foot_contacts()`, `bvh.to_feature_array()`, … all still work. Only the module-level functions moved:

| Old (`pybvh.features.*`) | New |
|---|---|
| `node_velocities`, `joint_velocities` | `pybvh.analysis.*` |
| `node_accelerations`, `joint_accelerations` | `pybvh.analysis.*` |
| `angular_velocities`, `root_trajectory` | `pybvh.analysis.*` |
| `foot_contacts`, `auto_detect_foot_joints` | `pybvh.analysis.*` |
| `to_feature_array`, `feature_array_layout` | `pybvh.features.*` *(unchanged path)* |

Rule of thumb: **motion descriptors → `analysis`; ML feature-array export stays in `features`.**

*(During 0.8.0 development the export half was briefly named `pybvh.packing`; that interim name never shipped in a release — `pybvh.packing.X` → `pybvh.features.X`.)*

---

## v0.8.0 — signal utilities: `pybvh.tools` → `pybvh.signal`

The array-pure signal utilities moved out of `pybvh.tools` (which shrinks to private helpers) into the new documented `pybvh.signal` module:

| Old (`pybvh.tools.*`) | New |
|---|---|
| `finite_difference` | `pybvh.signal.finite_difference` |
| `temporal_stats` | `pybvh.signal.temporal_stats` |
| `box_filter_smooth` | `pybvh.signal.box_filter_smooth` |
| `fft_magnitude` | `pybvh.signal.fft_magnitude` |
| `dominant_frequency` | `pybvh.signal.dominant_frequency` |
| `ramer_douglas_peucker` | `pybvh.signal.ramer_douglas_peucker` |

`pybvh.tools.test_file` (BVH path validation) is now the private `tools._validate_bvh_path`; `pybvh.tools.are_permutations` is removed (inlined at its single call site).

---

## v0.8.0 — normalization moved to pybvh-ml

Dataset-level z-score normalization is an ML-pipeline concern, and pybvh-ml already privately reimplemented the array-level core — the public trio moved there outright. Behavior is unchanged (same `mean` / `std` / `constant_channels` stats dict, same z-score math).

| Old (`pybvh.batch.*`, also exported as `pybvh.*`) | New |
|---|---|
| `compute_normalization_stats` | `pybvh_ml.compute_normalization_stats` |
| `normalize_array` | `pybvh_ml.normalize_array` |
| `denormalize_array` | `pybvh_ml.denormalize_array` |

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

## v0.8.0 — end sites are `BvhEndSite`, not plain `BvhNode`

`pybvh.bvhnode.BvhNode` is now the abstract-ish base class shared by all node kinds and no longer answers `is_end_site()`. End-site identity is carried by the new `BvhEndSite` class (checked via `isinstance` / `is_end_site()`); the generated `'EndSite<parent>'` display names carry no semantics anywhere.

| Old | New |
|---|---|
| `BvhNode(name, offset, parent)` *(as an end site)* | `BvhEndSite(name, offset, parent)` |
| `'EndSite'` name prefix marking end sites in `df_to_bvh` hierarchy dicts | structural: an entry with neither `'children'` nor `'rot_channels'` is an end site |

---

## v0.8.0 — analysis / geometry renames & signature changes

| Old | New | Notes |
|---|---|---|
| `pybvh.geometry.centroid` | `pybvh.geometry.center_of_mass` | Now shares the `Bvh.center_of_mass` method name. |
| `pybvh.analysis.lagged_correlation` | `pybvh.analysis.lagged_covariance` | Also mean-centers the signal (true covariance) — values change. |
| `pybvh.batch.relative_scale_factor` | `pybvh.analysis.relative_scale_factor` | Moved next to `skeleton_size`; also exported as `pybvh.relative_scale_factor`. |
| `foot_contacts(..., centered=)` | *(removed)* | Detection always runs in world frame; `coords=` accepts world-frame positions or a constant translation thereof. |
| `foot_contacts(..., vel_threshold=)` *(units/frame)* | `vel_threshold=` *(units/second)* | Migration: `new = old / frame_time`. Default `0.12·scale` u/s ≡ old `0.004·scale`/frame at 30 fps. |
| `foot_contacts(..., height_reference="fixed")` | `height_reference="floor"` | Parameter is now validated: `{"velocity", "floor"}`. |
| `walking_pace(bvh, foot_joints=)` | `walking_pace(bvh)` | The parameter was unused. |
| `velocity_reductions(speed, fs=1.0)` | `velocity_reductions(speed, fs)` | `fs` (Hz) is required — no implicit time base. |
| `active_duration(speed, threshold, frame_time=1.0)` | `active_duration(speed, threshold, fs)` | `fs` (Hz) is required; pass `fs = 1 / frame_time`. |

---

## v0.8.0 — transforms: radians unification & L/R mapping cleanup

All angle parameters in `pybvh.transforms` are now **radians** (matching `Bvh.joint_angles` and the `degrees=` flag convention in `rotations`/`analysis`/`geometry`). The same renames apply to the corresponding `Bvh` methods.

| Old | New | Notes |
|---|---|---|
| `add_noise(bvh, sigma_deg)` | `add_noise(bvh, sigma)` | `sigma` is in radians. Migration: `sigma = np.radians(sigma_deg)`. Negative `sigma`/`sigma_pos` now raise `ValueError`. |
| `add_noise(..., wrap=True)` *(default)* | `wrap=False` *(default)* | Wrapping silently corrupts channels legitimately outside `[-π, π]` (accumulated rotations); opt back in with `wrap=True`. |
| `rotate_vertical(bvh, angle_deg)` | `rotate_vertical(bvh, angle, degrees=False)` | Radians by default; pass `degrees=True` to keep degree inputs. `up_axis` is validated (signed axis string; `'y'` now raises a clear `ValueError` instead of `IndexError`). |
| `rotate_angles_vertical(..., angle_deg, ...)` | `rotate_angles_vertical(..., angle, ..., degrees=False)` | Array-level twin of `rotate_vertical`. |
| `random_rotate_vertical(bvh, angle_range=(-180, 180))` | `random_rotate_vertical(bvh, angle_range=(-np.pi, np.pi), degrees=False)` | |
| `random_translate_root(bvh, range_xyz=)` | `random_translate_root(bvh, offset_range=)` | Matches the `angle_range` / `factor_range` sibling names. |
| `mirror(bvh, left_right_mapping=)` | `mirror(bvh, lr_mapping=)` | Matches `Bvh.lr_mapping` and the loader parameter. An explicitly passed mapping with unknown joint names now raises `ValueError` listing them. |
| `transforms.auto_detect_lr_mapping(bvh)` | *(removed)* | Read `bvh.lr_mapping` — same dict, but `None` (not `{}`) when nothing is detected. `auto_detect_lr_pairs()` remains. |
| `drop_frames(bvh, rate)` | *(signature unchanged)* | Behavior change: kept frames are preserved bit-for-bit; only dropped frames are re-synthesized (previously every frame was re-canonicalized through a quaternion round-trip). |

---

## v0.8.0 — Bvh API surface

| Old | New | Notes |
|---|---|---|
| `to_quaternions()` / `from_quaternions(...)` | `to_quat()` / `from_quat(...)` | Representation *strings* are also `"quat"` everywhere (`to_feature_array`, `batch_to_numpy`, `rotations.convert`, `REPRESENTATION_CHANNELS`). |
| `rest_pose_coords()` / `rest_pose_coords(mode='coordinates')` | `rest_pose_positions()` | No motion-data dependence — now works on 0-frame Bvh objects. |
| `rest_pose_coords(mode='euler')` | `rest_pose_angles()` | Returns just the `(J, 3)` zeros; the tuple's root-position element is gone (it was always `zeros(3)`). |
| `hierarchy_info_as_dict()` | `to_hierarchy_dict()` | Pairs with the new `Bvh.from_df(hier, df)` classmethod. |
| `slice_frames(a, b, s)` | `bvh[a:b:s]` | Public method removed — the sequence protocol is the only spelling. |
| `concat(other)` | `bvh + other` / `bvh += other` | Public method removed — the operator is the only spelling. |
| `node_positions(frame_num=-1)` / `joint_positions(frame_num=-1)` | `node_positions(frame=None)` | `frame=None` (default) = all frames; `frame=-1` now returns the **last** frame (NumPy negative-index semantics). |
| `index(name, axis='joint'/'node')` | `index(name, space='joint'/'node')` | Parameter rename. |
| `write(new_filepath)` | `write(filepath)` | Parameter rename. |
| `scale((sx, sy, sz))` | *(removed)* | `scale()` is scalar-only — per-axis world factors on parent-local offsets are not geometrically meaningful under animation; non-scalar input raises `TypeError`. |
| descriptor methods with integer joint indices | names only (`str`) | `curvature` … `smoothness`, `range_of_motion` raise `TypeError` on ints (they were ambiguous between joint/node index spaces); use `bvh.index(name, space=...)` + the functional `pybvh.geometry` / `pybvh.analysis` API for index-based access. |
| `pybvh.api_rename_path()` | *(removed)* | This file stays in the repo / docs; the helper added no value. |

New in the same release (no old names): `Bvh.from_file(path)`, `Bvh.from_df(hier, df)`, `Bvh.from_rotmat(root_pos, rotmats)`, and method wrappers `bounding_ellipsoid()`, `movement_phase()`, `skeleton_size()`, `velocity_reductions()`. All motion-descriptor methods accept pre-computed positions via `coords=`. `bvh.world_up = 'auto'` (or `None`) now clears a manual override.

---

## v0.8.0 — bvhplot signatures

| Old | New | Notes |
|---|---|---|
| `render(..., fps=-1)` / `play(..., fps=-1)` | `fps=None` *(default)* | `fps` is `float \| None`: `None` = BVH frame rate, fractional rates (119.88) accepted, `fps <= 0` raises `ValueError` (previously `render(fps=0)` crashed with `ZeroDivisionError`). |
| `frame(bvh, coords_array)` | `frame(bvh, coords=coords_array)` | Pre-computed positions move to an explicit keyword; the positional `frame` parameter is an int frame index only (NumPy negative-index semantics — `frame=-1` is now the **last** frame, no longer an all-frames sentinel). |
| `render(backend=<typo>)` *(silent matplotlib fallback)* | raises `ValueError` | Validated against `{"auto", "opencv", "matplotlib"}`, mirroring `play()`. Under `"auto"`, `.gif`/`.webp`/`.apng`/`.html` always route to matplotlib/pillow even with cv2 installed; a forced `backend="opencv"` rejects extensions it cannot write. |
| `play()` return value | `None` | Documented and true for every backend (the k3d path previously documented returning the plot widget it deliberately didn't return). |

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
| `get_rest_pose(...)` | `rest_pose_positions()` *(named `rest_pose_coords` in v0.6.0–v0.7.x)* |

## Bvh class — Rotation conversions

| Old name | New name | Notes |
|---|---|---|
| `get_frames_as_rotmat()` | `to_rotmat()` | Returns 2-tuple `(root_pos, joint_data)` — third `joints` element removed. |
| `get_frames_as_6d()` | `to_6d()` | Same 2-tuple shape change. |
| `get_frames_as_quaternion()` | `to_quat()` *(named `to_quaternions` in v0.6.0–v0.7.x)* | Same 2-tuple shape change. |
| `get_frames_as_axisangle()` | `to_axisangle()` | Same 2-tuple shape change. |
| `set_frames_from_6d(...)` | `from_6d(...)` | |
| `set_frames_from_quaternion(...)` | `from_quat(...)` *(named `from_quaternions` in v0.6.0–v0.7.x)* | |
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

Unchanged: `translate_root()`. The v0.8.0 radians/parameter changes (see the *transforms: radians unification* section above) apply to the `Bvh` methods too: `add_noise(sigma)` in radians with `wrap=False` default, `rotate_vertical(angle, degrees=False)`, `random_rotate_vertical(angle_range=(-np.pi, np.pi))`, `random_translate_root(offset_range=)`, `mirror(lr_mapping=)`.

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

These bare-name renames happened in v0.6.0, while the functions still lived in the old mixed `pybvh.features`. As of v0.8.0 they live in `pybvh.analysis` (`to_feature_array` / `feature_array_layout` in `pybvh.features`).

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

Unchanged by v0.6.0: `translate_root()`, `auto_detect_lr_pairs()`, `mirror_angles()`. For the v0.8.0 radians/parameter changes to `add_noise`, `rotate_vertical`, `random_rotate_vertical`, `rotate_angles_vertical`, `random_translate_root`, `mirror`, and the removal of `auto_detect_lr_mapping`, see the *transforms: radians unification* section above.

### `pybvh.spatial_coord`

| Old name | New name |
|---|---|
| `frames_to_spatial_coord(...)` | `frames_to_spatial_coords(...)` |

### Modules with no changes

- `pybvh.io` — `read_bvh_file()`, `write_bvh_file()` unchanged
- `pybvh.rotations` — all 15 functions unchanged
- `pybvh.batch` — unchanged by v0.6.0 (v0.8.0 later moved `relative_scale_factor` to `pybvh.analysis` and the normalization trio to pybvh-ml — see above)
- `pybvh.df_to_bvh` — `df_to_bvh()` unchanged
- `pybvh.bvhplot` — `rest_pose()`, `frame()`, `trajectory()`, `render()`, `play()` unchanged
- `pybvh.bvhnode` — `BvhNode`, `BvhJoint`, `BvhRoot` unchanged
