# Changelog

All notable changes to **pybvh** are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.4.0] — 2026-03-31

### Added

- **`bvh.euler_orders`** — per-joint Euler rotation orders as strings (e.g. `['ZYX', 'ZYX', ...]`), eliminating the common `''.join(j.rot_channels)` boilerplate.
- **`bvh.edges`** — skeleton edge list as `(child_idx, parent_idx)` tuples in `joint_angles` index space.
- **`transforms.auto_detect_lr_pairs(bvh)`** — returns `list[tuple[int, int]]` index pairs ready for `mirror_angles()`, fixing the API mismatch where `auto_detect_lr_mapping` returned joint name pairs but `mirror_angles` expected integer indices.
- **`bvh == other`** — strict equality on `Bvh` objects via `__eq__`. Compares `joint_names`, `root_pos`, and `joint_angles` with `np.array_equal`.
- README "Philosophy" section clarifying pybvh's framework-agnostic stance and linking to pybvh-ml.
- 41 new tests (561 total).

### Fixed

- **`euler_to_rotmat` / `rotmat_to_euler`** now handle arbitrary batch dimensions (e.g. `(F, J, 3)` input). Previously limited to 2D `(N, 3)`, causing errors when passing `bvh.joint_angles` directly.
- **`_elementary_rotmat`** uses `np.ones_like` / `np.zeros_like` instead of shape-hardcoded arrays.

### Changed

- Extracted `pybvh/features.py` from `bvh.py` (ML pipeline feature functions).
- Extracted `pybvh/io.py` from `bvh.py` (consolidated `read_bvh_file` and `write_bvh_file`).
- Extracted `pybvh/transforms.py` from `bvh.py` (spatial augmentation transforms).
- Deleted `pybvh/read_bvh_file.py` (replaced by `io.py`).
- All public APIs remain unchanged — internal reorganization only.

---

## [0.3.1] — 2026-03-30

### Fixed

- Missing Python classifiers in `pyproject.toml`.
- Broken tutorial links in README.

---

## [0.3.0] — 2026-03-30

Major release covering four development phases since v0.2.0: rotation representations, performance, usability, and hardening.

### Added

#### Rotation Representations

- **6D rotation** (Zhou et al., CVPR 2019): `euler_to_rot6d`, `rot6d_to_euler`, `rotmat_to_rot6d`, `rot6d_to_rotmat`, plus `Bvh.get_frames_as_6d()` / `set_frames_from_6d()`.
- **Quaternions**: `euler_to_quat`, `quat_to_euler`, `rotmat_to_quat`, `quat_to_rotmat`, plus `Bvh.get_frames_as_quaternion()` / `set_frames_from_quaternion()`.
- **Axis-angle**: `euler_to_axisangle`, `axisangle_to_euler`, `rotmat_to_axisangle`, `axisangle_to_rotmat`, plus `Bvh.get_frames_as_axisangle()` / `set_frames_from_axisangle()`.
- **Quaternion SLERP**: `quat_slerp(q1, q2, t)` for spherical interpolation.
- **Euler order conversion**: `Bvh.single_joint_euler_angle()` and `Bvh.change_all_euler_orders()`.
- All rotation functions are batch-vectorized and available under `pybvh.rotations`.

#### Usability

- **Channel freeze protection**: `rot_channels` and `pos_channels` are frozen after `Bvh` construction — direct mutation raises `AttributeError`.
- **Frame operations**: `Bvh.slice_frames(start, end, step)`, `Bvh.concat(other)`, `Bvh.resample(target_fps)` (SLERP for rotations).
- **Joint extraction**: `Bvh.extract_joints(joint_names)` — removes unwanted joints and collapses offsets.
- **Skeleton retargeting**: `Bvh.change_skeleton()` now supports `name_mapping` dict and `strict` mode.
- **Uniform `inplace` API**: all mutation methods default to `inplace=False` (returns copy).

#### Batch Processing

- `read_bvh_directory(dirpath, pattern, sort, parallel)` — load all BVH files from a directory with optional threaded I/O.
- `batch_to_numpy(bvh_list, representation, pad)` — convert to NumPy arrays in any representation, with optional zero-padding.

#### ML Pipeline Features

- `Bvh.get_joint_velocities()` — finite differences of FK positions.
- `Bvh.get_joint_accelerations()` — second-order finite differences.
- `Bvh.get_angular_velocities()` — per-joint angular velocity via rotation matrix log map.
- `Bvh.get_root_relative_positions()` — root-subtracted positions per frame.
- `Bvh.get_root_trajectory()` — ground-plane position + heading sin/cos.
- `Bvh.get_foot_contacts()` — binary foot contact labels (velocity or height method, auto-detects foot joints).
- `Bvh.to_feature_array()` — one-stop flat NumPy array export combining rotations, velocities, and foot contacts.
- `compute_normalization_stats()`, `normalize_array()`, `denormalize_array()` — dataset-level normalization utilities.

#### Spatial Augmentation Transforms

- `transforms.mirror(bvh)` — left-right mirroring with auto-detected L/R pairs and lateral axis.
- `transforms.rotate_vertical(bvh, angle_deg)` / `random_rotate_vertical()`.
- `transforms.speed_perturbation(bvh, factor)` / `random_speed_perturbation()`.
- `transforms.add_joint_noise(bvh, sigma_deg)` — Gaussian noise with `[-180, 180]` wrapping.
- `transforms.translate_root(bvh, offset)` / `random_translate_root()`.
- `transforms.dropout_frames(bvh, drop_rate)` — frame dropout with SLERP interpolation.
- Array-level functions: `mirror_angles()`, `rotate_angles_vertical()`.
- All transforms also available as `Bvh` convenience methods.

#### New Properties

- `Bvh.node_index` — dict mapping node names to indices.
- `Bvh.joint_names` — list of non-end-site joint names.
- `Bvh.joint_count` — number of non-end-site joints.
- `Bvh.euler_column_names` — channel names for DataFrame/flat array mapping.

### Improved

- **Vectorized forward kinematics**: `frames_to_spatial_coord()` processes all frames in parallel per joint via batch matrix operations (replaces per-frame Python recursion).
- **Batch rotation matrix construction**: `batch_rotX/Y/Z` and `batch_get_premult_mat_rot` operate on `(N, 3)` arrays.
- **Pre-allocated parser**: `read_bvh_file` pre-allocates the frame array instead of O(n²) `np.append`.
- **Optimized DataFrame construction**: `get_df_constructor()` returns dict-of-arrays built directly from NumPy slices.
- Full type annotations across all source files with `@overload` on inplace methods. mypy clean.
- NumPy/SciPy docstrings on all public and private functions.
- 398 tests (up from 225).

### Breaking Changes

- `name2idx` property removed — use `node_index` instead (deprecated with warning).
- `single_joint_euler_angle` and `change_all_euler_orders` now default to `inplace=False` (previously `True`).

---

## [0.2.0] — 2026-02-17

**Not backward compatible** with v0.1.0 — internal data representation changed.

### Changed

- Replaced per-frame `frames` attribute with structured NumPy arrays: `root_pos` (2D `ndarray`) and `joint_angles` (3D `ndarray`).

### Fixed

- Matplotlib animation creation now checks if ffmpeg is present on the system.

---

## [0.1.0] — 2026-02-12

Initial release.

### Added

- BVH I/O: reader and writer for standard `.bvh` files.
- Kinematics: utilities for processing skeletal hierarchy and motion data.
- Visualization: plotting with Matplotlib.
- Rotation representations: utilities to convert between 3D rotation formats.
- NumPy array and optional Pandas DataFrame support.
- Python >= 3.9, NumPy >= 1.21, Matplotlib >= 3.7, Pandas >= 1.5 (optional).

---

[0.4.0]: https://github.com/VictorS-67/pybvh/compare/v0.3.1...HEAD
[0.3.1]: https://github.com/VictorS-67/pybvh/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/VictorS-67/pybvh/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/VictorS-67/pybvh/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/VictorS-67/pybvh/releases/tag/v0.1.0
