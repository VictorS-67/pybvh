# AI_ONBOARDING.md — pybvh

> **Purpose of this document**: Give any new AI agent (or human contributor) a complete, precise understanding of the pybvh codebase—its goals, architecture, data flow, every module, class, method, and design decision—so they can modify or extend it without guessing.

---

## 1. Project Identity

| Field | Value |
|---|---|
| **Name** | pybvh |
| **Language** | Python 3 (>= 3.9) |
| **Dependencies** | `numpy` (required), `matplotlib` (required), `pandas` (optional), `opencv-python` (optional, fast render), `k3d` (optional, Jupyter), `vedo` (optional, desktop) |
| **Primary use-case** | Reading, writing, and manipulating BVH (Biovision Hierarchy) motion capture files — serving ML pipelines, biomechanics research, game dev, and any workflow that consumes skeleton animation data |
| **Design principles** | **Fast** (NumPy-vectorised, pre-allocated arrays), **Lightweight** (minimal code surface, no ML framework deps), **Self-contained** (no scipy, no PyTorch, no TensorFlow) |
| **Version** | 0.4.0 |
| **Package** | Published on PyPI as `pybvh`. Install via `pip install pybvh`. Optional: `pip install "pybvh[pandas]"` |
| **CI/CD** | GitHub Actions: test workflow (push/PR, Python 3.9–3.12) + publish workflow (PyPI on release) + docs workflow (MkDocs to GitHub Pages on push to main) |
| **Type safety** | Full type annotations on all source files, `@overload` on inplace methods, mypy clean |
| **Tests** | 954 tests via pytest, covering all modules, edge cases, and battle tests across 3 real-world datasets |
| **Documentation** | MkDocs + mkdocstrings + Material theme, auto-deployed to GitHub Pages |

---

## 2. What is a BVH File?

A BVH file is a plain-text motion capture format with two sections:

### 2.1 HIERARCHY section
Defines a skeleton as a tree of joints. Each joint has:
- **OFFSET** — 3 floats (x, y, z) describing the bone vector from parent to this joint in the rest pose.
- **CHANNELS** — either 6 (root: 3 position + 3 rotation) or 3 (other joints: 3 rotation). Channel names encode both axis and type, e.g. `Zrotation Yrotation Xrotation`.
- **End Site** — leaf nodes with only an OFFSET (no channels). They represent the tip of a terminal bone.

### 2.2 MOTION section
- `Frames: N` — number of frames.
- `Frame Time: T` — seconds per frame (e.g. `0.033333` for 30 fps).
- N lines of floats — each line is one frame. Column order matches the depth-first traversal order of channels declared in the HIERARCHY.

**Key insight**: The HIERARCHY gives you the skeleton topology (offsets + rotation orders). The MOTION gives you per-frame Euler angles (and root translation). Combining them via forward kinematics yields 3D joint positions.

---

## 3. High-Level Architecture & Data Flow

```
                         ┌──────────────────────┐
   .bvh file ──────────► │  read_bvh_file()     │ ──────► Bvh object
                         └──────────────────────┘            │
                                                              │
   directory of ──────► read_bvh_directory() ──► list[Bvh]   │
   .bvh files                    │                            │
                                 ▼                            │
                        batch_to_numpy() ──► NumPy arrays     │
                                                              │
                    ┌─────────────────────────────────────────┤
                    │                                         │
                    ▼                                         ▼
            bvh.get_df_constructor()              bvh.get_spatial_coord()
                    │                                         │
                    ▼                                         ▼
            pd.DataFrame(...)                      NumPy array (3D positions)
                    │                                         │
                    ▼                                         ▼
              df_to_bvh() ──► Bvh object            plot.frame()
                    │                               plot.render() / plot.play()
                    ▼
            bvh.to_bvh_file() ──► .bvh file
```

**Central object**: `Bvh` — everything flows through it.

---

## 4. File-by-File Module Reference

### 4.1 `pybvh/__init__.py`
Public API surface. Exports:
```python
from .bvh import Bvh
from .io import read_bvh_file, write_bvh_file
from .df_to_bvh import df_to_bvh
from .spatial_coord import frames_to_spatial_coord
from .batch import (read_bvh_directory, batch_to_numpy,
                    compute_normalization_stats, normalize_array, denormalize_array)
from . import plot
from . import rotations
from . import transforms
from . import features
```

### 4.1b `pybvh/rotations.py` — Rotation Representation Conversions

Pure NumPy, batch-vectorised rotation conversions. No scipy dependency.

**Supported representations**:
- **Euler angles**: `(*, 3)` — degrees or radians, any of the 6 Tait-Bryan orders
- **Rotation matrices**: `(*, 3, 3)`
- **6D rotation** (Zhou et al., CVPR 2019): `(*, 6)` — continuous representation
- **Quaternions**: `(*, 4)` — `(w, x, y, z)` scalar-first, canonical `w >= 0`
- **Axis-angle**: `(*, 3)` — unit rotation axis scaled by rotation angle in radians

**Core functions**:

| Function | Input → Output |
|---|---|
| `euler_to_rotmat(angles, order, degrees)` | `(*, 3)` → `(*, 3, 3)` |
| `rotmat_to_euler(R, order, degrees)` | `(*, 3, 3)` → `(*, 3)` |
| `rotmat_to_rot6d(R)` | `(*, 3, 3)` → `(*, 6)` |
| `rot6d_to_rotmat(rot6d)` | `(*, 6)` → `(*, 3, 3)` (Gram-Schmidt) |
| `rotmat_to_quat(R)` | `(*, 3, 3)` → `(*, 4)` (Shepperd method) |
| `quat_to_rotmat(q)` | `(*, 4)` → `(*, 3, 3)` |
| `rotmat_to_axisangle(R)` | `(*, 3, 3)` → `(*, 3)` (log map) |
| `axisangle_to_rotmat(aa)` | `(*, 3)` → `(*, 3, 3)` (Rodrigues' formula) |
| `quat_slerp(q1, q2, t)` | `(*, 4)` × `(*, 4)` × scalar/array → `(*, 4)` |

**Convenience wrappers** (go through rotmat as intermediate):
`euler_to_rot6d`, `rot6d_to_euler`, `euler_to_quat`, `quat_to_euler`, `euler_to_axisangle`, `axisangle_to_euler`

**Internal helpers**: `_normalize`, `_elementary_rotmat` (single-axis batch rotation), `_extract_euler`

**Conventions**:
- Euler: intrinsic rotations, pre-multiplication `R = R1 @ R2 @ R3`
- Quaternion: `(w, x, y, z)` scalar-first, canonical `w >= 0`
- Axis-angle: zero vector = identity rotation; norm = rotation angle in `[0, π]`

### 4.2 `pybvh/bvhnode.py` — Node Class Hierarchy

Three classes forming an inheritance chain:

```
BvhNode  (end sites)
  └── BvhJoint  (interior joints)
        └── BvhRoot  (root joint — exactly one per skeleton)
```

#### `BvhNode`
- **Represents**: End sites (leaf bones with no rotation channels).
- **Attributes**: `name` (str), `offset` (np.array of 3 floats), `parent` (BvhNode | None).
- **Key methods**: `is_end_site()` → `True`, `is_root()` → `False`.
- All setters have validation (type-checked via `@property`).

#### `BvhJoint(BvhNode)`
- **Represents**: Interior joints with rotation channels.
- **Additional attributes**: `rot_channels` (list of 3 chars, e.g. `['Z', 'Y', 'X']`), `children` (list of BvhNode/subclass).
- `rot_channels` setter calls `_check_channels()` which validates it is a permutation of `['X', 'Y', 'Z']`.
- **Freeze mechanism**: After a `Bvh` object is constructed, `_frozen = True` is set on all joints. Direct assignment to `rot_channels` raises `AttributeError` — users must use `Bvh.single_joint_euler_angle()` or `Bvh.change_all_euler_orders()`. Internal code uses `_set_rot_channels_internal()` to bypass the freeze.
- `is_end_site()` → `False`.

#### `BvhRoot(BvhJoint)`
- **Represents**: The root joint (has position channels in addition to rotation).
- **Additional attribute**: `pos_channels` (list of 3 chars, e.g. `['X', 'Y', 'Z']`). Also frozen after Bvh construction.
- `is_root()` → `True`, `parent` is always `None`.

**Important**: The skeleton is a **tree**. You traverse it from `root` via `.children`, or walk up via `.parent`. The `Bvh.nodes` list is a **flat depth-first list** of all nodes (joints + end sites).

### 4.3 `pybvh/bvh.py` — The `Bvh` Class (Central Container)

#### Constructor
```python
Bvh(nodes=[BvhRoot()], root_pos=None, joint_angles=None,
    frame_frequency=0)
```

The constructor validates that `root.pos_channels == ['X', 'Y', 'Z']` (the only layout supported). If `root_pos` or `joint_angles` are `None`, they default to empty arrays with the correct shape. After construction, all node channel attributes are frozen.

#### Core Attributes (all property-validated)
| Attribute | Type | Description |
|---|---|---|
| `nodes` | `list[BvhNode]` | Flat depth-first list of all skeleton nodes |
| `root_pos` | `np.ndarray` (2D) | Shape `(F, 3)`. Root translation per frame |
| `joint_angles` | `np.ndarray` (3D) | Shape `(F, J, 3)`. Euler angles in degrees per joint per frame. Joint order follows `nodes` (end sites excluded) |
| `frame_frequency` | `float` | Seconds per frame (e.g. `1/30`) |
| `frame_count` | `int` (read-only) | Computed property: `len(self.root_pos)` |
| `euler_column_names` | `list[str]` (read-only) | Computed property generating BVH-style column names on the fly |
| `root` | `BvhRoot` | Shortcut to `nodes[0]` |
| `node_index` | `dict[str, int]` | Maps node name → integer index into `nodes`. One entry per node (including end sites). |
| `joint_names` | `list[str]` (read-only) | Names of non-end-site joints in topological order |
| `joint_count` | `int` (read-only) | Number of non-end-site joints |
| `euler_orders` | `list[str]` (read-only) | Per-joint Euler rotation orders as strings (e.g. `['ZYX', 'ZYX', ...]`). Ready for use with `pybvh.rotations` functions. Recomputed on each access. |
| `edges` | `list[tuple[int, int]]` (read-only) | Skeleton edge list as `(child_idx, parent_idx)` tuples in `joint_angles` index space. J joints → J-1 edges. Recomputed on each access. |

#### Data layout
Motion data is stored as two structured arrays:
- `root_pos[f]` — the `(X, Y, Z)` position of the root at frame `f`.
- `joint_angles[f, j]` — the 3 Euler angles (degrees) for joint `j` at frame `f`, in the order given by `nodes[j].rot_channels`.

End sites have NO entries in `joint_angles` (they have no channels). The joint index `j` corresponds to the j-th non-end-site node in `nodes`.

There is **no flat `.frames` property** — code should use `root_pos` and `joint_angles` directly.

#### Key Methods

| Method | Description |
|---|---|
| `__eq__(other)` | Strict equality: same `joint_names` and `np.array_equal` on `root_pos` and `joint_angles`. Returns `NotImplemented` for non-Bvh types. |
| `to_bvh_file(filepath, verbose=True)` | Serialize back to a `.bvh` file. Uses `%.6f` precision. Delegates to `io.write_bvh_file`. |
| `get_spatial_coord(frame_num=-1, centered="world")` | 3D joint positions via forward kinematics. Returns `(N, 3)` or `(F, N, 3)`. |
| `get_rest_pose(mode='euler'\|'coordinates')` | Rest/T-pose (all angles zero, root at origin). |
| `get_df_constructor(mode='euler'\|'coordinates', centered="world")` | Dict-of-arrays ready for `pd.DataFrame()`. |
| `hierarchy_info_as_dict()` | Deep-copied dict describing the skeleton tree. |
| `change_skeleton(new_skeleton, name_mapping=None, strict=False, inplace=False)` | Retarget: copy offsets from another skeleton. Supports name mapping. |
| `scale_skeleton(scale, inplace=False)` | Scale all offsets by a float or 3-element array. |
| `copy()` | Deep copy via `copy.deepcopy()`. |
| `get_frames_as_rotmat()` | → `(root_pos, joint_rotmats (F,J,3,3), joints)` |
| `get_frames_as_6d()` | → `(root_pos, joint_rot6d (F,J,6), joints)` |
| `get_frames_as_quaternion()` | → `(root_pos, joint_quats (F,J,4), joints)` |
| `get_frames_as_axisangle()` | → `(root_pos, joint_aa (F,J,3), joints)` |
| `set_frames_from_6d(root_pos, joint_rot6d, inplace=False)` | Set motion from 6D data. |
| `set_frames_from_quaternion(root_pos, joint_quats, inplace=False)` | Set motion from quaternion data. |
| `set_frames_from_axisangle(root_pos, joint_aa, inplace=False)` | Set motion from axis-angle data. |
| `single_joint_euler_angle(joint, new_order, inplace=False)` | Change Euler order of one joint. |
| `change_all_euler_orders(new_order, inplace=False)` | Change Euler order of all joints. |
| `slice_frames(start, end, step)` | Extract frame subset. Adjusts `frame_frequency` when step ≠ 1. |
| `concat(other)` | Concatenate motion from another Bvh. Validates skeleton compatibility. |
| `resample(target_fps)` | Resample motion to different frame rate (SLERP for rotations). |
| `extract_joints(joint_names)` | Extract joint subset, collapse offsets, rebuild hierarchy. |
| `get_joint_velocities(centered, in_frames, coords)` | Finite-difference joint velocities. Returns `(F-1, N, 3)`. Delegates to `features.get_joint_velocities`. |
| `get_joint_accelerations(centered, in_frames, coords)` | Second-order finite-difference accelerations. Returns `(F-2, N, 3)`. Delegates to `features.get_joint_accelerations`. |
| `get_angular_velocities(in_frames)` | Per-joint angular velocity via rotation matrix log map. Returns `(F-1, J, 3)`. Delegates to `features.get_angular_velocities`. |
| `get_root_relative_positions(centered, coords)` | All joint positions relative to root each frame. Returns `(F, N, 3)`. Delegates to `features.get_root_relative_positions`. |
| `get_root_trajectory(up_axis)` | Root ground-plane position + heading sin/cos. Returns `(F, 4)`. Delegates to `features.get_root_trajectory`. |
| `get_foot_contacts(foot_joints, method, threshold, centered, coords)` | Binary foot contact labels. Auto-detects foot joints. Returns `(F, num_feet)`. Delegates to `features.get_foot_contacts`. |
| `to_feature_array(representation, include_root_pos, include_velocities, include_foot_contacts, ...)` | One-stop ML export: flat `(F, D)` array composing rotations, velocities, and contacts. Delegates to `features.to_feature_array`. |
| `translate_root(offset, inplace=False)` | Shift root position by constant 3D offset. Delegates to `transforms.translate_root`. |
| `add_joint_noise(sigma_deg, sigma_pos, rng, inplace=False)` | Add Gaussian noise to angles/position. Delegates to `transforms.add_joint_noise`. |
| `speed_perturbation(factor)` | Change motion speed by resampling. Delegates to `transforms.speed_perturbation`. |
| `dropout_frames(drop_rate, rng, inplace=False)` | Replace random frames with SLERP interpolation. Delegates to `transforms.dropout_frames`. |
| `rotate_vertical(angle_deg, up_axis, inplace=False)` | Rotate motion around vertical axis. Delegates to `transforms.rotate_vertical`. |
| `mirror(left_right_mapping, lateral_axis, inplace=False)` | Left-right mirror the motion. Delegates to `transforms.mirror`. |

#### The `centered` Parameter (appears throughout the codebase)
Three modes controlling how root position is handled:
- `"world"` — Root at the actual saved coordinates from the BVH file.
- `"skeleton"` — Root forced to `(0, 0, 0)` in every frame.
- `"first"` — First frame's root is at `(0, 0, 0)`, subsequent frames move relative to that.

### 4.4 `pybvh/batch.py` — Batch Loading & NumPy Export

| Function | Description |
|---|---|
| `read_bvh_directory(dirpath, pattern="*.bvh", sort=True, parallel=False, max_workers=None)` | Load all matching BVH files from a directory. Optional `ThreadPoolExecutor` parallelism. Returns `list[Bvh]`. |
| `batch_to_numpy(bvh_list, representation="euler", include_root_pos=True, pad=False, pad_value=0.0)` | Convert list of Bvh to NumPy arrays. Validates skeleton compatibility. `pad=True` → `(B, F_max, D)`, `pad=False` → `list[(F_i, D)]`. Representations: `euler`, `6d`, `quaternion`, `axisangle`, `rotmat`. |
| `compute_normalization_stats(bvh_list, representation, include_root_pos)` | Per-channel mean/std across a dataset. Returns `{"mean": (D,), "std": (D,)}`. Zero-std channels set to 1.0. |
| `normalize_array(data, stats)` | Apply z-score normalization: `(data - mean) / std`. |
| `denormalize_array(data, stats)` | Reverse z-score normalization: `data * std + mean`. |

### 4.5 `pybvh/io.py` — BVH File I/O

Consolidates BVH file reading and writing into a single module (replaces the former `read_bvh_file.py`).

| Function | Description |
|---|---|
| `read_bvh_file(filepath)` → `Bvh` | Parse a `.bvh` file into a `Bvh` object (moved from `read_bvh_file.py`) |
| `write_bvh_file(bvh, filepath, verbose=True)` → `None` | Write a `Bvh` object to a `.bvh` file (extracted from `Bvh.to_bvh_file`) |

**Private helpers**:
- `_extract_bvh_file_info(filepath)` — returns `(nodes, frame_array, frame_frequency)`. Opens the file and reads line by line: parses `ROOT`, `JOINT`, `End Site` blocks constructing node objects, tracks parent-child relationships using a `parent_depth` counter, and after `Frame Time:` reads all frame data into a pre-allocated NumPy array.
- `_get_offset_channels(node)` — format a node's offset and channel declarations for BVH output.

After receiving the flat array, `read_bvh_file` splits it into `root_pos` (first 3 columns) and `joint_angles` (remaining columns reshaped to `(F, J, 3)`) before constructing the `Bvh` object.

**Performance**: Frame data is pre-allocated using `np.empty((frame_count, num_channels))`. Rows are filled via direct index assignment, giving O(n) performance.

### 4.6 `pybvh/spatial_coord.py` — Forward Kinematics

#### `frames_to_spatial_coord(nodes_container, root_pos=None, joint_angles=None, centered="world")`

**Output shape**: `(F, N, 3)` for multiple frames, or `(N, 3)` for a single frame, where N = total number of nodes (including end sites).

**Algorithm** (vectorized across all frames):
1. Pre-allocate output arrays: `positions (F, N, 3)` and `acc_rotmats (F, N, 3, 3)`.
2. Build topology arrays once from the node list.
3. Convert all angles to radians in one vectorized call.
4. For each node (in topological order), process **all frames at once** using batch matrix operations.
5. Add root position (unless skeleton-centered), broadcast across all nodes.

### 4.7 `pybvh/df_to_bvh.py` — DataFrame → Bvh Conversion

**Entry point**: `df_to_bvh(hier, df)` → `Bvh`

`hier` can be a list of BvhNode objects or a dict describing the hierarchy. Validates DataFrame columns match hierarchy, reorders if needed, then constructs a `Bvh` object.

### 4.8 `pybvh/tools.py` — Utility Functions

| Function | Purpose |
|---|---|
| `are_permutations(str1, str2)` | Check if two strings are permutations. Used to validate channel orders. |
| `test_file(filepath)` | Validate path exists and has `.bvh` extension. |
| `rotX/Y/Z(angle)` | 3×3 rotation matrices for a single axis (scalar, radians). |
| `get_premult_mat_rot(angles, order)` | Compose 3 Euler rotation matrices (scalar version). |
| `batch_rotX/Y/Z(angles)` | Batch rotation matrices: `(N,)` radians → `(N, 3, 3)`. |
| `batch_get_premult_mat_rot(angles, order)` | Batch Euler → rotation matrices: `(N, 3)` radians → `(N, 3, 3)`. Used by vectorized FK. |
| `get_main_direction(coord_array, tol)` | Return signed axis string (e.g. `'+y'`) for the dominant component, or `None` if norm < `tol`. |
| `extract_sign(ax)` | Return `True` if axis string is positive, `False` if negative. |
| `get_forw_up_axis(bvh_object, frame)` | Infer forward/upward axes using joint name heuristics (up) and left-right symmetry from rest-pose offsets (forward via cross product). Validates input, guarantees orthogonal axes. |
| `get_up_axis_index(bvh_object, frame)` | Return integer index (0=x, 1=y, 2=z) of the upward axis. |

### 4.9 `pybvh/plot/` — Visualization Package

Multi-backend visualization with five public functions:

| Function | Purpose |
|---|---|
| `plot.rest_pose(bvh, ...)` | Plot the T-pose / bind pose (all angles zero, root at origin). |
| `plot.frame(bvh, frame, centered, camera, ...)` | Static 3D matplotlib snapshot. Accepts single Bvh or list for side-by-side comparison. Camera presets: `"front"`, `"side"`, `"top"`, or `(azim, elev)`. |
| `plot.render(bvh, filepath, backend, camera, resolution, sync, ...)` | Export animation to video/GIF/HTML. OpenCV backend (~1000 fps) with matplotlib fallback. `sync="pad"` continues to the longest clip. |
| `plot.play(bvh, backend, sync, resolution, ...)` | Playback with 3-tier auto-detection: k3d (Jupyter), vedo (desktop), then OpenCV inline video (notebook) or matplotlib window (script). Auto-subsamples to 30fps for smooth playback. |
| `plot.trajectory(bvh, ...)` | 2D top-down root trajectory plot. Per-skeleton up-axis detection for correct projection in multi-skeleton overlays. |

**Submodules:**
- `_common.py` — Shared helpers: `get_skeleton_lines()`, `normalize_input()`, `compute_unified_limits()`, `get_camera_angles()`, `build_view_matrix()`, `ortho_project()`, `align_frame_counts()`
- `_matplotlib.py` — Matplotlib backend (frame, render, play, trajectory)
- `_opencv.py` — OpenCV fast render via orthographic 2D projection (~1000x faster than matplotlib for video export)
- `_k3d.py` — k3d Jupyter interactive backend with Play/slider widgets
- `_vedo.py` — vedo desktop interactive backend with keyboard controls

**Camera math:** `build_view_matrix()` replicates matplotlib's look-at camera algorithm (eye from spherical coords, cross-product u/v/w axes) so that both backends produce identical views. Front-view detection uses left-right symmetry to find the lateral axis, derives forward via cross product, then positions the camera facing the skeleton's chest.

**Optional dependencies:** `opencv-python` (fast render), `k3d` (notebook), `vedo` (desktop). Install via `pip install pybvh[opencv]`, `pybvh[interactive]`, `pybvh[viewer]`, or `pybvh[all-viz]`.

Axis detection functions (`get_forw_up_axis`, `extract_sign`) live in `tools.py`. `get_forw_up_axis` is re-exported from `plot.__init__` for backward compatibility.

### 4.10 `pybvh/transforms.py` — Spatial Augmentation Transforms

Data augmentation transforms for motion ML pipelines. All functions operate on `Bvh` objects and follow the `inplace=False` convention.

| Function | Description |
|---|---|
| `translate_root(bvh, offset, inplace=False)` | Add constant 3D offset to all root positions |
| `random_translate_root(bvh, range_xyz, rng)` | Random uniform translation per axis |
| `add_joint_noise(bvh, sigma_deg, sigma_pos, rng, inplace=False)` | Gaussian noise on joint angles (degrees) and optionally root position. Angles wrapped to [-180, 180] after noise |
| `speed_perturbation(bvh, factor)` | Resample to change speed (factor>1 = faster = fewer frames). Always returns new Bvh |
| `random_speed_perturbation(bvh, factor_range, rng)` | Random speed factor from uniform range |
| `dropout_frames(bvh, drop_rate, rng, inplace=False)` | Replace random frames with SLERP-interpolated values. Same frame count output |
| `rotate_vertical(bvh, angle_deg, up_axis, inplace=False)` | Rotate motion around vertical axis. Only modifies root position and root joint angles |
| `random_rotate_vertical(bvh, angle_range, up_axis, rng)` | Random vertical rotation from uniform range |
| `auto_detect_lr_mapping(bvh)` | Detect left/right joint pairs via "Left"↔"Right" and "L"↔"R" heuristics. Returns `dict[str, str]` of joint names |
| `auto_detect_lr_pairs(bvh)` | Returns `list[(left_idx, right_idx)]` index tuples in `joint_angles` index space, ready for `mirror_angles()`. Wraps `auto_detect_lr_mapping` |
| `mirror(bvh, left_right_mapping, lateral_axis, inplace=False)` | Reflect motion across lateral plane: negate lateral positions/offsets, swap L/R data, negate non-lateral Euler components |

**Mirror algorithm** (derived from reflection matrix math `R' = S @ R @ S`):
1. Negate root position lateral component
2. Negate ALL node offsets' lateral component
3. Swap offsets between L/R paired nodes (including end-site children)
4. Swap joint angle columns for L/R pairs
5. Negate Euler angle components whose rotation axis is NOT the lateral axis

**NumPy-level API** (for users working with pre-extracted arrays):

| Function | Description |
|---|---|
| `rotate_angles_vertical(joint_angles, root_pos, angle_deg, up_idx, root_order)` | Rotate root position and root Euler angles around the up axis. Returns `(new_angles, new_root_pos)` |
| `mirror_angles(joint_angles, root_pos, lr_joint_pairs, lateral_idx, rot_channels)` | Negate lateral root position, swap L/R columns, negate non-lateral Euler components. Returns `(new_angles, new_root_pos)`. Does NOT modify skeleton offsets |

The Bvh-level `rotate_vertical` and `mirror` delegate their array math to these functions.

**Thin wrappers on `Bvh`**: Each transform has a 3-5 line wrapper method on the `Bvh` class that delegates to `transforms.py`.

### 4.11 `pybvh/features.py` — Standalone ML Pipeline Feature Functions

Standalone functions for extracting ML pipeline features from `Bvh` objects. All take `bvh: Bvh` as their first argument. The corresponding `Bvh` class methods are thin wrappers that delegate to these functions.

| Function | Description |
|---|---|
| `get_joint_velocities(bvh, centered, in_frames, coords)` | Per-joint position velocities via finite differences |
| `get_joint_accelerations(bvh, centered, in_frames, coords)` | Per-joint accelerations via second-order finite differences |
| `get_angular_velocities(bvh, in_frames)` | Per-joint angular velocities via rotation matrix log map |
| `get_root_relative_positions(bvh, centered, coords)` | Joint positions relative to root at each frame |
| `get_root_trajectory(bvh, up_axis)` | Root ground-plane position + heading (sin/cos) |
| `get_foot_contacts(bvh, foot_joints, method, threshold, centered, coords)` | Binary foot contact labels per frame |
| `to_feature_array(bvh, representation, ...)` | Flat `(F, D)` feature array for ML pipelines |

---

## 5. Data Representation Details

### 5.1 Motion Data: `root_pos` + `joint_angles`
- **`root_pos`**: Shape `(F, 3)`. Column order always `(X, Y, Z)`.
- **`joint_angles`**: Shape `(F, J, 3)`. Euler angles in degrees. `J` = number of non-end-site nodes.

Example for `bvh_example.bvh`: `root_pos.shape = (56, 3)`, `joint_angles.shape = (56, 24, 3)`.

### 5.2 Spatial Coordinates Output
- Shape: `(N, 3)` for a single frame, `(F, N, 3)` for multiple frames.
- N = total number of nodes including end sites (29 for `bvh_example.bvh`).
- Order matches `Bvh.nodes` list order (depth-first).
- `node_index` maps `"JointName"` → integer index.

---

## 6. Forward Kinematics — The Math

Given a joint `J` with offset, parent's accumulated rotation `R_parent`, parent's position `P_parent`, and J's own rotation `R_J`:

$$P_J = R_{parent} \cdot \text{offset}_J + P_{parent}$$
$$R_{acc,J} = R_{parent} \cdot R_J$$

Rotation matrix from Euler angles uses **intrinsic** rotations with **pre-multiplication**:
$$R = R_{\text{first}} \cdot R_{\text{second}} \cdot R_{\text{third}}$$

where the order comes from the joint's `rot_channels`.

---

## 7. Coding Conventions & Patterns

1. **Property validation**: All core attributes use `@property` with setters that type-check inputs.
2. **Full type annotations**: All source files use `from __future__ import annotations`, `npt.NDArray`, `@overload` for inplace methods. mypy passes with 0 errors.
3. **NumPy throughout**: All numerical data as NumPy arrays. No ML framework dependencies.
4. **Deep copy safety**: `Bvh.copy()` uses `copy.deepcopy()`. `hierarchy_info_as_dict()` returns a deep copy.
5. **Channel freeze**: After `Bvh.__init__`, `rot_channels` and `pos_channels` are frozen. Mutation must go through Bvh methods.
6. **Uniform `inplace` convention**: All mutation methods default to `inplace=False` (returns copy). `inplace=True` modifies self, returns `None`.
7. **No pandas dependency**: pybvh never imports pandas. `get_df_constructor()` returns a dict-of-arrays.
8. **No ML framework dependencies**: Output is always NumPy. Users convert to PyTorch/TensorFlow themselves.
9. **Naming**: `_private` prefix for internal methods. `snake_case` everywhere.
10. **Errors**: Mix of `ValueError`, `Exception`, and `AttributeError`.

---

## 8. Testing Conventions

- **Framework**: pytest
- **Fixture files**: `bvh_data/bvh_example.bvh` (primary), plus `bvh_test1.bvh`, `bvh_test2.bvh`, `bvh_test3.bvh`, `standard_skeleton.bvh`
- **Numerical assertions**: `np.testing.assert_allclose` with `atol=1e-4` to `1e-10` depending on precision needs. File round-trips use `atol=1e-5` (due to `%.6f` formatting).
- **Round-trip tests**: BVH → file → BVH, BVH → DataFrame → BVH, BVH → {6D, quaternion, axis-angle} → BVH, Euler order conversion → re-conversion.
- **Test files**:
  - `tests/test_bvh.py` — File I/O, hierarchy, spatial coordinates, DataFrame conversion, skeleton operations, batch processing, freeze preservation, ML pipeline features (velocities, foot contacts, normalization, feature export), edge cases.
  - `tests/test_rotations.py` — All conversion paths, gimbal lock, 180° SLERP, analytical values.
- **Run command**: `conda run -n pybvh pytest tests/ -v`
- **Current count**: 954 tests, all passing (611 unit + 343 battle tests across 15 representative files from 3 datasets; + 23k-file smoke test via `--runslow`).
- **Note**: `tests/test_transforms_battle.py` uses private datasets from `internal_bvh_data/` and is gitignored — never publish or share this file.

---

## 9. Sample BVH Data Files

| File | Joints | Frames | FPS | Euler Order | Purpose |
|---|---|---|---|---|---|
| `bvh_example.bvh` | 24 | 56 | 30 | ZYX | Main test file |
| `bvh_test1.bvh` | 24 | 56 | 30 | ZYX | Additional test |
| `bvh_test2.bvh` | 23 | 61 | 120 | YXZ | Different order/fps |
| `bvh_test3.bvh` | 60 | 100 | 120 | Mixed | Large skeleton, mixed orders |
| `standard_skeleton.bvh` | 24 | 1 | - | ZYX | Reference skeleton for retargeting |

---

## 10. Quick Reference: Common Operations

```python
from pybvh import read_bvh_file, df_to_bvh, Bvh, rotations
from pybvh import read_bvh_directory, batch_to_numpy, plot
import pandas as pd

# Read
bvh = read_bvh_file("walk.bvh")

# Inspect
bvh.root_pos.shape          # (F, 3)
bvh.joint_angles.shape      # (F, J, 3)
bvh.joint_names              # ['Hips', 'Spine', ...]
bvh.joint_count              # 24
bvh.node_index['Hips']       # 0

# Spatial coordinates (forward kinematics)
coords = bvh.get_spatial_coord(centered="world")  # (F, N, 3)

# Rotation representations
root_pos, rot6d, joints = bvh.get_frames_as_6d()
root_pos, quats, joints = bvh.get_frames_as_quaternion()
root_pos, aa, joints    = bvh.get_frames_as_axisangle()

# Set frames back (inplace=False returns new Bvh)
bvh2 = bvh.set_frames_from_6d(root_pos, rot6d)

# Euler order conversion
bvh_xyz = bvh.change_all_euler_orders('XYZ')

# Frame operations
clip = bvh.slice_frames(10, 50)
combined = bvh.concat(other_bvh)
bvh_30fps = bvh.resample(30)

# Skeleton operations
bvh_scaled = bvh.scale_skeleton(0.01)
upper = bvh.extract_joints(["Hips", "Spine", "Neck", "Head"])

# Batch loading for ML
clips = read_bvh_directory("dataset/", parallel=True)
data = batch_to_numpy(clips, representation="6d", pad=True)  # (B, F_max, D)

# Standalone rotations
R = rotations.euler_to_rotmat([30, 45, 60], 'ZYX', degrees=True)
q = rotations.rotmat_to_quat(R)
q_mid = rotations.quat_slerp(q1, q2, t=0.5)

# ML pipeline features
vel = bvh.get_joint_velocities()                     # (F-1, N, 3) units/second
acc = bvh.get_joint_accelerations()                   # (F-2, N, 3) units/second^2
ang_vel = bvh.get_angular_velocities()                # (F-1, J, 3) radians/second
rel_pos = bvh.get_root_relative_positions()           # (F, N, 3) root-relative
traj = bvh.get_root_trajectory()                      # (F, 4) ground pos + heading
contacts = bvh.get_foot_contacts()                    # (F, num_feet) binary
feat = bvh.to_feature_array(representation="6d",      # (F, D) one-stop export
         include_velocities=True, include_foot_contacts=True)

# Normalization (dataset-level)
from pybvh import compute_normalization_stats, normalize_array, denormalize_array
stats = compute_normalization_stats(clips, representation="6d")
normalized = normalize_array(data, stats)
recovered = denormalize_array(normalized, stats)

# DataFrame (pybvh does NOT import pandas)
df = pd.DataFrame(bvh.get_df_constructor(mode='euler'))
bvh2 = df_to_bvh(bvh.nodes, df)

# Write & plot
bvh.to_bvh_file("output.bvh")
plot.frame(bvh, 0)
```

---

## 11. Extending the Codebase — Guidelines

1. **Add new rotation representations** in `rotations.py`. Keep them as pure NumPy batch-vectorized functions.
2. **Add new Bvh methods** in `bvh.py`. Follow the existing pattern: validate inputs in properties, delegate to helper modules.
3. **`euler_column_names` is computed**: Any operation that changes `rot_channels` only needs to update the node and `joint_angles` — column names reflect the change automatically.
4. **Test with fixtures**: Add tests using the existing fixtures. Include numerical assertions with known expected values.
5. **No new dependencies** unless absolutely necessary. Output NumPy arrays — let users convert to their ML framework of choice.
6. **Performance**: Pre-allocate arrays, vectorize with NumPy, avoid Python loops over frames.
7. **Type all new code**: Use `npt.NDArray[np.float64]` for returns, `npt.ArrayLike` for inputs. Add `@overload` for inplace methods.
8. **Caching opportunity**: `euler_orders` and `edges` properties recompute on every access (they traverse the node list). This is fine for single calls but wasteful in hot loops. If profiling shows these as bottlenecks, consider caching with invalidation on skeleton mutation (e.g. `change_all_euler_orders`, `extract_joints`).

---

## 12. Ecosystem & Scope Boundary

pybvh is the **foundation layer** in a two-library ecosystem:

```
pybvh-ml  (ML bridge: tensor packing, augmentation pipelines, PyTorch Datasets)
    │
    ▼
  pybvh   (BVH foundation: parsing, rotation math, transforms, motion analysis)
    │
    ▼
  NumPy
```

**pybvh never imports or knows about pybvh-ml.** The dependency flows one way.

### Scope rule
If a feature is useful to a biomechanics researcher, game developer, or anyone working with BVH data outside ML — it belongs in pybvh. If it only makes sense in an ML training context (tensor layouts, Dataset classes, augmentation pipelines, HDF5 export) — it belongs in pybvh-ml.

### API surface that pybvh-ml relies on
pybvh-ml is a primary consumer of pybvh's public API. When modifying pybvh, be aware that these entry points are used downstream:
- `bvh.root_pos`, `bvh.joint_angles`, `bvh.joint_count`, `bvh.joint_names` — data access
- `bvh.get_frames_as_quaternion()`, `bvh.get_frames_as_6d()`, etc. — representation conversion
- `bvh.euler_orders` — per-joint Euler order strings
- `bvh.edges` — skeleton edge list as index tuples
- `bvh.nodes`, `bvh.node_index` — skeleton topology
- `pybvh.transforms.auto_detect_lr_pairs()` — L/R index pair detection
- `pybvh.rotations.*` — rotation primitives (especially `quat_slerp`)
- `pybvh.features.*` — motion analysis features
- `pybvh.batch.*` — batch loading and normalization

### Design history: the emo_mocap review
The two-library split was motivated by a detailed external review from a developer integrating pybvh into an ML project (emo_mocap, emotion recognition from motion capture). The review proposed 13 improvements. Our analysis:
- **Implemented in pybvh**: `euler_orders` property, `auto_detect_lr_pairs`, `__eq__`, `edges` property, better docstrings (pending)
- **Implemented in pybvh-ml**: tensor packing (CTV/TVC/flat), skeleton graph metadata, array-level augmentation (quaternion + 6D), speed perturbation/dropout on arrays, HDF5 preprocessing, PyTorch Datasets, body-part partitions
- **Rejected**: linear Euler interpolation for dropout (mathematically unsound), framework-specific graph objects (too much coupling), `to_ml_tensor` as a Bvh method (extends `to_feature_array` instead)
- **Key principle established**: pybvh owns motion data; pybvh-ml owns how ML consumes it
