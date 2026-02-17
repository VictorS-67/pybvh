# AI_ONBOARDING.md — pybvh

> **Purpose of this document**: Give any new AI agent (or human contributor) a complete, precise understanding of the pybvh codebase—its goals, architecture, data flow, every module, class, method, and design decision—so they can modify or extend it without guessing.

---

## 1. Project Identity

| Field | Value |
|---|---|
| **Name** | pybvh |
| **Language** | Python 3 |
| **Dependencies** | `numpy`, `pandas`, `matplotlib` |
| **Primary use-case** | Pre-processing BVH (Biovision Hierarchy) motion capture files for skeleton-based machine learning pipelines |
| **Design principles** | **Fast** (NumPy-vectorised, pre-allocated arrays), **Lightweight** (minimal code surface), **Self-contained** (minimal external dependencies to reduce future maintenance burden) |

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
                    ┌─────────────────────────────────────────┤
                    │                                         │
                    ▼                                         ▼
            bvh.get_df_constructor()              bvh.get_spatial_coord()
                    │                                         │
                    ▼                                         ▼
            pd.DataFrame(...)                      NumPy array (3D positions)
                    │                                         │
                    ▼                                         ▼
              df_to_bvh() ──► Bvh object            plot.plot_frame()
                    │                               plot.plot_animation()
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
from .read_bvh_file import read_bvh_file
from .df_to_bvh import df_to_bvh
from .spatial_coord import frames_to_spatial_coord
from . import plot
from . import rotations
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

**Convenience wrappers** (go through rotmat as intermediate):
`euler_to_rot6d`, `rot6d_to_euler`, `euler_to_quat`, `quat_to_euler`, `euler_to_axisangle`, `axisangle_to_euler`

**Internal helpers**: `_normalize`, `_axis_angle_to_rotmat` (single-axis batch rotation), `_extract_euler`

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
- `is_end_site()` → `False`.

#### `BvhRoot(BvhJoint)`
- **Represents**: The root joint (has position channels in addition to rotation).
- **Additional attribute**: `pos_channels` (list of 3 chars, e.g. `['X', 'Y', 'Z']`).
- `is_root()` → `True`, `parent` is always `None`.

**Important**: The skeleton is a **tree**. You traverse it from `root` via `.children`, or walk up via `.parent`. The `Bvh.nodes` list is a **flat depth-first list** of all nodes (joints + end sites).

### 4.3 `pybvh/bvh.py` — The `Bvh` Class (Central Container)

#### Constructor
```python
Bvh(nodes=[BvhRoot()], root_pos=None, joint_angles=None,
    frame_frequency=0)
```

The constructor validates that `root.pos_channels == ['X', 'Y', 'Z']` (the only layout supported). If `root_pos` or `joint_angles` are `None`, they default to empty arrays with the correct shape.

#### Core Attributes (all property-validated)
| Attribute | Type | Description |
|---|---|---|
| `nodes` | `list[BvhNode]` | Flat depth-first list of all skeleton nodes |
| `root_pos` | `np.ndarray` (2D) | Shape `(F, 3)`. Root translation per frame |
| `joint_angles` | `np.ndarray` (3D) | Shape `(F, J, 3)`. Euler angles in degrees per joint per frame. Joint order follows `nodes` (end sites excluded) |
| `frame_frequency` | `float` | Seconds per frame (e.g. `1/30`) |
| `frame_count` | `int` (read-only) | Computed property: `len(self.root_pos)`. No setter. |
| `euler_column_names` | `list[str]` (read-only) | Computed property generating BVH-style column names on the fly, e.g. `['Hips_X_pos', 'Hips_Y_pos', 'Hips_Z_pos', 'Hips_Z_rot', ...]`. Not stored — rebuilt from `nodes` each time. |
| `root` | `BvhRoot` | Shortcut to `nodes[0]` |
| `name2idx` | `dict[str, int]` | Maps joint/node name → integer index into the `nodes` list. Created by `_create_name2idx()`. One entry per node (not per axis). |

#### Data layout
Motion data is stored as two structured arrays:
- `root_pos[f]` — the `(X, Y, Z)` position of the root at frame `f`.
- `joint_angles[f, j]` — the 3 Euler angles (degrees) for joint `j` at frame `f`, in the order given by `nodes[j].rot_channels`.

End sites have NO entries in `joint_angles` (they have no channels). The joint index `j` corresponds to the j-th non-end-site node in `nodes` (i.e., only joints/root have entries).

There is **no flat `.frames` property** — code should use `root_pos` and `joint_angles` directly.

#### Key Methods

| Method | Description |
|---|---|
| `to_bvh_file(filepath, verbose=True)` | Serialize back to a `.bvh` file. Uses recursive `rec_node_to_file()` for the hierarchy, then writes motion data by concatenating `root_pos[i]` and `joint_angles[i].ravel()` per frame with 6-decimal precision. |
| `get_spatial_coord(frame_num=-1, centered="world")` | Compute 3D joint positions via forward kinematics. Returns `(N, 3)` for one frame or `(F, N, 3)` for all frames, where N = number of nodes (including end sites). Delegates entirely to `frames_to_spatial_coord`. |
| `get_rest_pose(mode='euler'\|'coordinates')` | Returns the rest/T-pose. For `'euler'`: returns a tuple `(root_pos_zeros, joint_angles_zeros)` with matching dtype. For `'coordinates'`: returns `(N, 3)` array of 3D positions at rest. |
| `get_df_constructor(mode='euler'\|'coordinates', centered="world")` | Returns a dict-of-arrays (column name → 1D NumPy array) ready for `pd.DataFrame(bvh.get_df_constructor())`. Avoids importing pandas in pybvh. |
| `hierarchy_info_as_dict()` | Returns a deep-copied dict describing the skeleton tree. |
| `change_skeleton(new_skeleton, inplace=False)` | Retarget: copy offsets from another `Bvh` object's skeleton (joints must have matching names). |
| `scale_skeleton(scale, inplace=False)` | Scale all offsets by a float or 3-element array. |
| `copy()` | Deep copy via `copy.deepcopy()`. |
| `get_frames_as_rotmat()` | Convert Euler angles to rotation matrices `(F, J, 3, 3)`. Returns `(root_pos, joint_rotmats, joints)`. |
| `get_frames_as_6d()` | Convert Euler angles to 6D rotation representation `(F, J, 6)`. Returns `(root_pos, joint_rot6d, joints)`. |
| `get_frames_as_quaternion()` | Convert Euler angles to quaternions `(F, J, 4)`. Returns `(root_pos, joint_quats, joints)`. |
| `get_frames_as_axisangle()` | Convert Euler angles to axis-angle vectors `(F, J, 3)`. Returns `(root_pos, joint_aa, joints)`. |
| `set_frames_from_6d(root_pos, joint_rot6d)` | Set `root_pos` and `joint_angles` from 6D rotation data. Converts back to Euler angles using each joint's `rot_channels`. |
| `set_frames_from_quaternion(root_pos, joint_quats)` | Set `root_pos` and `joint_angles` from quaternion data. |
| `set_frames_from_axisangle(root_pos, joint_aa)` | Set `root_pos` and `joint_angles` from axis-angle data. |
| `single_joint_euler_angle(joint_name, new_order, inplace=True)` | Change Euler order of one joint. Updates `joint_angles` and the node's `rot_channels` atomically. |
| `change_all_euler_orders(new_order, inplace=True)` | Change Euler order of ALL joints to a unified order. |

#### The `centered` Parameter (appears throughout the codebase)
Three modes controlling how root position is handled:
- `"world"` — Root at the actual saved coordinates from the BVH file.
- `"skeleton"` — Root forced to `(0, 0, 0)` in every frame.
- `"first"` — First frame's root is at `(0, 0, 0)`, subsequent frames move relative to that.

### 4.4 `pybvh/read_bvh_file.py` — BVH Parser

**Entry point**: `read_bvh_file(filepath)` → `Bvh`

Internally calls `_extract_bvh_file_info(filepath)` which returns a 3-tuple `(nodes, frame_frequency, flat_frames)`:
1. Opens the file and reads line by line.
2. Parses `ROOT`, `JOINT`, `End Site` blocks, constructing `BvhRoot`/`BvhJoint`/`BvhNode` objects.
3. Tracks parent-child relationships using a `parent_depth` counter that increments on `}` lines.
4. After `Frame Time:` line, reads all frame data into a flat NumPy array.

After receiving the flat array, `read_bvh_file` splits it into `root_pos` (first 3 columns) and `joint_angles` (remaining columns reshaped to `(F, J, 3)`) before constructing the `Bvh` object.

**Helper**: `_get_offset_channels(node_type, f, line_number)` — reads 2–3 lines from the open file to extract offset and channel info for a node.

**Performance note**: Frame data is currently appended row-by-row with `np.append` — this is an O(n²) pattern and is a known area for optimization (pre-allocate based on `Frames: N`).

### 4.5 `pybvh/spatial_coord.py` — Forward Kinematics

**Purpose**: Convert Euler angle frames → 3D joint positions.

#### `frames_to_spatial_coord(nodes_container, root_pos=None, joint_angles=None, centered="world")`
The main workhorse. Accepts the structured `root_pos` `(F, 3)` and `joint_angles` `(F, J, 3)` arrays directly (or extracts them from a `Bvh` object).

**Output shape**: `(F, N, 3)` for multiple frames, or `(N, 3)` for a single frame, where N = total number of nodes (including end sites).

**Algorithm** (per frame):
1. Pre-allocate output array `(F, N, 3)`.
2. Build lookup dicts `node2jointidx` (node name → index in `joint_angles`) and `node2nodeidx` (node name → index in output) once.
3. Convert all angles to radians in one vectorized call.
4. For each frame, call `_fill_spatial_coords_rec()` which performs recursive forward kinematics:
   - **Root**: spatial coord = `[0, 0, 0]`, rotation matrix = `get_premult_mat_rot(angles, order)`.
   - **Interior joint**: `coord = parent_acc_rot_mat @ node.offset + parent_coord`; accumulated rotation = `parent_acc_rot_mat @ this_node_rot_mat`.
   - **End site**: `coord = parent_acc_rot_mat @ node.offset + parent_coord` (no own rotation).
5. After recursion, add root position (unless skeleton-centered), broadcast across all nodes.
6. For `"first"` centering, subtract first frame's root position from all frames.

**Node extraction**: Uses `hasattr`/`isinstance` checks (not duck-typing `try/except`) to determine whether the input is a `Bvh` object (extracts `nodes`, `root_pos`, `joint_angles`) or a plain list of nodes.

### 4.6 `pybvh/df_to_bvh.py` — DataFrame → Bvh Conversion

**Entry point**: `df_to_bvh(hier, df)` → `Bvh`

`hier` can be:
- A **list of BvhNode objects** (e.g. from an existing `Bvh.nodes`).
- A **dict** describing the hierarchy (keys = joint names, values = dicts with `offset`, `parent`, `children`, optional `rot_channels`/`pos_channels`).

**Pipeline**:
1. `_check_df_columns(df)` — Filter to valid columns matching pattern `{name}_{axis}_{pos|rot}`, verify `time` column exists, verify root pos+rot appear first.
2. If `hier` is a dict: `_complete_hier_dict()` fills missing channel info from DataFrame column order, then `_hier_dict_to_list()` converts to ordered node list.
3. `_check_df_match_with_hier()` — Reorder DataFrame columns to match hierarchy traversal order.
4. Extract flat frame array, split into `root_pos` (first 3 columns) and `joint_angles` (remaining columns reshaped to `(F, J, 3)`), compute `frame_frequency`, construct `Bvh`.

### 4.7 `pybvh/tools.py` — Utility Functions

| Function | Purpose |
|---|---|
| `are_permutations(str1, str2)` | Check if two strings are permutations. Used to validate channel orders. |
| `test_file(filepath)` | Validate that a path exists and has `.bvh` extension. Returns `Path` object. |
| `rotX(angle)`, `rotY(angle)`, `rotZ(angle)` | 3×3 rotation matrices for a single axis. Input must be in **radians**. |
| `get_premult_mat_rot(angles, order)` | Compose 3 intrinsic Euler rotation matrices into one via pre-multiplication: `R = R₁ @ R₂ @ R₃` where order matches the given channel order. |

**Design note**: Rotation functions assume radians input for performance (no internal conversion). The `spatial_coord` module converts degrees → radians once in bulk before calling these.

### 4.8 `pybvh/plot.py` — Visualization

| Function | Purpose |
|---|---|
| `plot_frame(bvh_object, frame, centered)` | Static 3D matplotlib plot of one frame. |
| `plot_animation(bvh_object, frames, centered, savefile, filepath, ...)` | Animated 3D plot using `matplotlib.animation.FuncAnimation`. Can save to `.mp4`. |

**Internal helpers**:
- `_get_forw_up_axis(bvh_object, frame)` — Heuristically determines forward/upward axes by analyzing joint positions in `(N, 3)` format. Uses broadcast subtraction (`frame - frame[0]`) for root-relative coordinates.
- `_setup_plt(...)` / `_setup_plt_animation_world(...)` — Configure matplotlib 3D axes with correct viewing angles and limits. Uses `root_pos = frame[0]` for the root position, and indexes into `(N, 3)` arrays.
- `_draw_skeleton(frame, bvh_object, lines)` — Update `Line3D` objects for each bone. Uses `bvh_object.name2idx` to look up node indices, then indexes into `frame[idx]` directly (no `*3` offset arithmetic).
- `_angle_up_forward(...)` — Convert forward/up axis labels to matplotlib elevation/azimuth angles.

### 4.9 `tests/test_bvh.py` — Test Suite

Run with: `pytest tests/test_bvh.py -v`

Uses `bvh_data/bvh_example.bvh` as fixture (29 nodes, 56 frames, 24 joints, 30fps).

Test classes:
- `TestReadBvhFile` — File loading, frame counts, node counts, channel validation.
- `TestNodeHierarchy` — End site types, joint types, parent-child consistency.
- `TestSpatialCoordinates` — Forward kinematics correctness for world/skeleton/first centering. Spatial coordinate shapes are `(29, 3)` (single frame) and `(56, 29, 3)` (all frames). Numerical assertions with known expected values.
- `TestDataFrameConversion` — Euler DataFrame construction, round-trip `Bvh → DataFrame → Bvh`.
- `TestFileRoundTrip` — Write → re-read produces matching data.
- `TestBvhMethods` — `copy()`, `__str__`, `__repr__`, `get_rest_pose`, `hierarchy_info_as_dict`.
- `TestEdgeCases` — Empty objects, invalid input validation.

---

## 5. Data Representation Details

### 5.1 Motion Data: `root_pos` + `joint_angles`
Motion data is stored as two separate structured arrays:

- **`root_pos`**: Shape `(F, 3)`. The root translation per frame. Column order is always `(X, Y, Z)` — the constructor validates `pos_channels == ['X', 'Y', 'Z']`.
- **`joint_angles`**: Shape `(F, J, 3)`. Euler angles in degrees for each joint per frame. `J` = number of non-end-site nodes. Joint order matches the depth-first traversal of `nodes` (skipping end sites). For joint `j`, the 3 values are in the order given by `nodes[j].rot_channels` (e.g., `['Z', 'Y', 'X']`).

Example for `bvh_example.bvh`: `root_pos.shape = (56, 3)`, `joint_angles.shape = (56, 24, 3)` — 56 frames, 24 joints.

### 5.2 The `euler_column_names` Property
A computed (not stored) property that generates BVH-style column names on the fly:
```python
['Hips_X_pos', 'Hips_Y_pos', 'Hips_Z_pos', 'Hips_Z_rot', 'Hips_Y_rot', 'Hips_X_rot', 'Spine_Z_rot', 'Spine_Y_rot', 'Spine_X_rot', ...]
```
Naming convention: `{JointName}_{Axis}_{pos|rot}`. Rebuilt from `nodes` each time it is accessed — no need to keep it in sync after skeleton modifications.

### 5.3 Spatial Coordinates Output
- Shape: `(N, 3)` for a single frame, `(F, N, 3)` for multiple frames.
- N = total number of nodes including end sites (29 for `bvh_example.bvh`).
- Each `[i]` entry is the `(X, Y, Z)` world position of node `i`.
- Order matches `Bvh.nodes` list order (depth-first).
- `name2idx` maps `"JointName"` → integer index `i` (one entry per node, not per axis).

### 5.4 DataFrame Column Convention
`get_df_constructor()` returns a **dict-of-arrays** (column name → 1D NumPy array), suitable for `pd.DataFrame(bvh.get_df_constructor())`. pybvh does **not** import pandas itself.

**Euler mode**: columns are `time`, `Hips_X_pos`, `Hips_Y_pos`, `Hips_Z_pos`, `Hips_Z_rot`, `Hips_Y_rot`, `Hips_X_rot`, `Spine_Z_rot`, ...

**Coordinates mode**: columns are `time`, `Hips_X`, `Hips_Y`, `Hips_Z`, `Spine_X`, `Spine_Y`, `Spine_Z`, ... (includes end sites)

---

## 6. Forward Kinematics — The Math

Given a joint `J` with:
- `offset` = bone vector from parent to J in rest pose
- Parent's accumulated rotation matrix `R_parent`
- Parent's world position `P_parent`
- J's own Euler angles → rotation matrix `R_J`

The world position of J is:
$$P_J = R_{parent} \cdot \text{offset}_J + P_{parent}$$

The accumulated rotation matrix passed to J's children:
$$R_{acc,J} = R_{parent} \cdot R_J$$

For the root:
$$P_{root} = (0, 0, 0) \quad \text{(before adding root translation)}$$
$$R_{acc,root} = R_{root}$$

Root translation is added at the end (broadcast across all nodes) unless in `"skeleton"` mode.

Rotation matrix from Euler angles uses **intrinsic** rotations with **pre-multiplication**:
$$R = R_{\text{first}} \cdot R_{\text{second}} \cdot R_{\text{third}}$$

where the order comes from the joint's `rot_channels` (e.g., `['Z', 'Y', 'X']` → `R_Z \cdot R_Y \cdot R_X`).

---

## 7. Coding Conventions & Patterns

1. **Property validation**: All core attributes use `@property` with setters that type-check and value-check inputs.
2. **NumPy throughout**: `root_pos`, `joint_angles`, spatial coords, offsets — all NumPy arrays.
3. **Deep copy safety**: `Bvh.copy()` uses `copy.deepcopy()`. `hierarchy_info_as_dict()` returns a deep copy. `df_to_bvh()` deep-copies the hierarchy input.
4. **No caching of spatial coordinates**: `get_spatial_coord()` always recomputes. The docstring advises users to cache results themselves.
5. **Naming**: `_private` prefix for internal methods. `snake_case` everywhere. Files are named after their main export.
6. **No type hints in signatures** (except `node_type: str` in one helper). Validation is done at runtime in setters.
7. **Errors**: Mix of `ValueError`, `Exception`, and `ImportError`. Generally raised from setters and file I/O.
8. **Type checking**: Uses `hasattr`/`isinstance` checks rather than duck-typing `try/except` blocks (e.g., in `spatial_coord.py` to distinguish Bvh objects from plain node lists).
9. **No pandas dependency**: pybvh never imports pandas. `get_df_constructor()` returns a dict-of-arrays that the user can pass to `pd.DataFrame()`.

---

## 8. Testing Conventions

- **Framework**: pytest.
- **Fixture file**: `bvh_data/bvh_example.bvh` (29 nodes, 56 frames, 24 joints, 30 fps, humanoid skeleton with Hips root).
- **Numerical assertions**: `np.testing.assert_allclose` with `atol=1e-4` to `1e-10` depending on precision needs.
- **Round-trip tests**: BVH → DataFrame → BVH, BVH → file → BVH, BVH → {6D, quaternion, axis-angle} → BVH, Euler order conversion → re-conversion.
- **Test files**:
  - `tests/test_bvh.py` — File I/O, hierarchy, spatial coordinates, DataFrame conversion, edge cases.
  - `tests/test_rotations.py` — Euler↔rotmat, 6D, quaternion, axis-angle, Bvh rotation methods, Euler order conversion.
- **Run command**: `pytest tests/ -v`
- **Current count**: 160 tests, all passing.

---

## 9. Sample BVH Data Files

| File | Purpose |
|---|---|
| `bvh_example.bvh` | Main test file. Humanoid, 29 nodes, 56 frames, 30fps. Z-up coordinate system. |
| `bvh_test1.bvh`, `bvh_test2.bvh`, `bvh_test3.bvh` | Additional test files (different skeletons / frame counts). |
| `standard_skeleton.bvh` | A reference skeleton for `change_skeleton()` use cases. |

---

## 10. Known Performance Characteristics & Bottlenecks

| Area | Current Approach | Note |
|---|---|---|
| **File parsing (frames)** | `np.append` in a loop — O(n²) | Should pre-allocate from `Frames: N` count |
| **Forward kinematics** | Recursive Python per frame, per node | Main bottleneck for large datasets. Per-frame Python loop is unavoidable with current recursive approach. |
| **Rotation matrices** | Individual `rotX/Y/Z` calls, `@` chaining | Correct but not batched across frames |
| **DataFrame construction** | Dict-of-arrays from NumPy slices | Fast — avoids per-frame Python loops |
| **Radian conversion** | Single `np.radians()` call on all frames at once | Already optimized |

---

## 11. Quick Reference: Common Operations

```python
from pybvh import read_bvh_file, df_to_bvh, Bvh, rotations
from pybvh import plot
import pandas as pd

# Read
bvh = read_bvh_file("bvh_data/bvh_example.bvh")

# Inspect
print(bvh)                              # "24 elements in the Hierarchy, 56 frames at ..."
print(bvh.root.name)                   # "Hips"
print(bvh.root_pos.shape)             # (56, 3)
print(bvh.joint_angles.shape)         # (56, 24, 3)
print(bvh.nodes[1].rot_channels)       # ['Z', 'Y', 'X']

# Spatial coordinates
coords_all = bvh.get_spatial_coord(centered="world")       # (56, 29, 3)
coords_one = bvh.get_spatial_coord(frame_num=0)             # (29, 3)

# Rotation representations (all return (root_pos, joint_data, joints))
root_pos, rotmats, joints = bvh.get_frames_as_rotmat()       # (56,3), (56,J,3,3), [nodes]
root_pos, rot6d,   joints = bvh.get_frames_as_6d()           # (56,3), (56,J,6),   [nodes]
root_pos, quats,   joints = bvh.get_frames_as_quaternion()    # (56,3), (56,J,4),   [nodes]
root_pos, aa,      joints = bvh.get_frames_as_axisangle()     # (56,3), (56,J,3),   [nodes]

# Set frames back from representations
bvh.set_frames_from_6d(root_pos, rot6d)
bvh.set_frames_from_quaternion(root_pos, quats)
bvh.set_frames_from_axisangle(root_pos, aa)

# Euler order conversion
bvh.single_joint_euler_angle('Spine', 'XYZ', inplace=True)
bvh.change_all_euler_orders('XYZ', inplace=True)

# Standalone rotation conversions
R = rotations.euler_to_rotmat([30, 45, 60], 'ZYX', degrees=True)
aa = rotations.rotmat_to_axisangle(R)
q = rotations.rotmat_to_quat(R)

# DataFrame (pybvh does NOT import pandas — user does)
df = pd.DataFrame(bvh.get_df_constructor(mode='euler'))
bvh2 = df_to_bvh(bvh.nodes, df)    # round-trip

# Write
bvh.to_bvh_file("output.bvh")

# Plot
fig, ax = plot.plot_frame(bvh, 0, centered="skeleton")

# Skeleton manipulation
bvh_scaled = bvh.scale_skeleton(0.01)  # new Bvh with scaled offsets
```

---

## 12. Extending the Codebase — Guidelines

1. **Add new rotation representations** in `rotations.py`. Keep them as pure NumPy batch-vectorized functions.
2. **Add new Bvh methods** in `bvh.py`. Follow the existing pattern: validate inputs in properties, call helper functions from other modules. Rotation-related methods should delegate to `rotations.py`.
3. **`euler_column_names` is computed**: It is generated on the fly from `nodes`. Any operation that changes `rot_channels` on a node (like `single_joint_euler_angle`) only needs to update the node and `joint_angles` — the column names will reflect the change automatically.
4. **Test with the fixture**: Add tests to `tests/test_rotations.py` (rotation-related) or `tests/test_bvh.py` (BVH I/O, hierarchy, spatial) using the `bvh_example` fixture. Include numerical assertions with known expected values.
5. **Avoid adding dependencies** unless absolutely necessary. If you must, prefer NumPy-based solutions over specialized libraries (no scipy).
6. **Performance**: Pre-allocate arrays, vectorize with NumPy, avoid Python loops over frames when possible.
