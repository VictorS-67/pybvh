# CONTEXT.md — pybvh

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
| **Version** | 0.7.0 |
| **Package** | Published on PyPI as `pybvh`. Install via `pip install pybvh`. Optional extras: `pybvh[opencv]` (fast render), `pybvh[interactive]` (k3d for Jupyter), `pybvh[viewer]` (vedo desktop), `pybvh[all-viz]` (all of the above), `pybvh[pandas]` (pandas integration) |
| **CI/CD** | GitHub Actions: test workflow (push/PR, Python 3.9–3.12) + publish workflow (PyPI on release) + docs workflow (MkDocs to GitHub Pages on push to main) |
| **Type safety** | Full type annotations on all source files, `@overload` on inplace methods, mypy clean |
| **Tests** | 1124 unit tests via pytest (plus ~23 000 parametrized `test_transforms_battle` cases across 3 real-world datasets, skipped unless the private fixtures are present) |
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
                bvh.to_df_dict()                    bvh.node_positions()
                    │                                         │
                    ▼                                         ▼
            pd.DataFrame(...)                      NumPy array (3D positions)
                    │                                         │
                    ▼                                         ▼
              df_to_bvh() ──► Bvh object       bvh.plot_frame() / bvh.render() /
                    │                          bvh.play() / bvhplot.frame([...])
                    ▼
              bvh.write() ──► .bvh file
```

**Central object**: `Bvh` — everything flows through it.

---

## 4. File-by-File Module Reference

### 4.1 `pybvh/__init__.py`
Public API surface. Exports:
```python
__version__ = "0.7.0"

from .bvh import Bvh
from .io import read_bvh_file, write_bvh_file
from .df_to_bvh import df_to_bvh
from .spatial_coord import frames_to_node_positions
from .batch import (read_bvh_directory, batch_to_numpy, harmonize,
                    compute_normalization_stats, normalize_array, denormalize_array)
from . import bvhplot
from . import rotations
from . import transforms
from . import analysis
from . import packing
```

**Module renamed `plot` → `bvhplot`** in v0.5.0 to avoid confusion with matplotlib's `plot()`.

### 4.2 `pybvh/rotations.py` — Rotation Representation Conversions

Pure NumPy, batch-vectorised rotation conversions. No scipy dependency. Supports Euler angles `(*, 3)`, rotation matrices `(*, 3, 3)`, 6D rotation (Zhou et al.) `(*, 6)`, quaternions `(*, 4)`, and axis-angle `(*, 3)`. Core functions convert between any pair via rotmat as the hub representation, plus `quat_slerp` for interpolation. Convenience wrappers provide direct paths (e.g. `euler_to_quat`). See source docstrings for method signatures.

**Conventions**:
- Euler: intrinsic rotations, pre-multiplication `R = R1 @ R2 @ R3`
- Quaternion: `(w, x, y, z)` scalar-first, canonical `w >= 0`
- Axis-angle: zero vector = identity rotation; norm = rotation angle in `[0, π]`

### 4.3 `pybvh/bvhnode.py` — Node Class Hierarchy

Three classes forming an inheritance chain:

```
BvhNode  (end sites)
  └── BvhJoint  (interior joints)
        └── BvhRoot  (root joint — exactly one per skeleton)
```

`BvhNode` represents end sites (leaf bones, no channels). `BvhJoint` adds `rot_channels` (list of 3 chars, e.g. `['Z', 'Y', 'X']`) and `children`. `BvhRoot` adds `pos_channels`.

**Freeze mechanism**: After a `Bvh` object is constructed, `_frozen = True` is set on all joints. Direct assignment to `rot_channels` raises `AttributeError` — users must use `Bvh.change_euler_order()`. Internal code uses `_set_rot_channels_internal()` to bypass the freeze.

The skeleton is a **tree**. Traverse from `root` via `.children`, or walk up via `.parent`. The `Bvh.nodes` list is a **flat depth-first list** of all nodes (joints + end sites). See source docstrings for method signatures.

### 4.4 `pybvh/bvh.py` — The `Bvh` Class (Central Container)

The central container holding skeleton + motion data. Constructor: `Bvh(nodes, root_pos, joint_angles, frame_time)`. Validates `root.pos_channels == ['X', 'Y', 'Z']`, freezes channel attributes after construction, and eagerly computes `_world_up_cached` via `_infer_world_up()`.

**Note**: the parameter was renamed from `frame_frequency` (misnamed — stored frame *time*, not frequency) to `frame_time` in v0.6.0. The old name was removed outright, no deprecation shim.

#### Data layout
- `root_pos`: Shape `(F, 3)`. Root translation per frame, column order always `(X, Y, Z)`.
- `joint_angles`: Shape `(F, J, 3)`. Euler angles in **radians** per joint per frame. Joint order follows `nodes` (end sites excluded). BVH files store degrees; pybvh converts at the I/O boundary in `read_bvh_file` / `write_bvh_file`.
- `frame_time`: Seconds per frame (e.g. `1/30`).
- `world_up`: Gravity axis (e.g. `'+y'`), auto-detected from frame 0 with rest-pose fallback. Settable; propagates through `copy()`/`slice_frames()`.
- Read-only properties: `frame_count`, `joint_names`, `joint_count`, `euler_orders`, `edges`, `node_index`, `joint_index`, `root`.

There is **no flat `.frames` property** — code should use `root_pos` and `joint_angles` directly.

The class provides methods for I/O (`write`, `node_positions`, `rest_pose_coords`, `to_df_dict`, `copy`), skeleton ops (`retarget`, `scale`, `change_euler_order`, `extract_joints`), topology checks (`matches_topology`, `matches_hierarchy`, `matches_channels`), rotation conversions (`to_rotmat`, `to_6d`, `to_quaternions`, `to_axisangle`, `from_6d`, `from_quaternions`, `from_axisangle`), frame ops (`slice_frames`, `concat`, `resample`), features (`joint_velocities`, `joint_accelerations`, `node_velocities`, `node_accelerations`, `angular_velocities`, `root_relative_positions`, `root_trajectory`, `foot_contacts`, `to_feature_array`, `feature_array_layout`, `auto_detect_foot_joints`), transforms (`translate_root`, `random_translate_root`, `add_noise`, `perturb_speed`, `random_perturb_speed`, `drop_frames`, `rotate_vertical`, `random_rotate_vertical`, `mirror`), reorientation (`reorient_world_up`, `reorient_rest_up`, `reorient_rest_forward`), orientation (`forward_at`), and visualization wrappers (`plot_rest_pose`, `plot_frame`, `plot_trajectory`, `render`, `play`). The `joint_index` and `lr_mapping` properties complement `node_index` for joint-axis lookups and L/R joint pairing respectively. The `source_path` attribute (populated by `read_bvh_file`) carries the on-disk origin for diagnostics. Many methods were renamed in v0.6.0 — old names were removed outright (no deprecation wrappers); see `pybvh/API_RENAME.md` for the complete old → new mapping. See source docstrings for method signatures.

#### The `centered` Parameter (appears throughout the codebase)
Three modes controlling how root position is handled:
- `"world"` — Root at the actual saved coordinates from the BVH file.
- `"skeleton"` — Root forced to `(0, 0, 0)` in every frame.
- `"first"` — First frame's root is at `(0, 0, 0)`, subsequent frames move relative to that.

### 4.5 `pybvh/batch.py` — Batch Loading & NumPy Export

Batch loading of BVH directories with optional parallelism, conversion to packed NumPy arrays, dataset-level harmonization, and normalization. Provides `read_bvh_directory` (supports `world_up=`, `lr_mapping=`, `skip_errors=`), `harmonize` (topology drop/raise + retarget + resample + reorient + Euler-order unification in one call; emits one summary `UserWarning` per call; pass `return_report=True` for a JSON-serializable `HarmonizeReport`), `batch_to_numpy` (representation-aware skeleton checks: full topology for `'euler'`/`'axisangle'`, hierarchy only for `'6d'`/`'quaternion'`/`'rotmat'`; error messages name clips by `source_path` when set), `compute_normalization_stats` (returns `mean`, `std`, `constant_channels`), `normalize_array`, and `denormalize_array`. See source docstrings for method signatures.

### 4.6 `pybvh/io.py` — BVH File I/O

Provides `read_bvh_file(filepath)` and `write_bvh_file(bvh, filepath)`. The reader parses the HIERARCHY into node objects, pre-allocates a NumPy array for frame data, and splits it into `root_pos` + `joint_angles`. The writer serializes back to `.bvh` format with `%.6f` precision. See source docstrings for method signatures.

### 4.7 `pybvh/spatial_coord.py` — Forward Kinematics

`frames_to_node_positions(nodes_container, root_pos, joint_angles, centered)` computes 3D joint positions via forward kinematics. Output shape: `(F, N, 3)` for multiple frames, `(N, 3)` for a single frame, where N = total nodes including end sites. Fully vectorized across all frames using batch matrix operations. See source docstrings for method signatures.

### 4.8 `pybvh/df_to_bvh.py` — DataFrame to Bvh Conversion

`df_to_bvh(hier, df)` converts a pandas DataFrame back to a `Bvh` object. `hier` can be a list of BvhNode objects or a dict describing the hierarchy. See source docstrings for method signatures.

### 4.9 `pybvh/tools.py` — Utility Functions

Low-level helpers: single-axis and batch rotation matrices (`rotX/Y/Z`, `batch_rotX/Y/Z`), Euler-to-rotmat composition (`get_premult_mat_rot`, `batch_get_premult_mat_rot`), string utilities (`are_permutations`, `get_main_direction`, `extract_sign`), and file validation (`test_file`). Also contains private orientation helpers (`_rest_upward`, `_rest_lateral`, `_infer_world_up`, `_compute_forward_at`, `_world_lateral_unit_at_frame`, `_signed_rotation_delta_around_axis`, `_validate_axis_string`) used by `Bvh.world_up`, `Bvh.forward_at()`, and follow-mode rendering. See source docstrings for method signatures.

### 4.10 `pybvh/bvhplot/` — Visualization Package

Quick-look visualization module with 5 public functions: `rest_pose`, `frame`, `render`, `play`, and `trajectory`. Supports multiple backends: matplotlib (default), OpenCV (fast video ~1000fps), k3d (Jupyter interactive), and vedo (desktop interactive with keyboard controls for playback, labels, trail, screenshots). Accepts single `Bvh` or list for multi-skeleton comparison. `render` supports `follow=True` for camera tracking. Optional dependencies installed via `pybvh[opencv]`, `pybvh[interactive]`, `pybvh[viewer]`, or `pybvh[all-viz]`. See source docstrings for method signatures.

### 4.11 `pybvh/transforms.py` — Spatial Augmentation Transforms

Data augmentation transforms operating on `Bvh` objects: `translate_root`, `random_translate_root`, `add_noise`, `perturb_speed`, `random_perturb_speed`, `drop_frames`, `rotate_vertical`, `random_rotate_vertical`, `auto_detect_lr_mapping`, `auto_detect_lr_pairs`, and `mirror`. Also provides coordinate-frame reorientation (`reorient_world_up`, `reorient_rest_up`, `reorient_rest_forward`) for dataset preprocessing. All follow the `inplace=False` convention. NumPy-level functions (`rotate_angles_vertical`, `mirror_angles`) are exposed for users working with pre-extracted arrays. See source docstrings for method signatures.

### 4.12 `pybvh/analysis.py` — Motion Analysis

Standalone functions for extracting motion descriptors: `joint_velocities` / `joint_accelerations` (shape `(F, J, 3)`, non-end-site joints — align with `joint_angles` axis), `node_velocities` / `node_accelerations` (shape `(F, N, 3)`, all nodes including end sites — useful for extremity tracking), `angular_velocities` (shape `(F, J, 3)` — rotations only exist on joints), `root_trajectory`, `foot_contacts`, and `auto_detect_foot_joints`. Bvh-bound functions take `bvh: Bvh` as their first argument; the corresponding `Bvh` class methods are thin wrappers.

v0.8.0 added (the descriptor-review primitives): **jerk** `node_jerk` / `joint_jerk` (third-derivative rung of the velocity→accel ladder, same `stencil`/`pad`); **smoothness** on a 1-D speed profile — `sparc` (spectral arc length), `dimensionless_jerk`, `log_dimensionless_jerk`, `number_of_peaks`, `speed_metric`, `integrated`/`mean`/`rms_squared_jerk`, plus a `smoothness(metric=…)` dispatcher (SPARC/DLJ/LDLJ validated against the Balasubramanian reference via golden fixtures); **reductions** `velocity_reductions`, `zero_crossings`, `active_segments` / `active_duration`; **`kinetic_energy`** (Σ‖v‖² or Σ½m‖v‖²); **gait** `cadence` / `stride_length` / `walking_pace` (from `foot_contacts` + root ground path); **`range_of_motion`**; and the covariance descriptors `cov3dj` / `lagged_correlation`. The array-pure kernels (smoothness, reductions, covariance) take signals/arrays directly; jerk/energy/gait/ROM-wrapper are Bvh-bound. See source docstrings for signatures.

### 4.13 `pybvh/packing.py` — ML Feature-Array Packing

`to_feature_array` and `feature_array_layout`: compose the `analysis` descriptors (rotations, root position, velocities, foot contacts) into a single flat `(F, D)` array for ML pipelines, plus the column-layout map describing it. Split out of the former `features.py` in v0.8.0 so the analysis layer stays free of pipeline-assembly concerns. See source docstrings for signatures.

### 4.14 `pybvh/geometry.py` — Position Descriptors (array-pure)

The position half of pybvh's geometry surface — the companion to `rotations.py` (§4.2, the orientation half). All functions are **array-pure** (plain NumPy point arrays in, arrays out; no `Bvh`), so downstream libraries build on them directly. Added in v0.8.0. Two shape conventions: point-set kernels take `(..., P, 3)` and reduce over the point axis `P` (`bounding_box`/`bounding_sphere`/`bounding_ellipsoid`/`centroid`/`verticality`); trajectory kernels take `(F, …, 3)` over the time axis (`path_length`, `straightness`, `curvature`, `torsion`, `movement_phase`, `ground_path`). Also inter-point relations (`inter_joint_distance`, `joint_angle`, `segment_axis_angle`, `triangle_area`, `point_to_plane_distance`, `point_to_segment_distance`) and pose ops (`pose_distance`, `mean_pose_subtract`). Derivative kernels route through `tools.finite_difference` (the shared stencil/pad convention, bit-identical to the velocity ladder). Zero-denominator ratios return `np.nan` consistently. Bounding sphere is Ritter's approximate 2-pass (vectorized, not exact Welzl); ellipsoid is PCA via batched `eigh` — no scipy.

---

## 5. Data Representation Details

### 5.1 Motion Data: `root_pos` + `joint_angles`
- **`root_pos`**: Shape `(F, 3)`. Column order always `(X, Y, Z)`.
- **`joint_angles`**: Shape `(F, J, 3)`. Euler angles in **radians**. `J` = number of non-end-site nodes.

Example for `bvh_example.bvh`: `root_pos.shape = (56, 3)`, `joint_angles.shape = (56, 24, 3)`.

### 5.2 Spatial Coordinates Output
- Shape: `(N, 3)` for a single frame, `(F, N, 3)` for multiple frames.
- N = total number of nodes including end sites (29 for `bvh_example.bvh`).
- Order matches `Bvh.nodes` list order (depth-first).
- `node_index` maps `"JointName"` → integer index into the N-axis (use for `node_positions()` output).
- `joint_index` maps `"JointName"` → integer index into the J-axis (use for `joint_angles`, which excludes end sites).

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
7. **No pandas dependency**: pybvh never imports pandas. `to_df_dict()` returns a dict-of-arrays that users can wrap in `pd.DataFrame(...)` themselves.
8. **No ML framework dependencies**: Output is always NumPy. Users convert to PyTorch/TensorFlow themselves.
9. **Naming**: `_private` prefix for internal methods. `snake_case` everywhere.
10. **Errors**: Mix of `ValueError`, `Exception`, and `AttributeError`.

---

## 8. Testing Conventions

- **Framework**: pytest
- **Fixture files**: `bvh_data/bvh_example.bvh` (primary), plus `bvh_test1.bvh`, `bvh_test2.bvh`, `bvh_test3.bvh`, `standard_skeleton.bvh`
- **Synthetic fixtures**: `tests/synthetic_bvh.py` — a library of 8 factory functions for programmatically creating BVH objects with known properties: `make_pos_y_up_bvh`, `make_neg_y_up_bvh`, `make_pos_z_up_bvh`, `make_neg_z_up_bvh`, `make_heterogeneous_euler_bvh`, `make_lowercase_lr_bvh`, `make_pos_y_up_rotating_bvh`, `make_simple_bvh`.
- **Numerical assertions**: `np.testing.assert_allclose` with `atol=1e-4` to `1e-10` depending on precision needs. File round-trips use `atol=1e-5` (due to `%.6f` formatting).
- **Round-trip tests**: BVH → file → BVH, BVH → DataFrame → BVH, BVH → {6D, quaternion, axis-angle} → BVH, Euler order conversion → re-conversion.
- **Test files**:
  - `tests/test_bvh.py` — File I/O, hierarchy, spatial coordinates, DataFrame conversion, skeleton operations, batch processing, freeze preservation, ML pipeline features (velocities, foot contacts, normalization, feature export), edge cases.
  - `tests/test_rotations.py` — All conversion paths, gimbal lock, 180° SLERP, analytical values.
  - `tests/test_plot.py` — Visualization module tests (bvhplot functions, backends, camera presets).
  - `tests/test_audit_fixes.py` — 86 audit tests verifying correctness of specific bug fixes and edge cases identified during code audits.
- **Run command**: `conda run -n pybvh pytest tests/ -v`
- **Current count**: 1124 tests, all passing.
- **Note**: `tests/test_transforms_battle.py` uses private datasets from `internal_bvh_data/` and is gitignored — never publish or share this file.

---

## 9. Sample BVH Data Files

| File | Joints | Nodes | Frames | FPS | World Up | Purpose |
|---|---|---|---|---|---|---|
| `bvh_example.bvh` | 24 | 29 | 75 | 30 | +z | Main test file (anger clip from DIEM-A dataset) |
| `bvh_test1.bvh` | 24 | 29 | 75 | 30 | +z | Additional Z-up test |
| `bvh_test2.bvh` | 23 | 28 | 61 | 120 | +y | Y-up test with root rotated ~180° from rest (regression fixture for `camera='front'`) |
| `bvh_test3.bvh` | 60 | 73 | 100 | 120 | +z* | Large skeleton, rest pose and first frame disagree on world up — triggers the `_infer_world_up` `UserWarning` |
| `standard_skeleton.bvh` | 24 | 29 | 1 | 120 | +z | Reference skeleton for retargeting |

*`bvh_test3` rest pose suggests `+y` but frame-0 head-hips is closer to `+z`; the new inference picks `+z` from the animation data. This is exactly the edge case the `world_up` warning was designed to catch.

---

## 10. Quick Reference: Common Operations

```python
from pybvh import read_bvh_file, df_to_bvh, Bvh, rotations, bvhplot
from pybvh import read_bvh_directory, batch_to_numpy
import pandas as pd

# Read
bvh = read_bvh_file("walk.bvh")

# Inspect
bvh.root_pos.shape          # (F, 3)
bvh.joint_angles.shape      # (F, J, 3)
bvh.joint_names              # ['Hips', 'Spine', ...]
bvh.joint_count              # 24
bvh.node_index['Hips']       # 0  — index into node_positions() (incl. end sites)
bvh.joint_index['Hips']      # 0  — index into joint_angles axis 1 (excl. end sites)
bvh.world_up                 # '+z' — auto-detected gravity axis
bvh.forward_at(0)            # '+y' — facing direction at frame 0
bvh.world_up = '+y'          # manual override (validated)

# Spatial coordinates (forward kinematics)
coords = bvh.node_positions(centered="world")  # (F, N, 3)
rest = bvh.rest_pose_coords(mode='coordinates')

# Rotation representations
root_pos, rot6d = bvh.to_6d()
root_pos, quats = bvh.to_quaternions()
root_pos, aa    = bvh.to_axisangle()

# Set frames back (inplace=False returns new Bvh)
bvh2 = bvh.from_6d(root_pos, rot6d)

# Euler order conversion (unified method)
bvh_xyz = bvh.change_euler_order('XYZ')           # all joints
bvh_hips = bvh.change_euler_order('XYZ', joint='Hips')  # one joint

# Frame operations
clip = bvh.slice_frames(10, 50)
combined = bvh.concat(other_bvh)
bvh_30fps = bvh.resample(30)

# Skeleton operations
bvh_scaled = bvh.scale(0.01)
retargeted = bvh.retarget(standard_skeleton)
upper = bvh.extract_joints(["Hips", "Spine", "Neck", "Head"])

# Transforms (augmentation)
noisy = bvh.add_noise(sigma_deg=1.0)
faster = bvh.perturb_speed(factor=1.5)
dropped = bvh.drop_frames(drop_rate=0.1)
mirrored = bvh.mirror()
rotated = bvh.rotate_vertical(90)
shifted = bvh.random_translate_root(rng=rng)  # random-variant method wrappers
jittered = bvh.random_rotate_vertical(rng=rng)
warped = bvh.random_perturb_speed(rng=rng)

# Reorientation (preprocessing)
bvh_zup = bvh.reorient_world_up('+z')        # rotate whole scene; character unchanged
bvh_rest = bvh.reorient_rest_up('+z')        # fix rest-pose / animation disagreement
bvh_fwd = bvh.reorient_rest_forward('+y')    # canonicalize rest-pose facing direction

# Skeleton topology checks
compatible = bvh.matches_topology(other)     # hierarchy AND channels (Euler orders)
same_graph = bvh.matches_hierarchy(other)    # names + parents + rest offsets only
same_chans = bvh.matches_channels(other)     # per-joint Euler orders only

# Batch loading for ML
clips = read_bvh_directory("dataset/", parallel=True, skip_errors=True)
data = batch_to_numpy(clips, representation="6d", pad=True)  # (B, F_max, D)

# One-shot dataset harmonization (topology / fps / up-axis / Euler order)
from pybvh import harmonize
clips, report = harmonize(clips, reference=ref, target_fps=30,
                          target_world_up='+z', target_euler_order='XYZ',
                          return_report=True)

# Standalone rotations
R = rotations.euler_to_rotmat([30, 45, 60], 'ZYX', degrees=True)
q = rotations.rotmat_to_quat(R)
q_mid = rotations.quat_slerp(q1, q2, t=0.5)

# ML pipeline features (de-prefixed)
vel = bvh.joint_velocities()                     # (F, J, 3) units/second (default stencil="central", pad="edge")
acc = bvh.joint_accelerations()                  # (F, J, 3) units/second^2
node_vel = bvh.node_velocities()                 # (F, N, 3) — all nodes incl. end sites
ang_vel = bvh.angular_velocities()               # (F, J, 3) radians/second
rel_pos = bvh.node_positions(centered='skeleton')# (F, N, 3) root-at-origin
traj = bvh.root_trajectory()                     # (F, 4) ground pos + heading
contacts = bvh.foot_contacts()                   # (F, num_feet) binary
feat = bvh.to_feature_array(representation="6d", # (F, D) one-stop export
         include_velocities=True, include_foot_contacts=True)
layout = bvh.feature_array_layout(                # slice map for unpacking
         representation="6d", include_velocities=True, include_foot_contacts=True)
# {'root_pos': slice(0, 3), 'rotations': slice(3, ...), 'velocities': ..., 'foot_contacts': ...}

# Normalization (dataset-level)
from pybvh import compute_normalization_stats, normalize_array, denormalize_array
stats = compute_normalization_stats(clips, representation="6d")
normalized = normalize_array(data, stats)
recovered = denormalize_array(normalized, stats)

# DataFrame (pybvh does NOT import pandas)
df = pd.DataFrame(bvh.to_df_dict(mode='euler'))
bvh2 = df_to_bvh(bvh.nodes, df)

# Write to file
bvh.write("output.bvh")

# Visualization — single-skeleton via Bvh wrappers
bvh.plot_rest_pose()
bvh.plot_frame(frame=0, camera='front')          # 'front' | 'side' | 'top' | (azim, elev)
bvh.plot_trajectory()
bvh.render("walk.mp4")                           # fast video export
bvh.render("walk.mp4", follow=True)              # camera tracks character rotation
bvh.play()                                       # interactive auto-backend

# Visualization — multi-skeleton via bvhplot module
bvhplot.frame([bvh1, bvh2], frame=0, labels=["A", "B"])
bvhplot.render([bvh1, bvh2], "compare.mp4", sync="pad")
bvhplot.trajectory([bvh1, bvh2], labels=["A", "B"])
```

---

## 11. Extending the Codebase — Guidelines

1. **Add new rotation representations** in `rotations.py`. Keep them as pure NumPy batch-vectorized functions.
2. **Add new Bvh methods** in `bvh.py`. Follow the existing pattern: validate inputs in properties, delegate to helper modules.
3. **`_euler_column_names` is computed on the fly**: Any operation that changes `rot_channels` only needs to update the node and `joint_angles` — the internal channel-name helper reflects the change automatically.
4. **Test with fixtures**: Add tests using the existing fixtures. Include numerical assertions with known expected values.
5. **No new dependencies** unless absolutely necessary. Output NumPy arrays — let users convert to their ML framework of choice.
6. **Performance**: Pre-allocate arrays, vectorize with NumPy, avoid Python loops over frames.
7. **Type all new code**: Use `npt.NDArray[np.float64]` for returns, `npt.ArrayLike` for inputs. Add `@overload` for inplace methods.
8. **Caching opportunity**: `euler_orders` and `edges` properties recompute on every access (they traverse the node list). This is fine for single calls but wasteful in hot loops. If profiling shows these as bottlenecks, consider caching with invalidation on skeleton mutation (e.g. `change_euler_order`, `extract_joints`).

---

## 12. Ecosystem & Scope Boundary

pybvh is the **foundation layer** in a three-library ecosystem:

```
pybvh-ml      (ML bridge: tensor packing, augmentation pipelines, PyTorch Datasets)
    │
    ▼
  pybvh       (BVH foundation: parsing, rotation math, transforms, motion analysis, quick visualization)
    │
    ▲
pybvh-blender (Blender addon: deep BVH inspection, joint panels, analysis overlays)
```

**pybvh never imports or knows about pybvh-ml or pybvh-blender.** Dependencies flow one way: `pybvh-ml -> pybvh` and `pybvh-blender -> pybvh`.

### Scope rules
- If a feature is useful to anyone working with BVH data (researcher, game dev, biomechanics) — it belongs in **pybvh**.
- If it only makes sense in an ML training context (tensor layouts, Datasets, HDF5 export) — it belongs in **pybvh-ml**.
- If it requires GUI widgets (property panels, graph editors, skeleton trees) for deep inspection — it belongs in **pybvh-blender**.
- If it's a quick visualization callable from Python (`bvhplot.play(bvh)`) — it belongs in **pybvh.bvhplot**.

### API surface that pybvh-ml relies on
pybvh-ml is a primary consumer of pybvh's public API. When modifying pybvh, be aware that these entry points are used downstream:
- `bvh.root_pos`, `bvh.joint_angles`, `bvh.joint_count`, `bvh.joint_names` — data access
- `bvh.to_quaternions()`, `bvh.to_6d()`, `bvh.to_rotmat()`, `bvh.to_axisangle()` — representation conversion. Return `(root_pos, joint_data)` 2-tuples (the third `joints` element in the old 3-tuple shape was removed in v0.6.0 — derivable from `bvh.nodes` / `bvh.joint_names` / `bvh.joint_index`).
- `bvh.from_6d()`, `bvh.from_quaternions()`, `bvh.from_axisangle()` — inverse representation conversion.
- `bvh.euler_orders` — per-joint Euler order strings
- `bvh.edges` — skeleton edge list as index tuples
- `bvh.nodes`, `bvh.node_index` — skeleton topology (indexes `node_positions()` output)
- `bvh.joint_index` — joint-only name → index dict (indexes `joint_angles` axis 1). Symmetric counterpart to `node_index`; preferred over `bvh.joint_names.index(name)`.
- `bvh.world_up`, `bvh.rest_up`, `bvh.rest_forward`, `bvh.forward_at(frame)`, `bvh.left_at(frame)` — orientation API. `world_up` / `forward_at` are animation-derived; `rest_up` / `rest_forward` are topology-derived (rest pose only). On clean files the two pairs agree. `(world_up, forward_at, left_at)` form an orthonormal right-hand-rule triple: `left = up × forward`.
- `bvh.lr_mapping` — cached L/R joint pair mapping, auto-detected at init via extended name heuristic (`Left`/`Right`, `.L`/`.R`, `_l`/`_r`, `mixamorig:` namespace, `.001` numbered suffix). `None` when no pairs detected. Settable via post-load setter or `lr_mapping=` kwarg on `read_bvh_file` / `read_bvh_directory` / `Bvh.__init__`. Consumed by `mirror()`, `forward_at()`, `left_at()`, `_rest_leftward`, `reorient_rest_forward`.
- `pybvh.transforms.auto_detect_lr_pairs()` — L/R index pair detection (module-level; reads `bvh.lr_mapping` internally)
- `pybvh.transforms.auto_detect_lr_mapping()` — L/R name pair detection (module-level; thin wrapper over `bvh.lr_mapping`)
- `bvh.random_translate_root()`, `bvh.random_rotate_vertical()`, `bvh.random_perturb_speed()` — method wrappers for the `random_*` augmentation variants (same signatures as the module-level functions)
- `bvh.matches_hierarchy(other, match_offsets=True, atol=1e-6)` — boolean predicate: True iff node names, parent structure, and (by default) rest offsets match. Pass `match_offsets=False` to ignore bone proportions when the caller is about to retarget. Channel layout / Euler orders are NOT compared.
- `bvh.matches_channels(other)` — boolean predicate: True iff per-joint Euler rotation orders and root position-channel order match. The serialization half previously conflated into `matches_topology`.
- `bvh.matches_topology(other)` — conjunction `matches_hierarchy(other) and matches_channels(other)`. Pre-0.7.0 this was a looser check (just `joint_names` + `euler_orders`); the new definition is stricter and additionally compares parent structure and rest offsets.
- `bvh.source_path` — on-disk origin set by `read_bvh_file`; preserved through `copy()` / `slice_frames()` / single-source `concat()`; surfaced in `batch_to_numpy` error messages and `HarmonizeReport.kept_sources` / `dropped_sources`.
- `pybvh.rotations.*` — rotation primitives (especially `quat_slerp`)
- `pybvh.analysis.*` — motion analysis descriptors (`joint_velocities`, `foot_contacts`, etc.); `pybvh.packing.*` — ML feature-array assembly (`to_feature_array`, `feature_array_layout`). Split from the former `pybvh.features` in v0.8.0.
- `pybvh.batch.*` — batch loading and normalization, plus `batch.harmonize(clips, *, reference, target_fps, target_world_up, target_rest_up, target_rest_forward, target_euler_order, on_incompatible, verbose, return_report)` for dataset-level preprocessing (topology drop/raise + retarget + resample + three-axis reorient + Euler-order unification, applied in the order world_up → rest_up → rest_forward → euler_order). Emits one summary `UserWarning` per call on drops. With `return_report=True`, returns `(clips, HarmonizeReport)` — a JSON-serializable audit trail with per-clip `applied_stages` records suitable for embedding alongside preprocessed datasets. `compute_normalization_stats` return dict includes `constant_channels` (bool mask of zero-std channels); `read_bvh_directory` accepts `skip_errors=` to tolerate corrupt files.

**Compatibility**: v0.6.0 removed all pre-0.6 `get_*` / `set_frames_from_*` / `scale_skeleton` / `change_skeleton` / `speed_perturbation` / `dropout_frames` etc. aliases outright (no deprecation cycle — the only known consumer, pybvh-ml, was briefed ahead of time). Code written against a pre-0.6 pybvh will `AttributeError` / `ImportError` until migrated. See `pybvh/API_RENAME.md` for the complete old → new mapping.

### Design history: the emo_mocap review
The two-library split was motivated by a detailed external review from a developer integrating pybvh into an ML project (emo_mocap, emotion recognition from motion capture). The review proposed 13 improvements. Our analysis:
- **Implemented in pybvh**: `euler_orders` property, `auto_detect_lr_pairs`, `__eq__`, `edges` property, better docstrings (pending)
- **Implemented in pybvh-ml**: tensor packing (CTV/TVC/flat), skeleton graph metadata, array-level augmentation (quaternion + 6D), speed perturbation/dropout on arrays, HDF5 preprocessing, PyTorch Datasets, body-part partitions
- **Rejected**: linear Euler interpolation for dropout (mathematically unsound), framework-specific graph objects (too much coupling), `to_ml_tensor` as a Bvh method (extends `to_feature_array` instead)
- **Key principle established**: pybvh owns motion data; pybvh-ml owns how ML consumes it
