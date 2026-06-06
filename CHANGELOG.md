# Changelog

All notable changes to **pybvh** are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

## [0.8.0] — Unreleased

_In progress. The breaking module split (below) lands first as a behavior-preserving step; theory-neutral motion-descriptor primitives (geometry, jerk/smoothness, gait, SE(3) transforms) follow within this same release._

### Upgrading from 0.7.0

- ⚠️ **`pybvh.features` split into `pybvh.analysis` + `pybvh.packing`.** The old module is removed — `import pybvh.features` now raises `ImportError`. Motion descriptors (`node_velocities`, `joint_velocities`, `node_accelerations`, `joint_accelerations`, `angular_velocities`, `root_trajectory`, `foot_contacts`, `auto_detect_foot_joints`) moved to **`pybvh.analysis`**; ML feature-array assembly (`to_feature_array`, `feature_array_layout`) moved to **`pybvh.packing`**. **`Bvh` method names are unchanged** — `bvh.joint_velocities()`, `bvh.to_feature_array()`, etc. all still work. Migration: `pybvh.features.X` → `pybvh.analysis.X` (or `pybvh.packing.X` for the two packing functions). Full mapping in `pybvh/API_RENAME.md`.

### Added

- **`pybvh.geometry`** — new array-pure module of position descriptors (the companion to `pybvh.rotations`): `inter_joint_distance`, `joint_angle`, `segment_axis_angle`, `triangle_area`, `point_to_plane_distance`, `point_to_segment_distance`, `bounding_box`, `bounding_sphere` (Ritter, approximate), `bounding_ellipsoid` (PCA), `centroid`, `com_displacement`, `verticality`, `path_length`, `straightness`, `curvature`, `torsion`, `movement_phase`, `ground_path`, `pose_distance`, `mean_pose_subtract`. NumPy-only; vectorized over the frame axis; zero-denominator ratios return `nan`.
- **`pybvh.tools.finite_difference`** — the single finite-difference convention shared by the kinematics ladder and the geometry derivative kernels (so derivatives composed across the two stay consistent).
- **`pybvh.analysis` motion primitives** — jerk (`node_jerk`, `joint_jerk`); smoothness on a 1-D speed profile (`sparc`, `dimensionless_jerk`, `log_dimensionless_jerk`, `number_of_peaks`, `speed_metric`, `integrated`/`mean`/`rms_squared_jerk`, and a `smoothness(metric=…)` dispatcher; SPARC/DLJ/LDLJ validated against the Balasubramanian reference); signal reductions (`velocity_reductions`, `zero_crossings`, `active_segments`, `active_duration`); `kinetic_energy`; gait (`cadence`, `stride_length`, `walking_pace`); `range_of_motion`; and covariance descriptors (`cov3dj`, `lagged_correlation`).

### Changed

- **Module layout.** `pybvh/features.py` → `pybvh/analysis.py` (descriptors) + `pybvh/packing.py` (feature-array assembly). Function behavior is byte-identical; only the import path changed.

### Removed

- **`pybvh.features` module.** Replaced by `pybvh.analysis` + `pybvh.packing`.

## [0.7.0] — 2026-05-14

### Upgrading from 0.6.0

Single big-bang release; everything below ships together. The most disruptive items first:

- ⚠️ **`Bvh.joint_angles` is now in radians.** Was degrees. Every value is ~57× smaller. Deg↔rad conversion lives at I/O: `read_bvh_file` converts on read, `write_bvh_file` converts on write, `to_df_dict` / `df_to_bvh` keep DataFrame columns in degrees for readability. Strip explicit `np.deg2rad(bvh.joint_angles)` / `np.radians(bvh.joint_angles)` — they're no-ops now.
- ⚠️ **`Bvh.joint_velocities` / `Bvh.joint_accelerations` return `(F, J, 3)`, not `(F, N, 3)`.** Per-node behavior is preserved under new names `node_velocities` / `node_accelerations`.
- ⚠️ **`Bvh.matches_topology` is stricter.** Now compares parent indices and rest offsets in addition to joint names and Euler orders.
- **`Bvh.joint_angles` / `Bvh.root_pos` return read-only views.** Mutation raises `ValueError: assignment destination is read-only`. Use copy → mutate → assign back.
- **`Bvh.spatial_coords` → `Bvh.node_positions` rename** (six-method symmetry across position/velocity/acceleration). Module-level `pybvh.frames_to_spatial_coords` similarly renamed to `frames_to_node_positions`.
- **End-site node names lose their space.** `'End Site Head'` → `'EndSiteHead'`.
- **`harmonize()` emits one summary `UserWarning` per call** instead of one per dropped clip. Programmatic callers should use `return_report=True`.

### Added

- **`Bvh.node_positions()`** — was `spatial_coords()` (renamed). Returns `(F, N, 3)` — per-node 3D positions, joints + end sites.
- **`Bvh.joint_positions()`** — new. Returns `(F, J, 3)` — joint-axis subset of `node_positions`. Index-aligns with `joint_angles` and `joint_velocities`.
- **`Bvh.node_velocities()` / `Bvh.node_accelerations()`** — was `joint_velocities` / `joint_accelerations` (renamed). Returns `(F, N, 3)`. Same `stencil`/`pad` parameters.
- **`Bvh.index(name, axis='joint'|'node')`** — unambiguous lookup that takes the axis explicitly. Avoids the silent `joint_index` vs `node_index` collision (integers differ for joints past the first end-site).
- **`Bvh.world_up_inferred` property** — read-only. Returns what the auto heuristic *would* pick, regardless of any manual `bvh.world_up = '+x'` override. Useful for auditing overrides.
- **`Bvh.matches_hierarchy(other, match_offsets=True, atol=1e-6)`** — predicate for joint names + parent structure + rest offsets. Pass `match_offsets=False` to ignore bone proportions (caller about to retarget).
- **`Bvh.matches_channels(other)`** — predicate for per-joint Euler rotation orders + root position-channel order.
- **`Bvh.source_path` attribute** — on-disk origin populated by `read_bvh_file`. Preserved through `copy()` / `slice_frames()` / single-source `concat()`. Surfaced in `__str__`, `batch_to_numpy` error messages, `HarmonizeReport`.
- **`pybvh.frames_to_node_positions`** — module-level function (renamed from `frames_to_spatial_coords`).
- **`harmonize(target_euler_order='XYZ')`** — pipeline stage that re-expresses every kept clip's joint angles in a uniform Euler order via `Bvh.change_euler_order`. Orientation-preserving.
- **`harmonize(return_report=True)` + `HarmonizeReport` dataclass** — JSON-serializable per-call audit. Carries `kept_indices` / `kept_sources` / `dropped_indices` / `dropped_sources` / `drop_reasons` / `applied_stages` (per-clip dicts recording every stage). Serialize via `json.dumps(dataclasses.asdict(report))`.

### Changed

- **`Bvh.joint_angles` unit**: degrees → radians. The deg↔rad conversion now lives entirely at I/O boundaries (`read_bvh_file`, `write_bvh_file`, `to_df_dict`, `df_to_bvh`). Internal rotation conversions (`to_rotmat`, `to_6d`, `from_*`, `change_euler_order`, transforms) work directly in radians; `degrees=True` kwargs were stripped from all internal callers. `pybvh.rotations.*` keeps its `degrees=` kwarg as the public bridge API. `angular_velocities` defaults stay (radians/sec; `degrees=True` opt-in).
- **`Bvh.joint_angles` / `Bvh.root_pos` getters return read-only views.** Closes the silent-corruption footgun where `angles = b.joint_angles; angles -= angles.mean(axis=0)` mutates the Bvh.
- **`Bvh.spatial_coords` renamed to `Bvh.node_positions`.** Six-method symmetry: `joint_*` always = `(F, J, 3)`, `node_*` always = `(F, N, 3)`. Module-level `pybvh.frames_to_spatial_coords` similarly renamed.
- **`joint_velocities` / `joint_accelerations` semantics**: per-joint `(F, J, 3)` (was per-node). Closes the latent layout asymmetry in `to_feature_array` — rotation and velocity blocks now share joint indexing.
- **`feature_array_layout(...)` signature**: `num_nodes` keyword argument removed; velocity block sizes from `num_joints`.
- **`Bvh.matches_topology` = `matches_hierarchy(other) and matches_channels(other)`** — closes the latent offset/parent-index gap.
- **`Bvh.lr_mapping` is bidirectional.** Both directions of every pair are present (`m['LeftArm'] == 'RightArm'` AND `m['RightArm'] == 'LeftArm'`). Setter accepts one-directional input and symmetrizes internally.
- **End-site names**: synthesized as `'EndSite<parent>'` instead of `'End Site <parent>'`. File I/O unchanged (BVH writes end-sites without name).
- **`Bvh.__str__` format**: `"24 joints, 75 frames at 30.0 fps (frame_time=0.033333s, from <basename>)"` — was "`elements in the Hierarchy`". `, from <basename>` appended when `source_path` is set.
- **`batch_to_numpy` is now representation-aware** — clips with different per-joint Euler orders no longer fail for `'6d'`/`'quaternion'`/`'rotmat'` (channel layout doesn't depend on Euler order there); only for `'euler'`/`'axisangle'`.
- **`batch_to_numpy` error messages are actionable** — name both clips by `source_path` (or index), identify the first divergent joint with both Euler orders, point at the recovery primitive.
- **`harmonize()` emits one summary `UserWarning` per call** (was per-clip). `UserWarning` channel; capturable via `warnings.catch_warnings()`. Set `verbose=False` to silence; `return_report=True` for structured drops.
- **`harmonize()`'s topology gate uses `matches_hierarchy(match_offsets=False)`** instead of `matches_topology`. Same 0.6.0 behavior for offset-divergent clips (accepted, retargeted) — but explicit now.
- **`add_noise(sigma_deg=...)`** still accepts degrees of noise (user-facing); internally converts to radians once and wraps to `[-π, π]` (was `[-180, 180]`).
- **`joint_index` / `node_index` property docstrings** include a prominent warning about the integer-collision silent footgun, recommending `bvh.index(name, axis=...)` for ambiguous call sites.
- **Velocity/acceleration docstrings** note that `centered='first'` and `centered='world'` produce identical results (constant offsets vanish under differentiation).

## [0.6.0] — 2026-04-19

### Upgrading from 0.5.1

Most of the changes in 0.6.0 are additive. The ones that are likely to bite an existing pipeline:

- **`joint_velocities`, `joint_accelerations`, `angular_velocities`, `to_feature_array(include_velocities=True)`, `root_trajectory(include_velocities=True)`** now return shape `(F, ...)` by default (previously `(F-1, ...)` / `(F-2, ...)`). Add `stencil="forward", pad="none"` to any call site that relied on the old leading-dimension drop.
- **`to_rotmat()`, `to_6d()`, `to_quaternions()`, `to_axisangle()` return `(root_pos, joint_data)` — 2-tuple, not 3-tuple.** The third `joints` element was removed; it was derivable from `bvh.nodes` / `bvh.joint_names` / `bvh.joint_index` and no caller used it. Rewrite `_, data, _ = bvh.to_X()` to `_, data = bvh.to_X()`.
- **`foot_contacts` default method changed from `"velocity"` to `"combined"`.** Pass `method="velocity"` explicitly if you need the old behaviour. The `threshold` parameter has been removed — use `vel_threshold` / `height_threshold`.
- **`scale(factor)` now scales `root_pos`** as well as bone offsets. If you were manually scaling `root_pos` after calling `scale`, remove that step or you will double-scale.
- **`frame_frequency` was misnamed** — it has always *stored frame time in seconds*, not frequency in fps. The `frame_time` rename in this release fixes the name. Note: if you ever wrote code that interpreted `bvh.frame_frequency` as fps (e.g. `fps = bvh.frame_frequency`), it was wrong. The correct expression is now `fps = 1.0 / bvh.frame_time`.
- **All old-API aliases are removed**, not deprecated — see "Removed" below. Because pybvh has no known production consumers beyond pybvh-ml (which has been briefed), this release does the rename work in one break rather than a two-release deprecation cycle. Migration: consult [API_RENAME.md](pybvh/API_RENAME.md) for every old→new mapping.
- **`root_relative_positions()` is removed.** Replace with `bvh.spatial_coords(centered='skeleton')` (mathematically identical).
- **`root_trajectory` heading reference changed** from a hardcoded `+x` to the rig's rest-pose forward. Heading numerical values change for rigs whose rest-forward is not `+x` (Mixamo, 3ds Max). Recalibrate if your model was trained against the old values.
- **`Bvh.write()` and `pybvh.io.write_bvh_file()` default flipped to `verbose=False`.** Previously printed a one-line "Successfully saved..." confirmation by default, which floods the terminal in preprocessing loops that write many files. Pass `verbose=True` explicitly if you want the per-file print back.

### Breaking changes

- **`joint_velocities`, `joint_accelerations`, `angular_velocities` default shape changed to `(F, ...)`** — previously `(F-1, ...)` / `(F-2, ...)` via forward differences. Two orthogonal parameters — `stencil` ("central" vs "forward") and `pad` ("edge" vs "none") — now control the finite-difference method and the boundary handling independently. Defaults `stencil="central", pad="edge"` match `np.gradient` semantics (central interior + one-sided edges) for linear velocities/accelerations, and the two-step relative rotation `R_rel = R_{i-1}^T @ R_{i+1}` (axis-angle divided by 2) interior + one-sided boundaries for `angular_velocities`. Result: `ω[i]` and `vel[i]` are time-aligned ("rate of change at frame i") so stacking them via `np.concat([vel, ω], axis=-1)` produces a consistent per-frame feature block. Key invariant: `np.gradient(bvh.joint_velocities(), bvh.frame_time, axis=0) == bvh.joint_accelerations()` exactly under the defaults. Users who need the old forward-difference behavior pass `stencil="forward", pad="none"`. See the stencil × pad tables in the per-function docstrings for the full 4-combination shape matrix.
- **`to_feature_array(include_velocities=True)` returns `(F, D)` by default** — previously `(F-1, D)`. The new default aligns all feature blocks at shape `(F, ...)` so row `i` in the feature array corresponds to frame `i` in the source motion. Pass `stencil="forward", pad="none"` to restore the previous first-frame-drop; `stencil="central", pad="none"` drops the first and last frames symmetrically and returns `(F-2, D)`.
- **`root_trajectory` heading reference is now rest-pose forward** — previously hardcoded to `ground_axes[0]` (i.e. always `+x` for the common up-axis conventions), which silently mis-reported heading for any rig whose rest-forward wasn't `+x` (Mixamo rigs facing `+z`, 3ds Max rigs facing `+y`, etc.). The heading now uses `_compute_forward_at(bvh, rest_pose_coords, world_up)` — matches the convention already used by `reorient_rest_forward`. Heading numerical values change for affected rigs; users who depended on the old (incorrect) offset must recalibrate.
- **`foot_contacts` default method is now `"combined"`** (velocity AND height) — previously `"velocity"` only. Matches the best-in-class open-source heuristic (HuMoR, Kovar 2002) and catches failure modes of either signal alone: a stationary foot in mid-air is no longer wrongly labeled as contact (velocity-only FP), and a low fast-sliding foot is no longer wrongly labeled either (height-only FP). Users can recover the old default by passing `method="velocity"` explicitly.
- **`foot_contacts` scale reference is now `skeleton_scale` (mean rest root→foot distance), not median bone length.** Median bone length is unstable: a skeleton with many short finger bones gets a median 2-3× smaller than the same skeleton without fingers, which inappropriately tightens the thresholds. `skeleton_scale` is topology-aware (uses only the leg chain, via the detected/supplied foot joints), orientation-independent (Euclidean distance — no axis projection, so rest-pose-vs-world-up mismatches don't break it), and monotonic with overall skeleton size. Default velocity threshold: `0.004 × skeleton_scale` (reverse-engineered from HuMoR-equivalent on SMPL, plus pivot-foot tolerance on real BVH). Default height clearance: `0.013 × skeleton_scale`. On finger-free skeletons these produce near-identical numerics to the previous `0.03 × median_bone / 0.10 × median_bone` defaults; on finger-rich skeletons (bvh_test3-style) they now track the actual skeleton scale.
- **`foot_contacts` parameter `threshold` removed** — the single `threshold` parameter was ambiguous once `method="combined"` began using both signals. Replace with `vel_threshold=` (for `method="velocity"`) or `height_threshold=` (for `method="height"`).
- **`foot_contacts` height method now separates floor estimation from clearance threshold** — previously `threshold = np.percentile(heights, 5) + 0.10 * bone` conflated "where is the floor" with "how close is close enough". The new `floor="auto"` (default) estimates the floor as the 2nd percentile of the per-frame minimum foot height, and `height_threshold` is a pure clearance above it. The auto-estimate is more robust to clips where feet never plant simultaneously. Users can pass an explicit `floor=<float>` to pin it. Numerical output is close to the old default but not identical.
- **`foot_contacts` temporal filters: `min_contact_duration` / `min_gap_duration` (seconds), both default `0.1 s`**. Replaces the earlier frame-based parameter. Specifying in seconds is fps-independent — the same physical motion at 30 fps and 120 fps gets the same filtering. Default 0.1 s (3 frames at 30 fps) is UnderPressure's published value and matches the pivot-gap duration observed on real BVH walking data. `min_contact_duration` runs a morphological open (removes contact jitter shorter than 100 ms); `min_gap_duration` runs a morphological close (bridges non-contact gaps where the joint briefly exceeds `vel_threshold` due to ankle rotation during stance). Set either to `0.0` to disable. Internally converted via `max(1, round(duration / frame_time))`, so sub-frame durations round to no-ops.
- **`foot_contacts(method="velocity")` frame-0 handling changed** — frame 0 now propagates from frame 1 (`contacts[0] = contacts[1]`) instead of being filled with `1.0`. Shape is still `(F, num_feet)`; the change makes frame 0 consistent with the rest of the clip rather than an arbitrary "assumed contact" sentinel.
- **`foot_contacts(method="height")` now raises `ValueError` when `world_up` is inconsistent with rest geometry** — specifically, when feet are above hips along the declared up axis at rest. Previously produced silently wrong contacts when `bvh.world_up` was auto-detected incorrectly (the ~5% of files where `_infer_world_up` gets it wrong).
- **`foot_contacts` auto-detection: topology filter + most-distal-wins**. Substring match on `"foot"`/`"toe"` is followed by a hard filter on tip descendants (end-site or toe-named child) — IK helpers like `LeftFootIK` are dropped. Among the remaining candidates, those whose subtree contains another candidate are also dropped — so on rigs with `Foot → ToeBase → EndSite`, only `ToeBase` is returned (the more distal ground-contacting joint). Detected order is deterministic: sorted by rest-pose height along `world_up`, alphabetical within ties. Call `bvh.auto_detect_foot_joints()` to preview the detection.

### Removed

Old API names renamed in this release are **removed outright** (no deprecation cycle). Because pybvh has no known production consumers beyond pybvh-ml, shipping deprecation wrappers that would be removed one release later wasn't worth the churn. See [API_RENAME.md](pybvh/API_RENAME.md) for the complete old → new mapping.

Names removed from the public surface:

- **`Bvh` class methods** — `get_frames_as_rotmat`, `get_frames_as_6d`, `get_frames_as_quaternion`, `get_frames_as_axisangle`, `set_frames_from_6d`, `set_frames_from_quaternion`, `set_frames_from_axisangle`, `get_spatial_coord`, `get_rest_pose`, `get_df_constructor`, `to_bvh_file`, `change_skeleton`, `scale_skeleton`, `single_joint_euler_angle`, `change_all_euler_orders`, `add_joint_noise`, `speed_perturbation`, `dropout_frames`, `get_joint_velocities`, `get_joint_accelerations`, `get_angular_velocities`, `get_root_relative_positions`, `get_root_trajectory`, `get_foot_contacts`, `root_relative_positions`.
- **`Bvh.frame_frequency` property** — replaced by `Bvh.frame_time` (note the semantic correction: seconds, not frequency).
- **Module-level functions** — `pybvh.transforms.{add_joint_noise, speed_perturbation, random_speed_perturbation, dropout_frames}`; `pybvh.features.{get_joint_velocities, get_joint_accelerations, get_angular_velocities, get_root_relative_positions, get_root_trajectory, get_foot_contacts, root_relative_positions}`; `pybvh.spatial_coord.frames_to_spatial_coord` (singular — was a typo for `frames_to_spatial_coords`).
- **`euler_column_names`** — the property renamed to the private `_euler_column_names`. Was never used outside tests.

### Changed (internal-only; no public-API behavior change)

- **Internal orientation helpers flipped to the leftward convention.** `_rest_lateral` → `_rest_leftward` (now `left − right`); `_world_lateral_unit_at_frame` → `_world_leftward_unit_at_frame` (same flip). `_compute_forward_at` now computes `forward = leftward × up` instead of `forward = up × lateral_right`, producing the same forward axis as before. Consumers that used the old helpers as vectors (`mirror`'s axis detection, the `follow` camera mode) were audited: mirror consumes only the axis letter (sign-insensitive), and the `follow` camera consumes the vector inside a signed-angle delta between two frames (sign cancels out), so the flip has no observable behavior change.

### Added

- **`Bvh.fps` property** — convenience inverse of `Bvh.frame_time` (`fps = 1.0 / frame_time`). Read/write: setting it rewrites `frame_time = 1.0 / fps`. Returns `0.0` when `frame_time == 0` (the unset sentinel) instead of raising. Mirrors how most user code thinks ("at 30 fps...") versus how the format stores ("seconds per frame"). Setter rejects non-positive values.
- **`pybvh.api_rename_path()` helper** — returns the on-disk path to the bundled `API_RENAME.md` reference (now shipped inside the package). Lets downstream migration scripts read the old → new symbol mapping without depending on internet access or repo layout.
- **`Bvh.left_at(frame=0)`** — orientation method returning the signed axis pointing toward the character's left in world coordinates at the given frame. Completes the `(world_up, forward_at, left_at)` orthonormal triple with the right-hand-rule convention `left = world_up × forward_at`. Frame-dependent (tracks hip/shoulder rotation), matching `forward_at`'s shape. Fills the gap downstream consumers were working around with `({"x","y","z"} - {up[1], fwd[1]}).pop()` + a manual sign guess.
- **`Bvh.rest_up` property** — skeleton's topological up axis, derived from rest-pose offsets only (pose-independent). Read-only; complements the existing animation-derived `world_up`. On a clean file the two agree; divergence indicates a file authored with rest pose in one convention and animated in another (fix with `reorient_rest_up`). Replaces the need to reach for the private `pybvh.tools._rest_upward` in downstream uniformity-diagnostic code.
- **`Bvh.rest_forward` property** — skeleton's topological forward axis, derived from the rest pose only. Read-only; complements `rest_up` (rest-pose orientation pair) and parallels `forward_at(frame)` (animation-derived). Useful for dataset uniformity checks and as the cheap guard inside `harmonize(target_rest_forward=...)`.
- **`coords=` parameter on `Bvh.forward_at()` and `Bvh.left_at()`** — pass a pre-computed `(F, N, 3)` spatial-coordinates array to skip the per-call forward kinematics. The selected frame's slice is taken via `coords[frame]`. Matches the existing pattern on `foot_contacts(coords=...)`. Useful for hot loops that scan many frames (dataset uniformity diagnostics, per-frame heading export). Removes the only remaining downstream reason to reach for the private `pybvh.tools._compute_forward_at`.
- **`batch.harmonize(target_rest_up=...)` and `batch.harmonize(target_rest_forward=...)`** — two reorient kwargs on the batch harmonizer beyond `target_world_up`. Each wraps the corresponding `Bvh.reorient_*` method and is guarded by an equality check (no-op when the clip is already aligned). Ordering inside `harmonize` is `world_up → rest_up → rest_forward`, matching the physical layering (world frame → rest topology up → rest topology facing). Removes the boilerplate list-comprehension that callers of pybvh-ml's `preprocess_directory` were writing to apply these uniformly.
- **Per-joint Euler orders in `rotations.euler_to_rotmat()` / `rotations.rotmat_to_euler()`** — pass a sequence of length-3 order strings (e.g. `['ZYX', 'ZYX', 'ZXY', ...]`, one per joint along axis `-2`) in addition to the existing single-order form. Joints sharing an order are grouped internally so the math vectorizes across all joints in one call. Removes the per-joint Python loop internal code (and downstream consumers like pybvh-ml) had to write.
- **`pybvh.rotations.convert(data, from_repr, to_repr, *, order=None, degrees=False)`** — string-dispatch converter between rotation representations (`"euler"`, `"rotmat"`, `"6d"`, `"quaternion"`, `"axisangle"`). Thin pivot through rotation matrices; all pairs reachable in one call. Accepts the same per-joint `order=` sequence as `euler_to_rotmat`.
- **`pybvh.rotations.REPRESENTATION_CHANNELS`** — public dict mapping representation name → per-joint channel count (`euler=3`, `axisangle=3`, `quaternion=4`, `6d=6`, `rotmat=9`). Replaces the private `_REPRESENTATION_WIDTHS` that used to be duplicated across modules.
- **`Bvh.node_edges` property** — parallel to `Bvh.edges` but in `nodes` index space (includes end sites). Use when the downstream graph treats end sites as real leaves (visual skeleton, per-bone styling, GCN inputs over the full topology).
- **`Bvh.lr_pairs` property** — cached left/right joint-pair list in `joint_angles` index space (`list[tuple[int, int]] | None`). Index-space counterpart to `Bvh.lr_mapping`, derived from the same cache. Both return `None` when no pairs are available — one consistent sentinel.
- **`centered=` is now applied to the `root_pos` block of `to_feature_array()`** — previously the parameter only reached velocities and foot contacts. With `centered="first"` the root_pos block is first-frame-zeroed; with `centered="skeleton"` it is all zeros; `centered="world"` (default) keeps the raw positions. Makes the output consistent with `spatial_coords(centered=...)`.
- **Orthogonal `stencil=` and `pad=` keywords** on `joint_velocities`, `joint_accelerations`, `angular_velocities`, `to_feature_array`, and `root_trajectory`. `stencil` picks the finite-difference method (`"central"`, default, or `"forward"`); `pad` picks boundary handling (`"edge"`, default — output has the same shape as the input — or `"none"` — drop boundary frames). All 4 combinations are supported: the previously unreachable `stencil="forward", pad="edge"` (forward differences padded to `(F, ...)`) and `stencil="central", pad="none"` (strict central returning `(F-2, ...)`) are now first-class. See the per-function docstrings for the full shape matrix.
- **`degrees=` keyword** on `angular_velocities` and `root_trajectory`. Default `False` (radians — current behaviour); `True` converts the output from radians to degrees. For `root_trajectory`, only the `heading_vel` column is affected — `ground_*_vel` columns are linear positions per second. Matches the `degrees=` convention on `pybvh.rotations` functions.
- **`root_trajectory(include_velocities=True)` parameter** — when `True`, appends `[ground_a_vel, ground_b_vel, heading_vel]` columns for a total `(F, 7)` output under the defaults. Leading dimension depends on the chosen `stencil` × `pad` combination. Heading velocity is computed from `np.unwrap`-ed heading, handling ±π wraparound correctly. Mirrors `to_feature_array`'s `include_velocities` convention. Also raises `ValueError` if `frame_time == 0` (consistent with `joint_velocities`).
- **`feature_array_layout(...)` pure function** — returns `{block_name: slice}` for every block (`root_pos`, `rotations`, `velocities`, `foot_contacts`) in a `to_feature_array` output. Keyword-only signature, no `Bvh` required (model-shape setup before data is loaded works). Also exposed as `bvh.feature_array_layout(...)` method.
- **`auto_detect_foot_joints(bvh)` pure function + `bvh.auto_detect_foot_joints()` method** — exposes the topology-based foot detection used internally by `foot_contacts`. Returns a deterministically ordered joint-name list (height + alphabetical) so callers can preview what `foot_contacts` will use and/or feed the list back in explicitly.
- **`foot_contacts` dual temporal filters** — `min_contact_duration` (open) and `min_gap_duration` (close), both in seconds, both default `0.1 s`. Open removes contact jitter; close bridges pivot-foot artefacts where the ToeBase joint velocity briefly spikes during ankle flexion even though the foot is physically planted. Both set to `0.0` to get raw per-frame output.
- **`foot_contacts(return_info=True)`** — opt-in structured return. When set, returns `(contacts, info)` where `info` is a dict with keys `joints`, `method`, `min_contact_duration`, `min_gap_duration`, `skeleton_scale` (present when auto-calibration ran), and — when the relevant signal participated — `vel_threshold`, `height_threshold`, `floor`. Makes detection self-documenting and gives downstream consumers (including a future learned backend in `pybvh-ml`) a stable place to introspect / override the parameters. Non-breaking: default `return_info=False` returns the same ndarray as today.
- **`foot_contacts(floor="auto" | float)`** — explicit floor handling. `"auto"` (default) estimates from the 2nd percentile of per-frame minimum foot height; a float pins the floor directly. Reported in raw world-axis coordinates (not sign-corrected), so `info["floor"]` is directly comparable to `bvh.root_pos[:, up_idx]`.
- **`Bvh.joint_index` property** — dict mapping joint name to its index in `joint_angles` axis 1 (joint-only, excludes end sites). Symmetric counterpart to the existing `bvh.node_index` (which indexes `bvh.nodes`, includes end sites). Recommended over `bvh.joint_names.index(name)` for joint-axis lookups; the two return *different* integers for the same name whenever an end site precedes the joint in the depth-first walk, and the mismatch is a common source of silent bugs. Both properties keep their pre-existing behaviour; this is purely additive.
- **`Bvh.lr_mapping` property** — cached left/right joint pair mapping (`dict[str, str] | None`). Auto-detected eagerly at `__init__` from joint names via an extended heuristic; settable via property setter or `lr_mapping=` kwarg at load time; carried through `bvh.copy()`. Used by `mirror()`, `forward_at()`, and `reorient_rest_forward` so they all share one canonical L/R source.
- **`lr_mapping=` parameter on `read_bvh_file()`, `read_bvh_directory()`, and `Bvh.__init__`** — pass an explicit mapping at load time to bypass auto-detection. Useful for skeletons whose naming conventions the heuristic can't parse, or when applying the same mapping to a whole dataset.
- **Extended L/R name heuristic** — `auto_detect_lr_mapping` (and the cached `bvh.lr_mapping`) now recognize: Blender-style `.L`/`.R` and lowercase `.l`/`.r` suffixes, underscore variants `_L`/`_R`/`_l`/`_r`, full-word `.Left`/`.Right` and `_Left`/`_Right` suffixes, Mixamo `mixamorig:` namespace prefix (auto-stripped before matching), and Blender numbered duplicates `.001` (auto-stripped before matching). Existing `Left`/`Right` substring and `L`/`R`+uppercase prefix rules preserved. Rules tried most-specific-first; mutual-match required (singletons aren't paired).
- **`Bvh.random_translate_root()`, `Bvh.random_rotate_vertical()`, `Bvh.random_perturb_speed()` method wrappers** — the stochastic-augmentation variants now have method forms matching the existing `Bvh.translate_root()`, `Bvh.rotate_vertical()`, `Bvh.perturb_speed()` pattern. Method-form augmentation chains stay clean (`bvh.mirror().random_rotate_vertical(rng=rng).add_noise(sigma_deg=1.0, rng=rng)`).
- **`world_up` parameter on `read_bvh_file()` and `read_bvh_directory()`** — pass `world_up="+y"` at load time to skip auto-detection and suppress the disagreement warning. Defaults to `"auto"` (current behavior). Also accepted by `Bvh.__init__`.
- **`warn_on_world_up_disagreement` parameter on `read_bvh_file()`** — set to `False` to silence the `UserWarning` when rest-pose and first-frame inferences disagree, without overriding the detected value.
- **`reorient_world_up(new_up)`** — apply a global rotation to the entire animation so the world vertical axis changes (e.g. Z-up to Y-up). The character looks visually identical; only the coordinate system changes. Restricted to axis-aligned rotations for lossless transformation. Available as both `Bvh` method and `transforms.reorient_world_up()`.
- **`reorient_rest_up(new_up)`** — rotate the skeleton's rest-pose offsets so its topological up aligns with `new_up`, compensating all joint rotations so FK positions are unchanged. Fixes files where rest pose and animation disagree on the up axis. Available as both `Bvh` method and `transforms.reorient_rest_up()`.
- **`reorient_rest_forward(new_forward)`** — rotate the skeleton's rest-pose offsets so the character faces `new_forward`, compensating all joint rotations so FK positions are unchanged. Available as both `Bvh` method and `transforms.reorient_rest_forward()`.
- **Shape validation on `root_pos` and `joint_angles` setters** — assigning an array with wrong shape (e.g. 1D for root_pos) now raises `ValueError` with a clear message.

- **`Bvh.world_up` property** — public gravity-axis attribute as a signed-axis string (e.g. `'+y'`). Auto-detected eagerly at `__init__` from the first animation frame's head-above-hips direction, with rest-pose topology as a fallback. Settable with validation (`ValueError` for anything outside `{+x, -x, +y, -y, +z, -z}`). Manual overrides propagate through `copy()`, `slice_frames()`, `mirror()`, `scale()`, `rotate_vertical()`, and `translate_root()`. `retarget()` re-infers from the new skeleton. A `UserWarning` is emitted when rest pose and first frame disagree on the dominant vertical axis (pointing the user at the manual override for the ~5% of files where this happens, e.g. Blender exports that author rest pose in one convention and animate in another).
- **`Bvh.forward_at(frame=0)` method** — character's world-space facing direction at a given frame, derived from actual joint positions (L/R joint symmetry projected onto the horizontal plane, crossed with `world_up`). Tracks root rotation / hip twist / shoulder rotation as the character moves; not just rest-pose topology. Returns a signed-axis string.
- **`bvhplot.render(..., follow=True)`** — new keyword that makes the camera **track the character's rotation smoothly** over the animation. Uses continuous signed-rotation-delta tracking (no 45° snap), so the view orbits in lockstep with the character's facing direction. Custom `(azim, elev)` tuples are fixed angles and ignore `follow` automatically. Available in both the OpenCV and matplotlib backends.
  - Follow mode pre-computes per-frame view matrices once, and derives a view-angle-invariant scale (`fixed_view_halves` passed to `ortho_project`) so the character doesn't appear to zoom in and out as the camera orbits.
- **Bvh visualization wrapper methods** — `bvh.plot_rest_pose()`, `bvh.plot_frame(frame=0)`, `bvh.plot_trajectory()`, `bvh.render(path)`, `bvh.play()` now exist on the `Bvh` class as convenience wrappers around the corresponding `pybvh.bvhplot` functions. Makes the common single-object case cleaner (`bvh.plot_frame()` instead of `pybvh.bvhplot.frame(bvh)`). Multi-skeleton comparisons still use the module-level functions which accept a list.
- **`ax` parameter** on `bvhplot.rest_pose()`, `frame()`, and `trajectory()` — pass an existing matplotlib Axes (must be 3D for the pose functions, 2D for trajectory) to draw into it instead of creating a new figure. Validated: wrong-dimension axes raise `ValueError` with a clear fix suggestion. Enables custom layouts like `plt.subplots(..., subplot_kw={'projection': '3d'})` grids.
- **Per-skeleton camera and bounding box** in side-by-side visualizations — previously, `frame([bvh1, bvh2], ...)` used the first skeleton's camera angles and a single unified bounding box for all subplots, which left mixed-up-axis or differently-scaled skeletons rotated on their side and squeezed into a corner. Now each subplot has its own camera orientation and cubic box centered on its own skeleton.
- **Start/end legend markers on `bvhplot.trajectory()`** — always-visible legend entries showing which end of each path is the start (circle) and which is the end (square), using neutral gray markers that don't conflict with per-skeleton line colors.
- **`facing_arrows=True` on `bvhplot.trajectory()`** — overlay ~10 small arrowheads along each skeleton's path showing the character's facing direction at those frames, computed from the `heading_sin`/`heading_cos` columns of `root_trajectory()`. Arrow length scales with the path's span (≈8 %), arrows use the skeleton's line color, and multi-skeleton clips each sample their own 10 positions. Default False (no visual change). Also accessible via `bvh.plot_trajectory(facing_arrows=True)`.
- **`tight=False` default on `bvhplot.trajectory()`** — axis range now matches the full horizontal extent of the skeleton across all joints and frames (matches `bvh.play()`'s bounding box) instead of auto-scaling to just the root path. Keeps motion scale honest relative to the character's body so a near-stationary clip doesn't get auto-zoomed into looking like a large walk. **Mildly breaking**: existing trajectory plots of small motions will show wider axes than before. Pass `tight=True` for the previous behaviour. Multi-skeleton plots use the union of per-skeleton extents.
- **`Bvh.matches_topology(other) -> bool`** — predicate for skeleton-topology compatibility. Returns True iff `joint_names` and `euler_orders` match; does NOT compare bone offsets or motion data (for that, use `==`). Surfaces the check previously inlined in `batch_to_numpy` and used by the new `batch.harmonize`. Primary use: precondition before batching or retargeting.
- **`batch.harmonize(clips, *, reference, target_fps, target_world_up, target_rest_up, target_rest_forward, on_incompatible, verbose)`** — dataset-level harmonization in one call. Applies per-clip: topology check vs `reference` (drop or raise per `on_incompatible`), retarget to `reference`'s bone offsets, resample to `target_fps` (when current fps differs by more than 0.01), then the three reorient stages applied in order `world_up → rest_up → rest_forward` (each guarded by an equality check). All targets are optional; any subset can be `None` to skip that stage. Default `on_incompatible="drop"` + `verbose=True` drops incompatible clips with a `UserWarning` per drop; `on_incompatible="raise"` rejects the whole batch on first mismatch. Replaces the ~15-line harmonization loop users would otherwise write by hand.
- **`constant_channels` key on `compute_normalization_stats` return** — bool array of shape `(D,)`, True where the raw standard deviation was below `1e-8` and the guard replaced it with `1.0`. Lets downstream code exclude constant channels from per-channel diagnostics (their normalized values are identically zero, not ~N(0,1)) without rediscovering which channels were guarded. Bool arrays round-trip cleanly through `np.savez` / `np.load`.
- **`read_bvh_directory(skip_errors=False)` parameter** — when True, files that fail to load emit a `UserWarning` and are skipped rather than propagating the exception. Default False keeps current strict behaviour. Useful for real-world datasets with occasional corrupt files.

### Fixed

- **`scale()` now scales `root_pos`** — previously only scaled node offsets, leaving root translation at original world-space position. `bvh.scale(0.01)` now correctly converts both bone lengths and root position (e.g. centimeters to meters). **Breaking**: users who manually scaled `root_pos` after `scale()` will now get double-scaling.
- **`rotate_vertical()` ignored negative up-axis sign** — `world_up='-y'` was treated identically to `'+y'`, producing wrong rotation direction. The sign is now extracted and applied to the rotation angle.
- **`foot_contacts()` height method inverted for negative up axes** — with `-y` up, the highest positions were flagged as contacts instead of the lowest. Heights are now normalized by the up-axis sign before comparison.
- **`mirror_angles()` used wrong Euler order for swapped joints** — negation happened after L/R data swap, applying the destination slot's Euler order to the source joint's data. Negation now occurs before the swap, using each joint's own order. Only affects skeletons where L/R paired joints have different Euler orders.
- **Proper Euler angle extraction (`rotmat_to_euler`)** — the sign formula for proper Euler orders (ZYZ, XYX, XZX, YXY, YZY, ZXZ) was incorrect, producing wrong angles. Both the safe case and gimbal lock fallback are now correct.
- **`__eq__` now compares `frame_time` and `euler_orders`** — previously, two Bvh objects with different frame rates or Euler orders but identical raw angles compared as equal.
- **`Bvh.__init__` mutable default argument** — `nodes=[BvhRoot()]` was shared across calls. Now uses `None` sentinel with fresh creation inside the body. Same fix for `BvhJoint` and `BvhRoot` `children` defaults.
- **`_world_up_cached` never invalidated** — `root_pos` and `joint_angles` are now properties whose setters clear the cached world-up, preventing stale orientation data after data mutation.
- **`extract_joints()` lost `_world_up_override`** — user-set `world_up` is now propagated to the extracted skeleton.
- **`Bvh.add_noise()` missing `wrap` parameter** — the wrapper now forwards the `wrap` keyword to `transforms.add_noise()`.
- **`from_6d`/`from_quaternions`/`from_axisangle` missing frame count validation** — mismatched `root_pos` and joint data frame counts now raise `ValueError`.
- **`spatial_coords()` rejected negative indices** — now supports Python-style negative indexing (`-2` for second-to-last frame). `-1` retains legacy "all frames" behavior.
- **Parser: uninitialized data when frame count is wrong** — now validates that actual data lines match the declared `Frames:` count.
- **Parser: blank lines caused crash** — empty lines in the hierarchy section are now skipped.
- **Parser: `print()` instead of raising** — missing frame count or frame time now raises `ValueError`.
- **`test_file()` raised `ImportError`** — now raises `FileNotFoundError` / `ValueError`.
- **`auto_detect_lr_mapping()` was case-sensitive** — now uses case-insensitive matching (e.g. `"leftArm"`) consistent with `_find_lr_joint_pairs`.
- **`__str__` showed "frequency" for frame_time** — now shows `"75 frames at 30.0 fps (frame_time=0.033333s)"`.
- **Error messages said "frame_frequency"** — updated to "frame_time" in `joint_velocities`, `joint_accelerations`, `angular_velocities`, and `concat` warning.
- Dead code removed: `all_names` in `auto_detect_lr_mapping`, `problem` in `_check_channels`, `er_str` in `_hier_dict_to_list`, `line_number = line_number` in `io.py`.
- **`camera='front'` showed the back of rotated-root characters** — `get_forw_up_axis()` used pose-independent rest-pose offsets, so for files where the animation rotates the character 180° from rest (e.g. `bvh_test2.bvh`), the camera was placed on the wrong side of the skeleton. The new `forward_at()` uses actual joint positions and is pose-dependent; `camera='front'` now correctly shows the character's chest/face regardless of root rotation.
- **`get_camera_angles()` ignored negative forward signs** — rotating a character 180° around the vertical axis kept the camera azimuth unchanged (instead of flipping by 180°). The fix applies the 180° azimuth offset to any negative forward, not just when the forward axis matches the up's default front axis.
- **`show=True` in static plots was unsafe for composition** — `bvhplot.rest_pose()`, `frame()`, and `trajectory()` now default to `show=False`. Users can still explicitly pass `show=True` for one-liner scripts; the new default makes it safe to call `ax.set_title(...)` and other customizations after the plot function returns.

### Changed

- **`mirror()` error message** — when no L/R pairs are available (auto-detect fails and no explicit mapping was provided), `mirror()` now raises `ValueError` pointing the user at concrete remediation: setting `bvh.lr_mapping = {...}` post-load or passing `lr_mapping=` at load time. Replaces the older "ensure joints contain 'Left'/'Right' in their names" message which only addressed one of the supported overrides.
- **`reorient_rest_forward()` now succeeds on skeletons whose source forward direction previously couldn't be derived** from L/R joint symmetry alone. Any skeleton for which `bvh.lr_mapping` resolves (via the extended name heuristic or an explicit user mapping) is now accepted.
- **`get_forw_up_axis()` and `get_up_axis_index()` removed** from `pybvh.tools` — replaced by the `Bvh.world_up` property and `Bvh.forward_at()` method. Both removed functions were internal-only with no documented use, so no deprecation shim.
- **`bvhplot.trajectory()` legend is now always present** (previously only when `labels` were passed), showing the start/end marker key.

---

## [0.5.1] — 2026-04-05

### Added

- **Vedo desktop viewer overhaul** — major performance and feature improvements:
  - **Merged mesh rendering** — all bones merged into 1 mesh, all joints into 1 mesh (2 VTK actors per skeleton instead of ~120). Bones updated via vectorized numpy `einsum`. Capable of ~490fps on modern hardware.
  - **Wall-clock timing** — frame advancement uses `time.perf_counter()` for correct playback speed regardless of frame drops.
  - **FPS selector** — left panel control with presets (15/30/60/120/native). Key `F` to cycle. 30fps cap removed for vedo backend.
  - **Ping-pong playback** — `L` key cycles through loop / ping-pong / off modes.
  - **Joint name labels** — `J` key toggles billboard text labels that always face the camera.
  - **Root trajectory trail** — `T` key shows root path projected on the floor, follows scrubbing.
  - **Screenshot with feedback** — `S` key saves PNG with brief "Saved: filename" overlay.
  - **Per-skeleton visibility** — `1`–`9` keys toggle individual skeletons.
  - **Help panel** — `H` key toggles right-side shortcut reference.
  - **Camera parameter** — `bvhplot.play()` now accepts `camera` parameter (was hardcoded to `"front"`).
  - **Auto camera detection** — initial view uses `get_forw_up_axis()` for correct front-facing orientation.
  - **Adaptive bone sizing** — bone radii scale proportionally to bone length (fingers thin, limbs thick).
  - **Floor grid** — positioned at skeleton's lowest point, denser grid (30x30), visible on all up-axis conventions.

- **Module renamed** from `pybvh.plot` to `pybvh.bvhplot` — avoids confusion with matplotlib. All imports, docs, and tests updated.
- **`bvhplot/CHARTER.md`** — scope document defining what bvhplot owns and what belongs in pybvh-blender.
- **Feature gap analysis** (`docs/feature_gap_analysis.md`) — comparison against 10 industry tools.
- **pybvh-blender implementation guide** (`docs/pybvh_blender_implementation_guide.md`) — complete handoff document for the Blender addon team.

### Fixed

- **Vedo key conflicts** — disabled vedo's default keyboard callbacks (L=lighting, arrows=transparency) that interfered with playback controls.
- **Division by zero** — guarded against `fps <= 0` and `num_frames = 0` in vedo viewer.
- **Dead code cleanup** — removed unused `frame_time` variable, stale helper functions, duplicate imports.
- **Inconsistent naming** — removed underscore-prefixed loop variables, unified state dict initialization.
- **Stale comments** — updated UI layout comments to match current code.

### Changed

- **Vedo minimum version** bumped to `>= 2024.5` (was `>= 2023.5`) in `pyproject.toml` and docstring.
- **Vedo viewer lighting** — switched from glossy specular to flat ambient-only. Eliminates color shifting on frame updates with merged meshes.
- **Ghost/onion-skin mode removed** — delegated to pybvh-blender for proper implementation.
- **UI compact layout** — removed "Controls" title, speed/FPS each on single line, transport buttons centered with `justify='bottom-center'`.

---

## [0.5.0] — 2026-04-03

### Added

- **`pybvh.bvhplot` module** — full visualization rewrite as a multi-file package with pluggable backends.
  - **`bvhplot.frame()`** — static 3D skeleton snapshot (matplotlib). Now accepts `camera` parameter (`"front"`, `"side"`, `"top"`, or `(azim, elev)`).
  - **`bvhplot.render()`** — fast video/GIF/HTML export. OpenCV backend (1000+ fps) with automatic matplotlib fallback.
  - **`bvhplot.play()`** — interactive playback with 3-tier auto-detection: k3d (Jupyter), vedo (desktop), OpenCV inline video or matplotlib fallback. Automatic 30fps subsampling for smooth playback.
  - **`bvhplot.trajectory()`** — 2D top-down root trajectory. Per-skeleton up-axis detection for correct projection when overlaying skeletons with different conventions.
  - **`bvhplot.rest_pose()`** — convenience wrapper for visualizing the T-pose / bind pose.
- **`sync` parameter** on `render()` and `play()` — `"truncate"` (default) or `"pad"` for side-by-side comparison of clips with different lengths.
- **`resolution` parameter** on `play()` — controls the OpenCV notebook fallback resolution.
- **Left-right symmetry axis detection** in `get_forw_up_axis()` — replaced fragile toe-based forward detection with robust left-right joint pair averaging from rest-pose offsets.
- **`build_view_matrix()`** rewritten to use matplotlib's look-at camera math — ensures OpenCV and matplotlib backends produce identical views.
- **Per-skeleton centering** in side-by-side OpenCV renders — each panel centers its skeleton independently.
- **GIF streaming** — `_render_gif()` uses a generator to avoid loading all frames into memory.
- 50 new visualization tests (954 total) including semantic front-view and backend-agreement checks.

### Fixed

- **Skeleton framing** — `compute_unified_limits()` now uses the larger of body span and trajectory extent, preventing walking skeletons from clipping out of frame.
- **View rotation framing** — `ortho_project()` computes view-space extent from rotated bounding box corners, preventing clipping at non-zero elevation angles.
- **Front-view orientation** — camera angles now correctly show the skeleton's chest (toes toward viewer) with right-handed axes for all up-axis conventions.
- **k3d bone connections** — added `indices_type='segment'` to fix spurious bones from triangle-mode index grouping.
- **k3d double plot** — `play_k3d()` returns `None` to prevent Jupyter auto-display of the returned widget.
- **k3d grid bounds** — grid set explicitly from full motion extent so skeleton stays within bounds during animation.
- **OpenCV codec noise** — reordered codec list to try `mp4v` first, eliminating `h264_v4l2m2m` stderr warnings.
- **Parameter validation** — `centered` and `sync` parameters now raise `ValueError` on invalid values instead of silently defaulting.

### Changed

- `pybvh/bvhplot.py` (single file) replaced by `pybvh/bvhplot/` package with `__init__.py`, `_common.py`, `_matplotlib.py`, `_opencv.py`, `_k3d.py`, `_vedo.py`.
- `render()` emits a warning when falling back from OpenCV to matplotlib in auto mode.

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

[Unreleased]: https://github.com/VictorS-67/pybvh/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/VictorS-67/pybvh/compare/v0.5.1...v0.6.0
[0.5.1]: https://github.com/VictorS-67/pybvh/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/VictorS-67/pybvh/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/VictorS-67/pybvh/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/VictorS-67/pybvh/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/VictorS-67/pybvh/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/VictorS-67/pybvh/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/VictorS-67/pybvh/releases/tag/v0.1.0
