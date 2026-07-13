# Changelog

All notable changes to **pybvh** are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.8.0] — 2026-07-06

A single release with two layers of work. First, a breaking module split plus a new layer of theory-neutral motion descriptors: trajectory geometry, jerk/smoothness, gait, and SE(3) rigid-transform math. Second, a whole-codebase review fix-up (~80 verified findings): silent-data-corruption fixes in the parser and transforms, one source of truth for the rotation math, a world-frame FK cache, a radians-first transforms API, short representation tokens, and a leaner `Bvh` surface. Everything ships as one break with a single migration path (0.x policy); the complete old → new ledger lives in [pybvh/API_RENAME.md](pybvh/API_RENAME.md).

### Breaking changes & migration

One row per rename / behavior change relative to v0.7.0. Rows marked † concern names or behaviors that only existed during 0.8.0 development (never in a shipped release) — listed for anyone tracking the development branch.

| Old | New | Migration |
|---|---|---|
| `pybvh.features` (mixed module) | `pybvh.analysis` (descriptors) + `pybvh.features` (feature-array export only) | `pybvh.features.X` → `pybvh.analysis.X` for descriptors; `to_feature_array` / `feature_array_layout` keep their `pybvh.features` path. `Bvh` method names are unchanged. |
| `to_quaternions()` / `from_quaternions()` | `to_quat()` / `from_quat()` | Rename call sites. |
| `"quaternion"` representation string | `"quat"` | Update strings passed to `to_feature_array` / `batch_to_numpy` / `rotations.convert`; `rotations.REPRESENTATION_CHANNELS` keys change too. |
| `bvh.slice_frames(a, b, s)` | *(removed)* | `bvh[a:b:s]`. |
| `bvh.concat(other)` | *(removed)* | `bvh + other` / `bvh += other`. |
| `node_positions(frame_num=-1)` / `joint_positions(frame_num=-1)` | `frame=None` | `None` (the new default) = all frames; `-1` now means the **last** frame (NumPy negative-index semantics). Parameter renamed `frame_num` → `frame`. |
| `rest_pose_coords()` | `rest_pose_positions()` / `rest_pose_angles()` | Positions no longer read motion data (works on 0-frame objects); the `mode='euler'` form is `rest_pose_angles()`, returning just the `(J, 3)` zeros. |
| `hierarchy_info_as_dict()` | `to_hierarchy_dict()` | Rename; pairs with the new `Bvh.from_df(hier, df)`. |
| `index(name, axis=...)` | `index(name, space=...)` | Keyword rename. |
| `write(new_filepath=...)` | `write(filepath=...)` | Keyword rename (positional callers unaffected). |
| `scale((sx, sy, sz))` | `scale(factor)` — scalar only | Per-axis factors are removed (per-axis world scaling of parent-local offsets is not geometrically meaningful under animation). |
| Descriptor methods with integer `joint=` | joint **names** only | `curvature` … `smoothness`, `range_of_motion` take joint names; for index-based access use the functional `pybvh.geometry` / `pybvh.analysis` API. |
| `add_noise(sigma_deg=...)` | `add_noise(sigma=...)` in **radians** | `sigma = np.radians(sigma_deg)`. Negative `sigma` / `sigma_pos` now raise `ValueError`. |
| `add_noise(..., wrap=True)` default | `wrap=False` default | Pass `wrap=True` to restore wrapping. |
| `rotate_vertical(angle_deg)` | `rotate_vertical(angle, degrees=False)` | Multiply by `np.pi / 180`, or pass `degrees=True`. Same for the array-level `rotate_angles_vertical`. |
| `random_rotate_vertical(angle_range=(-180, 180))` | radians default `(-np.pi, np.pi)` with a `degrees=False` flag | Pass `degrees=True` to keep degree ranges. |
| `random_translate_root(range_xyz=)` | `offset_range=` | Keyword rename. |
| `mirror(left_right_mapping=)` | `mirror(lr_mapping=)` | Keyword rename; unknown joint names in an explicit mapping now raise `ValueError`. |
| `transforms.auto_detect_lr_mapping(bvh)` | *(removed)* | Read `bvh.lr_mapping` (same symmetric dict; `None` — not `{}` — when nothing is detected). |
| `drop_frames` re-canonicalized every frame | kept frames preserved bit-for-bit | Output change only — no code change needed. |
| `centered="first"` subtracted the full 3-D first-frame root position | ground-plane-only (the `world_up` coordinate is untouched) | To reproduce the old output, additionally subtract the first frame's root height. |
| `foot_contacts(..., centered=)` | *(removed)* | Detection always runs on world-frame FK; `coords=` accepts world-frame positions or any constant translation thereof. |
| `foot_contacts(..., vel_threshold=)` in units/frame | units/**second** | `new = old / frame_time`. The new default `0.12 × scale` u/s equals the old default exactly at 30 fps. |
| `foot_contacts` foot speed from raw adjacent-frame differences | velocity signal conditioned over a fixed physical span (`vel_smooth_duration=1/30` s) | Labels at ≤ 30 fps are bit-identical (the window is 1 frame). Above 30 fps labels can change — they now match what the same clip gives at 30 fps instead of fragmenting on high-frequency jitter. Pass `vel_smooth_duration=0.0` for raw differencing. |
| `to_feature_array(stencil="forward", pad="none")` dropped the **first** frame | drops the **last** frame | Rows correspond to frames `0..F-2` (previously `1..F-1`). |
| End sites are plain `BvhNode` | `BvhEndSite` class | Construct end sites with `BvhEndSite`; test with `isinstance` / `is_end_site()` (`BvhNode` no longer answers it). |
| `Frame Time:` written as `%.6f` | full precision (`%.10g`) | Written files change on that line only; the read-side snap for truncated foreign files remains. |
| `pybvh.tools.get_premult_mat_rot` / `batch_get_premult_mat_rot` / `rotX/Y/Z` / `batch_rotX/Y/Z` | *(removed)* | `rotations.euler_to_rotmat` is the single Euler→rotmat implementation (e.g. `euler_to_rotmat([a, 0, 0], 'XYZ')` for a single-axis matrix). |
| `pybvh.tools.test_file` / `pybvh.tools.are_permutations` | *(removed)* | Internal helpers; path validation lives on privately under `read_bvh_file`. |
| `pybvh.api_rename_path()` | *(removed)* | `API_RENAME.md` stays a docs file (in the repo and on the docs site). |
| `compute_normalization_stats` / `normalize_array` / `denormalize_array` | moved to **pybvh-ml** | `from pybvh_ml import compute_normalization_stats, normalize_array, denormalize_array`. |
| `bvhplot.render(fps=-1)` / `play(fps=-1)` | `fps=None` (fractional floats accepted) | Replace `fps=-1` with `fps=None` or drop the argument; `fps <= 0` now raises `ValueError`. |
| `bvhplot.frame(bvh, coords_array)` | `frame(bvh, frame=0, *, coords=None)` | Pass pre-computed positions via `coords=`; `frame=-1` is the last frame (no longer an all-frames sentinel). |
| † `pybvh.packing` | `pybvh.features` | Interim dev name for the export half of the split; `pybvh.packing.X` → `pybvh.features.X`. |
| † `pybvh.tools.finite_difference` + signal utilities | `pybvh.signal` | Interim dev location; `pybvh.tools.X` → `pybvh.signal.X`. |
| † `lagged_correlation` | `lagged_covariance` — mean-centered | Rename; values change (a true covariance — the uncentered form let a constant offset dominate). |
| † `geometry.centroid` | `geometry.center_of_mass` | Rename; now shares the `Bvh.center_of_mass` method name. |
| † `pose_distance` returned the *squared* distance | true Euclidean distance | Square the result if a squared kernel is wanted. |
| † `velocity_reductions(speed, fs=1.0)` / `active_duration(..., frame_time=1.0)` | required `fs` (Hz) | Pass `fs = 1 / frame_time` — no implicit time base. |
| † `batch.relative_scale_factor` | `analysis.relative_scale_factor` | Moved next to its sibling `skeleton_size`; also exported as `pybvh.relative_scale_factor`. |
| † `foot_contacts(height_reference="fixed")` | `height_reference="floor"` | Parameter is validated against `{"velocity", "floor"}`. |
| † `walking_pace(bvh, foot_joints=)` | `walking_pace(bvh)` | The parameter was unused. |

### Added

- **`pybvh.geometry`** — new array-pure module of position descriptors (the companion to `pybvh.rotations`): `inter_joint_distance`, `joint_angle`, `segment_axis_angle`, `triangle_area`, `point_to_plane_distance`, `point_to_segment_distance`, `bounding_box`, `bounding_sphere` (Ritter, approximate), `bounding_ellipsoid` (PCA-aligned, radii grown by the worst point's ellipsoidal norm so it genuinely encloses every point), `center_of_mass`, `com_displacement`, `verticality`, `path_length`, `directness`, `curvature`, `torsion`, `movement_phase`, `ground_path`, `pose_distance`, `mean_pose_subtract`. NumPy-only; vectorized over the frame axis; zero-denominator ratios return `nan`; derivative kernels validate `stencil` / `pad` up front.
- **`pybvh.signal`** — new array-pure signal module shared by the analysis and geometry layers: `finite_difference` (the single stencil/pad derivative convention used by the velocity→acceleration→jerk ladder and the geometry derivative kernels, so derivatives composed across the two stay consistent), `temporal_stats` (mean/std/min/max/skewness/excess-kurtosis, manual moments), `box_filter_smooth` (cumsum moving average), `fft_magnitude`, `dominant_frequency`, `ramer_douglas_peucker` (polyline simplification).
- **`pybvh.analysis` motion primitives** — jerk (`node_jerk`, `joint_jerk`); smoothness on a 1-D speed profile (`sparc`, `dimensionless_jerk`, `log_dimensionless_jerk`, `number_of_peaks`, `speed_metric`, `integrated`/`mean`/`rms_squared_jerk`, and a `smoothness(metric=…)` dispatcher; SPARC/DLJ/LDLJ validated against the Balasubramanian reference); signal reductions (`velocity_reductions`, `zero_crossings`, `active_segments`, `active_duration` — time-based kernels take a required `fs`); `kinetic_energy`; gait (`cadence`, `stride_length`, `walking_pace`); `range_of_motion`; and covariance descriptors (`cov3dj`, `lagged_covariance`).
- **`pybvh.analysis.skeleton_size`** (absolute mean root-to-foot scale, the public name for the scale `foot_contacts` uses internally) and **`pybvh.analysis.relative_scale_factor`** (least-squares uniform scale between two poses; exported as `pybvh.relative_scale_factor`) — disambiguated size primitives.
- **`gait_parameters` — one-pass spatiotemporal gait analysis** (`Bvh.gait_parameters()` / `analysis.gait_parameters`). Returns a `GaitParameters` named tuple: `cadence`, `walking_pace`, `stride_length`, `stride_cv` (stride variability), `step_length`, `stance_fraction`, `double_support_fraction`, and left/right `asymmetry` — all measured from foot landings and contact timing, with `nan` for anything underdetermined. Its internal `foot_contacts` call runs with `adaptive=True` — calling `gait_parameters` declares the clip is locomotion, which is exactly the documented precondition for the adaptive thresholds, and the fixed thresholds under-detect stance on retargeted mocap (on the bundled CMU walk they yield a physically impossible `double_support_fraction` of 0). `cadence` / `stride_length` inherit this through the unified definition; pass an explicit `contacts=` array for full control over detection (`foot_contacts` itself keeps `adaptive=False` as its motion-type-agnostic default). Dynamic gait (joint torques, ground-reaction force, mechanical work) is out of scope — it needs a physical model and belongs in a downstream biomechanics layer.
- **`stride_length` is the foot-measured definition** — the distance between successive same-foot landings (the standard gait definition), pooled over feet and averaged. For straight steady walking this matches a root-path-÷-stride-count estimate; it differs (more correctly) on curved or uneven gait.
- **`pybvh.rotations` SE(3) rigid-transform math** — `se3_exp` / `se3_log` (4×4 transform ↔ se(3) twist `[ω, v]`, rotation-first, V-left-Jacobian-coupled), `screw_interpolate` (SE(3) SLERP analogue, broadcastable `t`), `se3_inverse` (closed-form rigid inverse `[Rᵀ, −Rᵀd]`), `relative_transform` (segment→segment bridge), and `rotation_geodesic_distance`. Quaternion-based rotation log keeps the SE(3) log/geodesic accurate at θ≈π. The module docstring now reads "rotation & rigid-transform math."
- **`pybvh.rotations.quat_multiply(q1, q2)`** — public Hamilton product, broadcasting over `(..., 4)`. The composition primitive downstream libraries kept reimplementing privately.
- **`Bvh` motion-descriptor methods** — thin wrappers for the primitives that are Bvh-bound or top-frequency single-joint queries: `curvature`, `torsion`, `path_length`, `directness`, `ground_path`, `inter_joint_distance`, `joint_angle`, `triangle_area`, `segment_axis_angle`, `bounding_box`, `bounding_sphere`, `bounding_ellipsoid`, `movement_phase`, `center_of_mass`, `com_displacement`, `verticality`, `node_jerk`, `joint_jerk`, `smoothness`, `kinetic_energy`, `cadence`, `stride_length`, `walking_pace`, `gait_parameters`, `range_of_motion`, `skeleton_size`, `velocity_reductions`. Relational/trajectory methods resolve names in node space (end sites first-class); `range_of_motion` resolves in joint space. All descriptor methods accept pre-computed positions via `coords=`.
- **`Bvh.from_rotmat(root_pos, joint_rotmats)`** — the missing importer for the pivot representation, same overload pattern as `from_6d` / `from_quat` / `from_axisangle`.
- **`Bvh.from_file(path)` and `Bvh.from_df(hier, df)` classmethods** — discoverable constructors delegating to `read_bvh_file` / `df_to_bvh`, completing the to/from pairs (`write`, `to_df_dict` + `to_hierarchy_dict`).
- **`bvh.world_up = 'auto'` (or `None`) clears a manual override** — re-enables auto-detection without rebuilding the object.
- **`velocity_reductions` gains `peak_acceleration`** — the mirror of `peak_deceleration` (the largest instantaneous speed *increase*, in units/second). `VelocityReductions` is `(peak, mean, peak_to_mean, peak_acceleration, peak_deceleration)`.
- **`kinetic_energy` accepts `masses` as a `{joint_name: mass}` mapping** — order-independent and validated for exact coverage (clear error on a missing/unknown joint, or a wrong-length array), instead of relying on a positionally-ordered `(J,)` vector. The array form still works.
- **`foot_contacts` strengthening** — the contact detector (the foundation under all gait metrics) gains: **hysteresis** (a Schmitt-trigger band, default ±25%, that suppresses borderline label flicker — it stabilized a borderline foot whose onset count previously swung 1↔3 under small threshold changes; `hysteresis=0` restores single-threshold behavior), **opt-in `adaptive` thresholds** (per-foot Otsu bimodal split with a fallback for non-walking feet — cut under-detection markedly on real data), and, under `return_info`, a per-foot **`confidence`** plus unsupervised diagnostics **`foot_skate`**, **`airborne_fraction`**, and **`height_at_contact`** that flag detection-quality problems with no ground truth needed.
- **`foot_contacts` frame-rate-robust velocity signal (`vel_smooth_duration`)** — the per-second threshold migration left the *signal* frame-rate dependent: adjacent-frame differencing at 120 fps picks up high-frequency jitter that 30 fps differencing averages out, fragmenting genuine stance phases. Displacement vectors are now box-averaged over `round(vel_smooth_duration / frame_time)` frames before taking the norm (default 1/30 s — the speed estimator always spans ~1/30 s regardless of capture rate; norm-of-mean, so oscillatory jitter cancels vectorially). A 1-frame no-op at ≤ 30 fps; at 120 fps (window 4) it healed a jitter-split stance on CMU walking data and repaired the downstream gait metrics (`stride_cv` 0.48 → 0.03, left/right `asymmetry` 0.65 → 0.02). `vel_smooth_duration=0.0` disables; the span and effective window are reported under `return_info` (`vel_smooth_duration`, `vel_smooth_frames`).
- **`foot_contacts` velocity-informed height threshold** (`height_reference="velocity"`, the default for `method="combined"`) — calibrates each foot's height threshold to where it actually sits during stance, fixing severe under-detection on **retargeted mocap** where the foot hovers above the floor (recovered per-foot stance from ~17–35% toward the ~60% physiological norm; airborne dropped from ~50% to single digits). It reduces *exactly* to the fixed threshold when the foot reaches the floor (clean rigs are unchanged), and a swing-presence guard keeps a held-airborne foot rejected. `height_reference="floor"` selects the floor-anchored behavior; the parameter is validated (`{"velocity", "floor"}`).
- **`cadence` / `stride_length` (and the `Bvh` wrappers) accept a pre-computed `contacts=` array** — like `gait_parameters`, so one `foot_contacts` run can feed the whole gait family.
- **`Bvh.floor_height`** — a lazily-cached canonical ground-plane height (raw world coordinate along `world_up`), unifying the floor estimate previously re-derived ad hoc. Computed from auto-detected feet (all nodes for footless rigs); invalidated whenever the motion is reassigned.
- **`BvhEndSite` class** — end sites are now a dedicated node class instead of plain `BvhNode`. End-site identity everywhere goes through `isinstance` / `is_end_site()`; generated display names like `'EndSiteHips'` are cosmetic only and carry no semantics (in dict hierarchies for `df_to_bvh`, an entry with neither `'children'` nor `'rot_channels'` is an end site — the `'EndSite'` name prefix no longer matters). ⚠️ `BvhNode` is now the abstract-ish base and no longer answers `is_end_site()`; construct end sites with `BvhEndSite`.
- **Rotation-first BVH roots load correctly** — `CHANNELS` entries are classified by token suffix (`…position` / `…rotation`), so 6-channel roots that declare rotations before positions parse with correct semantics (previously the first three motion columns were silently taken as root position, corrupting the clip). Layouts pybvh doesn't model — e.g. position channels on non-root joints — now raise a clear `ValueError` instead.
- **Documentation overhaul (discoverability-first)** — a library-wide **Feature Gallery** (`gallery/feature_gallery.ipynb`, ~57 figures: one picture and one call per visual capability, from `mirror`/`retarget`/centering modes through every 0.8.0 descriptor) is published as a docs page generated at deploy time by `scripts/export_gallery.py`: a click-to-jump thumbnail grid on top, every figure extracted as a cacheable lazy-loaded file, and stable-named copies (`centered-modes.png`, …) that six guide pages embed inline at the point of explanation (the exporter fails loudly if a refactor breaks one of those matches). The API reference gains a task→call **capability map** landing page, and the `Bvh`/`analysis`/`rotations` pages are restructured into themed groups. Three CI guards keep it honest: `tests/test_docs_api_coverage.py` (every public member appears in the docs, two-way), `tests/test_gallery_notebook.py` (jupytext pair in sync, committed outputs fresh and clean), and nbmake execution of the gallery notebook in `tutorials.yml` (GIF cells tagged `slow-on-pr`). README slimmed to a linked showcase with a hero GIF; docs home, guide cross-links, nav tabs, and a `docs` extras group (`pip install -e ".[docs]"`, includes Pillow for thumbnails) round it out. Docs deploys now run `mkdocs build --strict`, so a broken internal link fails CI. Gallery sources live in `gallery/` (with `gallery_plots.py`); the rendered GIF byproducts are no longer git-tracked (the committed copies are the notebook outputs and `docs/assets/`).

### Changed

- **Module layout.** `pybvh/features.py` (mixed) split into `pybvh/analysis.py` (motion descriptors) + `pybvh/features.py` (feature-array export only). Function behavior is byte-identical; only import paths change (see the migration table). The `Bvh` methods (`bvh.joint_velocities()`, `bvh.to_feature_array()`, …) are unchanged.
- ⚠️ **`to_quaternions` / `from_quaternions` → `to_quat` / `from_quat`; representation strings are short tokens.** The representation vocabulary is `{"euler", "quat", "6d", "axisangle", "rotmat"}` everywhere — `to_feature_array`, `feature_array_layout`, `batch_to_numpy`, `rotations.convert`, `rotations.REPRESENTATION_CHANNELS`, and their error messages.
- ⚠️ **`node_positions` / `joint_positions` take `frame: int | None = None`.** `None` (the new default) means all frames; integers follow NumPy semantics, so `frame=-1` is the **last** frame (previously the all-frames sentinel). The parameter is renamed from `frame_num` for consistency with `forward_at` / `plot_frame`.
- ⚠️ **`rest_pose_coords` → `rest_pose_positions()` + `rest_pose_angles()`.** Rest-pose positions are computed from zero angles directly (no motion-data dependence — works on 0-frame objects); the odd `mode='euler'` tuple form is now `rest_pose_angles()` returning just the `(J, 3)` zeros.
- ⚠️ **`hierarchy_info_as_dict` → `to_hierarchy_dict`** — the returned dict is built from copies directly (no trailing deepcopy) and pairs with the new `Bvh.from_df`.
- ⚠️ **`index(name, axis=...)` → `index(name, space=...)`** and **`write(new_filepath)` → `write(filepath)`** — parameter renames for call-site readability.
- ⚠️ **`scale` is scalar-only.** Per-axis `(sx, sy, sz)` factors are removed — world-axis scale factors applied to parent-local offsets are not geometrically meaningful once the skeleton animates. Non-scalar input raises `TypeError`.
- ⚠️ **Convenience descriptor methods resolve joints by name only.** `curvature` … `smoothness` and `range_of_motion` reject integer joint arguments (they were ambiguous between joint-space and node-space indices — a silent off-by-N trap on skeletons with end sites); index-based access stays available through the functional `pybvh.geometry` / `pybvh.analysis` API.
- ⚠️ **`centered="first"` is now ground-plane-only centering.** `node_positions` / `joint_positions` / `frames_to_node_positions` / `to_feature_array` subtract the first frame's root position in the two horizontal axes only — the `world_up` coordinate is untouched, so heights stay in world units (floor estimation and height-based features keep working on first-centered coords). Previously the full 3-D root position was subtracted. Migration: to reproduce the old behavior, additionally subtract the first frame's root height. `frames_to_node_positions` gains an `up: str = '+y'` parameter for this; `Bvh.node_positions` passes `bvh.world_up` automatically.
- **`Bvh` constructor cross-validation.** Passing only one of `root_pos` / `joint_angles`, arrays that disagree on frame count, or a `joint_angles` joint axis that doesn't match the skeleton's non-end-site joint count now raises `ValueError` instead of constructing a silently inconsistent object.
- **`Bvh.__eq__` compares the full hierarchy** (offsets, parent structure, end sites — via `matches_hierarchy`/`matches_channels`) in addition to frame timing and motion arrays. Previously two Bvhs with identical motion but different rest offsets compared equal.
- **World-frame FK is cached on the `Bvh` object.** `node_positions()` computes forward kinematics once per motion state and serves subsequent calls (any `centered` mode, single-frame or all-frames) from the cache; the cache is invalidated whenever motion data changes. Frame slicing, concatenation, and `resample` no longer deep-copy motion arrays they immediately overwrite, making `for frame in bvh:` linear instead of quadratic.
- **`frames_to_node_positions` keys skeleton topology by node identity** instead of node name, so skeletons with duplicate joint names no longer produce silently corrupted FK; it also raises `ValueError` when `root_pos` and `joint_angles` disagree on frame count.
- **`rotmat_to_axisangle` routes through the quaternion** — numerically stable near θ = π (the trace/arccos route lost ~1e-4 there) and vectorized (no per-element Python loop). It is also the base of `se3_log` / `rotation_geodesic_distance`.
- **Euler-order validation rejects degenerate orders** — orders with equal adjacent axes (e.g. `'XXY'`) now raise `ValueError` in all `rotations` entry points instead of silently extracting garbage angles.
- **`quat_to_rotmat` raises `ValueError` on zero-norm quaternions** instead of dividing by zero.
- ⚠️ **`Frame Time` is written at full precision** (`%.10g` instead of `%.6f`), so non-integer rates like 23.976 fps survive write→read round-trips losslessly. Written files change on the `Frame Time:` line only (e.g. `0.033333` → `0.03333333333`); the read-side snap-to-`1/N` salvage for 6-digit-truncated foreign files remains, is documented on the shared `_snap_frame_time` helper, and is also used by `df_to_bvh` (whose duplicate snapping code is gone).
- **`warn_on_world_up_disagreement` is a real flag** — `Bvh(...)` gains a `warn_on_disagreement` parameter that threads through world-up detection; `read_bvh_file` forwards it instead of suppressing the warning by message-text filtering.
- **BVH motion block parsed via `np.loadtxt`** — faster than the per-line Python loop, and a file whose data-line count disagrees with its `Frames:` declaration (either direction) now raises a clear `ValueError` (extra rows previously crashed with a bare `IndexError`). The hierarchy parser itself is now a brace-stack parser with a single token-driven node-block reader, so blank lines and line-order variations inside node blocks don't break parsing.
- **`df_to_bvh` hardening** — joint names containing underscores now work everywhere the `name_ax_pos/rot` column convention is parsed (split from the right); channel inference matches joint names exactly (previously `Hip` also matched `LHip` columns via substring search); the column-name pattern requires uppercase `[XYZ]` axes (matching `to_df_dict` output); frame time is derived from elapsed time `(t[-1] - t[0]) / (F - 1)` with a clear error for DataFrames of fewer than 2 rows; the DataFrame's degrees convention is documented in the docstring.
- **`BvhNode` family constructor hygiene** — default `offset` / `rot_channels` / `pos_channels` use `None` sentinels instead of shared mutable lists (all default-constructed joints previously shared one channel list, and frozen channel lists could be mutated through the caller's reference); `offset` is validated live to shape `(3,)`.
- ⚠️ **`foot_contacts` velocity threshold is in units/second.** The foot-speed signal and `vel_threshold` are now physical (units/second) instead of per-frame; the default is `0.12 × skeleton_scale` u/s, which equals the old `0.004 × skeleton_scale` per frame **exactly at 30 fps** (identical labels for 30 fps clips). Migration for an explicit threshold: `new = old / frame_time`. `frame_time == 0` now raises `ValueError` when the velocity signal (or a nonzero duration filter) needs a time base, instead of silently reinterpreting durations as frame counts.
- ⚠️ **`foot_contacts` drops `centered`** — the detector always works in world-frame FK internally (served by the FK cache). `coords=` remains and is documented as world-frame positions or any constant translation thereof (`to_feature_array(centered="skeleton")` no longer feeds root-relative coords into contact detection — labels there are now the correct world-frame ones).
- **`foot_contacts` floor is estimated from the coords in use.** `floor="auto"` always derives the floor from the positions the detector is actually running on; the cached `Bvh.floor_height` only fills/serves on the canonical world-coords + auto-detected-feet path (previously the canonical cached floor leaked into any auto-feet call, even with explicit `coords=`, and the detector ran FK + foot detection twice).
- ⚠️ **Gait scalars unified on `gait_parameters`.** `cadence` and `stride_length` are one-line projections of `gait_parameters` and, like it, accept a pre-computed `contacts=` array (also on the `Bvh` wrappers) so one `foot_contacts` run can feed the whole family; `gait_parameters` forwards its own FK coords into its internal `foot_contacts` call. `walking_pace` loses its unused `foot_joints` parameter and its docstring no longer claims the `stride_length × cadence / 2` identity holds "by construction" (it is only an approximation for straight steady gait).
- ⚠️ **`lagged_correlation` → `lagged_covariance`, mean-centered.** The kernel now subtracts the temporal mean before the lagged product (a true covariance — the old uncentered form let a constant offset dominate) and is named for what it computes. `cov3dj` is documented as the population covariance (divides by `F`).
- ⚠️ **`geometry.pose_distance` returns the true Euclidean distance** (`‖X₁ − X₂‖`, previously the *squared* distance). Migration: square the result if a squared kernel is wanted.
- ⚠️ **`geometry.centroid` → `geometry.center_of_mass`** — the functional kernel now shares the name of the `Bvh.center_of_mass` method that wraps it.
- ⚠️ **Array-pure time bases are explicit.** `velocity_reductions(speed, fs)` and `active_duration(speed, threshold, fs)` require the sampling rate (`fs` in Hz; `active_duration`'s old `frame_time=1.0` default silently returned a sample count). Migration: pass `fs=1/frame_time`.
- ⚠️ **`relative_scale_factor` moved from `pybvh.batch` to `pybvh.analysis`** (next to its sibling `skeleton_size`) and is exported at package level (`pybvh.relative_scale_factor`).
- **`skeleton_size` raises on unknown explicit joint names** instead of silently ignoring them; the `1.0` degenerate-rig fallback now applies only when *auto-detection* finds no feet.
- **`joint_velocities` / `joint_accelerations` / `joint_jerk` validate `coords`** — a joint-shaped `(F, J, 3)` input now raises a clear `ValueError` naming the required node shape `(F, N, 3)` (`node_positions()` output) instead of silently mis-indexing the node-axis subset.
- **`geometry` derivative kernels validate `stencil`/`pad` up front** (`curvature`, `torsion`, `movement_phase`) — a typo'd `pad` used to silently mean `"edge"`.
- ⚠️ **Transforms take radians.** `add_noise(sigma_deg)` → `add_noise(sigma)` (radians); `rotate_vertical(angle_deg)` → `rotate_vertical(angle, degrees=False)`; `random_rotate_vertical(angle_range=(-180, 180))` → radians default `(-π, π)` with a `degrees=False` flag; the array-level `rotate_angles_vertical` likewise takes `angle` in radians with a `degrees=` flag. This unifies the transforms module with `Bvh.joint_angles` (radians since v0.7.0) and the `degrees=` opt-in convention of `rotations`/`analysis`/`geometry`. Migration: multiply old degree arguments by `np.pi / 180`, or pass `degrees=True`. The `Bvh` method wrappers change identically.
- ⚠️ **`add_noise` no longer wraps by default.** The `wrap` default flips `True` → `False` — wrapping silently corrupts channels that legitimately hold values outside `[-π, π]` (accumulated rotations spanning multiple turns). Pass `wrap=True` for the old behavior. Negative `sigma` / `sigma_pos` now raise `ValueError` instead of silently no-opping.
- ⚠️ **`drop_frames` preserves kept frames exactly.** Only the dropped frames are re-synthesized (linear root interpolation + one vectorized quaternion-SLERP call); kept frames' Euler angles and root positions are bit-identical to the input. Previously *every* frame round-tripped through quaternions and back, silently re-canonicalizing the kept frames' angle values.
- ⚠️ **`mirror` parameter `left_right_mapping` → `lr_mapping`** — matches the `Bvh.lr_mapping` property and the `lr_mapping=` loader parameter. An explicitly passed mapping that names unknown joints now raises `ValueError` listing them (typos previously produced silently half-mirrored output), and L/R joints with mismatched end-site counts raise instead of silently mispairing.
- ⚠️ **`random_translate_root(range_xyz=)` → `offset_range=`** — matches the `angle_range` / `factor_range` sibling parameter names.
- **Axis-string validation in transforms.** `rotate_vertical(up_axis='y')` now raises a clear `ValueError` demanding a signed axis (previously an `IndexError`); `mirror(lateral_axis=)` accepts `'x'` or `'+x'`/`'-x'` (the sign is irrelevant for mirroring) and rejects anything else.
- **`Bvh.rest_up` returns `None` for degenerate skeletons** (single-node rigs, all-zero offsets, or no motion data) instead of an arbitrary axis or a crash.
- ⚠️ **`to_feature_array` forward-stencil trim drops the *last* frame.** With `include_velocities=True, stencil="forward", pad="none"`, the root-position / rotation / foot-contact blocks now drop the last frame instead of the first — a forward difference labels frame `i` with `(x[i+1] − x[i]) / dt`, so the frame without a defined velocity is the final one, and every output row now describes a single frame consistently across all blocks. Migration: rows correspond to frames `0..F-2` (previously `1..F-1`).
- **`batch_to_numpy` extracts each clip via `features.to_feature_array`** — one implementation owns the flat `(F, D)` layout, the valid-representation set, and its error message (previously duplicated in `pybvh.batch`).
- **`Bvh.__repr__` / `__str__` rebuilt from structured fields** (`joint_names`, `joint_count`, `fps`) — robust to joint names containing spaces or quotes, which broke the previous string munging.
- ⚠️ **`bvhplot.render(fps=)` / `bvhplot.play(fps=)` is `float | None`.** `None` (the new default, replacing the `-1` sentinel) means the BVH's native frame rate; fractional rates like `119.88` are accepted exactly, and `fps <= 0` raises `ValueError` (previously `render(fps=0)` crashed with `ZeroDivisionError`). Migration: replace `fps=-1` with `fps=None`, or just drop the argument. The matplotlib writer saves at the exact requested rate instead of quantizing it through an integer frame interval.
- **`bvhplot.render(backend=)` is validated** against `{"auto", "opencv", "matplotlib"}`, mirroring `play()` — a typo'd backend name now raises `ValueError` instead of silently falling back to matplotlib, and a forced `backend="opencv"` raises `ValueError` on file extensions OpenCV cannot write (`.gif` / `.html` / `.webp` / `.apng`) instead of failing deep inside the codec.
- **`bvhplot.frame(bvh, frame=0, *, coords=None)`** — pre-computed positions move to an explicit `coords=` keyword instead of overloading the `frame` argument with an int-or-array union; `play()` is documented (truthfully, for every backend) to return `None`.

### Removed

- ⚠️ **Public `slice_frames` / `concat` methods** — `bvh[a:b:c]` and `bvh + other` / `bvh += other` are the only spellings (the implementations live on privately under `__getitem__` / `__add__`).
- ⚠️ **`transforms.auto_detect_lr_mapping`** — it was a thin back-compat shim over `Bvh.lr_mapping`. Read `bvh.lr_mapping` instead (same symmetric dict, but `None` — not `{}` — when nothing is detected). `auto_detect_lr_pairs()` remains for the index-pair form.
- ⚠️ **`pybvh.tools` rotation-matrix helpers** — `rotX/Y/Z`, `batch_rotX/Y/Z`, `get_premult_mat_rot`, `batch_get_premult_mat_rot`. `rotations.euler_to_rotmat` (which also powers forward kinematics) is the single Euler→rotmat implementation.
- **`pybvh.tools.test_file` and `pybvh.tools.are_permutations`** — internal helpers removed from the module surface. Path validation lives on as the private `tools._validate_bvh_path` used by `read_bvh_file`; the permutation check is inlined at its single call site.
- **`pybvh.api_rename_path()`** — the v0.6.0 helper that returned the on-disk path of the bundled `API_RENAME.md`. The rename ledger stays in the repo and on the docs site; the helper added no value over reading it there.
- ⚠️ **`compute_normalization_stats` / `normalize_array` / `denormalize_array` moved to pybvh-ml.** Dataset-level z-score normalization is an ML-pipeline concern, and pybvh-ml already privately reimplemented the array-level core — the trio moves there rather than being duplicated across the two libraries. Behavior is unchanged (same stats dict with `mean` / `std` / `constant_channels`, same z-score math). Migration: `from pybvh_ml import compute_normalization_stats, normalize_array, denormalize_array`.

### Fixed

- **`foot_contacts` no longer extrapolates a contact across an open clip boundary.** A foot cut off mid-toe-off (clip ends while it is lifting) was held "planted" all the way to the last frame, inflating `stance_fraction` and `double_support_fraction` near either end of *any* clip. Two cleanup layers were speculatively extending across the edge and both are now boundary-aware: the hysteresis band no longer fills a weak run past its raw-threshold support at an open edge, and the morphological open/close no longer length-filters runs that touch frame 0 or the last frame (a truncated run's observed length is only a lower bound, so it can't be judged "short"). Interior behavior is unchanged. On a real walk cut mid-stride this removed a spurious 21-frame terminal double-support block; the remaining double-support is the genuine weight-transfer phase.
- **`velocity_reductions` directional rates clamp at 0** — `peak_acceleration` and `peak_deceleration` are both `>= 0` now (`0` when the speed never rises / never falls), matching the documented contract; previously `peak_deceleration` could come back negative for a monotonically accelerating profile.
- **`dimensionless_jerk` returns `nan` for an all-zero speed profile** (zero peak → undefined normalization), matching the degenerate-input convention of `sparc` and `speed_metric`, instead of dividing by zero.
- **`extract_joints` preserves `source_path` and a user-set `lr_mapping`** (filtered to the kept joints) — the docstring already promised the former.
- **`resample` validates `target_fps > 0` and always sets the new `frame_time`** — previously a clip too short to interpolate (< 2 frames) returned with the *old* timing.
- **`bvhplot.render()` routes by output extension under `backend="auto"`** — `.gif` / `.webp` / `.apng` / `.html` outputs always go to the matplotlib/pillow writer even when OpenCV is installed; previously cv2 users got a misleading codec `RuntimeError` on these documented formats.

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

[0.8.0]: https://github.com/VictorS-67/pybvh/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/VictorS-67/pybvh/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/VictorS-67/pybvh/compare/v0.5.1...v0.6.0
[0.5.1]: https://github.com/VictorS-67/pybvh/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/VictorS-67/pybvh/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/VictorS-67/pybvh/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/VictorS-67/pybvh/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/VictorS-67/pybvh/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/VictorS-67/pybvh/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/VictorS-67/pybvh/releases/tag/v0.1.0
