# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: pybvh
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Batch Processing

# %% [markdown]
# Real-world motion capture datasets contain hundreds or thousands of BVH files. `pybvh.batch` provides utilities for loading directories, validating compatibility, converting to NumPy, and computing dataset-level normalization statistics.
#
# Before you can batch-process a dataset, three classical problems need to be addressed:
#
# 1. **Heterogeneous sources.** Different files may use different up-axis conventions (`+y` vs `+z`), different frame rates, and different skeleton topologies. These have to be **harmonized** before batching.
# 2. **Shape uniformity.** ML models consume tensors of fixed shape. BVH clips have variable length; we either pad to a common length or keep them as a list.
# 3. **Feature scaling.** Joint rotations, root positions, and velocities have very different numerical scales. Models trained on raw values are gradient-dominated by the largest-scale channels. **Per-channel normalization** fixes this.
#
# This tutorial walks through all three in order, then puts them together into a complete dataset-preparation pipeline.

# %%
import numpy as np
np.set_printoptions(precision=4, suppress=True)

import pybvh
from pybvh import batch
from pathlib import Path

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "tutorials" else Path.cwd()
bvh_folder = REPO_ROOT / "bvh_data"
output_folder = Path('./output')
output_folder.mkdir(exist_ok=True)

# %% [markdown]
# # Loading a directory

# %% [markdown]
# `read_bvh_directory()` loads every matching BVH file in a directory and returns a `list[Bvh]`. Files are sorted alphabetically by default (`sort=True`), which matters for reproducibility — shuffling should happen downstream, after deterministic loading.

# %%
bvh_list = batch.read_bvh_directory(bvh_folder)

print(f'Loaded {len(bvh_list)} BVH files:')
for i, b in enumerate(bvh_list):
    print(f'  [{i}] {b.joint_count:>2d} joints, {b.frame_count:>3d} frames, '
          f'fps={1/b.frame_time:>5.0f}, world_up={b.world_up}')

# %% [markdown]
# Notice that the loaded files are heterogeneous: different joint counts, frame rates, and up-axis conventions. We'll unify them below.

# %% [markdown]
# ## Filtering with patterns
#
# The `pattern` parameter accepts glob patterns. Common uses: restrict to a sub-category (`'*walk*.bvh'`), exclude metadata files (`'subject*.bvh'`), or handle nested structure (`'*/*.bvh'`).

# %%
# Only files starting with 'bvh_test' (excludes bvh_example.bvh and standard_skeleton.bvh)
test_files = batch.read_bvh_directory(bvh_folder, pattern='bvh_test*.bvh')
print(f'Loaded {len(test_files)} of {len(bvh_list)} files matching "bvh_test*.bvh"')

# %% [markdown]
# ## Parallel loading
#
# For large datasets, `parallel=True` uses a thread pool to overlap file I/O. Useful when loading thousands of files; negligible benefit for small fixtures. `max_workers` caps the pool size (defaults to Python's `ThreadPoolExecutor` default, typically CPU count).

# %%
bvh_list_parallel = batch.read_bvh_directory(bvh_folder, parallel=True, max_workers=4)
print(f'Loaded {len(bvh_list_parallel)} files in parallel')

# %% [markdown]
# ## Robustness: skipping corrupt files
#
# Real datasets sometimes contain malformed or corrupt BVH files — a half-finished export, a stray non-BVH file with a `.bvh` extension, etc. By default `read_bvh_directory` propagates the first failure and aborts the load. Pass `skip_errors=True` to instead emit a `UserWarning` per bad file and return only the successes:
#
# ```python
# clips = batch.read_bvh_directory('big_dataset/', parallel=True, skip_errors=True)
# ```
#
# Silent skipping is opt-in precisely because it can hide real problems (you silently lose data). Only enable it when occasional corrupt files are expected and logging the skip is enough.

# %% [markdown]
# ## Dataset-wide conventions at load time
#
# Two parameters apply a convention uniformly across the whole dataset at load time:
#
# - **`world_up='+z'`** — force every file's `world_up` metadata to the given value, skipping auto-detection. Use when you **know** all your files follow one convention but pybvh's heuristic mis-identifies some (the ~5% edge case). This only sets the metadata; it does **not** rotate the data — for genuine rotation, see *Up-axis unification* below.
# - **`lr_mapping={...}`** — apply one explicit left/right joint pair mapping to every file. Useful for datasets with non-standard naming conventions the auto-detect heuristic can't parse.

# %%
# Force every loaded file to be interpreted as +z up
bvh_list_zup = batch.read_bvh_directory(bvh_folder, world_up='+z')
print('After world_up="+z" at load:')
for i, b in enumerate(bvh_list_zup):
    print(f'  [{i}] world_up={b.world_up}')

# %% [markdown]
# # Harmonizing heterogeneous datasets

# %% [markdown]
# When clips come from different sources, three mismatches commonly block batching: **skeleton topology**, **frame rate**, and **up-axis convention**. Each has a corresponding pybvh tool. Apply them in a loop over your clips before `batch_to_numpy`.

# %% [markdown]
# ## Skeleton unification
#
# Two sub-problems:
#
# - **Different bone proportions, same topology**: `bvh.retarget(reference)` copies bone offsets from a reference skeleton while preserving joint angles. Use when clips share the same joint names and hierarchy but represent differently-sized performers.
# - **Different joint counts or names**: `bvh.extract_joints([common_names])` keeps a shared subset, collapsing removed joints' offsets into the nearest kept descendant. See [Tutorial 2](2.Spatial_coordinates.ipynb) for details.
#
# If neither tool applies — e.g., finger-rich files vs. body-only files with no sensible common subset — the practical answer is to **drop the incompatible files**.

# %%
# Retarget every compatible clip to a reference skeleton's bone proportions
reference = pybvh.read_bvh_file(bvh_folder / 'standard_skeleton.bvh')
clip_a = pybvh.read_bvh_file(bvh_folder / 'bvh_test1.bvh')
clip_b = pybvh.read_bvh_file(bvh_folder / 'bvh_example.bvh')

print('Before retargeting — Spine offset varies:')
print(f'  reference: {reference.nodes[1].offset}')
print(f'  clip_a:    {clip_a.nodes[1].offset}')
print(f'  clip_b:    {clip_b.nodes[1].offset}')

retargeted = [c.retarget(reference) for c in [clip_a, clip_b]]

print('\nAfter retargeting — all match the reference:')
for i, c in enumerate(retargeted):
    print(f'  clip_{chr(ord("a")+i)}:    {c.nodes[1].offset}')

# %% [markdown]
# ## Frame-rate unification
#
# Clips recorded at different fps can't be batched directly — their feature arrays have different time resolutions. `bvh.resample(target_fps)` uses quaternion SLERP to produce a clip at the target frame rate while preserving the underlying motion.
#
# (This is also the primitive `transforms.perturb_speed` builds on, covered in [Tutorial 5](5.Transforms.ipynb).)

# %%
bvh_30 = pybvh.read_bvh_file(bvh_folder / 'bvh_test1.bvh')   # 30 fps
bvh_120 = pybvh.read_bvh_file(bvh_folder / 'bvh_test2.bvh')  # 120 fps

print('Before unification:')
for name, b in [('bvh_30', bvh_30), ('bvh_120', bvh_120)]:
    print(f'  {name:7s}  {b.frame_count:>3d} frames @ {1/b.frame_time:>4.0f} fps')

target_fps = 30
unified = [b if abs(1/b.frame_time - target_fps) < 0.1 else b.resample(target_fps)
           for b in [bvh_30, bvh_120]]

print(f'\nAfter resampling to {target_fps} fps:')
for i, c in enumerate(unified):
    print(f'  clip_{i}:   {c.frame_count:>3d} frames @ {1/c.frame_time:>4.0f} fps')

# %% [markdown]
# ## Up-axis unification
#
# When files come from tools with different up-axis conventions (Maya commonly exports `+y`; Blender commonly `+z`), their world coordinates live in different frames. Visualizing them side-by-side or computing trajectories produces nonsense until they're rotated into a common axis.
#
# `bvh.reorient_world_up(new_up)` rotates the entire scene so the world vertical axis changes, without altering how the character looks (covered in more depth in [Tutorial 5](5.Transforms.ipynb)).

# %%
bvh_yup = pybvh.read_bvh_file(bvh_folder / 'bvh_test2.bvh')  # +y
bvh_zup = pybvh.read_bvh_file(bvh_folder / 'bvh_test1.bvh')  # +z

print('Before unification:')
for name, b in [('bvh_yup', bvh_yup), ('bvh_zup', bvh_zup)]:
    print(f'  {name:7s}  world_up={b.world_up}')

target_up = '+z'
unified = [b if b.world_up == target_up else b.reorient_world_up(target_up)
           for b in [bvh_yup, bvh_zup]]

print(f'\nAfter reorienting to {target_up}:')
for i, c in enumerate(unified):
    print(f'  clip_{i}:   world_up={c.world_up}')

# %% [markdown]
# ## Harmonizing everything at once
#
# The three subsections above applied `retarget`, `resample`, and `reorient_world_up` one at a time. In practice you'll apply them together to every clip in a dataset, after first checking topology compatibility. `batch.harmonize()` composes all of that behind one call:
#
# - Topology check vs a reference skeleton (drop or raise on mismatch)
# - Retarget bone proportions to match the reference
# - Resample to a target fps (with a small tolerance to avoid no-op copies)
# - Reorient into a target up axis
#
# Any of `reference`, `target_fps`, `target_up` may be `None` to skip that stage. Clips that don't match the reference's topology are dropped with a `UserWarning` by default (or raise with `on_incompatible='raise'`). For workflows `batch.harmonize` doesn't fit — e.g. using `extract_joints` to reduce clips to a common joint subset instead of dropping mismatched files — fall back on the three primitives directly.

# %%
import warnings

reference = pybvh.read_bvh_file(bvh_folder / 'bvh_example.bvh')
raw = [pybvh.read_bvh_file(bvh_folder / name) for name in
       ['bvh_example.bvh', 'bvh_test1.bvh', 'bvh_test2.bvh']]

with warnings.catch_warnings():
    warnings.simplefilter('ignore')  # quiet per-drop warnings for a tidy cell output
    harmonized = batch.harmonize(
        raw,
        reference=reference,
        target_fps=30,
        target_world_up='+z',
        verbose=False,
    )

print(f'In: {len(raw)}  Out: {len(harmonized)} '
      f'(bvh_test2 dropped — different topology)')
for i, c in enumerate(harmonized):
    print(f'  clip {i}: {c.joint_count} joints, {c.frame_count} frames '
          f'@ {1/c.frame_time:.0f} fps, up={c.world_up}')

# %% [markdown]
# # Batch conversion to NumPy

# %% [markdown]
# Once clips are harmonized, `batch.batch_to_numpy()` converts a list of `Bvh` objects into NumPy arrays with a flat per-frame feature vector. It validates skeleton compatibility first — mismatched clips raise `ValueError` before any conversion happens.
#
# Build a small demo batch from one clip sliced at different ranges so all share the same skeleton:

# %%
base = pybvh.read_bvh_file(bvh_folder / 'bvh_test1.bvh')
clips = [base, base.slice_frames(0, 40), base.slice_frames(20, 75)]

print(f'Clip frame counts: {[c.frame_count for c in clips]}')

# %% [markdown]
# ## Variable-length vs. padded output
#
# `pad=False` (default) returns one 2D array per clip — good when clip length is a property of the data (e.g., variable-length sequence models). `pad=True` zero-pads to the longest clip and returns a single 3D tensor — good for fixed-length batching.

# %%
arrays = batch.batch_to_numpy(clips, representation='6d')
print(f'pad=False → {type(arrays).__name__} of:')
for i, a in enumerate(arrays):
    print(f'  clip {i}: shape {a.shape}')

padded = batch.batch_to_numpy(clips, representation='6d', pad=True)
print(f'\npad=True  → single array: shape {padded.shape}  (B, F_max, D)')

# %% [markdown]
# ## Feature-column layout
#
# The flat feature dimension `D` is structured:
#
# `D = 3 (root position X, Y, Z) + J × rep_dim (flattened joint rotations)`
#
# with `rep_dim` depending on the representation:
#
# | Representation | `rep_dim` per joint | Notes |
# |---|---|---|
# | `euler`     | 3 | Raw Euler angles, degrees |
# | `6d`        | 6 | First two columns of rotation matrix (Zhou et al., 2019) |
# | `quaternion`| 4 | Scalar-first (w, x, y, z) |
# | `axisangle` | 3 | Rotation vector; norm = angle in radians |
# | `rotmat`    | 9 | Full 3×3 rotation matrix, flattened |
#
# Pass `include_root_pos=False` to drop the 3 leading position columns when your model conditions on rotation only.

# %%
print(f'Joint count: {base.joint_count}; expected D = 3 + J × rep_dim\n')
for rep in ['euler', '6d', 'quaternion', 'axisangle', 'rotmat']:
    arr = batch.batch_to_numpy(clips, representation=rep)[0]
    rep_dim = (arr.shape[1] - 3) // base.joint_count
    print(f'  {rep:11s}  rep_dim = {rep_dim}  →  D = {arr.shape[1]}')

# %% [markdown]
# ## Validation
#
# Two ways to check whether clips can be batched:
#
# - **`bvh.matches_topology(other)`** — boolean predicate (same `joint_names` and `euler_orders`). Use this to filter out incompatible clips ahead of time.
# - **`batch_to_numpy(...)`** — runs the same check internally and raises `ValueError` on mismatch, with a message pointing at the first divergent joint. Use this when bad data should hard-fail rather than be silently filtered.

# %%
# Soft predicate — no exception, just a boolean
ex = pybvh.read_bvh_file(bvh_folder / 'bvh_example.bvh')   # 24 joints
t1 = pybvh.read_bvh_file(bvh_folder / 'bvh_test1.bvh')     # 24 joints, same skeleton
with warnings.catch_warnings():
    warnings.simplefilter('ignore')  # bvh_test3 emits a rest/animation warning on load
    t3 = pybvh.read_bvh_file(bvh_folder / 'bvh_test3.bvh')  # 60 joints

print(f'bvh_example vs bvh_test1: {ex.matches_topology(t1)}')
print(f'bvh_example vs bvh_test3: {ex.matches_topology(t3)}')

# %%
# Files with different joint counts cannot batch together
import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore')  # bvh_test3 emits a rest/animation warning on load
    incompat_a = pybvh.read_bvh_file(bvh_folder / 'bvh_test1.bvh')  # 24 joints
    incompat_b = pybvh.read_bvh_file(bvh_folder / 'bvh_test3.bvh')  # 60 joints

try:
    batch.batch_to_numpy([incompat_a, incompat_b])
except ValueError as e:
    print(f'Caught: {e}')

# %% [markdown]
# # Normalization

# %% [markdown]
# ML models trained on raw motion features are gradient-dominated by the largest-scale channels. Concretely:
#
# - **Root positions** are in file units (typically centimeters) with magnitudes in the 10-100 range.
# - **Euler angles** are in degrees, range `[-180, 180]`.
# - **Quaternions, 6D components** are in `[-1, 1]`.
#
# Without normalization, a neural network sees the position channels ~100× more strongly than rotation channels. **Per-channel z-score normalization** — subtract mean, divide by std — brings every channel to the same scale.

# %% [markdown]
# ## Computing statistics
#
# `compute_normalization_stats()` concatenates all clips along the time axis and computes per-channel mean and std across the resulting `(total_frames, D)` tensor.

# %%
stats = batch.compute_normalization_stats(clips, representation='6d')

print(f'Stats keys: {list(stats.keys())}')
print(f'Mean shape: {stats["mean"].shape}  (D channels)')
print(f'Std shape:  {stats["std"].shape}')

print(f'\nRoot position channels (0–2) — centimeter-scale:')
print(f'  mean: {stats["mean"][:3]}')
print(f'  std:  {stats["std"][:3]}')
print(f'\nFirst rotation channel (index 3, root 6D[0]) — unit-scale:')
print(f'  mean: {stats["mean"][3]:.3f}, std: {stats["std"][3]:.3f}')

# %% [markdown]
# ## Applying and reversing
#
# `normalize_array()` applies `(data - mean) / std`; `denormalize_array()` reverses it.

# %%
arrays = batch.batch_to_numpy(clips, representation='6d')
normalized = [batch.normalize_array(arr, stats) for arr in arrays]

# Round-trip check
recovered = batch.denormalize_array(normalized[0], stats)
print(f'Round-trip max error: {np.max(np.abs(arrays[0] - recovered)):.2e}')

# %% [markdown]
# ## Verifying the result
#
# If normalization worked, the concatenated normalized data should have per-channel mean ≈ 0 and std ≈ 1:

# %%
concat = np.concatenate(normalized, axis=0)

# Constant channels (same value across the whole dataset) have their std
# protected at 1.0, so their post-norm std is 0, not 1. Use the mask from
# compute_normalization_stats to exclude them from the verification.
active = ~stats['constant_channels']

print(f'Post-normalization over {concat.shape[0]} frames, {concat.shape[1]} channels '
      f'({(~active).sum()} skipped as constant):')
print(f'  Max |per-channel mean|:    {np.abs(concat.mean(axis=0)[active]).max():.2e}')
print(f'  Max |per-channel std − 1|: {np.abs(concat.std(axis=0)[active] - 1).max():.2e}')

# %% [markdown]
# ## Caveats
#
# Three details that bite users normalizing motion data:
#
# - **Use unpadded arrays for stats.** `compute_normalization_stats` calls `batch_to_numpy(..., pad=False)` internally, so padded zeros don't bias the mean toward zero. If you compute stats by hand on pre-padded data, drop the padding first.
# - **Train-only stats.** Including validation or test clips when computing stats leaks information from held-out data into the normalization. Standard pattern: compute `stats` on the training split, then apply the same `stats` to validation and test at inference time.
# - **Zero-std channels are protected.** Channels that are constant across the dataset (std = 0) would cause division by zero. `compute_normalization_stats` sets their std to `1.0` automatically, and exposes which channels were guarded via the `constant_channels` bool mask in the returned dict (the verification cell above uses it to exclude those channels from the mean/std checks).

# %% [markdown]
# ## Saving and loading stats
#
# Normalization stats are part of the pipeline, not the model. Save them alongside the model so inference code can denormalize predictions back into BVH units.

# %%
stats_path = output_folder / 'norm_stats.npz'
np.savez(stats_path, **stats)

# Later (e.g., at inference):
loaded = dict(np.load(stats_path))
match = (np.allclose(stats['mean'], loaded['mean']) and
         np.allclose(stats['std'],  loaded['std']))
print(f'Saved to {stats_path.name} and reloaded — round-trip OK: {match}')

stats_path.unlink()  # clean up

# %% [markdown]
# # End-to-end pipeline

# %% [markdown]
# Complete dataset-preparation workflow, combining everything above:

# %%
import warnings

# 1. Load raw files
raw = batch.read_bvh_directory(bvh_folder, pattern='bvh_*.bvh')
print(f'Step 1 — Loaded {len(raw)} files')

# 2. Harmonize (topology check / retarget / resample / reorient) in one call.
#    Clips incompatible with the reference skeleton are dropped with a warning.
#    We use bvh_example as the canonical rig here; `standard_skeleton.bvh`
#    uses a different Euler order, which would reject every clip.
reference = pybvh.read_bvh_file(bvh_folder / 'bvh_example.bvh')
with warnings.catch_warnings():
    warnings.simplefilter('ignore')  # quiet the per-drop warnings for tutorial output
    harmonized = batch.harmonize(
        raw,
        reference=reference,
        target_fps=30,
        target_world_up='+z',
        verbose=False,
    )
print(f'Step 2 — Harmonized, kept {len(harmonized)} of {len(raw)} clips')

# 3. Convert to arrays
arrays = batch.batch_to_numpy(harmonized, representation='6d')
print(f'Step 3 — Converted, D = {arrays[0].shape[1]}')

# 4. Compute normalization stats
stats = batch.compute_normalization_stats(harmonized, representation='6d')
print(f'Step 4 — Stats computed over D = {len(stats["mean"])} channels')

# 5. Normalize and save
normalized = [batch.normalize_array(a, stats) for a in arrays]
np.savez(output_folder / 'dataset.npz',
         **{f'clip_{i}': a for i, a in enumerate(normalized)})
np.savez(output_folder / 'stats.npz', **stats)
print(f'Step 5 — Saved {len(normalized)} normalized clips + stats to {output_folder}/')

# Clean up (for tutorial reruns)
(output_folder / 'dataset.npz').unlink()
(output_folder / 'stats.npz').unlink()

# %% [markdown]
# For ML-framework-specific downstream work — PyTorch `Dataset` classes, DataLoaders, collate functions, augmentation pipelines, HDF5 packing — see [pybvh-ml](https://github.com/VictorS-67/pybvh-ml), the companion library that builds on top of pybvh.

# %% [markdown]
# # Summary

# %% [markdown]
# | Function / method | Purpose |
# |---|---|
# | `batch.read_bvh_directory(dir)` | Load all matching BVH files |
# | `bvh.retarget(reference)` | Copy bone offsets from a reference skeleton |
# | `bvh.resample(target_fps)` | Resample to a target frame rate (SLERP) |
# | `bvh.reorient_world_up(axis)` | Rotate scene into a common up axis |
# | `batch.batch_to_numpy(clips)` | Convert to `(F_i, D)` arrays or `(B, F_max, D)` tensor |
# | `batch.compute_normalization_stats(clips)` | Per-channel dataset mean and std |
# | `batch.normalize_array(data, stats)` | Z-score normalize |
# | `batch.denormalize_array(data, stats)` | Reverse normalization |
#
# Key parameters (selection — see the [ML pipeline guide](https://victors-67.github.io/pybvh/guide/ml-pipeline/) for the full reference):
#
# - **`read_bvh_directory`**: `pattern`, `sort`, `parallel`, `max_workers`, `world_up`, `lr_mapping`
# - **`batch_to_numpy` / `compute_normalization_stats`**: `representation`, `include_root_pos`, `pad`, `pad_value`

# %% [markdown]
# # What's next
#
# - [Tutorial 5 — Transforms](5.Transforms.ipynb) covered augmentation transforms (mirror, yaw rotation, noise, speed). Apply them after normalization as a stochastic augmentation step.
# - [Tutorial 6 — Motion Features](6.Features.ipynb) covered velocities, foot contacts, and `to_feature_array()` — richer alternatives to raw joint angles.
# - For ML-framework integration (PyTorch, TensorFlow), see [pybvh-ml](https://github.com/VictorS-67/pybvh-ml).
