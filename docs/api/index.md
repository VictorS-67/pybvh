# API Reference

## Find a function

The fastest route from "I want to…" to the exact call. Everything visual is also drawn, one picture per feature, in the [Gallery](../gallery/index.md).

| I want to… | Call | Reference |
|---|---|---|
| Load / write a BVH file | `pybvh.read_bvh_file(path)` / `bvh.write(path)` | [I/O](io.md) |
| Load a whole directory | `read_bvh_directory("data/", parallel=True)` | [Batch](batch.md) |
| Reconcile mixed skeletons / fps / up-axes | `harmonize(clips)` | [`batch.harmonize`][pybvh.batch.harmonize] |
| Get 3D joint positions (forward kinematics) | `bvh.node_positions()` | [`Bvh.node_positions`][pybvh.bvh.Bvh.node_positions] |
| Convert rotation representations | `bvh.to_quat()`, `bvh.to_6d()`, … | [Rotations](rotations.md) |
| Interpolate rotations / rigid transforms | `rotations.quat_slerp(...)` / `rotations.screw_interpolate(...)` | [Rotations & SE(3)](rotations.md#se3-rigid-transforms) |
| Detect foot contacts | `bvh.foot_contacts()` | [`analysis.foot_contacts`][pybvh.analysis.foot_contacts] |
| Measure gait (cadence, stride, symmetry) | `bvh.gait_parameters()` | [`analysis.gait_parameters`][pybvh.analysis.gait_parameters] |
| Score movement smoothness (SPARC, jerk) | `bvh.smoothness(joint, metric="sparc")` | [`analysis.smoothness`][pybvh.analysis.smoothness] |
| Velocities / accelerations / jerk | `bvh.joint_velocities()` … `bvh.node_jerk()` | [Analysis](analysis.md) |
| Trajectory shape (curvature, path length, …) | `bvh.curvature(joint)`, `bvh.path_length(joint)` | [Geometry](geometry.md) |
| Pose extent & centre of mass | `bvh.bounding_box()`, `bvh.center_of_mass()` | [Geometry](geometry.md) |
| Relative pose of two segments, SE(3) twists | `rotations.relative_transform(...)`, `se3_log` | [Rotations & SE(3)](rotations.md#se3-rigid-transforms) |
| Augment data (mirror, rotate, noise, …) | `bvh.mirror()`, `bvh.rotate_vertical(a)`, … | [Transforms](transforms.md) |
| Export one ML-ready feature array | `bvh.to_feature_array(representation="6d")` | [`features.to_feature_array`][pybvh.features.to_feature_array] |
| Stack many clips into one array | `batch_to_numpy(clips, pad=True)` | [`batch.batch_to_numpy`][pybvh.batch.batch_to_numpy] |
| Smooth / differentiate / FFT a signal | `signal.box_filter_smooth(...)`, `signal.fft_magnitude(...)` | [Signal](signal.md) |
| Visualize (snapshot, video, interactive) | `bvh.plot_frame()`, `bvh.render("out.mp4")`, `bvh.play()` | [Visualization](bvhplot.md) |
| Edit the skeleton (retarget, scale, subset) | `bvh.retarget(ref)`, `bvh.extract_joints([...])` | [`Bvh.retarget`][pybvh.bvh.Bvh.retarget] |
| Slice / concatenate / resample frames | `bvh[10:50]`, `bvh + other`, `bvh.resample(30)` | [Bvh Class](bvh.md) |
| Fix the up axis or facing convention | `bvh.reorient_world_up('+z')`, `bvh.reorient_rest_up('+z')` | [`Bvh.reorient_world_up`][pybvh.bvh.Bvh.reorient_world_up] |
| Round-trip through pandas | `bvh.to_df_dict()` / `pybvh.df_to_bvh(...)` | [Bvh Class](bvh.md) |

## Modules at a glance

| Module | Owns | Page |
|---|---|---|
| `pybvh.Bvh` | the central container: skeleton + motion, with every high-level method | [Bvh Class](bvh.md) |
| `pybvh.io` | reading and writing `.bvh` files | [I/O](io.md) |
| `pybvh.rotations` | conversions between all rotation representations, SLERP, SE(3) rigid-transform math | [Rotations & SE(3)](rotations.md) |
| `pybvh.geometry` | array-pure position descriptors: trajectories, bounding volumes, centre of mass | [Geometry](geometry.md) |
| `pybvh.transforms` | augmentation: mirror, rotate, translate, noise, speed, dropout | [Transforms](transforms.md) |
| `pybvh.analysis` | motion dynamics: velocities → jerk, foot contacts, gait, smoothness, covariance | [Analysis](analysis.md) |
| `pybvh.features` | the flat `(F, D)` ML feature-array export and its column layout | [Features](features.md) |
| `pybvh.signal` | array-pure signal utilities: finite differences, stats, smoothing, FFT | [Signal](signal.md) |
| `pybvh.batch` | directory-level loading, harmonization, batched NumPy export | [Batch](batch.md) |
| `pybvh.bvhplot` | visualization: snapshots, video/GIF export, interactive playback | [Visualization](bvhplot.md) |

## Top-level exports

::: pybvh
    options:
      show_submodules: false
      members:
        - read_bvh_file
        - write_bvh_file
        - read_bvh_directory
        - batch_to_numpy
        - harmonize
        - HarmonizeReport
        - df_to_bvh
        - frames_to_node_positions
        - relative_scale_factor
