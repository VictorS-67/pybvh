# Motion Descriptors

pybvh computes a layer of **theory-neutral motion descriptors**. They are geometric and dynamic properties, measured directly from the motion. They are properties of the *data*, not of any particular consumer. A biomechanics researcher, a game developer, and an ML engineer all read the same curvature, smoothness, and gait numbers. Everything is pure NumPy in and out, so it composes with any framework.

**Every descriptor on this page is drawn in the [Feature Gallery](../gallery/index.md)**, with one figure and one call each. If you want the visual overview before the prose, skim the gallery first.

The descriptors are in four modules. The modules mirror the two halves of the BVH data model (rotations and positions), plus their dynamics and the shared signal layer:

| Module | Owns | Examples |
|---|---|---|
| [`pybvh.geometry`](../api/geometry.md) | positions in R³ | `curvature`, `bounding_box`, `center_of_mass`, `inter_joint_distance` |
| [`pybvh.analysis`](../api/analysis.md) | motion dynamics | `node_jerk`, `smoothness`, `kinetic_energy`, `cadence` |
| [`pybvh.rotations`](../api/rotations.md) | orientation & rigid transforms | `se3_exp`/`se3_log`, `screw_interpolate`, `rotation_geodesic_distance` |
| [`pybvh.signal`](../api/signal.md) | array-pure signal utilities | `finite_difference`, `box_filter_smooth`, `fft_magnitude`, `temporal_stats` |

## Geometry — what the body traces out

`pybvh.geometry` measures points and trajectories:

- How far a joint travels (`path_length`), and how directly (`directness`).
- How sharply its path turns (`curvature`), and how it twists out of plane (`torsion`).
- The size and shape of the whole pose (`bounding_box`, `bounding_sphere`, `bounding_ellipsoid`, `verticality`).
- The pose's centre of mass (`center_of_mass`, `com_displacement`).
- Relations between joints (`inter_joint_distance`, `joint_angle`, `triangle_area`).

Derivative-based kernels share pybvh's one finite-difference convention with the velocity ladder (`signal.finite_difference`). So geometry and kinematics derivatives stay consistent when you combine them.

## Analysis — how the body moves

`pybvh.analysis` adds dynamics on top of the kinematics it already owned:

- `node_jerk`/`joint_jerk`: the third rung of the velocity → acceleration ladder.
- `node_speed_derivative`/`joint_speed_derivative`: the ladder's tangential rung. It is the per-frame `d‖v‖/dt`, positive while a joint speeds up and negative while it brakes. It is distinct from `node_accelerations`, whose norm also counts direction changes.
- **Smoothness** of a speed profile, via SPARC, dimensionless jerk, log dimensionless jerk, number of peaks, and a `smoothness(metric=…)` dispatcher. Every speed-profile kernel also reduces a `(T, K)` stack of profiles per column.
- `kinetic_energy`.
- Gait: `cadence`, `stride_length`, `walking_pace`.
- Peak-to-peak `range_of_motion`.
- Covariance descriptors: `cov3dj`, `lagged_covariance`.

## SE(3) — rigid-transform features

`pybvh.rotations` handles rigid transforms alongside its rotation conversions:

- The exp/log maps between 4×4 transforms and se(3) twists `[ω, v]` (`se3_exp`/`se3_log`).
- Screw-motion interpolation, the SE(3) analogue of SLERP.
- The segment-to-segment `relative_transform` bridge.
- `rotation_geodesic_distance`.

These are the building blocks for Lie-group skeletal features.

## Array-pure kernels vs `Bvh` methods

There are two layers, by design:

- **Array-pure kernels** take plain NumPy arrays and return arrays. They are all of `pybvh.geometry`, the smoothness functions, the covariance descriptors, and the SE(3) math. A downstream library can call them with no `Bvh` at all.
- **`Bvh` methods** wrap the primitives that are either skeleton-bound or common single-joint queries. A skeleton-bound primitive needs `world_up`, `frame_time`, or foot detection. Examples: `bvh.curvature("RightHand")`, `bvh.bounding_box()`, `bvh.smoothness(joint)`, `bvh.cadence()`.

    Relational and trajectory methods resolve names in **node space**, so end sites (fingertips, toe tips, head top) are first-class. `range_of_motion` resolves in **joint** space, because rotations exist only on joints.

    Joints are addressed by **name only**. Integer indices raise `TypeError`, because they were ambiguous between the two index spaces. For index-based access, resolve with `bvh.index(name, space=...)`, and use the array-pure kernels.

    Every descriptor method also accepts pre-computed positions via `coords=`, for hot loops. The FK cache serves the default path anyway.

!!! note "Name-based primitives assume one consistent skeleton"
    Descriptors addressed by joint name (`inter_joint_distance(pairs)`, foot and centre-of-mass detection) assume every clip shares the same skeleton. Reconciling differing skeletons across a dataset is a harmonization concern (`pybvh.batch.harmonize`; `pybvh.analysis.relative_scale_factor` for size), not a descriptor concern.

## See also

- The [Feature Gallery](../gallery/index.md) — every descriptor drawn, one figure and one call each (sections 6-12).
- The [Motion descriptors tutorial](https://github.com/VictorS-67/pybvh/blob/main/tutorials/8.Motion_descriptors.ipynb) walks through these on a real clip with closed-form sanity checks.
- API reference: [geometry](../api/geometry.md), [analysis](../api/analysis.md), [rotations](../api/rotations.md), [signal](../api/signal.md).
