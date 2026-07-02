# Motion Descriptors

pybvh computes a layer of **theory-neutral motion descriptors** — geometric and
dynamic properties measured directly from the motion. They are properties of the
*data*, not of any particular consumer: a biomechanics researcher, a game
developer, and an ML engineer all read the same curvature, smoothness, and gait
numbers. Everything is pure NumPy in and out, so it composes with any framework.

The descriptors live in three modules, mirroring the two halves of the BVH data
model (rotations and positions) plus their dynamics:

| Module | Owns | Examples |
|---|---|---|
| [`pybvh.geometry`](../api/geometry.md) | positions in R³ | `curvature`, `bounding_box`, `center_of_mass`, `inter_joint_distance` |
| [`pybvh.analysis`](../api/analysis.md) | motion dynamics | `node_jerk`, `smoothness`, `kinetic_energy`, `cadence` |
| [`pybvh.rotations`](../api/rotations.md) | orientation & rigid transforms | `se3_exp`/`se3_log`, `screw_interpolate`, `rotation_geodesic_distance` |

## Geometry — what the body traces out

`pybvh.geometry` measures points and trajectories: how far a joint travels
(`path_length`) and how directly (`directness`); how sharply its path turns
(`curvature`) and twists out of plane (`torsion`); the size and shape of the
whole pose (`bounding_box`, `bounding_sphere`, `bounding_ellipsoid`,
`verticality`); its centre of mass (`centroid`, `com_displacement`); and
relations between joints (`inter_joint_distance`, `joint_angle`,
`triangle_area`). Derivative-based kernels share pybvh's one finite-difference
convention with the velocity ladder (`tools.finite_difference`), so geometry and
kinematics derivatives stay consistent when you combine them.

## Analysis — how the body moves

`pybvh.analysis` adds dynamics on top of the kinematics it already owned:
`node_jerk`/`joint_jerk` (the third rung of the velocity → acceleration ladder);
**smoothness** of a speed profile via SPARC, dimensionless jerk, log
dimensionless jerk, number of peaks, and a `smoothness(metric=…)` dispatcher;
`kinetic_energy`; gait (`cadence`, `stride_length`, `walking_pace`); peak-to-peak
`range_of_motion`; and covariance descriptors (`cov3dj`, `lagged_correlation`).

## SE(3) — rigid-transform features

`pybvh.rotations` handles rigid transforms alongside its rotation conversions:
the exp/log maps between 4×4 transforms and se(3) twists `[ω, v]`
(`se3_exp`/`se3_log`), screw-motion interpolation (the SE(3) analogue of SLERP),
the segment-to-segment `relative_transform` bridge, and
`rotation_geodesic_distance`. These are the building blocks for Lie-group
skeletal features.

## Array-pure kernels vs `Bvh` methods

Two layers, by design:

- **Array-pure kernels** take plain NumPy arrays and return arrays — all of
  `pybvh.geometry`, the smoothness functions, the covariance descriptors, and the
  SE(3) math. A downstream library can call them with no `Bvh` at all.
- **`Bvh` methods** wrap the primitives that are either skeleton-bound (they need
  `world_up`, `frame_time`, or foot detection) or common single-joint queries —
  e.g. `bvh.curvature("RightHand")`, `bvh.bounding_box()`, `bvh.smoothness(joint)`,
  `bvh.cadence()`. Relational and trajectory methods resolve names in **node
  space**, so end sites (fingertips, toe tips, head top) are first-class;
  `range_of_motion` resolves in **joint** space, since rotations exist only on
  joints.

!!! note "Name-based primitives assume one consistent skeleton"
    Descriptors addressed by joint name (`inter_joint_distance(pairs)`, foot and
    centre-of-mass detection) assume every clip shares the same skeleton.
    Reconciling differing skeletons across a dataset is a `pybvh.batch` concern
    (`harmonize`, `relative_scale_factor`), not a descriptor concern.

## See also

- The [Motion descriptors tutorial](https://github.com/VictorS-67/pybvh/blob/main/tutorials/8.Motion_descriptors.ipynb)
  walks through these on a real clip with closed-form sanity checks.
- API reference: [geometry](../api/geometry.md), [analysis](../api/analysis.md),
  [rotations](../api/rotations.md).
