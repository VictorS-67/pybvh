# Analysis

::: pybvh.analysis
    options:
      members: false

## Velocities & accelerations

The finite-difference kinematics ladder over FK positions and rotations.

::: pybvh.analysis.node_velocities
    options:
      heading_level: 3

::: pybvh.analysis.joint_velocities
    options:
      heading_level: 3

::: pybvh.analysis.node_accelerations
    options:
      heading_level: 3

::: pybvh.analysis.joint_accelerations
    options:
      heading_level: 3

::: pybvh.analysis.angular_velocities
    options:
      heading_level: 3

## Root trajectory & foot contacts

Ground-plane root features and binary contact labels — the contact signals are drawn in the [Gallery](../gallery/index.md).

::: pybvh.analysis.root_trajectory
    options:
      heading_level: 3

::: pybvh.analysis.foot_contacts
    options:
      heading_level: 3

::: pybvh.analysis.auto_detect_foot_joints
    options:
      heading_level: 3

## Jerk

The third rung of the velocity → acceleration → jerk ladder.

::: pybvh.analysis.node_jerk
    options:
      heading_level: 3

::: pybvh.analysis.joint_jerk
    options:
      heading_level: 3

## Smoothness metrics

Array-pure kernels on a 1-D speed profile, plus the `smoothness(metric=…)` dispatcher.

::: pybvh.analysis.sparc
    options:
      heading_level: 3

::: pybvh.analysis.dimensionless_jerk
    options:
      heading_level: 3

::: pybvh.analysis.log_dimensionless_jerk
    options:
      heading_level: 3

::: pybvh.analysis.number_of_peaks
    options:
      heading_level: 3

::: pybvh.analysis.speed_metric
    options:
      heading_level: 3

::: pybvh.analysis.integrated_squared_jerk
    options:
      heading_level: 3

::: pybvh.analysis.mean_squared_jerk
    options:
      heading_level: 3

::: pybvh.analysis.rms_squared_jerk
    options:
      heading_level: 3

::: pybvh.analysis.smoothness
    options:
      heading_level: 3

## Signal reductions

Scalar summaries of a speed or activity signal.

::: pybvh.analysis.VelocityReductions
    options:
      heading_level: 3

::: pybvh.analysis.velocity_reductions
    options:
      heading_level: 3

::: pybvh.analysis.zero_crossings
    options:
      heading_level: 3

::: pybvh.analysis.active_segments
    options:
      heading_level: 3

::: pybvh.analysis.active_duration
    options:
      heading_level: 3

## Energy, gait & range of motion

Kinetic energy and the spatiotemporal gait parameters (all computed in one pass by `gait_parameters`).

::: pybvh.analysis.kinetic_energy
    options:
      heading_level: 3

::: pybvh.analysis.cadence
    options:
      heading_level: 3

::: pybvh.analysis.stride_length
    options:
      heading_level: 3

::: pybvh.analysis.walking_pace
    options:
      heading_level: 3

::: pybvh.analysis.GaitParameters
    options:
      heading_level: 3

::: pybvh.analysis.gait_parameters
    options:
      heading_level: 3

::: pybvh.analysis.range_of_motion
    options:
      heading_level: 3

## Scale & covariance descriptors

Skeleton-size normalization and fixed-size sequence statistics.

::: pybvh.analysis.skeleton_size
    options:
      heading_level: 3

::: pybvh.analysis.relative_scale_factor
    options:
      heading_level: 3

::: pybvh.analysis.cov3dj
    options:
      heading_level: 3

::: pybvh.analysis.lagged_covariance
    options:
      heading_level: 3
