# Rotations & SE(3)

::: pybvh.rotations
    options:
      members: false

## Representation conversions

Batch-vectorized conversions between Euler angles, rotation matrices, quaternions, 6D, and axis-angle.

::: pybvh.rotations.euler_to_rotmat
    options:
      heading_level: 3

::: pybvh.rotations.rotmat_to_euler
    options:
      heading_level: 3

::: pybvh.rotations.rotmat_to_rot6d
    options:
      heading_level: 3

::: pybvh.rotations.rot6d_to_rotmat
    options:
      heading_level: 3

::: pybvh.rotations.euler_to_rot6d
    options:
      heading_level: 3

::: pybvh.rotations.rot6d_to_euler
    options:
      heading_level: 3

::: pybvh.rotations.rotmat_to_quat
    options:
      heading_level: 3

::: pybvh.rotations.quat_to_rotmat
    options:
      heading_level: 3

::: pybvh.rotations.euler_to_quat
    options:
      heading_level: 3

::: pybvh.rotations.quat_to_euler
    options:
      heading_level: 3

::: pybvh.rotations.rotmat_to_axisangle
    options:
      heading_level: 3

::: pybvh.rotations.axisangle_to_rotmat
    options:
      heading_level: 3

::: pybvh.rotations.euler_to_axisangle
    options:
      heading_level: 3

::: pybvh.rotations.axisangle_to_euler
    options:
      heading_level: 3

## Quaternion utilities

Composition and spherical interpolation.

::: pybvh.rotations.quat_multiply
    options:
      heading_level: 3

::: pybvh.rotations.quat_slerp
    options:
      heading_level: 3

## The convert dispatcher

One entry point that routes between any pair of representations by name.

::: pybvh.rotations.convert
    options:
      heading_level: 3

## SE(3) rigid transforms

Twists, the exp/log maps, screw interpolation, and segment-relative poses — each drawn in the [Gallery](../gallery/index.md).

::: pybvh.rotations.se3_exp
    options:
      heading_level: 3

::: pybvh.rotations.se3_log
    options:
      heading_level: 3

::: pybvh.rotations.se3_inverse
    options:
      heading_level: 3

::: pybvh.rotations.screw_interpolate
    options:
      heading_level: 3

::: pybvh.rotations.relative_transform
    options:
      heading_level: 3

::: pybvh.rotations.rotation_geodesic_distance
    options:
      heading_level: 3

::: pybvh.rotations.mean_rotation
    options:
      heading_level: 3
