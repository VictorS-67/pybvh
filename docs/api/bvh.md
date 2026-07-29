# Bvh Class

::: pybvh.bvh.Bvh
    options:
      members: false

## Construction & I/O

Create a `Bvh` from a file, a DataFrame, or another instance; write it back losslessly.

::: pybvh.bvh.Bvh.from_file
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.from_df
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.write
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.copy
    options:
      heading_level: 3

## Data, metadata & skeleton introspection

The raw motion arrays, the joint hierarchy, and the two index spaces (see the [Core Concepts guide](../guide/core-concepts.md)).

::: pybvh.bvh.Bvh.nodes
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.root
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.root_pos
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.joint_angles
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.frame_count
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.frame_time
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.fps
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.joint_names
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.joint_count
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.euler_orders
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.node_index
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.joint_index
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.index
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.joint_tips
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.edges
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.node_edges
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.fk_topology
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.matches_hierarchy
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.matches_channels
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.matches_topology
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.to_hierarchy_dict
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.to_df_dict
    options:
      heading_level: 3

## Orientation & frames of reference

Up axis, facing direction, L/R pairs, and the reorientation family (see the [World Up guide](../guide/world-up.md)).

::: pybvh.bvh.Bvh.world_up
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.world_up_inferred
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.up_axis
    options:
      heading_level: 3

::: pybvh.tools.Axis
    options:
      heading_level: 3

::: pybvh.tools.parse_axis
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.rest_up
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.rest_up_axis
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.rest_forward
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.forward_axis
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.floor_height
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.forward_at
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.left_at
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.facing_frame
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.lr_mapping
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.lr_pairs
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.node_lr_pairs
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.has_lr_geometry
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.reorient_world_up
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.reorient_rest_up
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.reorient_rest_forward
    options:
      heading_level: 3

## Forward kinematics

From joint angles to 3D positions — the `centered` modes are drawn in the [Gallery](../gallery/index.md).

::: pybvh.bvh.Bvh.node_positions
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.joint_positions
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.rest_pose_positions
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.rest_pose_angles
    options:
      heading_level: 3

## Rotation representations

Conversions to and from every representation (see [Choosing a Representation](../guide/choosing-rotations.md)).

::: pybvh.bvh.Bvh.to_rotmat
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.to_6d
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.to_quat
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.to_axisangle
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.from_rotmat
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.from_6d
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.from_quat
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.from_axisangle
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.change_euler_order
    options:
      heading_level: 3

## Frame & skeleton operations

Timeline and skeleton editing (see the [Skeleton Operations guide](../guide/skeleton-ops.md)). Slicing and concatenation use plain Python syntax: `bvh[10:50]` returns the frame range as a new `Bvh`, and `bvh_a + bvh_b` concatenates two clips with matching skeletons.

::: pybvh.bvh.Bvh.resample
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.extract_joints
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.retarget
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.scale
    options:
      heading_level: 3

## Kinematics, trajectory & contacts

The velocity ladder, the root-trajectory features, and foot-contact detection (see the [Feature Export guide](../guide/feature-export.md)).

::: pybvh.bvh.Bvh.joint_velocities
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.node_velocities
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.joint_accelerations
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.node_accelerations
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.joint_speed_derivative
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.node_speed_derivative
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.angular_velocities
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.root_trajectory
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.foot_contacts
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.ground_contacts
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.auto_detect_foot_joints
    options:
      heading_level: 3

## Feature export

The one-stop flat `(F, D)` array for ML pipelines.

::: pybvh.bvh.Bvh.to_feature_array
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.feature_array_layout
    options:
      heading_level: 3

## Trajectory & pose geometry

Position descriptors — each drawn in the [Gallery](../gallery/index.md); array-pure kernels in [`pybvh.geometry`](geometry.md).

::: pybvh.bvh.Bvh.curvature
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.torsion
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.movement_phase
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.path_length
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.directness
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.ground_path
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.inter_joint_distance
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.joint_angle
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.triangle_area
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.segment_axis_angle
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.bounding_box
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.bounding_sphere
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.bounding_ellipsoid
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.center_of_mass
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.com_displacement
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.verticality
    options:
      heading_level: 3

## Dynamics, smoothness & gait

Dynamic descriptors (see the [Motion Descriptors guide](../guide/motion-descriptors.md)); array-pure kernels in [`pybvh.analysis`](analysis.md).

::: pybvh.bvh.Bvh.node_jerk
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.joint_jerk
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.smoothness
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.velocity_reductions
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.kinetic_energy
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.skeleton_size
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.cadence
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.stride_length
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.walking_pace
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.gait_parameters
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.range_of_motion
    options:
      heading_level: 3

## Augmentation transforms

Seeded data augmentation (see the [Data Augmentation guide](../guide/augmentation.md)); array-level functions in [`pybvh.transforms`](transforms.md).

::: pybvh.bvh.Bvh.mirror
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.rotate_vertical
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.translate_root
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.add_rotation_noise
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.add_position_noise
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.perturb_speed
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.drop_frames
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.random_translate_root
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.random_rotate_vertical
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.random_perturb_speed
    options:
      heading_level: 3

## Visualization

Quick-look plotting on the object; multi-skeleton comparison lives in [`pybvh.bvhplot`](bvhplot.md).

::: pybvh.bvh.Bvh.plot_rest_pose
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.plot_frame
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.plot_trajectory
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.render
    options:
      heading_level: 3

::: pybvh.bvh.Bvh.play
    options:
      heading_level: 3
