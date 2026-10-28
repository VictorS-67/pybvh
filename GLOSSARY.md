# pybvh

The library that reads, writes, transforms and analyzes BVH motion capture files, and the vocabulary its code, docs, issues and reviews use for what a BVH file contains. `CONTEXT.md` is the architecture reference; this file is the language.

## Language

### The skeleton

**Node**:
One entry of a skeleton's hierarchy: the root, a joint or an end site. `Bvh.nodes` lists them in the file's depth-first order. A node passed to `Bvh(...)` is owned by that clip and frozen; a changed skeleton goes to a new `Bvh`.

**Root**:
The first node, the only one with position channels; its translation per frame is the root position.

**Joint**:
A node with rotation channels: the root and every interior node. `J` counts them.

**End site**:
A childless node with an offset and no channels: the tip of a terminal bone. Its name is generated from its parent's (`EndSite` + parent name), so the two end sites of one joint share a name.
_Avoid_: leaf

**Offset**:
A node's bone vector from its parent in the rest pose, as the file declares it.

**Bone**:
The segment between a node and its parent, what bvhplot draws; a node's offset is its bone's rest-pose vector.

**Channels**:
The per-node list of axes a file declares, in its order: three rotation channels on a joint, plus three position channels on the root.

**Euler order**:
A joint's rotation channel order as a string (`'ZYX'`): the sequence of intrinsic rotations its angles apply.
_Avoid_: rotation order

**Hierarchy**:
The tree of nodes with their names and offsets, as the HIERARCHY section declares it. `matches_hierarchy` compares exactly that.

**Topology**:
A hierarchy together with its channel layout. Two clips with the same topology share the same joint-angle columns; `matches_topology` compares it, and `a + b`, `a += b` and `a[i:j] = b` require it.

**Skeleton**:
The everyday word for the hierarchy a clip carries beside its motion; use hierarchy and topology where the exact comparison matters.

**Node table**:
The skeleton as plain data: one `dict` per node in `nodes` order, with `name`, `parent` (the parent's index, `None` on the root), `offset`, `rot_channels`, and `pos_channels` on the root. The named twin of `FkTopology`.
_Avoid_: hierarchy dict (the name-keyed form removed in v0.10.0)

**FkTopology**:
The skeleton as arrays for forward kinematics: offsets, parent indices, joint columns and Euler orders. An FK input bundle, with no names and no orientation.

**Rest pose**:
The skeleton with every rotation zero, positions the summed offsets. Rest up, rest left and rest forward are read from it alone: its coordinates are the skeleton's own frame, not the world's.
_Avoid_: T-pose, bind pose

**Bone chain**:
In bvhplot, the run of bones from a limb's root to its tip that shares one color: left warm, right cool, spine dark.

### Motion

**Clip**:
One `Bvh`: a skeleton and its motion, usually one file.
_Avoid_: take, recording

**Derived state**:
What a clip holds that is computed from its skeleton, motion or world up, such as the forward-kinematics positions or the scene ground: rebuilt on demand after an input changes, never copied to a derived clip. Unlike a descriptor, it is held by the clip rather than computed per call.
_Avoid_: cache

**Frame**:
One row of the motion: the root position and every joint's angles at one instant. `F` counts them.

**Frame time**:
Seconds per frame, the file's `Frame Time:`; `fps` is derived from it. Zero means unset.
_Avoid_: frame frequency

**Joint angles**:
The `(F, J, 3)` array of Euler angles, in radians, each joint in its Euler order; degrees only in files and DataFrames.

**Root position**:
The `(F, 3)` translation of the root, always `(X, Y, Z)`.

**Animation**:
The motion as distinct from the rest pose: what world up, the facing and the heading are read from.

**Centered**:
How the root position is treated when posing: `"world"` keeps the file's coordinates, `"skeleton"` pins the root at the origin, `"first"` subtracts the first frame's root position in the two ground axes only.

**Round trip**:
Reading a clip, writing it (to a file, a DataFrame or a rotation representation) and reading it back to the same clip within float precision. The fidelity standard.

**Harmonize**:
Bringing a set of clips to one topology, frame rate, orientation and Euler order in one call, with a `HarmonizeReport` of what was applied to each.

### Index spaces

**Node space**:
An axis of length `N` indexed like `nodes`, end sites included: positions, node velocities, `node_index`, `node_edges`, `node_lr_pairs`.

**Joint space**:
An axis of length `J` indexed like `joint_angles`, end sites excluded: angles, angular velocities, `joint_index`, `edges`, `lr_pairs`.

**Edges**:
The parent-child pairs of a skeleton as index tuples, in joint space (`edges`) or node space (`node_edges`); both are views of `fk_topology.parent_idx`.

**L/R pairs**:
Left and right counterparts detected from names (`lr_mapping`), as index pairs in joint space (`lr_pairs`) or node space (`node_lr_pairs`); what mirroring and the facing geometry use.

### Orientation

Orientation facts belong to one of two frames, and no fact combines an axis of one with an axis of the other (ADR 0004). The `rest_*` facts are the skeleton's own frame, read from the rest pose alone; world up, the facing and the heading are the world's, read from the motion. The root rotation is what maps a skeleton fact into the world, on each frame.

Names follow one rule: a name ending in `_axis` holds an `Axis`, and a signed axis string naming world up is called `world_up`.

**Axis string**:
A signed axis such as `'+y'` or `'-z'`, the form every orientation property speaks; `parse_axis` turns it into an `Axis(index, sign, vector)`.

**World up**:
The signed axis that points against gravity in the animation, guessed from the first frame, else from rest up as a labelled guess, and inherited by every clip derived from the clip. Settable, and a set value is kept until the user changes it.
_Avoid_: vertical axis

**Rest up**:
The skeleton's own up, read from the rest pose alone. It agrees with world up on a clean file and differs on one whose rest pose was authored in another convention than its motion.

**Rest left**:
The skeleton's own left, read from the rest pose's L/R geometry against rest up: the axis `mirror` reflects along.

**Rest forward**:
The direction the rest pose faces in the skeleton's own frame: rest left × rest up. Not a direction in the world; the heading is where it meets the world.

**Facing**:
The horizontal direction the whole body points at a frame, in the world: the frame's own leftward against world up, else the root's heading at that frame when rest forward is measured (a heading built on a defaulted rest forward would offer a guessed direction as a measured one), else the house default, with forward = left × up. `facing_frame()` gives the continuous `(forward, left, up)` basis, its `valid` false on the fallback rows; `forward_at` and `left_at` snap it to axis labels.

**Heading**:
The root's own orientation projected on the ground: the root rotation applied to rest forward, as `root_trajectory` reports it, `NaN` on a frame where that projection is too short to read. Not the direction of travel, and not the facing: a side-stepping character keeps its heading. Built on rest forward even when rest forward is the house default, with its warning; the heading and its rate of turn then both depend on that guessed reference whenever the root pitches or rolls, so they are reliable only for a root that turns about world up alone. The facing takes no heading from a defaulted rest forward.

**Orientation report**:
How each orientation fact of a clip was obtained, set, measured or defaulted, beside its value (`bvh.orientation_report`, an `OrientationReport`). Per-frame facts are not in it: a facing basis marks its fallback rows itself.

**Reorientation**:
Changing a clip's world up, rest up or rest forward to a target; the three-axis step of harmonize, which applies them in that order.

### Rotation representations

**Euler angles**:
Intrinsic rotations applied in the joint's Euler order and pre-multiplied, `R = R1 @ R2 @ R3`.

**Quaternion**:
Scalar-first `(w, x, y, z)`, canonical with `w >= 0`.

**6D rotation**:
The continuous six-number form of Zhou et al.; `to_6d` and `from_6d`.

**Axis-angle**:
A vector whose direction is the axis and whose norm is the angle in `[0, π]`; the zero vector is the identity.

**Rotation matrix**:
The hub every conversion routes through; the chordal mean of a batch of them is `mean_rotation`.

### Analysis

**Descriptor**:
A quantity computed from a clip's motion or geometry: velocities, jerk, path length, a bounding box, gait parameters. Relational and trajectory descriptors resolve in node space, `range_of_motion` in joint space.

**Feature array**:
The flat `(F, D)` array `to_feature_array` packs descriptors into, described by its layout.
_Avoid_: feature (for a single descriptor)

**Velocity ladder**:
Velocities, accelerations and jerk by the one finite-difference stencil, with the speed derivative as the tangential rung.

**Scene ground**:
Where a clip's geometry bottoms out: the robust 2nd percentile of the per-frame minimum height over all nodes, end sites included (`Bvh.floor_height`). What a floor is drawn at.
_Avoid_: floor (alone), ground level

**Contact reference**:
The ground level a contact detector estimates per call over the joints it tests (`floor="auto"`, reported as the contact report's `contact_reference`). Never the scene ground; the two differ by about a toe length (ADR 0002).

**Contact**:
A frame in which a joint is planted, from velocity and height thresholds: `foot_contacts` on the detected feet, `ground_contacts` on any joint set.

**Contact report**:
What one contact detection applied and compared, returned with the contacts by `return_info=True` (`ContactReport`): the method and settings, each joint's thresholds, the skeleton scale, the contact reference, the speed and clearance signals, and the diagnostics. Every field present, `None` where the method does not use it; per-joint values `(n_joints,)`, never averaged.

**Threshold source**:
Where one joint's contact threshold came from, in the contact report (`vel_threshold_source`, `height_threshold_source`): given by the caller, the default scaled by the skeleton scale (`"scale"`, `"margin"`), the stance calibration (`"stance"`), Otsu's split (`"otsu"`), or a bound standing in for a split outside it (`"otsu_clamped"`). A fallback reads as what stands in, never as the method that failed. Beside it, `skeleton_scale_measured` says whether the skeleton scale was measured or `1.0` stood in.

**Gait parameters**:
The eight fields of `GaitParameters`, from contacts and the root's ground path: `cadence`, `walking_pace`, `stride_length`, `stride_cv`, `step_length`, `stance_fraction`, `double_support_fraction` and `asymmetry`. `gait_parameters` computes them in one pass; `cadence` and `stride_length` are its projections, and `walking_pace` is also computed alone from the root path.

**Smoothness**:
A metric on a speed profile: SPARC, dimensionless jerk and the others `smoothness(metric=)` dispatches to.

### Visualization (bvhplot)

**Scene**:
The pure-data input every backend draws: one view per skeleton, changed through intent-named operations (`subsampled`, `offset`, `spread`, `scaled`, `size_matched`, `looped`). It holds no `Bvh`.

**View**:
One skeleton's coords, bones, floor, axes and metadata (`SkeletonView`), complete and read-only; `make_scene` builds one from a `Bvh`, tests build them from arrays.

**Viewport**:
The geometry of one picture: cube, framing box, floor plane, camera and schedule, projection. One frozen value per picture, computed when the picture is made; no backend computes its own.

**Cube**:
The viewport's scale: a center and a half-span with a 5% margin. The floor's reach and the perspective camera's target are multiples of it.

**Framing box**:
What axes or the orthographic projection are fitted to: the cube for a still, the box the whole clip sweeps for a clip.

**Body size**:
A view's height along its rest up, in the coords' unit; what is drawn on a body (capsule radius, label lift, bone width) is sized from it.

**Schedule**:
The camera azimuth per frame: fixed, turntable or follow. A constant schedule is stored as none and framed as a fixed camera.

**Style**:
The one styling object every bvhplot function accepts: the presets `paper`, `debug` and `dark`, with overrides.

**Backend**:
The toolkit a picture is drawn with: matplotlib, OpenCV, k3d, the vedo viewer or the vedo offscreen renderer, chosen by the call unless named: `render` by the file's extension and the installed packages, `play` by the session (notebook, display) and the installed packages; `frame` draws with matplotlib unless named. Each declares the options it reads (ADR 0006) and the `Style` values it draws.

**Video target**:
Where an animation is written: the path, the rate written, the resolution and the codec. One writer serves every backend.

**Sequence**:
The motion-paper still: equidistantly sampled poses in one figure, lighter the earlier.

**Animation options**:
What an animation draws beside its Scene, apart from the Style and the video target: the camera's schedule, ghosts, the trail and the frame counter. Built by `render` and `play` from their options; a still (`frame`) has none.

**Ghost**:
A trailing pose drawn faded behind the current one, spaced in seconds of clip time.

**Trail**:
The root's path drawn on the floor (`trajectory=`).
_Avoid_: trace

**Spread**:
Laying a comparison's skeletons side by side, each further toward the first skeleton's own left.

**Comparison**:
Several clips in one scene, in a figure or a playback.

**Turntable**:
A camera revolving around the scene, one revolution per period.

### Ecosystem

**Consumer**:
A library built on pybvh (pybvh-ml, pybvh-blender). Dependencies flow one way: pybvh never imports or knows a consumer.
