# bvh_data — sample motion

Small BVH clips used by the test suite, tutorials, and the feature gallery. This directory is **not** shipped in the installed package (`prune bvh_data` in `MANIFEST.in`); it is repository-only.

## `cmu_12_01_walk.bvh` — real walking (CMU subject 12, trial 1)

A real motion-capture walk: Carnegie Mellon University Graphics Lab Motion Capture Database, subject 12, trial 1, in the cgspeed BVH conversion by Bruce Hahne. Used in the feature gallery to illustrate the gait descriptors (`gait_parameters`, `foot_contacts`) on genuine human locomotion rather than synthetic data. 524 frames at 120 fps (~4.4 s); a clean, near-straight steady walk of a few gait cycles.

Source: [mocap.cs.cmu.edu](http://mocap.cs.cmu.edu) · BVH conversion: [cgspeed CMU BVH release](https://sites.google.com/a/cgspeed.com/cgspeed/motion-capture).

**License — CMU Graphics Lab Motion Capture Database** (verbatim, as distributed with the cgspeed conversion): *"This data is free for use in research and commercial projects worldwide. If you publish results obtained using this data, we would appreciate it if you would send the citation to your published paper to jkh+mocap@cs.cmu.edu, and also would add this text to your acknowledgments section: 'The data used in this project was obtained from mocap.cs.cmu.edu. The database was created with funding from NSF EIA-0196217.'"* CMU places no restrictions on use; the cgspeed conversion adds none.

**Acknowledgment:** The data used in this project was obtained from mocap.cs.cmu.edu. The database was created with funding from NSF EIA-0196217.

## Other clips

pybvh's own bundled test fixtures: small clips and a reference skeleton exercised by the unit tests and used as the default subject across most of the feature gallery. Each is there for what it has that the others lack:

| File | What it is for |
|---|---|
| `bvh_example.bvh` | The tests' default clip: an anger clip from the DIEM-A dataset, Z-up at 30 fps, with four different Euler orders across its joints. |
| `bvh_test1.bvh` | A byte-identical copy of `bvh_example.bvh`; the tutorials, the gallery, the README's hero animation and several tests load it under this name. |
| `bvh_test2.bvh` | Y-up at 120 fps, one Euler order (`YXZ`) on every joint, and a root turned so the character faces −Z while its rest pose faces +Z: the regression clip for `camera="front"`. |
| `bvh_test3.bvh` | A large skeleton (60 joints, mixed Euler orders) whose rest pose is Y-up while its animation is Z-up, so reading it warns that the two disagree: the clip for world-up inference and `reorient_rest_up`. |
| `standard_skeleton.bvh` | The node names of `bvh_example.bvh` with other bone lengths and one Euler order, as a single frame at rest: the reference skeleton `retarget` maps onto. |
