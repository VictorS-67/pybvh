# A joint's position channels are its whole position relative to its parent, stored beside root_pos

Through v0.9.0 pybvh refused a BVH file whose joints other than the root declare position channels (`pybvh does not model position channels on non-root joints`). Some exporters write six channels (3 positions + 3 rotations) on every joint: the root is then often a static reference node and the hips joint carries the travel. Six channels also let a bone stretch frame by frame and let props and markers move on their own. The motion model had no place for such a value: `root_pos` was the only position array, and `FkTopology` had no field for a joint's position. Issue #186 implements the decisions below.

## Decision

**1. A joint's position channels are its whole position relative to its parent.** In each frame they replace the joint's `OFFSET`; the `OFFSET` describes the rest pose only. A joint with `OFFSET 0 10 0` and the frame values `0 12 0` is 12 above its parent in that frame, not 22. The root already works this way: its position channels are its position, and forward kinematics does not add its offset.

**2. The values are stored compact, beside `root_pos`, as the file's values.** `Bvh.joint_translations`, shape `(F, P, 3)`, holds the position channels of the P joints other than the root that declare them, in node order, in `X Y Z` order: the file's values, not the difference from the `OFFSET`. `root_pos` keeps its name, shape and meaning: the root is in the world frame, a translating joint in its parent's frame. A clip with no translating joint has P = 0. The array is one entry of the clip's motion list, interpolated linearly.

**3. Which joints translate is a skeleton fact.** The `CHANNELS` lines fix it, as they fix the Euler orders: `BvhJoint.pos_channels` is `('X', 'Y', 'Z')` on a translating joint and `()` on a joint that only rotates. `FkTopology` gains a fifth field, `translation_idx`: per node, its column in `joint_translations`, or −1 to use its `OFFSET`. `matches_channels` compares the declaration, never the values.

**4. A joint other than the root declares 3 rotation channels, or 3 position + 3 rotation channels.** Anything else is refused, and the error names the joint. The root declares 3 position + 3 rotation channels, or none (`CHANNELS 0`, a static root, which stays at its `OFFSET` with no rotation; #186 also implements this). Position channels are read in any axis order and stored, and written, as `Xposition Yposition Zposition` before the rotation channels, the rule of the root's position order (#179). Position tokens interleaved with rotation tokens are read as "translate first, then rotate".

## Why

**The exporters replace the offset.** Readers disagree on this point: some add the position channels to the `OFFSET`. But the exporters that we read and the six-channel files that we measured all replace it: a joint that does not move has frame values equal to its `OFFSET`, and under addition every bone would be twice its length. Blender's exporter writes the pose displacement plus the `OFFSET`.

**The file's values round-trip as they are.** A difference from the `OFFSET` would be recomputed on every write and change whenever an operation edits the offsets. The file's values are written back unchanged, so a write followed by a read gives an equal clip.

**A compact array leaves the 3-channel clips as they were.** Most clips have no translating joint. With P = 0 their motion, their files, their node tables and their `FkTopology` values (apart from the new field, all −1) do not change, and no array of `(F, J, 3)` positions fills with offsets that carry no information.

## Considered options

- **Add the position channels to the `OFFSET`**: rejected. Every bone of a measured six-channel file would be twice its length.
- **Store the difference from the `OFFSET`**: rejected. It is not what the file holds, and it goes stale whenever an operation edits the offsets.
- **A dense `(F, J, 3)` array, with the offset on every joint that does not translate**: rejected. It enlarges every clip for the few that translate, and it hides which joints the file declares as translating.
- **Fold the root into the same array**: rejected. The root is in the world frame and every consumer of `root_pos` reads it there; a translating joint is in its parent's frame.
- **Drop position channels whose values equal the `OFFSET` on read**: rejected. A joint that declares position channels keeps them; dropping or adding them explicitly is a separate issue.

## Consequences

`FkTopology` has five fields, all required, so code that unpacks or builds four breaks. A four-field form with the fifth all −1 would place every translating joint at its `OFFSET` with no error. `frames_to_node_positions` takes `joint_translations=`, required when the skeleton has a translating joint and the motion is passed. A file whose position channels are declared in another order than `X Y Z`, or interleaved with the rotation channels, changes on its first write only. Each operation that edits offsets or frames states what it does to `joint_translations` (#186's table of operations).
