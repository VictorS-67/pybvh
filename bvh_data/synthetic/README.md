# bvh_data/synthetic — invented clips for conventions the bundled files lack

Every file here is **synthetic**: authored by `scripts/generate_synthetic_bvh.py`, five frames each,
deterministic. Skeleton *shapes* follow conventions of publicly documented rigs and exporters (named
in the table); every offset and motion value is invented in the generator. No numerical table or frame is copied
from any third-party dataset. The generator, profiles and output are licensed under
[pybvh's MIT licence](../../LICENSE). Regenerate with
`python3 scripts/generate_synthetic_bvh.py --check` from the repository root; `manifest.json` records
per-file conventions, the expected pelvis, and SHA-256 hashes.

What they show, and why pybvh needs them: in several widely used exports a joint other than the root
declares **six channels**, three positions and three rotations. The position channels give the joint's
complete position relative to its parent, frame by frame, the same quantity its `OFFSET` gives for the
rest pose. In most joints that value never changes and equals the `OFFSET`; in a few it moves: the
pelvis under a **static reference root** carries the whole body's travel, and an animator can stretch a
bone. Some exports also put a non-body node beside or above the body (a reference point, a Footsteps
node, a prop), so **node 0 is not always the pelvis**.

Units are centimetres, Y is up, the character faces roughly +Z. Names follow the `LeftUpLeg` style of
`standard_skeleton.bvh`. The anatomical pelvis is authored per profile and recorded as
`expected_pelvis` in the manifest.
Names are stable profile-local IDs (identity mapping, no imported names to normalise).

Frame 0 is a neutral posture with a known body yaw; later frames add small authored rotations.
Default time is `0.03333333333333333`; the Blender-style profile uses `0.033333` and the
hands profile `0.016667`. There is no Blender dependency and these are not actual vendor exports.
The quadruped's head points chiefly forward: use declared `+y` for animal checks, not a claim
that a head-direction heuristic recovers gravity.

Generation uses seed 40067 with explicit per-profile seeds in the manifest. Perturbations use
SHA-256 of `seed:joint:frame:axis`, first 64 bits mapped to ±0.2 degrees (no PRNG state).
Numbers use six decimals with negative zero canonicalised; UTF-8 and LF, with no timestamps,
absolute paths, network inputs or BVH comments. Regeneration is supported on stock Python 3.9+.
`--check` audits files after writing; every generation also audits before publishing its manifest.
For read-only verification run `python3 scripts/audit_synthetic_bvh.py bvh_data/synthetic`.
Rigid non-root positions repeat OFFSET tokens verbatim; root positions remain complete world
positions. These exact-equality rules apply to this authored corpus, not arbitrary imported data.

The rotation-only control and rigid six-channel control have identical body world positions;
`reference_humanoid_6ch` adds a static ancestor without changing those positions. The far props
body differs only by `(120, 0, 600)` in every frame. The rotation-only props companion keeps
attachments fixed at their offsets and illustrates pelvis identification using rotation-only joints.
The Footsteps zero-offset humanoid and dog identify a distinct body joint at the root's location;
only the deliberately large-offset humanoid companion separates their coordinates.

As of pybvh 0.8.2, the reader rejects position channels on joints other than the root;
these files document and exercise that convention.
The two rotation-only files read; the props companion exposes root-based `world_up` choosing
`+z` although the body's head-minus-hips direction is clearly `+y`.

| file | convention illustrated | shape after | six-channel non-root joints | pelvis |
|---|---|---|---|---|
| `reference_humanoid_6ch.bvh` | static reference root; the character's travel is in the Hips' position channels; every other joint carries its OFFSET | MotionBuilder-style reference-node export (Bandai Namco dataset shape) | all 22 non-root joints | Hips |
| `footsteps_humanoid_6ch.bvh` | moving root with a Footsteps leaf beside the body; the pelvis is a joint under the root, coincident with it (zero offset, zero position) | 3ds Max Biped export shape | all 23 non-root joints | Hips |
| `footsteps_humanoid_offset_6ch.bvh` | moving root with a Footsteps leaf beside the body; the pelvis is a joint under the root; here the pelvis sits at a large nonzero offset from the root, so root-based and pelvis-based results differ (deliberately synthetic) | 3ds Max Biped export shape | all 23 non-root joints | Hips |
| `hips_hands_6ch.bvh` | the root is the pelvis; every joint declares six channels; each hand has six child joints (five fingers and a helper) | re-solved motion-capture skeleton shape (LaFAN1 re-solve / ZeroEGGS) | all 53 non-root joints | Hips |
| `quadruped_tails_6ch.bvh` | quadruped; the hips have five child joints (torso, two tail chains, two hind legs); the front-limb junction lies deeper in the torso | Biped-rigged dog capture shape (Tencent Robotics X lifelike dataset) | all 34 non-root joints | Hips |
| `blender_mixed.bvh` | Blender's default export split: bones connected to their parent are rotation-only, unconnected bones carry six channels with their rest position repeated | Blender BVH exporter output shape | Neck, LeftShoulder, RightShoulder, LeftUpLeg, RightUpLeg | Hips |
| `mixed_zxy_positions.bvh` | non-root joints declaring position channels in orders other than X Y Z, one joint interleaving position and rotation tokens, several rotation orders; a format stress case, not an exporter's habit | authored | Neck, LeftShoulder, RightShoulder, LeftUpLeg, RightUpLeg | Hips |
| `synthetic_stretch.bvh` | a non-root joint whose local position changes from frame to frame in all three components while its parent rotates; frame 2 writes a zero position for a joint whose OFFSET is not zero; a translated joint with a translated child and End Site below it | authored | LeftShoulder, LeftArm, RightShoulder | Hips |
| `rotation_only_control.bvh` | the same body and motion as reference_humanoid_6ch written the ordinary way: the root is the pelvis, its position channels carry the travel, every other joint is rotation-only; world positions of the body match that file | pybvh's bundled clips | none | Hips |
| `rigid_6ch_control.bvh` | rotation_only_control with six channels declared on every joint and each non-root joint's OFFSET repeated as its position in every frame; world positions identical to the rotation-only file | pybvh's bundled clips | all 21 non-root joints | Hips |
| `reference_props_humanoid_6ch.bvh` | static reference root with three unlike children: the travelling body (far from the origin), a two-joint prop that moves on its own, and a single marker; the root has three children but is not the pelvis | reference-node export with tracked props | all 25 non-root joints | Hips |
| `reference_props_rotation_only.bvh` | rotation-only anatomy companion: static reference, far-offset body and two unlike props; readable before non-root position support | authored reference-node attachment | none | Hips |
