# The body anchor is where the left/right pairs meet, and the body is found from it

ADR 0011 makes a `Bvh` one hierarchy that holds at most one body, and names the body-level operations. They need one joint to measure from, where through v0.9.0 they read node 0. On `reference_humanoid_6ch` in the synthetic corpus, which holds the body of `rotation_only_control` at the same world positions under a reference root, `skeleton_size` gave 16.09 for 93.35, and `foot_contacts` with the height method raised "feet are above hips". On `reference_props_humanoid_6ch` and `footsteps_humanoid_offset_6ch`, world up read `+z` for a body that stands along `+y`. This record says how that joint, the body anchor, is found, how the body is found from it, and what reads it. Issue #205 implements it.

## Decision

**1. The rule.** In this order:

1. An anchor that the user set is the anchor. Source `"set"`.
2. With no L/R pair in the mapping in effect, the anchor is the root. Source `"default"`.
3. Under a static root, each child of the root heads a tree. When a second tree has at least half the joints of the largest, the clip has no body. Otherwise the largest tree is kept, with the pairs whose two joints are inside it.
4. Each pair has a meeting joint: the lowest joint that both joints' chains pass through.
5. The candidates are the meeting joints at the shallowest depth.
6. One candidate is the anchor. With several, the one with the most joints below it is the anchor when each other candidate has fewer than half as many. Otherwise the clip has no body, source `"ambiguous"`. A joint chosen by steps 3 to 6 has the source `"lr_pairs"`.

Sizes count joints, the candidate included, end sites excluded. Under a static root with no body, the candidates are one per tree with at least half the joints of the largest: the tree's largest shallowest meeting joint, or the tree's head when no pair lies inside it. The rule reads the skeleton and the L/R mapping in effect, never the rest positions or the motion.

```
Reference                root                         __0  (CHANNELS 0)
└─ Hips ★                ├─ ik_foot_root  [3]         ├─ ik_foot.L
   ├─ legs               └─ pelvis ★      [21]        ├─ ik_foot.R
   └─ Spine …                                         └─ hips ★
one candidate            the larger one               the largest tree
```

The legs meet at the hips, and the arms meet deeper in the tree, at the chest, so on most rigs the anchor is the hips. The rule defines the anchor, not anatomy.

**2. The body and its climb.** The body is the anchor and every node below it. When no joint below the anchor has a head name (`head`, `neck`, `chest`, `spine`), or no joint below it has a foot name (`leg`, `foot`, `toe`, `toes`), each read as a word of the name, the body starts higher: at the closest joint above the anchor that has the missing name below it. With both missing, it starts at the higher of the two. When no joint above has the missing name, the body starts at the anchor. The anchor does not move.

```
Root            ← the body starts here: no head name is below the anchor
├─ Spine1 … Head, arms
└─ Hips ★       ← the anchor: the legs meet here
```

**3. Who reads what.**

| Reader | Reads |
|---|---|
| the world-up guess at frame 0, rest up from the head joint, the test that skips a head joint lying on the reference point | the body anchor |
| the travel descriptors: `root_trajectory`, the `centered="skeleton"` and `centered="first"` modes, the gait descriptors | the body anchor's position |
| the heading, and the check that frame 0's rotation carries rest up onto another axis (ADR 0005, decision 2) | the body anchor's world rotation: its own rotation composed with those of the joints above it |
| the head joint, the L/R pairs of rest left and rest forward, `auto_detect_feet` | joints inside the body |
| the leg-pair reading of rest up | the pairs whose two joints lie in two branches of the body anchor |
| the "feet below hips" check and the threshold scale of `foot_contacts`, `skeleton_size` | the lowest joint that has the given feet in two or more of its branches; when there is none (one foot, or feet along one leg), the body anchor |

`ground_contacts` keeps the root as its scale reference, and drawing draws the root's trail: both are hierarchy-level. `frames_to_node_positions` given a node list or an `FkTopology` centres on the root, since those carry no L/R mapping. No function is renamed: `root_trajectory` and the `centered` modes keep their names, and their docstrings say that they read the body anchor.

**4. The surface.** `Bvh.body_anchor` reads the anchor as a node of the clip, or `None` for a clip with no body. It is set with a joint's name or the node itself, through `Bvh.index` (ADR 0009): an integer, an end site, a missing or repeated name and a node of another clip are refused. Setting `None` returns to the rule, as `world_up = None` returns to the guess. `read_bvh_file(..., body_anchor="Hips")`, `Bvh.from_file` and `read_bvh_directory` set it at load, by name. `OrientationReport` gains `body_anchor`, `body_anchor_source` (`"set"`, `"lr_pairs"`, `"default"`, `"ambiguous"`) and `body_anchor_candidates`: the joints that step 6 compared, largest first, or under a static root the candidate of each tree that step 3 compared. It is empty for `"set"` and `"default"`.

**5. The lifecycle (ADR 0003).** An anchor that the user set is one of the user's settings, as a set L/R mapping is. Assigning the motion or `lr_mapping`, and pybvh's own skeleton edits, keep it. Every derived clip and a pickle carry it; `copy(nodes=)` carries it to the joint at the same position, under its new name. A derived clip that no longer holds the joint drops the setting and uses the rule. An anchor found by the rule is derived state. It is cleared when the skeleton changes or the L/R mapping is set, and it is never inherited: a derived clip finds its own. A new event, *the body anchor setting changed*, clears what is computed from the anchor and kept. The inheritance of world up does not change.

## Why

**The pairs say where a body is, and the names and the shape alone do not.** A reference root, a `Bip01` root and a `CHANNELS 0` root each sit above the body, and props and helpers sit beside it. The legs do not meet at any of them: on every biped of the synthetic corpus they meet at the hips, whatever stands above them. A pair that meets at a static root, such as two IK targets, is set apart by step 3. The pairs are already detected from the names (`lr_mapping`), and the user can set them, so the rule adds no input.

**Size settles a helper pair, and comparable sizes mean two figures.** A pair of helpers beside the body can meet at a joint of the same depth as the hips: on `helper_pair_moving_root`, that joint counts 3 joints and the hips count 22. Two figures give two candidates of the same size. One factor, half, separates the two cases, and the result is a body or an error, never a guess with a warning.

**The body is wider than the anchor's subtree on some rigs.** When the hips hang beside the spine under a root (`hips_beside_spine`), the head is not below the anchor, and the head joint must still be found. When the chest is the anchor beside the hips (`chest_and_hips_same_depth`), the feet are not below it. Climbing to the closest joint above that has the missing names below it finds them, and the anchor, and with it the direction and the travel, stays where the pairs meet.

**The sizes are read where the feet branch.** The size of a body is the distance from its feet to where they part. On `chest_and_hips_same_depth`, whose anchor is the chest, that joint is the hips, so `skeleton_size` gives 93.35, the size of `rotation_only_control`, though the anchor is not the hips.

## How it sits with ADRs 0003, 0004 and 0005

**ADR 0004, decision 1, is amended.** Rest up is read from the body anchor, not from the root: the main axis from the anchor to the first joint of the body whose name holds `head`, else `neck`, `chest` or `spine`, as a word; else the axis opposite the legs, from the far ends of two branches of the anchor that an L/R pair is split across, to the anchor; else the axis of largest spread, signed from the anchor. The walk that named the pelvis as the first node down from the root with more than one child is removed. The body anchor's world rotation, not the root rotation, is what carries a skeleton fact into the world: the heading is the anchor's world rotation applied to rest forward, also in the heading stage of the facing chain (ADR 0004, decision 2). The two frames, the facing chain and the report stand.

**ADR 0003's table gains the anchor's rows**, and the user's settings gain a set body anchor, with one more event, *the body anchor setting changed*:

| Derived value | Computed from | Cleared by |
| --- | --- | --- |
| body anchor found by the rule | skeleton, L/R mapping in effect | skeleton changed, L/R mapping setting changed |
| orientation report (ADR 0004) | as before, and the body anchor setting | as before, and body anchor setting changed |
| guessed world up, a guess of the clip's own | as before, and the body anchor | as before, and body anchor setting changed |
| scene ground | as before | as before, and body anchor setting changed |

The rule's anchor is not cleared by the setting: the setting is read ahead of the rule. An inherited world up stays when the anchor is set, as it stays when the L/R mapping is set.

**ADR 0005 stands.** The guess is the one it records, with the body anchor in place of the root: the head's direction from the anchor at frame 0, and the warning when the anchor's rotation at frame 0 carries rest up onto another axis. Only the reference point changes. On a clip whose anchor is its root, every bundled clip outside the synthetic corpus, each value is the same as before.

## Considered options

For the rule:

- **The first node with two children down from the root**, the walk of ADR 0004: rejected. It is wrong when a reference root carries props: the root itself has several children.
- **The meeting point of the pairs with the head joint**: rejected. It is silent on a helper pair or a second figure beside the body.
- **The first candidate in file order, with a warning**: rejected. File order says nothing about which candidate is the body, and the warning does not change the numbers.
- **Moving the anchor up to where the head is below it**: rejected. It can land on a root that stays on the ground, which is what the anchor replaces. The body climbs instead, and the anchor stays.
- **A factor other than "fewer than half", with a warning between two factors**: rejected. The result is a body or an error. A band in which the rule picks a candidate and warns would give numbers that may describe the wrong figure.

## Consequences

A clip whose anchor is its root gives the results it gave before, bit for bit. On a clip with a root above or beside the body, `world_up`, `rest_up`, the detected feet, the contacts, `skeleton_size`, the travel descriptors and the heading change, and `foot_contacts` stops raising "feet are above hips". The accepted limit: when the hips and the chest are candidates at the same depth, the chest side is the larger one, because of the fingers, so the chest is the anchor. Direction and travel are then measured from the chest. The body, the feet and the sizes are still found, through decision 2 and the last row of decision 3. `bvh.body_anchor = "Hips"` corrects it. A skeleton whose pairs the detection does not find takes the root, labelled `"default"`; the naming conventions that the detection does not read are #202. The CHANGELOG of v0.10.0 lists each change with its way out.
