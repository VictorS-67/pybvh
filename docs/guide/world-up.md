# World Up and Orientation

## What is world_up?

BVH files don't declare which axis is "up". Different software uses different conventions: Y-up is standard in game engines and Maya, Z-up in Blender and CAD tools. pybvh needs to know the up axis for:

- **Visualization** -- camera orientation, bounding box projection
- **Foot contact detection** -- which direction is "ground"
- **`rotate_vertical()`** -- which axis to rotate around
- **`root_trajectory()`** -- projecting the root path onto the ground plane

## Two frames: the skeleton's and the world's

A BVH file says which way is up in two places: the rest pose, the skeleton with every rotation zero, written in the file's coordinates, and the motion, which places that skeleton in the world frame by frame. pybvh reads each orientation fact from one of the two, and never combines an axis of one with an axis of the other:

| Frame | Read from | Facts |
|---|---|---|
| The skeleton's own frame | the rest pose alone, never `world_up` or the motion | `rest_up`, rest left, `rest_forward` |
| The world frame | the motion, against `world_up` | `world_up`, the facing (`forward_at`, `left_at`, `facing_frame`), the heading (`root_trajectory`) |

The rest pose's coordinates are not world coordinates: only the root rotation of a frame says how the skeleton stands in the world at that frame, and it is the one thing that carries a skeleton fact into the world (the heading is the root rotation applied to `rest_forward`). On a clean file the two frames agree, rest up is world up, and the distinction changes no value. On a file whose rest pose is authored in another convention than its motion, it does. The bundled `bvh_test3.bvh` animates z-up with a rest pose authored y-up:

```python
b3 = pybvh.read_bvh_file("bvh_data/bvh_test3.bvh")   # warns: rest pose and first frame disagree
b3.world_up        # '+z' — read from the motion
b3.rest_up         # '+y' — the rest pose was authored y-up
b3.rest_forward    # '+z' — the rest pose's own facing: its left '+x' × its up '+y'
b3.forward_at(0)   # '+y' — where the body faces in the world at frame 0
```

`rest_forward` lies along world up here, and that is correct: it is a direction in the skeleton's frame, and the root rotation turns it onto the ground on every frame.

## Auto-detection

When a clip is read from a file or built from arrays, `Bvh.__init__` guesses `world_up` from frame 0: the direction from the root to the head (the first joint whose name holds `head` as a word, else `neck`, `chest` or `spine`, the joint `rest_up` reads too), snapped to its main axis when its largest component is at least twice the second. When frame 0 cannot decide (no such joint, no frames, or a crouched, lying or leaning first pose), the skeleton's own up, `rest_up`, stands in: a guess that assumes the rest pose is authored upright. When the rest pose is degenerate too, pybvh warns and falls back to `'+y'`. The result is a signed axis string like `'+y'` or `'+z'`, and `bvh.orientation_report.world_up_source` says which of the three gave it (`"first_frame"`, `"rest_up"`, `"default"`; see [The orientation report](#the-orientation-report)).

```python
bvh = pybvh.read_bvh_file("walk.bvh")
bvh.world_up  # '+z' (auto-detected)
```

## The disagreement warning

Some BVH files author the rest pose in one convention and animate in another. For example, a Blender export where the rest pose skeleton points along Y but the animation plays in Z-up world space. When the first frame and the rest pose name different axes, pybvh emits a `UserWarning`, once per guess (at load, for a clip read from a file), and trusts the animation data (frame 0) over the rest pose. The clips derived from it inherit its axis (next section) and do not warn again.

In the maintainer's corpus of 23,020 clips, 75 have a rest up other than the world up guessed for them: 51 rest poses authored in another convention than their motion, and 24 clips whose first frame is not upright, so that the guess itself is wrong. If you see the warning, check whether `world_up` matches your expectation and override it if needed.

## Set, inferred or inherited

A clip's world up is in one of three states:

- **Set**: you assigned it (`bvh.world_up = '+y'`), passed `world_up=` at load, or the clip is the result of `reorient_world_up`. It stays until you assign another axis.
- **Inferred**: guessed from the clip's own first frame, as above. A clip read from a file, or built with `Bvh(...)`, `Bvh.from_df` or `df_to_bvh`, starts here.
- **Inherited**: a clip pybvh derives from another reports its source's axis without guessing from its own frames. That covers `copy()`, slicing (`bvh[a:b]`), `a + b` and `a += b` (from `a`), `resample`, every transform (`mirror`, `scale`, `rotate_vertical`, `translate_root`, the noise functions, the reorientations), `retarget`, `change_euler_order`, `extract_joints` and a clip loaded from a pickle. A set axis travels the same way: the derived clip has it set too.

Cutting, resampling or transforming a clip does not change which way gravity points, but a fresh guess on the derived clip could: a slice whose first frame is a crouch, a fall or a lying pose is ambiguous on its own and would fall back to the rest pose, where its source was not. Inheriting the answer makes every derived clip agree with its source, and makes the guess on the source the only one.

```python
bvh = pybvh.read_bvh_file("walk.bvh")   # inferred from frame 0: '+z'
clip = bvh[200:260]                       # inherited: '+z', whatever frame 200 shows
clip.orientation_report.world_up_inherited   # True; world_up_source is the source's, 'first_frame'
clip.world_up_inferred                    # what a guess from clip's own frames would give
```

The report tells the three states apart: `world_up_source` is `"set"` for a set axis and otherwise names what the guess was read from, and `world_up_inherited` is `True` for a guess made on the clip this one was derived from. `world_up_inferred` always runs the heuristic on the clip's own motion and changes nothing. To have a derived clip use its own guess instead, assign it: `clip.world_up = clip.world_up_inferred` (the axis is then set).

### When the guess starts again

Only when the recording itself is replaced: assigning `root_pos`, `joint_angles` or frames (`bvh[i] = other`), or loading new motion with `from_rotmat`, `from_6d`, `from_quat` or `from_axisangle`. The clip then guesses again from its new first frame, whether its axis was inferred or inherited, and the clips derived from it afterwards inherit the new guess. None of these clears a set axis.

## Manual override

```python
bvh.world_up = '+y'  # override auto-detection
```

Validated input -- only accepts `{+x, -x, +y, -y, +z, -z}`. Anything else raises `ValueError`.

The override is a set axis, so it propagates to every clip derived from this one, `retarget()` included: new bone lengths do not change which way is up. Assign `"auto"` (or `None`) to clear the override and return to the guessed axis, inferred or inherited. A pickle keeps the override, but BVH files have no world-up field, so manual overrides are lost on write/read round trips and must be re-applied.

## The orientation report

`world_up`, `rest_up` and `rest_forward` always answer, a default included, so their value alone cannot say whether it was measured. `bvh.orientation_report` can: it returns a frozen `OrientationReport` holding each clip-level orientation fact beside how it was obtained.

```python
report = b3.orientation_report
# OrientationReport(rest_up='+y', rest_up_source='named_joint', rest_left='+x',
#                   rest_forward='+z', rest_forward_source='lr_geometry',
#                   world_up='+z', world_up_source='first_frame',
#                   world_up_inherited=False, lr_mapping_source='names')
```

| Field | Values | Says |
|---|---|---|
| `rest_up_source` | `"named_joint"`, `"leg_pair"`, `"largest_spread"`, `None` | rest up read from a joint named for the head, neck, chest or spine; else opposite the legs, the L/R pair split across the pelvis's branches; else from the axis along which the rest pose spreads most; `None` when the rest pose carries no direction |
| `rest_left` | a signed axis or `None` | the skeleton's own left (next sections) |
| `rest_forward_source` | `"lr_geometry"`, `"default"` | rest forward measured as rest left × rest up, or the house default for a rest pose without usable L/R geometry |
| `world_up_source` | `"set"`, `"first_frame"`, `"rest_up"`, `"default"` | world up set by you, guessed from frame 0, guessed from rest up because frame 0 could not decide, or the house `'+y'` |
| `world_up_inherited` | `bool` | the guess was made on the clip this one was derived from |
| `lr_mapping_source` | `"set"`, `"names"`, `None` | the L/R mapping set by you, detected from joint names, or absent |

Reading the sources is how a pipeline tells a measurement from a default without parsing warnings:

```python
report = bvh.orientation_report
if report.world_up_source in ("rest_up", "default"):
    ...  # frame 0 could not decide world up: check it, or set it
if report.rest_forward_source == "default":
    ...  # no usable L/R geometry: rest_forward and the heading are a house default
```

Reading the report never warns. A warning about a default fires when the defaulted fact itself is read (`world_up`, once per guess; `rest_forward`; the facing), and points to the report. The report is computed once and kept until the skeleton changes, `world_up` or `lr_mapping` is assigned, or the motion is assigned. Facts that change from frame to frame are not in it: `forward_at`, `left_at` and `facing_frame` carry them, and `facing_frame().valid` marks the frames whose facing could not be measured.

## forward_at(frame)

Returns the character's world-space facing direction at a given frame as a signed axis string:

```python
bvh.forward_at(0)   # '-z' (facing direction at frame 0)
bvh.forward_at(60)  # '+x' (character has turned by frame 60)
```

Derived from actual joint positions: L/R joint pairs are averaged in world space to find the leftward direction, measured against `world_up`, and forward is that leftward crossed with `world_up`. This tracks the character's real facing as they rotate through the animation, not the rest pose.

Used internally by bvhplot's camera auto-orientation and follow mode.

`forward_at()` reads the L/R pair list from `bvh.lr_mapping`. If your skeleton's joint names don't follow the expected conventions (`Left*`/`Right*`, `.L`/`.R`, `_l`/`_r`, `mixamorig:` namespace, `.001` numbered duplicates), `bvh.lr_mapping` will be `None` (`orientation_report.lr_mapping_source` is `None`), and `forward_at()` falls back to a house default, an arbitrary-but-stable horizontal axis for `world_up`, with a warning. To get a meaningful forward, set the mapping explicitly — see the [Augmentation guide](augmentation.md#lr-pair-detection) for the override patterns.

## left_at(frame)

Returns the character's world-space leftward direction at a given frame as a signed axis string:

```python
bvh.left_at(0)   # '-x' — character's left-hand direction at frame 0
```

Together with `world_up` and `forward_at(frame)`, `left_at(frame)` completes the character's local frame as an orthonormal right-hand-rule triple:

```
left = world_up × forward        (equivalently, forward = left × world_up)
```

Positive step along `left_at` moves from the character's right side toward their left (right-shoulder → left-shoulder direction). This matches the standard rigging convention used in Blender, Maya, and Unity.

The triple drawn on a pose:

![The orientation triple drawn as three arrows on a skeleton: world_up in blue, forward_at in red, left_at in green](../gallery/img/orientation-triad.png)

`left_at()` shares the L/R pair machinery with `forward_at()`, so the same `bvh.lr_mapping` caveats apply. The two always form a right-handed triple with `world_up`, on every frame: `left_at` never returns world up.

### coords= (skip per-call FK)

Both `forward_at()` and `left_at()` accept a pre-computed `(F, N, 3)` spatial-coordinates array via `coords=` to skip the per-call forward kinematics. Useful when you need orientation axes across many frames:

```python
coords = bvh.node_positions()    # one FK pass
forwards = [bvh.forward_at(f, coords=coords) for f in range(bvh.frame_count)]
lefts    = [bvh.left_at(f,    coords=coords) for f in range(bvh.frame_count)]
```

`coords` holds the clip's frames, one row per frame: a frame whose leftward cannot be read falls back to the root's heading at that frame (next section).

## facing_frame() — the continuous basis

`forward_at()` / `left_at()` snap the facing direction to the nearest of the six signed world axes — useful for canonicalization checks, but lossy: through a gentle turn the label stays constant while the true facing rotates. `bvh.facing_frame()` returns the same construction *before* the snap, as a `FacingFrame(forward, left, up, valid)` named tuple: three `(F, 3)` unit-vector arrays — a per-frame orthonormal right-handed basis (`forward × left = up`) — plus a `(F,)` bool array that is `False` on frames whose basis is a fallback rather than a measurement:

```python
frame = bvh.facing_frame()        # or bvh.facing_frame(coords=coords), one row per clip frame
frame.forward[120]                # continuous facing vector at frame 120
frame.valid[120]                  # False if that vector is a fallback
```

This is a yaw-only, gravity-aligned *facing* frame: `up` is always exactly `world_up`, so the basis only rotates about the vertical — it is deliberately not a pelvis orientation (no roll/pitch), and distinct from `root_trajectory()`'s heading, which is a second facing estimate built differently (the root bone's rotation applied to the rest forward — it inherits pelvis twist and is not a direction of travel). See [`analysis.facing_frame`](../api/analysis.md#pybvh.analysis.facing_frame) for the full policy.

### When a frame's facing cannot be measured

One chain gives every frame its leftward, measured against `world_up`, and the facing is built from it:

1. the frame's own leftward, from its L/R pairs;
2. where that cannot be read (the L/R pairs line up with world up: the body lies on its side), the root's heading at that frame turned a quarter turn about up, when `rest_forward` is measured and the heading can be read;
3. otherwise the house default, an arbitrary-but-stable horizontal axis for `world_up`.

Forward is always left × up, so the basis is right-handed with world up on every frame, fallback frames included, and `valid` is `False` on the frames of steps 2 and 3. A frame that falls to the house default warns, naming the frame; on a skeleton with no L/R pairs at all every frame does, and the warning names the skeleton.

## Rest-pose axes: rest_up, rest left and rest_forward

`world_up` and `forward_at(frame)` are read from the motion. `rest_up`, rest left and `rest_forward` are the skeleton's own frame: read from the rest pose alone, never from `world_up` or the animation:

```python
bvh.rest_up                         # '+y' — the skeleton's own up
bvh.orientation_report.rest_left    # '+x' — the skeleton's own left
bvh.rest_forward                    # '+z' — the direction the rest pose faces: rest left × rest up
```

- **`rest_up`** is the main axis from the root to the first joint whose name holds `head` as a word, else `neck`, `chest` or `spine`, in any case and after a `foo:` namespace or a `.001` suffix: `Head`, `mixamorig:Head`, `Bip01 Head`, `CC_Base_Head` and `J_Bip_C_Head` are heads, `Forehead` is not. Without one, it is the axis opposite the legs: the first L/R pair whose two joints lie in two different branches of the pelvis, the first node down from the root with more than one child (the root itself on a usual rig, a `Hips` below a `Root`), read from the pelvis to the far ends of those two branches, the feet. A mapping of the arms alone, both below the spine, names no legs. Without such a pair either, it is the axis along which the rest pose spreads most, a weak reading: a T-pose wider than tall spreads most along its arms, and hips mid-body put as much of the body below the root as above. `rest_up_source` says which (`"named_joint"`, `"leg_pair"`, `"largest_spread"`).
- **Rest left** averages the left-minus-right rest positions of the L/R pairs, removes the `rest_up` component and snaps the rest to an axis. `mirror` reflects along it. It is in the report only.
- **`rest_forward`** is rest left × rest up. Without usable L/R geometry (no pairs, or pairs that lie along rest up) it is the house default for rest up (`'+y'` gives `'+z'`, `'+z'` gives `'+x'`, `'+x'` gives `'+y'`), reading it warns, `has_lr_geometry` is `False` and `rest_forward_source` is `"default"`.

On a clean BVH file, `rest_up == world_up` and `rest_forward` matches `forward_at(0)` when the root rotation at frame 0 is identity. When they disagree:

- `rest_up != world_up` — the file was authored with the rest pose in one up convention and animated in another (the case that triggers the disagreement warning; fix with `reorient_rest_up`). `rest_forward` is then a direction in the rest pose's convention and may lie along world up, as on `bvh_test3.bvh` above.
- `rest_forward != forward_at(0)` — the character simply starts the animation facing a different direction from their rest pose. This is normal; `forward_at(0)` is what the viewer sees.

All three are read-only. To actually change `rest_up` or `rest_forward`, call the corresponding `reorient_*` method (see "Reorienting data" below) which rotates the rest-pose offsets and compensates joint rotations so FK positions are unchanged.

## The heading

`root_trajectory()` reports the root's heading as a `(sin, cos)` pair: the root rotation of each frame applied to `rest_forward`, projected on the ground plane of `world_up`. The root rotation is what carries the skeleton's own forward into the world, so the heading follows the body whichever convention the rest pose was authored in: on `bvh_test3.bvh` it lies a median 2.3° from the measured facing (`facing_frame().forward`).

On a frame where the root's forward stands along world up, its projection on the ground is too short to have a direction, and the heading is `NaN` there rather than an arbitrary angle:

```python
import numpy as np
from pybvh import rotations

bvh = pybvh.read_bvh_file("bvh_data/bvh_test1.bvh")   # world up '+z', rest forward '+y'
bvh.world_up = "+z"   # set it: assigning the motion below would guess world up again, from the tilted frame
tilt = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])   # turns '+y' onto '+z'
angles = bvh.joint_angles.copy()
angles[:, 0] = rotations.rotmat_to_euler(
    np.broadcast_to(tilt, (bvh.frame_count, 3, 3)), bvh.nodes[0].rot_channels)
bvh.joint_angles = angles
bvh.root_trajectory()[:3, 2:4]   # NaN: the root faces straight up
```

When `rest_forward` is the house default (no usable L/R geometry; reading it warns), `root_trajectory()` still builds the heading on it, and both the heading and its rate of turn (`heading_vel`) depend on that guessed reference whenever the root pitches or rolls: a tilted root's forward projects on the ground at an angle, and turns there at a rate, that differ with the forward it starts from. On a defaulted `rest_forward` they are reliable only for a root that turns about world up alone. The facing chain takes no heading from such a `rest_forward` (step 2 above), so a facing never offers a guessed direction as a measured one.

With `include_velocities=True`, the `NaN` reaches only the `heading_vel` samples whose stencil reads that frame (with the default central stencil, the frames either side). Code that feeds the columns to a model or a sum masks the rows (`np.isnan(traj[:, 2])`) or fills them. For a facing that is never `NaN`, read `facing_frame()`, whose `valid` flags its fallback frames.

## mirror works in the skeleton's frame

`mirror` reflects the skeleton's offsets across its own sagittal plane, along rest left, and the motion follows by the same reflection. It never reads `world_up`, so the result is a valid mirror whatever world up says, a wrong guess included. Pass `lateral='x'` (or `'y'`, `'z'`) to choose the axis yourself.

## Negative up axes

`'-y'` means "more negative Y = higher". This is uncommon but valid. It affects:

- **`rotate_vertical()`** -- angle direction is inverted
- **`foot_contacts()`** -- ground is at the positive end of the axis
- **Camera orientation** -- flipped vertical in visualization

All of these handle the sign correctly.

## Reorienting data

The `world_up` setter only changes metadata.  To actually **rewrite the data** so a file uses a different coordinate system or rest-pose orientation:

```python
# Change the world coordinate system (Z-up -> Y-up)
bvh_y = bvh.reorient_world_up('+y')  # character looks identical, coords change

# Fix a rest-pose / animation disagreement
bvh_fixed = bvh.reorient_rest_up('+y')  # FK positions unchanged, rest pose rotated

# Change the rest-pose forward direction
bvh_fwd = bvh.reorient_rest_forward('+z')  # FK positions unchanged
```

`reorient_rest_forward` turns the rest pose about `rest_up`, in the skeleton's own frame: `rest_up` is the same afterwards and `rest_forward` is the target, which is named in the rest pose's frame and must not be parallel to `rest_up`. All three are restricted to axis-aligned rotations (multiples of 90 degrees) for lossless transformation.

## Quick reference

```python
# Load with explicit world_up (skips auto-detection and warning)
bvh = pybvh.read_bvh_file("walk.bvh", world_up="+y")

# Or auto-detect (default)
bvh = pybvh.read_bvh_file("walk.bvh")
bvh.world_up           # '+z' (auto-detected)
bvh.forward_at(0)      # '-z' (facing direction at frame 0)
bvh.world_up = '+y'    # manual metadata override

# How each orientation fact was obtained
bvh.orientation_report.world_up_source      # 'set', 'first_frame', 'rest_up' or 'default'
bvh.orientation_report.rest_forward_source  # 'lr_geometry' or 'default'

# Suppress disagreement warning for bulk processing
clips = pybvh.read_bvh_directory("dataset/", warn_on_world_up_disagreement=False)
```

!!! info "See also"
    [Core Concepts](core-concepts.md) — the index spaces and `centered` modes this page's conventions feed into · [Transforms API](../api/transforms.md) — the reorientation functions at array level · [Gallery](../gallery/index.md) — the orientation triple and reorientation drawn (section 2)
