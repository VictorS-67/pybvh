#!/usr/bin/env python3
"""Generate pybvh's synthetic BVH corpus into ``bvh_data/synthetic/``.

Every clip here is invented: the skeleton shapes follow conventions seen in
publicly documented rigs and exporters (a static reference root above the
body, a Biped-style root with a Footsteps node, an all-six-channel humanoid
with detailed hands, a quadruped with tail chains, Blender's 3/6-channel
split), but every offset and motion value is authored in this file.
No number is copied from a third-party dataset. The corpus exists so that
pybvh has small, redistributable examples of BVH conventions its bundled
clips do not show — above all position channels on joints other than the
root, and roots that are not the character's pelvis.

The generator depends on the standard library only and never imports
pybvh, so it can generate examples independently of the reader's supported
channel layouts. Every
value is analytic: small sinusoidal rotations after a clean first pose,
plus SHA-256-keyed bounded perturbations. Runs produce identical bytes
on the verified Python 3.9/3.13 environments (six-decimal quantisation).
Generator, profiles and output are licensed under the repository MIT terms.

Run from the repository root::

    python3 scripts/generate_synthetic_bvh.py            # write the corpus
    python3 scripts/generate_synthetic_bvh.py --check    # write, then audit

``--check`` re-parses every written file with an independent minimal parser
and verifies: one ROOT, balanced braces, motion width equal to the declared
channel count on every row, exactly five frames, rigid six-channel joints
carrying their OFFSET verbatim in every frame, the root's position order,
and that the recorded anatomical pelvis names a joint of the file.

Outputs: ``<name>.bvh`` per profile, ``manifest.json`` (provenance,
conventions, expected pelvis, SHA-256 per file) and ``README.md``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

from audit_synthetic_bvh import audit

GENERATOR_VERSION = "2.1.0"
SEED = 40067
FRAMES = 5
DEFAULT_FRAME_TIME = "0.03333333333333333"

POS = ("Xposition", "Yposition", "Zposition")
ROT_ZXY = ("Zrotation", "Xrotation", "Yrotation")
ROT_XYZ = ("Xrotation", "Yrotation", "Zrotation")
ROT_ZYX = ("Zrotation", "Yrotation", "Xrotation")
ROT_YXZ = ("Yrotation", "Xrotation", "Zrotation")


# --------------------------------------------------------------------------
# Skeleton description
# --------------------------------------------------------------------------

@dataclass
class Joint:
    name: str
    offset: tuple[float, float, float]
    channels: tuple[str, ...] = POS + ROT_ZXY
    children: list["Joint"] = field(default_factory=list)
    end_site: tuple[float, float, float] | None = None

    def add(self, child: "Joint") -> "Joint":
        self.children.append(child)
        return child

    def walk(self):
        yield self
        for c in self.children:
            yield from c.walk()

    def find(self, name: str) -> "Joint":
        for j in self.walk():
            if j.name == name:
                return j
        raise KeyError(name)

    def has_positions(self) -> bool:
        return any(c.endswith("position") for c in self.channels)


def chain(parent: Joint, names: list[str], offsets: list[tuple], channels, end_site):
    """Attach a chain of joints under ``parent``; return the last joint."""
    node = parent
    for name, off in zip(names, offsets):
        node = node.add(Joint(name, off, channels))
    node.end_site = end_site
    return node


def set_layout(root: Joint, policy) -> None:
    """Assign channel layouts: ``policy(joint, is_root) -> tuple``."""
    for j in root.walk():
        j.channels = policy(j, j is root)


# --------------------------------------------------------------------------
# Body builders (centimetres, Y up, character faces +Z)
# --------------------------------------------------------------------------

def humanoid_body(hips: Joint, hands: str = "simple") -> Joint:
    """Torso, arms and legs under ``hips`` (offsets in cm, Y up).

    ``hands``: ``"simple"`` gives one hand joint with an End Site;
    ``"fingers"`` gives five three-joint fingers plus a helper leaf, so each
    hand has six child joints.
    """
    spine = hips.add(Joint("Spine", (0.0, 10.0, -1.0)))
    spine1 = spine.add(Joint("Spine1", (0.0, 12.0, 0.5)))
    chest = spine1.add(Joint("Chest", (0.0, 14.0, 0.5)))
    neck = chest.add(Joint("Neck", (0.0, 14.0, 0.0)))
    head = neck.add(Joint("Head", (0.0, 10.0, 1.0)))
    head.end_site = (0.0, 12.0, 0.0)
    for side, sx in (("Left", 1.0), ("Right", -1.0)):
        sh = chest.add(Joint(f"{side}Shoulder", (sx * 8.0, 12.0, 0.0)))
        arm = sh.add(Joint(f"{side}Arm", (sx * 14.0, 0.0, 0.0)))
        fore = arm.add(Joint(f"{side}ForeArm", (sx * 27.0, 0.0, 0.0)))
        hand = fore.add(Joint(f"{side}Hand", (sx * 25.0, 0.0, 0.0)))
        if hands == "simple":
            hand.end_site = (sx * 10.0, 0.0, 0.0)
        else:
            fingers = [("Thumb", (sx * 3.0, -1.0, 3.0), (sx * 3.0, 0.0, 2.0)),
                       ("Index", (sx * 9.0, 0.0, 2.5), (sx * 4.0, 0.0, 0.0)),
                       ("Middle", (sx * 9.0, 0.0, 0.5), (sx * 4.5, 0.0, 0.0)),
                       ("Ring", (sx * 8.5, 0.0, -1.5), (sx * 4.0, 0.0, 0.0)),
                       ("Pinky", (sx * 8.0, 0.0, -3.5), (sx * 3.0, 0.0, 0.0))]
            for fname, base, seg in fingers:
                chain(hand, [f"{side}{fname}{k}" for k in (1, 2, 3)],
                      [base, seg, seg], hand.channels, seg)
            helper = hand.add(Joint(f"{side}HandHelper", (sx * 2.0, -3.0, 0.0)))
            helper.end_site = (0.0, -2.0, 0.0)
        upleg = hips.add(Joint(f"{side}UpLeg", (sx * 9.0, -3.0, 0.0)))
        leg = upleg.add(Joint(f"{side}Leg", (0.0, -42.0, 0.0)))
        foot = leg.add(Joint(f"{side}Foot", (0.0, -41.0, 0.0)))
        toe = foot.add(Joint(f"{side}ToeBase", (0.0, -6.0, 13.0)))
        toe.end_site = (0.0, 0.0, 7.0)
    return hips


def quadruped_body(hips: Joint) -> Joint:
    """Dog-like body under ``hips``: torso to a front-limb junction, two tail
    chains, two hind legs (five child joints on the hips)."""
    spine = hips.add(Joint("Spine", (0.0, 3.0, 12.0)))
    spine1 = spine.add(Joint("Spine1", (0.0, 2.0, 12.0)))
    spine2 = spine1.add(Joint("Spine2", (0.0, 1.0, 12.0)))
    neck = spine2.add(Joint("Neck", (0.0, 6.0, 8.0)))
    neck1 = neck.add(Joint("Neck1", (0.0, 5.0, 6.0)))
    head = neck1.add(Joint("Head", (0.0, 4.0, 6.0)))
    head.end_site = (0.0, 0.0, 12.0)
    for side, sx in (("Left", 1.0), ("Right", -1.0)):
        sh = spine2.add(Joint(f"{side}Shoulder", (sx * 6.0, -4.0, 4.0)))
        arm = sh.add(Joint(f"{side}Arm", (sx * 2.0, -16.0, 0.0)))
        fore = arm.add(Joint(f"{side}ForeArm", (0.0, -18.0, -1.0)))
        hand = fore.add(Joint(f"{side}Hand", (0.0, -12.0, 2.0)))
        hand.end_site = (0.0, -3.0, 5.0)
    chain(hips, [f"Tail{k}" for k in range(1, 6)],
          [(0.0, 2.0, -8.0)] + [(0.0, -1.0, -7.0)] * 4, hips.channels, (0.0, -1.0, -6.0))
    chain(hips, [f"TailGuide{k}" for k in range(1, 6)],
          [(0.0, -1.0, -7.0)] + [(0.0, -1.5, -6.5)] * 4, hips.channels, (0.0, -1.0, -5.0))
    for side, sx in (("Left", 1.0), ("Right", -1.0)):
        upleg = hips.add(Joint(f"{side}UpLeg", (sx * 7.0, -3.0, -3.0)))
        leg = upleg.add(Joint(f"{side}Leg", (0.0, -22.0, -6.0)))
        foot = leg.add(Joint(f"{side}Foot", (0.0, -18.0, 4.0)))
        toe = foot.add(Joint(f"{side}Toe", (0.0, -9.0, 3.0)))
        toe.end_site = (0.0, -2.0, 6.0)
    return hips


# --------------------------------------------------------------------------
# Motion: analytic, small, deterministic
# --------------------------------------------------------------------------

class Motion:
    """Per-joint rotations (degrees, X Y Z about the joint's local axes) and
    optional local-position overrides, for frames 0..FRAMES-1."""

    def __init__(self, seed: int, jitter_deg: float = 0.2):
        self.seed = seed
        self.jitter = jitter_deg
        self.pos_override: dict[str, callable] = {}
        self.rot_override: dict[str, callable] = {}
        self._jit: dict[tuple, float] = {}

    def _j(self, key) -> float:
        if key not in self._jit:
            # A stable integer hash makes each sample independent of traversal
            # order, other profiles, and Python's random/hash implementations.
            token = ":".join(str(x) for x in (self.seed,) + key).encode("ascii")
            unit = int.from_bytes(hashlib.sha256(token).digest()[:8], "big") / (2**64 - 1)
            self._jit[key] = self.jitter * (2 * unit - 1)
        return self._jit[key]

    def rotation(self, joint: Joint, t: int, is_root: bool) -> tuple[float, float, float]:
        if joint.name in self.rot_override:
            return self.rot_override[joint.name](t)
        if t == 0:
            return (0.0, 12.0 if is_root else 0.0, 0.0)
        ph = 2.0 * math.pi * t / (FRAMES - 1)
        n = joint.name
        rx = ry = rz = 0.0
        if is_root:
            ry = 12.0                      # the character does not face exactly +Z
            rx = 1.5 * math.sin(ph)
        elif "UpLeg" in n:
            rx = (18.0 if n.startswith("Left") else -18.0) * math.sin(ph)
        elif n.endswith("Leg"):
            rx = 8.0 + 6.0 * math.sin(ph + (0.0 if n.startswith("Left") else math.pi))
        elif "Foot" in n:
            rx = 4.0 * math.cos(ph)
        elif "Arm" in n and "Fore" not in n:
            rx = (-14.0 if n.startswith("Left") else 14.0) * math.sin(ph)
            rz = 6.0 if n.startswith("Left") else -6.0
        elif "ForeArm" in n:
            rx = -10.0 - 5.0 * math.sin(ph)
        elif n.startswith("Spine") or n == "Chest":
            ry = 3.0 * math.sin(ph)
        elif n.startswith("Neck") or n == "Head":
            rx = 2.0 * math.cos(ph)
        elif "Tail" in n:
            ry = 9.0 * math.sin(ph + 0.4 * int(n[-1]))
        elif any(f in n for f in ("Thumb", "Index", "Middle", "Ring", "Pinky")):
            rz = (10.0 if n.startswith("Left") else -10.0) + 2.0 * math.sin(ph)
        return (rx + self._j((n, t, "x")), ry + self._j((n, t, "y")), rz + self._j((n, t, "z")))

    def position(self, joint: Joint, t: int) -> tuple[float, float, float]:
        """Local position written to the file. Default: the OFFSET itself
        (a rigid joint); overrides supply travel or stretch."""
        if joint.name in self.pos_override:
            return self.pos_override[joint.name](t)
        return joint.offset


def travel(base: tuple[float, float, float], step: float = 6.0, bob: float = 1.2):
    """Walking-style travel along +Z with a small vertical bob."""
    def f(t: int):
        ph = 2.0 * math.pi * t / (FRAMES - 1)
        return (base[0], base[1] + bob * math.sin(ph), base[2] + step * t)
    return f


# --------------------------------------------------------------------------
# Profiles
# --------------------------------------------------------------------------

@dataclass
class Profile:
    name: str
    root: Joint
    motion: Motion
    frame_time: str
    up_axis: str
    describes: str            # the convention the profile illustrates
    shape_after: str          # rig family whose *shape* inspired it
    expected_pelvis: str      # anatomical answer, by joint name
    notes: str = ""


def all_six(j: Joint, is_root: bool) -> tuple:
    return POS + ROT_ZXY


def root_only(j: Joint, is_root: bool) -> tuple:
    return POS + ROT_ZXY if is_root else ROT_ZXY


def profiles() -> list[Profile]:
    out: list[Profile] = []

    # 1. Static reference root, travelling Hips (Bandai-style).
    ref = Joint("Reference", (0.0, 0.0, 0.0))
    hips = ref.add(Joint("Hips", (0.0, 95.0, 0.0)))
    humanoid_body(hips)
    set_layout(ref, all_six)
    m = Motion(SEED + 1)
    m.rot_override["Reference"] = lambda t: (0.0, 0.0, 0.0)
    m.pos_override["Reference"] = lambda t: (0.0, 0.0, 0.0)
    m.pos_override["Hips"] = travel((0.0, 95.0, 0.0))
    m.rot_override["Hips"] = lambda t: (1.5 * math.sin(2 * math.pi * t / 4), 12.0, 0.0)
    out.append(Profile("reference_humanoid_6ch", ref, m, DEFAULT_FRAME_TIME, "Y",
                       "static reference root; the character's travel is in the Hips' position channels; every other joint carries its OFFSET",
                       "MotionBuilder-style reference-node export (Bandai Namco dataset shape)", "Hips"))

    # 2. Biped-style moving root with a Footsteps leaf and a zero-offset pelvis.
    def biped(hips_offset: tuple, prof_name: str, note: str):
        root = Joint("Bip01", (0.0, 0.0, 0.0))
        steps = root.add(Joint("Footsteps", (0.0, -95.0, 0.0)))
        steps.end_site = (0.0, 0.0, 10.0)
        h = root.add(Joint("Hips", hips_offset))
        humanoid_body(h)
        set_layout(root, all_six)
        mm = Motion(SEED + 2)
        mm.pos_override["Bip01"] = travel((0.0, 95.0, 0.0))
        mm.rot_override["Bip01"] = lambda t: (0.0, 12.0 + 2.0 * math.sin(math.pi * t / 4), 0.0)
        mm.rot_override["Hips"] = lambda t: (1.5 * math.sin(2 * math.pi * t / 4), 0.0, 0.0)
        return Profile(prof_name, root, mm, DEFAULT_FRAME_TIME, "Y",
                       "moving root with a Footsteps leaf beside the body; the pelvis is a joint under the root" + note,
                       "3ds Max Biped export shape", "Hips")
    out.append(biped((0.0, 0.0, 0.0), "footsteps_humanoid_6ch",
                     ", coincident with it (zero offset, zero position)"))
    out.append(biped((0.0, 40.0, 400.0), "footsteps_humanoid_offset_6ch",
                     "; here the pelvis sits at a large nonzero offset from the root, so root-based and pelvis-based results differ (deliberately synthetic)"))

    # 3. Hips-root humanoid, every joint six-channel, detailed hands.
    hips = Joint("Hips", (0.0, 0.0, 0.0))
    humanoid_body(hips, hands="fingers")
    set_layout(hips, all_six)
    m = Motion(SEED + 3)
    m.pos_override["Hips"] = travel((0.0, 95.0, 0.0), step=4.0)
    out.append(Profile("hips_hands_6ch", hips, m, "0.016667", "Y",
                       "the root is the pelvis; every joint declares six channels; each hand has six child joints (five fingers and a helper)",
                       "re-solved motion-capture skeleton shape (LaFAN1 re-solve / ZeroEGGS)", "Hips",
                       "60 fps"))

    # 4. Quadruped: moving root, Footsteps sibling, five-child hips with two tail chains.
    root = Joint("Root", (0.0, 0.0, 0.0))
    steps = root.add(Joint("Footsteps", (0.0, -55.0, 0.0)))
    steps.end_site = (0.0, 0.0, 8.0)
    hips = root.add(Joint("Hips", (0.0, 0.0, 0.0)))
    quadruped_body(hips)
    set_layout(root, all_six)
    m = Motion(SEED + 4)
    m.pos_override["Root"] = travel((0.0, 55.0, 0.0), step=9.0, bob=1.5)
    m.rot_override["Root"] = lambda t: (0.0, -8.0, 0.0)
    out.append(Profile("quadruped_tails_6ch", root, m, DEFAULT_FRAME_TIME, "Y",
                       "quadruped; the hips have five child joints (torso, two tail chains, two hind legs); the front-limb junction lies deeper in the torso",
                       "Biped-rigged dog capture shape (Tencent Robotics X lifelike dataset)", "Hips"))

    # 5. Blender-style: connected bones rotation-only, unconnected bones six-channel.
    unconnected = {"LeftUpLeg", "RightUpLeg", "LeftShoulder", "RightShoulder", "Neck"}
    hips = Joint("Hips", (0.0, 0.0, 0.0))
    humanoid_body(hips)
    set_layout(hips, lambda j, r: POS + ROT_ZXY if (r or j.name in unconnected) else ROT_ZXY)
    m = Motion(SEED + 5)
    m.pos_override["Hips"] = travel((0.0, 95.0, 0.0))
    out.append(Profile("blender_mixed", hips, m, "0.033333", "Y",
                       "Blender's default export split: bones connected to their parent are rotation-only, unconnected bones carry six channels with their rest position repeated",
                       "Blender BVH exporter output shape", "Hips"))

    # 6. Non-root position orders other than XYZ, one interleaved joint.
    hips = Joint("Hips", (0.0, 0.0, 0.0))
    humanoid_body(hips)
    def zxy_layout(j: Joint, r: bool):
        if r:
            return POS + ROT_ZXY
        if j.name == "LeftUpLeg":
            return ("Zposition", "Xposition", "Yposition") + ROT_XYZ
        if j.name == "RightUpLeg":
            return ("Yposition", "Zposition", "Xposition") + ROT_ZYX
        if j.name == "Neck":
            return ("Zrotation", "Xposition", "Xrotation", "Yposition", "Yrotation", "Zposition")
        if j.name in ("LeftShoulder", "RightShoulder"):
            return POS + ROT_YXZ
        return ROT_ZXY
    set_layout(hips, zxy_layout)
    m = Motion(SEED + 6)
    m.pos_override["Hips"] = travel((0.0, 95.0, 0.0))
    out.append(Profile("mixed_zxy_positions", hips, m, DEFAULT_FRAME_TIME, "Y",
                       "non-root joints declaring position channels in orders other than X Y Z, one joint interleaving position and rotation tokens, several rotation orders; a format stress case, not an exporter's habit",
                       "authored", "Hips", "artificial layout"))

    # 7. Explicit bone stretch: a shoulder whose local position changes every frame.
    hips = Joint("Hips", (0.0, 0.0, 0.0))
    humanoid_body(hips)
    six = {"LeftShoulder", "LeftArm", "RightShoulder"}
    set_layout(hips, lambda j, r: POS + ROT_ZXY if (r or j.name in six) else ROT_ZXY)
    m = Motion(SEED + 7)
    m.pos_override["Hips"] = lambda t: (0.0, 95.0, 0.0)
    m.rot_override["Hips"] = lambda t: (0.0, 30.0 * t / (FRAMES - 1), 0.0)   # parent turns
    ls = hips.find("LeftShoulder").offset
    def stretch(t: int):
        # frame 0: at rest; frame 2: file position exactly zero (displacement = -OFFSET);
        # other frames: displaced in all three components.
        if t == 2:
            return (0.0, 0.0, 0.0)
        return (ls[0] + 3.0 * t, ls[1] - 1.5 * t, ls[2] + 2.0 * t)
    m.pos_override["LeftShoulder"] = stretch
    # Keep a known yawing parent and a straight descendant chain for arithmetic
    # checks at the zero-position frame and the fully stretched frame.
    for name in ("Spine", "Spine1", "Chest", "Neck", "Head",
                 "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand"):
        m.rot_override[name] = lambda t: (0.0, 0.0, 0.0)
    out.append(Profile("synthetic_stretch", hips, m, DEFAULT_FRAME_TIME, "Y",
                       "a non-root joint whose local position changes from frame to frame in all three components while its parent rotates; frame 2 writes a zero position for a joint whose OFFSET is not zero; a translated joint with a translated child and End Site below it",
                       "authored", "Hips", "artificial motion: no measured file in hand shows bone stretch; the format allows it"))

    # 8. Rotation-only control with the same body and motion as profile 1.
    hips = Joint("Hips", (0.0, 0.0, 0.0))
    humanoid_body(hips)
    set_layout(hips, root_only)
    m = Motion(SEED + 1)  # same seed as profile 1: identical joint rotations
    m.pos_override["Hips"] = travel((0.0, 95.0, 0.0))
    m.rot_override["Hips"] = lambda t: (1.5 * math.sin(2 * math.pi * t / 4), 12.0, 0.0)
    out.append(Profile("rotation_only_control", hips, m, DEFAULT_FRAME_TIME, "Y",
                       "the same body and motion as reference_humanoid_6ch written the ordinary way: the root is the pelvis, its position channels carry the travel, every other joint is rotation-only; world positions of the body match that file",
                       "pybvh's bundled clips", "Hips"))

    # 9. Six-channel version of the control: every joint rigid, positions == OFFSET.
    hips = Joint("Hips", (0.0, 0.0, 0.0))
    humanoid_body(hips)
    set_layout(hips, all_six)
    m = Motion(SEED + 1)
    m.pos_override["Hips"] = travel((0.0, 95.0, 0.0))
    m.rot_override["Hips"] = lambda t: (1.5 * math.sin(2 * math.pi * t / 4), 12.0, 0.0)
    out.append(Profile("rigid_6ch_control", hips, m, DEFAULT_FRAME_TIME, "Y",
                       "rotation_only_control with six channels declared on every joint and each non-root joint's OFFSET repeated as its position in every frame; world positions identical to the rotation-only file",
                       "pybvh's bundled clips", "Hips"))

    # 10. Static reference root carrying a body and two unlike props.
    ref = Joint("Reference", (0.0, 0.0, 0.0))
    hips = ref.add(Joint("Hips", (120.0, 95.0, 600.0)))
    humanoid_body(hips)
    sword = ref.add(Joint("Sword", (150.0, 100.0, 630.0)))
    tip = sword.add(Joint("SwordTip", (0.0, 0.0, 40.0)))
    tip.end_site = (0.0, 0.0, 30.0)
    marker = ref.add(Joint("Marker", (-40.0, 0.0, 60.0)))
    marker.end_site = (0.0, 5.0, 0.0)
    set_layout(ref, all_six)
    m = Motion(SEED + 1)  # near/far bodies have identical local poses
    m.rot_override["Reference"] = lambda t: (0.0, 0.0, 0.0)
    m.pos_override["Reference"] = lambda t: (0.0, 0.0, 0.0)
    m.pos_override["Hips"] = travel((120.0, 95.0, 600.0))
    m.rot_override["Hips"] = lambda t: (1.5 * math.sin(2 * math.pi * t / 4), 12.0, 0.0)
    m.pos_override["Sword"] = lambda t: (150.0 + 1.0 * t, 100.0 + 2.0 * math.sin(math.pi * t / 4), 630.0 + 6.0 * t)
    m.rot_override["Sword"] = lambda t: (-20.0 + 5.0 * t, 0.0, 10.0)
    m.rot_override["SwordTip"] = lambda t: (0.0, 0.0, 0.0)
    m.rot_override["Marker"] = lambda t: (0.0, 0.0, 0.0)
    out.append(Profile("reference_props_humanoid_6ch", ref, m, DEFAULT_FRAME_TIME, "Y",
                       "static reference root with three unlike children: the travelling body (far from the origin), a two-joint prop that moves on its own, and a single marker; the root has three children but is not the pelvis",
                       "reference-node export with tracked props", "Hips"))

    # This companion isolates anatomy from unsupported position declarations.
    # A rotation-only attachment stays at its OFFSET; only the body pose and
    # prop orientation animate, so it is intentionally not a travelling copy.
    fixed = copy.deepcopy(ref)
    set_layout(fixed, root_only)
    fixed_motion = copy.deepcopy(m)
    fixed_motion.pos_override = {"Reference": lambda t: (0.0, 0.0, 0.0)}
    out.append(Profile("reference_props_rotation_only", fixed, fixed_motion,
                       DEFAULT_FRAME_TIME, "Y",
                       "rotation-only anatomy companion: static reference, far-offset body and two unlike props; readable before non-root position support",
                       "authored reference-node attachment", "Hips",
                       "body and prop positions stay at rest; their rotations animate"))

    return out


# --------------------------------------------------------------------------
# Serialisation
# --------------------------------------------------------------------------

def fmt(x: float) -> str:
    s = f"{x:.6f}"
    return "0.000000" if s == "-0.000000" else s


def channel_values(joint: Joint, motion: Motion, t: int, is_root: bool) -> list[str]:
    rx, ry, rz = motion.rotation(joint, t, is_root)
    px, py, pz = motion.position(joint, t)
    table = {"Xposition": px, "Yposition": py, "Zposition": pz,
             "Xrotation": rx, "Yrotation": ry, "Zrotation": rz}
    return [fmt(table[c]) for c in joint.channels]


def serialise(p: Profile) -> str:
    lines: list[str] = ["HIERARCHY"]

    def node(j: Joint, depth: int, is_root: bool):
        ind = "\t" * depth
        lines.append(f"{ind}{'ROOT' if is_root else 'JOINT'} {j.name}")
        lines.append(f"{ind}{{")
        lines.append(f"{ind}\tOFFSET {fmt(j.offset[0])} {fmt(j.offset[1])} {fmt(j.offset[2])}")
        lines.append(f"{ind}\tCHANNELS {len(j.channels)} {' '.join(j.channels)}")
        for c in j.children:
            node(c, depth + 1, False)
        if j.end_site is not None:
            e = j.end_site
            lines.append(f"{ind}\tEnd Site")
            lines.append(f"{ind}\t{{")
            lines.append(f"{ind}\t\tOFFSET {fmt(e[0])} {fmt(e[1])} {fmt(e[2])}")
            lines.append(f"{ind}\t}}")
        lines.append(f"{ind}}}")

    node(p.root, 0, True)
    lines.append("MOTION")
    lines.append(f"Frames: {FRAMES}")
    lines.append(f"Frame Time: {p.frame_time}")
    for t in range(FRAMES):
        row: list[str] = []
        for j in p.root.walk():
            row.extend(channel_values(j, p.motion, t, j is p.root))
        lines.append(" ".join(row))
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Manifest and README
# --------------------------------------------------------------------------

README_HEAD = """# bvh_data/synthetic — invented clips for conventions the bundled files lack

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
"""


def write_outputs(out_dir: Path, do_check: bool) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"generator": "scripts/generate_synthetic_bvh.py", "generator_version": GENERATOR_VERSION,
                "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "audit_sha256": hashlib.sha256(Path(__file__).with_name("audit_synthetic_bvh.py").read_bytes()).hexdigest(),
                "seed": SEED, "frames": FRAMES, "units": "centimetres", "up_axis": "Y",
                "synthetic": True, "forward_axis": "+z",
                "format": "UTF-8, LF, six decimals, canonical positive zero",
                "perturbation": "SHA-256(seed:joint:frame:axis), first 64 bits mapped to +/-0.2 degrees; frame 0 unperturbed",
                "licence": "same as pybvh (MIT); all values authored in the generator, none copied",
                "files": []}
    readme_rows = []
    failures = 0
    for p in profiles():
        text = serialise(p)
        path = out_dir / f"{p.name}.bvh"
        path.write_bytes(text.encode("utf-8"))
        entry = {"file": path.name, "profile": p.name, "profile_revision": 2, "seed": p.motion.seed,
                 "synthetic": True, "copied_components": [],
                 "provenance": "Generic authored topology, original offsets and analytic motion; conventional joint names; no source file or numerical data imported",
                 "source_url": None, "source_sha256": None,
                 "licence": "MIT (repository LICENSE)", "frames": FRAMES,
                 "describes": p.describes,
                 "shape_after": p.shape_after, "frame_time": p.frame_time, "up_axis": p.up_axis,
                 "expected_pelvis": p.expected_pelvis, "notes": p.notes,
                 "sha256": hashlib.sha256(text.encode()).hexdigest()}
        try:
            entry.update(audit(path.read_text(encoding="utf-8"), p.expected_pelvis,
                               authored=p))
            status = "ok"
        except (AssertionError, ValueError, KeyError, IndexError) as e:  # reported, not raised
            failures += 1
            status = f"AUDIT FAILED: {e}"
            entry["audit_failure"] = str(e)
        manifest["files"].append(entry)
        six_joints = entry.get("six_channel_non_root_joints", [])
        six = ", ".join(six_joints) or "none"
        if len(six_joints) > 5:
            six = (f"all {len(six_joints)} non-root joints"
                   if len(six_joints) == entry["joints"] - 1
                   else f"{len(six_joints)} non-root joints")
        readme_rows.append(f"| `{path.name}` | {p.describes} | {p.shape_after} | {six} | {p.expected_pelvis} |")
        print(f"{status:>8}  {path.name:36s} joints={entry.get('joints','?'):>3} width={entry.get('width','?'):>4} "
              f"six={len(entry.get('six_channel_non_root_joints', [])):>3} moving={entry.get('moving_six_channel_joints')}")
    (out_dir / "manifest.json").write_bytes((json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
    (out_dir / "README.md").write_bytes((README_HEAD + "\n".join(readme_rows) + "\n").encode("utf-8"))
    if do_check:
        print(f"\naudit: {len(manifest['files']) - failures} passed, {failures} failed")
    return 1 if failures else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="bvh_data/synthetic", type=Path)
    ap.add_argument("--check", action="store_true", help="audit every written file and fail on any defect")
    a = ap.parse_args(argv)
    return write_outputs(a.out, a.check)


if __name__ == "__main__":
    sys.exit(main())
