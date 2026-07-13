"""Synthetic BVH fixture library for controlled testing.

Provides helper functions that build Bvh objects programmatically with known
geometry and motion, enabling deterministic tests for axis-sign handling,
mirroring, scaling, and other operations that depend on skeleton structure.

All fixtures use this skeleton topology:

    Root (Hips)
    +-- Spine --> Head (End Site)
    +-- LeftLeg --> LeftFoot (End Site)
    +-- RightLeg --> RightFoot (End Site)

6 joints, 4 end sites = 10 nodes total.
"""
from __future__ import annotations

import numpy as np

from pybvh.bvh import Bvh
from pybvh.bvhnode import BvhRoot, BvhJoint, BvhNode, BvhEndSite


# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------

def _build_skeleton(
    up_idx: int,
    up_sign: int,
    lat_idx: int,
    fwd_idx: int,
    root_rot_order: str = "ZYX",
    left_rot_order: str = "ZYX",
    right_rot_order: str = "ZYX",
    left_name: str = "LeftLeg",
    right_name: str = "RightLeg",
) -> list[BvhNode]:
    """Build the 10-node skeleton with offsets along the requested axes.

    Parameters
    ----------
    up_idx : int
        Index of the up axis (0=X, 1=Y, 2=Z).
    up_sign : int
        +1 or -1.  Head is placed in the ``up_sign`` direction from root.
    lat_idx : int
        Index of the lateral axis.
    fwd_idx : int
        Index of the forward axis.
    root_rot_order, left_rot_order, right_rot_order : str
        Euler order strings for the respective joints.
    left_name, right_name : str
        Names for L/R leg joints (allows lowercase testing).
    """
    def _off(axis: int, value: float) -> list[float]:
        """Return a 3-element offset with *value* on *axis*, zeros elsewhere."""
        o = [0.0, 0.0, 0.0]
        o[axis] = value
        return o

    # Offsets — head above, feet below, laterally separated
    head_end_offset = _off(up_idx, 5.0 * up_sign)
    spine_offset = _off(up_idx, 10.0 * up_sign)
    left_offset = [0.0, 0.0, 0.0]
    left_offset[lat_idx] = -3.0
    left_offset[up_idx] = -5.0 * up_sign
    right_offset = [0.0, 0.0, 0.0]
    right_offset[lat_idx] = 3.0
    right_offset[up_idx] = -5.0 * up_sign
    foot_end_offset = _off(up_idx, -5.0 * up_sign)

    # --- Build nodes (bottom-up so we can assign children) ---
    head_end = BvhEndSite("Head", offset=head_end_offset)
    spine = BvhJoint("Spine", offset=spine_offset, rot_channels=list(root_rot_order),
                      children=[head_end])
    head_end.parent = spine

    left_foot_end = BvhEndSite("LeftFoot", offset=foot_end_offset)
    left_leg = BvhJoint(left_name, offset=left_offset, rot_channels=list(left_rot_order),
                         children=[left_foot_end])
    left_foot_end.parent = left_leg

    right_foot_end = BvhEndSite("RightFoot", offset=foot_end_offset)
    right_leg = BvhJoint(right_name, offset=right_offset, rot_channels=list(right_rot_order),
                          children=[right_foot_end])
    right_foot_end.parent = right_leg

    root = BvhRoot("Hips", offset=[0.0, 0.0, 0.0],
                    rot_channels=list(root_rot_order),
                    children=[spine, left_leg, right_leg])
    spine.parent = root
    left_leg.parent = root
    right_leg.parent = root

    # Flat depth-first list (matches BVH traversal order)
    nodes = [root, spine, head_end, left_leg, left_foot_end, right_leg, right_foot_end]
    return nodes


def _build_motion(
    n_frames: int,
    up_idx: int,
    up_sign: int,
    fwd_idx: int,
    n_joints: int = 4,
    add_rotations: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Build motion arrays with known values.

    Root translates forward at a known height.  Joint angles are zero by
    default, which guarantees FK pos/neg equivalence (negating the up-axis
    component of offsets and root_pos produces negated up-axis FK positions,
    with all other axes unchanged).

    When ``add_rotations=True``, root gets a 90-degree rotation and legs get
    cyclic oscillation.  This breaks the simple pos/neg FK equivalence but is
    useful for testing rotation-dependent functions.

    Returns (root_pos, joint_angles) with shapes (n_frames, 3) and
    (n_frames, n_joints, 3).
    """
    root_pos = np.zeros((n_frames, 3), dtype=np.float64)
    # Root height (on up axis)
    root_pos[:, up_idx] = 100.0 * up_sign
    # Root moves forward over time
    t = np.linspace(0, 50, n_frames)
    root_pos[:, fwd_idx] = t

    joint_angles = np.zeros((n_frames, n_joints, 3), dtype=np.float64)

    if add_rotations:
        # Root rotates 90 degrees (channel 0 of root joint)
        joint_angles[:, 0, 0] = np.linspace(0, 90, n_frames)
        # Legs cycle
        leg_cycle = 15.0 * np.sin(np.linspace(0, 2 * np.pi, n_frames))
        if n_joints >= 3:
            joint_angles[:, 2, 1] = leg_cycle
        if n_joints >= 4:
            joint_angles[:, 3, 1] = -leg_cycle

    return root_pos, joint_angles


# ---------------------------------------------------------------------------
#  Public fixture functions
# ---------------------------------------------------------------------------

N_FRAMES = 10
FRAME_TIME = 1.0 / 30.0


def make_pos_y_up_bvh() -> Bvh:
    """Standard +Y up skeleton with 10 frames of known motion."""
    up_idx, up_sign, lat_idx, fwd_idx = 1, 1, 0, 2
    nodes = _build_skeleton(up_idx, up_sign, lat_idx, fwd_idx)
    n_joints = sum(1 for n in nodes if not n.is_end_site())
    root_pos, joint_angles = _build_motion(N_FRAMES, up_idx, up_sign, fwd_idx, n_joints)
    bvh = Bvh(nodes=nodes, root_pos=root_pos, joint_angles=joint_angles,
              frame_time=FRAME_TIME)
    bvh.world_up = '+y'
    return bvh


def make_clip_bvh(n_frames: int, frame_time: float) -> Bvh:
    """+Y up skeleton with ``n_frames`` zero-rotation frames at ``frame_time``.

    For tests that sculpt trajectories via the ``coords=`` escape hatch at
    controlled frame rates (the fixed-length fixtures below are all
    ``N_FRAMES`` frames at 30 fps).
    """
    up_idx, up_sign, lat_idx, fwd_idx = 1, 1, 0, 2
    nodes = _build_skeleton(up_idx, up_sign, lat_idx, fwd_idx)
    n_joints = sum(1 for n in nodes if not n.is_end_site())
    root_pos, joint_angles = _build_motion(n_frames, up_idx, up_sign, fwd_idx,
                                            n_joints)
    bvh = Bvh(nodes=nodes, root_pos=root_pos, joint_angles=joint_angles,
              frame_time=frame_time)
    bvh.world_up = '+y'
    return bvh


def make_neg_y_up_bvh() -> Bvh:
    """Inverted -Y up skeleton.  Same physical motion as pos_y, negated Y."""
    up_idx, up_sign, lat_idx, fwd_idx = 1, -1, 0, 2
    nodes = _build_skeleton(up_idx, up_sign, lat_idx, fwd_idx)
    n_joints = sum(1 for n in nodes if not n.is_end_site())
    root_pos, joint_angles = _build_motion(N_FRAMES, up_idx, up_sign, fwd_idx, n_joints)
    bvh = Bvh(nodes=nodes, root_pos=root_pos, joint_angles=joint_angles,
              frame_time=FRAME_TIME)
    bvh.world_up = '-y'
    return bvh


def make_pos_z_up_bvh() -> Bvh:
    """Standard +Z up skeleton with 10 frames of known motion."""
    up_idx, up_sign, lat_idx, fwd_idx = 2, 1, 0, 1
    nodes = _build_skeleton(up_idx, up_sign, lat_idx, fwd_idx)
    n_joints = sum(1 for n in nodes if not n.is_end_site())
    root_pos, joint_angles = _build_motion(N_FRAMES, up_idx, up_sign, fwd_idx, n_joints)
    bvh = Bvh(nodes=nodes, root_pos=root_pos, joint_angles=joint_angles,
              frame_time=FRAME_TIME)
    bvh.world_up = '+z'
    return bvh


def make_neg_z_up_bvh() -> Bvh:
    """Inverted -Z up skeleton.  Same physical motion as pos_z, negated Z."""
    up_idx, up_sign, lat_idx, fwd_idx = 2, -1, 0, 1
    nodes = _build_skeleton(up_idx, up_sign, lat_idx, fwd_idx)
    n_joints = sum(1 for n in nodes if not n.is_end_site())
    root_pos, joint_angles = _build_motion(N_FRAMES, up_idx, up_sign, fwd_idx, n_joints)
    bvh = Bvh(nodes=nodes, root_pos=root_pos, joint_angles=joint_angles,
              frame_time=FRAME_TIME)
    bvh.world_up = '-z'
    return bvh


def make_heterogeneous_euler_bvh() -> Bvh:
    """+Y up skeleton where LeftLeg uses ZYX and RightLeg uses XYZ.

    For testing mirror with heterogeneous Euler orders.
    """
    up_idx, up_sign, lat_idx, fwd_idx = 1, 1, 0, 2
    nodes = _build_skeleton(up_idx, up_sign, lat_idx, fwd_idx,
                            left_rot_order="ZYX", right_rot_order="XYZ")
    n_joints = sum(1 for n in nodes if not n.is_end_site())
    root_pos, joint_angles = _build_motion(N_FRAMES, up_idx, up_sign, fwd_idx, n_joints)
    bvh = Bvh(nodes=nodes, root_pos=root_pos, joint_angles=joint_angles,
              frame_time=FRAME_TIME)
    bvh.world_up = '+y'
    return bvh


def make_lowercase_lr_bvh() -> Bvh:
    """+Y up skeleton with lowercase L/R names: 'leftLeg', 'rightLeg'.

    For testing case-sensitivity of the L/R name detection heuristic
    behind ``Bvh.lr_mapping``.
    """
    up_idx, up_sign, lat_idx, fwd_idx = 1, 1, 0, 2
    nodes = _build_skeleton(up_idx, up_sign, lat_idx, fwd_idx,
                            left_name="leftLeg", right_name="rightLeg")
    n_joints = sum(1 for n in nodes if not n.is_end_site())
    root_pos, joint_angles = _build_motion(N_FRAMES, up_idx, up_sign, fwd_idx, n_joints)
    bvh = Bvh(nodes=nodes, root_pos=root_pos, joint_angles=joint_angles,
              frame_time=FRAME_TIME)
    bvh.world_up = '+y'
    return bvh


def make_dot_lr_bvh() -> Bvh:
    """+Y up skeleton using Blender-style '.L' / '.R' suffix naming."""
    up_idx, up_sign, lat_idx, fwd_idx = 1, 1, 0, 2
    nodes = _build_skeleton(up_idx, up_sign, lat_idx, fwd_idx,
                            left_name="Leg.L", right_name="Leg.R")
    n_joints = sum(1 for n in nodes if not n.is_end_site())
    root_pos, joint_angles = _build_motion(N_FRAMES, up_idx, up_sign, fwd_idx, n_joints)
    bvh = Bvh(nodes=nodes, root_pos=root_pos, joint_angles=joint_angles,
              frame_time=FRAME_TIME)
    bvh.world_up = '+y'
    return bvh


def make_underscore_lr_bvh() -> Bvh:
    """+Y up skeleton using '_l' / '_r' suffix naming (lowercase)."""
    up_idx, up_sign, lat_idx, fwd_idx = 1, 1, 0, 2
    nodes = _build_skeleton(up_idx, up_sign, lat_idx, fwd_idx,
                            left_name="leg_l", right_name="leg_r")
    n_joints = sum(1 for n in nodes if not n.is_end_site())
    root_pos, joint_angles = _build_motion(N_FRAMES, up_idx, up_sign, fwd_idx, n_joints)
    bvh = Bvh(nodes=nodes, root_pos=root_pos, joint_angles=joint_angles,
              frame_time=FRAME_TIME)
    bvh.world_up = '+y'
    return bvh


def make_namespace_lr_bvh() -> Bvh:
    """+Y up skeleton using Mixamo-style 'mixamorig:LeftLeg' namespace prefix."""
    up_idx, up_sign, lat_idx, fwd_idx = 1, 1, 0, 2
    nodes = _build_skeleton(up_idx, up_sign, lat_idx, fwd_idx,
                            left_name="mixamorig:LeftLeg",
                            right_name="mixamorig:RightLeg")
    n_joints = sum(1 for n in nodes if not n.is_end_site())
    root_pos, joint_angles = _build_motion(N_FRAMES, up_idx, up_sign, fwd_idx, n_joints)
    bvh = Bvh(nodes=nodes, root_pos=root_pos, joint_angles=joint_angles,
              frame_time=FRAME_TIME)
    bvh.world_up = '+y'
    return bvh


def make_numbered_lr_bvh() -> Bvh:
    """+Y up skeleton using '.L.001' / '.R.001' numbered-duplicate suffix."""
    up_idx, up_sign, lat_idx, fwd_idx = 1, 1, 0, 2
    nodes = _build_skeleton(up_idx, up_sign, lat_idx, fwd_idx,
                            left_name="Leg.L.001", right_name="Leg.R.001")
    n_joints = sum(1 for n in nodes if not n.is_end_site())
    root_pos, joint_angles = _build_motion(N_FRAMES, up_idx, up_sign, fwd_idx, n_joints)
    bvh = Bvh(nodes=nodes, root_pos=root_pos, joint_angles=joint_angles,
              frame_time=FRAME_TIME)
    bvh.world_up = '+y'
    return bvh


def make_bare_substring_lr_bvh() -> Bvh:
    """+Y up skeleton using 'LeftEye' / 'RightEye' — bare substring, no delimiter.

    Regression fixture: strategy A must still detect these when the
    more-specific suffix and prefix rules don't match.
    """
    up_idx, up_sign, lat_idx, fwd_idx = 1, 1, 0, 2
    nodes = _build_skeleton(up_idx, up_sign, lat_idx, fwd_idx,
                            left_name="LeftEye", right_name="RightEye")
    n_joints = sum(1 for n in nodes if not n.is_end_site())
    root_pos, joint_angles = _build_motion(N_FRAMES, up_idx, up_sign, fwd_idx, n_joints)
    bvh = Bvh(nodes=nodes, root_pos=root_pos, joint_angles=joint_angles,
              frame_time=FRAME_TIME)
    bvh.world_up = '+y'
    return bvh


def make_nameless_lr_bvh() -> Bvh:
    """+Y up skeleton with cryptic joint names ('J1','J2','J3',...).

    lr_mapping auto-detection should return ``None`` for this skeleton —
    no L/R naming cues exist.
    """
    up_idx, up_sign, lat_idx, fwd_idx = 1, 1, 0, 2
    nodes = _build_skeleton(up_idx, up_sign, lat_idx, fwd_idx,
                            left_name="J2", right_name="J3")
    n_joints = sum(1 for n in nodes if not n.is_end_site())
    root_pos, joint_angles = _build_motion(N_FRAMES, up_idx, up_sign, fwd_idx, n_joints)
    bvh = Bvh(nodes=nodes, root_pos=root_pos, joint_angles=joint_angles,
              frame_time=FRAME_TIME)
    bvh.world_up = '+y'
    return bvh


def make_singleton_lr_bvh() -> Bvh:
    """+Y up skeleton with 'LeftLeg' but NO pairable counterpart.

    The 'right' side is named 'OtherLeg' (no L/R cue), so mutual-match
    should leave 'LeftLeg' unpaired — lr_mapping becomes ``None``.
    """
    up_idx, up_sign, lat_idx, fwd_idx = 1, 1, 0, 2
    nodes = _build_skeleton(up_idx, up_sign, lat_idx, fwd_idx,
                            left_name="LeftLeg", right_name="OtherLeg")
    n_joints = sum(1 for n in nodes if not n.is_end_site())
    root_pos, joint_angles = _build_motion(N_FRAMES, up_idx, up_sign, fwd_idx, n_joints)
    bvh = Bvh(nodes=nodes, root_pos=root_pos, joint_angles=joint_angles,
              frame_time=FRAME_TIME)
    bvh.world_up = '+y'
    return bvh


def make_pos_y_up_rotating_bvh() -> Bvh:
    """+Y up skeleton with root rotation (90 deg over clip) and leg cycling.

    Use this for tests that need non-trivial rotations (root_trajectory
    heading, rotate_vertical direction).  Does NOT have a clean pos/neg FK
    equivalence with make_neg_y_up_bvh (use zero-angle fixtures for that).
    """
    up_idx, up_sign, lat_idx, fwd_idx = 1, 1, 0, 2
    nodes = _build_skeleton(up_idx, up_sign, lat_idx, fwd_idx)
    n_joints = sum(1 for n in nodes if not n.is_end_site())
    root_pos, joint_angles = _build_motion(N_FRAMES, up_idx, up_sign, fwd_idx,
                                            n_joints, add_rotations=True)
    bvh = Bvh(nodes=nodes, root_pos=root_pos, joint_angles=joint_angles,
              frame_time=FRAME_TIME)
    bvh.world_up = '+y'
    return bvh


def make_simple_bvh(world_up: str = '+y') -> Bvh:
    """Minimal skeleton with 1 frame for quick tests (scale, eq, extract, etc.).

    Uses the same topology but only 1 frame with non-zero root position for
    easy numerical verification.
    """
    sign_char, axis_char = world_up[0], world_up[1]
    up_idx = {'x': 0, 'y': 1, 'z': 2}[axis_char]
    up_sign = 1 if sign_char == '+' else -1
    lat_idx = 0 if up_idx != 0 else 1
    fwd_idx = ({0, 1, 2} - {up_idx, lat_idx}).pop()

    nodes = _build_skeleton(up_idx, up_sign, lat_idx, fwd_idx)
    n_joints = sum(1 for n in nodes if not n.is_end_site())
    root_pos, joint_angles = _build_motion(1, up_idx, up_sign, fwd_idx, n_joints)
    # Give root a recognizable position for numerical tests
    root_pos[0] = [10.0, 20.0, 30.0]
    bvh = Bvh(nodes=nodes, root_pos=root_pos, joint_angles=joint_angles,
              frame_time=FRAME_TIME)
    bvh.world_up = world_up
    return bvh


def make_disagreement_bvh() -> Bvh:
    """BVH where rest-pose offsets suggest +Z up but animation plays with +Y up.

    Triggers the _infer_world_up disagreement warning. After
    reorient_rest_up('+y'), the disagreement should disappear.
    """
    # Build skeleton with Z-up offsets (head above root on Z axis)
    up_idx, up_sign, lat_idx, fwd_idx = 2, 1, 0, 1  # +Z up
    nodes = _build_skeleton(up_idx, up_sign, lat_idx, fwd_idx)
    n_joints = sum(1 for n in nodes if not n.is_end_site())

    # But make root_pos heights on Y axis (as if the animation is Y-up)
    root_pos, joint_angles = _build_motion(N_FRAMES, 1, 1, 2, n_joints,
                                            add_rotations=True)
    # root_pos[:, 1] = 100 (Y height), root_pos[:, 2] = 0 (no Z height)
    # This makes frame-0 head-hips direction point along Y (animation says +Y)
    # while rest-pose offsets point along Z (topology says +Z)

    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.filterwarnings("ignore", message="Rest pose suggests world up")
        bvh = Bvh(nodes=nodes, root_pos=root_pos, joint_angles=joint_angles,
                  frame_time=FRAME_TIME)
    # The auto-detection picked +Y from animation; rest-pose says +Z
    # Don't override — let auto-detection stand
    return bvh
