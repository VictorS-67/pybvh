"""The orientation a skeleton's geometry implies.

The skeleton's own frame read from the rest pose alone (rest up, rest
left, rest forward), the world up axis inferred from a clip's first frame,
the left/right joint pairs found by name and their resolution into
joint-space and node-space index pairs, the body anchor where those pairs
meet, and the per-frame leftward,
forward and facing basis, with the house default for a skeleton that
carries no left/right information; and :class:`OrientationReport`, which
says for each clip-level fact whether it was set, measured or defaulted.
:class:`~pybvh.bvh.Bvh` exposes their results as ``world_up``,
``rest_up``, ``rest_forward``, ``lr_mapping``, ``lr_pairs``, ``body_anchor``,
``forward_at``, ``left_at`` and ``facing_frame``; ``transforms``,
``analysis`` and bvhplot call them rather than keep their own. The axis
vocabulary they are written in (signed axis strings, :class:`Axis`,
:func:`get_main_direction`) is :mod:`pybvh.tools`'.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, NamedTuple

import numpy as np
import numpy.typing as npt

from ._warnings import user_stacklevel
from .tools import _AXIS_CHAR_TO_IDX, _axis_to_vector, get_main_direction

if TYPE_CHECKING:
    from .bvhnode import BvhNode

# ===========================================================================
# ORIENTATION HELPERS (internal)
# ===========================================================================
#
# The facts belong to two frames, and no function combines an axis of one
# with an axis of the other (ADR 0004):
#
#   1. THE SKELETON'S OWN FRAME — read from the rest pose alone, never from
#      world up, on plain data (node names, rest positions, L/R pairs):
#      _up_marker_index → _infer_rest_up; _infer_rest_left (the L/R pairs
#      against rest up), read together with one set of pairs by
#      _infer_rest_up_and_left; _infer_rest_forward (rest left × rest up,
#      or the house default keyed on rest up). Used by mirror, the rest
#      reorientations and root_trajectory's heading, where the body
#      anchor's world rotation carries rest forward into the world.
#   2. THE WORLD FRAME — world up, guessed from the first frame through the
#      same up marker (_infer_world_up); the heading (_heading_units: the
#      body anchor's world rotation carries rest forward into the world,
#      NaN where its ground projection is unreadable); and the
#      facing at a frame, read from that frame's joint positions against
#      world up.
#
# The L/R mapping is detected from joint names and resolved into index
# pairs (_detect_lr_mapping_by_names, _resolve_lr_pairs,
# _resolve_node_lr_pairs), also on plain data. Bvh gathers every input.
#
# The facing chain, per frame, one up throughout (→ = "used by"):
#
#     Bvh._facing_lr_pairs ──→ bvhplot SkeletonView.lr_pairs
#     (the body's pairs only)  (the follow camera)
#           │                       │
#           ▼                       ▼
#     _leftward_units_from_pairs ── the ONLY place the per-frame leftward
#           │   (array kernel)      geometry is computed; bvhplot
#           │                       follow-mode azimuths read it too
#           ▼
#     _facing_basis: the frame's leftward, else the heading at that
#           │        frame (_heading_units), else the house default
#           │        (_fallback_leftward); forward = left × up
#           ├──→ Bvh._facing_at ──→ analysis.facing_frame
#           └──→ _facing_labels (the snap) ──→ Bvh.forward_at / Bvh.left_at
#
# Public surface: `Bvh.orientation_report` (OrientationReport: each fact
# below with its source), `Bvh.world_up` (property + setter), `Bvh.rest_up`,
# `Bvh.rest_forward`, `Bvh.has_lr_geometry`, the parsed `Bvh.world_up_axis` /
# `Bvh.rest_forward_axis` / `Bvh.rest_up_axis` properties,
# `Bvh.forward_at(frame=0)` / `Bvh.left_at(frame=0)` (methods), and — via
# `analysis.facing_frame` — the continuous vector basis. Everything here but
# OrientationReport is underscore-prefixed and intended for internal use by
# pybvh modules.
# ---------------------------------------------------------------------------

# Arbitrary-but-stable forward per up axis (rest up for rest forward, world up
# for the per-frame facing), used when a skeleton carries no usable L/R
# orientation information.
_FALLBACK_FORWARD = {"y": "+z", "z": "+x", "x": "+y"}


# The body parts whose direction from the body anchor says which way is up, most
# telling first.
_UP_BODY_PARTS = ("head", "neck", "chest", "spine")

# The words of a joint name that say it is part of a leg or a foot: the body
# starts above the body anchor when none of its joints holds one
# (_find_body).
_FOOT_NAMES = ("leg", "foot", "toe", "toes")

# What each orientation fact was read from, as OrientationReport names it.
_RestUpSource = Literal["named_joint", "leg_pair", "largest_spread"]
_RestForwardSource = Literal["lr_geometry", "default"]
_WorldUpGuessSource = Literal["first_frame", "rest_up", "default"]
_WorldUpSource = Literal["set", _WorldUpGuessSource]
_LrMappingSource = Literal["set", "names"]


# ---------------------------------------------------------------------------
# Joint names (pure string logic), read by the up marker and L/R detection
# ---------------------------------------------------------------------------

_NUMBER_SUFFIX_RE = re.compile(r"\.\d+$")

# A word of a name: a capitalized or lowercase run of letters, a run of
# capitals not followed by a lowercase letter (an acronym, or a name in
# capitals), or a run of digits. Anything else separates words.
_NAME_WORD_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")


def _split_name(name: str) -> tuple[str, str, str]:
    """Split a joint name into its namespace, its base and its number suffix.

    The one normalization of joint names: a ``foo:`` namespace (Mixamo's
    ``mixamorig:``) and a trailing ``.NNN`` suffix (Blender's ``.001``
    duplicates) are not part of the name. Returns ``(namespace, base,
    suffix)``, the namespace with its colon, each ``''`` when absent, so
    ``namespace + base + suffix == name``.
    """
    before_colon, colon, after_namespace = name.rpartition(":")
    base = _NUMBER_SUFFIX_RE.sub("", after_namespace)
    return before_colon + colon, base, after_namespace[len(base) :]


def _name_words(name: str) -> set[str]:
    """The words of a joint name's base, lowercased.

    Words are split at anything not a letter or a digit, at a lowercase
    letter followed by a capital, and between letters and digits:
    ``Bip01 Head`` gives ``bip``, ``01``, ``head``; ``HeadTop_End`` gives
    ``head``, ``top``, ``end``; ``Forehead`` gives ``forehead`` alone.
    """
    _, base, _ = _split_name(name)
    return {word.lower() for word in _NAME_WORD_RE.findall(base)}


class _RestSkeleton(NamedTuple):
    """A skeleton in its rest pose, as the plain data its own frame is read from.

    ``Bvh`` gathers it; rest up, the world-up guess and the orientation
    report read it.

    Attributes
    ----------
    node_names : sequence of str
        Every node's name, in node order (end sites included); names may
        repeat.
    rest_positions : ndarray of shape (N, 3)
        The rest pose's node positions, in the same order.
    parent_idx : ndarray of shape (N,)
        Each node's parent index, ``-1`` for the root; a parent comes
        before its children.
    is_end_site : ndarray of shape (N,), bool
        Which nodes are end sites.
    lr_pairs : ndarray of shape (P, 2)
        ``(left_idx, right_idx)`` rows into the N axis: the joint pairs of
        the L/R mapping in effect.
    static_root : bool
        Whether the root declares no channels (``CHANNELS 0``): each of its
        children then heads a tree of its own for the body anchor.
    """

    node_names: Sequence[str]
    rest_positions: npt.NDArray[np.float64]
    parent_idx: npt.NDArray[np.intp]
    is_end_site: npt.NDArray[np.bool_]
    lr_pairs: npt.NDArray[np.intp]
    static_root: bool = False


# ---------------------------------------------------------------------------
# The body anchor: where the L/R pairs meet, read from the hierarchy alone
# ---------------------------------------------------------------------------

# How the body anchor was found, as OrientationReport names it.
_BodyAnchorSource = Literal["set", "lr_pairs", "default", "ambiguous"]


class _BodyAnchor(NamedTuple):
    """The joint the body's measures start from, and how it was found.

    Attributes
    ----------
    node : int or None
        The anchor's index in node order; ``None`` when the clip has no
        body.
    source : {"set", "lr_pairs", "default", "ambiguous"}
        ``"set"``: the joint the user set (``Bvh.body_anchor``).
        ``"lr_pairs"``: found where the L/R pairs meet
        (:func:`_find_body_anchor`). ``"default"``: no pair, so the root.
        ``"ambiguous"``: no body, as no candidate is more than twice as
        large as each other one.
    candidates : tuple of int
        The meeting joints the rule compared, the most joints below them
        first; empty for ``"set"`` and ``"default"``.
    """

    node: int | None
    source: _BodyAnchorSource
    candidates: tuple[int, ...]


def _find_body_anchor(skeleton: _RestSkeleton) -> _BodyAnchor:
    """The body anchor the rule finds, from the hierarchy and the L/R pairs.

    Each pair has a meeting joint: the lowest joint that both joints'
    chains to the root pass through. The candidates are the meeting
    joints at the shallowest depth, the root at depth 0, so the legs,
    which meet at the hips, win over the arms, which meet lower, at the
    chest, and a root above the hips or a helper beside them is passed
    over. Of several candidates, the anchor is the one with the most
    joints below it, itself included and end sites excluded, when each
    other candidate has fewer than half as many; otherwise the clip has
    no body (two figures under one root): the anchor is ``None`` and the
    source ``"ambiguous"``. At least half is the tie: 10 joints against
    20 give no body, 9 against 20 the larger one. Without a pair, the
    anchor is the root. Neither the rest positions nor the motion are
    read.

    Under a static root (``CHANNELS 0``, as Blender's exporter writes
    above several parentless bones), each child of the root heads a tree,
    and the trees are compared before the meeting joints. When a second
    tree has at least half the joints of the largest, the clip has no
    body: the candidates are those of each tree with at least half the
    largest's joints, the largest tree first, a tree's candidate being
    its largest shallowest meeting joint, or its head when no pair lies
    inside it. Otherwise only the largest tree's pairs are read, the
    pairs whose two joints are both inside it, so a pair of IK targets
    that are two trees, which meets at the root, is not counted; when
    none is left, the anchor is the root, by default.

    Parameters
    ----------
    skeleton : _RestSkeleton
        The hierarchy (``parent_idx``, ``is_end_site``), the pairs of the
        L/R mapping in effect (``lr_pairs``) and whether the root is
        static (``static_root``).

    Returns
    -------
    _BodyAnchor
    """
    parent_idx = skeleton.parent_idx
    pairs = [(int(a), int(b)) for a, b in skeleton.lr_pairs]
    if not pairs:
        return _BodyAnchor(0, "default", ())
    depth = np.zeros(len(parent_idx), dtype=np.intp)
    for node in range(1, len(parent_idx)):
        depth[node] = depth[parent_idx[node]] + 1
    # A node's joints, itself included, end sites excluded: summed from the
    # leaves up, a parent coming before its children.
    joints_below = (~skeleton.is_end_site).astype(np.intp)
    for node in range(len(parent_idx) - 1, 0, -1):
        joints_below[parent_idx[node]] += joints_below[node]

    def shallowest_meeting_joints(pairs: list[tuple[int, int]]) -> list[int]:
        """The pairs' meeting joints at the shallowest depth, the most
        joints below them first."""
        meeting = {_meeting_joint(parent_idx, depth, a, b) for a, b in pairs}
        shallowest = min(depth[joint] for joint in meeting)
        return sorted(
            (joint for joint in meeting if depth[joint] == shallowest),
            key=lambda joint: (-joints_below[joint], joint),
        )

    if skeleton.static_root:
        # Each child of the root heads a tree; each node's tree is its head.
        tree = np.full(len(parent_idx), -1, dtype=np.intp)
        for node in range(1, len(parent_idx)):
            tree[node] = node if parent_idx[node] == 0 else tree[parent_idx[node]]
        heads = sorted(
            (node for node in range(1, len(parent_idx)) if parent_idx[node] == 0),
            key=lambda head: (-joints_below[head], head),
        )
        largest = joints_below[heads[0]]
        compared = [head for head in heads if 2 * joints_below[head] >= largest]
        pairs_in = {
            head: [(a, b) for a, b in pairs if tree[a] == head and tree[b] == head]
            for head in compared
        }
        if len(compared) > 1:
            # A tree's candidate is where its pairs meet, or its head when
            # no pair lies inside it.
            return _BodyAnchor(
                None,
                "ambiguous",
                tuple(
                    shallowest_meeting_joints(pairs_in[head])[0] if pairs_in[head] else head
                    for head in compared
                ),
            )
        pairs = pairs_in[heads[0]]
        if not pairs:
            return _BodyAnchor(0, "default", ())
    candidates = shallowest_meeting_joints(pairs)
    largest = joints_below[candidates[0]]
    if any(2 * joints_below[other] >= largest for other in candidates[1:]):
        return _BodyAnchor(None, "ambiguous", tuple(candidates))
    return _BodyAnchor(candidates[0], "lr_pairs", tuple(candidates))


def _meeting_joint(
    parent_idx: npt.NDArray[np.intp], depth: npt.NDArray[np.intp], a: int, b: int
) -> int:
    """The lowest node that the chains from *a* and from *b* to the root
    both pass through, each chain holding its own node."""
    while depth[a] > depth[b]:
        a = int(parent_idx[a])
    while depth[b] > depth[a]:
        b = int(parent_idx[b])
    while a != b:
        a, b = int(parent_idx[a]), int(parent_idx[b])
    return a


def _find_body(skeleton: _RestSkeleton, anchor: int) -> npt.NDArray[np.bool_]:
    """The nodes of the body: the joint it starts at and every node below.

    The body starts at the body anchor, unless no joint below the anchor
    has a head name (``head``, ``neck``, ``chest``, ``spine``, the names
    the head joint is found with) or none has a foot name (``leg``, ``foot``, ``toe``,
    ``toes``), each read as a word of the joint's name
    (:func:`_name_words`). For a missing name, the body starts at the
    closest joint above the anchor that has a joint with that name below
    it; with both missing, at the higher of the two joints. When no
    joint above has the missing name, the anchor stays the start for
    that name. So on a rig whose hips sit beside its spine, under a
    common root, the body starts at that root and holds the head, and a
    reference root or a helper beside a whole body stays outside it. End
    sites' names are not read. The anchor itself does not move.

    Parameters
    ----------
    skeleton : _RestSkeleton
        The node names and the hierarchy (``node_names``,
        ``parent_idx``, ``is_end_site``).
    anchor : int
        The body anchor's index (:func:`_find_body_anchor`).

    Returns
    -------
    ndarray of shape (N,), bool
        Which nodes are in the body.
    """
    parent_idx = skeleton.parent_idx
    words = [
        set() if end_site else _name_words(name)
        for name, end_site in zip(skeleton.node_names, skeleton.is_end_site, strict=True)
    ]
    above = []  # the anchor's ancestors, the closest first
    node = int(parent_idx[anchor])
    while node >= 0:
        above.append(node)
        node = int(parent_idx[node])

    # For each missing name, how many joints above the anchor the climb
    # goes: 0 for the anchor's parent.
    climbs = []
    for parts in (_UP_BODY_PARTS, _FOOT_NAMES):
        named = np.array([not name_words.isdisjoint(parts) for name_words in words])
        # A node's named joints, itself included, summed from the leaves
        # up; a parent comes before its children.
        named_in_subtree = named.astype(np.intp)
        for child in range(len(parent_idx) - 1, 0, -1):
            named_in_subtree[parent_idx[child]] += named_in_subtree[child]
        named_below = named_in_subtree - named
        if named_below[anchor]:
            continue
        climb = next((rank for rank, joint in enumerate(above) if named_below[joint]), None)
        if climb is not None:
            climbs.append(climb)
    start = above[max(climbs)] if climbs else anchor

    in_body = np.zeros(len(parent_idx), dtype=bool)
    in_body[start] = True
    for child in range(start + 1, len(parent_idx)):
        in_body[child] = in_body[parent_idx[child]]
    return in_body


def _size_reference(parent_idx: npt.NDArray[np.intp], feet: Sequence[int], anchor: int) -> int:
    """The joint the sizes of a set of feet are read from.

    The lowest joint that has the feet in two or more of its branches:
    the meeting joint of all the feet, where the legs meet on a biped and
    where the hind and the front chains part on a quadruped. When there
    is none, because one foot is given or the feet lie along one leg (the
    meeting joint is then one of the feet), it is the body anchor. The
    hips and not the anchor, because the anchor can sit above the hips
    (the chest, on a rig whose chest and hips meet the pairs at one
    depth), and a size measured from the chest is not the leg's. Read by
    the "feet below hips" check and the threshold scale of the contacts,
    and by ``skeleton_size``.

    Parameters
    ----------
    parent_idx : ndarray of shape (N,)
        Each node's parent index, ``-1`` for the root; a parent comes
        before its children.
    feet : sequence of int
        The feet's node indices, at least one.
    anchor : int
        The body anchor's index (:func:`_find_body_anchor`).

    Returns
    -------
    int
        The size reference's node index.
    """
    depth = np.zeros(len(parent_idx), dtype=np.intp)
    for node in range(1, len(parent_idx)):
        depth[node] = depth[parent_idx[node]] + 1
    meeting = int(feet[0])
    for foot in feet[1:]:
        meeting = _meeting_joint(parent_idx, depth, meeting, int(foot))
    return anchor if meeting in feet else meeting


def _up_marker_index(skeleton: _RestSkeleton, anchor: int) -> int | None:
    """The node whose direction from the body anchor says which way is up.

    The one body-part lookup for "up", shared by the rest up and the
    world-up guess so the two read the same node: the first joint of the
    body (:func:`_find_body`), in node order, whose name holds ``head`` as a word, else ``neck``, ``chest``
    or ``spine``, in any case, after its namespace and number suffix
    (:func:`_name_words`). So ``Head``, ``mixamorig:Head``, ``Bip01
    Head``, ``CC_Base_Head``, ``J_Bip_C_Head`` and ``HeadTop_End`` (a
    joint at the top of the head) are heads, and ``Forehead`` is not. The
    alternative, the whole name alone, misses every rig whose convention
    prefixes its joints; a substring match would take ``Forehead``. End
    sites are not looked up: an end site's name is generated from its
    parent's and names no body part of its own. A joint whose rest
    position coincides with the anchor's carries no direction and is
    skipped, the anchor itself included.

    Parameters
    ----------
    skeleton : _RestSkeleton
        The node names and the rest pose.
    anchor : int
        The body anchor's index (:func:`_find_body_anchor`), which the
        directions are measured from.

    Returns
    -------
    int or None
        The node's index, or ``None`` when no node qualifies.
    """
    local = skeleton.rest_positions - skeleton.rest_positions[anchor]
    in_body = _find_body(skeleton, anchor)
    words = [
        _name_words(name) if inside and not end_site else set()
        for name, end_site, inside in zip(
            skeleton.node_names, skeleton.is_end_site, in_body, strict=True
        )
    ]
    for part in _UP_BODY_PARTS:
        # The first node of a repeated name, by position in the walk.
        for idx, name_words in enumerate(words):
            if part in name_words and get_main_direction(local[idx]) is not None:
                return idx
    return None


# The smallest extent a rest pose must have along some axis to say which way
# is up; below it every node is taken to sit at one point.
_MIN_REST_SPREAD = 1e-6


class _RestUp(NamedTuple):
    """The skeleton's own up and what it was read from.

    ``source`` is ``"named_joint"`` (the up marker,
    :func:`_up_marker_index`), ``"leg_pair"`` (no marker: opposite the
    legs, :func:`_up_from_leg_pair`) or ``"largest_spread"`` (neither:
    the axis along which the body spreads most in the rest pose).
    """

    axis: str
    source: _RestUpSource


def _up_from_leg_pair(skeleton: _RestSkeleton, anchor: int) -> str | None:
    """Rest up opposite the legs: the L/R pair that meets at the body anchor.

    The legs are the first L/R pair, in the mapping's order, whose two
    joints lie below two different children of the anchor, so a pair of
    arms, both below the spine, is never read as the legs, and the legs
    are found whichever of their joints the mapping names. Each leg's
    branch is the anchor's child it lies below, with every node under
    that child; its far end is the node of the branch, end sites
    included, farthest from the anchor, and up is the main axis from the
    middle of the two far ends to the anchor. Read from the anchor, not
    from the root: a root on the floor under the hips sits level with
    the feet, and a reference root may carry props beside the hips.

    Parameters
    ----------
    skeleton : _RestSkeleton
        The hierarchy, the rest pose and the pairs of the L/R mapping in
        effect.
    anchor : int
        The body anchor's index (:func:`_find_body_anchor`).

    Returns
    -------
    str or None
        The signed axis string, or ``None`` without a pair that meets at
        the anchor, or when the far ends coincide with the anchor.
    """
    parent_idx = skeleton.parent_idx
    # For each node, the anchor's child whose subtree holds it, -1 outside
    # the anchor's subtree. Parents precede children, so one pass in node
    # order hands each node its parent's branch; a child of the anchor
    # starts its own.
    branch = np.full(len(parent_idx), -1, dtype=np.intp)
    for node in range(anchor + 1, len(parent_idx)):
        parent = parent_idx[node]
        branch[node] = node if parent == anchor else branch[parent]
    for left, right in skeleton.lr_pairs:
        left_branch, right_branch = int(branch[left]), int(branch[right])
        if left_branch >= 0 and right_branch >= 0 and left_branch != right_branch:
            break
    else:
        return None

    positions = skeleton.rest_positions
    anchor_position = positions[anchor]
    far_ends = []
    for leg in (left_branch, right_branch):
        below = np.flatnonzero(branch == leg)
        distances = np.linalg.norm(positions[below] - anchor_position, axis=1)
        far_ends.append(positions[below[np.argmax(distances)]])
    return get_main_direction(anchor_position - np.mean(far_ends, axis=0))


def _infer_rest_up(skeleton: _RestSkeleton, anchor: int) -> _RestUp | None:
    """The skeleton's own up, read from the rest pose alone, with its source.

    The main axis from the body anchor to the node
    :func:`_up_marker_index` finds; without one, the axis opposite the
    legs, the L/R pair that meets at the anchor
    (:func:`_up_from_leg_pair`); without such a pair, the axis along
    which the body (:func:`_find_body`) spreads most in the rest pose,
    signed by the side of the anchor its nodes lie on. The largest spread is the weakest reading: a
    T-pose wider than tall spreads most along its arms, and hips
    mid-body put as much of it below the anchor as above.

    Parameters
    ----------
    skeleton : _RestSkeleton
        The skeleton in its rest pose, with the pairs of the L/R mapping
        in effect.
    anchor : int
        The body anchor's index (:func:`_find_body_anchor`).

    Returns
    -------
    _RestUp or None
        The signed axis string (e.g. ``'+z'``) and its source, or ``None``
        when the rest pose carries no direction (a single node, or every
        node at one point).
    """
    local = skeleton.rest_positions - skeleton.rest_positions[anchor]
    marker = _up_marker_index(skeleton, anchor)
    if marker is not None:
        axis = get_main_direction(local[marker])
        # The marker is chosen for having a direction from the anchor.
        assert axis is not None
        return _RestUp(axis, "named_joint")

    leg_pair_up = _up_from_leg_pair(skeleton, anchor)
    if leg_pair_up is not None:
        return _RestUp(leg_pair_up, "leg_pair")

    # The body's nodes alone: a prop or a reference root beside the body
    # would spread the pose along an axis the body does not stand on.
    body = local[_find_body(skeleton, anchor)]
    spread = np.ptp(body, axis=0)
    if float(np.max(spread)) < _MIN_REST_SPREAD:
        return None  # degenerate rest pose: all joints coincide
    up_idx = int(np.argmax(spread))
    # The sign of the mean: the side of the anchor most of the body lies on.
    sign = "+" if np.mean(body[:, up_idx]) >= 0 else "-"
    return _RestUp(sign + "xyz"[up_idx], "largest_spread")


# ---------------------------------------------------------------------------
# L/R name-detection heuristics (pure string logic on joint names)
# ---------------------------------------------------------------------------

_SUFFIX_LR_TABLE = [
    # (left_suffix, right_suffix) — matched against the trailing substring
    # of the base name (after namespace and number-suffix strip). Ordered
    # most-specific-first so e.g. ".Left" / ".Right" wins over ".L" / ".R"
    # if both would parse.
    (".Left", ".Right"),
    ("_Left", "_Right"),
    (".left", ".right"),
    ("_left", "_right"),
    (".L", ".R"),
    ("_L", "_R"),
    (".l", ".r"),
    ("_l", "_r"),
]


def _try_suffix_partner(
    base: str, joint_names: set[str], prefix: str, number_suffix: str
) -> str | None:
    """If ``base`` matches a known L/R suffix, return the partner's full name."""
    for left_suf, right_suf in _SUFFIX_LR_TABLE:
        if base.endswith(left_suf):
            partner_base = base[: -len(left_suf)] + right_suf
            partner_full = prefix + partner_base + number_suffix
            if partner_full in joint_names:
                return partner_full
        elif base.endswith(right_suf):
            partner_base = base[: -len(right_suf)] + left_suf
            partner_full = prefix + partner_base + number_suffix
            if partner_full in joint_names:
                return partner_full
    return None


# A side given as a word of its own between delimiters: the suffix
# rule's delimiters, ``.`` and ``_``, and a space (``Bip01 L Thigh``).
# The capture group keeps the delimiters in the split, so the name is
# rebuilt as it was.
_SIDE_TOKEN_DELIMITER_RE = re.compile(r"([._ ])")

# Each side word, lowercased, with its partner and whether it is the left.
_SIDE_TOKENS = {
    "l": ("r", True),
    "r": ("l", False),
    "left": ("right", True),
    "right": ("left", False),
}

# A rule of the L/R name detection: from a name's base, each candidate for
# its partner's base, with whether the name is the left half.
_LrNameRule = Callable[[str], Iterator[tuple[str, bool]]]


def _side_token_partners(base: str) -> Iterator[tuple[str, bool]]:
    """``b_l_upleg`` to ``b_r_upleg``: a delimited side word, swapped in its own case."""
    parts = _SIDE_TOKEN_DELIMITER_RE.split(base)
    # Words sit at the even indices, delimiters at the odd ones.
    for i in range(0, len(parts), 2):
        token = parts[i]
        side = _SIDE_TOKENS.get(token.lower())
        if side is None:
            continue
        partner_token, is_left = side
        if token.isupper():
            partner_token = partner_token.upper()
        elif token[0].isupper():
            partner_token = partner_token.capitalize()
        yield "".join(parts[:i] + [partner_token] + parts[i + 1 :]), is_left


# The Japanese side characters, read as the name's first character only.
_JAPANESE_SIDES = {"左": ("右", True), "右": ("左", False)}


def _japanese_prefix_partners(base: str) -> Iterator[tuple[str, bool]]:
    """``左足`` to ``右足``: the side character first, swapped."""
    side = _JAPANESE_SIDES.get(base[:1])
    if side is not None:
        partner_char, is_left = side
        yield partner_char + base[1:], is_left


# A lowercase side letter, with its partner and whether it is the left.
_SIDE_LETTERS = {"l": ("r", True), "r": ("l", False)}


def _lowercase_prefix_partners(base: str) -> Iterator[tuple[str, bool]]:
    """``lThigh`` to ``rThigh``: a lowercase side letter before an uppercase letter."""
    side = _SIDE_LETTERS.get(base[:1])
    if side is not None and base[1:2].isupper():
        partner_letter, is_left = side
        yield partner_letter + base[1:], is_left


def _bare_prefix_partners(base: str) -> Iterator[tuple[str, bool]]:
    """``lfemur`` to ``rfemur``: a lowercase side letter with no delimiter."""
    side = _SIDE_LETTERS.get(base[:1])
    if side is not None and len(base) > 1:
        partner_letter, is_left = side
        yield partner_letter + base[1:], is_left


# How many pairs the bare prefix must give on a skeleton to count: one
# pair alone can be two names that differ by a first l / r by chance
# (``light`` / ``right``), and a convention names a whole skeleton.
_MIN_BARE_PREFIX_PAIRS = 2

# The rules tried after the three of ``_detect_lr_mapping_by_names``'s
# first pass, in order, each with the fewest pairs it must give to count.
_LR_NAME_RULES: tuple[tuple[_LrNameRule, int], ...] = (
    (_side_token_partners, 1),
    (_japanese_prefix_partners, 1),
    (_lowercase_prefix_partners, 1),
    (_bare_prefix_partners, _MIN_BARE_PREFIX_PAIRS),
)


def _pairs_by_rule(
    rule: _LrNameRule, joint_names: Sequence[str], paired: set[str]
) -> list[tuple[str, str]]:
    """The ``(left, right)`` pairs *rule* finds among the names not yet *paired*."""
    names = set(joint_names)
    pairs: list[tuple[str, str]] = []
    taken = set(paired)
    for name in joint_names:
        if name in taken:
            continue
        ns_prefix, base, number_suffix = _split_name(name)
        for partner_base, is_left in rule(base):
            partner = ns_prefix + partner_base + number_suffix
            if partner in names and partner not in taken:
                pairs.append((name, partner) if is_left else (partner, name))
                taken.update((name, partner))
                break
    return pairs


def _detect_lr_mapping_by_names(joint_names: Sequence[str]) -> dict[str, str]:
    """Internal: detect L/R joint pairs by extended name heuristics.

    Called by ``Bvh`` on first use of its L/R mapping, when the user set none.
    Rules tried most-specific-first so delimited suffixes win over bare
    substring matches:

    1. Delimited suffix — ``arm.L`` / ``arm.R``, ``arm_L`` / ``arm_R``,
       and their lowercase and ``.Left`` / ``_Left`` variants. Names are
       normalized by stripping a ``foo:`` namespace prefix (Mixamo's
       ``mixamorig:``) and a trailing ``.NNN`` number suffix (Blender's
       ``.001`` duplicates) before the suffix match; the partner is
       rebuilt with the same prefix/number suffix.
    2. Substring ``left`` / ``right`` (case-insensitive) — handles
       ``LeftArm`` / ``RightArm``, ``leftArm`` / ``rightArm``, and bare
       cases like ``LeftEye`` / ``RightEye`` with no delimiter. Namespace
       prefix is stripped before matching.
    3. Prefix ``L*`` / ``R*`` followed by an uppercase letter — handles
       ``LArm`` / ``RArm``.

    The rules of ``_LR_NAME_RULES`` follow, each in its own pass over
    the names the earlier rules left unpaired, so they cannot change a
    pair the three above give:

    4. A side word of its own, ``l``, ``r``, ``left`` or ``right`` in
       any case, between ``.``, ``_`` or spaces — ``b_l_upleg``,
       ``L_legUpper0001_bind_JNT``, ``Bip01 L Thigh``; the partner swaps
       that word in the same case.
    5. ``左`` / ``右`` as the first character — ``左足`` / ``右足``.
    6. A lowercase ``l`` / ``r`` before an uppercase letter — ``lThigh``.
    7. A lowercase ``l`` / ``r`` with no delimiter — ``lfemur``. Counted
       only when it gives ``_MIN_BARE_PREFIX_PAIRS`` pairs or more.

    A name is paired by the first rule that pairs it. Mutual-match
    requirement: both halves must exist as joint names. Singletons are
    not paired.

    Returns ``{}`` if no pairs detected. See ``bvh.lr_mapping`` for the
    public API that wraps this function.

    Parameters
    ----------
    joint_names : sequence of str
        The joints' names (end sites excluded), in joint order.
    """
    names = set(joint_names)
    mapping: dict[str, str] = {}
    seen: set[str] = set()

    for name in joint_names:
        if name in seen:
            continue

        partner: str | None = None

        ns_prefix, base, number_suffix = _split_name(name)

        # Strategy 1: delimited suffix (most specific)
        partner = _try_suffix_partner(base, names, ns_prefix, number_suffix)

        # Strategy 2: "left"/"right" substring (case-insensitive)
        # Work on the post-namespace base (no number suffix) so Mixamo
        # names like "mixamorig:LeftArm" match against "LeftArm".
        if partner is None:
            lower = base.lower()
            if "left" in lower:
                partner_base = (
                    base.replace("Left", "Right").replace("left", "right").replace("LEFT", "RIGHT")
                )
                candidate = ns_prefix + partner_base + number_suffix
                if candidate in names:
                    partner = candidate
            elif "right" in lower:
                partner_base = (
                    base.replace("Right", "Left").replace("right", "left").replace("RIGHT", "LEFT")
                )
                candidate = ns_prefix + partner_base + number_suffix
                if candidate in names:
                    partner = candidate

        # Strategy 3: "L" / "R" prefix followed by uppercase
        if partner is None and len(base) >= 2:
            if base[0] == "L" and base[1].isupper():
                partner_base = "R" + base[1:]
                candidate = ns_prefix + partner_base + number_suffix
                if candidate in names:
                    partner = candidate
            elif base[0] == "R" and base[1].isupper():
                partner_base = "L" + base[1:]
                candidate = ns_prefix + partner_base + number_suffix
                if candidate in names:
                    partner = candidate

        if partner is not None and partner not in seen:
            # Normalise: always store the "Left" / "L" version as key
            left, right = _order_lr_pair(name, partner)
            mapping[left] = right
            seen.add(left)
            seen.add(right)

    # The later rules pair only the names the three above left unpaired,
    # so that no pair they give can change.
    for rule, min_pairs in _LR_NAME_RULES:
        pairs = _pairs_by_rule(rule, joint_names, seen)
        if len(pairs) < min_pairs:
            continue
        for left, right in pairs:
            mapping[left] = right
            seen.update((left, right))

    return mapping


def _order_lr_pair(a: str, b: str) -> tuple[str, str]:
    """Return ``(left_name, right_name)``."""
    # Prefer substring form first (most common)
    for kw in ("Left", "left", "LEFT"):
        if kw in a:
            return (a, b)
        if kw in b:
            return (b, a)
    _, a_base, _ = _split_name(a)
    _, b_base, _ = _split_name(b)
    for left_suf, _right_suf in _SUFFIX_LR_TABLE:
        if a_base.endswith(left_suf):
            return (a, b)
        if b_base.endswith(left_suf):
            return (b, a)
    # Fallback: "L" prefix → left
    if a_base.startswith("L") and (len(a_base) < 2 or a_base[1].isupper()):
        return (a, b)
    if b_base.startswith("L") and (len(b_base) < 2 or b_base[1].isupper()):
        return (b, a)
    return (a, b)


def _iter_unique_lr_pairs(mapping: dict[str, str]):
    """Yield each ``(name_a, name_b)`` pair from an L/R mapping exactly once.

    Tolerates symmetric mappings (the public form of ``Bvh.lr_mapping``,
    which contains both directions of every pair) without double-counting.
    """
    seen: set[frozenset[str]] = set()
    for a, b in mapping.items():
        key = frozenset((a, b))
        if key in seen:
            continue
        seen.add(key)
        yield a, b


def _resolve_lr_pairs(
    mapping: dict[str, str] | None,
    name2idx: dict[str, int],
    *,
    strict: bool = False,
) -> list[tuple[int, int]]:
    """Resolve an L/R name mapping to ``(left_idx, right_idx)`` pairs.

    The single pair-resolution loop shared by every L/R consumer
    (``mirror``, ``Bvh.lr_pairs``, the orientation heuristics).
    Symmetric mappings (the public form of ``Bvh.lr_mapping``) are
    deduplicated to one tuple per pair via :func:`_iter_unique_lr_pairs`.

    Parameters
    ----------
    mapping : dict or None
        L/R name mapping.  ``None`` or empty resolves to ``[]``.
    name2idx : dict
        Name → index lookup defining the target index space (e.g.
        ``bvh.joint_index`` or ``bvh.node_index``).
    strict : bool, optional
        If True, raise ``ValueError`` listing every mapped name missing
        from ``name2idx`` (used to surface typos in explicitly passed
        mappings).  Default False: pairs with unknown names are skipped.
    """
    if not mapping:
        return []
    pairs: list[tuple[int, int]] = []
    unknown: set[str] = set()
    for left_name, right_name in _iter_unique_lr_pairs(mapping):
        if left_name in name2idx and right_name in name2idx:
            pairs.append((name2idx[left_name], name2idx[right_name]))
        else:
            unknown.update(n for n in (left_name, right_name) if n not in name2idx)
    if strict and unknown:
        raise ValueError(
            f"lr_mapping references unknown joint names: "
            f"{sorted(unknown)}. Check against `bvh.joint_names`."
        )
    return pairs


def _resolve_node_lr_pairs(
    mapping: dict[str, str] | None,
    node_names: Sequence[str],
    parent_idx: npt.NDArray[np.intp],
    is_end_site: npt.NDArray[np.bool_],
) -> tuple[list[tuple[int, int]], list[tuple[str, int, str, int]]]:
    """Resolve an L/R name mapping to ``nodes``-space pairs, end sites included.

    The single implementation behind :attr:`Bvh.node_lr_pairs` and
    :func:`~pybvh.transforms.mirror`, which need the same pairs but
    disagree on what to do about an unpairable joint — so this reports
    that case rather than deciding it.

    Parameters
    ----------
    mapping : dict or None
        L/R name mapping.  ``None`` or empty resolves to ``([], [])``.
    node_names : sequence of str
        Every node's name, in node order (depth first, as
        :attr:`Bvh.nodes`), end sites included.
    parent_idx : ndarray of shape (N,)
        Each node's parent index, ``-1`` for the root.
    is_end_site : ndarray of shape (N,), bool
        Which nodes are end sites.

    Returns
    -------
    pairs : list of (int, int)
        ``(left_idx, right_idx)`` in ``nodes`` index space: every joint
        pair in *mapping* order first, then the end-site pairs in that
        same order.  Deterministic, so downstream metadata built from it
        reproduces across runs.
    mismatches : list of (str, int, str, int)
        ``(left_name, left_end_count, right_name, right_end_count)`` for
        each pair whose two joints carry different numbers of end sites,
        so their tips cannot be zipped.  The joint pair itself is still
        returned in *pairs*; only its end sites are omitted.

    Notes
    -----
    Joint names are looked up among joints only — an end site whose
    generated display name collides with a joint's cannot shadow it — and
    a joint's end sites are found by position, as the end-site children
    of that joint in node order, since two end sites under one joint
    carry the same generated name.
    """
    if not mapping:
        return [], []

    joint_by_name: dict[str, int] = {}
    end_sites_of: dict[int, list[int]] = {}
    for i, name in enumerate(node_names):
        if is_end_site[i]:
            end_sites_of.setdefault(int(parent_idx[i]), []).append(i)
        else:
            joint_by_name[name] = i

    joint_pairs: list[tuple[int, int]] = []
    end_site_pairs: list[tuple[int, int]] = []
    mismatches: list[tuple[str, int, str, int]] = []

    for left_name, right_name in _iter_unique_lr_pairs(mapping):
        left_joint = joint_by_name.get(left_name)
        right_joint = joint_by_name.get(right_name)
        if left_joint is None or right_joint is None:
            continue
        joint_pairs.append((left_joint, right_joint))

        left_ends = end_sites_of.get(left_joint, [])
        right_ends = end_sites_of.get(right_joint, [])
        if len(left_ends) != len(right_ends):
            mismatches.append((left_name, len(left_ends), right_name, len(right_ends)))
            continue
        end_site_pairs.extend(zip(left_ends, right_ends, strict=True))

    return joint_pairs + end_site_pairs, mismatches


def _infer_rest_left(
    rest_positions: npt.NDArray[np.float64],
    lr_pairs: npt.ArrayLike,
    rest_up: str | None,
) -> str | None:
    """The skeleton's own left, read from the rest pose alone.

    Averages the left-minus-right rest positions over the L/R pairs,
    removes the component along rest up and snaps the rest to the
    nearest signed axis: the direction from the skeleton's right toward
    its left, in its own frame. :func:`~pybvh.transforms.mirror`
    reflects along it, and rest forward is built from it.

    Parameters
    ----------
    rest_positions : ndarray of shape (N, 3)
        The rest pose's node positions.
    lr_pairs : array_like of shape (P, 2)
        ``(left_idx, right_idx)`` rows into the N axis.
    rest_up : str or None
        The skeleton's own up, the axis :func:`_infer_rest_up` gives.

    Returns
    -------
    str or None
        Signed axis string (e.g. ``'-x'``), or ``None`` when there is no
        rest up, no pair, or the averaged offset lies along rest up.
    """
    if rest_up is None:
        return None
    pairs = np.asarray(lr_pairs, dtype=np.intp).reshape(-1, 2)
    if pairs.shape[0] == 0:
        return None

    avg_leftward = np.mean(rest_positions[pairs[:, 0]] - rest_positions[pairs[:, 1]], axis=0)
    avg_leftward[_AXIS_CHAR_TO_IDX[rest_up[1]]] = 0.0
    # With the up component removed, the snap can only land on a lateral
    # axis, or find nothing left to read.
    return get_main_direction(avg_leftward)


def _infer_rest_up_and_left(
    skeleton: _RestSkeleton, anchor: int
) -> tuple[_RestUp | None, str | None]:
    """Rest up and rest left, both read with the skeleton's own L/R pairs.

    Without a named up joint, rest up is read from the pairs (the leg
    pair), so a rest left measured with one mapping's pairs against the
    rest up of another's mixes two skeletons' frames. Every reader of
    rest left goes through here: the orientation report with the pairs of
    the mapping in effect, :func:`~pybvh.transforms.mirror` with those
    of the mapping it mirrors with. *anchor* is the body anchor's index
    (:func:`_find_body_anchor`), which rest up is read from.

    Returns
    -------
    tuple of (_RestUp or None, str or None)
        :func:`_infer_rest_up`'s reading and :func:`_infer_rest_left`'s
        against its axis.
    """
    rest_up = _infer_rest_up(skeleton, anchor)
    rest_up_axis = None if rest_up is None else rest_up.axis
    pairs = _lr_pairs_in_body(skeleton.lr_pairs, _find_body(skeleton, anchor))
    return rest_up, _infer_rest_left(skeleton.rest_positions, pairs, rest_up_axis)


def _lr_pairs_in_body(
    lr_pairs: npt.NDArray[np.intp], in_body: npt.NDArray[np.bool_]
) -> npt.NDArray[np.intp]:
    """The rows of *lr_pairs* whose two nodes are both in the body.

    *in_body* is :func:`_find_body`'s mask. Rest left and the facing read
    these pairs, so a second figure or a prop beside the body is never
    averaged into the body's left. The rows keep their order, so a body
    that holds every pair reads the same pairs as the whole skeleton.
    """
    return lr_pairs[in_body[lr_pairs].all(axis=1)]


# The up axis the house defaults are keyed on when a skeleton has none: the
# same '+y' the world-up guess falls back to.
_HOUSE_UP = "+y"


def _infer_rest_forward(rest_left: str | None, rest_up: str | None) -> str:
    """The direction the rest pose faces in the skeleton's own frame.

    ``rest_left × rest_up`` when both exist. Without a rest left (no
    usable L/R geometry) it is the house default for rest up,
    arbitrary but stable and perpendicular to it; without a rest up
    either, the default for ``'+y'``. Never read against world up: the
    rest pose's coordinates are the skeleton's frame, and only a root
    rotation places them in the world (ADR 0004).

    The default is returned silently; the caller that hands it on as a
    fact warns (:func:`_warn_rest_forward_defaulted`).
    """
    if rest_left is None or rest_up is None:
        return _FALLBACK_FORWARD[(rest_up or _HOUSE_UP)[1]]
    forward = get_main_direction(
        np.cross(_axis_to_vector(rest_left), _axis_to_vector(rest_up))  # type: ignore[arg-type]  # NumPy 2.2's stub types a float cross product as floating, not float64
    )
    # Rest left is perpendicular to rest up by construction, so the cross
    # product is a unit axis.
    assert forward is not None
    return forward


def _warn_rest_forward_defaulted(rest_forward: str, rest_up: str | None) -> None:
    """Warn that :attr:`Bvh.rest_forward` is the house default, not a measurement."""
    keyed_on = f"rest up {rest_up!r}" if rest_up is not None else f"the house up {_HOUSE_UP!r}"
    # The returned axis is a house default and its type cannot say so,
    # the same silent substitution _infer_world_up warns about. Python's
    # default filter shows this once per line of the user's code.
    warnings.warn(
        f"No usable left/right geometry in this skeleton's rest pose, so "
        f"the direction it faces cannot be measured; falling back to "
        f"{rest_forward!r}, the arbitrary-but-stable axis for {keyed_on}. "
        f"rest_forward, rest_forward_axis and the heading root_trajectory "
        f"builds on it are a default here, not a property of this file. "
        f"Set `bvh.lr_mapping = {{...}}` to fix it; "
        f"`bvh.orientation_report.rest_forward_source` is 'default' here.",
        UserWarning,
        stacklevel=user_stacklevel(),
    )


class _WorldUpGuess(NamedTuple):
    """A guessed world up and what it was read from.

    ``source`` is ``"first_frame"`` (the up marker above the body anchor
    at frame 0), ``"rest_up"`` (the first frame could not decide, so the
    skeleton's own up stands in) or ``"default"`` (neither said
    anything: the house up).
    """

    axis: str
    source: _WorldUpGuessSource


# Frame 0 decides world up when the largest component of its
# marker-above-anchor direction is at least this many times the second; below
# it the direction has no clear main axis (a torso bent or leaning far from
# every axis), and the frame is ambiguous. A straight horizontal pose with
# its head near a horizontal axis decides, on that axis.
_FIRST_FRAME_DOMINANCE = 2.0


def _dominant_axis(direction: npt.NDArray[np.float64]) -> str | None:
    """*direction*'s main axis, when its largest component is at least
    ``_FIRST_FRAME_DOMINANCE`` times the second; ``None`` otherwise, and
    for a direction too short to read (:func:`get_main_direction`'s
    threshold)."""
    second, largest = np.sort(np.abs(direction))[-2:]
    # Written as a failed acceptance, not as `largest < ...`, so that a NaN
    # component (a NaN root position written by the user) does not decide.
    if not largest >= _FIRST_FRAME_DOMINANCE * second:
        return None
    return get_main_direction(direction)


def _infer_world_up(
    skeleton: _RestSkeleton,
    anchor: int,
    frame0_positions: npt.NDArray[np.float64] | None,
) -> _WorldUpGuess:
    """Guess world up from a clip's first frame, with its source.

    The direction from the body anchor to the up marker (:func:`_up_marker_index`,
    the node rest up reads) at frame 0, snapped to its main axis, when
    its largest component is at least twice the second. Otherwise (no
    frame, no marker, an ambiguous frame-0 pose) the skeleton's own up
    (:func:`_infer_rest_up`) stands in, and without one the house up
    ``'+y'``. Returns the guess silently: the reader of world up warns
    (:func:`_warn_world_up_defaulted`, :func:`_warn_world_up_disagreement`,
    :func:`_warn_rest_up_contradicted_by_anchor`).

    Parameters
    ----------
    skeleton : _RestSkeleton
        The skeleton the clip animates, in its rest pose.
    anchor : int
        The body anchor's index (:func:`_find_body_anchor`).
    frame0_positions : ndarray of shape (N, 3) or None
        The node positions at frame 0, ``None`` for a clip without frames.

    Returns
    -------
    _WorldUpGuess
    """
    marker = _up_marker_index(skeleton, anchor)
    if frame0_positions is not None and marker is not None:
        frame_up = _dominant_axis(frame0_positions[marker] - frame0_positions[anchor])
        if frame_up is not None:
            return _WorldUpGuess(frame_up, "first_frame")

    rest_up = _infer_rest_up(skeleton, anchor)
    if rest_up is not None:
        return _WorldUpGuess(rest_up.axis, "rest_up")
    return _WorldUpGuess(_HOUSE_UP, "default")


def _warn_world_up_defaulted() -> None:
    """Warn that world up is the house default, not a measurement."""
    # Neither the animation nor the rest pose said anything, so '+y' is
    # invented, not measured, and the returned string cannot say so; the
    # same silent substitution _warn_facing_defaulted_no_pairs warns about.
    warnings.warn(
        f"Could not infer a world up axis: this skeleton has no "
        f"recognizable head/neck/chest/spine joint to read the first "
        f"frame from, and its rest pose is degenerate (single node, or "
        f"all offsets coincident). Falling back to {_HOUSE_UP!r}, the house "
        f"default — anything derived from up (floor_height, "
        f"foot_contacts, facing, ground-plane projections) rests on a "
        f"guess here. Set `bvh.world_up = '<axis>'` to fix it; "
        f"`bvh.orientation_report.world_up_source` is 'default' here.",
        UserWarning,
        stacklevel=user_stacklevel(),
    )


def _warn_world_up_disagreement(rest_up: str, frame_up: str) -> None:
    """Warn that the rest pose and the first frame disagree on world up.

    Only the axis letter is compared by the caller, so a sign flip alone
    (rest pose ``'+y'``, first frame ``'-y'``) never warns.
    """
    # Two causes give it: a rest pose authored in another convention than
    # the motion, where frame 0 is right, and a first frame that is not
    # upright, where the rest pose's axis may be. The guess cannot tell
    # them apart, so the warning names both.
    warnings.warn(
        f"Rest pose suggests world up is {rest_up!r} but at the first "
        f"animation frame the direction from the body anchor to the head joint "
        f"(else the neck, chest or spine) is closer to {frame_up!r}. "
        f"Using {frame_up!r} from the animation data. Either the rest pose "
        f"is authored in another convention than the motion, and "
        f"{frame_up!r} is right, or the first frame is not upright (the "
        f"performer lies, kneels or bows), and {rest_up!r} may be. "
        f"If {frame_up!r} is wrong for your file, set it explicitly via "
        f"`bvh.world_up = '<axis>'`. `bvh.orientation_report` shows the "
        f"source of each axis.",
        UserWarning,
        stacklevel=user_stacklevel(),
    )


def _world_up_disagrees(guess: _WorldUpGuess, rest_up: str | None) -> bool:
    """Whether a first-frame guess names another axis than rest up.

    Only the axis letter counts: a sign flip alone is not a disagreement.
    """
    return guess.source == "first_frame" and rest_up is not None and rest_up[1] != guess.axis[1]


def _anchor_contradicts_rest_up(
    guess: _WorldUpGuess, anchor_rotation: npt.NDArray[np.float64] | None
) -> str | None:
    """The axis the body anchor's world rotation at frame 0 carries rest up
    onto, when a guess that fell back on rest up names another axis.

    The body anchor's world rotation is what carries a fact of the
    skeleton's own frame into the world (ADR 0004); the rest-up fallback
    skips it. Rest up
    carried by it is snapped as frame 0's head direction is
    (:func:`_dominant_axis`), and only the axis letter is compared, as in
    :func:`_world_up_disagrees`. A disagreement says the rest pose may be
    authored in another convention than the motion; it is a warning, not
    a better guess, since a pelvis tilted at frame 0 reads a horizontal
    axis as confidently.

    Parameters
    ----------
    guess : _WorldUpGuess
        The clip's guess; only a ``"rest_up"`` one is read.
    anchor_rotation : ndarray of shape (3, 3) or None
        The body anchor's world rotation matrix at frame 0, ``None`` for a
        clip without frames.

    Returns
    -------
    str or None
        The carried axis, or ``None`` when the guess did not fall back on
        rest up, the carried direction does not decide, or it names rest
        up's own axis.
    """
    if guess.source != "rest_up" or anchor_rotation is None:
        return None
    carried = _dominant_axis(anchor_rotation @ _axis_to_vector(guess.axis))
    if carried is None or carried[1] == guess.axis[1]:
        return None
    return carried


def _warn_rest_up_contradicted_by_anchor(rest_up: str, carried: str) -> None:
    """Warn that world up fell back on rest up while the body anchor's world
    rotation at frame 0 carries rest up onto another axis."""
    # The fallback is otherwise silent: the report's "rest_up" source is
    # its only trace, and on a rest pose authored in another convention
    # than the motion it is wrong.
    warnings.warn(
        f"World up fell back on the rest pose's up {rest_up!r}: at the "
        f"first animation frame the direction from the body anchor to the head "
        f"joint (else the neck, chest or spine) does not decide it. But "
        f"the body anchor's rotation at that frame carries the rest pose's up "
        f"onto {carried!r}, so the rest pose may be authored in another "
        f"convention than the motion, and world up may be {carried!r}. "
        f"Check the clip and set it explicitly via "
        f"`bvh.world_up = '<axis>'`; `bvh.orientation_report.world_up_source` "
        f"is 'rest_up' here.",
        UserWarning,
        stacklevel=user_stacklevel(),
    )


# ---------------------------------------------------------------------------
# The orientation report: every clip-level fact, with how it was obtained
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OrientationReport:
    """How each orientation fact of a clip was obtained.

    Returned by :attr:`Bvh.orientation_report <pybvh.bvh.Bvh.orientation_report>`.
    ``rest_up``, ``rest_forward`` and ``world_up`` are the values the
    members of the same names return; each ``*_source`` says whether the
    value was set, measured or defaulted, and how, so a default never
    passes for a measurement (``rest_forward`` and ``world_up`` always
    answer, whatever the clip carries, on a clip with a body). Frozen: a
    clip's report is replaced, never edited, when what it was read from
    changes.

    On a clip with no body (``body_anchor`` ``None``) the facts read from
    the body are ``None``, with their sources: ``rest_up``, ``rest_left``,
    ``rest_forward`` and a guessed ``world_up``. The members of the same
    names raise there; the report does not. A ``world_up`` set or
    inherited is reported as on any clip.

    The facts belong to two frames (ADR 0004). The *skeleton's own
    frame* is read from the rest pose alone, the skeleton with every
    rotation zero, and never from world up: ``rest_up``, ``rest_left``,
    ``rest_forward``. The *world frame* is read from the motion:
    ``world_up``. The world rotation of the body anchor at a frame is
    what carries a skeleton fact into the world. Per-frame facts (the facing at a
    frame) are not here: :meth:`~pybvh.bvh.Bvh.forward_at`,
    :meth:`~pybvh.bvh.Bvh.left_at` and
    :meth:`~pybvh.bvh.Bvh.facing_frame` carry them, with
    ``FacingFrame.valid`` as the per-frame mark.

    Attributes
    ----------
    rest_up : str or None
        The skeleton's own up, a signed axis string, ``None`` when the
        rest pose carries no direction (a single node, or every node at
        one point).
    rest_up_source : {"named_joint", "leg_pair", "largest_spread"} or None
        ``"named_joint"``: the main axis from the body anchor to the
        first joint of the body whose name holds ``head`` as a word, else ``neck``,
        ``chest`` or ``spine`` (``Head``, ``mixamorig:Head``, ``Bip01
        Head``; not ``Forehead``).
        ``"leg_pair"``: no such joint, so the axis opposite the legs:
        from the far ends of the two branches of the body anchor that
        an L/R pair is split across, to the anchor.
        ``"largest_spread"``: no such pair either, so the axis along which the body spreads most in the rest pose, signed by
        the side most of the body lies on, a weak reading (wrong in axis
        on a T-pose wider than tall, in sign when the hips sit
        mid-body). ``None`` with ``rest_up``.
    rest_left : str or None
        The skeleton's own left: the left-minus-right rest positions of
        the L/R pairs, averaged, with the ``rest_up`` component removed,
        snapped to an axis. ``None`` without pairs, without ``rest_up``,
        or when the pairs lie along ``rest_up``.
    rest_forward : str or None
        The direction the rest pose faces in the skeleton's own frame;
        ``None`` on a clip with no body.
    rest_forward_source : {"lr_geometry", "default"} or None
        ``"lr_geometry"``: ``rest_left × rest_up``, measured.
        ``"default"``: no ``rest_left``, so the house default for
        ``rest_up`` (``'+y'`` gives ``'+z'``, ``'+z'`` gives ``'+x'``,
        ``'+x'`` gives ``'+y'``; ``'+z'`` without a ``rest_up``),
        arbitrary but stable. ``None`` with ``rest_forward``.
    world_up : str or None
        The world's vertical, a signed axis string; ``None`` on a clip
        with no body, where it is neither set nor inherited.
    world_up_source : {"set", "first_frame", "rest_up", "default"} or None
        ``"set"``: assigned by the user, passed as ``world_up=`` at load,
        or the target of ``reorient_world_up``. ``"first_frame"``:
        guessed from the direction of the ``rest_up`` node from the
        body anchor at frame 0. ``"rest_up"``: the first frame could not decide
        (no frames, no such node, or a crouched, lying or leaning first
        pose), so the skeleton's own up stands in, a guess that assumes
        the rest pose is authored upright in the world.
        ``"default"``: neither said anything, so the house up ``'+y'``.
        ``None`` with ``world_up``.
    world_up_inherited : bool
        True when the guess was made on the clip this one was derived
        from (a slice, a copy, a transformed clip, a pickle) and taken
        over as it was: ``world_up_source`` is then the source of that
        original guess. False for an axis set by the user and for a
        guess made on this clip's own frames.
    lr_mapping_source : {"set", "names"} or None
        Where the L/R mapping the rest left is read from comes from:
        ``"set"`` by the user (``bvh.lr_mapping = {...}`` or
        ``lr_mapping=`` at load), ``"names"`` detected from the joint
        names, ``None`` when there is none (the names give no pairs, or
        the user cleared the mapping).
    body_anchor : BvhNode or None
        The joint the body's measures start from, a node of the clip:
        what :attr:`Bvh.body_anchor <pybvh.bvh.Bvh.body_anchor>` returns.
        ``None`` when the clip has no body.
    body_anchor_source : {"set", "lr_pairs", "default", "ambiguous"}
        ``"set"``: the joint the user set, assigned to
        :attr:`Bvh.body_anchor <pybvh.bvh.Bvh.body_anchor>` or passed as
        ``body_anchor=`` at load; the rule is not run.
        ``"lr_pairs"``: where the L/R pairs of the mapping in effect
        meet. Each pair meets at the lowest joint that both its joints'
        chains to the root pass through; the anchor is the meeting joint
        closest to the root, so the legs, which meet at the hips, win
        over the arms, which meet at the chest. Of several at the same
        depth, the one with the most joints below it, end sites not
        counted, when each other one has fewer than half as many.
        ``"ambiguous"``: no body, as two of those meeting joints are
        that close in size, for instance two figures under one root;
        the anchor is then ``None``. ``"default"``: no pair, so the root.
    body_anchor_candidates : tuple of BvhNode
        The joints that competed for the anchor, the most joints below
        them first: the meeting joints the rule compared, the anchor alone
        when the pairs meet at one joint closest to the root. Under a
        static root, when the trees below it give no body, the candidate
        of each tree with at least half the joints of the largest, the
        largest tree first; when the largest tree is kept, the meeting
        joints compared inside it. Empty for ``"set"`` and
        ``"default"``.
    """

    rest_up: str | None
    rest_up_source: _RestUpSource | None
    rest_left: str | None
    rest_forward: str | None
    rest_forward_source: _RestForwardSource | None
    world_up: str | None
    world_up_source: _WorldUpSource | None
    world_up_inherited: bool
    lr_mapping_source: _LrMappingSource | None
    body_anchor: BvhNode | None
    body_anchor_source: _BodyAnchorSource
    body_anchor_candidates: tuple[BvhNode, ...]


def _build_orientation_report(
    skeleton: _RestSkeleton,
    world_up: str | _WorldUpGuess | None,
    world_up_inherited: bool,
    lr_mapping_source: _LrMappingSource | None,
    body_anchor: _BodyAnchor,
    nodes: Sequence[BvhNode],
) -> OrientationReport:
    """Read the skeleton's own frame from the rest pose, beside world up.

    Parameters
    ----------
    skeleton : _RestSkeleton
        The skeleton in its rest pose, with the pairs of the L/R mapping
        in effect.
    world_up : str or _WorldUpGuess or None
        World up as the caller settled it: a signed axis string the user
        set (reported as ``"set"``), or the guess it stands on
        (:func:`_infer_world_up`), whose source is reported; ``None``
        when there is neither, on a clip with no body.
    world_up_inherited : bool
        Whether that guess was made on the clip this one was derived
        from; False for an axis the user set.
    lr_mapping_source : {"set", "names"} or None
        The source of the L/R mapping the skeleton's pairs come from, the
        caller's setting.
    body_anchor : _BodyAnchor
        The body anchor as the caller found it, in node index space.
        With no body (``node`` ``None``), the facts read from the body
        are ``None``.
    nodes : sequence of BvhNode
        The clip's nodes, which the anchor's indices are read into.

    Returns
    -------
    OrientationReport
    """
    world_up_axis: str | None
    world_up_source: _WorldUpSource | None
    if world_up is None:
        world_up_axis = world_up_source = None
    elif isinstance(world_up, _WorldUpGuess):
        world_up_axis, world_up_source = world_up
    else:
        world_up_axis, world_up_source = world_up, "set"
    rest_forward: str | None = None
    rest_forward_source: _RestForwardSource | None = None
    rest_up = rest_left = None
    if body_anchor.node is not None:
        rest_up, rest_left = _infer_rest_up_and_left(skeleton, body_anchor.node)
        rest_forward = _infer_rest_forward(rest_left, None if rest_up is None else rest_up.axis)
        rest_forward_source = "lr_geometry" if rest_left is not None else "default"
    return OrientationReport(
        rest_up=None if rest_up is None else rest_up.axis,
        rest_up_source=None if rest_up is None else rest_up.source,
        rest_left=rest_left,
        rest_forward=rest_forward,
        rest_forward_source=rest_forward_source,
        world_up=world_up_axis,
        world_up_source=world_up_source,
        world_up_inherited=world_up_inherited,
        lr_mapping_source=lr_mapping_source,
        body_anchor=None if body_anchor.node is None else nodes[body_anchor.node],
        body_anchor_source=body_anchor.source,
        body_anchor_candidates=tuple(nodes[index] for index in body_anchor.candidates),
    )


def _leftward_units_from_pairs(
    coords: npt.NDArray[np.float64],
    lr_pairs: npt.NDArray[np.intp],
    up_vec: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Per-frame leftward **unit vectors**, measured against world up.

    The single implementation of the facing chain's first stage, and of
    the leftward bvhplot's follow camera turns with: average ``(left_pos
    - right_pos)`` over the L/R pairs, project it on the ground (the
    plane normal to ``up_vec``), normalize. Continuous, not snapped.

    Parameters
    ----------
    coords : ndarray of shape (F, N, 3)
        World-space node positions per frame, in the skeleton's length
        unit, indexed along N by ``lr_pairs``.
    lr_pairs : int ndarray of shape (P, 2)
        ``(left_idx, right_idx)`` rows into the N axis; ``(0, 2)`` means
        no pairs, and every frame comes back invalid.
    up_vec : ndarray of shape (3,)
        Signed world-up unit vector (:func:`_axis_to_vector` of the
        world-up string).

    Returns
    -------
    leftward : ndarray of shape (F, 3)
        Leftward unit vector per frame. Rows where ``valid`` is False
        carry no usable direction and must not be consumed.
    valid : ndarray of shape (F,), bool
        False for every frame when there are no pairs; per frame, False
        where the averaged leftward lies along world up or is zero
        (shorter than ``_MIN_GROUND_NORM`` on the ground).
    """
    num_frames = coords.shape[0]
    lr_pairs = np.asarray(lr_pairs, dtype=np.intp).reshape(-1, 2)
    if lr_pairs.shape[0] == 0:
        return (np.zeros((num_frames, 3)), np.zeros(num_frames, dtype=bool))

    left_idx = lr_pairs[:, 0]
    right_idx = lr_pairs[:, 1]

    leftward = np.mean(coords[:, left_idx] - coords[:, right_idx], axis=1)
    return _ground_units(leftward, up_vec)


# The shortest a direction projected on the ground may be and still be read,
# for the leftward and the heading alike.
_MIN_GROUND_NORM = 1e-6


def _ground_units(
    vectors: npt.NDArray[np.float64],
    up_vec: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """*vectors* projected on the ground (the plane normal to *up_vec*), as units.

    Returns the projected rows, normalised where they are readable, and
    ``valid``, ``False`` on the rows whose projection is shorter than
    ``_MIN_GROUND_NORM`` (a vector along up, or zero); those rows are
    left as projected and carry no direction.
    """
    projected = vectors - (vectors @ up_vec)[:, np.newaxis] * up_vec
    norms = np.linalg.norm(projected, axis=1)
    valid = norms >= _MIN_GROUND_NORM
    projected[valid] /= norms[valid, np.newaxis]
    return projected, valid


def _heading_units(
    anchor_rotmats: npt.NDArray[np.float64],
    rest_forward: str,
    world_up: str,
) -> npt.NDArray[np.float64]:
    """The body's heading per frame: rest forward carried into the world.

    The body anchor's world rotation at each frame applied to *rest_forward* (a
    direction in the skeleton's own frame), projected on the ground of
    *world_up* and normalised. Where that projection is too short to read
    (the root's forward stands along world up) the row is ``NaN``, the
    :mod:`pybvh.geometry` policy for a degenerate input: a number there
    would be an arbitrary angle.

    Parameters
    ----------
    anchor_rotmats : ndarray of shape (F, 3, 3)
        The body anchor's world rotation matrix per frame.
    rest_forward : str
        Signed axis string, the direction the rest pose faces.
    world_up : str
        Signed axis string for the world vertical axis.

    Returns
    -------
    ndarray of shape (F, 3)
        Unit ground vectors, ``NaN`` on the unreadable rows.
    """
    forward = np.einsum("fij,j->fi", anchor_rotmats, _axis_to_vector(rest_forward))
    heading, valid = _ground_units(forward, _axis_to_vector(world_up))
    heading[~valid] = np.nan
    return heading


def _fallback_leftward(world_up: str) -> npt.NDArray[np.float64]:
    """The house default facing, stored once as a leftward.

    ``up × forward``, where forward is ``_FALLBACK_FORWARD``'s
    arbitrary-but-stable axis for *world_up*, so that the chain's
    ``forward = left × up`` gives that axis back. A unit axis vector
    perpendicular to world up.
    """
    up_vec = _axis_to_vector(world_up)
    return np.cross(up_vec, _axis_to_vector(_FALLBACK_FORWARD[world_up[1]]))  # type: ignore[return-value]  # NumPy 2.2's stub types a float cross product as floating, not float64


def _facing_basis(
    coords: npt.NDArray[np.float64],
    lr_pairs: npt.NDArray[np.intp],
    world_up: str,
    anchor_rotmats: npt.NDArray[np.float64] | None = None,
    rest_forward: str | None = None,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.bool_],
    npt.NDArray[np.bool_],
]:
    """The facing chain: the per-frame basis under ``facing_frame``.

    One chain gives each row a leftward measured against world up: the
    frame's own leftward (:func:`_leftward_units_from_pairs`); where that
    is unreadable (the left/right axis vertical, the body on its side, or
    the pairs coinciding),
    the heading at that frame turned a quarter turn about up
    (``left = up × heading``); where that is unreadable too, or not
    given, the house default (:func:`_fallback_leftward`). Then
    ``forward = left × up`` and ``up`` is the exact world-up vector on
    every row, so ``forward × left = up`` everywhere, fallback rows
    included, and left never lies along world up. One up throughout:
    nothing here reads the rest pose (ADR 0004).

    Parameters
    ----------
    coords : ndarray of shape (F, N, 3)
        World-space node positions per frame.
    lr_pairs : int ndarray of shape (P, 2)
        ``(left_idx, right_idx)`` rows into the N axis.
    world_up : str
        Signed axis string for the world vertical axis.
    anchor_rotmats : ndarray of shape (F, 3, 3), optional
        The body anchor's world rotation at each row's frame.
    rest_forward : str, optional
        The direction the rest pose faces, read with *anchor_rotmats* into
        the heading (:func:`_heading_units`). Pass it only
        when it is measured: the heading of a defaulted rest forward is
        no measurement either. Without both, the heading stage is
        skipped.

    Returns
    -------
    forward, left, up : ndarray of shape (F, 3) each
        Unit vectors forming a right-handed basis per row.
    valid : ndarray of shape (F,), bool
        True where the row is the frame's own measurement, False on
        every fallback row.
    defaulted : ndarray of shape (F,), bool
        True on the rows that carry the house default.
    """
    up_vec = _axis_to_vector(world_up)
    leftward, valid = _leftward_units_from_pairs(coords, lr_pairs, up_vec)

    defaulted = ~valid
    if anchor_rotmats is not None and rest_forward is not None and defaulted.any():
        # Only the rows the frame's own leftward leaves unread.
        rows = np.flatnonzero(defaulted)
        heading = _heading_units(anchor_rotmats[rows], rest_forward, world_up)
        readable = ~np.isnan(heading).any(axis=1)
        leftward[rows[readable]] = np.cross(up_vec, heading[readable])
        defaulted[rows[readable]] = False
    if defaulted.any():
        leftward[defaulted] = _fallback_leftward(world_up)

    forward = np.cross(leftward, up_vec)
    up = np.tile(up_vec, (coords.shape[0], 1))
    return forward, leftward, up, valid, defaulted  # type: ignore[return-value]  # NumPy 2.2's stub types a float cross product as floating, not float64


def _facing_labels(forward: npt.NDArray[np.float64], world_up: str) -> tuple[str, str]:
    """The snapped labels of one facing row: ``(forward, left)``.

    *forward* (one row of :func:`_facing_basis`) snapped to the nearest
    signed axis, and left the axis ``up × forward`` of that label, so the
    labels form a right-handed triple with world up even on a row at 45
    degrees between two axes.
    """
    forward_label = get_main_direction(forward)
    # A forward row is a unit vector perpendicular to world up, so the
    # snap never degenerates and never lands on the up axis.
    assert forward_label is not None
    left_label = get_main_direction(
        np.cross(_axis_to_vector(world_up), _axis_to_vector(forward_label))  # type: ignore[arg-type]  # NumPy 2.2's stub types a float cross product as floating, not float64
    )
    assert left_label is not None
    return forward_label, left_label


def _warn_facing_defaulted_no_pairs(world_up: str) -> None:
    """Warn that every facing row carries the house default: the skeleton
    has no L/R pairs to read a facing from."""
    fallback = _FALLBACK_FORWARD[world_up[1]]
    # The returned axis is a house default, not a measurement, and the
    # return type cannot say so, the same silent substitution
    # _infer_world_up warns about. Python's default filter shows this once
    # per line of the user's code, so a loop over frames stays quiet.
    warnings.warn(
        f"No usable left/right geometry on this skeleton, so its facing "
        f"direction cannot be measured; falling back to {fallback!r}, the "
        f"arbitrary-but-stable horizontal axis for world_up={world_up!r}. "
        f"The facing (forward_at, left_at, facing_frame) is a default "
        f"here, not a property of this file. Set `bvh.lr_mapping = {{...}}` "
        f"to fix it; `bvh.orientation_report.lr_mapping_source` says where "
        f"the L/R mapping comes from, None when there is none.",
        UserWarning,
        stacklevel=user_stacklevel(),
    )


def _warn_facing_defaulted_at_frames(world_up: str, frames: Sequence[int]) -> None:
    """Warn that the facing rows of clip *frames* carry the house default."""
    shown = ", ".join(str(f) for f in frames[:5])
    more = f" and {len(frames) - 5} more" if len(frames) > 5 else ""
    _warn_rows_facing_defaulted(world_up, f"At frame{'s' if len(frames) > 1 else ''} {shown}{more}")


def _warn_facing_defaulted_on_pose(world_up: str) -> None:
    """Warn that the facing of a pose with no clip frame is the house default."""
    _warn_rows_facing_defaulted(world_up, "On the pose given")


def _warn_rows_facing_defaulted(world_up: str, where: str) -> None:
    """The warning of the two above, for the rows *where* names."""
    fallback = _FALLBACK_FORWARD[world_up[1]]
    warnings.warn(
        f"{where}, the facing direction cannot be measured: the left/right "
        f"joints give no direction on the ground (they line up with "
        f"world_up={world_up!r}, a body on its side, or coincide), and the "
        f"body's heading cannot stand in (it lies along world_up too, or "
        f"rest_forward is itself a default). Falling back to {fallback!r}, "
        f"the arbitrary-but-stable horizontal axis for that up. "
        f"`facing_frame().valid` is False on such frames.",
        UserWarning,
        stacklevel=user_stacklevel(),
    )
