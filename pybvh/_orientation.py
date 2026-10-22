"""The orientation a skeleton's geometry implies.

The skeleton's own frame read from the rest pose alone (rest up, rest
left, rest forward), the world up axis inferred from a clip's first frame,
the left/right joint pairs found by name and their resolution into
joint-space and node-space index pairs, and the per-frame leftward,
forward and facing basis, with the house default for a skeleton that
carries no left/right information; and :class:`OrientationReport`, which
says for each clip-level fact whether it was set, measured or defaulted.
:class:`~pybvh.bvh.Bvh` exposes their results as ``world_up``,
``rest_up``, ``rest_forward``, ``lr_mapping``, ``lr_pairs``,
``forward_at``, ``left_at`` and ``facing_frame``; ``transforms``,
``analysis`` and bvhplot call them rather than keep their own. The axis
vocabulary they are written in (signed axis strings, :class:`Axis`,
:func:`get_main_direction`) is :mod:`pybvh.tools`'.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, NamedTuple

import numpy as np
import numpy.typing as npt

from ._warnings import user_stacklevel
from .tools import _AXIS_CHAR_TO_IDX, _axis_to_vector, get_main_direction

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
#      reorientations and root_trajectory's heading, where the root
#      rotation carries rest forward into the world.
#   2. THE WORLD FRAME — world up, guessed from the first frame through the
#      same up marker (_infer_world_up); the root's heading
#      (_root_heading_units: the root rotation carries rest forward into
#      the world, NaN where its ground projection is unreadable); and the
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
#           │                       │
#           ▼                       ▼
#     _leftward_units_from_pairs ── the ONLY place the per-frame leftward
#           │   (array kernel)      geometry is computed; bvhplot
#           │                       follow-mode azimuths read it too
#           ▼
#     _facing_basis: the frame's leftward, else the root's heading at that
#           │        frame (_root_heading_units), else the house default
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


# The body parts whose direction from the root says which way is up, most
# telling first.
_UP_BODY_PARTS = ("head", "neck", "chest", "spine")

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
    """

    node_names: Sequence[str]
    rest_positions: npt.NDArray[np.float64]
    parent_idx: npt.NDArray[np.intp]
    is_end_site: npt.NDArray[np.bool_]
    lr_pairs: npt.NDArray[np.intp]


def _up_marker_index(skeleton: _RestSkeleton) -> int | None:
    """The node whose direction from the root says which way is up.

    The one body-part lookup for "up", shared by the rest up and the
    world-up guess so the two read the same node: the first joint, in node
    order, whose name holds ``head`` as a word, else ``neck``, ``chest``
    or ``spine``, in any case, after its namespace and number suffix
    (:func:`_name_words`). So ``Head``, ``mixamorig:Head``, ``Bip01
    Head``, ``CC_Base_Head``, ``J_Bip_C_Head`` and ``HeadTop_End`` (a
    joint at the top of the head) are heads, and ``Forehead`` is not. The
    alternative, the whole name alone, misses every rig whose convention
    prefixes its joints; a substring match would take ``Forehead``. End
    sites are not looked up: an end site's name is generated from its
    parent's and names no body part of its own. A joint whose rest
    position coincides with the root's carries no direction and is
    skipped, the root itself included.

    Parameters
    ----------
    skeleton : _RestSkeleton
        The node names and the rest pose.

    Returns
    -------
    int or None
        The node's index, or ``None`` when no node qualifies.
    """
    local = skeleton.rest_positions - skeleton.rest_positions[0]
    words = [
        set() if end_site else _name_words(name)
        for name, end_site in zip(skeleton.node_names, skeleton.is_end_site, strict=True)
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
    the axis along which the rest pose spreads most).
    """

    axis: str
    source: _RestUpSource


def _pelvis_branches(skeleton: _RestSkeleton) -> tuple[int, npt.NDArray[np.intp]] | None:
    """The node the legs hang from, and which of its branches each node is in.

    The pelvis is the first node, walking down from the root while a node
    has a single joint child, with two or more joint children: the root
    itself on a usual rig (``Hips`` -> legs and spine), ``Hips`` on a
    ``Root`` -> ``Hips`` -> legs rig. End sites are not counted as children:
    no joint of a pair lies below one.

    Returns
    -------
    tuple of (int, ndarray of shape (N,)) or None
        The pelvis's index, and for each node the index of the pelvis's
        child whose subtree holds it (``-1`` for the pelvis and the nodes
        above it); ``None`` when the walk ends without a branch.
    """
    parent_idx = skeleton.parent_idx
    joint_children: list[list[int]] = [[] for _ in parent_idx]
    for node in range(1, len(parent_idx)):
        if not skeleton.is_end_site[node]:
            joint_children[parent_idx[node]].append(node)
    pelvis = 0
    while len(joint_children[pelvis]) == 1:
        pelvis = joint_children[pelvis][0]
    if len(joint_children[pelvis]) < 2:
        return None
    branch = np.full(len(parent_idx), -1, dtype=np.intp)
    # Parents precede children, so one pass in node order hands each node
    # its parent's branch; a child of the pelvis starts its own.
    for node in range(pelvis + 1, len(parent_idx)):
        parent = parent_idx[node]
        branch[node] = node if parent == pelvis else branch[parent]
    return pelvis, branch


def _up_from_leg_pair(skeleton: _RestSkeleton) -> str | None:
    """Rest up opposite the legs: the L/R pair split across the pelvis.

    The legs are the first L/R pair, in the mapping's order, whose two
    joints lie in two different branches of the pelvis
    (:func:`_pelvis_branches`), so a pair of arms, both below the spine,
    is never read as the legs, and the legs are found whichever of their
    joints the mapping names. Each leg's far end is the node of its
    branch, end sites included, farthest from the pelvis, and up is the
    main axis from the middle of the two far ends to the pelvis. Read from
    the pelvis, not from the root: a root on the floor under the hips sits
    level with the feet.

    Returns
    -------
    str or None
        The signed axis string, or ``None`` without a pair split across
        the pelvis, or when the far ends coincide with the pelvis.
    """
    found = _pelvis_branches(skeleton)
    if found is None:
        return None
    pelvis, branch = found
    for left, right in skeleton.lr_pairs:
        left_branch, right_branch = int(branch[left]), int(branch[right])
        if left_branch >= 0 and right_branch >= 0 and left_branch != right_branch:
            break
    else:
        return None

    positions = skeleton.rest_positions
    hip = positions[pelvis]
    far_ends = []
    for leg in (left_branch, right_branch):
        below = np.flatnonzero(branch == leg)
        distances = np.linalg.norm(positions[below] - hip, axis=1)
        far_ends.append(positions[below[np.argmax(distances)]])
    return get_main_direction(hip - np.mean(far_ends, axis=0))


def _infer_rest_up(skeleton: _RestSkeleton) -> _RestUp | None:
    """The skeleton's own up, read from the rest pose alone, with its source.

    The main axis from the root to the node :func:`_up_marker_index`
    finds; without one, the axis opposite the legs, the L/R pair split
    across the pelvis (:func:`_up_from_leg_pair`); without such a pair, the
    axis along which the rest pose spreads most, signed by the side its
    nodes lie on. The largest spread is the weakest reading: a T-pose
    wider than tall spreads most along its arms, and hips mid-body put
    as much of it below the root as above.

    Parameters
    ----------
    skeleton : _RestSkeleton
        The skeleton in its rest pose, with the pairs of the L/R mapping
        in effect.

    Returns
    -------
    _RestUp or None
        The signed axis string (e.g. ``'+z'``) and its source, or ``None``
        when the rest pose carries no direction (a single node, or every
        node at one point).
    """
    local = skeleton.rest_positions - skeleton.rest_positions[0]
    marker = _up_marker_index(skeleton)
    if marker is not None:
        axis = get_main_direction(local[marker])
        # The marker is chosen for having a direction from the root.
        assert axis is not None
        return _RestUp(axis, "named_joint")

    leg_pair_up = _up_from_leg_pair(skeleton)
    if leg_pair_up is not None:
        return _RestUp(leg_pair_up, "leg_pair")

    spread = np.ptp(local, axis=0)
    if float(np.max(spread)) < _MIN_REST_SPREAD:
        return None  # degenerate rest pose: all joints coincide
    up_idx = int(np.argmax(spread))
    # The sign of the mean: the side of the root most of the body lies on.
    sign = "+" if np.mean(local[:, up_idx]) >= 0 else "-"
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

    Mutual-match requirement: both halves must exist as joint names.
    Singletons are not paired.

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


def _infer_rest_up_and_left(skeleton: _RestSkeleton) -> tuple[_RestUp | None, str | None]:
    """Rest up and rest left, both read with the skeleton's own L/R pairs.

    Without a named up joint, rest up is read from the pairs (the leg
    pair), so a rest left measured with one mapping's pairs against the
    rest up of another's mixes two skeletons' frames. Every reader of
    rest left goes through here: the orientation report with the pairs of
    the mapping in effect, :func:`~pybvh.transforms.mirror` with those
    of the mapping it mirrors with.

    Returns
    -------
    tuple of (_RestUp or None, str or None)
        :func:`_infer_rest_up`'s reading and :func:`_infer_rest_left`'s
        against its axis.
    """
    rest_up = _infer_rest_up(skeleton)
    rest_up_axis = None if rest_up is None else rest_up.axis
    return rest_up, _infer_rest_left(skeleton.rest_positions, skeleton.lr_pairs, rest_up_axis)


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

    ``source`` is ``"first_frame"`` (the up marker above the root at
    frame 0), ``"rest_up"`` (the first frame could not decide, so the
    skeleton's own up stands in) or ``"default"`` (neither said
    anything: the house up).
    """

    axis: str
    source: _WorldUpGuessSource


# Frame 0 decides world up when the largest component of its
# marker-above-root direction is at least this many times the second; below
# it the pose is crouched, lying or leaning, and the frame is ambiguous.
_FIRST_FRAME_DOMINANCE = 2.0


def _dominant_axis(direction: npt.NDArray[np.float64]) -> str | None:
    """*direction*'s main axis, when its largest component is at least
    ``_FIRST_FRAME_DOMINANCE`` times the second; ``None`` otherwise, and
    for a direction too short to read (:func:`get_main_direction`'s
    threshold)."""
    second, largest = np.sort(np.abs(direction))[-2:]
    if largest < _FIRST_FRAME_DOMINANCE * second:
        return None
    return get_main_direction(direction)


def _infer_world_up(
    skeleton: _RestSkeleton,
    frame0_positions: npt.NDArray[np.float64] | None,
) -> _WorldUpGuess:
    """Guess world up from a clip's first frame, with its source.

    The direction from the root to the up marker (:func:`_up_marker_index`,
    the node rest up reads) at frame 0, snapped to its main axis, when
    its largest component is at least twice the second. Otherwise (no
    frame, no marker, an ambiguous frame-0 pose) the skeleton's own up
    (:func:`_infer_rest_up`) stands in, and without one the house up
    ``'+y'``. Returns the guess silently: the reader of world up warns
    (:func:`_warn_world_up_defaulted`, :func:`_warn_world_up_disagreement`,
    :func:`_warn_rest_up_contradicted_by_root`).

    Parameters
    ----------
    skeleton : _RestSkeleton
        The skeleton the clip animates, in its rest pose.
    frame0_positions : ndarray of shape (N, 3) or None
        The node positions at frame 0, ``None`` for a clip without frames.

    Returns
    -------
    _WorldUpGuess
    """
    marker = _up_marker_index(skeleton)
    if frame0_positions is not None and marker is not None:
        frame_up = _dominant_axis(frame0_positions[marker] - frame0_positions[0])
        if frame_up is not None:
            return _WorldUpGuess(frame_up, "first_frame")

    rest_up = _infer_rest_up(skeleton)
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
        f"animation frame the direction from the root to the head joint "
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


def _root_contradicts_rest_up(
    guess: _WorldUpGuess, root_rotation: npt.NDArray[np.float64] | None
) -> str | None:
    """The axis frame 0's root rotation carries rest up onto, when a guess
    that fell back on rest up names another axis.

    The root rotation is what carries a fact of the skeleton's own frame
    into the world (ADR 0004); the rest-up fallback skips it. Rest up
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
    root_rotation : ndarray of shape (3, 3) or None
        The root's rotation matrix at frame 0, ``None`` for a clip
        without frames.

    Returns
    -------
    str or None
        The carried axis, or ``None`` when the guess did not fall back on
        rest up, the carried direction does not decide, or it names rest
        up's own axis.
    """
    if guess.source != "rest_up" or root_rotation is None:
        return None
    carried = _dominant_axis(root_rotation @ _axis_to_vector(guess.axis))
    if carried is None or carried[1] == guess.axis[1]:
        return None
    return carried


def _warn_rest_up_contradicted_by_root(rest_up: str, carried: str) -> None:
    """Warn that world up fell back on rest up while the root rotation of
    frame 0 carries rest up onto another axis."""
    # The fallback is otherwise silent: the report's "rest_up" source is
    # its only trace, and on a rest pose authored in another convention
    # than the motion it is wrong.
    warnings.warn(
        f"World up fell back on the rest pose's up {rest_up!r}: at the "
        f"first animation frame the direction from the root to the head "
        f"joint (else the neck, chest or spine) does not decide it. But "
        f"the root's rotation at that frame carries the rest pose's up "
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
    answer, whatever the clip carries). Frozen: a clip's report is
    replaced, never edited, when what it was read from changes.

    The facts belong to two frames (ADR 0004). The *skeleton's own
    frame* is read from the rest pose alone, the skeleton with every
    rotation zero, and never from world up: ``rest_up``, ``rest_left``,
    ``rest_forward``. The *world frame* is read from the motion:
    ``world_up``. The root rotation of a frame is what carries a
    skeleton fact into the world. Per-frame facts (the facing at a
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
        ``"named_joint"``: the main axis from the root to the first
        joint whose name holds ``head`` as a word, else ``neck``,
        ``chest`` or ``spine`` (``Head``, ``mixamorig:Head``, ``Bip01
        Head``; not ``Forehead``).
        ``"leg_pair"``: no such joint, so the axis opposite the legs:
        from the far ends of the two branches of the pelvis (the first
        node down from the root with more than one child) that an L/R
        pair is split across, to the pelvis. ``"largest_spread"``: no
        such pair either,
        so the axis along which the rest pose spreads most, signed by
        the side most of the body lies on, a weak reading (wrong in axis
        on a T-pose wider than tall, in sign when the hips sit
        mid-body). ``None`` with ``rest_up``.
    rest_left : str or None
        The skeleton's own left: the left-minus-right rest positions of
        the L/R pairs, averaged, with the ``rest_up`` component removed,
        snapped to an axis. ``None`` without pairs, without ``rest_up``,
        or when the pairs lie along ``rest_up``.
    rest_forward : str
        The direction the rest pose faces in the skeleton's own frame.
    rest_forward_source : {"lr_geometry", "default"}
        ``"lr_geometry"``: ``rest_left × rest_up``, measured.
        ``"default"``: no ``rest_left``, so the house default for
        ``rest_up`` (``'+y'`` gives ``'+z'``, ``'+z'`` gives ``'+x'``,
        ``'+x'`` gives ``'+y'``; ``'+z'`` without a ``rest_up``),
        arbitrary but stable.
    world_up : str
        The world's vertical, a signed axis string.
    world_up_source : {"set", "first_frame", "rest_up", "default"}
        ``"set"``: assigned by the user, passed as ``world_up=`` at load,
        or the target of ``reorient_world_up``. ``"first_frame"``:
        guessed from the direction of the ``rest_up`` node above the
        root at frame 0. ``"rest_up"``: the first frame could not decide
        (no frames, no such node, or a crouched, lying or leaning first
        pose), so the skeleton's own up stands in, a guess that assumes
        the rest pose is authored upright in the world.
        ``"default"``: neither said anything, so the house up ``'+y'``.
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
    """

    rest_up: str | None
    rest_up_source: _RestUpSource | None
    rest_left: str | None
    rest_forward: str
    rest_forward_source: _RestForwardSource
    world_up: str
    world_up_source: _WorldUpSource
    world_up_inherited: bool
    lr_mapping_source: _LrMappingSource | None


def _build_orientation_report(
    skeleton: _RestSkeleton,
    world_up: str | _WorldUpGuess,
    world_up_inherited: bool,
    lr_mapping_source: _LrMappingSource | None,
) -> OrientationReport:
    """Read the skeleton's own frame from the rest pose, beside world up.

    Parameters
    ----------
    skeleton : _RestSkeleton
        The skeleton in its rest pose, with the pairs of the L/R mapping
        in effect.
    world_up : str or _WorldUpGuess
        World up as the caller settled it: a signed axis string the user
        set (reported as ``"set"``), or the guess it stands on
        (:func:`_infer_world_up`), whose source is reported.
    world_up_inherited : bool
        Whether that guess was made on the clip this one was derived
        from; False for an axis the user set.
    lr_mapping_source : {"set", "names"} or None
        The source of the L/R mapping the skeleton's pairs come from, the
        caller's setting.

    Returns
    -------
    OrientationReport
    """
    world_up_source: _WorldUpSource
    if isinstance(world_up, _WorldUpGuess):
        world_up_axis, world_up_source = world_up
    else:
        world_up_axis, world_up_source = world_up, "set"
    rest_up, rest_left = _infer_rest_up_and_left(skeleton)
    rest_up_axis = None if rest_up is None else rest_up.axis
    return OrientationReport(
        rest_up=rest_up_axis,
        rest_up_source=None if rest_up is None else rest_up.source,
        rest_left=rest_left,
        rest_forward=_infer_rest_forward(rest_left, rest_up_axis),
        rest_forward_source="lr_geometry" if rest_left is not None else "default",
        world_up=world_up_axis,
        world_up_source=world_up_source,
        world_up_inherited=world_up_inherited,
        lr_mapping_source=lr_mapping_source,
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


def _root_heading_units(
    root_rotmats: npt.NDArray[np.float64],
    rest_forward: str,
    world_up: str,
) -> npt.NDArray[np.float64]:
    """The root's heading per frame: rest forward carried into the world.

    The root rotation of each frame applied to *rest_forward* (a
    direction in the skeleton's own frame), projected on the ground of
    *world_up* and normalised. Where that projection is too short to read
    (the root's forward stands along world up) the row is ``NaN``, the
    :mod:`pybvh.geometry` policy for a degenerate input: a number there
    would be an arbitrary angle.

    Parameters
    ----------
    root_rotmats : ndarray of shape (F, 3, 3)
        The root's rotation matrix per frame.
    rest_forward : str
        Signed axis string, the direction the rest pose faces.
    world_up : str
        Signed axis string for the world vertical axis.

    Returns
    -------
    ndarray of shape (F, 3)
        Unit ground vectors, ``NaN`` on the unreadable rows.
    """
    forward = np.einsum("fij,j->fi", root_rotmats, _axis_to_vector(rest_forward))
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
    root_rotmats: npt.NDArray[np.float64] | None = None,
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
    the root's heading at that frame turned a quarter turn about up
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
    root_rotmats : ndarray of shape (F, 3, 3), optional
        The root's rotation at each row's frame.
    rest_forward : str, optional
        The direction the rest pose faces, read with *root_rotmats* into
        the root's heading (:func:`_root_heading_units`). Pass it only
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
    if root_rotmats is not None and rest_forward is not None and defaulted.any():
        # Only the rows the frame's own leftward leaves unread.
        rows = np.flatnonzero(defaulted)
        heading = _root_heading_units(root_rotmats[rows], rest_forward, world_up)
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
        f"root's heading cannot stand in (it lies along world_up too, or "
        f"rest_forward is itself a default). Falling back to {fallback!r}, "
        f"the arbitrary-but-stable horizontal axis for that up. "
        f"`facing_frame().valid` is False on such frames.",
        UserWarning,
        stacklevel=user_stacklevel(),
    )
