"""The orientation a skeleton's geometry implies.

The skeleton's own frame read from the rest pose alone (rest up, rest
left, rest forward), the world up axis inferred from a clip's first frame,
the left/right joint pairs found by name and their resolution into
joint-space and node-space index pairs, and the per-frame leftward,
forward and facing basis, with the house default for a skeleton that
carries no left/right information.
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
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ._warnings import user_stacklevel
from .tools import _AXIS_CHAR_TO_IDX, _axis_to_vector, get_main_direction

if TYPE_CHECKING:
    from .bvh import Bvh

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
#      against rest up); _infer_rest_forward (rest left × rest up, or the
#      house default keyed on rest up). Used by mirror, the rest
#      reorientations and root_trajectory's heading, where the root
#      rotation carries rest forward into the world.
#   2. THE WORLD FRAME — world up, guessed from the first frame through the
#      same up marker (_infer_world_up), and the facing at a frame, read
#      from that frame's joint positions against world up.
#
# The L/R mapping is detected from joint names and resolved into index
# pairs (_detect_lr_mapping_by_names, _resolve_lr_pairs,
# _resolve_node_lr_pairs), also on plain data. Bvh gathers every input.
#
# The facing chain, per frame (→ = "used by"):
#
#     _facing_lr_pairs ──→ bvhplot SkeletonView.lr_pairs
#           │                       │
#           ▼                       ▼
#     _leftward_units_from_pairs ── the ONLY place the per-frame leftward
#           │   (array kernel)      geometry is computed; bvhplot
#           │                       follow-mode azimuths read it too
#           ▼
#     _world_leftward_units (Bvh-reading resolver over the kernel)
#           │        └──→ _world_leftward_unit_at_frame (1-frame view)
#           │                   │
#           │                   ▼
#           │             _compute_forward_at, ──→ Bvh.forward_at /
#           │             _compute_left_at         Bvh.left_at
#           ▼               (axis-string snap)
#     _facing_basis ──→ analysis.facing_frame
#        (+ _fallback_forward_vector, shared with the axis-string snappers)
#
# Public surface: `Bvh.world_up` (property + setter), `Bvh.rest_up`,
# `Bvh.rest_forward`, `Bvh.has_lr_geometry`, the parsed `Bvh.world_up_axis` /
# `Bvh.rest_forward_axis` / `Bvh.rest_up_axis` properties,
# `Bvh.forward_at(frame=0)` / `Bvh.left_at(frame=0)` (methods), and — via
# `analysis.facing_frame` — the continuous vector basis. Everything here is
# underscore-prefixed and intended for internal use by pybvh modules.
# ---------------------------------------------------------------------------

# Arbitrary-but-stable forward per up axis (rest up for rest forward, world up
# for the per-frame facing), used when a skeleton carries no usable L/R
# orientation information.
_FALLBACK_FORWARD = {"y": "+z", "z": "+x", "x": "+y"}


# The body parts whose direction from the root says which way is up, most
# telling first.
_UP_BODY_PARTS = ("head", "neck", "chest", "spine")


def _up_marker_index(
    node_names: Sequence[str],
    rest_positions: npt.NDArray[np.float64],
) -> int | None:
    """The node whose direction from the root says which way is up.

    The one body-part lookup for "up", shared by the rest up and the
    world-up guess so the two read the same node: the first node, in node
    order, named ``head``, else ``neck``, ``chest`` or ``spine`` (whole
    name, any case, so ``mixamorig:Head`` does not match). A node whose
    rest position coincides with the root's carries no direction and is
    skipped, the root itself included.

    Parameters
    ----------
    node_names : sequence of str
        Every node's name, in node order (end sites included).
    rest_positions : ndarray of shape (N, 3)
        The rest pose's node positions, in the same order.

    Returns
    -------
    int or None
        The node's index, or ``None`` when no node qualifies.
    """
    local = rest_positions - rest_positions[0]
    lowered = [name.lower() for name in node_names]
    for part in _UP_BODY_PARTS:
        # The first node of a repeated name, by position in the walk.
        for idx, name in enumerate(lowered):
            if name == part and get_main_direction(local[idx]) is not None:
                return idx
    return None


def _infer_rest_up(
    node_names: Sequence[str],
    rest_positions: npt.NDArray[np.float64],
) -> str | None:
    """The skeleton's own up, read from the rest pose alone.

    The main axis from the root to the node :func:`_up_marker_index`
    finds; without one, the axis along which the rest pose spreads most,
    signed by the side its nodes lie on.

    Parameters
    ----------
    node_names : sequence of str
        Every node's name, in node order (end sites included).
    rest_positions : ndarray of shape (N, 3)
        The rest pose's node positions, in the same order.

    Returns
    -------
    str or None
        Signed axis string (e.g. ``'+z'``), or ``None`` when the rest pose
        carries no direction (a single node, or every node at one point).
    """
    local = rest_positions - rest_positions[0]
    marker = _up_marker_index(node_names, rest_positions)
    if marker is not None:
        return get_main_direction(local[marker])

    spread = np.ptp(local, axis=0)
    if float(np.max(spread)) < 1e-6:
        return None  # degenerate rest pose: all joints coincide
    up_idx = int(np.argmax(spread))
    # The sign of the mean: the side of the root most of the body lies on.
    sign = "+" if np.mean(local[:, up_idx]) >= 0 else "-"
    return sign + "xyz"[up_idx]


# ---------------------------------------------------------------------------
# L/R name-detection heuristics (pure string logic on joint names)
# ---------------------------------------------------------------------------

_NUMBER_SUFFIX_RE = re.compile(r"\.\d+$")


def _strip_number_suffix(name: str) -> str:
    """Strip trailing ``.NNN`` suffix like Blender's ``.001`` duplicates."""
    return _NUMBER_SUFFIX_RE.sub("", name)


def _strip_namespace_prefix(name: str) -> tuple[str, str]:
    """Strip a ``foo:`` namespace prefix (Mixamo and similar).

    Returns ``(prefix, base)``. If no namespace, prefix is empty.
    """
    if ":" in name:
        prefix, _, base = name.rpartition(":")
        return prefix + ":", base
    return "", name


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

        # Decompose name for suffix rule: strip namespace and number suffix
        ns_prefix, after_ns = _strip_namespace_prefix(name)
        base = _strip_number_suffix(after_ns)
        number_suffix = after_ns[len(base) :]  # e.g. '.001' or ''

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
    # Normalize for suffix check: strip namespace + number suffix
    _, a_after = _strip_namespace_prefix(a)
    _, b_after = _strip_namespace_prefix(b)
    a_base = _strip_number_suffix(a_after)
    b_base = _strip_number_suffix(b_after)
    for left_suf, _right_suf in _SUFFIX_LR_TABLE:
        if a_base.endswith(left_suf):
            return (a, b)
        if b_base.endswith(left_suf):
            return (b, a)
    # Fallback: "L" prefix → left
    a_prefix_check = _strip_number_suffix(a_after)
    b_prefix_check = _strip_number_suffix(b_after)
    if a_prefix_check.startswith("L") and (len(a_prefix_check) < 2 or a_prefix_check[1].isupper()):
        return (a, b)
    if b_prefix_check.startswith("L") and (len(b_prefix_check) < 2 or b_prefix_check[1].isupper()):
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
        end_site_pairs.extend(zip(left_ends, right_ends))

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
        The skeleton's own up, as :func:`_infer_rest_up` gives it.

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
        f"Set `bvh.lr_mapping = {{...}}` to fix it, or check "
        f"`bvh.has_lr_geometry` to detect it.",
        UserWarning,
        stacklevel=user_stacklevel(),
    )


def _infer_world_up(bvh: Bvh, warn: bool = True) -> str:
    """Infer the world vertical axis from the first animation frame.

    Uses frame 0's head-above-hips direction as the primary source (this
    reflects the actual world-space orientation at playback time, not
    just the rest-pose topology). Falls back to rest-pose topology if:
      - there are no animation frames
      - the first frame's head-hips direction is ambiguous (no dominant
        component clearly larger than the others)
      - the relevant joints are not present

    Issues a UserWarning when the first-frame inference succeeds but
    disagrees with the rest-pose topology (i.e. a different dominant
    axis) — this is a strong signal that the BVH file authored its rest
    pose in one convention and animates in another.

    Parameters
    ----------
    bvh : Bvh
        The skeleton with populated animation data.
    warn : bool, optional
        Whether to warn when the first frame and the rest pose disagree
        on the up axis (default True). Only the axis letter is compared,
        so a sign flip alone (rest pose ``'+y'``, first frame ``'-y'``)
        never warns. It gates only that warning: the warning that the
        axis fell back to the invented ``'+y'`` is always issued, because
        that result is a guess the returned string cannot flag.

    Returns
    -------
    str
        Signed axis string (e.g. '+y').
    """
    # Compute rest-pose topology once; used both as fallback and as the
    # reference to compare against when emitting the disagreement warning.
    node_names = [node.name for node in bvh.nodes]
    rest = bvh._rest_pose()
    rest_up = _infer_rest_up(node_names, rest)

    def fallback() -> str:
        """The rest-pose axis, or the house default when there is none."""
        if rest_up is not None:
            return rest_up
        # Neither the animation nor the rest pose said anything, so '+y'
        # is invented, not measured, and the returned string cannot say
        # so — same silent substitution _fallback_forward_vector warns
        # about, same treatment.
        warnings.warn(
            "Could not infer a world up axis: this skeleton has no "
            "recognizable head/neck/chest/spine joint to read the first "
            "frame from, and its rest pose is degenerate (single node, or "
            "all offsets coincident). Falling back to '+y', the house "
            "default — anything derived from up (floor_height, "
            "foot_contacts, facing, ground-plane projections) rests on a "
            "guess here. Set `bvh.world_up = '<axis>'` to fix it.",
            UserWarning,
            stacklevel=user_stacklevel(),
        )
        return "+y"

    # Need animation frames to do first-frame inference
    if bvh.frame_count == 0:
        return fallback()

    # The node rest up reads ("head-ish"), against the root ("hips").
    head_idx = _up_marker_index(node_names, rest)
    hips_idx = 0

    if head_idx is None:
        return fallback()

    frame0 = bvh.node_positions(frame=0)
    head_hips = frame0[head_idx] - frame0[hips_idx]

    # Check for a clear dominant axis: the largest component must be
    # strictly larger than the second-largest (ratio > 2). Otherwise
    # the pose is ambiguous (character crouched / lying / leaning).
    abs_components = np.abs(head_hips)
    sorted_abs = np.sort(abs_components)
    if sorted_abs[-1] < 2.0 * sorted_abs[-2] or sorted_abs[-1] < 1e-6:
        return fallback()

    # Clear winner from animation data
    frame_up = get_main_direction(head_hips)
    if frame_up is None:
        return fallback()

    # Warn if rest-pose topology disagrees (different dominant axis)
    if warn and rest_up is not None and rest_up[1] != frame_up[1]:
        warnings.warn(
            f"Rest pose suggests world up is {rest_up!r} but the first "
            f"animation frame's head-hips direction is closer to "
            f"{frame_up!r}. Using {frame_up!r} from the animation data. "
            f"If this is wrong for your file, set it explicitly via "
            f"`bvh.world_up = '<axis>'`.",
            UserWarning,
            stacklevel=user_stacklevel(),
        )

    return frame_up


def _world_leftward_units(
    bvh: Bvh,
    coords: npt.NDArray[np.float64],
    world_up: str,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Per-frame world-space leftward **unit vectors** for a whole clip.

    The single implementation of the facing construction's first stage:
    average ``(left_pos - right_pos)`` across matching L/R joint pairs
    in world space, project onto the plane perpendicular to
    ``world_up``, normalize. Continuous (not snapped to a signed axis);
    matches the ``up × forward`` right-hand-rule convention used by
    :meth:`Bvh.left_at`. Every consumer of the leftward geometry —
    :func:`_facing_basis`, the single-frame axis-string snappers, the
    bvhplot follow camera — routes through here.

    Parameters
    ----------
    bvh : Bvh
        The skeleton (used for L/R pairs).
    coords : np.ndarray
        Spatial coordinates of shape (F, N, 3).
    world_up : str
        Signed axis string for the world vertical axis.

    Returns
    -------
    leftward : ndarray of shape (F, 3)
        Leftward unit vector per frame. Rows where ``valid`` is False
        carry no usable direction and must not be consumed.
    valid : ndarray of shape (F,), bool
        False for every frame when no L/R pairs exist; per-frame False
        where the averaged leftward is degenerate (parallel to world
        up, or zero). Callers should handle invalid frames by falling
        back to the topological rest-pose direction.
    """
    return _leftward_units_from_pairs(coords, _facing_lr_pairs(bvh), _axis_to_vector(world_up))


def _facing_lr_pairs(bvh: Bvh) -> npt.NDArray[np.intp]:
    """The L/R pairs the facing construction averages, in node index space.

    Joint pairs only, as :attr:`Bvh.lr_mapping` names them, resolved into
    ``nodes`` index space so they index ``node_positions()`` output. End
    sites are not included: that is :attr:`Bvh.node_lr_pairs`' job, for
    mirroring positions, not for reading the facing. Returns an ``(P, 2)``
    int array, ``(0, 2)`` when the rig has no L/R mapping, so array code
    needs no ``None`` branch.

    The Bvh-reading half of :func:`_world_leftward_units`; the array
    half is :func:`_leftward_units_from_pairs`. bvhplot stores this
    array on its ``SkeletonView`` so the follow camera needs no Bvh.
    """
    pairs = _resolve_lr_pairs(bvh.lr_mapping, bvh.node_index)
    return np.asarray(pairs, dtype=np.intp).reshape(-1, 2)


def _leftward_units_from_pairs(
    coords: npt.NDArray[np.float64],
    lr_pairs: npt.NDArray[np.intp],
    up_vec: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Array kernel of :func:`_world_leftward_units`: no Bvh, arrays in.

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
    leftward, valid
        As :func:`_world_leftward_units`.
    """
    num_frames = coords.shape[0]
    lr_pairs = np.asarray(lr_pairs, dtype=np.intp).reshape(-1, 2)
    if lr_pairs.shape[0] == 0:
        return (np.zeros((num_frames, 3)), np.zeros(num_frames, dtype=bool))

    left_idx = lr_pairs[:, 0]
    right_idx = lr_pairs[:, 1]

    leftward = np.mean(coords[:, left_idx] - coords[:, right_idx], axis=1)
    # Project onto plane perpendicular to world_up
    leftward = leftward - (leftward @ up_vec)[:, np.newaxis] * up_vec
    norms = np.linalg.norm(leftward, axis=1)
    valid = norms >= 1e-6
    leftward[valid] /= norms[valid, np.newaxis]
    return leftward, valid


def _world_leftward_unit_at_frame(
    bvh: Bvh,
    frame_coords: npt.NDArray[np.float64],
    world_up: str,
) -> npt.NDArray[np.float64] | None:
    """Single-frame view of :func:`_world_leftward_units`.

    Takes (N, 3) coordinates for one frame; returns the leftward unit
    vector, or ``None`` for the degenerate cases where the vectorized
    form flags the frame invalid (no L/R pairs, or leftward parallel to
    world up / zero).
    """
    leftward, valid = _world_leftward_units(bvh, frame_coords[np.newaxis], world_up)
    if not valid[0]:
        return None
    return leftward[0]


def _rest_forward_from_topology(
    bvh: Bvh,
    world_up: str,
) -> npt.NDArray[np.float64] | None:
    """Forward measured from the topological rest leftward, or ``None``.

    The middle step of the facing fallback chain: the rest left
    (the L/R axis measured against the *rest-pose* up) crossed with
    ``world_up``. Returns ``None`` when there is no usable rest leftward,
    or when it is (anti)parallel to ``world_up`` — possible when
    rest-pose topology and ``world_up`` disagree — i.e. exactly the
    cases where the caller must fall through to the arbitrary
    ``_FALLBACK_FORWARD`` default, which :func:`_fallback_forward_vector`
    does.
    """
    rest_left = bvh._rest_left_of(bvh._lr_mapping)
    if rest_left is None:
        return None
    forward = np.cross(_axis_to_vector(rest_left), _axis_to_vector(world_up))
    # Cross of two distinct signed axes is exactly a unit axis vector;
    # it is zero only in the (anti)parallel case described above.
    if float(np.linalg.norm(forward)) < 1e-6:
        return None
    return forward  # type: ignore[return-value]  # NumPy 2.2's stub types a float cross product as floating, not float64


def _fallback_forward_vector(
    bvh: Bvh,
    world_up: str,
) -> npt.NDArray[np.float64]:
    """Unit forward vector for frames carrying no usable L/R direction.

    The vector form of the fallback chain documented on
    :meth:`Bvh.forward_at`: the topological rest-pose leftward crossed
    with ``world_up``; when the skeleton has no usable rest-pose L/R
    geometry either (or it is parallel to ``world_up``), the
    arbitrary-but-stable ``_FALLBACK_FORWARD`` axis for this up axis.
    Always axis-aligned and perpendicular to ``world_up``, so snapping
    it reproduces the string fallback exactly.
    """
    forward = _rest_forward_from_topology(bvh, world_up)
    if forward is not None:
        return forward
    fallback = _FALLBACK_FORWARD[world_up[1]]
    # The returned axis is a house default, not a measurement, and the
    # return type cannot say so — mirroring `_infer_world_up`, which warns
    # on the same class of silent substitution. Python's default filter
    # shows this once per line of the user's code, so a loop over frames
    # stays quiet.
    warnings.warn(
        f"No usable left/right geometry on this skeleton, so its facing "
        f"direction cannot be measured; falling back to {fallback!r}, the "
        f"arbitrary-but-stable horizontal axis for world_up={world_up!r}. "
        f"The facing (forward_at, left_at, facing_frame) is a default "
        f"here, not a property of this file. Set `bvh.lr_mapping = {{...}}` to fix it, "
        f"or check `bvh.has_lr_geometry` to detect it.",
        UserWarning,
        stacklevel=user_stacklevel(),
    )
    return _axis_to_vector(fallback)


def _facing_basis(
    bvh: Bvh,
    coords: npt.NDArray[np.float64],
    world_up: str,
) -> tuple[
    npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.bool_]
]:
    """Vectorized per-frame facing basis — the geometry under ``facing_frame``.

    Builds the right-handed orthonormal triple from the leftward
    geometry: ``forward = leftward × up``, ``left = up × forward``,
    ``up`` = the exact ``world_up`` unit vector on every frame. Frames
    flagged invalid by :func:`_world_leftward_units` receive the
    constant :func:`_fallback_forward_vector` basis instead — the same
    fallback chain the axis-string snappers use — and are reported as
    such in ``valid``, which is passed through unchanged so the report
    tracks the fallback by construction.

    Parameters
    ----------
    bvh : Bvh
        The skeleton (used for L/R pairs and the rest-pose fallback).
    coords : np.ndarray
        Spatial coordinates of shape (F, N, 3).
    world_up : str
        Signed axis string for the world vertical axis.

    Returns
    -------
    forward, left, up : ndarray of shape (F, 3) each
        Unit vectors forming a right-handed basis per frame
        (``forward × left = up``).
    valid : ndarray of shape (F,), bool
        False on the rows that carry the constant fallback basis.
    """
    up_vec = _axis_to_vector(world_up)
    leftward, valid = _world_leftward_units(bvh, coords, world_up)

    # leftward is unit and perpendicular to up on valid rows, so both
    # crosses below give unit vectors and `left` re-orthogonalizes to
    # leftward itself (up × (leftward × up) = leftward).
    forward = np.cross(leftward, up_vec)
    left = np.cross(up_vec, forward)
    if not valid.all():
        fallback_forward = _fallback_forward_vector(bvh, world_up)
        forward[~valid] = fallback_forward
        left[~valid] = np.cross(up_vec, fallback_forward)

    up = np.tile(up_vec, (coords.shape[0], 1))
    return forward, left, up, valid  # type: ignore[return-value]  # NumPy 2.2's stub types a float cross product as floating, not float64


def _compute_forward_at(
    bvh: Bvh,
    frame_coords: npt.NDArray[np.float64],
    world_up: str,
) -> str:
    """Compute the character's world-space forward direction at a given frame.

    Uses the actual joint positions at the frame (not rest-pose offsets),
    so the result tracks root rotation, hip twist, and shoulder rotation
    as the character moves.

    Algorithm:
      1. Get the continuous world-space leftward vector via
         :func:`_world_leftward_unit_at_frame`.
      2. ``forward = cross(leftward, world_up_vec)`` — follows the
         ``up × forward = leftward`` right-hand-rule convention.
      3. Snap to nearest signed axis via :func:`get_main_direction`.
      4. Fallback: if leftward is degenerate (parallel to world_up) or
         no L/R pairs found, use :func:`_fallback_forward_vector`
         (rest-pose leftward crossed with world_up, then the arbitrary
         per-up-axis default).

    Parameters
    ----------
    bvh : Bvh
        The skeleton.
    frame_coords : np.ndarray
        Spatial coordinates of shape (N, 3) for a single frame.
    world_up : str
        Signed axis string for the world vertical axis.

    Returns
    -------
    str
        Signed axis string (e.g. '-z') giving the character's forward
        direction in world space at the given frame.
    """
    leftward_vec = _world_leftward_unit_at_frame(bvh, frame_coords, world_up)

    if leftward_vec is None:
        forward_vec = _fallback_forward_vector(bvh, world_up)
    else:
        forward_vec = np.cross(leftward_vec, _axis_to_vector(world_up))  # type: ignore[assignment]  # NumPy 2.2's stub types a float cross product as floating, not float64

    # Both branches yield a unit vector perpendicular to world_up, so
    # the snap never degenerates and never lands on the up axis.
    forward_ax = get_main_direction(forward_vec)
    assert forward_ax is not None
    return forward_ax


def _compute_left_at(
    bvh: Bvh,
    frame_coords: npt.NDArray[np.float64],
    world_up: str,
) -> str:
    """Compute the character's world-space leftward direction at a given frame.

    Companion of :func:`_compute_forward_at` — implements
    :meth:`Bvh.left_at`.  Snaps the continuous world-space leftward
    vector (averaged left-minus-right over L/R joint pairs) to the
    nearest signed axis, falling back to the rest-pose leftward when the
    frame's leftward is degenerate, and finally to a left axis derived
    from the same arbitrary horizontal forward `_compute_forward_at`
    uses — so the (up, forward, left) triple stays consistent even
    without any L/R information.

    Parameters
    ----------
    bvh : Bvh
        The skeleton.
    frame_coords : np.ndarray
        Spatial coordinates of shape (N, 3) for a single frame.
    world_up : str
        Signed axis string for the world vertical axis.

    Returns
    -------
    str
        Signed axis string (e.g. '-x') pointing toward the character's
        left in world space at the given frame.
    """
    left_vec = _world_leftward_unit_at_frame(bvh, frame_coords, world_up)

    if left_vec is not None:
        # Unit and perpendicular to world_up by construction, so the
        # snap never degenerates and never lands on the up axis.
        left_ax = get_main_direction(left_vec)
        assert left_ax is not None
        return left_ax

    rest_left = bvh._rest_left_of(bvh._lr_mapping)
    if rest_left is not None:
        return rest_left

    # No usable L/R information — derive left from the fallback forward
    # via the right-hand rule (leftward = up × forward). Routed through
    # _fallback_forward_vector so this path emits the same warning: a
    # defaulted "left" is exactly as unmeasured as a defaulted forward.
    fwd_vec = _fallback_forward_vector(bvh, world_up)
    up_vec = _axis_to_vector(world_up)
    left_ax = get_main_direction(np.cross(up_vec, fwd_vec))  # type: ignore[arg-type]  # NumPy 2.2's stub types a float cross product as floating, not float64
    assert left_ax is not None  # axis-aligned cross never degenerate
    return left_ax
