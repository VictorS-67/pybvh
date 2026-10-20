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
from typing import TYPE_CHECKING, Literal, NamedTuple

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
#     _facing_lr_pairs ──→ bvhplot SkeletonView.lr_pairs
#           │                       │
#           ▼                       ▼
#     _leftward_units_from_pairs ── the ONLY place the per-frame leftward
#           │   (array kernel)      geometry is computed; bvhplot
#           │                       follow-mode azimuths read it too
#           ▼
#     _facing_basis: the frame's leftward, else the root's heading at that
#           │        frame (_root_heading_units), else the house default
#           │        (_fallback_leftward); forward = left × up
#           ├──→ Bvh._facing_basis ──→ analysis.facing_frame
#           └──→ _facing_labels (the snap) ──→ Bvh.forward_at / Bvh.left_at
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


class _WorldUpGuess(NamedTuple):
    """A guessed world up and what it was read from.

    ``source`` is ``"first_frame"`` (the up marker above the root at
    frame 0), ``"rest_up"`` (the first frame could not decide, so the
    skeleton's own up stands in) or ``"default"`` (neither said
    anything: the house up).
    """

    axis: str
    source: Literal["first_frame", "rest_up", "default"]


# How much the largest component of frame 0's marker-above-root direction
# must exceed the second for the frame to decide world up; below it the
# pose is crouched, lying or leaning, and the frame is ambiguous.
_FIRST_FRAME_DOMINANCE = 2.0


def _infer_world_up(
    node_names: Sequence[str],
    rest_positions: npt.NDArray[np.float64],
    frame0_positions: npt.NDArray[np.float64] | None,
) -> _WorldUpGuess:
    """Guess world up from a clip's first frame, with its source.

    The direction from the root to the up marker (:func:`_up_marker_index`,
    the node rest up reads) at frame 0, snapped to its main axis, when
    its largest component is more than twice the second. Otherwise (no
    frame, no marker, an ambiguous frame-0 pose) the skeleton's own up
    (:func:`_infer_rest_up`) stands in, and without one the house up
    ``'+y'``. Returns the guess silently: the reader of world up warns
    (:func:`_warn_world_up_defaulted`, :func:`_warn_world_up_disagreement`).

    Parameters
    ----------
    node_names : sequence of str
        Every node's name, in node order (end sites included).
    rest_positions : ndarray of shape (N, 3)
        The rest pose's node positions.
    frame0_positions : ndarray of shape (N, 3) or None
        The node positions at frame 0, ``None`` for a clip without frames.

    Returns
    -------
    _WorldUpGuess
    """
    marker = _up_marker_index(node_names, rest_positions)
    if frame0_positions is not None and marker is not None:
        marker_above_root = frame0_positions[marker] - frame0_positions[0]
        second, largest = np.sort(np.abs(marker_above_root))[-2:]
        if largest >= _FIRST_FRAME_DOMINANCE * second and largest >= 1e-6:
            frame_up = get_main_direction(marker_above_root)
            if frame_up is not None:
                return _WorldUpGuess(frame_up, "first_frame")

    rest_up = _infer_rest_up(node_names, rest_positions)
    if rest_up is not None:
        return _WorldUpGuess(rest_up, "rest_up")
    return _WorldUpGuess(_HOUSE_UP, "default")


def _warn_world_up_defaulted() -> None:
    """Warn that world up is the house default, not a measurement."""
    # Neither the animation nor the rest pose said anything, so '+y' is
    # invented, not measured, and the returned string cannot say so; the
    # same silent substitution _warn_facing_defaulted warns about.
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
    # A strong signal that the file authored its rest pose in one
    # convention and animates in another.
    warnings.warn(
        f"Rest pose suggests world up is {rest_up!r} but the first "
        f"animation frame's head-hips direction is closer to "
        f"{frame_up!r}. Using {frame_up!r} from the animation data. "
        f"If this is wrong for your file, set it explicitly via "
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


def _facing_lr_pairs(bvh: Bvh) -> npt.NDArray[np.intp]:
    """The L/R pairs the facing construction averages, in node index space.

    Joint pairs only, as :attr:`Bvh.lr_mapping` names them, resolved into
    ``nodes`` index space so they index ``node_positions()`` output. End
    sites are not included: that is :attr:`Bvh.node_lr_pairs`' job, for
    mirroring positions, not for reading the facing. Returns an ``(P, 2)``
    int array, ``(0, 2)`` when the rig has no L/R mapping, so array code
    needs no ``None`` branch.

    What :func:`_leftward_units_from_pairs` and :func:`_facing_basis`
    read. bvhplot stores this array on its ``SkeletonView`` so the
    follow camera needs no Bvh.
    """
    pairs = _resolve_lr_pairs(bvh.lr_mapping, bvh.node_index)
    return np.asarray(pairs, dtype=np.intp).reshape(-1, 2)


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
    is unreadable (the left/right axis vertical, the body on its side),
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


def _warn_facing_defaulted(world_up: str, frames: Sequence[int] | None) -> None:
    """Warn that some facing rows carry the house default, not a measurement.

    *frames* names the clip frames of those rows; an empty sequence
    means the skeleton has no L/R pairs at all, so every row defaults,
    and ``None`` that the rows are a pose with no clip frame.
    """
    fallback = _FALLBACK_FORWARD[world_up[1]]
    # The returned axis is a house default, not a measurement, and the
    # return type cannot say so, the same silent substitution
    # _infer_world_up warns about. Python's default filter shows this once
    # per line of the user's code, so a loop over frames stays quiet.
    if frames is not None and len(frames) == 0:
        message = (
            f"No usable left/right geometry on this skeleton, so its facing "
            f"direction cannot be measured; falling back to {fallback!r}, the "
            f"arbitrary-but-stable horizontal axis for world_up={world_up!r}. "
            f"The facing (forward_at, left_at, facing_frame) is a default "
            f"here, not a property of this file. Set `bvh.lr_mapping = {{...}}` "
            f"to fix it, or check `bvh.has_lr_geometry` to detect it."
        )
    else:
        if frames is None:
            where = "On the pose given"
        else:
            shown = ", ".join(str(f) for f in frames[:5])
            more = f" and {len(frames) - 5} more" if len(frames) > 5 else ""
            where = f"At frame{'s' if len(frames) > 1 else ''} {shown}{more}"
        message = (
            f"{where}, the facing direction cannot be measured: the left/right "
            f"joints line up with world_up={world_up!r} (the body on its side), "
            f"and the root's heading gives no ground direction either. Falling "
            f"back to {fallback!r}, the arbitrary-but-stable horizontal axis "
            f"for that up. `facing_frame().valid` is False on such frames."
        )
    warnings.warn(message, UserWarning, stacklevel=user_stacklevel())
