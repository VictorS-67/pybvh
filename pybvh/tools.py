from __future__ import annotations

from pathlib import Path
from typing import Sequence, TYPE_CHECKING, Union

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from .bvh import Bvh



def are_permutations(str1: str, str2: str) -> bool:
    """Test if two strings are permutations of each other.

    Parameters
    ----------
    str1 : str
        First string.
    str2 : str
        Second string.

    Returns
    -------
    result : bool
        True if the strings are permutations of each other.
    """
    if len(str1) != len(str2):
        return False

    char_freq: dict[str, int] = {}

    for char in str1:
        char_freq[char] = char_freq.get(char, 0) + 1

    for char in str2:
        if char not in char_freq or char_freq[char] == 0:
            return False
        char_freq[char] -= 1

    return True

#--------------------------------------------------------------------------------------------


def test_file(filepath: str | Path) -> Path:
    """Validate that a filepath exists and points to a .bvh file.

    Parameters
    ----------
    filepath : str or Path
        Path to the file to validate.

    Returns
    -------
    filepath : Path
        The validated filepath as a Path object.

    Raises
    ------
    ImportError
        If the file is not a .bvh file or does not exist.
    """
    filepath = Path(filepath)
    if filepath.suffix != '.bvh':
        raise ValueError(f'{filepath} is not a bvh file')
    elif not filepath.exists():
        raise FileNotFoundError(f'could not find the file {filepath}')
    return filepath

#--------------------------------------------------------------------------------------------

# rotations matrices
# since the goal is efficiency with those, we want to minize the overhead
# therefore we assume that the angle is already in radians

def rotX(angle: float) -> npt.NDArray[np.float64]:
    """Compute a 3x3 rotation matrix around the X axis.

    Parameters
    ----------
    angle : float
        Rotation angle in radians.

    Returns
    -------
    R : np.ndarray, shape (3, 3)
        Rotation matrix around X.
    """
    return np.array([[1, 0, 0],
                     [0, np.cos(angle), -np.sin(angle)],
                     [0, np.sin(angle), np.cos(angle)]])

def rotY(angle: float) -> npt.NDArray[np.float64]:
    """Compute a 3x3 rotation matrix around the Y axis.

    Parameters
    ----------
    angle : float
        Rotation angle in radians.

    Returns
    -------
    R : np.ndarray, shape (3, 3)
        Rotation matrix around Y.
    """
    return np.array([[np.cos(angle), 0, np.sin(angle)],
                     [0, 1, 0],
                     [-np.sin(angle), 0, np.cos(angle)]])

def rotZ(angle: float) -> npt.NDArray[np.float64]:
    """Compute a 3x3 rotation matrix around the Z axis.

    Parameters
    ----------
    angle : float
        Rotation angle in radians.

    Returns
    -------
    R : np.ndarray, shape (3, 3)
        Rotation matrix around Z.
    """
    return np.array([[np.cos(angle), -np.sin(angle), 0],
                     [np.sin(angle),  np.cos(angle), 0],
                     [0, 0, 1]])

def get_premult_mat_rot(
    angles: npt.NDArray[np.float64],
    order: Union[str, Sequence[str]],
) -> npt.NDArray[np.float64]:
    """Convert 3 intrinsic Euler angles to a rotation matrix via pre-multiplication.

    The resulting matrix R can be applied as v' = R @ v to rotate a vector.

    Parameters
    ----------
    angles : np.ndarray, shape (3,)
        Euler angles in radians.
    order : str or list of str
        Euler rotation order, e.g. ``'XYZ'`` or ``['X', 'Y', 'Z']``.

    Returns
    -------
    R : np.ndarray, shape (3, 3)
        Combined rotation matrix.
    """
    order2fun = {'X':rotX,
                 'Y':rotY,
                 'Z':rotZ}
    return order2fun[order[0]](angles[0]) @ order2fun[order[1]](angles[1]) @ order2fun[order[2]](angles[2])


#--------------------------------------------------------------------------------------------

def batch_rotX(angles: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute rotation matrices around the X axis for a batch of angles.

    Parameters
    ----------
    angles : np.ndarray, shape (N,)
        Rotation angles in radians.

    Returns
    -------
    R : np.ndarray, shape (N, 3, 3)
        Batch of rotation matrices around X.
    """
    c = np.cos(angles)
    s = np.sin(angles)
    N = len(angles)
    R = np.zeros((N, 3, 3))
    R[:, 0, 0] = 1
    R[:, 1, 1] = c
    R[:, 1, 2] = -s
    R[:, 2, 1] = s
    R[:, 2, 2] = c
    return R

def batch_rotY(angles: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute rotation matrices around the Y axis for a batch of angles.

    Parameters
    ----------
    angles : np.ndarray, shape (N,)
        Rotation angles in radians.

    Returns
    -------
    R : np.ndarray, shape (N, 3, 3)
        Batch of rotation matrices around Y.
    """
    c = np.cos(angles)
    s = np.sin(angles)
    N = len(angles)
    R = np.zeros((N, 3, 3))
    R[:, 0, 0] = c
    R[:, 0, 2] = s
    R[:, 1, 1] = 1
    R[:, 2, 0] = -s
    R[:, 2, 2] = c
    return R

def batch_rotZ(angles: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute rotation matrices around the Z axis for a batch of angles.

    Parameters
    ----------
    angles : np.ndarray, shape (N,)
        Rotation angles in radians.

    Returns
    -------
    R : np.ndarray, shape (N, 3, 3)
        Batch of rotation matrices around Z.
    """
    c = np.cos(angles)
    s = np.sin(angles)
    N = len(angles)
    R = np.zeros((N, 3, 3))
    R[:, 0, 0] = c
    R[:, 0, 1] = -s
    R[:, 1, 0] = s
    R[:, 1, 1] = c
    R[:, 2, 2] = 1
    return R

def batch_get_premult_mat_rot(
    angles: npt.NDArray[np.float64],
    order: Union[str, Sequence[str]],
) -> npt.NDArray[np.float64]:
    """Convert batched Euler angles to rotation matrices via pre-multiplication.

    Parameters
    ----------
    angles : np.ndarray, shape (N, 3)
        Euler angles in radians, one triplet per row.
    order : str or list of str
        Euler rotation order, e.g. ``'ZYX'`` or ``['Z', 'Y', 'X']``.

    Returns
    -------
    R : np.ndarray, shape (N, 3, 3)
        Batch of combined rotation matrices.
    """
    order2fun = {'X': batch_rotX, 'Y': batch_rotY, 'Z': batch_rotZ}
    R1 = order2fun[order[0]](angles[:, 0])
    R2 = order2fun[order[1]](angles[:, 1])
    R3 = order2fun[order[2]](angles[:, 2])
    return R1 @ R2 @ R3  # (N,3,3) @ (N,3,3) @ (N,3,3) via numpy broadcasting

#--------------------------------------------------------------------------------------------

# Axis detection utilities
# These are used by bvhplot and by ML pipeline methods (foot contacts, root trajectory)

def get_main_direction(
    coord_array: npt.NDArray[np.float64],
    tol: float = 1e-6,
) -> str | None:
    """Return the signed axis string (e.g. ``'+y'``) for the dominant component.

    Parameters
    ----------
    coord_array : np.ndarray
        1-D array of length 3 representing an (x, y, z) vector.
    tol : float, optional
        Minimum vector norm to consider valid (default ``1e-6``).
        Vectors shorter than this return ``None``.

    Returns
    -------
    main_dir : str or None
        Signed axis label such as ``'+x'``, ``'-z'``, etc.
        Returns ``None`` if the vector norm is below *tol*.
    """
    if float(np.linalg.norm(coord_array)) < tol:
        return None

    main_direction_idx = int(np.argmax(np.abs(coord_array)))
    if coord_array[main_direction_idx] < 0:
        main_dir = "-"
    else:
        main_dir = "+"

    if main_direction_idx == 0:
        main_dir += "x"
    elif main_direction_idx == 1:
        main_dir += "y"
    elif main_direction_idx == 2:
        main_dir += "z"
    else:
        raise ValueError("Invalid index")

    return main_dir


def extract_sign(ax: str) -> bool:
    """Return ``True`` if the axis string has a ``'+'`` sign, ``False`` if ``'-'``.

    Parameters
    ----------
    ax : str
        Signed axis string, e.g. ``'+x'`` or ``'-z'``.

    Returns
    -------
    is_positive : bool
        ``True`` for positive, ``False`` for negative.
    """
    if ax[0] == '+':
        return True
    elif ax[0] == '-':
        return False
    else:
        raise ValueError("The sign of the axis should be either '+' or '-'.")


# ===========================================================================
# ORIENTATION HELPERS (internal)
# ===========================================================================
#
# This block replaces the old monolithic `get_forw_up_axis()` function with
# a set of single-purpose helpers. They cover three distinct concepts that
# the old function conflated:
#
#   1. WORLD UP — gravity axis of the BVH coordinate system. Constant per
#      file. Auto-detected from the first animation frame with rest-pose
#      topology as fallback.
#   2. TOPOLOGICAL LATERAL/UP — the character's own L/R and vertical axes
#      in rest pose. Used by mirror/retarget; independent of the animation.
#   3. FRAME-SPECIFIC FORWARD — the direction the character is *actually*
#      facing in world space at a given frame, derived from live joint
#      positions. Used for camera placement in visualizations.
#
# Layer diagram (→ = "used by", top = primitives, bottom = public API):
#
#     _axis_to_vector                _validate_axis_string
#           │                               │
#           ├──────────┐                    │
#           ▼          ▼                    ▼
#   _rest_offset_   _find_lr_         [Bvh.world_up setter]
#   from_node     joint_pairs
#           │          │
#           ├──────────┤
#           ▼          ▼
#       _rest_upward, _rest_leftward
#           │          │       │
#           │          │       └──→ used by transforms.mirror()
#           │          ▼
#           │    _world_leftward_unit_at_frame
#           │          │       │
#           │          ▼       │
#           │    _compute_     │
#           │    forward_at    │   _signed_rotation_delta_around_axis
#           │          │       │          │
#           ▼          ▼       │          │
#       _infer_world_up        │          │
#           │                  │          │
#           ▼                  ▼          ▼
#   ┌──────────────┐   ┌───────────┐   ┌──────────────────────┐
#   │ Bvh.world_up │   │Bvh.forward│   │ bvhplot follow-mode  │
#   │  (property)  │   │    _at()  │   │ (render_mpl/opencv)  │
#   └──────────────┘   └───────────┘   └──────────────────────┘
#
# Public surface: ONLY `Bvh.world_up` (property + setter) and
# `Bvh.forward_at(frame=0)` (method). Everything else is underscore-prefixed
# and intended for internal use by pybvh modules.
# ---------------------------------------------------------------------------

_VALID_AXIS_STRINGS = frozenset(
    {'+x', '-x', '+y', '-y', '+z', '-z'})

_AXIS_CHAR_TO_IDX = {'x': 0, 'y': 1, 'z': 2}


def _axis_to_vector(ax: str) -> npt.NDArray[np.float64]:
    """Convert a signed axis string ('+y') to a unit vector ([0, 1, 0])."""
    vec = np.zeros(3)
    vec[_AXIS_CHAR_TO_IDX[ax[1]]] = 1.0 if ax[0] == '+' else -1.0
    return vec


def _rest_upward(bvh: Bvh) -> str:
    """Infer the skeleton's topological up axis from the rest pose.

    Uses the same heuristic the old get_forw_up_axis used: iterate named
    body parts ("head", "neck", "chest", "spine") in priority order and
    check which axis their rest-pose cumulative offset dominates. Falls
    back to the axis with the largest spread across all joints.

    Parameters
    ----------
    bvh : Bvh
        The skeleton.

    Returns
    -------
    str
        Signed axis string (e.g. '+z').
    """
    rest = bvh.rest_pose_coords(mode='coordinates')  # (N, 3)
    local_coord = rest - rest[0]  # root at origin

    up_body_parts = ["head", "neck", "chest", "spine"]
    for part_name in up_body_parts:
        for joint in bvh.nodes:
            if joint.name.lower() == part_name:
                coord = local_coord[bvh.node_index[joint.name]]
                direction = get_main_direction(coord)
                if direction is not None:
                    return direction

    # Fallback: axis with largest spread across all joints
    spread = np.ptp(local_coord, axis=0)
    up_idx_fallback = int(np.argmax(spread))
    # Use the mean to determine sign (positive mean = positive axis direction)
    sign = '+' if np.mean(local_coord[:, up_idx_fallback]) >= 0 else '-'
    return sign + 'xyz'[up_idx_fallback]


def _rest_offset_from_node(node: object) -> npt.NDArray[np.float64]:
    """Cumulative rest-pose offset from root for a joint (walks parent chain)."""
    offset = np.zeros(3)
    current = node
    while current is not None:
        offset = offset + np.array(current.offset)  # type: ignore[attr-defined]
        current = current.parent  # type: ignore[attr-defined]
    return offset


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


def _find_lr_joint_pairs(
    bvh: Bvh, mapping: dict[str, str] | None = None,
) -> list[tuple[object, object]]:
    """Return ``(left_node, right_node)`` pairs from an L/R name mapping.

    Parameters
    ----------
    bvh : Bvh
        The skeleton.
    mapping : dict or None
        If provided, pair up joints according to this mapping. If None,
        reads ``bvh.lr_mapping`` (the cached auto-detected or
        user-supplied mapping). Returns an empty list if the mapping
        is empty or any referenced name is unknown.
    """
    if mapping is None:
        mapping = bvh.lr_mapping
    if not mapping:
        return []
    ni = bvh.node_index
    pairs: list[tuple[object, object]] = []
    for left_name, right_name in _iter_unique_lr_pairs(mapping):
        if left_name in ni and right_name in ni:
            pairs.append((bvh.nodes[ni[left_name]], bvh.nodes[ni[right_name]]))
    return pairs


def _rest_leftward(
    bvh: Bvh, mapping: dict[str, str] | None = None,
) -> str | None:
    """Infer the skeleton's leftward axis from rest-pose L/R symmetry.

    Averages the left-minus-right rest-pose cumulative offsets across all
    matching L/R joint pairs, projects onto the horizontal plane, and
    returns the dominant signed axis — a vector pointing from the
    character's right toward their left. Convention matches the public
    :meth:`Bvh.left_at` method (``up × forward``) via the right-hand
    rule. Used by transforms that need a stable topological orientation
    reference (mirror, retarget).

    Parameters
    ----------
    bvh : Bvh
        The skeleton.
    mapping : dict or None
        If provided, pair up joints according to this explicit mapping
        instead of reading ``bvh.lr_mapping``. Useful when the caller
        wants to compute leftward for a mapping that isn't cached on
        the object.

    Returns
    -------
    str or None
        Signed axis string (e.g. ``'-x'``) pointing toward the
        character's left, or ``None`` if no L/R pairs are available or
        the averaged offset is degenerate.
    """
    up_ax = _rest_upward(bvh)
    up_idx = _AXIS_CHAR_TO_IDX[up_ax[1]]

    pairs = _find_lr_joint_pairs(bvh, mapping=mapping)
    if not pairs:
        return None

    leftward_vectors = [
        _rest_offset_from_node(l) - _rest_offset_from_node(r)
        for l, r in pairs
    ]
    avg_leftward = np.mean(leftward_vectors, axis=0)
    avg_leftward[up_idx] = 0.0  # project to ground plane

    leftward_ax = get_main_direction(avg_leftward)
    if leftward_ax is None or leftward_ax[1] == up_ax[1]:
        return None
    return leftward_ax


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

    Returns
    -------
    str
        Signed axis string (e.g. '+y').
    """
    # Compute rest-pose topology once; used both as fallback and as the
    # reference to compare against when emitting the disagreement warning.
    try:
        rest_up = _rest_upward(bvh)
    except Exception:
        rest_up = None

    # Need animation frames to do first-frame inference
    if bvh.frame_count == 0:
        return rest_up if rest_up is not None else '+y'

    # Try to find a "head-ish" and "hips-ish" joint in the skeleton.
    # Priority: head -> neck -> last joint in spine chain.
    name_lookup = {n.name.lower(): i for i, n in enumerate(bvh.nodes)}
    head_idx = None
    for candidate in ('head', 'neck', 'chest', 'spine'):
        if candidate in name_lookup:
            head_idx = name_lookup[candidate]
            break
    # Hips = the root (index 0)
    hips_idx = 0

    if head_idx is None or head_idx == hips_idx:
        return rest_up if rest_up is not None else '+y'

    frame0 = bvh.node_positions(frame_num=0)
    head_hips = frame0[head_idx] - frame0[hips_idx]

    # Check for a clear dominant axis: the largest component must be
    # strictly larger than the second-largest (ratio > 2). Otherwise
    # the pose is ambiguous (character crouched / lying / leaning).
    abs_components = np.abs(head_hips)
    sorted_abs = np.sort(abs_components)
    if sorted_abs[-1] < 2.0 * sorted_abs[-2] or sorted_abs[-1] < 1e-6:
        return rest_up if rest_up is not None else '+y'

    # Clear winner from animation data
    frame_up = get_main_direction(head_hips)
    if frame_up is None:
        return rest_up if rest_up is not None else '+y'

    # Warn if rest-pose topology disagrees (different dominant axis)
    if warn and rest_up is not None and rest_up[1] != frame_up[1]:
        import warnings
        warnings.warn(
            f"Rest pose suggests world up is {rest_up!r} but the first \n"
            f"animation frame's head-hips direction is closer to {frame_up!r}. \n"
            f"Using {frame_up!r} from the animation data. If this is wrong \n"
            f"for your file, set it explicitly via `bvh.world_up = '<axis>'`.",
            UserWarning,
            stacklevel=2,
        )

    return frame_up


def _world_leftward_unit_at_frame(
    bvh: Bvh,
    frame_coords: npt.NDArray[np.float64],
    world_up: str,
) -> npt.NDArray[np.float64] | None:
    """Return the character's world-space leftward **unit vector** at a frame.

    Continuous (not snapped to a signed axis), projected onto the plane
    perpendicular to ``world_up``. Averages ``(left_pos - right_pos)``
    across matching L/R joint pairs in world space at the given frame —
    a unit vector pointing from the character's right toward their
    left. Matches the ``up × forward`` right-hand-rule convention used
    by :meth:`Bvh.left_at`.

    Returns ``None`` if no L/R pairs exist or the averaged leftward is
    degenerate (parallel to world up, or zero). Callers should handle
    ``None`` by falling back to the topological ``_rest_leftward``.
    """
    up_vec = _axis_to_vector(world_up)
    pairs = _find_lr_joint_pairs(bvh)
    if not pairs:
        return None

    leftward_diffs = [
        frame_coords[bvh.node_index[l.name]] - frame_coords[bvh.node_index[r.name]]  # type: ignore[attr-defined]
        for l, r in pairs
    ]
    avg_leftward = np.mean(leftward_diffs, axis=0)
    # Project onto plane perpendicular to world_up
    avg_leftward = avg_leftward - np.dot(avg_leftward, up_vec) * up_vec
    norm = float(np.linalg.norm(avg_leftward))
    if norm < 1e-6:
        return None
    return avg_leftward / norm


def _signed_rotation_delta_around_axis(
    v_from: npt.NDArray[np.float64],
    v_to: npt.NDArray[np.float64],
    axis_vec: npt.NDArray[np.float64],
) -> float:
    """Signed angle (degrees) rotating ``v_from`` to ``v_to`` around ``axis_vec``.

    Positive = counter-clockwise when viewed from the direction of
    ``axis_vec``. Both input vectors are assumed unit-length and already
    in the plane perpendicular to ``axis_vec``.
    """
    cos_a = float(np.clip(np.dot(v_from, v_to), -1.0, 1.0))
    sin_a = float(np.dot(np.cross(v_from, v_to), axis_vec))
    return float(np.degrees(np.arctan2(sin_a, cos_a)))


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
         no L/R pairs found, use :func:`_rest_leftward` and cross with
         world_up.

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
    up_vec = _axis_to_vector(world_up)
    leftward_vec = _world_leftward_unit_at_frame(bvh, frame_coords, world_up)

    # Fallback: use rest-pose leftward (topology) if current-frame
    # leftward is degenerate or no L/R pairs exist.
    if leftward_vec is None:
        rest_left = _rest_leftward(bvh)
        if rest_left is None:
            # Truly no information available — pick an arbitrary horizontal
            # direction so we at least return a valid axis.
            fallback = {'y': '+z', 'z': '+x', 'x': '+y'}
            return fallback[world_up[1]]
        leftward_vec = _axis_to_vector(rest_left)

    forward_vec = np.cross(leftward_vec, up_vec)
    forward_ax = get_main_direction(forward_vec)
    if forward_ax is None or forward_ax[1] == world_up[1]:
        fallback = {'y': '+z', 'z': '+x', 'x': '+y'}
        return fallback[world_up[1]]
    return forward_ax


def _validate_axis_string(value: object) -> str:
    """Validate and normalize a signed axis string.

    Accepts '+x'/'-x'/'+y'/'-y'/'+z'/'-z' (case-insensitive in the
    axis char). Returns the normalized form. Raises ValueError on
    any other input.
    """
    normalized = value.lower() if isinstance(value, str) and len(value) == 2 else value
    if normalized not in _VALID_AXIS_STRINGS:
        raise ValueError(
            f"Axis must be one of {sorted(_VALID_AXIS_STRINGS)}, got {value!r}")
    return normalized  # type: ignore[return-value]


def _axis_aligned_rotation(
    from_ax: str,
    to_ax: str,
) -> npt.NDArray[np.float64]:
    """3x3 rotation matrix mapping signed axis ``from_ax`` to ``to_ax``.

    Only works for axis-aligned unit vectors.  Returns exact matrices
    (entries are 0, 1, or -1) for lossless coordinate transforms.

    Parameters
    ----------
    from_ax, to_ax : str
        Signed axis strings, e.g. ``'+y'``, ``'-z'``.

    Returns
    -------
    R : ndarray of shape (3, 3)
        Orthogonal rotation matrix with ``det = +1``.
    """
    from_vec = _axis_to_vector(from_ax)
    to_vec = _axis_to_vector(to_ax)

    if np.allclose(from_vec, to_vec):
        return np.eye(3, dtype=np.float64)

    if np.allclose(from_vec, -to_vec):
        # 180-degree rotation around a perpendicular axis
        from_idx = _AXIS_CHAR_TO_IDX[from_ax[1]]
        perp_idx = (from_idx + 1) % 3
        rot_funcs = {0: rotX, 1: rotY, 2: rotZ}
        R = rot_funcs[perp_idx](np.pi)
        return np.round(R).astype(np.float64)

    # 90-degree rotation: cross product gives the rotation axis and sign
    cross = np.cross(from_vec, to_vec)
    rot_idx = int(np.argmax(np.abs(cross)))
    rot_sign = np.sign(cross[rot_idx])
    rot_funcs = {0: rotX, 1: rotY, 2: rotZ}
    R = rot_funcs[rot_idx](rot_sign * np.pi / 2)
    return np.round(R).astype(np.float64)

