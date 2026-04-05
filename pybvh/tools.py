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
        raise ImportError(f'{filepath} is not a bvh file')
    elif not filepath.exists():
        raise ImportError(f'could not find the file {filepath}')
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


def get_forw_up_axis(bvh_object: Bvh, frame: npt.NDArray[np.float64]) -> dict[str, str]:
    """Infer the forward and upward axes from a skeleton frame (human only).

    Uses heuristics based on joint names to determine the upward axis
    (looks for "head", "neck", "chest", "spine" in priority order,
    skipping joints at the root origin) and left-right symmetry from
    rest-pose offsets for the forward axis (pose-independent). The
    lateral axis is found by averaging the right-minus-left offset of
    all matching Left/Right joint pairs, then forward is derived via
    ``cross(up, lateral)``.

    Parameters
    ----------
    bvh_object : Bvh
        The BVH object containing the skeleton hierarchy.
    frame : np.ndarray
        Spatial coordinates of shape ``(N, 3)`` for a single frame.

    Returns
    -------
    directions : dict
        Dictionary with keys ``'forward'`` and ``'upward'``, each
        mapping to a signed axis string (e.g. ``'+y'``, ``'-z'``).

    Raises
    ------
    ValueError
        If *frame* has wrong shape or node count doesn't match.
    """
    # --- Input validation ---
    if frame.ndim != 2 or frame.shape[1] != 3:
        raise ValueError(f"Expected frame shape (N, 3), got {frame.shape}")
    if frame.shape[0] != len(bvh_object.nodes):
        raise ValueError(
            f"Frame has {frame.shape[0]} nodes but skeleton has "
            f"{len(bvh_object.nodes)} nodes")

    # work with local coordinates (root at origin)
    local_coord = frame - frame[0]  # (N, 3) - (3,) broadcast

    # --- Up-axis detection ---
    # Iterate named body parts in priority order; skip zero-offset joints
    up_body_parts = ["head", "neck", "chest", "spine"]
    up_ax: str | None = None
    for part_name in up_body_parts:
        for joint in bvh_object.nodes:
            if joint.name.lower() == part_name:
                coord = local_coord[bvh_object.node_index[joint.name]]
                up_ax = get_main_direction(coord)
                if up_ax is not None:
                    break
        if up_ax is not None:
            break

    # Fallback: axis with largest spread across all joints (always positive)
    if up_ax is None:
        spread = np.ptp(local_coord, axis=0)  # (3,)
        up_idx_fallback = int(np.argmax(spread))
        up_ax = "+" + "xyz"[up_idx_fallback]

    up_idx = {"x": 0, "y": 1, "z": 2}[up_ax[1]]

    # --- Forward-axis detection (pose-independent) ---
    # Primary: use left-right symmetry from rest-pose offsets to find the
    # lateral axis, then derive forward = cross(up, lateral).
    # This is more robust than toe end-sites because nearly every humanoid
    # skeleton has Left/Right named joints and we can average multiple pairs.
    forward_ax: str | None = None

    # Cumulative rest-pose offset from root for a joint
    def _rest_offset(node: object) -> npt.NDArray[np.float64]:
        offset = np.zeros(3)
        current = node
        while current is not None:
            offset = offset + np.array(current.offset)
            current = current.parent
        return offset

    # Find matching Left/Right joint pairs
    left_joints: dict[str, object] = {}
    right_joints: dict[str, object] = {}
    for node in bvh_object.nodes:
        if node.is_end_site():
            continue
        lower = node.name.lower()
        if "left" in lower:
            left_joints[node.name] = node
        elif "right" in lower:
            right_joints[node.name] = node

    lateral_vectors: list[npt.NDArray[np.float64]] = []
    for lname, lnode in left_joints.items():
        rname = lname.replace("Left", "Right").replace("left", "right")
        if rname in right_joints:
            diff = _rest_offset(right_joints[rname]) - _rest_offset(lnode)
            lateral_vectors.append(diff)

    if lateral_vectors:
        avg_lateral = np.mean(lateral_vectors, axis=0)
        avg_lateral[up_idx] = 0.0  # project to ground plane
        lateral_ax = get_main_direction(avg_lateral)
        if lateral_ax is not None and lateral_ax[1] != up_ax[1]:
            up_vec = np.zeros(3)
            up_vec[up_idx] = 1.0 if up_ax[0] == "+" else -1.0
            lat_vec = np.zeros(3)
            lat_idx = {"x": 0, "y": 1, "z": 2}[lateral_ax[1]]
            lat_vec[lat_idx] = 1.0 if lateral_ax[0] == "+" else -1.0
            fwd_vec = np.cross(up_vec, lat_vec)
            forward_ax = get_main_direction(fwd_vec)

    # Fallback: default mapping from up-axis
    default_up2front: dict[str, str] = {"x": "+y", "y": "+z", "z": "+x"}
    if forward_ax is None:
        forward_ax = default_up2front[up_ax[1]]

    # --- Orthogonality guard ---
    if forward_ax[1] == up_ax[1]:
        forward_ax = default_up2front[up_ax[1]]

    return {'forward': forward_ax, 'upward': up_ax}


def get_up_axis_index(bvh_object: Bvh, frame: npt.NDArray[np.float64]) -> int:
    """Return the integer index (0=x, 1=y, 2=z) of the upward axis.

    Convenience wrapper around :func:`get_forw_up_axis` for methods
    that need the up-axis as an integer index.

    Parameters
    ----------
    bvh_object : Bvh
        The BVH object containing the skeleton hierarchy.
    frame : np.ndarray
        Spatial coordinates of shape ``(N, 3)`` for a single frame.

    Returns
    -------
    up_idx : int
        0 for x, 1 for y, 2 for z.
    """
    directions = get_forw_up_axis(bvh_object, frame)
    axis_char = directions['upward'][1]  # e.g. 'y' from '+y'
    return {'x': 0, 'y': 1, 'z': 2}[axis_char]
