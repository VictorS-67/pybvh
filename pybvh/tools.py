from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union

import numpy as np
import numpy.typing as npt



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
