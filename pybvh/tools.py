from pathlib import Path
import numpy as np



def are_permutations(str1, str2):
    """
    Test if two strings are permutations of each others. 
    Return a bool object.
    """
    if len(str1) != len(str2):
        return False
    
    char_freq = {}
    
    for char in str1:
        char_freq[char] = char_freq.get(char, 0) + 1
        
    for char in str2:
        if char not in char_freq or char_freq[char] == 0:
            return False
        char_freq[char] -= 1
            
    return True

#--------------------------------------------------------------------------------------------

    
def test_file(filepath):
    """
    Test if a filepath exists and is a bvh file. Return a Path object.
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

def rotX(angle):
    """ angle as rad already"""
    return np.array([[1, 0, 0],
                     [0, np.cos(angle), -np.sin(angle)],
                     [0, np.sin(angle), np.cos(angle)]])

def rotY(angle):
    """ angle as rad already"""
    return np.array([[np.cos(angle), 0, np.sin(angle)],
                     [0, 1, 0],
                     [-np.sin(angle), 0, np.cos(angle)]])

def rotZ(angle):
    """ angle as rad already"""
    return np.array([[np.cos(angle), -np.sin(angle), 0],
                     [np.sin(angle),  np.cos(angle), 0],
                     [0, 0, 1]])

def get_premult_mat_rot(angles, order):
    """
    Transform 3 instrinsic euler angles in a (matrix premultiply) rotation matrix.
    Input : - angles : np.1d array of 3 euler angles in radian
            - order : string 'XYZ' or list ['X', 'Y', 'Z'] of the euler order
                        of the angles in the given 'angles' argument

    To obtain a rotated vector given R the rotation matrix obtain here,
    the formula is v' = Rv, with v' the new rotated vector.
    """
    order2fun = {'X':rotX,
                 'Y':rotY,
                 'Z':rotZ}
    return order2fun[order[0]](angles[0]) @ order2fun[order[1]](angles[1]) @ order2fun[order[2]](angles[2])


#--------------------------------------------------------------------------------------------

def batch_rotX(angles):
    """Batch rotation matrices around X axis. angles: (N,) radians -> (N,3,3)"""
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

def batch_rotY(angles):
    """Batch rotation matrices around Y axis. angles: (N,) radians -> (N,3,3)"""
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

def batch_rotZ(angles):
    """Batch rotation matrices around Z axis. angles: (N,) radians -> (N,3,3)"""
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

def batch_get_premult_mat_rot(angles, order):
    """
    Batch Euler angles to rotation matrices via pre-multiplication.
    angles: (N, 3) radians, order: list of 3 chars e.g. ['Z','Y','X']
    Returns: (N, 3, 3)
    """
    order2fun = {'X': batch_rotX, 'Y': batch_rotY, 'Z': batch_rotZ}
    R1 = order2fun[order[0]](angles[:, 0])
    R2 = order2fun[order[1]](angles[:, 1])
    R3 = order2fun[order[2]](angles[:, 2])
    return R1 @ R2 @ R3  # (N,3,3) @ (N,3,3) @ (N,3,3) via numpy broadcasting

#--------------------------------------------------------------------------------------------

