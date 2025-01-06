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

#--------------------------------------------------------------------------------------------

