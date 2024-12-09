from pathlib import Path

def _are_permutations(str1, str2):
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

    
def _test_file(filepath):
    """
    function to test if a filepath exists and is a bvh file. Return a Path object.
    """
    filepath = Path(filepath)
    if filepath.suffix != '.bvh':
        raise ImportError(f'{filepath} is not a bvh file')
    elif not filepath.exists():
        raise ImportError(f'could not find the file {filepath}')
    return filepath

#--------------------------------------------------------------------------------------------


