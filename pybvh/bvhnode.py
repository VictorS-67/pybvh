import numpy as np
import numbers

from .methods import _are_permutations

class BvhNode:
    """
    Class that represent a bvh nodes in the bvh Hierarchy.
    EndSites in a bvh file will become BvhNode objects. Parent class for BvhJoint class.
    This class has 2 attributes, offset and parent.
    Offset accepts a list or np array of 3 numbers.
    parent accepts either None, or a BvhNode class/subclass object.
    """


    def __init__(self, name, offset=[0.0, 0.0, 0.0], parent = None):
        self.name = name
        self.offset = offset
        self.parent = parent

    @property
    def name(self):
        return self._name
    @name.setter
    def name(self, value):
        if not isinstance(value, str):
            raise ValueError("name should be a string type")
        self._name = value

    @property
    def offset(self):
        return self._offset
    @offset.setter
    def offset(self, value):
        # ensure that we have 3 numbers. Convert to numpy list
        #if len(value) != 3 or any([not isinstance(x, numbers.Number) for x in value]):
        #    raise ValueError("offset should be a list or numpy array of 3 numbers")
        self._offset = np.array(value)

    @property
    def parent(self):
        return self._parent
    @parent.setter
    def parent(self, value):
        #parent needs to be either None or an instance of BvhNode
        if value != None and not isinstance(value, BvhNode):
            raise ValueError("parent should either be None or a BvhNode class/subclasse object")
        self._parent = value

    def __str__(self):
        return f'{self.name}'
        
    def __repr__(self):
        return f'BvhNode(name = {self.name}, offset = {self.offset}, parent = {self.parent})'



#---------------------------------------------------------------------------------------------

class BvhJoint(BvhNode):
    """
    Subclass of BvhNode class. It has 4 attributes: offset, rot_channels, children, and parent.
    offset accepts a list or np array of 3 numbers.
    rot_channels accept a list of three letters X Y or Z or a string of three letters X Y or Z
    children accepts list of BvhNode class/subclass objects.
    parent accepts either None, or a BvhNode class/subclass object.
    """
    def __init__(self, name, offset=[0.0, 0.0, 0.0],
                 rot_channels=['Z', 'Y', 'X'], children=[],
                 parent = None):
        #inheritance
        super().__init__(name, offset, parent)

        self.rot_channels = rot_channels
        self.children = children
        
    @property
    def rot_channels(self):
        return self._rot_channels
    @rot_channels.setter
    def rot_channels(self, value):
        self._rot_channels = self._check_channels(value)

    @property
    def children(self):
        return self._children
    @children.setter
    def children(self, value):
        if (not isinstance(value, list)) or any([not isinstance(x, BvhNode) for x in value]):
            raise ValueError("children should be a list of BvhNode class/subclasse objects")
        self._children = value

    
    def __str__(self):
        return f'JOINT {self.name}'

    def __repr__(self):
        children_list = []
        for child in self.children:
            if 'End Site' in child.name :
                children_list.append(f'{child.__str__()}')
            else:
                children_list.append(f'BvhJoint({child.__str__()})')
        return f'BvhJoint(name = {self.name}, offset = {self.offset}, rot_channels = {self.rot_channels}, children = {str(children_list)}, parent = {self.parent})'


    def _check_channels(self, value):
        # we will check if the channels are either a list of 3 elements,
        # or a string of 3 elements, belonging to a permutation of 'XYZ'
        # we return the result as a list of 3 characters
        problem = True
        er = ValueError("the channels should be a list or a string of 3 elements, one of each from 'X' 'Y' 'Z'")
        if isinstance(value, str):
            if not _are_permutations('XYZ', value):
                raise er
            else:
                return list(value)
        elif isinstance(value, list):
            try:
                str_conv = ''.join(value)
            except:
                raise er
            if not _are_permutations('XYZ', str_conv):
                raise er
            else:
                return value
        else:
            raise er



#---------------------------------------------------------------------------------------------

class BvhRoot(BvhJoint):
    """
    Subclass of BvhJoint class. It has 5 attributes: offset, pos_channels,
    rot_channels, children, and parent.
    Offset and rot_channels both accept list or np array of 3 numbers.
    children accepts list of BvhNode class/subclass objects.
    parent accepts either None, or a BvhNode class/subclass object.
    """
    def __init__(self, name='root', offset=[0.0, 0.0, 0.0],
                 pos_channels=['X', 'Y', 'Z'], rot_channels=['Z', 'Y', 'X'],
                 children=[], parent = None):
        #inheritance
        super().__init__(name, offset, rot_channels, children, parent)

        self.pos_channels = pos_channels
        
    @property
    def pos_channels(self):
        return self._pos_channels
    @pos_channels.setter
    def pos_channels(self, value):
        self._pos_channels = self._check_channels(value)

    def __str__(self):
        return f'ROOT {self.name}'

    def __repr__(self):
        super_str = super().__repr__()
        #the parent classe repr is f'BvhJoint(name = {self.name}, offset = {self.offset},
        #  rot_channels = {self.rot_channels}, children = {str(children_list)}, parent = {self.parent})'
        super_str_list = super_str.split(',')
        super_str_list[0] = f'BvhRoot(name = {self.name}'
        super_str_list.insert(2, f' pos_channels = {self.pos_channels}')
        return ','.join(super_str_list)
        #return f'BvhRoot(name = {self.name}, offset = {self.offset}, pos_channels = {self.pos_channels},
        #  rot_channels = {self.rot_channels}, children = {str(children_list)}, parent = {self.parent})'
