from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .tools import are_permutations

class BvhNode:
    """A BVH hierarchy node, representing end-sites or serving as base for joints.

    Attributes
    ----------
    name : str
        Name of the node.
    offset : np.ndarray
        3-element array of positional offset values.
    parent : BvhNode or None
        Parent node in the hierarchy, or None if this is a root.
    """


    def __init__(self, name: str, offset: list[float] | npt.NDArray[np.float64] = [0.0, 0.0, 0.0], parent: BvhNode | None = None) -> None:
        self.name = name
        self.offset = offset  # type: ignore[assignment]
        self.parent = parent

    @property
    def name(self) -> str:
        return self._name
    @name.setter
    def name(self, value: str) -> None:
        if not isinstance(value, str):
            raise ValueError("name should be a string type")
        self._name = value

    @property
    def offset(self) -> npt.NDArray[np.float64]:
        return self._offset
    @offset.setter
    def offset(self, value: list[float] | npt.NDArray[np.float64]) -> None:
        # ensure that we have 3 numbers. Convert to numpy list
        #if len(value) != 3 or any([not isinstance(x, numbers.Number) for x in value]):
        #    raise ValueError("offset should be a list or numpy array of 3 numbers")
        self._offset: npt.NDArray[np.float64] = np.array(value, dtype=np.float64)

    @property
    def parent(self) -> BvhNode | None:
        return self._parent
    @parent.setter
    def parent(self, value: BvhNode | None) -> None:
        #parent needs to be either None or an instance of BvhNode
        if value != None and not isinstance(value, BvhNode):
            raise ValueError("parent should either be None or a BvhNode class/subclasse object")
        self._parent = value

    def __str__(self) -> str:
        return f'{self.name}'

    def __repr__(self) -> str:
        return f'BvhNode(name = {self.name}, offset = {self.offset}, parent = {self.parent})'

    def is_end_site(self) -> bool:
        return True

    def is_root(self) -> bool:
        return False


#---------------------------------------------------------------------------------------------

class BvhJoint(BvhNode):
    """A BVH joint node with rotation channels and children.

    Attributes
    ----------
    name : str
        Name of the joint.
    offset : np.ndarray
        3-element array of positional offset values.
    rot_channels : list of str
        Rotation channel order as a permutation of ``['X', 'Y', 'Z']``.
    children : list of BvhNode
        Child nodes in the hierarchy.
    parent : BvhNode or None
        Parent node, or None if this is a root.
    """
    def __init__(self, name: str, offset: list[float] | npt.NDArray[np.float64] = [0.0, 0.0, 0.0],
                 rot_channels: list[str] | str = ['Z', 'Y', 'X'], children: list[BvhNode] = [],
                 parent: BvhNode | None = None) -> None:
        #inheritance
        super().__init__(name, offset, parent)

        self._frozen = False
        self.rot_channels = rot_channels  # type: ignore[assignment]
        self.children = children

    @property
    def rot_channels(self) -> list[str]:
        return self._rot_channels
    @rot_channels.setter
    def rot_channels(self, value: list[str] | str) -> None:
        if getattr(self, '_frozen', False):
            raise AttributeError(
                "rot_channels is frozen. Use "
                "Bvh.single_joint_euler_angle(joint_name, new_order) or "
                "Bvh.change_all_euler_orders(new_order) to change rotation order.")
        self._rot_channels = self._check_channels(value)

    def _set_rot_channels_internal(self, value: list[str] | str) -> None:
        """Set rot_channels bypassing the freeze check.

        Parameters
        ----------
        value : list of str or str
            Rotation channel order as a permutation of ``'XYZ'``.
        """
        self._rot_channels = self._check_channels(value)

    @property
    def children(self) -> list[BvhNode]:
        return self._children
    @children.setter
    def children(self, value: list[BvhNode]) -> None:
        if (not isinstance(value, list)) or any([not isinstance(x, BvhNode) for x in value]):
            raise ValueError("children should be a list of BvhNode class/subclasse objects")
        self._children = value


    def __str__(self) -> str:
        return f'JOINT {self.name}'

    def __repr__(self) -> str:
        children_list = []
        for child in self.children:
            if child.is_end_site():
                children_list.append(f'{child.__str__()}')
            else:
                children_list.append(f'BvhJoint({child.__str__()})')
        return f'BvhJoint(name = {self.name}, offset = {self.offset}, rot_channels = {self.rot_channels}, children = {str(children_list)}, parent = {self.parent})'


    def _check_channels(self, value: list[str] | str) -> list[str]:
        # we will check if the channels are either a list of 3 elements,
        # or a string of 3 elements, belonging to a permutation of 'XYZ'
        # we return the result as a list of 3 characters
        problem = True
        er = ValueError("the channels should be a list or a string of 3 elements, one of each from 'X' 'Y' 'Z'")
        if isinstance(value, str):
            if not are_permutations('XYZ', value):
                raise er
            else:
                return list(value)
        elif isinstance(value, list):
            try:
                str_conv = ''.join(value)
            except:
                raise er
            if not are_permutations('XYZ', str_conv):
                raise er
            else:
                return value
        else:
            raise er

    def is_end_site(self) -> bool:
        return False


#---------------------------------------------------------------------------------------------

class BvhRoot(BvhJoint):
    """A BVH root joint with both position and rotation channels.

    Attributes
    ----------
    name : str
        Name of the root joint.
    offset : np.ndarray
        3-element array of positional offset values.
    pos_channels : list of str
        Position channel order as a permutation of ``['X', 'Y', 'Z']``.
    rot_channels : list of str
        Rotation channel order as a permutation of ``['X', 'Y', 'Z']``.
    children : list of BvhNode
        Child nodes in the hierarchy.
    parent : BvhNode or None
        Parent node, or None.
    """
    def __init__(self, name: str = 'root', offset: list[float] | npt.NDArray[np.float64] = [0.0, 0.0, 0.0],
                 pos_channels: list[str] | str = ['X', 'Y', 'Z'], rot_channels: list[str] | str = ['Z', 'Y', 'X'],
                 children: list[BvhNode] = [], parent: BvhNode | None = None) -> None:
        #inheritance
        super().__init__(name, offset, rot_channels, children, parent)

        self.pos_channels = pos_channels  # type: ignore[assignment]

    @property
    def pos_channels(self) -> list[str]:
        return self._pos_channels
    @pos_channels.setter
    def pos_channels(self, value: list[str] | str) -> None:
        if getattr(self, '_frozen', False):
            raise AttributeError(
                "pos_channels is frozen after construction and cannot be changed.")
        self._pos_channels = self._check_channels(value)

    def __str__(self) -> str:
        return f'ROOT {self.name}'

    def __repr__(self) -> str:
        super_str = super().__repr__()
        #the parent classe repr is f'BvhJoint(name = {self.name}, offset = {self.offset},
        #  rot_channels = {self.rot_channels}, children = {str(children_list)}, parent = {self.parent})'
        super_str_list = super_str.split(',')
        super_str_list[0] = f'BvhRoot(name = {self.name}'
        super_str_list.insert(2, f' pos_channels = {self.pos_channels}')
        return ','.join(super_str_list)
        #return f'BvhRoot(name = {self.name}, offset = {self.offset}, pos_channels = {self.pos_channels},
        #  rot_channels = {self.rot_channels}, children = {str(children_list)}, parent = {self.parent})'


    def is_root(self) -> bool:
        return True
