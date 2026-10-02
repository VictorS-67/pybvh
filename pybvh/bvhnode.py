"""Nodes of a BVH skeleton hierarchy.

A skeleton is a tree of :class:`BvhRoot` (exactly one, first), :class:`BvhJoint`
and :class:`BvhEndSite` nodes, all subclasses of :class:`BvhNode`. Each node
carries its name, its rest offset from its parent and links to its parent
and children; the motion itself lives in the :class:`~pybvh.bvh.Bvh` that
holds the nodes. Trees are usually built by the reader or by
:func:`~pybvh.nodes_from_table`, which wire both directions of every link.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from typing import TypeGuard


class BvhNode:
    """Base class for BVH hierarchy nodes.

    ``BvhNode`` itself carries only the data shared by every node kind (name, offset, parent). Concrete hierarchies are built from its subclasses: :class:`BvhRoot`, :class:`BvhJoint`, and :class:`BvhEndSite`. Node-kind checks go through :meth:`is_end_site` / :meth:`is_root` (never through name conventions), so a bare ``BvhNode`` cannot answer :meth:`is_end_site`.

    Attributes
    ----------
    name : str
        Name of the node.
    offset : np.ndarray
        3-element array of positional offset values, every component
        finite: the setter rejects NaN and infinity, and a ``None``
        component, which NumPy would convert to NaN.
    parent : BvhNode or None
        Parent node in the hierarchy, or None if this is a root.
    """

    def __init__(
        self,
        name: str,
        offset: list[float] | npt.NDArray[np.float64] | None = None,
        parent: BvhNode | None = None,
    ) -> None:
        self.name = name
        self.offset = offset if offset is not None else [0.0, 0.0, 0.0]  # type: ignore[assignment]
        self.parent = parent

    @property
    def name(self) -> str:
        """The node's name, as the file spells it.

        Names are not unique: real files repeat joint names, and the
        reader names an end site after its parent, so two end sites of one
        joint share a name. pybvh builds and traverses the tree by node
        identity and position, never by name. Lookups by name do exist
        (``Bvh.index``, ``Bvh.node_index``, ``Bvh.joint_index``,
        retargeting); ``node_index`` keeps the last node of a repeated
        name. Assigning anything but a ``str`` raises ``ValueError``.
        """
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        if not isinstance(value, str):
            raise ValueError("name should be a string type")
        self._name = value

    @property
    def offset(self) -> npt.NDArray[np.float64]:
        """Rest-pose offset from the parent, shape ``(3,)``.

        In the parent's frame and in the file's length unit (the unit of
        ``root_pos``). Assigning accepts any sequence of three finite
        numbers and stores it as a new float64 array; NaN, infinity, a
        ``None`` component or a wrong length raise ``ValueError``.
        """
        return self._offset

    @offset.setter
    def offset(self, value: list[float] | npt.NDArray[np.float64]) -> None:
        try:
            offset_arr = np.array(value, dtype=np.float64)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"offset should be a list or numpy array of 3 finite numbers, got {value!r}"
            ) from e
        if offset_arr.shape != (3,):
            raise ValueError(
                f"offset should be a list or numpy array of 3 finite "
                f"numbers, got shape {offset_arr.shape}"
            )
        # np.array([None, 0, 0], dtype=float64) is [nan, 0, 0]: a None
        # component (a JSON null) would otherwise pass silently.
        if not np.all(np.isfinite(offset_arr)):
            raise ValueError(
                f"offset should be a list or numpy array of 3 finite numbers, got {value!r}"
            )
        self._offset: npt.NDArray[np.float64] = offset_arr

    @property
    def parent(self) -> BvhNode | None:
        """The node this one hangs from, ``None`` for the root.

        Assigning anything but ``None`` or a :class:`BvhNode` raises
        ``ValueError``. Setting the parent does not add this node to the
        parent's ``children``: each direction of the link is set on its
        own.
        """
        return self._parent

    @parent.setter
    def parent(self, value: BvhNode | None) -> None:
        # parent needs to be either None or an instance of BvhNode
        if value is not None and not isinstance(value, BvhNode):
            raise ValueError("parent should either be None or a BvhNode class/subclasse object")
        self._parent = value

    def __str__(self) -> str:
        return f"{self.name}"

    def __repr__(self) -> str:
        return f"BvhNode(name = {self.name}, offset = {self.offset}, parent = {self.parent})"

    def is_end_site(self) -> bool:
        """Whether this node is an end site, a channel-less leaf.

        The node-kind check pybvh uses everywhere instead of name
        conventions. A bare ``BvhNode`` is not a node kind, so it raises
        ``NotImplementedError``; each subclass answers.

        Returns
        -------
        bool
        """
        raise NotImplementedError(
            "BvhNode is the abstract base class; build hierarchies from "
            "BvhRoot, BvhJoint, and BvhEndSite."
        )

    def is_root(self) -> bool:
        """Whether this node is the root of its hierarchy.

        ``False`` for every node kind but :class:`BvhRoot`.

        Returns
        -------
        bool
        """
        return False


# ---------------------------------------------------------------------------------------------


class BvhEndSite(BvhNode):
    """A BVH End Site — a channel-less leaf marking the tip of a bone chain.

    End sites carry only an offset (the bone-tip position relative to the parent joint); they have no channels, no children, and no motion data. End-site identity is carried by this class — check it via :meth:`is_end_site` or ``isinstance``. Generated display names like ``'EndSiteHips'`` are cosmetic only and carry no semantics.

    Attributes
    ----------
    name : str
        Display name of the end site.
    offset : np.ndarray
        3-element array of positional offset values.
    parent : BvhNode or None
        Parent joint in the hierarchy.
    """

    def __repr__(self) -> str:
        return f"BvhEndSite(name = {self.name}, offset = {self.offset}, parent = {self.parent})"

    def is_end_site(self) -> bool:
        """Always ``True``: an end site has an offset and nothing else.

        Returns
        -------
        bool
        """
        return True


# ---------------------------------------------------------------------------------------------


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

    def __init__(
        self,
        name: str,
        offset: list[float] | npt.NDArray[np.float64] | None = None,
        rot_channels: list[str] | str | None = None,
        children: list[BvhNode] | None = None,
        parent: BvhNode | None = None,
    ) -> None:
        # inheritance
        super().__init__(name, offset, parent)

        self._frozen = False
        self.rot_channels = rot_channels if rot_channels is not None else ["Z", "Y", "X"]  # type: ignore[assignment]
        self.children = children if children is not None else []

    @property
    def rot_channels(self) -> list[str]:
        """Euler rotation order, e.g. ``['Z', 'Y', 'X']``.

        The order of the joint's rotation channels as the file's
        ``CHANNELS`` line lists them, which is also the order of its
        three angles in ``Bvh.joint_angles``: intrinsic rotations,
        pre-multiplied, as :func:`~pybvh.rotations.euler_to_rotmat` reads
        them. Assigning accepts a three-letter string or a list of three
        one-letter strings, a permutation of ``'XYZ'`` either way, and
        stores a new list; anything else raises ``ValueError``.

        Frozen by the :class:`~pybvh.bvh.Bvh` constructor, which freezes
        every node it is given, because the stored angles are only
        meaningful in their own order: assigning then raises
        ``AttributeError``, and ``Bvh.change_euler_order`` converts the
        angles and the order together. A joint that reaches a ``Bvh`` only
        through ``Bvh.nodes`` assignment is not frozen.
        """
        return self._rot_channels

    @rot_channels.setter
    def rot_channels(self, value: list[str] | str) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError(
                "rot_channels is frozen. Use "
                "Bvh.change_euler_order(order, joint=joint_name) or "
                "Bvh.change_euler_order(order) to change rotation order."
            )
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
        """The joints and end sites directly below this one, in file order.

        Assigning anything but a list of :class:`BvhNode` raises
        ``ValueError``. Like ``parent``, it sets one direction of the
        link only: the children's ``parent`` is left as it was.
        """
        return self._children

    @children.setter
    def children(self, value: list[BvhNode]) -> None:
        if (not isinstance(value, list)) or any([not isinstance(x, BvhNode) for x in value]):
            raise ValueError("children should be a list of BvhNode class/subclasse objects")
        self._children = value

    def __str__(self) -> str:
        return f"JOINT {self.name}"

    def __repr__(self) -> str:
        children_list = []
        for child in self.children:
            if child.is_end_site():
                children_list.append(f"{child.__str__()}")
            else:
                children_list.append(f"BvhJoint({child.__str__()})")
        return f"BvhJoint(name = {self.name}, offset = {self.offset}, rot_channels = {self.rot_channels}, children = {str(children_list)}, parent = {self.parent})"

    def _check_channels(self, value: list[str] | str) -> list[str]:
        # A string of 3 characters or a list of 3 one-character strings,
        # a permutation of 'XYZ' either way. A list is checked element by
        # element, not joined: ['XY', 'Z'] joins to 'XYZ' but would write
        # a CHANNELS line of five tokens (XYrotation) the reader rejects.
        # The result is a new list, never the caller's own: channel lists
        # are frozen after Bvh construction and must not be mutable from
        # the outside.
        error = ValueError(
            "the channels should be a string of 3 characters or a list of "
            "3 one-character strings, one of each from 'X' 'Y' 'Z'"
        )
        if isinstance(value, str):
            axes = list(value)
        elif isinstance(value, list):
            if not all(isinstance(axis, str) and len(axis) == 1 for axis in value):
                raise error
            axes = list(value)
        else:
            raise error
        if sorted(axes) != ["X", "Y", "Z"]:
            raise error
        return axes

    def is_end_site(self) -> bool:
        """Always ``False``: a joint carries rotation channels.

        Returns
        -------
        bool
        """
        return False


def _is_joint(node: BvhNode) -> TypeGuard[BvhJoint]:
    """Whether ``node`` is not an end site, narrowed for the type checker to a joint.

    The test is :meth:`BvhNode.is_end_site`, the node-kind check pybvh
    uses everywhere, so a node class of the caller's own that answers
    ``False`` counts as a joint, as it does at run time, and is expected
    to carry ``rot_channels`` and ``children``. ``isinstance(node,
    BvhJoint)`` is the stricter alternative: it would skip such a node.
    """
    return not node.is_end_site()


# ---------------------------------------------------------------------------------------------


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

    def __init__(
        self,
        name: str = "root",
        offset: list[float] | npt.NDArray[np.float64] | None = None,
        pos_channels: list[str] | str | None = None,
        rot_channels: list[str] | str | None = None,
        children: list[BvhNode] | None = None,
        parent: BvhNode | None = None,
    ) -> None:
        # inheritance
        super().__init__(name, offset, rot_channels, children, parent)

        self.pos_channels = pos_channels if pos_channels is not None else ["X", "Y", "Z"]  # type: ignore[assignment]

    @property
    def pos_channels(self) -> list[str]:
        """Order of the root's three position channels.

        Accepts the same forms as ``rot_channels``. A
        :class:`~pybvh.bvh.Bvh` supports only ``['X', 'Y', 'Z']``, the
        order of every ``root_pos`` row, and raises ``ValueError`` at
        construction for any other. The ``Bvh`` constructor freezes it
        with ``rot_channels``: from then on, assigning raises
        ``AttributeError``.
        """
        return self._pos_channels

    @pos_channels.setter
    def pos_channels(self, value: list[str] | str) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError("pos_channels is frozen after construction and cannot be changed.")
        self._pos_channels = self._check_channels(value)

    def __str__(self) -> str:
        return f"ROOT {self.name}"

    def __repr__(self) -> str:
        super_str = super().__repr__()
        # the parent classe repr is f'BvhJoint(name = {self.name}, offset = {self.offset},
        #  rot_channels = {self.rot_channels}, children = {str(children_list)}, parent = {self.parent})'
        super_str_list = super_str.split(",")
        super_str_list[0] = f"BvhRoot(name = {self.name}"
        super_str_list.insert(2, f" pos_channels = {self.pos_channels}")
        return ",".join(super_str_list)
        # return f'BvhRoot(name = {self.name}, offset = {self.offset}, pos_channels = {self.pos_channels},
        #  rot_channels = {self.rot_channels}, children = {str(children_list)}, parent = {self.parent})'

    def is_root(self) -> bool:
        """Always ``True``.

        Returns
        -------
        bool
        """
        return True
