"""Nodes of a BVH skeleton hierarchy.

A skeleton is a tree of :class:`BvhRoot` (exactly one, first), :class:`BvhJoint`
and :class:`BvhEndSite` nodes, all subclasses of :class:`BvhNode`. Each node
carries its name, its rest offset from its parent and links to its parent
and children; the motion itself lives in the :class:`~pybvh.bvh.Bvh` that
holds the nodes. Trees are usually built by the reader or by
:func:`~pybvh.nodes_from_table`, which wire both directions of every link.

A node is mutable until a :class:`~pybvh.bvh.Bvh` owns it: the ``Bvh``
constructor freezes every node it is given. From then on, assigning its
``name``, ``offset``, ``parent``, ``children`` or channels raises
``AttributeError`` naming the way to change a skeleton
(``table = bvh.to_node_table()``, edit the rows, then
``bvh.copy(nodes=table)``; see :meth:`~pybvh.bvh.Bvh.copy`), ``offset``
is a read-only array and ``children`` a tuple. A clip's derived values (forward
kinematics, indexes, floor height) are computed from its skeleton, so the
skeleton cannot change under them.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias, TypeGuard

import numpy as np
import numpy.typing as npt


def _frozen_error(node: BvhNode, attribute: str) -> AttributeError:
    """The refusal of an edit to a node a :class:`~pybvh.bvh.Bvh` owns."""
    return AttributeError(
        f"{attribute} of node {node.name!r} cannot be changed: a Bvh owns "
        f"this node, and the skeleton of a Bvh is frozen. To change it, edit "
        f"the skeleton as a table and copy the clip onto it: "
        f"table = bvh.to_node_table(), edit the rows, then "
        f"bvh.copy(nodes=table)."
    )


def _is_empty_order(value: object) -> bool:
    """Whether a channel order is given as empty (``[]``, ``()``, ``''``)."""
    return isinstance(value, Sequence) and len(value) == 0


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
        self._frozen = False
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
        name. Assigning anything but a ``str`` raises ``ValueError``,
        and assigning on a node a ``Bvh`` owns raises ``AttributeError``.
        """
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        if self._frozen:
            raise _frozen_error(self, "name")
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

        Once a ``Bvh`` owns the node, assigning raises ``AttributeError``
        and the array returned is read-only: writing into it raises
        ``ValueError``. ``offset.copy()`` is a writable copy.
        """
        if self._frozen:
            # A view of the read-only array: unlike the array itself, which
            # owns its data, a view cannot be made writeable again.
            return self._offset.view()
        return self._offset

    @offset.setter
    def offset(self, value: list[float] | npt.NDArray[np.float64]) -> None:
        if self._frozen:
            raise _frozen_error(self, "offset")
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
        own. Assigning on a node a ``Bvh`` owns raises ``AttributeError``.
        """
        return self._parent

    @parent.setter
    def parent(self, value: BvhNode | None) -> None:
        if self._frozen:
            raise _frozen_error(self, "parent")
        # parent needs to be either None or an instance of BvhNode
        if value is not None and not isinstance(value, BvhNode):
            raise ValueError("parent should either be None or a BvhNode class/subclasse object")
        self._parent = value

    def _freeze(self) -> None:
        """Refuse every later edit: a :class:`~pybvh.bvh.Bvh` calls this on
        each node it takes. Nothing unfreezes a node."""
        self._frozen = True
        self._offset.flags.writeable = False

    def __setstate__(self, state: dict[str, object]) -> None:
        # copy.deepcopy and pickle rebuild a node from its __dict__, and the
        # offset array they rebuild is writable: a frozen node's must not be.
        self.__dict__.update(state)
        if self._frozen:
            self._offset.flags.writeable = False

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
    """A BVH joint node with rotation channels, position channels if it translates, and children.

    Attributes
    ----------
    name : str
        Name of the joint.
    offset : np.ndarray
        3-element array of positional offset values.
    pos_channels : tuple of str
        ``()`` on a joint that only rotates (the default), ``('X', 'Y',
        'Z')`` on a translating joint.
    rot_channels : tuple of str
        Rotation channel order as a permutation of ``('X', 'Y', 'Z')``.
    children : list or tuple of BvhNode
        Child nodes in the hierarchy; a tuple once a ``Bvh`` owns the node.
    parent : BvhNode or None
        Parent node, or None if this is a root.
    """

    def __init__(
        self,
        name: str,
        offset: list[float] | npt.NDArray[np.float64] | None = None,
        rot_channels: str | Sequence[str] | None = None,
        children: list[BvhNode] | None = None,
        parent: BvhNode | None = None,
        pos_channels: str | Sequence[str] | None = None,
    ) -> None:
        # inheritance
        super().__init__(name, offset, parent)

        self.pos_channels = pos_channels if pos_channels is not None else ()
        self.rot_channels = rot_channels if rot_channels is not None else ("Z", "Y", "X")
        self.children = children if children is not None else []

    @property
    def pos_channels(self) -> tuple[str, ...]:
        """The node's position channels: ``()`` or ``('X', 'Y', 'Z')``.

        On a :class:`BvhJoint`, empty by default: a joint that only
        rotates, whose position relative to its parent is its ``offset``
        in every frame. ``('X', 'Y', 'Z')`` on a joint whose ``CHANNELS``
        line declares three position channels besides its rotation
        channels, a translating joint. On a :class:`BvhRoot`, ``('X',
        'Y', 'Z')`` by default, the order of every ``root_pos`` row, and
        ``()`` on a static root, together with an empty ``rot_channels``.
        Position channels, where present, are stored in ``X Y Z`` order
        whatever order the file lists them in.
        Assigning accepts the forms of ``rot_channels`` and an empty
        sequence; any other order raises ``ValueError`` naming the
        expected ``'XYZ'``. Frozen by the :class:`~pybvh.bvh.Bvh`
        constructor with the rest of the node: assigning then raises
        ``AttributeError``.
        """
        return self._pos_channels

    @pos_channels.setter
    def pos_channels(self, value: str | Sequence[str]) -> None:
        if self._frozen:
            raise _frozen_error(self, "pos_channels")
        if _is_empty_order(value):
            self._pos_channels: tuple[str, ...] = ()
            return
        axes = self._check_channels(value)
        if axes != ("X", "Y", "Z"):
            raise ValueError(self._position_order_error("".join(axes)))
        self._pos_channels = axes

    def _position_order_error(self, order: str) -> str:
        return (
            f"Non-standard position channel order {order!r} on joint "
            f"{self.name!r} is not supported. Expected 'XYZ': position "
            f"channels are stored in X Y Z order."
        )

    @property
    def rot_channels(self) -> tuple[str, ...]:
        """Euler rotation order, a tuple such as ``('Z', 'Y', 'X')``.

        The order of the joint's rotation channels as the file's
        ``CHANNELS`` line lists them, which is also the order of its
        three angles in ``Bvh.joint_angles``: intrinsic rotations,
        pre-multiplied, as :func:`~pybvh.rotations.euler_to_rotmat` reads
        them. Assigning accepts a three-letter string or a sequence (a
        list, a tuple) of three one-letter strings, a permutation of
        ``'XYZ'`` either way, and stores it as a tuple, so the order never
        changes in place; anything else raises ``ValueError``. On a
        :class:`BvhRoot` only, an empty sequence is accepted too: a
        static root (``CHANNELS 0``) declares no rotation channels. Compare it
        with a tuple or as a string (``"".join(joint.rot_channels) ==
        "ZYX"``): a tuple never equals a list.

        Frozen by the :class:`~pybvh.bvh.Bvh` constructor with the rest of
        the node: assigning then raises ``AttributeError``, and
        ``Bvh.change_euler_order`` converts the angles and the order
        together, since the stored angles are only meaningful in their own
        order.
        """
        return self._rot_channels

    @rot_channels.setter
    def rot_channels(self, value: str | Sequence[str]) -> None:
        if self._frozen:
            raise AttributeError(
                "rot_channels is frozen. Use "
                "Bvh.change_euler_order(order, joint=node), with this node, or "
                "Bvh.change_euler_order(order) to change rotation order."
            )
        self._rot_channels = self._check_rotation_channels(value)

    @property
    def children(self) -> list[BvhNode] | tuple[BvhNode, ...]:
        """The joints and end sites directly below this one, in file order.

        A list until a ``Bvh`` owns the node, so a tree can be built by
        appending to it; a tuple from then on. Assigning anything but a
        list of :class:`BvhNode` raises ``ValueError``, and assigning on a
        node a ``Bvh`` owns raises ``AttributeError``. Like ``parent``, it
        sets one direction of the link only: the children's ``parent`` is
        left as it was.
        """
        return self._children

    @children.setter
    def children(self, value: list[BvhNode]) -> None:
        if self._frozen:
            raise _frozen_error(self, "children")
        if (not isinstance(value, list)) or any([not isinstance(x, BvhNode) for x in value]):
            raise ValueError("children should be a list of BvhNode class/subclasse objects")
        self._children: list[BvhNode] | tuple[BvhNode, ...] = value

    def _freeze(self) -> None:
        super()._freeze()
        self._children = tuple(self._children)

    def __str__(self) -> str:
        return f"JOINT {self.name}"

    def __repr__(self) -> str:
        children_list = []
        for child in self.children:
            if child.is_end_site():
                children_list.append(f"{child.__str__()}")
            else:
                children_list.append(f"BvhJoint({child.__str__()})")
        # A joint that only rotates keeps the repr it had before #186. The
        # root shows its position channels even when it has none, so that
        # a static root reads as one.
        pos = f"pos_channels = {self.pos_channels}, " if self.pos_channels or self.is_root() else ""
        return f"{type(self).__name__}(name = {self.name}, offset = {self.offset}, {pos}rot_channels = {self.rot_channels}, children = {str(children_list)}, parent = {self.parent})"

    def _check_channels(self, value: str | Sequence[str]) -> tuple[str, ...]:
        # A string of 3 characters or a sequence (a list, a tuple) of 3
        # one-character strings, a permutation of 'XYZ' either way. A
        # sequence is checked element by element, not joined: ['XY', 'Z']
        # joins to 'XYZ' but would write a CHANNELS line of five tokens
        # (XYrotation) the reader rejects. The result is a tuple, so a
        # channel order never changes in place: on a node a Bvh owns, that
        # would reinterpret the stored angles under another order, and on
        # one not yet owned it would bypass this check.
        error = ValueError(
            "the channels should be a string of 3 characters or a sequence "
            "of 3 one-character strings, one of each from 'X' 'Y' 'Z'"
        )
        if isinstance(value, str):
            axes = tuple(value)
        elif isinstance(value, Sequence):
            if not all(isinstance(axis, str) and len(axis) == 1 for axis in value):
                raise error
            axes = tuple(value)
        else:
            raise error
        if sorted(axes) != ["X", "Y", "Z"]:
            raise error
        return axes

    def _check_rotation_channels(self, value: str | Sequence[str]) -> tuple[str, ...]:
        # A joint other than the root always declares an Euler order.
        return self._check_channels(value)

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


def _is_translating_joint(node: BvhNode) -> bool:
    """Whether ``node`` is a joint other than the root that declares position channels.

    Such a joint's position relative to its parent is a column of
    ``Bvh.joint_translations`` in each frame, in place of its ``offset``.
    The root's position channels are ``root_pos``, not a translation.
    """
    return _is_joint(node) and not node.is_root() and bool(_pos_channels(node))


def _pos_channels(node: BvhNode) -> tuple[str, ...]:
    """``node``'s position channels, ``()`` on a node without any.

    A node class of the caller's own that answers ``False`` to
    :meth:`BvhNode.is_end_site` counts as a joint (see :func:`_is_joint`)
    and may carry no ``pos_channels``: it only rotates.
    """
    return tuple(getattr(node, "pos_channels", ()))


def _euler_order(joint: BvhJoint) -> str:
    """``joint``'s Euler order as a string, ``'XYZ'`` on a static root.

    A static root declares no rotation channel, and its row of
    ``Bvh.joint_angles`` is zero in every frame: every order gives it the
    identity. It reads ``'XYZ'`` so that every column of ``joint_angles``
    has an order, which the rotation conversions and forward kinematics
    require.
    """
    return "".join(joint.rot_channels) or "XYZ"


# ---------------------------------------------------------------------------------------------


class BvhRoot(BvhJoint):
    """A BVH root joint with both position and rotation channels, or neither.

    A root declares 3 position + 3 rotation channels, or no channel at
    all (``CHANNELS 0``, a static root, see :meth:`is_static`). One order
    empty and the other not is refused when the root is built and when a
    :class:`~pybvh.bvh.Bvh` is given it; the setters allow it, so that a
    root can change state by two assignments.

    Attributes
    ----------
    name : str
        Name of the root joint.
    offset : np.ndarray
        3-element array of positional offset values.
    pos_channels : tuple of str
        ``('X', 'Y', 'Z')`` (the default), or ``()`` on a static root.
    rot_channels : tuple of str
        Rotation channel order as a permutation of ``('X', 'Y', 'Z')``,
        or ``()`` on a static root.
    children : list or tuple of BvhNode
        Child nodes in the hierarchy; a tuple once a ``Bvh`` owns the node.
    parent : BvhNode or None
        Parent node, or None.
    """

    def __init__(
        self,
        name: str = "root",
        offset: list[float] | npt.NDArray[np.float64] | None = None,
        pos_channels: str | Sequence[str] | None = None,
        rot_channels: str | Sequence[str] | None = None,
        children: list[BvhNode] | None = None,
        parent: BvhNode | None = None,
    ) -> None:
        # inheritance
        super().__init__(
            name,
            offset,
            rot_channels,
            children,
            parent,
            pos_channels if pos_channels is not None else ("X", "Y", "Z"),
        )
        self._check_channel_set()

    def is_static(self) -> bool:
        """Whether the root declares no channels (``CHANNELS 0``).

        A static root has empty ``pos_channels`` and ``rot_channels``: it
        stays at its ``offset`` with no rotation in every frame, a
        reference node above the skeleton, as in the files Blender's
        exporter writes for an armature with several root bones. A root
        has both channel orders or neither.

        Returns
        -------
        bool
        """
        return not self.pos_channels and not self.rot_channels

    def _check_channel_set(self) -> None:
        """Raise unless the root has both channel orders or neither.

        Checked when the root is built and when a
        :class:`~pybvh.bvh.Bvh` is given it, not by the setters: a root
        changes from one state to the other by two assignments, and the
        state between them has one order empty.
        """
        if bool(self.pos_channels) != bool(self.rot_channels):
            raise ValueError(
                f"Root {self.name!r} has pos_channels "
                f"{''.join(self.pos_channels)!r} and rot_channels "
                f"{''.join(self.rot_channels)!r}: a root declares both "
                f"(3 position + 3 rotation channels) or neither "
                f"(CHANNELS 0, a static root); only one of them empty is "
                f"not a BVH declaration."
            )

    def _position_order_error(self, order: str) -> str:
        return (
            f"Non-standard root position channel order {order!r} is not "
            f"supported. Expected 'XYZ', the order of every root_pos row "
            f"(root {self.name!r})."
        )

    def _check_rotation_channels(self, value: str | Sequence[str]) -> tuple[str, ...]:
        # A root may declare no rotation channels (a static root).
        if _is_empty_order(value):
            return ()
        return self._check_channels(value)

    def __str__(self) -> str:
        return f"ROOT {self.name}"

    def is_root(self) -> bool:
        """Always ``True``.

        Returns
        -------
        bool
        """
        return True


# ---------------------------------------------------------------------------------------------


NodeSelector: TypeAlias = str | BvhNode
"""A node of a clip, selected by its name or given as the node itself.

Every parameter of :class:`~pybvh.bvh.Bvh`, :mod:`pybvh.analysis` and
:mod:`pybvh.features` that selects one node takes a ``NodeSelector``, and one
that selects a set of nodes takes a sequence of them.
:meth:`~pybvh.bvh.Bvh.index` resolves it, the same way for every entry point:

- a :class:`BvhNode` is found by identity among the clip's ``nodes``, so a
  node of another clip, or one taken before the clip's skeleton was
  replaced, is refused;
- a name must match exactly one node of the index space the entry point
  works in. A name that two nodes share is refused, and the node itself
  selects either of them.

An integer is not a ``NodeSelector``: it does not say whether it counts in
joint space or in node space. ``bvh.nodes[i]`` is node ``i``. Integers stay in
the array-level functions (:func:`pybvh.transforms.mirror_angles`,
:mod:`pybvh.geometry`), where the array fixes the space. The name maps, an L/R
mapping, the ``name_mapping`` of :meth:`~pybvh.bvh.Bvh.retarget` and the
``masses`` of :func:`~pybvh.analysis.kinetic_energy` given as a mapping, key
values by names: they are not selections.
"""
