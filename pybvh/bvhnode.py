from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt


class BvhNode:
    """Base class for BVH hierarchy nodes.

    ``BvhNode`` itself carries only the data shared by every node kind (name, offset, parent). Concrete hierarchies are built from its subclasses: :class:`BvhRoot`, :class:`BvhJoint`, and :class:`BvhEndSite`. Node-kind checks go through :meth:`is_end_site` / :meth:`is_root` (never through name conventions), so a bare ``BvhNode`` cannot answer :meth:`is_end_site`.

    Attributes
    ----------
    name : str
        Name of the node.
    offset : np.ndarray
        3-element array of positional offset values.
    parent : BvhNode or None
        Parent node in the hierarchy, or None if this is a root.
    """


    def __init__(self, name: str, offset: list[float] | npt.NDArray[np.float64] | None = None, parent: BvhNode | None = None) -> None:
        self.name = name
        self.offset = offset if offset is not None else [0.0, 0.0, 0.0]  # type: ignore[assignment]
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
        try:
            offset_arr = np.array(value, dtype=np.float64)
        except (TypeError, ValueError) as e:
            raise ValueError("offset should be a list or numpy array of 3 numbers") from e
        if offset_arr.shape != (3,):
            raise ValueError(
                f"offset should be a list or numpy array of 3 numbers, "
                f"got shape {offset_arr.shape}")
        self._offset: npt.NDArray[np.float64] = offset_arr

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
        raise NotImplementedError(
            "BvhNode is the abstract base class; build hierarchies from "
            "BvhRoot, BvhJoint, and BvhEndSite.")

    def is_root(self) -> bool:
        return False


#---------------------------------------------------------------------------------------------

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
        return f'BvhEndSite(name = {self.name}, offset = {self.offset}, parent = {self.parent})'

    def is_end_site(self) -> bool:
        return True


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
    def __init__(self, name: str, offset: list[float] | npt.NDArray[np.float64] | None = None,
                 rot_channels: list[str] | str | None = None, children: list[BvhNode] | None = None,
                 parent: BvhNode | None = None) -> None:
        #inheritance
        super().__init__(name, offset, parent)

        self._frozen = False
        self.rot_channels = rot_channels if rot_channels is not None else ['Z', 'Y', 'X']  # type: ignore[assignment]
        self.children = children if children is not None else []

    @property
    def rot_channels(self) -> list[str]:
        return self._rot_channels
    @rot_channels.setter
    def rot_channels(self, value: list[str] | str) -> None:
        if getattr(self, '_frozen', False):
            raise AttributeError(
                "rot_channels is frozen. Use "
                "Bvh.change_euler_order(order, joint=joint_name) or "
                "Bvh.change_euler_order(order) to change rotation order.")
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
        # we return the result as a new list of 3 characters (never the
        # caller's own list — channel lists are frozen after Bvh
        # construction and must not be mutable from the outside)
        er = ValueError("the channels should be a list or a string of 3 elements, one of each from 'X' 'Y' 'Z'")
        if isinstance(value, str):
            if sorted(value) != ['X', 'Y', 'Z']:
                raise er
            return list(value)
        elif isinstance(value, list):
            try:
                str_conv = ''.join(value)
            except:
                raise er
            if sorted(str_conv) != ['X', 'Y', 'Z']:
                raise er
            return list(value)
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
    def __init__(self, name: str = 'root', offset: list[float] | npt.NDArray[np.float64] | None = None,
                 pos_channels: list[str] | str | None = None, rot_channels: list[str] | str | None = None,
                 children: list[BvhNode] | None = None, parent: BvhNode | None = None) -> None:
        #inheritance
        super().__init__(name, offset, rot_channels, children, parent)

        self.pos_channels = pos_channels if pos_channels is not None else ['X', 'Y', 'Z']  # type: ignore[assignment]

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


#---------------------------------------------------------------------------------------------
# The node tree as a whole
#---------------------------------------------------------------------------------------------

def _check_node_tree(nodes: Sequence[BvhNode]) -> None:
    """Raise unless ``nodes`` is one tree in depth-first order, wired both ways.

    The depth-first walk of ``children`` from ``nodes[0]`` must visit
    exactly ``nodes``, in order, by identity, and every node reached must
    have the node it was reached from as its ``parent``. This is what
    :func:`~pybvh.io.write_bvh_file` (which walks ``children``) and
    ``joint_angles`` (whose columns follow ``nodes``) both rely on, and it
    is checked once, when a :class:`~pybvh.Bvh` is built. O(N), no FK.
    """
    if len(nodes) == 0:
        raise ValueError("nodes must hold at least one node, the root.")
    total = len(nodes)
    position = {id(node): i for i, node in enumerate(nodes)}

    def describe(node: BvhNode | None) -> str:
        if node is None:
            return "None"
        if id(node) in position:
            return f"nodes[{position[id(node)]}] ({node.name!r})"
        return f"{node.name!r}, which is not in nodes"

    stack: list[tuple[BvhNode, BvhNode | None]] = [(nodes[0], None)]
    visited = 0
    while stack:
        reached, reached_from = stack.pop()
        if visited == total:
            raise ValueError(
                f"The depth-first walk of children from nodes[0] reaches "
                f"{describe(reached)}, a child of {describe(reached_from)}, "
                f"after all {total} nodes were visited: a node is listed "
                f"more than once in the children lists, or a child is not "
                f"in nodes.")
        expected = nodes[visited]
        if reached is not expected:
            raise ValueError(
                f"The depth-first walk of children from nodes[0] reaches "
                f"{describe(reached)}, a child of {describe(reached_from)}, "
                f"where {describe(expected)} is listed. nodes must list the "
                f"tree in depth-first order (each joint followed by its whole "
                f"subtree, as a .bvh file writes it), with every node listed "
                f"exactly once in its parent's children.")
        if reached_from is None:
            if reached.parent is not None:
                raise ValueError(
                    f"nodes[0] ({reached.name!r}) is the root, but its parent "
                    f"is {describe(reached.parent)}; the root's parent must "
                    f"be None.")
        elif reached.parent is not reached_from:
            raise ValueError(
                f"{describe(reached)} is in the children of "
                f"{describe(reached_from)}, but its parent is "
                f"{describe(reached.parent)}; parent and children must agree.")
        visited += 1
        if not reached.is_end_site():
            stack.extend(
                (child, reached) for child in reversed(reached.children))  # type: ignore[attr-defined]

    if visited != total:
        unreached = nodes[visited]
        raise ValueError(
            f"{describe(unreached)} is not reached by the depth-first walk "
            f"of children from nodes[0]; its parent is "
            f"{describe(unreached.parent)}. Every node must be listed in "
            f"its parent's children.")


#---------------------------------------------------------------------------------------------
# The node table: a skeleton as plain data, and the one builder of node trees
#---------------------------------------------------------------------------------------------

_TABLE_KEYS = ('name', 'parent', 'offset', 'rot_channels', 'pos_channels')


def nodes_to_table(nodes: Sequence[BvhNode]) -> list[dict[str, Any]]:
    """Export a node tree as a node table: one plain ``dict`` per node.

    The node table is the skeleton as plain data, the named twin of
    :class:`~pybvh.FkTopology`: ``nodes`` in the order they are given
    (for :attr:`Bvh.nodes`, the depth-first order of the file), each
    node an entry whose ``parent`` is the *index* of its parent's entry.
    Position is identity. Names play no part in it, because BVH names
    are not unique: two end sites under one joint share their generated
    display name, and duplicate joint names occur in real files, so a
    name-keyed export (the ``{name: ...}`` dict of
    :meth:`Bvh.to_hierarchy_dict`) loses nodes that this table keeps.

    Each entry has these keys, in this order:

    ``name``
        The node's name, a ``str``. For an end site, the display name
        (``'EndSiteHips'``), which is cosmetic.
    ``parent``
        The index of the parent's entry in this table, ``None`` for the
        root (entry 0) and nowhere else.
    ``offset``
        The rest offset from the parent, an ``ndarray`` of shape ``(3,)``
        and a copy, as ``node.offset`` is everywhere in pybvh; a caller
        writing JSON calls ``.tolist()``.
    ``rot_channels``
        The Euler order, e.g. ``['Z', 'Y', 'X']``, on the root and on
        joints only. **An entry without it is an end site**; there is no
        kind flag. Channel orders are never inferred, so the rule is
        unambiguous.
    ``pos_channels``
        The position channel order, on the root only: ``['X', 'Y', 'Z']``
        for every root a :class:`~pybvh.Bvh` accepts.

    There is no ``children`` key: children are derived from ``parent``,
    in table order, so the two cannot disagree. Entries are fresh objects,
    safe to mutate.

    Relation to :class:`~pybvh.FkTopology`: ``offsets`` is the stacked
    ``offset`` values, ``parent_idx`` is ``parent`` with ``None`` read as
    ``-1``, ``joint_idx`` counts up over the entries that have
    ``rot_channels`` and ``euler_orders`` is those ``rot_channels``
    joined. The table adds the names and splits the root's channels;
    ``FkTopology`` stays an FK input bundle with no names.

    Conventions, and the alternatives they were chosen over: the table is
    a ``list`` of ``dict``, not a DataFrame and not a class, because a
    plain list has one index and position is the identity here, whereas
    a DataFrame has a label index and a positional one that sorting,
    filtering or concatenating pull apart, which makes an integer parent
    reference ambiguous between the two. The root's ``parent`` is
    ``None`` rather than ``FkTopology``'s ``-1``, because ``-1`` is a
    valid Python index (the last entry): ``None`` cannot index anything
    by accident, and the conversion to ``parent_idx`` is one expression.

    This function exports what it is given and validates nothing: a
    node list that is not in depth-first order gives a table that is not
    either, which :func:`nodes_from_table` then rejects.

    Parameters
    ----------
    nodes : sequence of BvhNode
        A node tree, every parent listed before its children, such as
        :attr:`Bvh.nodes`.

    Returns
    -------
    list of dict
        One entry per node, in ``nodes`` order.

    Raises
    ------
    ValueError
        If a node's parent is not in ``nodes``.

    See Also
    --------
    nodes_from_table : The inverse, and the validation of the format.
    FkTopology.from_nodes : The same positions, as arrays for FK.

    Example
    -------
    The rig of one joint carrying two end sites, which no name-keyed
    export can hold:

    >>> nodes_to_table(bvh.nodes)
    [{'name': 'Hips', 'parent': None, 'offset': array([0., 0., 0.]),
      'pos_channels': ['X', 'Y', 'Z'], 'rot_channels': ['Z', 'Y', 'X']},
     {'name': 'Hand', 'parent': 0, 'offset': array([0., 1., 0.]),
      'rot_channels': ['Z', 'Y', 'X']},
     {'name': 'EndSiteHand', 'parent': 1, 'offset': array([1., 0., 0.])},
     {'name': 'EndSiteHand', 'parent': 1, 'offset': array([0., 0., 2.])}]
    """
    position = {id(node): i for i, node in enumerate(nodes)}
    table: list[dict[str, Any]] = []
    for i, node in enumerate(nodes):
        if node.parent is None:
            parent: int | None = None
        else:
            try:
                parent = position[id(node.parent)]
            except KeyError:
                raise ValueError(
                    f"Node {node.name!r} (index {i}) has a parent that is "
                    f"not in the node list.") from None
        entry: dict[str, Any] = {
            'name': node.name,
            'parent': parent,
            'offset': node.offset.copy(),
        }
        if node.is_root():
            entry['pos_channels'] = list(node.pos_channels)  # type: ignore[attr-defined]
        if not node.is_end_site():
            entry['rot_channels'] = list(node.rot_channels)  # type: ignore[attr-defined]
        table.append(entry)
    return table


def nodes_from_table(table: Sequence[Mapping[str, Any]]) -> list[BvhNode]:
    """Build a node tree from a node table, wired and checked.

    The inverse of :func:`nodes_to_table`, and the way to build a
    skeleton from plain data without wiring parents and children by
    hand. It returns fresh :class:`BvhRoot`, :class:`BvhJoint` and
    :class:`BvhEndSite` objects, one per entry in table order, with
    ``parent`` and ``children`` wired from the indices, so the two cannot
    disagree, and it checks the tree once, here (see *Raises*).

    The format is the one :func:`nodes_to_table` writes, one ``dict``
    per node with ``name``, ``parent`` (the index of the parent's entry,
    ``None`` on the root), ``offset`` (three numbers), ``rot_channels``
    (root and joints) and ``pos_channels`` (root only). An entry without
    ``rot_channels`` is an end site; there is no kind flag and no
    ``children`` key. Position is identity, so names need not be unique:
    two end sites under one joint, or two joints named alike, come back
    as they were. The relation to :class:`~pybvh.FkTopology` is the one
    :func:`nodes_to_table` describes: ``parent`` with ``None`` as ``-1``
    is ``parent_idx``, and the entries with ``rot_channels`` are the
    joint columns, in order.

    **The table must be in depth-first order**: entry 0 is the root and
    every joint is followed by its whole subtree, the order a ``.bvh``
    file writes. The order is required, not repaired: ``joint_angles``
    columns follow the node order, and :func:`~pybvh.io.write_bvh_file`
    writes the depth-first order of ``children``, so a breadth-first
    table silently reordered to depth-first would move the columns away
    from the joints they describe, and left as it is would write a file
    whose channel order disagrees with the motion it carries. A table
    out of order raises instead, naming the entry.

    Two leniencies, so a table can be written by hand: a root entry
    without ``pos_channels`` gets ``['X', 'Y', 'Z']``, and an end-site
    entry without ``name`` gets ``'EndSite' + parent name``, the
    parser's own rule. The root entry may also omit ``parent``, whose
    only value is ``None``. Nothing else is inferred.

    Parameters
    ----------
    table : sequence of dict
        One entry per node, in depth-first order.

    Returns
    -------
    list of BvhNode
        Fresh nodes in table order, ``parent`` and ``children`` wired.

    Raises
    ------
    ValueError
        Naming the entry's index and name, when:

        - entry 0 has a parent, or any other entry has ``parent`` set to
          ``None`` (exactly one root, at index 0);
        - ``parent`` is not an ``int`` in ``[0, i)`` (parents precede
          their children; no self or forward reference);
        - the parent entry is an end site (end sites are leaves);
        - the depth-first walk of the tree from entry 0 does not visit
          the entries as ``0, 1, ..., N-1``;
        - ``offset`` is missing or not three numbers, or a channel list
          is not a permutation of ``'XYZ'`` (``None`` included: the node
          constructors would default it, this function infers nothing);
        - ``pos_channels`` sits on an entry other than the root;
        - an entry has a key the format does not define (a misspelt
          ``rot_channels`` would otherwise silently turn a joint into an
          end site).

    See Also
    --------
    nodes_to_table : The inverse, and the format in full.
    FkTopology : The same positions as arrays, without names.

    Example
    -------
    >>> nodes = nodes_from_table([
    ...     {'name': 'Hips', 'parent': None, 'offset': [0, 0, 0],
    ...      'rot_channels': 'ZYX'},
    ...     {'name': 'Hand', 'parent': 0, 'offset': [0, 1, 0],
    ...      'rot_channels': 'ZYX'},
    ...     {'parent': 1, 'offset': [1, 0, 0]},
    ...     {'parent': 1, 'offset': [0, 0, 2]},
    ... ])
    >>> [node.name for node in nodes]
    ['Hips', 'Hand', 'EndSiteHand', 'EndSiteHand']
    """
    if len(table) == 0:
        raise ValueError("A node table needs at least one entry, the root.")
    nodes: list[BvhNode] = []
    for i, entry in enumerate(table):
        if not isinstance(entry, Mapping):
            raise ValueError(
                f"Node table entry {i} is a {type(entry).__name__}, not a "
                f"dict with the keys {', '.join(_TABLE_KEYS)}.")
        label = _entry_label(i, entry)
        unknown_keys = sorted(set(entry) - set(_TABLE_KEYS))
        if unknown_keys:
            raise ValueError(
                f"Node table {label} has key(s) {unknown_keys} the format "
                f"does not define; the keys are {', '.join(_TABLE_KEYS)}. "
                f"(An entry without rot_channels is an end site, so a "
                f"misspelt key would silently change the node's kind.)")
        parent = _table_parent(entry, i, nodes, label)
        node = _node_from_entry(entry, i, parent, label)
        if parent is not None:
            parent.children.append(node)
        nodes.append(node)
    _check_table_order(nodes)
    return nodes


def _entry_label(index: int, entry: Mapping[str, Any]) -> str:
    """``"entry 3 ('Hand')"``, or ``"entry 3"`` when the name is unusable."""
    name = entry.get('name')
    if isinstance(name, str):
        return f"entry {index} ({name!r})"
    return f"entry {index}"


def _table_parent(entry: Mapping[str, Any], index: int, built: list[BvhNode],
                  label: str) -> BvhJoint | None:
    """Resolve an entry's ``parent`` index to the node built for it."""
    parent_index = entry.get('parent')
    if index == 0:
        if parent_index is not None:
            raise ValueError(
                f"Node table {label} has parent {parent_index!r}; entry 0 is "
                f"the root and its parent must be None.")
        return None
    if parent_index is None:
        raise ValueError(
            f"Node table {label} has parent None; only entry 0, the root, "
            f"has no parent. A node table holds exactly one root.")
    if isinstance(parent_index, bool) or not isinstance(parent_index, (int, np.integer)):
        raise ValueError(
            f"Node table {label} has parent {parent_index!r}, which is not "
            f"an int: parent is the index of the parent's entry in the table.")
    if not 0 <= parent_index < index:
        raise ValueError(
            f"Node table {label} has parent {int(parent_index)}, which is "
            f"not an index in [0, {index}): parents precede their children "
            f"in a node table, and an entry cannot be its own parent.")
    parent = built[parent_index]
    if parent.is_end_site():
        raise ValueError(
            f"Node table {label} has parent entry {int(parent_index)} "
            f"({parent.name!r}), which is an end site (an entry without "
            f"rot_channels). End sites are leaves.")
    return parent  # type: ignore[return-value]


def _node_from_entry(entry: Mapping[str, Any], index: int,
                     parent: BvhJoint | None, label: str) -> BvhNode:
    """Build the node an entry describes; its kind follows from its keys."""
    is_end_site = 'rot_channels' not in entry
    if index == 0 and is_end_site:
        raise ValueError(
            f"Node table {label} has no rot_channels; entry 0 is the root, "
            f"which carries the skeleton's base rotation. (An entry without "
            f"rot_channels is an end site, and end sites are leaves.)")
    if 'pos_channels' in entry and index != 0:
        kind = ("an end site (no rot_channels), which has no channels at all"
                if is_end_site else "a joint")
        raise ValueError(
            f"Node table {label} has pos_channels but is {kind}; only the "
            f"root (entry 0) has position channels.")
    if entry.get('offset') is None:
        raise ValueError(
            f"Node table {label} has no offset; offset is the rest offset "
            f"from the parent, three numbers.")
    # The node constructors turn a None channel order into their default
    # (ZYX, XYZ); here nothing is inferred, so None is rejected.
    for key in ('rot_channels', 'pos_channels'):
        if key in entry and entry[key] is None:
            how_to_omit = ("omit the key instead: an entry without "
                           "rot_channels is an end site"
                           if key == 'rot_channels' else
                           "omit the key instead: the root then gets XYZ")
            raise ValueError(
                f"Node table {label} has {key} None; a channel order is a "
                f"permutation of 'XYZ', and nothing is inferred, so "
                f"{how_to_omit}.")

    name = entry.get('name')
    if name is None:
        if not is_end_site:
            raise ValueError(
                f"Node table {label} has no name; only an end-site entry "
                f"may omit it (it is then named 'EndSite' + parent name).")
        assert parent is not None  # entry 0 is never an end site
        name = 'EndSite' + parent.name

    # The node setters validate name, offset and channels; the entry's
    # index is what their messages lack.
    try:
        if index == 0:
            return BvhRoot(name, entry['offset'],
                           entry.get('pos_channels', ['X', 'Y', 'Z']),
                           entry['rot_channels'], [], None)
        if is_end_site:
            return BvhEndSite(name, entry['offset'], parent)
        return BvhJoint(name, entry['offset'], entry['rot_channels'], [], parent)
    except ValueError as e:
        raise ValueError(f"Node table {label}: {e}") from e


def _check_table_order(nodes: list[BvhNode]) -> None:
    """Raise unless the tree's depth-first walk visits ``nodes`` in order.

    Called on a tree whose ``children`` were wired from parent indices in
    table order, so every node is reached exactly once and only the
    order can be wrong.
    """
    position = {id(node): i for i, node in enumerate(nodes)}
    stack = [nodes[0]]
    visited = 0
    while stack:
        reached = stack.pop()
        expected = nodes[visited]
        if reached is not expected:
            raise ValueError(
                f"Node table entry {visited} ({expected.name!r}) is out of "
                f"depth-first order: the walk from entry 0 reaches entry "
                f"{position[id(reached)]} ({reached.name!r}) at position "
                f"{visited}. A node table lists the tree as a .bvh file "
                f"writes it, each joint followed by its whole subtree, "
                f"because joint_angles columns follow that order.")
        visited += 1
        if not reached.is_end_site():
            stack.extend(reversed(reached.children))  # type: ignore[attr-defined]
