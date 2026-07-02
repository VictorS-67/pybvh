"""BVH file I/O — reading and writing ``.bvh`` motion capture files.

Public functions:

- :func:`read_bvh_file` — parse a ``.bvh`` file into a :class:`~pybvh.bvh.Bvh`
- :func:`write_bvh_file` — write a :class:`~pybvh.bvh.Bvh` to a ``.bvh`` file
"""
from __future__ import annotations

from pathlib import Path
from typing import TextIO

import numpy as np
import numpy.typing as npt

from .bvhnode import BvhNode, BvhJoint, BvhRoot, BvhEndSite
from .bvh import Bvh
from .tools import test_file


# ----------------------------------------------------------------
#  Reading
# ----------------------------------------------------------------

def read_bvh_file(
    filepath: str | Path,
    world_up: str = "auto",
    warn_on_world_up_disagreement: bool = True,
    lr_mapping: dict[str, str] | None = None,
) -> Bvh:
    """Parse a BVH motion capture file and return a Bvh object.

    Parameters
    ----------
    filepath : str or Path
        Path to the BVH file.
    world_up : str, optional
        World vertical axis.  ``"auto"`` (default) auto-detects from
        animation data.  Pass a signed axis string like ``"+y"`` to skip
        auto-detection and suppress the disagreement warning.
    warn_on_world_up_disagreement : bool, optional
        If True (default) and ``world_up="auto"``, emit a ``UserWarning``
        when rest-pose and first-frame inferences disagree.
    lr_mapping : dict or None, optional
        Explicit left/right joint pair mapping
        (``{"arm.L": "arm.R", ...}``). If provided, skips the name-based
        auto-detection for this file. Use for skeletons whose naming
        conventions the heuristic can't parse.

    Returns
    -------
    bvh : Bvh
        A Bvh object containing the skeleton hierarchy, root positions,
        joint angles, and frame time.

    Notes
    -----
    BVH files store joint angles in degrees; pybvh holds them in radians
    on :attr:`Bvh.joint_angles`. This function converts on read;
    :func:`write_bvh_file` converts back on write. Round-trip is lossless
    within float precision.
    """
    node_list, frame_array, frame_time = _extract_bvh_file_info(filepath)
    num_joints = len([n for n in node_list if not n.is_end_site()])
    root_pos = frame_array[:, :3].astype(np.float64)
    # BVH stores angles in degrees; pybvh holds them in radians.
    joint_angles_deg = frame_array[:, 3:].reshape(frame_array.shape[0], num_joints, 3).astype(np.float64)
    joint_angles = np.deg2rad(joint_angles_deg)
    return Bvh(nodes=node_list, root_pos=root_pos, joint_angles=joint_angles,
               frame_time=frame_time, world_up=world_up,
               lr_mapping=lr_mapping, source_path=str(filepath),
               warn_on_disagreement=warn_on_world_up_disagreement)


def _snap_frame_time(frame_time: float) -> float:
    """Snap a frame time to an exact ``1/N`` rate when it looks truncated.

    Foreign BVH files commonly store ``Frame Time`` truncated to 6 digits (e.g. ``0.033333`` for 30 fps), which makes resample-and-back round-trips drift. When the nearest exact ``1/N`` is within 0.01% of the literal value, return ``1/N``; otherwise return the input unchanged — non-integer rates like 23.976 fps are NOT snapped, since ``1/24 != 1/23.976``. This is a read-side salvage for truncated files; pybvh itself writes frame times at full precision.
    """
    if frame_time <= 0:
        return frame_time
    rate = round(1.0 / frame_time)
    if rate == 0:
        return frame_time
    snapped = 1.0 / rate
    if abs(snapped - frame_time) / frame_time < 1e-4:
        return snapped
    return frame_time


def _extract_bvh_file_info(filepath: str | Path) -> tuple[list[BvhNode], npt.NDArray[np.float64], float]:
    """Extract node hierarchy, frame data, and frame time from a BVH file.

    The returned frame array is normalized to pybvh's internal column layout: the root's 3 position columns first, then all rotation columns (root's, then each joint's in hierarchy order). Files whose root declares rotation channels before position channels are reordered on read; :func:`write_bvh_file` always writes position-first.
    """
    node_list: list[BvhNode] = []
    # Brace stack of currently-open node blocks: a node is pushed when its
    # block opens and popped on the matching '}', so the parent of a new
    # node is always the innermost open node (stack top).
    open_nodes: list[BvhNode] = []
    # line number if we need to report a problem in the file
    line_number: int = 0
    frame_count: int = 0
    frame_time: float = 0.0
    # Column indices of the root's position/rotation channels within its
    # own channel block, in file order (used to normalize rotation-first
    # roots to the internal position-first layout).
    root_pos_cols: list[int] = []
    root_rot_cols: list[int] = []

    filepath = test_file(filepath)

    with open(filepath, "r") as f:
        #---------- first, read the hierarchy (first part of the file)
        for raw_line in f:
            line_number += 1
            line = raw_line.split()
            if not line:
                continue
            token = line[0]

            if token in ('ROOT', 'JOINT'):
                name = line[1]
                if token == 'ROOT' and node_list:
                    raise ValueError(
                        f"Second ROOT '{name}' at line {line_number} in file "
                        f"{filepath}: pybvh models single-root skeletons only")
                if token == 'JOINT' and not open_nodes:
                    raise ValueError(
                        f"JOINT '{name}' outside any ROOT block at line "
                        f"{line_number} in file {filepath}")
                node_type = 'root' if token == 'ROOT' else 'joint'
                try:
                    offset, channels, line_number = _read_node_block(node_type, f, line_number)
                except Exception as e:
                    raise ValueError(
                        f"Could not read the offset or channels of the {node_type} "
                        f"{name}, at line {line_number} in file {filepath}: {e}") from e

                rot_channels = [ax for kind, ax in channels if kind == 'rot']
                if token == 'ROOT':
                    pos_channels = [ax for kind, ax in channels if kind == 'pos']
                    root_pos_cols = [i for i, (kind, _) in enumerate(channels) if kind == 'pos']
                    root_rot_cols = [i for i, (kind, _) in enumerate(channels) if kind == 'rot']
                    node: BvhNode = BvhRoot(name, offset, pos_channels, rot_channels, [], None)
                else:
                    parent_node = open_nodes[-1]
                    node = BvhJoint(name, offset, rot_channels, [], parent_node)
                    parent_node.children = parent_node.children + [node]  # type: ignore[attr-defined]
                node_list.append(node)
                open_nodes.append(node)

            elif token == 'End':
                if not open_nodes:
                    raise ValueError(
                        f"End Site outside any ROOT block at line {line_number} "
                        f"in file {filepath}")
                parent_node = open_nodes[-1]
                try:
                    offset, channels, line_number = _read_node_block('end_site', f, line_number)
                except Exception as e:
                    raise ValueError(
                        f"Could not read the offset of the End Site "
                        f"at line {line_number} in file {filepath}: {e}") from e
                # The generated name is display-only; end-site identity is
                # carried by the BvhEndSite class.
                node = BvhEndSite('EndSite' + parent_node.name, offset, parent_node)
                parent_node.children = parent_node.children + [node]  # type: ignore[attr-defined]
                node_list.append(node)
                # End-site blocks are fully consumed by _read_node_block
                # (including their closing '}'), so they never go on the stack.

            elif token == '}':
                if not open_nodes:
                    raise ValueError(
                        f"Unmatched '}}' at line {line_number} in file {filepath}")
                open_nodes.pop()

            elif token == 'Frames:':
                frame_count = int(line[1])

            elif token == 'Frame' and len(line) > 2 and line[1] == 'Time:':
                # Snap 6-digit-truncated exact 1/N rates (see _snap_frame_time).
                frame_time = _snap_frame_time(float(line[2]))
                # --- we close the loop related to reading the hierarchy ---
                break
            # Other tokens ('HIERARCHY', 'MOTION') carry no data — skipped.

        #small test to see if we reach the end of the hierarchy with no trouble.
        if not node_list:
            raise ValueError(f"No ROOT declaration found in {filepath}")
        if frame_count == 0 or frame_time == 0.0:
            raise ValueError(
                f"Frame count ({frame_count}) or frame time ({frame_time}) "
                f"is missing or zero in {filepath}")

        #----------  End of the Hierarchy part. After the hierarchy comes the frames data.

        # Expected channels: 6 for root (3 pos + 3 rot), 3 for each other non-end-site joint
        non_end_site_nodes = [n for n in node_list if not n.is_end_site()]
        num_channels = 3 + 3 * len(non_end_site_nodes)

        frame_array = np.loadtxt(f, ndmin=2)
        if frame_array.shape[0] != frame_count:
            raise ValueError(
                f"BVH declares {frame_count} frames but file contains "
                f"{frame_array.shape[0]} data lines")
        if frame_array.shape[1] != num_channels:
            raise ValueError(
                f"BVH motion lines have {frame_array.shape[1]} values per "
                f"frame but the hierarchy declares {num_channels} channels")

    # Normalize rotation-first (or interleaved) root channel layouts to the
    # internal position-first layout; within-block channel order is preserved.
    canonical_root_cols = root_pos_cols + root_rot_cols
    if canonical_root_cols != [0, 1, 2, 3, 4, 5]:
        frame_array = frame_array[:, canonical_root_cols + list(range(6, num_channels))]

    return (node_list, frame_array, frame_time)


def _parse_channels_line(parts: list[str]) -> list[tuple[str, str]]:
    """Parse a ``CHANNELS`` line into ``(kind, axis)`` tuples in file order.

    ``kind`` is ``'pos'`` or ``'rot'``, classified by token suffix (``endswith('position')`` / ``endswith('rotation')``, case-insensitive); ``axis`` is ``'X'``/``'Y'``/``'Z'``.
    """
    try:
        declared_count = int(parts[1])
    except (IndexError, ValueError) as e:
        raise ValueError(f"malformed CHANNELS line: {' '.join(parts)!r}") from e
    tokens = parts[2:]
    if len(tokens) != declared_count:
        raise ValueError(
            f"CHANNELS declares {declared_count} channels but lists {len(tokens)}")
    channels: list[tuple[str, str]] = []
    for tok in tokens:
        tok_lower = tok.lower()
        if tok_lower.endswith('position'):
            kind = 'pos'
        elif tok_lower.endswith('rotation'):
            kind = 'rot'
        else:
            raise ValueError(
                f"unrecognized channel token {tok!r} "
                f"(expected e.g. 'Xposition' or 'Yrotation')")
        axis = tok[0].upper()
        if axis not in ('X', 'Y', 'Z'):
            raise ValueError(f"unrecognized axis in channel token {tok!r}")
        channels.append((kind, axis))
    return channels


def _read_node_block(node_type: str, f: TextIO, line_number: int) -> tuple[list[float], list[tuple[str, str]], int]:
    """Read the lines opening a node's block: ``{``, ``OFFSET``, and ``CHANNELS``.

    Token-driven — each line is dispatched on its leading token, and completeness is validated per node type at the end, so blank lines and line-order variations don't break parsing. Joint/root blocks are left open (their children follow); end-site blocks are consumed through their closing ``}``.

    Returns ``(offset, channels, line_number)`` where ``channels`` is the ``(kind, axis)`` list from :func:`_parse_channels_line` (empty for end sites).
    """
    if node_type not in ('root', 'joint', 'end_site'):
        raise ValueError('node_type should be either root, joint or end_site')

    offset: list[float] | None = None
    channels: list[tuple[str, str]] | None = None

    for raw_ln in f:
        line_number += 1
        parts = raw_ln.split()
        if not parts:
            continue
        token = parts[0]
        if token == '{':
            continue
        elif token == '}':
            if node_type != 'end_site':
                raise ValueError(
                    f"{node_type} block closed before OFFSET/CHANNELS were read")
            break
        elif token == 'OFFSET':
            offset = [float(x) for x in parts[1:]]
        elif token == 'CHANNELS':
            channels = _parse_channels_line(parts)
        else:
            raise ValueError(f"unexpected token {token!r} in {node_type} block")
        if node_type != 'end_site' and offset is not None and channels is not None:
            break

    # ---- per-node-type validation ----
    if offset is None:
        raise ValueError(f"{node_type.replace('_', ' ')} missing OFFSET line")
    if len(offset) != 3:
        raise ValueError(f"OFFSET must have 3 values, got {len(offset)}")

    if node_type == 'end_site':
        if channels is not None:
            raise ValueError("end site must not declare CHANNELS")
        return (offset, [], line_number)

    if channels is None:
        raise ValueError(f"{node_type} missing CHANNELS line")
    num_pos = sum(1 for kind, _ in channels if kind == 'pos')
    num_rot = len(channels) - num_pos
    if node_type == 'root':
        if num_pos != 3 or num_rot != 3:
            raise ValueError(
                f"root must have 3 position + 3 rotation channels, "
                f"got {num_pos} position + {num_rot} rotation")
    else:
        if num_pos != 0:
            raise ValueError(
                f"pybvh does not model position channels on non-root joints "
                f"(got {num_pos} position channels)")
        if num_rot != 3:
            raise ValueError(
                f"joint must have exactly 3 rotation channels, got {num_rot}")

    return (offset, channels, line_number)


# ----------------------------------------------------------------
#  Writing
# ----------------------------------------------------------------

def write_bvh_file(bvh: Bvh, filepath: str | Path, verbose: bool = False) -> None:
    """Write a Bvh object to a ``.bvh`` file.

    Parameters
    ----------
    bvh : Bvh
        The motion data to write.
    filepath : str or Path
        Destination file path.  Must have a ``.bvh`` extension.
    verbose : bool, optional
        If True, print a one-line confirmation to stdout on success.
        Default ``False`` — preprocessing loops that write many files
        shouldn't flood the terminal by default.

    Raises
    ------
    ValueError
        If the file extension is not ``.bvh``.
    FileNotFoundError
        If the parent directory does not exist.

    Notes
    -----
    pybvh stores joint angles in radians, but the BVH format requires
    degrees; this function converts on write.
    """
    filepath = Path(filepath)
    if filepath.suffix != '.bvh':
        raise ValueError(f"{filepath.name} is not a .bvh file")
    elif not filepath.parent.exists():
        raise FileNotFoundError(f"directory does not exist: {filepath.parent}")

    def offset_to_str(node: BvhNode) -> str:
        offset_str = 'OFFSET'
        for num in node.offset:
            offset_str += ' ' + f'{num:.6f}'
        return offset_str

    def channels_to_str(node: BvhNode) -> str:
        chanels_str = 'CHANNELS'
        if node.parent is None:
            chanels_str += ' 6'
            for pos_ax in node.pos_channels:  # type: ignore[attr-defined]
                chanels_str += ' ' + pos_ax + 'position'
        else:
            chanels_str += ' 3'

        for rot_ax in node.rot_channels:  # type: ignore[attr-defined]
            chanels_str += ' ' + rot_ax + 'rotation'

        return chanels_str

    def rec_node_to_file(node: BvhNode, file: TextIO, depth: int = 0) -> None:
        if node.is_end_site():
            print('\t'*depth + 'End Site', file=file)
            print('\t'*depth + '{', file=file)
            print('\t'*(depth+1) + offset_to_str(node), file=file)
            print('\t'*depth + '}', file=file)
        else:
            if node.parent is None:
                type_str = 'ROOT'
            else:
                type_str = 'JOINT'
            print('\t'*depth + type_str + ' ' + node.name, file=file)
            print('\t'*depth +'{', file=file)
            print('\t'*(depth+1) + offset_to_str(node), file=file)
            print('\t'*(depth+1) + channels_to_str(node), file=file)
            for child in node.children:  # type: ignore[attr-defined]
                rec_node_to_file(child, file=file, depth=depth+1)
            print('\t'*depth +'}', file=file)

    with open(filepath, "w") as f:
        f.write('HIERARCHY\n')

        rec_node_to_file(bvh.root, file=f)

        f.write('MOTION\n')
        f.write(f'Frames: {bvh.frame_count}\n')
        # Full precision (10 significant digits) so non-integer rates like
        # 23.976 fps survive round-trips; the old '%.6f' truncation lost them.
        f.write(f'Frame Time: {bvh.frame_time:.10g}\n')

        F = bvh.frame_count
        if F > 0:
            # pybvh stores angles in radians; BVH format requires degrees.
            joint_angles_deg = np.rad2deg(bvh.joint_angles)
            motion = np.column_stack([bvh.root_pos,
                                      joint_angles_deg.reshape(F, -1)])
            np.savetxt(f, motion, fmt='%.6f', delimiter=' ')

    if verbose:
        print(f'Successfully saved the file {filepath.name} at the location\n{filepath.parent.absolute()}')
