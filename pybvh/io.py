"""BVH file I/O — reading and writing ``.bvh`` motion capture files.

Public functions:

- :func:`read_bvh_file` — parse a ``.bvh`` file into a :class:`~pybvh.bvh.Bvh`
- :func:`write_bvh_file` — write a :class:`~pybvh.bvh.Bvh` to a ``.bvh`` file
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import TextIO

import numpy as np
import numpy.typing as npt

from .bvhnode import BvhNode, BvhJoint, BvhRoot
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
    """
    node_list, frame_array, frame_time = _extract_bvh_file_info(filepath)
    num_joints = len([n for n in node_list if not n.is_end_site()])
    root_pos = frame_array[:, :3].astype(np.float64)
    joint_angles = frame_array[:, 3:].reshape(frame_array.shape[0], num_joints, 3).astype(np.float64)
    if warn_on_world_up_disagreement:
        return Bvh(nodes=node_list, root_pos=root_pos, joint_angles=joint_angles,
                   frame_time=frame_time, world_up=world_up,
                   lr_mapping=lr_mapping)
    with warnings.catch_warnings():
        # Specific filter: only suppress the rest/animation-frame disagreement
        # warning; everything else still propagates.
        warnings.filterwarnings("ignore", message="Rest pose suggests world up")
        return Bvh(nodes=node_list, root_pos=root_pos, joint_angles=joint_angles,
                   frame_time=frame_time, world_up=world_up,
                   lr_mapping=lr_mapping)

def _extract_bvh_file_info(filepath: str | Path) -> tuple[list[BvhNode], npt.NDArray[np.float64], float]:
    """Extract node hierarchy, frame data, and frame time from a BVH file."""
    #list of BvhNode objects in the file hierarchy
    node_list: list[BvhNode] = []
    # this is a flag variable to tell us if the parent of the joint
    # we are working on is the directly previous joint in the file, or not
    parent_is_previous: bool = True
    # this is necessary when the parent of the joint we are working on
    # is not directly the previous joint in the file
    parent_depth: int = 1
    # line number if we need to report a problem in the file
    line_number: int = 0
    frame_count: int = 0
    frame_time: float = 0.0

    filepath = test_file(filepath)

    with open(filepath, "r") as f:
        #---------- first, read the hierarchy in the file (first part of the file)

        # read the file line by line
        for raw_line in f:
            line_number += 1
            line = raw_line.split()
            if not line:
                continue
            # if the line starts with ROOT, then the next 3 lines are about the information of the root
            # we want to save them in a BvhRoot object

            if line[0] == 'ROOT':
                name = line[1]
                try:
                    offset, pos_channels, rot_channels, line_number = _get_offset_channels('root', f, line_number)
                except Exception as e:
                    raise ValueError(
                        f"Could not read the offset or channels of the root {name}, "
                        f"at line {line_number} in file {filepath}: {e}") from e

                node_list.append(BvhRoot(name, offset, pos_channels, rot_channels, [], None))  # type: ignore[arg-type]

            # if the line starts with JOINT,
            # then the next 3 lines are about the information of this joint
            # we want to save them in a BvhJoint object
            elif line[0] == 'JOINT':
                name = line[1]
                try:
                    offset, pos_channels, rot_channels, line_number = _get_offset_channels('joint', f, line_number)
                except Exception as e:
                    raise ValueError(
                        f"Could not read the offset or channels of the joint {name}, "
                        f"at line {line_number} in file {filepath}: {e}") from e
                #since it is a joint, pos_channels == None

                if parent_is_previous:
                    # if this joint is direclty folowing the previous one in the file,
                    # its parent is the previous Joint in the list
                    parent_node = node_list[-1]
                    parent_depth = 1
                else:
                    # if this joint is not directly following its parent in the file,
                    # that means that there was and 'End Site' above it,
                    # with also a certain number of '}'. Those increased the parent_depth variable.
                    # To get its parent, we will walk backward the chain of parent-child links
                    # as many times as the parent_depth tell us to.
                    # the first node of the chain needs to be the previous end site
                    parent_node = node_list[-1]
                    parent_depth -= 1
                    for i in range(parent_depth):
                        parent_node = parent_node.parent  # type: ignore[assignment]

                node_list.append(BvhJoint(name, offset, rot_channels, [], parent_node))  # type: ignore[arg-type]
                # let's not forget that, we gave this Joint a parent,
                # but then we also need to tell the parent that this is its child
                # so we link its parent directly to the node we just added in the list
                parent_node.children = parent_node.children + [node_list[-1]]  # type: ignore[attr-defined]
                parent_is_previous = True

            elif line[0] == 'End':
                try:
                    offset, pos_channels, rot_channels, line_number = _get_offset_channels('end_site', f, line_number)
                except Exception as e:
                    raise ValueError(
                        f"Could not read the offset or channels of the End Site "
                        f"at line {line_number} in file {filepath}: {e}") from e
                # since it is an end site, pos_channels == None and rot_channels == None

                #for an end site, the parent is always just before in the list
                parent_node = node_list[-1]
                parent_depth = 1

                node_list.append(BvhNode('End Site '+parent_node.name, offset, parent_node))  # type: ignore[arg-type]
                parent_node.children = parent_node.children + [node_list[-1]]  # type: ignore[attr-defined]

            elif line[0] == '}':
                # to corectly assign the parent to a node,
                # we need to increase the parent_depth variable every time
                # we read a '}' in the file
                parent_depth += 1
                parent_is_previous = False

            elif line[0] == "Frames:":
                frame_count = int(line[1])  # noqa: redefinition OK

            elif line[0] == "Frame" and line[1] == "Time:":
                frame_time = float(line[2])  # noqa: redefinition OK
                # If the file's 6-digit Frame Time obviously truncates an
                # exact 1/N rate (e.g. ``0.033333`` for 30 fps), snap to
                # 1/N so resample-and-back round-trips don't drift.  Only
                # applied when the snapped value is within 0.01% of the
                # literal — non-integer rates like 23.976 fps are NOT
                # snapped, since `1/24 ≠ 1/23.976`.
                if frame_time > 0:
                    snapped = 1.0 / round(1.0 / frame_time)
                    if abs(snapped - frame_time) / frame_time < 1e-4:
                        frame_time = snapped
                # --- we close the loop related to reading the hierarchy ---
                break
        #small test to see if we reach the end of the hierarchy with no trouble.
        if frame_count == 0 or frame_time == 0.0:
            raise ValueError(
                f"Frame count ({frame_count}) or frame time ({frame_time}) "
                f"is missing or zero in {filepath}")

        #----------  End of the Hierarchy part. After the hierarchy comes the frames data.

        # Calculate number of channels: 6 for root (3 pos + 3 rot), 3 for each other non-end-site joint
        non_end_site_nodes = [n for n in node_list if not n.is_end_site()]
        num_channels = 3 + 3 * len(non_end_site_nodes)  # 3 root pos + 3 rot per joint (including root)

        frame_array = np.empty((frame_count, num_channels))
        frame_number = 0
        for data_line in f:
            data_parts = data_line.split()
            if not data_parts:
                continue
            frame_array[frame_number] = [float(x) for x in data_parts]
            frame_number += 1

        if frame_number != frame_count:
            raise ValueError(
                f"BVH declares {frame_count} frames but file contains "
                f"{frame_number} data lines")


    #-----------------end of reading the file
    # frame_template is a list we created of the form [jointName_ax_pos/rot].
    # ex : [Hips_X_pos, Hips_Y_pos, Hips_Z_pos, Hips_X_rot, ...]
    return (node_list, frame_array, frame_time)


def _get_offset_channels(node_type: str, f: TextIO, line_number: int) -> tuple[list[float] | None, list[str] | None, list[str] | None, int]:
    """Read offset and channel lines for a single node from the open file."""
    offset: list[float] | None = None
    rot_channels: list[str] | None = None
    pos_channels: list[str] | None = None

    # i is used to get out of the subloop after 3 or 2 lines
    # depending on the node type
    i=0

    if node_type == 'root':
        for raw_ln in f:
            line_number += 1
            parts = raw_ln.split()
            if parts[0] == 'OFFSET':
                offset = [float(x) for x in parts[1:]]
            elif parts[0] == 'CHANNELS':
                pos_channels, rot_channels = [x[0] for x in parts[2:5]], [x[0] for x in parts[5:]]
            #after 3 lines we get out of the subloop
            if i == 2:
                break
            i += 1
        #checking that the information is complete
        if offset is None or pos_channels is None or rot_channels is None:
            raise ValueError("root missing OFFSET or CHANNELS line")
        if len(offset) != 3 or len(pos_channels) != 3 or len(rot_channels) != 3:
            raise ValueError(
                f"root must have 3 OFFSET + 3 position + 3 rotation channels, "
                f"got {len(offset)}/{len(pos_channels)}/{len(rot_channels)}")
    elif node_type == 'joint':
        for raw_ln in f:
            line_number += 1
            parts = raw_ln.split()
            if parts[0] == 'OFFSET':
                offset = [float(x) for x in parts[1:]]
            elif parts[0] == 'CHANNELS':
                rot_channels = [x[0] for x in parts[2:]]
            #after 3 lines we get out of the subloop
            if i ==2:
                break
            i += 1
        #checking that the information is complete
        if offset is None or rot_channels is None:
            raise ValueError("joint missing OFFSET or CHANNELS line")
        if len(offset) != 3 or len(rot_channels) != 3:
            raise ValueError(
                f"joint must have 3 OFFSET + 3 rotation channels, "
                f"got {len(offset)}/{len(rot_channels)}")
    elif node_type == 'end_site':
        for raw_ln in f:
            line_number += 1
            parts = raw_ln.split()
            if parts[0] == 'OFFSET':
                offset = [float(x) for x in parts[1:]]
            #after 2 lines we get out of the subloop
            if i ==1:
                break
            i += 1
        #checking that the information is complete
        if offset is None:
            raise ValueError("end site missing OFFSET line")
        if len(offset) != 3:
            raise ValueError(
                f"end site OFFSET must have 3 values, got {len(offset)}")
    else:
        raise ValueError('node_type should be either root, joint or end_site')

    return (offset, pos_channels, rot_channels, line_number)


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
    Exception
        If the file extension is not ``.bvh`` or the parent directory
        does not exist.
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
        f.write(f'Frame Time: {bvh.frame_time:.6f}\n')

        F = bvh.frame_count
        if F > 0:
            motion = np.column_stack([bvh.root_pos,
                                      bvh.joint_angles.reshape(F, -1)])
            np.savetxt(f, motion, fmt='%.6f', delimiter=' ')

    if verbose:
        print(f'Succesfully saved the file {filepath.name} at the location\n{filepath.parent.absolute()}')
