"""BVH file I/O — reading and writing ``.bvh`` motion capture files.

Public functions:

- :func:`read_bvh_file` — parse a ``.bvh`` file into a :class:`~pybvh.bvh.Bvh`
- :func:`write_bvh_file` — write a :class:`~pybvh.bvh.Bvh` to a ``.bvh`` file
"""

from __future__ import annotations

import math
from fractions import Fraction
from itertools import chain
from pathlib import Path
from typing import Any, NamedTuple, TextIO

import numpy as np
import numpy.typing as npt

from .bvh import Bvh
from .bvhnode import BvhNode
from .node_tree import nodes_from_table
from .tools import _validate_bvh_path, _validate_frame_time

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
        when rest-pose and first-frame inferences disagree. The setting
        is kept on the returned clip for its lifetime, not spent at
        load: ``world_up`` is inferred again when you assign the motion
        (``root_pos``, ``joint_angles`` or frames), and each of those
        inferences follows it, as does :attr:`Bvh.world_up_inferred`.
        Clips that a :class:`Bvh` operation derives from this one
        (slices, ``copy()``, transformed copies, ``extract_joints``,
        ``resample``) inherit it along with the world up itself; a
        concatenation ``a + b`` follows ``a``, as it does for
        ``frame_time``. A clip rebuilt from its data
        (:func:`~pybvh.df_to_bvh`, :meth:`Bvh.from_df`) is a new clip,
        like one read from a file, and starts from the default. The
        alternative, a one-shot suppression of the load-time inference
        only, would re-warn on the first motion assignment of a clip
        you asked to keep quiet.
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

    Raises
    ------
    ValueError
        If the file is malformed, for example if its ``Frame Time`` is
        missing, zero, negative, NaN or infinite.

    Notes
    -----
    BVH files store joint angles in degrees; pybvh holds them in radians
    on :attr:`Bvh.joint_angles`. This function converts on read;
    :func:`write_bvh_file` converts back on write.

    Three things the reader normalizes rather than preserving verbatim,
    so a round-trip is lossless in *motion* but not byte-for-byte:

    - **Frame time is recovered** when the file's value is the frame
      time of a round rate, truncated or rounded by its exporter.
      Exporters often print ``Frame Time`` at 6 decimals (``0.033333``
      for 30 fps, which is 30.0003 fps), and a resample and back
      then drifts. The reader looks for the rate with the fewest
      decimals, from 0 to 3, whose frame time, rounded or truncated to
      the decimals the file prints, gives exactly the file's text. When
      exactly one rate with that many decimals fits, ``bvh.frame_time``
      is ``1 / rate``; when several fit, or none with up to 3 decimals,
      it is the value as read. ``0.033333`` and ``0.033`` read as
      ``1 / 30``, ``0.004166`` as ``1 / 240`` and ``0.033367`` as
      ``1 / 29.97``; ``0.017`` (every whole rate from 56 to 60 fps
      fits) and ``0.008341`` (119.88 and 119.89 fps) are kept as read.
      A 29.97 fps file becomes ``1 / 29.97``, which is 1.000001e-6
      relative from NTSC's exact ``1001 / 30000``; set
      ``bvh.frame_time = 1001 / 30000`` after reading when you need
      that. The rule recovers lost digits, not round numbers, so a text
      of 17 significant digits or more, which names one float exactly,
      is kept as read: ``0.033366700033366700`` reads as itself, one
      float from ``1 / 29.97``. Since the text alone cannot tell a short
      frame time from a truncated one, :func:`write_bvh_file` writes all
      17 significant digits (its Notes say why): a clip whose
      ``frame_time`` is ``0.033`` is written ``0.033000000000000002``
      and reads back as ``0.033``. None of the other BVH readers
      surveyed recovers a rate: those that use the value take the text
      as printed, and one (fairmotion) rounds the rate to a whole
      number. Where the rule recovers a rate, the printed value differs
      from pybvh's by less than one unit of its last decimal
      (``0.033333`` against ``1 / 30``); where it keeps the text, the
      two agree. The rounded rate differs from pybvh's wherever pybvh's
      is not whole (``0.033367`` gives 30 fps, ``0.017`` 59 fps), and,
      in a file at 6 decimals, on some whole rates above 720 fps
      (``0.001386`` gives 722 fps, where pybvh reads 721).
    - **Root channels are reordered** to position-first. A file
      declaring rotations before positions parses correctly but writes
      back in pybvh's canonical order.
    - **Offsets and motion values are written at 6 decimal places**
      (the frame time at 17 significant digits, which reads back as
      the same float), so re-reading a written file
      quantizes at ~1e-6 in the file's own units.

    Values pybvh infers or you set — ``world_up``, ``lr_mapping`` — have
    nowhere to live in the BVH format and are lost on write; re-apply
    them after reading.
    """
    return _bvh_from_parsed(
        _extract_bvh_file_info(filepath),
        filepath,
        world_up=world_up,
        warn_on_world_up_disagreement=warn_on_world_up_disagreement,
        lr_mapping=lr_mapping,
    )


class _ParsedBvh(NamedTuple):
    """What :func:`_extract_bvh_file_info` reads from a BVH file."""

    nodes: list[BvhNode]
    frames: npt.NDArray[np.float64]  # root position first; angles in degrees
    frame_time: float


def _bvh_from_parsed(
    parsed: _ParsedBvh,
    filepath: str | Path,
    world_up: str = "auto",
    warn_on_world_up_disagreement: bool = True,
    lr_mapping: dict[str, str] | None = None,
) -> Bvh:
    """The :class:`Bvh` of a parsed file: :func:`read_bvh_file`'s second
    step, after :func:`_extract_bvh_file_info`.

    Kept apart because this step can warn (the world up axis is inferred
    here) and the parse cannot: :func:`~pybvh.batch.read_bvh_directory`
    parses on worker threads and runs this step on the caller's, the
    only thread whose stack holds the caller's line for the warning.
    """
    node_list, frame_array, frame_time = parsed
    num_joints = len([n for n in node_list if not n.is_end_site()])
    root_pos = frame_array[:, :3].astype(np.float64)
    # BVH stores angles in degrees; pybvh holds them in radians.
    joint_angles_deg = (
        frame_array[:, 3:].reshape(frame_array.shape[0], num_joints, 3).astype(np.float64)
    )
    joint_angles = np.deg2rad(joint_angles_deg)
    return Bvh(
        nodes=node_list,
        root_pos=root_pos,
        joint_angles=joint_angles,
        frame_time=frame_time,
        world_up=world_up,
        lr_mapping=lr_mapping,
        source_path=str(filepath),
        warn_on_world_up_disagreement=warn_on_world_up_disagreement,
    )


#: The most decimals a recovered rate has: enough for 23.976 and 119.88 fps.
_RECOVERED_RATE_MAX_DECIMALS = 3

#: The significant digits that name one float64 exactly, whatever its value.
_FLOAT64_EXACT_DIGITS = 17


def _frame_time_from_text(text: str) -> float:
    """The frame time a ``Frame Time`` text means: the recovered rate's, or the text's value.

    A writer that rounded *text* to its last printed decimal started
    from a frame time in ``[x - u/2, x + u/2)``, and one that truncated
    from ``[x, x + u)``, where ``x`` is the printed value and ``u`` one
    unit of its last decimal. When :func:`_rate_with_fewest_decimals`
    finds one rate whose frame time lies in their union, the result is
    ``1 / rate``; otherwise it is ``x``. A text of
    :data:`_FLOAT64_EXACT_DIGITS` significant digits or more lost none of
    its float's digits, so it is ``x`` too. :func:`read_bvh_file` states
    the rule for the user.
    """
    printed = Fraction(text)
    # Zero is the unset frame time, refused by the caller; it has no rate.
    if printed <= 0 or _significant_digits(text) >= _FLOAT64_EXACT_DIGITS:
        return float(printed)
    unit = _unit_of_last_decimal(text)
    rate = _rate_with_fewest_decimals(printed - unit / 2, printed + unit)
    if rate is None:
        return float(printed)
    # 1 / float(rate), not float(1 / rate): the frame time ``bvh.fps = 29.97``
    # sets, so a recovered clip and one given its rate in code are equal.
    return 1 / float(rate)


def _significant_digits(text: str) -> int:
    """The significant digits *text* prints, trailing zeros included: ``0.0830`` gives 3."""
    mantissa, _, _ = text.lower().partition("e")
    return len(mantissa.lstrip("+-").replace(".", "").lstrip("0"))


def _unit_of_last_decimal(text: str) -> Fraction:
    """One unit of the last decimal *text* prints: ``0.0083`` gives 1/10000, ``8.3e-3`` too."""
    mantissa, _, exponent = text.lower().partition("e")
    _, _, decimals = mantissa.partition(".")
    return Fraction(10) ** (int(exponent or 0) - len(decimals))


def _rate_with_fewest_decimals(low: Fraction, high: Fraction) -> Fraction | None:
    """The rate with the fewest decimals whose frame time lies in ``[low, high)``.

    Rates with 0 decimals are tried first, then 1, up to
    :data:`_RECOVERED_RATE_MAX_DECIMALS`. The search stops at the first
    count of decimals at which a rate fits: the rate when it is the only
    one, ``None`` when several fit. ``None`` too when none fits. *low*
    must be positive.
    """
    for decimals in range(_RECOVERED_RATE_MAX_DECIMALS + 1):
        step = Fraction(1, 10**decimals)
        # The rate k * step fits when low <= 1 / (k * step) < high, that is
        # when 1 / (high * step) < k <= 1 / (low * step): a count of whole
        # numbers, so no candidate is enumerated.
        largest_k = math.floor(1 / (low * step))
        fitting = largest_k - math.floor(1 / (high * step))
        if fitting == 1:
            return largest_k * step
        if fitting > 1:
            return None
    return None


def _extract_bvh_file_info(filepath: str | Path) -> _ParsedBvh:
    """Extract node hierarchy, frame data, and frame time from a BVH file.

    The hierarchy is read token by token into node-table rows (see :func:`~pybvh.nodes_to_table`), one per ROOT, JOINT or End Site in file order, and the tree is built once by :func:`~pybvh.nodes_from_table`, which wires ``parent`` and ``children`` and checks the tree. Syntax errors (a second ROOT, a JOINT or End Site outside the ROOT block, an unmatched ``}``, an unreadable block) are raised here, with the line and the file.

    The returned frame array is normalized to pybvh's internal column layout: the root's 3 position columns first, then all rotation columns (root's, then each joint's in hierarchy order). Files whose root declares rotation channels before position channels are reordered on read; :func:`write_bvh_file` always writes position-first.
    """
    table: list[dict[str, Any]] = []
    # Brace stack of the open ROOT/JOINT blocks, as row indices: a row is
    # pushed when its block opens and popped on the matching '}', so the
    # parent of a new node is always the innermost open block (stack top).
    open_rows: list[int] = []
    # line number if we need to report a problem in the file
    line_number: int = 0
    frame_count: int = 0
    frame_time: float = 0.0
    # Column indices of the root's position/rotation channels within its
    # own channel block, in file order (used to normalize rotation-first
    # roots to the internal position-first layout).
    root_pos_cols: list[int] = []
    root_rot_cols: list[int] = []

    filepath = _validate_bvh_path(filepath)

    with open(filepath) as f:
        # ---------- first, read the hierarchy (first part of the file)
        for raw_line in f:
            line_number += 1
            line = raw_line.split()
            if not line:
                continue
            token = line[0]

            if token in ("ROOT", "JOINT"):
                name = line[1]
                if token == "ROOT" and table:
                    raise ValueError(
                        f"Second ROOT '{name}' at line {line_number} in file "
                        f"{filepath}: pybvh models single-root skeletons only"
                    )
                if token == "JOINT" and not open_rows:
                    raise ValueError(
                        f"JOINT '{name}' outside any ROOT block at line "
                        f"{line_number} in file {filepath}"
                    )
                node_type = "root" if token == "ROOT" else "joint"
                try:
                    offset, channels, line_number = _read_node_block(node_type, f, line_number)
                except Exception as e:
                    raise ValueError(
                        f"Could not read the offset or channels of the {node_type} "
                        f"{name}, at line {line_number} in file {filepath}: {e}"
                    ) from e

                row: dict[str, Any] = {
                    "name": name,
                    "parent": open_rows[-1] if token == "JOINT" else None,
                    "offset": offset,
                }
                if token == "ROOT":
                    row["pos_channels"] = [ax for kind, ax in channels if kind == "pos"]
                    root_pos_cols = [i for i, (kind, _) in enumerate(channels) if kind == "pos"]
                    root_rot_cols = [i for i, (kind, _) in enumerate(channels) if kind == "rot"]
                row["rot_channels"] = [ax for kind, ax in channels if kind == "rot"]
                open_rows.append(len(table))
                table.append(row)

            elif token == "End":
                if not open_rows:
                    raise ValueError(
                        f"End Site outside any ROOT block at line {line_number} in file {filepath}"
                    )
                try:
                    offset, channels, line_number = _read_node_block("end_site", f, line_number)
                except Exception as e:
                    raise ValueError(
                        f"Could not read the offset of the End Site "
                        f"at line {line_number} in file {filepath}: {e}"
                    ) from e
                # No name: the builder gives an end site its display name,
                # 'EndSite' + the parent's name. End-site blocks are fully
                # consumed by _read_node_block (including their closing
                # '}'), so they never go on the stack.
                table.append({"parent": open_rows[-1], "offset": offset})

            elif token == "}":
                if not open_rows:
                    raise ValueError(f"Unmatched '}}' at line {line_number} in file {filepath}")
                open_rows.pop()

            elif token == "Frames:":
                frame_count = int(line[1])

            elif token == "Frame" and len(line) > 2 and line[1] == "Time:":
                # Checked here, although the Bvh setter applies the same
                # rule, so that the error names the file and the line.
                try:
                    _validate_frame_time(float(line[2]))
                except ValueError as e:
                    raise ValueError(
                        f"Could not read the frame time at line {line_number} "
                        f"in file {filepath}: {e}"
                    ) from e
                frame_time = _frame_time_from_text(line[2])
                # --- we close the loop related to reading the hierarchy ---
                break
            # Other tokens ('HIERARCHY', 'MOTION') carry no data — skipped.

        # small test to see if we reach the end of the hierarchy with no trouble.
        if not table:
            raise ValueError(f"No ROOT declaration found in {filepath}")
        if frame_count == 0 or frame_time == 0.0:
            raise ValueError(
                f"Frame count ({frame_count}) or frame time ({frame_time}) "
                f"is missing or zero in {filepath}"
            )

        node_list = nodes_from_table(table)

        # ----------  End of the Hierarchy part. After the hierarchy comes the frames data.

        # Expected channels: 6 for root (3 pos + 3 rot), 3 for each other non-end-site joint
        non_end_site_nodes = [n for n in node_list if not n.is_end_site()]
        num_channels = 3 + 3 * len(non_end_site_nodes)

        # loadtxt warns, rather than raises, on input with no data row; the
        # first one is found here so that the parse never warns (it runs on
        # read_bvh_directory's worker threads, see _bvh_from_parsed). A data
        # row is what loadtxt reads as one: text left after a '#' comment is
        # cut off. The rest of the file is streamed to loadtxt behind it.
        first_row = next((row for row in f if row.split("#", 1)[0].strip()), None)
        if first_row is None:
            raise ValueError(f"BVH declares {frame_count} frames but file contains 0 data lines")
        frame_array = np.loadtxt(chain([first_row], f), ndmin=2)
        if frame_array.shape[0] != frame_count:
            raise ValueError(
                f"BVH declares {frame_count} frames but file contains "
                f"{frame_array.shape[0]} data lines"
            )
        if frame_array.shape[1] != num_channels:
            raise ValueError(
                f"BVH motion lines have {frame_array.shape[1]} values per "
                f"frame but the hierarchy declares {num_channels} channels"
            )

    # Normalize rotation-first (or interleaved) root channel layouts to the
    # internal position-first layout; within-block channel order is preserved.
    canonical_root_cols = root_pos_cols + root_rot_cols
    if canonical_root_cols != [0, 1, 2, 3, 4, 5]:
        frame_array = frame_array[:, canonical_root_cols + list(range(6, num_channels))]

    return _ParsedBvh(node_list, frame_array, frame_time)


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
        raise ValueError(f"CHANNELS declares {declared_count} channels but lists {len(tokens)}")
    channels: list[tuple[str, str]] = []
    for tok in tokens:
        tok_lower = tok.lower()
        if tok_lower.endswith("position"):
            kind = "pos"
        elif tok_lower.endswith("rotation"):
            kind = "rot"
        else:
            raise ValueError(
                f"unrecognized channel token {tok!r} (expected e.g. 'Xposition' or 'Yrotation')"
            )
        axis = tok[0].upper()
        if axis not in ("X", "Y", "Z"):
            raise ValueError(f"unrecognized axis in channel token {tok!r}")
        channels.append((kind, axis))
    return channels


def _read_node_block(
    node_type: str, f: TextIO, line_number: int
) -> tuple[list[float], list[tuple[str, str]], int]:
    """Read the lines opening a node's block: ``{``, ``OFFSET``, and ``CHANNELS``.

    Token-driven — each line is dispatched on its leading token, and completeness is validated per node type at the end, so blank lines and line-order variations don't break parsing. Joint/root blocks are left open (their children follow); end-site blocks are consumed through their closing ``}``.

    Returns ``(offset, channels, line_number)`` where ``channels`` is the ``(kind, axis)`` list from :func:`_parse_channels_line` (empty for end sites).
    """
    if node_type not in ("root", "joint", "end_site"):
        raise ValueError("node_type should be either root, joint or end_site")

    offset: list[float] | None = None
    channels: list[tuple[str, str]] | None = None

    for raw_ln in f:
        line_number += 1
        parts = raw_ln.split()
        if not parts:
            continue
        token = parts[0]
        if token == "{":
            continue
        elif token == "}":
            if node_type != "end_site":
                raise ValueError(f"{node_type} block closed before OFFSET/CHANNELS were read")
            break
        elif token == "OFFSET":
            offset = [float(x) for x in parts[1:]]
        elif token == "CHANNELS":
            channels = _parse_channels_line(parts)
        else:
            raise ValueError(f"unexpected token {token!r} in {node_type} block")
        if node_type != "end_site" and offset is not None and channels is not None:
            break

    # ---- per-node-type validation ----
    if offset is None:
        raise ValueError(f"{node_type.replace('_', ' ')} missing OFFSET line")
    if len(offset) != 3:
        raise ValueError(f"OFFSET must have 3 values, got {len(offset)}")

    if node_type == "end_site":
        if channels is not None:
            raise ValueError("end site must not declare CHANNELS")
        return (offset, [], line_number)

    if channels is None:
        raise ValueError(f"{node_type} missing CHANNELS line")
    num_pos = sum(1 for kind, _ in channels if kind == "pos")
    num_rot = len(channels) - num_pos
    if node_type == "root":
        if num_pos != 3 or num_rot != 3:
            raise ValueError(
                f"root must have 3 position + 3 rotation channels, "
                f"got {num_pos} position + {num_rot} rotation"
            )
    else:
        if num_pos != 0:
            raise ValueError(
                f"pybvh does not model position channels on non-root joints "
                f"(got {num_pos} position channels)"
            )
        if num_rot != 3:
            raise ValueError(f"joint must have exactly 3 rotation channels, got {num_rot}")

    return (offset, channels, line_number)


# ----------------------------------------------------------------
#  Writing
# ----------------------------------------------------------------


def write_bvh_file(
    bvh: Bvh, filepath: str | Path, verbose: bool = False, overwrite: bool = True
) -> None:
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
    overwrite : bool, optional
        If True (default), replace an existing file at ``filepath``.
        Pass ``False`` to refuse instead, raising ``FileExistsError`` —
        worth doing when the destination is hand-authored data rather
        than a regenerable output.

    Raises
    ------
    ValueError
        If the file extension is not ``.bvh``, or if ``bvh.frame_time``
        is 0 (unset): a file must state its frame time, and
        :func:`read_bvh_file` refuses ``Frame Time: 0``.
    FileNotFoundError
        If the parent directory does not exist.
    FileExistsError
        If ``filepath`` exists and ``overwrite=False``.

    Notes
    -----
    pybvh stores joint angles in radians, but the BVH format requires
    degrees; this function converts on write.

    Offsets and motion values are written with 6 decimal places, the de
    facto BVH convention (and what most DCC tools emit) — motion data
    re-read from a written file is therefore quantized at ~1e-6 in the
    skeleton's length unit and in degrees. ``Frame Time`` is the
    exception: it is written at 17 significant digits with trailing
    zeros kept (``f"{frame_time:#.17g}"``: ``0.033333333333333333`` for
    30 fps, ``0.12000000000000000`` for 0.12 s), so the file holds the
    clip's frame time without loss. A truncated frame time compounds
    across every frame of a resample, while a truncated coordinate does
    not. The shortest text that reads back as the same float (Python's
    ``repr``) is not enough: :func:`read_bvh_file` takes a ``Frame
    Time`` with few decimals for an exporter's truncated rate, so
    ``0.12`` would read back as ``1 / 8`` and ``0.033`` as ``1 / 30``.
    The reader keeps a text of 17 significant digits as read, so every
    frame time this function writes reads back as the same float.

    The BVH format carries no place for ``world_up``, ``lr_mapping``, or
    ``source_path``; those are lost on write and re-inferred on read.
    Root channels are always written position-first, whatever order the
    source file declared.
    """
    filepath = Path(filepath)
    if filepath.suffix != ".bvh":
        raise ValueError(f"{filepath.name} is not a .bvh file")
    elif not filepath.parent.exists():
        raise FileNotFoundError(f"directory does not exist: {filepath.parent}")
    # checked before the file is opened — "w" truncates on open, so a late
    # check would already have destroyed the data it is meant to protect
    elif not overwrite and filepath.exists():
        raise FileExistsError(f"{filepath} already exists; pass overwrite=True to replace it.")
    if bvh.frame_time == 0:
        raise ValueError(
            f"Cannot write {filepath.name}: the clip's frame_time is 0 (unset), and a "
            "BVH file must state its frame time. Set bvh.frame_time (or bvh.fps) first."
        )

    def offset_to_str(node: BvhNode) -> str:
        offset_str = "OFFSET"
        for num in node.offset:
            offset_str += " " + f"{num:.6f}"
        return offset_str

    def channels_to_str(node: BvhNode) -> str:
        chanels_str = "CHANNELS"
        if node.parent is None:
            chanels_str += " 6"
            for pos_ax in node.pos_channels:  # type: ignore[attr-defined]
                chanels_str += " " + pos_ax + "position"
        else:
            chanels_str += " 3"

        for rot_ax in node.rot_channels:  # type: ignore[attr-defined]
            chanels_str += " " + rot_ax + "rotation"

        return chanels_str

    def rec_node_to_file(node: BvhNode, file: TextIO, depth: int = 0) -> None:
        if node.is_end_site():
            print("\t" * depth + "End Site", file=file)
            print("\t" * depth + "{", file=file)
            print("\t" * (depth + 1) + offset_to_str(node), file=file)
            print("\t" * depth + "}", file=file)
        else:
            if node.parent is None:
                type_str = "ROOT"
            else:
                type_str = "JOINT"
            print("\t" * depth + type_str + " " + node.name, file=file)
            print("\t" * depth + "{", file=file)
            print("\t" * (depth + 1) + offset_to_str(node), file=file)
            print("\t" * (depth + 1) + channels_to_str(node), file=file)
            for child in node.children:  # type: ignore[attr-defined]
                rec_node_to_file(child, file=file, depth=depth + 1)
            print("\t" * depth + "}", file=file)

    with open(filepath, "w") as f:
        f.write("HIERARCHY\n")

        rec_node_to_file(bvh.root, file=f)

        f.write("MOTION\n")
        f.write(f"Frames: {bvh.frame_count}\n")
        f.write(f"Frame Time: {float(bvh.frame_time):#.{_FLOAT64_EXACT_DIGITS}g}\n")

        F = bvh.frame_count
        if F > 0:
            # pybvh stores angles in radians; BVH format requires degrees.
            joint_angles_deg = np.rad2deg(bvh.joint_angles)
            motion = np.column_stack([bvh.root_pos, joint_angles_deg.reshape(F, -1)])
            np.savetxt(f, motion, fmt="%.6f", delimiter=" ")

    if verbose:
        print(
            f"Successfully saved the file {filepath.name} at the location\n{filepath.parent.absolute()}"
        )
