#!/usr/bin/env python3
"""Independently audit the authored corpus; Python 3.9+, stdlib only, MIT.

This parser consumes serialized files without using the generator's tree or
pybvh. The anatomical pelvis is an author-supplied field; the audit checks
that it names a joint of the file.

Run: python3 scripts/audit_synthetic_bvh.py bvh_data/synthetic
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Node:
    name: str
    parent: int | None
    offset: list[str]
    channels: list[str]
    children: list[int] = field(default_factory=list)
    end: bool = False
    column: int = 0


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(tokens):
    return all(math.isfinite(float(t)) for t in tokens)


def parse_bvh(text):
    """Strict recursive line parser retaining decimal tokens and End Sites."""
    lines = [line.split() for line in text.splitlines() if line.strip()]
    cursor = 0
    nodes = []
    width = 0

    def take():
        nonlocal cursor
        require(cursor < len(lines), "unexpected end of file")
        row = lines[cursor]
        cursor += 1
        return row

    def expect(tokens):
        require(take() == tokens, "expected " + " ".join(tokens))

    def node(parent=None, root=False):
        nonlocal width
        header = take()
        end = header == ["End", "Site"]
        if root:
            require(len(header) == 2 and header[0] == "ROOT", "one ROOT required")
        elif not end:
            require(len(header) == 2 and header[0] == "JOINT", "expected JOINT or End Site")
        name = "EndSite" + nodes[parent].name if end else header[1]
        expect(["{"])
        off = take()
        require(len(off) == 4 and off[0] == "OFFSET" and finite(off[1:]), "finite XYZ OFFSET required")
        channels = []
        if not end:
            decl = take()
            require(len(decl) >= 2 and decl[0] == "CHANNELS", "missing CHANNELS")
            channels = decl[2:]
            require(int(decl[1]) == len(channels), "CHANNELS count mismatch")
            rotations = [c for c in channels if c.endswith("rotation")]
            positions = [c for c in channels if c.endswith("position")]
            require(sorted(rotations) == [a + "rotation" for a in "XYZ"], "three distinct rotations required")
            require(not positions or sorted(positions) == [a + "position" for a in "XYZ"], "three distinct positions required")
            require(len(channels) == len(rotations) + len(positions) in (3, 6), "illegal layout")
            if root:
                require(positions == [a + "position" for a in "XYZ"], "root XYZ position order")
        idx = len(nodes)
        nodes.append(Node(name, parent, off[1:], channels, end=end, column=width))
        width += len(channels)
        if parent is not None:
            require(parent < idx and not nodes[parent].end, "parent traversal order")
            nodes[parent].children.append(idx)
        while cursor < len(lines) and lines[cursor] != ["}"]:
            require(not end, "End Site cannot have children")
            node(idx)
        expect(["}"])

    expect(["HIERARCHY"])
    node(root=True)
    expect(["MOTION"])
    declaration = take()
    require(len(declaration) == 2 and declaration[0] == "Frames:", "missing Frames")
    frames = int(declaration[1])
    timing = take()
    require(len(timing) == 3 and timing[:2] == ["Frame", "Time:"], "missing Frame Time")
    require(finite(timing[2:]) and float(timing[2]) > 0, "positive finite frame time")
    rows = lines[cursor:]
    require(frames == len(rows) == 5, "exactly five frames required")
    require(all(len(row) == width and finite(row) for row in rows), "motion width or finite values")
    names = [n.name.casefold() for n in nodes]
    require(len(set(names)) == len(names), "case-folded node-name collision")
    joints = [n.name for n in nodes if not n.end]
    for name in joints:
        if name.startswith("Left"):
            require("Right" + name[4:] in joints, "missing right partner: " + name)
        if name.startswith("Right"):
            require("Left" + name[5:] in joints, "missing left partner: " + name)
    return nodes, rows, timing[2]


def audit(text, expected_pelvis, authored=None):
    nodes, rows, frame_time = parse_bvh(text)
    joints = [n for n in nodes if not n.end]
    names = [n.name for n in nodes]
    require(expected_pelvis in [n.name for n in joints], "missing anatomical pelvis")
    six, rigid, moving, changing_position, changing_rotation = [], [], [], [], []
    position_samples = {}
    original = {j.name: j for j in authored.root.walk()} if authored else None
    if original is not None:
        require(list(original) == [j.name for j in joints], "authored joint traversal mismatch")
        require(frame_time == authored.frame_time, "authored frame time mismatch")
    for i, n in enumerate(nodes):
        if n.end:
            if original is not None:
                expected_tip = original[nodes[n.parent].name].end_site
                require(expected_tip is not None and list(map(float, n.offset)) == list(expected_tip), "authored End Site mismatch")
            continue
        samples = [{c: row[n.column + k] for k, c in enumerate(n.channels)} for row in rows]
        positions = [[s[a + "position"] for a in "XYZ"] for s in samples] if "Xposition" in n.channels else []
        rotations = [[s[a + "rotation"] for a in "XYZ"] for s in samples]
        if any(r != rotations[0] for r in rotations[1:]):
            changing_rotation.append(n.name)
        if positions:
            position_samples[n.name] = positions
            if any(r != positions[0] for r in positions[1:]):
                changing_position.append(n.name)
            if i:
                six.append(n.name)
                (rigid if all(r == n.offset for r in positions) else moving).append(n.name)
        if original is not None:
            j = original[n.name]
            require(list(j.channels) == n.channels, "authored channel layout: " + n.name)
            require(list(map(float, n.offset)) == list(j.offset), "authored offset: " + n.name)
            parent = None if n.parent is None else nodes[n.parent].name
            expected_parent = next((p.name for p in original.values() if j in p.children), None)
            require(parent == expected_parent, "authored parent: " + n.name)
            if positions:
                for t, actual in enumerate(positions):
                    # Independent formatting; the generator's fmt is not imported.
                    target = [format(v if abs(v) >= 0.0000005 else 0.0, ".6f")
                              for v in authored.motion.position(j, t)]
                    require(actual == target, "authored local position: " + n.name)
                if i and n.name not in authored.motion.pos_override:
                    require(all(r == n.offset for r in positions), "rigid OFFSET tokens: " + n.name)
            if i and n.name in authored.motion.pos_override:
                require(bool(positions), "animated local position requires channels: " + n.name)
    return dict(joints=len(joints), end_sites=len(nodes) - len(joints),
                width=sum(len(n.channels) for n in nodes),
                six_channel_non_root_joints=six, rigid_six_channel_joints=rigid,
                moving_six_channel_joints=moving, position_varying_joints=changing_position,
                rotation_varying_joints=changing_rotation,
                position_samples_xyz=position_samples,
                root_position_channels=[c for c in nodes[0].channels if c.endswith("position")],
                pelvis_node_index=names.index(expected_pelvis),
                pelvis_joint_index=[n.name for n in joints].index(expected_pelvis),
                node_identities=[{"id": n.name, "name": n.name, "node_index": i,
                                  "parent_id": None if n.parent is None else nodes[n.parent].name,
                                  "end_site": n.end, "offset": n.offset, "channels": n.channels}
                                 for i, n in enumerate(nodes)],
                reader_status_0_8_2="rejected: position channels on a non-root joint" if six else "reads")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.directory / "manifest.json").read_text(encoding="utf-8"))
    require(sorted(p.name for p in args.directory.glob("*.bvh")) == sorted(e["file"] for e in manifest["files"]), "file inventory")
    for entry in manifest["files"]:
        data = (args.directory / entry["file"]).read_bytes()
        require(hashlib.sha256(data).hexdigest() == entry["sha256"], "SHA-256: " + entry["file"])
        result = audit(data.decode("utf-8"), entry["expected_pelvis"])
        require(all(entry[k] == v for k, v in result.items()), "manifest annotation mismatch: " + entry["file"])
        print(entry["file"] + ": integrity, motion, identities and hash OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
