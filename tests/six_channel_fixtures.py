"""Skeletons and files with translating joints or a static root (#186),
shared by the test modules of #186.

- :func:`reference_hips_table`: a reference root, a ``Hips`` joint that
  translates, a ``Spine`` joint and an end site, as a node table.
- :func:`rotating_spine_twin`: a clip on that skeleton, with the spine
  declaring rotation channels only.
- :func:`blender_style_text`: the text of a file with a static root above
  a translating ``Hips`` and a ``Prop`` that only rotates.
"""

from __future__ import annotations

from pybvh import Bvh, nodes_from_table


def reference_hips_table(
    *, spine_translates: bool = False, hips_order: str = "ZYX", spine_order: str = "ZYX"
) -> list[dict]:
    """Reference root -> Hips (translates) -> Spine (translates or not) -> end site."""
    spine = {"name": "Spine", "parent": 1, "offset": [0, 10, 0], "rot_channels": spine_order}
    if spine_translates:
        spine["pos_channels"] = "XYZ"
    return [
        {"name": "Reference", "parent": None, "offset": [0, 0, 0], "rot_channels": "ZYX"},
        {
            "name": "Hips",
            "parent": 0,
            "offset": [0, 90, 0],
            "pos_channels": "XYZ",
            "rot_channels": hips_order,
        },
        spine,
        {"parent": 2, "offset": [0, 5, 0]},
    ]


def rotating_spine_twin(clip: Bvh) -> Bvh:
    """``clip``'s skeleton with the spine (node 2) declaring rotation
    channels only, and ``clip``'s motion less the spine's translations."""
    table = clip.to_node_table()
    del table[2]["pos_channels"]
    return Bvh(
        nodes=nodes_from_table(table),
        root_pos=clip.root_pos,
        joint_angles=clip.joint_angles,
        joint_translations=clip.joint_translations[:, :1],
        frame_time=clip.frame_time,
        world_up="+y",
    )


_BLENDER_STYLE_HIERARCHY = """HIERARCHY
ROOT __0
{
\tOFFSET 0 2 0
\tCHANNELS 0
\tJOINT Hips
\t{
\t\tOFFSET 0 90 0
\t\tCHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
\t\tEnd Site
\t\t{
\t\t\tOFFSET 0 10 0
\t\t}
\t}
\tJOINT Prop
\t{
\t\tOFFSET 30 0 0
\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\tEnd Site
\t\t{
\t\t\tOFFSET 0 4 0
\t\t}
\t}
}
"""

# Frame 1 moves the hips to 10 90 5 from the root and turns them 90
# degrees about Z, and turns the prop 90 degrees about X. Frame 2 turns
# both about every axis.
_BLENDER_STYLE_FRAMES = [
    "0 90 0 0 0 0 0 0 0",
    "10 90 5 90 0 0 0 0 90",
    "-3 85 2 30 -60 170 45 89 -10",
]


def blender_style_text(frame_count: int) -> str:
    """A file as Blender's exporter writes one: a static root 2 above the
    origin, and under it the hips, which translate and turn, and a prop,
    which only rotates; the first *frame_count* frames, of 3."""
    frames = _BLENDER_STYLE_FRAMES[:frame_count]
    motion = f"MOTION\nFrames: {len(frames)}\nFrame Time: 0.1\n"
    return _BLENDER_STYLE_HIERARCHY + motion + "".join(f"{frame}\n" for frame in frames)
