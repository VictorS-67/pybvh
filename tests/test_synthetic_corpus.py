"""The synthetic corpus in ``bvh_data/synthetic/``: its generators and its reading.

Two scripts write the corpus: ``generate_synthetic_bvh.py`` writes twelve
clips, ``manifest.json`` and ``README.md``, and audits each clip with
``audit_synthetic_bvh.py``; ``generate_lying_down_bvh.py`` writes the
lying-down clip. The test runs both into a temporary directory and compares
their output with the committed files byte for byte, so that a change to a
script and its regenerated output land together. The lying-down clip takes
its skeleton from ``bvh_data/bvh_test1.bvh``, so a change to that fixture
fails this test too, until the clip is regenerated.

pybvh reads every clip, ten of which declare position channels on a joint
other than the root (#186), and each round-trips: write then read gives an
equal clip, and the second write the bytes of the first. Forward kinematics
gives the positions the generator intends: the values its manifest samples,
the travel and turns of its motion formulas, worked by hand here, and the
equalities it states between clips (a rigid six-channel body and its
rotation-only twin, a body under a static reference and the same body as the
root). The ``CHANNELS 0`` root, which no corpus clip declares, is checked by
hand in ``tests/test_static_root.py``.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pybvh import read_bvh_file

REPO = Path(__file__).resolve().parent.parent
CORPUS = REPO / "bvh_data" / "synthetic"
SCRIPTS = REPO / "scripts"


def _run(script: str, *args: str) -> None:
    subprocess.run(
        [sys.executable, str(SCRIPTS / script), *args],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )


def test_generators_reproduce_the_committed_corpus(tmp_path):
    _run("generate_synthetic_bvh.py", "--check", "--out", str(tmp_path))
    _run("generate_lying_down_bvh.py", "--out", str(tmp_path))

    written = sorted(path.name for path in tmp_path.iterdir())
    committed = sorted(path.name for path in CORPUS.iterdir())
    assert written == committed
    differing = [
        name for name in written if (tmp_path / name).read_bytes() != (CORPUS / name).read_bytes()
    ]
    assert differing == []


# File and its declared frame count (the ``Frames:`` line).
READABLE = [
    ("blender_mixed.bvh", 5),
    ("bvh_synthetic_lying_down.bvh", 120),
    ("footsteps_humanoid_6ch.bvh", 5),
    ("footsteps_humanoid_offset_6ch.bvh", 5),
    ("hips_hands_6ch.bvh", 5),
    ("mixed_zxy_positions.bvh", 5),
    ("quadruped_tails_6ch.bvh", 5),
    ("reference_humanoid_6ch.bvh", 5),
    ("reference_props_humanoid_6ch.bvh", 5),
    ("reference_props_rotation_only.bvh", 5),
    ("rigid_6ch_control.bvh", 5),
    ("rotation_only_control.bvh", 5),
    ("synthetic_stretch.bvh", 5),
]


def test_every_corpus_clip_is_listed():
    assert {name for name, _ in READABLE} == {path.name for path in CORPUS.glob("*.bvh")}


@pytest.mark.filterwarnings("ignore:Rest pose suggests world up")
@pytest.mark.parametrize(("name", "frames"), READABLE, ids=[name for name, _ in READABLE])
def test_clip_reads_and_round_trips(name, frames, tmp_path):
    clip = read_bvh_file(CORPUS / name)
    assert clip.frame_count == frames
    first = tmp_path / "first.bvh"
    clip.write(first)
    reread = read_bvh_file(first)
    assert reread == clip
    second = tmp_path / "second.bvh"
    reread.write(second)
    assert second.read_bytes() == first.read_bytes()


# --------------------------------------------------------------------------
# Forward kinematics against the generator's intended positions
# --------------------------------------------------------------------------

MANIFEST = json.loads((CORPUS / "manifest.json").read_text())
PROFILES = {entry["file"]: entry for entry in MANIFEST["files"]}
FRAMES = range(MANIFEST["frames"])


def _read(name: str):
    return read_bvh_file(CORPUS / name, world_up="+y")


def _positions(clip, name: str) -> np.ndarray:
    """The ``(F, 3)`` world positions of the node ``name``."""
    return clip.node_positions()[:, clip.node_index[name]]


def _yaw(degrees: float, vector) -> np.ndarray:
    """``vector`` turned by ``degrees`` about +Y (a BVH ``Yrotation``)."""
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    x, y, z = vector
    return np.array([c * x + s * z, y, -s * x + c * z])


def _travel(base, step=6.0, bob=1.2) -> np.ndarray:
    """The generator's walking travel: along +Z, with a vertical bob."""
    return np.array(
        [
            [base[0], base[1] + bob * math.sin(2 * math.pi * t / 4), base[2] + step * t]
            for t in FRAMES
        ]
    )


@pytest.mark.parametrize("name", sorted(PROFILES), ids=sorted(PROFILES))
def test_the_position_samples_are_read_as_the_file_values(name):
    # The manifest's samples are the values the generator wrote, per joint
    # that declares position channels, in X Y Z order.
    clip = _read(name)
    for joint, samples in PROFILES[name]["position_samples_xyz"].items():
        expected = np.array(samples, dtype=float)
        node = clip.node_index[joint]
        if node == 0:
            np.testing.assert_array_equal(clip.root_pos, expected)
        else:
            column = clip.fk_topology.translation_idx[node]
            np.testing.assert_array_equal(clip.joint_translations[:, column], expected)


@pytest.mark.parametrize("name", sorted(PROFILES), ids=sorted(PROFILES))
def test_a_translating_joint_lies_at_its_sampled_distance_from_its_parent(name):
    # A joint's position channels are its whole position relative to its
    # parent (ADR 0010), so its distance to its parent is the length of its
    # values, whatever the parent's rotation: OFFSET + values would give
    # another length for every joint whose values are not zero.
    clip = _read(name)
    parents = clip.fk_topology.parent_idx
    for joint, samples in PROFILES[name]["position_samples_xyz"].items():
        node = clip.node_index[joint]
        if node == 0:
            continue
        bone = clip.node_positions()[:, node] - clip.node_positions()[:, parents[node]]
        np.testing.assert_allclose(
            np.linalg.norm(bone, axis=1),
            np.linalg.norm(np.array(samples, dtype=float), axis=1),
            atol=1e-9,
        )


class TestStaticReferenceRoot:
    """``reference_humanoid_6ch``: a static reference root, travelling hips."""

    def test_the_reference_stays_at_the_origin(self):
        clip = _read("reference_humanoid_6ch.bvh")
        np.testing.assert_array_equal(_positions(clip, "Reference"), np.zeros((5, 3)))

    def test_the_hips_carry_the_travel(self):
        clip = _read("reference_humanoid_6ch.bvh")
        np.testing.assert_allclose(_positions(clip, "Hips"), _travel((0.0, 95.0, 0.0)), atol=1e-6)

    def test_the_body_is_the_rotation_only_control_s_body(self):
        # The generator writes the same body and motion the ordinary way in
        # rotation_only_control: the root is the hips, which carry the travel.
        clip = _read("reference_humanoid_6ch.bvh")
        control = _read("rotation_only_control.bvh")
        np.testing.assert_allclose(
            clip.node_positions()[:, 1:], control.node_positions(), atol=1e-9
        )

    def test_the_travel_descriptors_describe_the_hips(self):
        # They read the body anchor, the hips, not the root, which stays at
        # the origin: under "skeleton" centering the hips stay still and the
        # reference node travels the other way.
        clip = _read("reference_humanoid_6ch.bvh")
        hips = _travel((0.0, 95.0, 0.0))
        assert clip.walking_pace() > 0.0
        np.testing.assert_allclose(clip.root_trajectory()[:, :2], hips[:, [0, 2]], atol=1e-6)
        centred = clip.node_positions(centered="skeleton")
        np.testing.assert_array_equal(centred[:, clip.node_index["Hips"]], np.zeros((5, 3)))
        np.testing.assert_allclose(centred[:, clip.node_index["Reference"]], -hips, atol=1e-6)

    def test_the_props_body_is_the_same_body_moved_by_120_0_600(self):
        clip = _read("reference_props_humanoid_6ch.bvh")
        alone = _read("reference_humanoid_6ch.bvh")
        body = [clip.node_index[node.name] for node in alone.nodes[1:]]
        np.testing.assert_allclose(
            clip.node_positions()[:, body],
            alone.node_positions()[:, 1:] + [120.0, 0.0, 600.0],
            atol=1e-9,
        )

    def test_the_sword_moves_on_its_own(self):
        clip = _read("reference_props_humanoid_6ch.bvh")
        expected = [
            [150.0 + t, 100.0 + 2.0 * math.sin(math.pi * t / 4), 630.0 + 6.0 * t] for t in FRAMES
        ]
        np.testing.assert_allclose(_positions(clip, "Sword"), expected, atol=1e-6)
        np.testing.assert_array_equal(
            _positions(clip, "Marker"), np.tile([-40.0, 0.0, 60.0], (5, 1))
        )


def test_a_rigid_six_channel_clip_has_the_fk_of_its_rotation_only_twin():
    # rigid_6ch_control repeats each joint's OFFSET as its position in every
    # frame; rotation_only_control declares rotations only.
    rigid = _read("rigid_6ch_control.bvh")
    control = _read("rotation_only_control.bvh")
    assert rigid.joint_translations.shape == (5, 21, 3)
    np.testing.assert_allclose(rigid.node_positions(), control.node_positions(), atol=1e-9)


class TestFootsteps:
    """A moving root with a Footsteps leaf and the pelvis a joint below it."""

    def test_a_zero_offset_pelvis_coincides_with_the_root(self):
        clip = _read("footsteps_humanoid_6ch.bvh")
        np.testing.assert_allclose(_positions(clip, "Hips"), _positions(clip, "Bip01"), atol=1e-9)

    def test_a_far_pelvis_turns_with_the_root(self):
        # Bip01 travels and yaws by 12 + 2 sin(pi t / 4) degrees; the hips
        # sit at 0 40 400 from it in every frame.
        clip = _read("footsteps_humanoid_offset_6ch.bvh")
        expected = [
            root + _yaw(12.0 + 2.0 * math.sin(math.pi * t / 4), (0.0, 40.0, 400.0))
            for t, root in zip(FRAMES, _travel((0.0, 95.0, 0.0)), strict=True)
        ]
        np.testing.assert_allclose(_positions(clip, "Hips"), expected, atol=1e-5)

    def test_the_quadruped_s_hips_coincide_with_the_root_and_footsteps_lie_below(self):
        clip = _read("quadruped_tails_6ch.bvh")
        root = _travel((0.0, 55.0, 0.0), step=9.0, bob=1.5)
        np.testing.assert_allclose(_positions(clip, "Hips"), root, atol=1e-6)
        np.testing.assert_allclose(
            _positions(clip, "Footsteps"), root - [0.0, 55.0, 0.0], atol=1e-6
        )


class TestStretch:
    """``synthetic_stretch``: the left shoulder moves under a yawing parent.

    The hips stay at 0 95 0 and yaw by 30 t / 4 degrees; the spine, chest
    and left arm do not rotate, so the chest is 36 straight above the hips
    and the shoulder's values, turned by the yaw, place it from the chest.
    """

    CHEST = np.array([0.0, 131.0, 0.0])

    def test_the_shoulder_follows_its_values_turned_by_the_yaw(self):
        clip = _read("synthetic_stretch.bvh")
        values = [(8.0 + 3.0 * t, 12.0 - 1.5 * t, 2.0 * t) for t in FRAMES]
        values[2] = (0.0, 0.0, 0.0)
        expected = [self.CHEST + _yaw(30.0 * t / 4, v) for t, v in zip(FRAMES, values, strict=True)]
        np.testing.assert_allclose(_positions(clip, "LeftShoulder"), expected, atol=1e-6)

    def test_a_zero_value_puts_the_shoulder_on_the_chest(self):
        # Frame 2 writes 0 0 0 for a shoulder whose OFFSET is 8 12 0.
        clip = _read("synthetic_stretch.bvh")
        np.testing.assert_allclose(
            clip.node_positions(2)[clip.node_index["LeftShoulder"]], self.CHEST, atol=1e-6
        )
        np.testing.assert_allclose(
            clip.node_positions(2)[clip.node_index["LeftArm"]],
            self.CHEST + _yaw(15.0, (14.0, 0.0, 0.0)),
            atol=1e-6,
        )
