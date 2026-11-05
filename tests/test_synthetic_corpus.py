"""The synthetic corpus in ``bvh_data/synthetic/``: its generators and its reading.

Two scripts write the corpus: ``generate_synthetic_bvh.py`` writes twelve
clips, ``manifest.json`` and ``README.md``, and audits each clip with
``audit_synthetic_bvh.py``; ``generate_lying_down_bvh.py`` writes the
lying-down clip. The test runs both into a temporary directory and compares
their output with the committed files byte for byte, so that a change to a
script and its regenerated output land together.

pybvh reads three of the thirteen clips. The other ten declare position
channels on a joint other than the root, which the reader refuses; each
refusal is pinned with its message, so that supporting that dialect turns a
pinned refusal into a load, one file at a time.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

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
    ("bvh_synthetic_lying_down.bvh", 120),
    ("reference_props_rotation_only.bvh", 5),
    ("rotation_only_control.bvh", 5),
]

# File, the first joint that declares position channels, and its line.
REFUSED = [
    ("blender_mixed.bvh", "Neck", 18),
    ("footsteps_humanoid_6ch.bvh", "Footsteps", 6),
    ("footsteps_humanoid_offset_6ch.bvh", "Footsteps", 6),
    ("hips_hands_6ch.bvh", "Spine", 6),
    ("mixed_zxy_positions.bvh", "Neck", 18),
    ("quadruped_tails_6ch.bvh", "Footsteps", 6),
    ("reference_humanoid_6ch.bvh", "Hips", 6),
    ("reference_props_humanoid_6ch.bvh", "Hips", 6),
    ("rigid_6ch_control.bvh", "Spine", 6),
    ("synthetic_stretch.bvh", "LeftShoulder", 32),
]


def test_every_corpus_clip_is_either_read_or_refused():
    listed = {name for name, _ in READABLE} | {name for name, _, _ in REFUSED}
    assert listed == {path.name for path in CORPUS.glob("*.bvh")}


@pytest.mark.parametrize(("name", "frames"), READABLE, ids=[name for name, _ in READABLE])
def test_clip_without_non_root_positions_reads(name, frames):
    clip = read_bvh_file(CORPUS / name)
    assert clip.frame_count == frames


@pytest.mark.parametrize(("name", "joint", "line"), REFUSED, ids=[name for name, _, _ in REFUSED])
def test_clip_with_non_root_positions_is_refused(name, joint, line):
    path = CORPUS / name
    with pytest.raises(ValueError) as refusal:
        read_bvh_file(path)
    assert str(refusal.value) == (
        f"Could not read the offset or channels of the joint {joint}, at line {line} "
        f"in file {path}: pybvh does not model position channels on non-root joints "
        "(got 3 position channels)"
    )
