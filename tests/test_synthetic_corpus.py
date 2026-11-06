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
equal clip, and the second write the bytes of the first.
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
