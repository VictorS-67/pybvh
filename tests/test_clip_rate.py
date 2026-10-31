"""A clip's rate: the one refusal of an unset clip where a rate is needed."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from pybvh import read_bvh_file

BVH_DIR = Path(__file__).parent.parent / "bvh_data"


@pytest.fixture(scope="module")
def walk():
    """Sixty frames of the CMU walk, which has feet for the contact and
    gait operations."""
    return read_bvh_file(BVH_DIR / "cmu_12_01_walk.bvh")[0:60]


@pytest.fixture
def unset(walk):
    clip = walk.copy()
    clip.frame_time = 0
    return clip


def _refusal(operation: str, *, in_frames: bool = False) -> str:
    """The one message an unset clip gets, as a regex."""
    way_out = "Set bvh.frame_time (or bvh.fps)"
    if in_frames:
        way_out += ", or pass in_frames=True"
    return re.escape(
        f"{operation} needs the clip's frame rate, but its frame_time is 0 (unset). {way_out}"
    )


# Each operation that needs a rate: the name its message gives, and a call.
RATE_OPERATIONS = [
    pytest.param("resample()", lambda clip: clip.resample(30), id="resample"),
    # The check comes before the shortcut of a clip too short to interpolate.
    pytest.param("resample()", lambda clip: clip[0:1].resample(30), id="resample-one-frame"),
]


class TestUnsetClipRefused:
    @pytest.mark.parametrize(("operation", "call"), RATE_OPERATIONS)
    def test_refuses_with_the_operation_and_the_way_out(self, unset, operation, call):
        with pytest.raises(ValueError, match=_refusal(operation)):
            call(unset)
