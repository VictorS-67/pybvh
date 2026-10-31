"""A clip's rate: the one refusal of an unset clip where a rate is needed."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from pybvh import batch, read_bvh_file

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


def _refusal(operation: str, *, in_frames: bool = False, alternative: str | None = None) -> str:
    """The one message an unset clip gets, as a regex."""
    way_out = "Set bvh.frame_time (or bvh.fps)"
    if in_frames:
        way_out += ", or pass in_frames=True for per-frame units"
    if alternative is not None:
        way_out += f", or {alternative}"
    return re.escape(
        f"{operation} needs the clip's frame rate, but its frame_time is 0 (unset). {way_out}"
    )


# Each operation that needs a rate: the name its message gives, and a call.
RATE_OPERATIONS = [
    pytest.param("resample()", lambda clip: clip.resample(30), id="resample"),
    # The check comes before the shortcut of a clip too short to interpolate.
    pytest.param("resample()", lambda clip: clip[0:1].resample(30), id="resample-one-frame"),
    pytest.param(
        "root_trajectory(include_velocities=True)",
        lambda clip: clip.root_trajectory(include_velocities=True),
        id="root_trajectory",
    ),
    pytest.param("perturb_speed()", lambda clip: clip.perturb_speed(1.2), id="perturb_speed"),
    pytest.param(
        "foot_contacts() with method='combined'",
        lambda clip: clip.foot_contacts(),
        id="foot_contacts-speed",
    ),
    pytest.param(
        "ground_contacts() with method='velocity'",
        lambda clip: clip.ground_contacts(["LeftFoot"], method="velocity"),
        id="ground_contacts-speed",
    ),
    pytest.param("movement_phase()", lambda clip: clip.movement_phase("Head"), id="movement_phase"),
    pytest.param("smoothness()", lambda clip: clip.smoothness("Head"), id="smoothness"),
    pytest.param(
        "velocity_reductions()",
        lambda clip: clip.velocity_reductions("Head"),
        id="velocity_reductions",
    ),
    pytest.param("walking_pace()", lambda clip: clip.walking_pace(), id="walking_pace"),
]

_DURATIONS_OFF = "disable them with min_contact_duration=0.0 and min_gap_duration=0.0"

# The contact detectors' duration filters, in seconds, offer a way out of their own.
DURATION_OPERATIONS = [
    pytest.param(
        "foot_contacts()'s duration filters",
        lambda clip, **off: clip.foot_contacts(method="height", **off),
        id="foot_contacts",
    ),
    pytest.param(
        "ground_contacts()'s duration filters",
        lambda clip, **off: clip.ground_contacts(["LeftFoot"], method="height", **off),
        id="ground_contacts",
    ),
]

# The per-second descriptors, which offer per-frame units instead.
IN_FRAMES_OPERATIONS = [
    pytest.param(name, id=name)
    for name in (
        "node_velocities",
        "node_accelerations",
        "node_speed_derivative",
        "angular_velocities",
        "node_jerk",
    )
]


class TestUnsetClipRefused:
    @pytest.mark.parametrize(("operation", "call"), RATE_OPERATIONS)
    def test_refuses_with_the_operation_and_the_way_out(self, unset, operation, call):
        with pytest.raises(ValueError, match=_refusal(operation)):
            call(unset)

    @pytest.mark.parametrize(("operation", "call"), DURATION_OPERATIONS)
    def test_duration_filters_refuse_and_name_the_way_to_disable_them(self, unset, operation, call):
        with pytest.raises(ValueError, match=_refusal(operation, alternative=_DURATIONS_OFF)):
            call(unset)

    @pytest.mark.parametrize(("operation", "call"), DURATION_OPERATIONS)
    def test_duration_filters_off_detect_contacts(self, unset, operation, call):
        contacts = call(unset, min_contact_duration=0.0, min_gap_duration=0.0)
        assert len(contacts) == unset.frame_count

    @pytest.mark.parametrize("name", IN_FRAMES_OPERATIONS)
    def test_per_second_descriptors_refuse_and_offer_in_frames(self, unset, name):
        with pytest.raises(ValueError, match=_refusal(f"{name}()", in_frames=True)):
            getattr(unset, name)()

    @pytest.mark.parametrize("name", IN_FRAMES_OPERATIONS)
    def test_per_second_descriptors_in_frames_return_results(self, walk, unset, name):
        np.testing.assert_array_equal(
            getattr(unset, name)(in_frames=True), getattr(walk, name)(in_frames=True)
        )

    def test_harmonize_names_the_clip_by_its_index(self, walk, unset):
        expected = re.escape(
            "harmonize() needs the frame rate of the clip at index 1, but its frame_time is 0"
        )
        with pytest.raises(ValueError, match=expected):
            batch.harmonize([walk, unset], target_fps=30)


class TestWalkingPace:
    def test_a_clip_too_short_to_measure_has_no_pace(self, walk):
        assert np.isnan(walk[0:1].walking_pace())
