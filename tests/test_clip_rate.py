"""A clip's rate: the one refusal of an unset clip where a rate is needed."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from pybvh import batch, bvhplot, read_bvh_file

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


def _refusal(
    operation: str,
    *,
    in_frames: bool = False,
    alternative: str | None = None,
    index: int | None = None,
) -> str:
    """The one message an unset clip gets, as a regex."""
    needed = (
        "the clip's frame rate" if index is None else f"the frame rate of the clip at index {index}"
    )
    way_out = "Set bvh.frame_time (or bvh.fps)"
    if in_frames:
        way_out += ", or pass in_frames=True for per-frame units"
    if alternative is not None:
        way_out += f", or {alternative}"
    return re.escape(f"{operation} needs {needed}, but its frame_time is 0 (unset). {way_out}")


def _rated(clip):
    """A copy of ``clip`` at 30 fps, to stand beside it in a list."""
    rated = clip.copy()
    rated.fps = 30
    return rated


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
IN_FRAMES_NAMES = (
    "node_velocities",
    "node_accelerations",
    "node_speed_derivative",
    "angular_velocities",
    "node_jerk",
)

# Every operation that needs a rate: the message an unset clip gets, and a call.
# One list, so an operation that comes to need a rate joins it here.
RATE_OPERATIONS = [
    pytest.param(_refusal("resample()"), lambda clip: clip.resample(30), id="resample"),
    # The check comes before the shortcut of a clip too short to interpolate.
    pytest.param(
        _refusal("resample()"), lambda clip: clip[0:1].resample(30), id="resample-one-frame"
    ),
    pytest.param(
        _refusal("root_trajectory(include_velocities=True)"),
        lambda clip: clip.root_trajectory(include_velocities=True),
        id="root_trajectory",
    ),
    pytest.param(
        _refusal("perturb_speed()"), lambda clip: clip.perturb_speed(1.2), id="perturb_speed"
    ),
    pytest.param(
        _refusal("foot_contacts() with method='combined'"),
        lambda clip: clip.foot_contacts(),
        id="foot_contacts-speed",
    ),
    pytest.param(
        _refusal("ground_contacts() with method='velocity'"),
        lambda clip: clip.ground_contacts(["LeftFoot"], method="velocity"),
        id="ground_contacts-speed",
    ),
    pytest.param(
        _refusal("movement_phase()"), lambda clip: clip.movement_phase("Head"), id="movement_phase"
    ),
    pytest.param(_refusal("smoothness()"), lambda clip: clip.smoothness("Head"), id="smoothness"),
    pytest.param(
        _refusal("velocity_reductions()"),
        lambda clip: clip.velocity_reductions("Head"),
        id="velocity_reductions",
    ),
    pytest.param(_refusal("walking_pace()"), lambda clip: clip.walking_pace(), id="walking_pace"),
    pytest.param(
        _refusal("gait_parameters()"), lambda clip: clip.gait_parameters(), id="gait_parameters"
    ),
    # Contacts given, nothing reads the speed, but cadence and pace are per second.
    pytest.param(
        _refusal("gait_parameters()"),
        lambda clip: clip.gait_parameters(contacts=np.zeros((clip.frame_count, 2))),
        id="gait_parameters-contacts-given",
    ),
    # Contacts given, a cadence is still per second.
    pytest.param(
        _refusal("cadence()"),
        lambda clip: clip.cadence(contacts=np.zeros((clip.frame_count, 2))),
        id="cadence-contacts-given",
    ),
    # No contacts given, a stride length detects them first, which needs the rate.
    pytest.param(
        _refusal("foot_contacts() with method='combined'"),
        lambda clip: clip.stride_length(),
        id="stride_length-contacts-detected",
    ),
    pytest.param(
        _refusal("harmonize()", index=1),
        lambda clip: batch.harmonize([_rated(clip), clip], target_fps=30),
        id="harmonize",
    ),
    pytest.param(_refusal("Writing take.bvh"), lambda clip: clip.write("take.bvh"), id="write"),
    # bvhplot's ghost trail and match_fps need each clip's own rate, fps= given or not.
    pytest.param(
        _refusal("render() with ghost=2", alternative="drop ghost="),
        lambda clip: bvhplot.render(clip, "x.gif", backend="matplotlib", fps=10, ghost=2),
        id="render-ghost",
    ),
    pytest.param(
        _refusal("render() with match_fps='lowest'", alternative="drop match_fps=", index=1),
        lambda clip: bvhplot.render(
            [_rated(clip), clip], "x.gif", backend="matplotlib", fps=10, match_fps="lowest"
        ),
        id="render-match_fps",
    ),
    *(
        pytest.param(
            _refusal(operation, alternative=_DURATIONS_OFF), call, id=f"{param.id}-durations"
        )
        for param in DURATION_OPERATIONS
        for operation, call in [param.values]
    ),
    *(
        pytest.param(
            _refusal(f"{name}()", in_frames=True),
            lambda clip, name=name: getattr(clip, name)(),
            id=name,
        )
        for name in IN_FRAMES_NAMES
    ),
]


class TestUnsetClipRefused:
    @pytest.mark.parametrize(("message", "call"), RATE_OPERATIONS)
    def test_refuses_with_the_operation_and_the_way_out(
        self, unset, message, call, tmp_path, monkeypatch
    ):
        # Any file a call would write lands in tmp_path.
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match=message):
            call(unset)

    @pytest.mark.parametrize(("operation", "call"), DURATION_OPERATIONS)
    def test_duration_filters_off_detect_contacts(self, unset, operation, call):
        contacts = call(unset, min_contact_duration=0.0, min_gap_duration=0.0)
        assert len(contacts) == unset.frame_count

    @pytest.mark.parametrize("name", IN_FRAMES_NAMES)
    def test_per_second_descriptors_in_frames_return_results(self, walk, unset, name):
        np.testing.assert_array_equal(
            getattr(unset, name)(in_frames=True), getattr(walk, name)(in_frames=True)
        )


class TestWalkingPace:
    def test_a_clip_too_short_to_measure_has_no_pace(self, walk):
        assert np.isnan(walk[0:1].walking_pace())


class TestStrideLength:
    def test_contacts_given_an_unset_clip_has_its_stride_length(self):
        # A stride length is a distance: with contacts given, it needs no rate.
        # The value is what stride_length returned on 82dcc39, before unset
        # clips were refused.
        clip = read_bvh_file(BVH_DIR / "cmu_12_01_walk.bvh")[0:300]
        # The contacts stride_length detects itself on the rated clip.
        contacts = clip.foot_contacts(adaptive=True)
        clip.frame_time = 0
        assert clip.stride_length(contacts=contacts) == pytest.approx(20.542627741776613)
