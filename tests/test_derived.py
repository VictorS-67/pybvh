"""Tests for the lifecycle of a clip's derived state (`pybvh._derived`).

The module holds every value a `Bvh` computes from its skeleton, motion,
world up or L/R mapping setting, and one method per change to those inputs. Each change must clear
exactly the values computed from what changed, and nothing else: a value left
in place after its input changed is stale, and a value cleared needlessly is a
recomputation every caller pays for. The expected rows below are ADR 0003's
table, with ADR 0004's row for the orientation report.
"""

from __future__ import annotations

import numpy as np
import pytest

from pybvh._derived import DerivedState

ALL_VALUES = {
    "node_positions",
    "rest_pose_positions",
    "floor_height",
    "inferred_world_up",
    "inherited_world_up",
    "world_up_guess_announced",
    "node_index",
    "joint_index",
    "names_lr_mapping",
    "orientation_report",
}
MOTION_DEPENDENTS = {"node_positions", "floor_height"}
# The guessed world up, inferred on the clip or inherited from the clip it
# was derived from, and the guess world_up warned about, survive every
# change but the user replacing the motion.
GUESSED_WORLD_UP = {"inferred_world_up", "inherited_world_up", "world_up_guess_announced"}
# But the user setting the L/R mapping starts the clip's own guess again
# (rest up, which it may be read from, reads the mapping), and with it the
# floor measured along it; not the inherited guess or the guess warned about.
GUESSED_HERE = {"inferred_world_up", "floor_height"}

# The orientation report reads the skeleton, world up and the L/R mapping
# setting, and none of pybvh's own motion writes (ADR 0004).
REPORT = {"orientation_report"}

CLEARED_BY = {
    "skeleton_changed": ALL_VALUES - GUESSED_WORLD_UP,
    "motion_changed": MOTION_DEPENDENTS,
    "user_replaced_motion": MOTION_DEPENDENTS | GUESSED_WORLD_UP | REPORT,
    "world_up_changed": {"floor_height"} | REPORT,
    "lr_mapping_changed": GUESSED_HERE | REPORT,
}


def _filled() -> DerivedState:
    derived = DerivedState()
    derived.node_positions = np.zeros((2, 3, 3))
    derived.rest_pose_positions = np.zeros((3, 3))
    derived.floor_height = 1.5
    derived.inferred_world_up = "+y"
    derived.inherited_world_up = "+z"
    derived.world_up_guess_announced = "+y"
    derived.node_index = {"Hips": 0, "Spine": 1, "EndSite": 2}
    derived.joint_index = {"Hips": 0, "Spine": 1}
    derived.names_lr_mapping = {"LeftArm": "RightArm"}
    derived.orientation_report = object()  # type: ignore[assignment]
    return derived


def test_a_new_state_holds_no_value():
    derived = DerivedState()
    for name in ALL_VALUES:
        assert getattr(derived, name) is None, name


@pytest.mark.parametrize("change", sorted(CLEARED_BY))
def test_a_change_clears_exactly_its_row(change):
    derived = _filled()
    kept = {name: getattr(derived, name) for name in ALL_VALUES - CLEARED_BY[change]}

    getattr(derived, change)()

    for name in CLEARED_BY[change]:
        assert getattr(derived, name) is None, name
    for name, value in kept.items():
        assert getattr(derived, name) is value, name


@pytest.mark.parametrize("name", ["node_positions", "rest_pose_positions"])
def test_stored_positions_are_read_only(name):
    derived = DerivedState()
    setattr(derived, name, np.zeros((2, 3)))
    with pytest.raises(ValueError, match="read-only"):
        getattr(derived, name)[0, 0] = 1.0
