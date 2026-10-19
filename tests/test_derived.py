"""Tests for the lifecycle of a clip's derived state (`pybvh._derived`).

The module holds every value a `Bvh` computes from its skeleton, motion or
world up, and one method per change to those inputs. Each change must clear
exactly the values computed from what changed, and nothing else: a value left
in place after its input changed is stale, and a value cleared needlessly is a
recomputation every caller pays for. The expected rows below are ADR 0003's
table as it stands until the guessed world up is inherited (#112).
"""

from __future__ import annotations

import numpy as np
import pytest

from pybvh._derived import DerivedState

ALL_VALUES = {"node_positions", "floor_height", "guessed_world_up"}
MOTION_DEPENDENTS = {"node_positions", "guessed_world_up", "floor_height"}

CLEARED_BY = {
    "skeleton_changed": ALL_VALUES,
    "motion_changed": MOTION_DEPENDENTS,
    "user_replaced_motion": MOTION_DEPENDENTS,
}


def _filled() -> DerivedState:
    derived = DerivedState()
    derived.node_positions = np.zeros((2, 3, 3))
    derived.floor_height = 1.5
    derived.guessed_world_up = "+y"
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


def test_stored_positions_are_read_only():
    derived = DerivedState()
    derived.node_positions = np.zeros((2, 3, 3))
    with pytest.raises(ValueError, match="read-only"):
        derived.node_positions[0, 0, 0] = 1.0
