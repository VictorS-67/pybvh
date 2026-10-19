"""Shared fixtures for the pybvh test suite."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

import pybvh.analysis
import pybvh.bvh


@dataclass
class Computations:
    """How many times each derived value of a clip was computed.

    Attributes
    ----------
    fk : int
        Full-clip forward kinematics (``frames_to_node_positions`` called
        by ``Bvh`` with a 2-D ``root_pos``). The single-frame FK that the
        world-up inference runs is not counted.
    world_up : int
        Every call of ``_infer_world_up`` made by ``Bvh``.
    floor_height : int
        Every call of ``analysis._compute_floor_height``.
    rest_pose : int
        Every call of ``_build_rest_pose`` made by ``Bvh``.
    """

    fk: int = 0
    world_up: int = 0
    floor_height: int = 0
    rest_pose: int = 0


@pytest.fixture
def computations(monkeypatch):
    """Count the computations of a clip's derived values (ADR 0003).

    Tests read the counts to show that a value is computed once, computed
    again after the change that should clear it, and left alone by the
    changes that should not.
    """
    counts = Computations()

    real_fk = pybvh.bvh.frames_to_node_positions
    real_infer = pybvh.bvh._infer_world_up
    real_floor = pybvh.analysis._compute_floor_height
    real_rest = pybvh.bvh._build_rest_pose

    def counted_fk(*args, **kwargs):
        if np.ndim(kwargs.get("root_pos")) == 2:
            counts.fk += 1
        return real_fk(*args, **kwargs)

    def counted_infer(*args, **kwargs):
        counts.world_up += 1
        return real_infer(*args, **kwargs)

    def counted_floor(*args, **kwargs):
        counts.floor_height += 1
        return real_floor(*args, **kwargs)

    def counted_rest(*args, **kwargs):
        counts.rest_pose += 1
        return real_rest(*args, **kwargs)

    monkeypatch.setattr(pybvh.bvh, "frames_to_node_positions", counted_fk)
    monkeypatch.setattr(pybvh.bvh, "_infer_world_up", counted_infer)
    monkeypatch.setattr(pybvh.analysis, "_compute_floor_height", counted_floor)
    monkeypatch.setattr(pybvh.bvh, "_build_rest_pose", counted_rest)
    return counts
