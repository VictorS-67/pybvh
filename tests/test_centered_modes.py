"""The values of ``centered``, one list for every callable that takes it.

``"world"`` keeps the positions as forward kinematics gives them,
``"first_ground"`` subtracts the first frame's body-anchor position in
the ground plane, and ``"anchor"`` puts the body anchor at the origin on
every frame. The 0.9.0 names, ``"first"`` and ``"skeleton"``, are
refused with no alias (#214).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pytest

import pybvh
from pybvh import analysis, bvhplot, features, read_bvh_file
from pybvh.spatial_coord import frames_to_node_positions

matplotlib.use("Agg")

NEW_VALUES = ("world", "first_ground", "anchor")
OLD_VALUES = ("skeleton", "first")
# The message lists the accepted values, in this order.
ACCEPTED = r"\['world', 'first_ground', 'anchor'\]"

BVH_DATA = Path(__file__).parent.parent / "bvh_data"


@pytest.fixture(scope="module")
def clip():
    return read_bvh_file(BVH_DATA / "bvh_test1.bvh")


# Each public callable that takes ``centered`` and computes positions or a
# quantity from them, called on a clip with only ``centered`` varying.
COMPUTING = {
    "Bvh.node_positions": lambda b, c: b.node_positions(centered=c),
    "Bvh.node_positions(frame=)": lambda b, c: b.node_positions(frame=3, centered=c),
    "Bvh.joint_positions": lambda b, c: b.joint_positions(centered=c),
    "Bvh.to_df_dict": lambda b, c: b.to_df_dict(mode="coordinates", centered=c),
    "Bvh.node_velocities": lambda b, c: b.node_velocities(centered=c),
    "Bvh.joint_velocities": lambda b, c: b.joint_velocities(centered=c),
    "Bvh.node_accelerations": lambda b, c: b.node_accelerations(centered=c),
    "Bvh.joint_accelerations": lambda b, c: b.joint_accelerations(centered=c),
    "Bvh.node_speed_derivative": lambda b, c: b.node_speed_derivative(centered=c),
    "Bvh.joint_speed_derivative": lambda b, c: b.joint_speed_derivative(centered=c),
    "Bvh.node_jerk": lambda b, c: b.node_jerk(centered=c),
    "Bvh.joint_jerk": lambda b, c: b.joint_jerk(centered=c),
    "Bvh.kinetic_energy": lambda b, c: b.kinetic_energy(centered=c),
    "Bvh.to_feature_array": lambda b, c: b.to_feature_array(centered=c),
    "analysis.node_velocities": lambda b, c: analysis.node_velocities(b, centered=c),
    "analysis.joint_velocities": lambda b, c: analysis.joint_velocities(b, centered=c),
    "analysis.node_accelerations": lambda b, c: analysis.node_accelerations(b, centered=c),
    "analysis.joint_accelerations": lambda b, c: analysis.joint_accelerations(b, centered=c),
    "analysis.node_speed_derivative": lambda b, c: analysis.node_speed_derivative(b, centered=c),
    "analysis.joint_speed_derivative": lambda b, c: analysis.joint_speed_derivative(b, centered=c),
    "analysis.node_jerk": lambda b, c: analysis.node_jerk(b, centered=c),
    "analysis.joint_jerk": lambda b, c: analysis.joint_jerk(b, centered=c),
    "analysis.kinetic_energy": lambda b, c: analysis.kinetic_energy(b, centered=c),
    "features.to_feature_array": lambda b, c: features.to_feature_array(b, centered=c),
    "frames_to_node_positions": lambda b, c: frames_to_node_positions(b, centered=c),
}

# The bvhplot entry points, drawn on the matplotlib backend without a window.
DRAWING = {
    "bvhplot.frame": lambda b, c, tmp: bvhplot.frame(b, 0, centered=c, show=False),
    "bvhplot.sequence": lambda b, c, tmp: bvhplot.sequence(b, centered=c, n_poses=2, show=False),
    "bvhplot.render": lambda b, c, tmp: bvhplot.render(
        b[:2], tmp / "out.gif", centered=c, backend="matplotlib"
    ),
    "bvhplot.play": lambda b, c, tmp: bvhplot.play(b[:2], centered=c, backend="matplotlib"),
    "bvhplot.trajectory": lambda b, c, tmp: bvhplot.trajectory(b, centered=c, show=False),
}


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


@pytest.mark.parametrize("value", NEW_VALUES)
@pytest.mark.parametrize("name", COMPUTING)
def test_a_computing_callable_accepts_the_new_values(clip, name, value):
    result = COMPUTING[name](clip, value)
    values = result.values() if isinstance(result, dict) else [result]
    assert all(np.all(np.isfinite(v)) for v in values)


@pytest.mark.parametrize("value", OLD_VALUES)
@pytest.mark.parametrize("name", COMPUTING)
def test_a_computing_callable_refuses_the_old_values(clip, name, value):
    with pytest.raises(ValueError, match=ACCEPTED):
        COMPUTING[name](clip, value)


@pytest.mark.parametrize("value", NEW_VALUES)
@pytest.mark.parametrize("name", ["bvhplot.frame", "bvhplot.sequence", "bvhplot.trajectory"])
def test_a_drawing_callable_accepts_the_new_values(clip, name, value, tmp_path):
    DRAWING[name](clip, value, tmp_path)


@pytest.mark.parametrize("value", OLD_VALUES)
@pytest.mark.parametrize("name", DRAWING)
def test_a_drawing_callable_refuses_the_old_values(clip, name, value, tmp_path):
    with pytest.raises(ValueError, match=ACCEPTED):
        DRAWING[name](clip, value, tmp_path)


def test_anchor_puts_the_body_anchor_at_the_origin(clip):
    positions = clip.node_positions(centered="anchor")
    anchor = clip.nodes.index(clip.body_anchor)
    np.testing.assert_array_equal(positions[:, anchor], 0.0)


def test_first_ground_keeps_the_height_and_zeroes_the_ground_plane(clip):
    world = clip.node_positions(centered="world")
    first_ground = clip.node_positions(centered="first_ground")
    anchor = clip.nodes.index(clip.body_anchor)
    up = pybvh.tools._axis_to_vector(clip.world_up).astype(bool)
    np.testing.assert_array_equal(first_ground[..., up], world[..., up])
    np.testing.assert_allclose(first_ground[0, anchor, ~up], 0.0, atol=1e-12)
