"""The values of ``centered``, one list for every callable that takes it.

``"world"`` keeps the positions as forward kinematics gives them,
``"first_ground"`` subtracts the first frame's body-anchor position in
the ground plane, ``"anchor"`` puts the body anchor at the origin on
every frame, and ``"anchor_ground"`` subtracts each frame's body-anchor
position in the ground plane (#215). The 0.9.0 names, ``"first"`` and ``"skeleton"``, are
refused with no alias (#214).
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib
import numpy as np
import pytest

import pybvh
from pybvh import analysis, bvhplot, features, read_bvh_file, signal
from pybvh.spatial_coord import frames_to_node_positions

matplotlib.use("Agg")

NEW_VALUES = ("world", "first_ground", "anchor", "anchor_ground")
OLD_VALUES = ("skeleton", "first")
# The message lists the accepted values, in this order.
ACCEPTED = r"\['world', 'first_ground', 'anchor', 'anchor_ground'\]"

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


# The test of each mode's result, below, calls not every callable under
# "world": this one does.
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


# ---------------------------------------------------------------------------
# Each mode's output, derived from the same callable under "world"
# ---------------------------------------------------------------------------
#
# A branch that reads a stale value, or falls through to another mode, still
# gives finite output; these tests compare each result with the one its mode
# defines. The expected result is derived from the callable's "world" output
# and the body anchor's world positions, so it needs no stored values:
#
# - positions: "anchor" subtracts the anchor's position on every frame, all
#   three axes; "first_ground" subtracts its first-frame position in the two
#   ground axes only; "anchor_ground" subtracts its position on every frame
#   in the two ground axes only;
# - vector derivatives: "first_ground" equals "world", since a constant offset
#   vanishes; "anchor" subtracts the anchor's own derivative of the same order,
#   and "anchor_ground" that derivative in the two ground axes only;
# - speed derivatives and kinetic energy: "first_ground" equals "world", for
#   the same reason; under "anchor" and "anchor_ground" the expected value is
#   computed from the expected velocities.

# A clip whose body anchor is its root, and one whose anchor (Hips) sits
# below a reference root and starts away from the origin, so that every
# branch of the modes on either kind of clip is reached.
RELATION_CLIPS = {
    "anchor is the root": BVH_DATA / "bvh_test1.bvh",
    "anchor below a reference root": BVH_DATA / "synthetic" / "reference_props_humanoid_6ch.bvh",
}
CENTRED_VALUES = ("first_ground", "anchor", "anchor_ground")
# The frame the single-frame entry of COMPUTING poses.
FRAME = 3


# Some callables take another path once the clip holds its world positions
# (``Bvh.node_positions`` with ``frame=``), so each test reads a new clip,
# and reads it in both states.
CACHE_STATES = ("nothing cached", "world positions cached")


@pytest.fixture(params=list(RELATION_CLIPS))
def relation_clip_path(request):
    return RELATION_CLIPS[request.param]


def _anchor_world(clip):
    """The body anchor's world positions, ``(F, 3)``."""
    return clip.node_positions(centered="world")[:, clip.nodes.index(clip.body_anchor)]


def _ground(clip, position):
    """*position* with its component along the clip's up axis set to zero."""
    ground = np.array(position, dtype=np.float64)
    ground[..., "xyz".index(clip.world_up[-1].lower())] = 0.0
    return ground


def _offset(clip, value):
    """What *value* subtracts from a world position, ``(F, 3)``."""
    anchor = _anchor_world(clip)
    if value == "anchor":
        return anchor
    if value == "anchor_ground":
        return _ground(clip, anchor)
    return np.broadcast_to(_ground(clip, anchor[0]), anchor.shape)


def _joints(clip):
    """The indices, in node order, of the nodes that are not end sites."""
    return [i for i, node in enumerate(clip.nodes) if not node.is_end_site()]


def _anchor_derivative(clip, order):
    return signal.finite_difference(_anchor_world(clip), clip.frame_time, order=order)


def _node_derivative(clip, value, order):
    """The expected ``order``-th derivative of every node under *value*."""
    world = (analysis.node_velocities, analysis.node_accelerations, analysis.node_jerk)[order - 1](
        clip, centered="world"
    )
    if value == "first_ground":
        return world
    anchor = _anchor_derivative(clip, order)
    if value == "anchor_ground":
        anchor = _ground(clip, anchor)
    return world - anchor[:, np.newaxis]


def _node_speed_derivative(clip, value):
    speed = np.linalg.norm(_node_derivative(clip, value, 1), axis=-1)
    return signal.finite_difference(speed, clip.frame_time)


def _kinetic_energy(clip, value):
    velocities = _node_derivative(clip, value, 1)[:, _joints(clip)]
    return np.sum(velocities**2, axis=(-2, -1))


def _to_df_dict(clip, value):
    world = clip.to_df_dict(mode="coordinates", centered="world")
    offset = _offset(clip, value)
    expected = {"time": world["time"]}
    for column, values in world.items():
        if column != "time":
            expected[column] = values - offset[:, "XYZ".index(column[-1])]
    return expected


def _to_feature_array(clip, value):
    # With the defaults, only the root position (the first three columns)
    # depends on ``centered``.
    expected = clip.to_feature_array(centered="world").copy()
    expected[:, :3] -= _offset(clip, value)
    return expected


def _world_positions(clip, value):
    return clip.node_positions(centered="world") - _offset(clip, value)[:, np.newaxis]


# For each name in COMPUTING, its expected output under a centred value.
EXPECTED = {
    "Bvh.node_positions": _world_positions,
    # One frame posed is that frame's row of the whole clip (#215).
    "Bvh.node_positions(frame=)": lambda b, c: _world_positions(b, c)[FRAME],
    "Bvh.joint_positions": lambda b, c: _world_positions(b, c)[:, _joints(b)],
    "Bvh.to_df_dict": _to_df_dict,
    "Bvh.node_velocities": lambda b, c: _node_derivative(b, c, 1),
    "Bvh.joint_velocities": lambda b, c: _node_derivative(b, c, 1)[:, _joints(b)],
    "Bvh.node_accelerations": lambda b, c: _node_derivative(b, c, 2),
    "Bvh.joint_accelerations": lambda b, c: _node_derivative(b, c, 2)[:, _joints(b)],
    "Bvh.node_speed_derivative": _node_speed_derivative,
    "Bvh.joint_speed_derivative": lambda b, c: _node_speed_derivative(b, c)[:, _joints(b)],
    "Bvh.node_jerk": lambda b, c: _node_derivative(b, c, 3),
    "Bvh.joint_jerk": lambda b, c: _node_derivative(b, c, 3)[:, _joints(b)],
    "Bvh.kinetic_energy": _kinetic_energy,
    "Bvh.to_feature_array": _to_feature_array,
    "analysis.node_velocities": lambda b, c: _node_derivative(b, c, 1),
    "analysis.joint_velocities": lambda b, c: _node_derivative(b, c, 1)[:, _joints(b)],
    "analysis.node_accelerations": lambda b, c: _node_derivative(b, c, 2),
    "analysis.joint_accelerations": lambda b, c: _node_derivative(b, c, 2)[:, _joints(b)],
    "analysis.node_speed_derivative": _node_speed_derivative,
    "analysis.joint_speed_derivative": lambda b, c: _node_speed_derivative(b, c)[:, _joints(b)],
    "analysis.node_jerk": lambda b, c: _node_derivative(b, c, 3),
    "analysis.joint_jerk": lambda b, c: _node_derivative(b, c, 3)[:, _joints(b)],
    "analysis.kinetic_energy": _kinetic_energy,
    "features.to_feature_array": _to_feature_array,
    "frames_to_node_positions": _world_positions,
}


def test_every_computing_callable_has_an_expected_output():
    assert EXPECTED.keys() == COMPUTING.keys()


@pytest.mark.parametrize("cache", CACHE_STATES)
@pytest.mark.parametrize("value", CENTRED_VALUES)
@pytest.mark.parametrize("name", COMPUTING)
def test_a_computing_callable_gives_what_its_mode_defines(relation_clip_path, name, value, cache):
    clip = read_bvh_file(relation_clip_path)
    if cache == "world positions cached":
        clip.node_positions()
    actual = COMPUTING[name](clip, value)
    expected = EXPECTED[name](clip, value)
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        pairs = [(actual[k], expected[k]) for k in expected]
    else:
        pairs = [(actual, expected)]
    for got, want in pairs:
        # A world output that is not finite would let a non-finite result
        # match it.
        assert np.all(np.isfinite(want))
        np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-9)


# ---------------------------------------------------------------------------
# "anchor_ground": the travel removed, the height kept (#215)
# ---------------------------------------------------------------------------

SYNTHETIC = BVH_DATA / "synthetic"


def _read_quietly(path):
    """The clip, read without the world-up warnings some corpus clips give."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return read_bvh_file(path)


# rotation_only_control: the hips travel along +z and bob around a height of
# 95 along +y, its world up.
CONTROL_HIPS_HEIGHT = np.array([95.0, 96.2, 95.0, 93.8, 95.0])


def test_anchor_ground_keeps_the_hips_above_the_origin_at_their_height():
    clip = _read_quietly(SYNTHETIC / "rotation_only_control.bvh")
    hips = clip.nodes.index(clip.body_anchor)
    expected = np.zeros((clip.frame_count, 3))
    expected[:, 1] = CONTROL_HIPS_HEIGHT
    np.testing.assert_allclose(
        clip.node_positions(centered="anchor_ground")[:, hips], expected, atol=1e-9
    )
    for frame in range(clip.frame_count):
        np.testing.assert_allclose(
            clip.node_positions(frame=frame, centered="anchor_ground")[hips],
            expected[frame],
            atol=1e-9,
        )


def test_anchor_ground_kinetic_energy_keeps_a_jump_and_drops_the_travel():
    walk = _read_quietly(SYNTHETIC / "rotation_only_control.bvh")
    # The same clip with the hips held above one point: a jump in place.
    jump = walk.copy()
    root_pos = jump.root_pos.copy()
    root_pos[:, [0, 2]] = root_pos[0, [0, 2]]
    jump.root_pos = root_pos
    np.testing.assert_allclose(
        jump.kinetic_energy(centered="anchor_ground"),
        jump.kinetic_energy(centered="world"),
        rtol=1e-9,
        atol=1e-9,
    )
    # "anchor" drops the jump along with the travel.
    assert not np.allclose(
        jump.kinetic_energy(centered="anchor"), jump.kinetic_energy(centered="world")
    )
    # On the walk, "anchor_ground" drops the travel: it gives the jump's energy.
    np.testing.assert_allclose(
        walk.kinetic_energy(centered="anchor_ground"),
        jump.kinetic_energy(centered="world"),
        rtol=1e-9,
        atol=1e-9,
    )


def test_anchor_ground_on_a_topology_centres_on_the_root():
    # An FkTopology carries no L/R mapping to find a body from, so the mode
    # centres on the root, also on a clip whose body is below its root, and
    # on one that has no body.
    for name in ("reference_props_humanoid_6ch.bvh", "two_figures_moving_root.bvh"):
        clip = _read_quietly(SYNTHETIC / name)
        world = frames_to_node_positions(
            clip.fk_topology,
            clip.root_pos,
            clip.joint_angles,
            joint_translations=clip.joint_translations,
        )
        positions = frames_to_node_positions(
            clip.fk_topology,
            clip.root_pos,
            clip.joint_angles,
            centered="anchor_ground",
            world_up="+y",
            joint_translations=clip.joint_translations,
        )
        expected = world - np.asarray(clip.root_pos * [1.0, 0.0, 1.0])[:, np.newaxis]
        np.testing.assert_allclose(positions, expected, atol=1e-9)


def test_anchor_ground_on_a_topology_needs_a_world_up():
    clip = read_bvh_file(BVH_DATA / "bvh_test1.bvh")
    with pytest.raises(ValueError, match="centered='anchor_ground' needs an up axis"):
        frames_to_node_positions(
            clip.fk_topology, clip.root_pos, clip.joint_angles, centered="anchor_ground"
        )


@pytest.mark.parametrize("value", ["world", "first_ground", "anchor_ground"])
def test_a_mode_that_keeps_the_height_draws_the_clip_s_floor(value, monkeypatch):
    # The heights stay world heights, so the drawn floor is the clip's own
    # (Bvh.floor_height), not the lowest point of the pose drawn.
    import pybvh.bvhplot._matplotlib as mpl_backend

    scenes = []
    monkeypatch.setattr(
        mpl_backend,
        "frame_mpl",
        lambda scene, style, **kwargs: scenes.append(scene) or (None, None),
    )
    clip = _read_quietly(SYNTHETIC / "rotation_only_control.bvh")
    bvhplot.frame(clip, 1, centered=value)
    (scene,) = scenes
    assert scene.views[0].floor_height == clip.floor_height
    assert clip.floor_height != scene.views[0].coords[..., 1].min()


# ---------------------------------------------------------------------------
# One frame posed is that frame's row of the whole clip (#215)
# ---------------------------------------------------------------------------

BUNDLED_CLIPS = sorted(BVH_DATA.rglob("*.bvh"))


@pytest.mark.parametrize("cache", CACHE_STATES)
@pytest.mark.parametrize("path", BUNDLED_CLIPS, ids=lambda p: p.name)
def test_a_frame_posed_alone_is_its_row_of_the_whole_clip(path, cache):
    clip = _read_quietly(path)
    if clip.frame_count == 0:
        pytest.skip("a clip with no frames has no row to pose")
    frames = (0, clip.frame_count // 2, clip.frame_count - 1, -1)
    if cache == "world positions cached":
        clip.node_positions()
    for value in NEW_VALUES:
        # A clip whose world positions are not cached yet poses one frame
        # with no whole-clip FK: read the rows from a copy.
        try:
            whole = clip.copy().node_positions(centered=value)
        except ValueError as refusal:
            # A clip with no body refuses a centred mode: one frame too.
            for frame in frames:
                with pytest.raises(ValueError, match=re.escape(str(refusal))):
                    clip.node_positions(frame=frame, centered=value)
            continue
        for frame in frames:
            np.testing.assert_allclose(
                clip.node_positions(frame=frame, centered=value),
                whole[frame],
                rtol=1e-9,
                atol=1e-9,
                err_msg=f"centered={value!r}, frame={frame}",
            )


def test_frame_draws_row_k_of_the_whole_clip_under_first_ground(monkeypatch):
    # The hips of rotation_only_control travel along +z: on its last frame,
    # "first_ground" keeps them at their travel from frame 0.
    import pybvh.bvhplot._matplotlib as mpl_backend

    scenes = []
    monkeypatch.setattr(
        mpl_backend,
        "frame_mpl",
        lambda scene, style, **kwargs: scenes.append(scene) or (None, None),
    )
    clip = _read_quietly(SYNTHETIC / "rotation_only_control.bvh")
    last = clip.frame_count - 1
    bvhplot.frame(clip, last, centered="first_ground")
    (scene,) = scenes
    drawn = scene.views[0].coords
    np.testing.assert_allclose(drawn, clip.node_positions(centered="first_ground")[last : last + 1])
    hips = clip.nodes.index(clip.body_anchor)
    assert drawn[0, hips, 2] != pytest.approx(0.0)
