"""Guard of the motion list: every frame operation carries every motion array.

``Bvh`` keeps one private list of its motion arrays, and each operation that
carries the arrays as they are (shape checks, derived clips, pickles, copies,
slicing, joins, splices, equality) or interpolates them (resampling, dropped
frames) reads them from it. These tests run each operation once over the list
as it is and once with a test-only array added to it, so that an
array added to the list and missed by an operation fails here. The last class
runs every operation on a clip with one array of the list a frame short.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from pybvh import frames_to_node_positions, nodes_from_table, transforms
from pybvh.bvh import Bvh, _MotionArray

FRAMES = 6

# A test-only motion array: two linear values per frame, no skeleton dim.
EXTRA = _MotionArray(name="extra", frame_shape=(2,), interpolation="linear")


@pytest.fixture(params=["as_is", "with_extra"])
def motion_list(request, monkeypatch):
    """The motion list the test runs over: as it is, or with ``EXTRA`` added."""
    if request.param == "with_extra":
        monkeypatch.setattr(Bvh, "_MOTION_ARRAYS", (*Bvh._MOTION_ARRAYS, EXTRA))
    return Bvh._MOTION_ARRAYS


def _skeleton_table():
    # The spine translates, so that every skeleton dim of the list has a
    # length above 0 on this skeleton.
    return [
        {"name": "Hips", "parent": None, "offset": [0, 0, 0], "rot_channels": "ZYX"},
        {
            "name": "Spine",
            "parent": 0,
            "offset": [0, 1, 0],
            "pos_channels": "XYZ",
            "rot_channels": "XYZ",
        },
        {"name": "Head", "parent": 1, "offset": [0, 1, 0]},
    ]


def _frame_shape(entry):
    """The entry's shape after the frame axis on the skeleton above."""
    nodes = nodes_from_table(_skeleton_table())
    return tuple(dim if isinstance(dim, int) else dim.count(nodes) for dim in entry.frame_shape)


def _tagged_motion(motion_list, first=0.0):
    """One array per entry, each value telling its array, frame and component.

    Frame ``f`` of the ``k``-th array holds ``first + 100 k + f`` plus a
    fraction for the component, so a frame in the wrong place, a frame
    taken from another array or an array left behind changes some value.
    """
    motion = {}
    for k, entry in enumerate(motion_list):
        shape = (FRAMES, *_frame_shape(entry))
        components = np.arange(int(np.prod(shape[1:]))).reshape(shape[1:]) / 1000
        frames = np.arange(FRAMES).reshape((FRAMES,) + (1,) * (len(shape) - 1))
        motion[entry.name] = first + 100 * k + frames + components
    return motion


def _clip(motion, frame_time=0.1):
    """A clip on the two-joint skeleton, holding *motion*."""
    empty = Bvh(nodes=nodes_from_table(_skeleton_table()), frame_time=frame_time)
    return empty._derive_clip(motion)


def _assert_holds(clip, motion):
    """Each array of *clip* equals the one *motion* names."""
    assert clip._motion().keys() == motion.keys()
    for name, expected in motion.items():
        np.testing.assert_array_equal(clip._motion()[name], expected, err_msg=name)


class TestShapeChecks:
    def test_an_empty_clip_holds_no_frame_of_any_array(self, motion_list):
        clip = Bvh(nodes=nodes_from_table(_skeleton_table()))
        # No frame, and no joint: the shapes Bvh() has always given.
        expected = {
            "root_pos": (0, 3),
            "joint_angles": (0, 0, 3),
            "joint_translations": (0, 0, 3),
            "extra": (0, 2),
        }
        for entry in motion_list:
            assert clip._motion()[entry.name].shape == expected[entry.name]

    def test_a_clip_holds_every_array_it_is_built_with(self, motion_list):
        motion = _tagged_motion(motion_list)
        _assert_holds(_clip(motion), motion)

    @pytest.mark.parametrize("position", [0, -1], ids=["first", "last"])
    def test_a_wrong_shape_is_refused(self, motion_list, position):
        entry = motion_list[position]
        motion = _tagged_motion(motion_list)
        motion[entry.name] = np.zeros((FRAMES, 5))
        with pytest.raises(ValueError, match=f"{entry.name} must have shape"):
            _clip(motion)

    def test_a_wrong_frame_count_is_refused(self, motion_list):
        motion = _tagged_motion(motion_list)
        last = motion_list[-1].name
        motion[last] = motion[last][:-1]
        with pytest.raises(ValueError, match="disagree on frame count"):
            _clip(motion)

    def test_set_motion_replaces_each_array(self, motion_list):
        clip = _clip(_tagged_motion(motion_list))
        replacement = _tagged_motion(motion_list, first=0.5)
        clip._set_motion(**replacement)
        _assert_holds(clip, replacement)

    def test_set_motion_refuses_an_array_not_in_the_list(self, motion_list):
        clip = _clip(_tagged_motion(motion_list))
        with pytest.raises(TypeError, match="no motion array named \\['root_pso'\\]"):
            clip._set_motion(root_pso=np.zeros((FRAMES, 3)))

    def test_set_motion_refuses_a_wrong_shape(self, motion_list):
        clip = _clip(_tagged_motion(motion_list))
        name = motion_list[-1].name
        with pytest.raises(ValueError, match=f"{name} must have shape"):
            clip._set_motion(**{name: np.zeros((FRAMES, 5, 5, 5))})


class TestCarryingOperations:
    def test_a_derived_clip_holds_the_arrays_it_is_given(self, motion_list):
        source = _clip(_tagged_motion(motion_list))
        motion = _tagged_motion(motion_list, first=0.25)
        _assert_holds(source._derive_clip(motion), motion)

    def test_a_derived_clip_refuses_a_missing_array(self, motion_list):
        source = _clip(_tagged_motion(motion_list))
        for entry in motion_list:
            motion = _tagged_motion(motion_list, first=0.25)
            del motion[entry.name]
            with pytest.raises(TypeError, match=f"missing motion arrays \\['{entry.name}'\\]"):
                source._derive_clip(motion)

    def test_a_derived_clip_refuses_an_array_not_in_the_list(self, motion_list):
        source = _clip(_tagged_motion(motion_list))
        motion = {**_tagged_motion(motion_list), "root_pso": np.zeros((FRAMES, 3))}
        with pytest.raises(TypeError, match="no motion array named \\['root_pso'\\]"):
            source._derive_clip(motion)

    def test_a_pickle_round_trip_keeps_every_array(self, motion_list):
        motion = _tagged_motion(motion_list)
        _assert_holds(pickle.loads(pickle.dumps(_clip(motion))), motion)

    def test_a_copy_keeps_every_array(self, motion_list):
        motion = _tagged_motion(motion_list)
        _assert_holds(_clip(motion).copy(), motion)

    def test_a_copy_on_a_new_skeleton_keeps_every_array(self, motion_list):
        motion = _tagged_motion(motion_list)
        _assert_holds(_clip(motion).copy(nodes=_skeleton_table()), motion)

    def test_a_copy_owns_its_arrays(self, motion_list):
        motion = _tagged_motion(motion_list)
        source = _clip(motion)
        duplicate = source.copy()
        for name in motion:
            assert not np.shares_memory(duplicate._motion()[name], source._motion()[name])

    @pytest.mark.parametrize(
        ("key", "frames"),
        [
            (slice(1, 4), slice(1, 4)),
            (slice(0, None, 2), slice(0, None, 2)),
            (slice(None, None, -1), slice(None, None, -1)),
            # An integer keeps its frame as a one-frame clip.
            (3, slice(3, 4)),
            (-1, slice(-1, None)),
        ],
        ids=["step_1", "step_2", "reversed", "integer", "negative_integer"],
    )
    def test_slicing_keeps_the_selected_frames_of_every_array(self, motion_list, key, frames):
        motion = _tagged_motion(motion_list)
        expected = {name: array[frames] for name, array in motion.items()}
        _assert_holds(_clip(motion)[key], expected)

    def test_joining_appends_the_frames_of_every_array(self, motion_list):
        first = _tagged_motion(motion_list)
        second = _tagged_motion(motion_list, first=0.5)
        expected = {name: np.concatenate([first[name], second[name]]) for name in first}
        _assert_holds(_clip(first) + _clip(second), expected)

    def test_joining_in_place_appends_the_frames_of_every_array(self, motion_list):
        first = _tagged_motion(motion_list)
        second = _tagged_motion(motion_list, first=0.5)
        expected = {name: np.concatenate([first[name], second[name]]) for name in first}
        clip = _clip(first)
        clip += _clip(second)
        _assert_holds(clip, expected)

    def test_splicing_writes_the_frames_of_every_array(self, motion_list):
        target = _tagged_motion(motion_list)
        source = _tagged_motion(motion_list, first=0.5)
        expected = {name: array.copy() for name, array in target.items()}
        for name in expected:
            expected[name][2:4] = source[name][0:2]
        clip = _clip(target)
        clip[2:4] = _clip(source)[0:2]
        _assert_holds(clip, expected)

    def test_equality_sees_a_difference_in_any_array(self, motion_list):
        motion = _tagged_motion(motion_list)
        for entry in motion_list:
            changed = {name: array.copy() for name, array in motion.items()}
            changed[entry.name][3] += 1.0
            assert _clip(motion) != _clip(changed), entry.name
        assert _clip(motion) == _clip(_tagged_motion(motion_list))


def _square_motion(motion_list):
    """``_tagged_motion`` with each linear array's frame ``f`` at ``f**2``.

    Frame ``f`` of the ``k``-th linear array holds ``100 k + f**2`` plus a
    fraction for the component: linear interpolation between two frames
    gives a value that keeping either frame does not, so an array left out
    of the interpolation changes some value.
    """
    motion = _tagged_motion(motion_list)
    for entry in motion_list:
        if entry.interpolation == "linear":
            frames = np.arange(FRAMES, dtype=float).reshape(
                (FRAMES,) + (1,) * len(entry.frame_shape)
            )
            motion[entry.name] = motion[entry.name] - frames + frames**2
    return motion


def _linear_offsets(motion_list):
    """Each linear array's frame 0: ``100 k`` plus the component fraction."""
    motion = _tagged_motion(motion_list)
    return {
        entry.name: motion[entry.name][0]
        for entry in motion_list
        if entry.interpolation == "linear"
    }


class TestInterpolatingOperations:
    def test_resampling_interpolates_every_linear_array(self, motion_list):
        resampled = _clip(_square_motion(motion_list), frame_time=0.1).resample(20.0)
        # Twice the rate: new frame i sits at old frame i / 2, so an odd i
        # is the mean of the squares on either side.
        squares = np.array([0, 0.5, 1, 2.5, 4, 6.5, 9, 12.5, 16, 20.5, 25])
        for name, offset in _linear_offsets(motion_list).items():
            expected = offset + squares.reshape((-1,) + (1,) * offset.ndim)
            np.testing.assert_allclose(
                resampled._motion()[name], expected, atol=1e-12, err_msg=name
            )

    def test_resampling_gives_every_array_the_new_frame_count(self, motion_list):
        resampled = _clip(_tagged_motion(motion_list), frame_time=0.1).resample(20.0)
        for entry in motion_list:
            assert resampled._motion()[entry.name].shape == (11, *_frame_shape(entry)), entry.name


class _FixedDraws:
    """A stand-in for ``rng`` whose draws are fixed: at ``drop_rate=0.5``,
    frames 1, 2 and 4 of six are dropped and 0, 3 and 5 kept."""

    def random(self, size):
        return np.array([0.9, 0.1, 0.1, 0.9, 0.1, 0.9])


def _drop_with_method(clip):
    return clip.drop_frames(0.5, rng=_FixedDraws())


def _drop_with_method_inplace(clip):
    clip.drop_frames(0.5, rng=_FixedDraws(), inplace=True)
    return clip


def _drop_with_transform(clip):
    return transforms.drop_frames(clip, 0.5, rng=_FixedDraws())


def _drop_with_transform_inplace(clip):
    transforms.drop_frames(clip, 0.5, rng=_FixedDraws(), inplace=True)
    return clip


DROPPERS = [
    _drop_with_method,
    _drop_with_method_inplace,
    _drop_with_transform,
    _drop_with_transform_inplace,
]


class TestDroppedFrames:
    @pytest.mark.parametrize("drop", DROPPERS)
    def test_dropped_frames_of_every_linear_array_are_interpolated(self, motion_list, drop):
        result = drop(_clip(_square_motion(motion_list)))
        # Kept frames 0, 3, 5 hold squares 0, 9, 25; frames 1 and 2 are a
        # third and two thirds of the way from 0 to 9, frame 4 halfway
        # from 9 to 25.
        squares = np.array([0, 3, 6, 9, 17, 25])
        for name, offset in _linear_offsets(motion_list).items():
            expected = offset + squares.reshape((-1,) + (1,) * offset.ndim)
            np.testing.assert_allclose(result._motion()[name], expected, atol=1e-12, err_msg=name)

    @pytest.mark.parametrize("drop", DROPPERS)
    def test_kept_frames_of_every_array_are_untouched(self, motion_list, drop):
        motion = _square_motion(motion_list)
        result = drop(_clip(motion))
        for name, array in motion.items():
            np.testing.assert_array_equal(
                result._motion()[name][[0, 3, 5]], array[[0, 3, 5]], err_msg=name
            )


class TestUnknownInterpolation:
    """An entry whose kind no operation knows is refused, not skipped."""

    @pytest.fixture
    def odd_list(self, monkeypatch):
        odd = _MotionArray(name="odd", frame_shape=(2,), interpolation="cubic")
        monkeypatch.setattr(Bvh, "_MOTION_ARRAYS", (*Bvh._MOTION_ARRAYS, odd))
        return Bvh._MOTION_ARRAYS

    def test_resampling_refuses_it(self, odd_list):
        clip = _clip(_tagged_motion(odd_list), frame_time=0.1)
        with pytest.raises(
            ValueError,
            match=r"resample\(\) does not know interpolation 'cubic' of motion array 'odd'",
        ):
            clip.resample(20.0)

    @pytest.mark.parametrize("drop", DROPPERS)
    def test_dropping_frames_refuses_it(self, odd_list, drop):
        clip = _clip(_tagged_motion(odd_list))
        with pytest.raises(
            ValueError,
            match=r"drop_frames\(\) does not know interpolation 'cubic' of motion array 'odd'",
        ):
            drop(clip)


class TestPickleFormat:
    """The state a pickle holds keeps its keys, so pickles read both ways."""

    def test_a_state_from_before_the_motion_list_unpickles(self):
        # The keys of the state pybvh pickled before the motion list.
        root_pos = np.arange(6.0).reshape(2, 3)
        joint_angles = np.arange(12.0).reshape(2, 2, 3) / 10
        # No joint translated before #186.
        table = _skeleton_table()
        del table[1]["pos_channels"]
        state = {
            "nodes": tuple(nodes_from_table(table)),
            "root_pos": root_pos,
            "joint_angles": joint_angles,
            "frame_time": 0.1,
            "source_path": None,
            "world_up_override": None,
            "world_up_guess": ("+y", "rest_up"),
            "warn_on_world_up_disagreement": True,
            "lr_mapping_setting": "names",
        }
        clip = Bvh.__new__(Bvh)
        clip.__setstate__(state)
        np.testing.assert_array_equal(clip.root_pos, root_pos)
        np.testing.assert_array_equal(clip.joint_angles, joint_angles)
        # Pickled before any joint translated: no translating joint.
        assert clip.joint_translations.shape == (2, 0, 3)

    def test_a_state_keeps_the_keys_earlier_versions_read(self):
        clip = _clip(_tagged_motion(Bvh._MOTION_ARRAYS))
        # Earlier versions read their keys and ignore joint_translations and
        # body_anchor_setting.
        assert set(clip.__getstate__()) == {
            "joint_translations",
            "body_anchor_setting",
            "nodes",
            "root_pos",
            "joint_angles",
            "frame_time",
            "source_path",
            "world_up_override",
            "world_up_guess",
            "warn_on_world_up_disagreement",
            "lr_mapping_setting",
        }


# --- A clip whose arrays disagree on frame count -----------------------------
#
# The setters check no frame count, so that a user replaces the arrays one at
# a time; every operation that reads the motion then refuses a clip whose
# arrays disagree, with the motion list's ``ValueError``, instead of
# broadcasting one array across the frames of another.


def _disagreeing_clip(entry):
    """``_clip`` over the tagged motion, with *entry*'s array one frame short.

    Written to the attribute, as the array's setter writes it, so that the
    test-only array is shortened as the others are. World up is set, so
    that no operation raises through the world-up guess, which runs
    forward kinematics.
    """
    clip = _clip(_tagged_motion(Bvh._MOTION_ARRAYS))
    clip.world_up = "+y"
    setattr(clip, entry.attribute, getattr(clip, entry.attribute)[:-1].copy())
    return clip


def _held_arrays(clip):
    """A copy of each array *clip* holds, read past every check."""
    return {entry.name: getattr(clip, entry.attribute).copy() for entry in Bvh._MOTION_ARRAYS}


def _write(clip, tmp_path):
    clip.write(tmp_path / "clip.bvh")


def _from_rotmat_inplace(clip, tmp_path):
    rotmats = np.tile(np.eye(3), (FRAMES, clip.joint_count, 1, 1))
    clip.from_rotmat(rotmats, inplace=True)


SINGLE_CLIP_OPERATIONS = {
    "copy": lambda clip, _: clip.copy(),
    "copy_on_nodes": lambda clip, _: clip.copy(nodes=_skeleton_table()),
    "pickle": lambda clip, _: pickle.dumps(clip),
    "slice": lambda clip, _: clip[1:3],
    "integer_index": lambda clip, _: clip[0],
    "resample": lambda clip, _: clip.resample(20.0),
    "perturb_speed": lambda clip, _: transforms.perturb_speed(clip, 1.5),
    "drop_frames": lambda clip, _: _drop_with_method(clip),
    "drop_frames_inplace": lambda clip, _: _drop_with_transform_inplace(clip),
    "translate_root_inplace": lambda clip, _: transforms.translate_root(
        clip, [1.0, 0.0, 0.0], inplace=True
    ),
    "add_rotation_noise_inplace": lambda clip, _: transforms.add_rotation_noise(
        clip, 1.0, rng=np.random.default_rng(0), inplace=True
    ),
    "add_position_noise_inplace": lambda clip, _: transforms.add_position_noise(
        clip, 1.0, rng=np.random.default_rng(0), inplace=True
    ),
    "rotate_vertical_inplace": lambda clip, _: transforms.rotate_vertical(clip, 30.0, inplace=True),
    "reorient_world_up_inplace": lambda clip, _: transforms.reorient_world_up(
        clip, "+z", inplace=True
    ),
    "scale_inplace": lambda clip, _: clip.scale(2.0, inplace=True),
    "retarget_inplace": lambda clip, _: clip.retarget(
        Bvh(nodes=nodes_from_table(_skeleton_table())), inplace=True
    ),
    "change_euler_order_inplace": lambda clip, _: clip.change_euler_order("XYZ", inplace=True),
    "from_rotmat_inplace": _from_rotmat_inplace,
    "extract_joints": lambda clip, _: clip.extract_joints(["Hips"]),
    "node_positions": lambda clip, _: clip.node_positions(),
    "node_positions_first_frame": lambda clip, _: clip.node_positions(frame=0),
    "node_positions_last_frame_centered_first": lambda clip, _: clip.node_positions(
        frame=-1, centered="first_ground"
    ),
    "frames_to_node_positions": lambda clip, _: frames_to_node_positions(clip),
    "root_trajectory": lambda clip, _: clip.root_trajectory(),
    "to_df_dict": lambda clip, _: clip.to_df_dict(),
    "write": _write,
    "to_feature_array": lambda clip, _: clip.to_feature_array(),
}


def _join(clip, other):
    return clip + other


def _join_in_place(clip, other):
    clip += other


def _splice_all(clip, other):
    clip[:] = other


TWO_CLIP_OPERATIONS = {
    "join": _join,
    "join_in_place": _join_in_place,
    "splice_all": _splice_all,
}


class TestDisagreeingFrameCounts:
    """Each operation refuses a clip whose arrays disagree, before it changes anything."""

    @pytest.mark.parametrize("operation", list(SINGLE_CLIP_OPERATIONS))
    def test_an_operation_refuses_it(self, motion_list, operation, tmp_path):
        for entry in motion_list:
            clip = _disagreeing_clip(entry)
            before = _held_arrays(clip)
            with pytest.raises(ValueError, match="disagree on frame count"):
                SINGLE_CLIP_OPERATIONS[operation](clip, tmp_path)
            for name, array in _held_arrays(clip).items():
                np.testing.assert_array_equal(array, before[name], err_msg=f"{entry.name}: {name}")

    @pytest.mark.parametrize("side", ["self", "other"])
    @pytest.mark.parametrize("operation", list(TWO_CLIP_OPERATIONS))
    def test_a_two_clip_operation_refuses_it_on_either_side(self, motion_list, operation, side):
        for entry in motion_list:
            disagreeing = _disagreeing_clip(entry)
            agreeing = _clip(_tagged_motion(motion_list, first=0.5))
            agreeing.world_up = "+y"
            clip, other = (disagreeing, agreeing) if side == "self" else (agreeing, disagreeing)
            before = _held_arrays(clip)
            with pytest.raises(ValueError, match="disagree on frame count"):
                TWO_CLIP_OPERATIONS[operation](clip, other)
            for name, array in _held_arrays(clip).items():
                np.testing.assert_array_equal(array, before[name], err_msg=f"{entry.name}: {name}")

    def test_equality_compares_it_without_raising(self, motion_list):
        for entry in motion_list:
            assert _disagreeing_clip(entry) == _disagreeing_clip(entry), entry.name
            assert _disagreeing_clip(entry) != _clip(_tagged_motion(motion_list)), entry.name
