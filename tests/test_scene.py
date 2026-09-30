"""Tests for the Scene/SkeletonView container (bvhplot Phase 0)."""
from __future__ import annotations

import ast
import copy
import dataclasses
import pathlib
import pickle

import numpy as np
import pytest

from pybvh import bvhplot, read_bvh_file
from pybvh.analysis import root_trajectory
from pybvh.bvhplot._from_bvh import (
    make_scene,
    get_skeleton_lines,
    get_bone_chains,
    get_camera_angles,
)
from pybvh.bvhplot._scene import Scene, SkeletonView
from pybvh.tools import _resolve_lr_pairs
from synthetic_bvh import make_nameless_lr_bvh
from synthetic_scene import make_array_scene, make_array_view

BVH_PATH = "bvh_data/cmu_12_01_walk.bvh"


@pytest.fixture(scope="module")
def bvh():
    return read_bvh_file(BVH_PATH)


@pytest.fixture(scope="module")
def coords(bvh):
    return bvh.node_positions()


class TestMakeScene:
    def test_single_skeleton(self, bvh, coords):
        scene = make_scene([bvh], [coords], "front", None)
        assert isinstance(scene, Scene)
        assert scene.num_skeletons == 1
        assert scene.num_frames == coords.shape[0]
        view = scene.views[0]
        np.testing.assert_array_equal(view.coords, coords)
        assert np.shares_memory(view.coords, coords)  # borrowed, not copied
        assert view.label is None

    def test_view_fields_match_helpers(self, bvh, coords):
        """Scene assembly must agree with the individual helpers it wraps."""
        scene = make_scene([bvh], [coords], "front", None)
        view = scene.views[0]

        assert view.bones == get_skeleton_lines(bvh)

        az, el, up = get_camera_angles(bvh, coords[0], "front")
        assert view.azimuth == az
        assert view.elevation == el
        assert view.up_axis == up

    def test_labels_assigned_per_view(self, bvh, coords):
        scene = make_scene([bvh, bvh], [coords, coords], "front", ["a", "b"])
        assert [v.label for v in scene.views] == ["a", "b"]

    def test_missing_labels_are_none(self, bvh, coords):
        scene = make_scene([bvh, bvh], [coords, coords], "front", ["only"])
        assert [v.label for v in scene.views] == ["only", None]

    def test_camera_tuple_passthrough(self, bvh, coords):
        scene = make_scene([bvh], [coords], (33.0, 12.0), None)
        assert scene.views[0].azimuth == 33.0
        assert scene.views[0].elevation == 12.0


class TestViewCarriesSkeletonFacts:
    """A view holds every skeleton fact a backend draws from, so the
    backends never need the Bvh it was built from."""

    def test_timing_names_and_rest_pose(self, bvh, coords):
        view = make_scene([bvh], [coords], "front", None).views[0]
        assert view.frame_time == bvh.frame_time
        assert view.node_names == [n.name for n in bvh.nodes]
        np.testing.assert_allclose(view.rest_coords, bvh.rest_pose_positions())
        assert view.rest_coords.shape == (len(bvh.nodes), 3)

    def test_orientation_facts(self, bvh, coords):
        view = make_scene([bvh], [coords], "front", None).views[0]
        assert view.up == bvh.world_up
        assert view.up_index == bvh.up_axis.index
        assert view.up_sign == bvh.up_axis.sign
        np.testing.assert_array_equal(view.up_vector, bvh.up_axis.vector)
        assert view.forward_axis == bvh.forward_at(0)

    def test_a_view_states_its_up_axis_once(self):
        """One signed string is the field; the letter, column, sign and
        vector are read from it, so no view can hold two up axes."""
        names = {f.name for f in dataclasses.fields(SkeletonView)}
        assert "up" in names
        assert not names & {"up_axis", "up_index", "up_sign", "up_vector"}

    @pytest.mark.parametrize("up, letter, index, sign, vector", [
        ("+x", "x", 0, 1.0, [1.0, 0.0, 0.0]),
        ("-x", "x", 0, -1.0, [-1.0, 0.0, 0.0]),
        ("+y", "y", 1, 1.0, [0.0, 1.0, 0.0]),
        ("-y", "y", 1, -1.0, [0.0, -1.0, 0.0]),
        ("+z", "z", 2, 1.0, [0.0, 0.0, 1.0]),
        ("-z", "z", 2, -1.0, [0.0, 0.0, -1.0]),
    ])
    def test_up_forms_are_derived_from_the_signed_string(
            self, up, letter, index, sign, vector):
        forward = "+x" if letter != "x" else "+y"
        view = dataclasses.replace(
            make_array_view(), up=up, forward_axis=forward)
        assert view.up_axis == letter
        assert view.up_index == index
        assert view.up_sign == sign
        assert view.up_vector.dtype == np.float64
        np.testing.assert_array_equal(view.up_vector, vector)

    def test_up_vector_is_a_fresh_array(self):
        view = make_array_view()
        view.up_vector[:] = 7.0
        np.testing.assert_array_equal(view.up_vector, [0.0, 1.0, 0.0])

    def test_below_floor_follows_the_sign_of_up(self):
        view = dataclasses.replace(make_array_view(), floor_height=2.0)
        assert view.below_floor(0.5) == 1.5
        flipped = dataclasses.replace(view, up="-y")
        assert flipped.below_floor(0.5) == 2.5

    def test_lr_pairs_are_the_facing_geometrys_joint_pairs(self, bvh, coords):
        """The pairs the follow camera averages: joints only, resolved the
        way tools resolves them, so follow azimuths match the Bvh path."""
        view = make_scene([bvh], [coords], "front", None).views[0]
        expected = _resolve_lr_pairs(bvh.lr_mapping, bvh.node_index)
        assert view.lr_pairs.dtype == np.intp
        assert view.lr_pairs.tolist() == [list(p) for p in expected]
        assert len(expected) > 0
        for left, right in view.lr_pairs:
            assert not bvh.nodes[left].is_end_site()
            assert not bvh.nodes[right].is_end_site()

    def test_lr_pairs_empty_when_rig_has_none(self):
        rig = make_nameless_lr_bvh()
        assert rig.node_lr_pairs is None
        view = make_scene([rig], [rig.node_positions()], "front", None).views[0]
        assert view.lr_pairs.shape == (0, 2)

    def test_bone_chains_parallel_to_bones(self, bvh, coords):
        view = make_scene([bvh], [coords], "front", None).views[0]
        assert len(view.bone_chains) == len(view.bones)
        chains = get_bone_chains(bvh)
        for chain_name, bone_indices in chains.items():
            for i in bone_indices:
                assert view.bone_chains[i] == chain_name
        claimed = {i for idxs in chains.values() for i in idxs}
        for i in range(len(view.bones)):
            if i not in claimed:
                assert view.bone_chains[i] == "spine"

    def test_root_heading_is_root_trajectory_heading(self, bvh, coords):
        view = make_scene([bvh], [coords], "front", None,
                          clip_frames=slice(None)).views[0]
        np.testing.assert_allclose(
            view.root_heading, root_trajectory(bvh)[:, 2:4])

    def test_root_heading_follows_truncated_coords(self, bvh, coords):
        short = coords[:40]
        view = make_scene([bvh], [short], "front", None,
                          clip_frames=slice(None)).views[0]
        assert view.root_heading.shape == (40, 2)
        np.testing.assert_allclose(
            view.root_heading, root_trajectory(bvh)[:40, 2:4])

    def test_root_heading_follows_padded_coords(self, bvh, coords):
        extra = 7
        padded = np.concatenate(
            [coords, np.repeat(coords[-1:], extra, axis=0)], axis=0)
        view = make_scene([bvh], [padded], "front", None,
                          clip_frames=slice(None)).views[0]
        heading = root_trajectory(bvh)[:, 2:4]
        assert view.root_heading.shape == (coords.shape[0] + extra, 2)
        np.testing.assert_allclose(view.root_heading[:-extra], heading)
        np.testing.assert_allclose(
            view.root_heading[-extra:], np.repeat(heading[-1:], extra, axis=0))

    def test_root_heading_for_a_single_frame(self, bvh, coords):
        one = coords[-1:]
        view = make_scene([bvh], [one], "front", None,
                          clip_frames=-1).views[0]
        assert view.root_heading.shape == (1, 2)
        np.testing.assert_allclose(
            view.root_heading[0], root_trajectory(bvh)[-1, 2:4])

    def test_root_heading_follows_a_slice_of_the_clip(self, bvh, coords):
        view = make_scene([bvh], [coords[10:50:2]], "front", None,
                          clip_frames=slice(10, 50, 2)).views[0]
        np.testing.assert_allclose(
            view.root_heading, root_trajectory(bvh)[10:50:2, 2:4])

    def test_one_clip_frame_needs_one_row_coords(self, bvh, coords):
        with pytest.raises(ValueError, match="names one clip frame"):
            make_scene([bvh], [coords], "front", None, clip_frames=3)

    def test_root_heading_none_unless_the_coords_are_the_clips(
            self, bvh, coords):
        """The default attaches no clip fact: the caller must say which
        clip frames the coords are before a heading is aligned to them."""
        view = make_scene([bvh], [coords], "front", None).views[0]
        assert view.root_heading is None

    def test_router_says_which_clip_frames_the_coords_are(self, bvh, coords):
        from pybvh.bvhplot import _prepare
        heading = root_trajectory(bvh)[:, 2:4]

        whole = _prepare(bvh, None, "world", "front").views[0]
        np.testing.assert_allclose(whole.root_heading, heading)

        one = _prepare(bvh, 7, "world", "front").views[0]
        np.testing.assert_allclose(one.root_heading, heading[7:8])

        supplied = _prepare(bvh, coords[:5], "world", "front").views[0]
        assert supplied.root_heading is None

    def test_rest_pose_scene_carries_no_clip_heading(self, bvh, monkeypatch):
        """A rest pose is not a clip frame: attaching frame 0's heading
        to it would describe a different pose than the coords do."""
        import pybvh.bvhplot as bvhplot
        import pybvh.bvhplot._matplotlib as mpl_backend
        captured = []

        def fake_frame_mpl(scene, style, **kwargs):
            captured.append(scene)
            return None, None

        monkeypatch.setattr(mpl_backend, "frame_mpl", fake_frame_mpl)
        bvhplot.rest_pose(bvh, show=False)
        view = captured[0].views[0]
        assert view.root_heading is None
        np.testing.assert_allclose(view.coords[0], bvh.rest_pose_positions())

    def test_scene_frame_time_is_the_first_views(self, bvh, coords):
        scene = make_scene([bvh, bvh], [coords, coords], "front", None)
        assert scene.frame_time == bvh.frame_time


class TestSceneMethods:
    def test_a_view_holds_no_box(self):
        """What a picture frames is computed when the picture is made,
        so no Scene operation can leave a stale box behind."""
        names = {f.name for f in dataclasses.fields(SkeletonView)}
        assert not names & {"center", "half_span", "lo", "hi"}
        assert not hasattr(Scene, "unified_box")

    def test_subsampled_slices_every_frame_indexed_field(self, bvh, coords):
        scene = make_scene([bvh], [coords], "front", ["lbl"],
                           clip_frames=slice(None))
        step = 4
        sub = scene.subsampled(step)
        view, original = sub.views[0], scene.views[0]
        np.testing.assert_array_equal(view.coords, coords[::step])
        np.testing.assert_array_equal(
            view.root_heading, original.root_heading[::step])
        assert view.frame_time == pytest.approx(original.frame_time * step)
        assert sub.frame_time == view.frame_time
        # untouched by subsampling
        assert view.floor_height == original.floor_height
        assert view.label == "lbl"
        assert view.azimuth == original.azimuth
        assert np.shares_memory(scene.views[0].coords, coords)
        np.testing.assert_array_equal(scene.views[0].coords, coords)  # original untouched

    def test_subsampled_rejects_step_below_one(self, bvh, coords):
        scene = make_scene([bvh], [coords], "front", None)
        with pytest.raises(ValueError, match="step"):
            scene.subsampled(0)

    def test_offset_moves_coords_and_floor_together(self, bvh, coords):
        scene = make_scene([bvh], [coords], "front", None)
        view = scene.views[0]
        off = np.zeros(3)
        off[view.up_index] = 2.5
        moved = scene.offset([off]).views[0]
        np.testing.assert_allclose(moved.coords, coords + off)
        assert moved.floor_height == pytest.approx(view.floor_height + 2.5)
        # translation-invariant facts are kept
        np.testing.assert_array_equal(moved.root_heading, view.root_heading)
        assert moved.frame_time == view.frame_time

    def test_lateral_offset_leaves_the_floor(self, bvh, coords):
        scene = make_scene([bvh], [coords], "front", None)
        view = scene.views[0]
        off = np.ones(3)
        off[view.up_index] = 0.0
        moved = scene.offset([off]).views[0]
        assert moved.floor_height == view.floor_height

    def test_offset_length_mismatch_raises(self, bvh, coords):
        scene = make_scene([bvh], [coords], "front", None)
        with pytest.raises(ValueError, match="offsets"):
            scene.offset([np.zeros(3), np.zeros(3)])

    def test_spread_single_view_is_identity(self, bvh, coords):
        scene = make_scene([bvh], [coords], "front", None)
        assert scene.spread("auto") is scene
        assert scene.spread(3.0) is scene

    def test_spread_moves_later_views_along_the_lateral_axis(self, bvh, coords):
        scene = make_scene([bvh, bvh], [coords, coords], "front", None)
        first = scene.views[0]
        fwd_idx = {"x": 0, "y": 1, "z": 2}[first.forward_axis[1]]
        lat_idx = next(i for i in range(3)
                       if i != first.up_index and i != fwd_idx)
        spread = scene.spread(3.0)
        np.testing.assert_array_equal(spread.views[0].coords, coords)
        diff = spread.views[1].coords - coords
        assert np.allclose(diff[..., lat_idx], 3.0)
        for axis in range(3):
            if axis != lat_idx:
                assert np.allclose(diff[..., axis], 0.0)
        # a lateral move leaves the floor where it was
        assert spread.views[1].floor_height == first.floor_height

    def test_spread_auto_uses_the_first_views_lateral_extent(self, bvh, coords):
        scene = make_scene([bvh, bvh], [coords, coords], "front", None)
        first = scene.views[0]
        fwd_idx = {"x": 0, "y": 1, "z": 2}[first.forward_axis[1]]
        lat_idx = next(i for i in range(3)
                       if i != first.up_index and i != fwd_idx)
        width = float(np.ptp(coords[..., lat_idx]))
        spread = scene.spread("auto")
        diff = spread.views[1].coords - coords
        assert np.allclose(diff[..., lat_idx], max(width, 0.1) * 1.2)

    def test_spread_zero_is_identity(self, bvh, coords):
        scene = make_scene([bvh, bvh], [coords, coords], "front", None)
        assert scene.spread(0.0) is scene

    def test_views_are_frozen(self, bvh, coords):
        scene = make_scene([bvh], [coords], "front", None)
        with pytest.raises(Exception):
            scene.views[0].frame_time = 1.0  # type: ignore[misc]

    def test_subsampled_keeps_a_missing_heading_missing(self):
        view = dataclasses.replace(make_array_view(12), root_heading=None)
        sub = Scene(views=[view]).subsampled(3).views[0]
        assert sub.root_heading is None
        np.testing.assert_array_equal(sub.coords, view.coords[[0, 3, 6, 9]])
        assert sub.frame_time == pytest.approx(3 / 30)

    def test_subsampled_keeps_the_only_frame_of_a_one_frame_view(self):
        """The one frame is kept at any step; its frame time still grows
        by the step, as for every other view."""
        view = make_array_view(1)
        sub = Scene(views=[view]).subsampled(4).views[0]
        np.testing.assert_array_equal(sub.coords, view.coords)
        np.testing.assert_array_equal(sub.root_heading, view.root_heading)
        assert sub.frame_time == pytest.approx(4 / 30)

    def test_offset_on_a_negative_up_view_moves_the_floor_with_the_feet(self):
        """floor_height is a coordinate along the up axis, not a signed
        height: moving a '-y' view by +2 in y moves its floor to +2
        more, where the feet (the coordinate maximum) went."""
        view = make_array_view(up="-y")
        moved = Scene(views=[view]).offset([np.array([0.0, 2.0, 0.0])]).views[0]
        np.testing.assert_allclose(moved.coords, view.coords + [0.0, 2.0, 0.0])
        assert moved.floor_height == pytest.approx(view.floor_height + 2.0)
        assert moved.floor_height == pytest.approx(moved.coords[..., 1].max())

    def test_spread_on_negative_up_views_moves_along_positive_x(self):
        """With '-y' up and '+z' forward the lateral axis is x, and later
        views move toward +x, which on this rig is the character's
        right: the direction is the axis's, not the character's."""
        first = make_array_view(up="-y")
        second = make_array_view(up="-y")
        spread = Scene(views=[first, second]).spread(3.0)
        np.testing.assert_array_equal(spread.views[0].coords, first.coords)
        np.testing.assert_allclose(
            spread.views[1].coords, second.coords + [3.0, 0.0, 0.0])
        assert spread.views[1].floor_height == second.floor_height

    def test_spread_auto_on_negative_up_views_uses_the_x_extent(self):
        first = make_array_view(up="-y")
        width = float(np.ptp(first.coords[..., 0]))
        spread = Scene(views=[first, make_array_view(up="-y")]).spread("auto")
        np.testing.assert_allclose(
            spread.views[1].coords - first.coords,
            np.broadcast_to([1.2 * width, 0.0, 0.0], first.coords.shape))


class TestViewIsCheckedAtConstruction:
    """An inconsistent view raises where it is built, naming the field,
    instead of drawing something wrong in a backend later."""

    def test_a_consistent_view_builds(self):
        view = make_array_view(n_frames=5)
        assert view.coords.shape[0] == 5

    @pytest.mark.parametrize("changes, message", [
        (dict(coords=np.zeros((9, 3))), "coords must have shape"),
        (dict(coords=np.zeros((4, 9, 2))), "coords must have shape"),
        (dict(coords=np.zeros((0, 9, 3)), root_heading=None),
         "at least one frame"),
        (dict(coords=np.zeros((4, 0, 3)), node_names=[],
              rest_coords=np.zeros((0, 3))), "at least one frame"),
        (dict(up="y"), "up must be one of"),
        (dict(up="+Y"), "up must be one of"),
        (dict(up="up"), "up must be one of"),
        (dict(forward_axis="z"), "forward_axis must be one of"),
        (dict(forward_axis="-y"), "lies along the up axis"),
        (dict(node_names=["only"]), "node_names has 1 entries"),
        (dict(rest_coords=np.zeros((4, 3))), "rest_coords must have shape"),
        (dict(bone_chains=["spine"]), "bone_chains has 1 entries"),
        (dict(bones=[(0, 9)], bone_chains=["spine"]),
         r"bones names nodes \[9\]"),
        (dict(bones=[(-1, 2)], bone_chains=["spine"]),
         r"bones names nodes \[-1\]"),
        (dict(lr_pairs=np.array([[3, 12]])), r"lr_pairs names nodes \[12\]"),
        (dict(lr_pairs=np.array([3, 5])), "lr_pairs must be"),
        (dict(lr_pairs=np.empty((3, 0), dtype=np.intp)), "lr_pairs must be"),
        (dict(lr_pairs=np.array([[0.0, 1.5]])),
         "lr_pairs must hold integer node indices"),
        (dict(lr_pairs=np.array([[0.0, np.nan]])),
         "lr_pairs must hold integer node indices"),
        (dict(bones=[(0, 1.5)], bone_chains=["spine"]),
         "bones must hold integer node indices"),
        (dict(bones=[(0, 1, 2)], bone_chains=["spine"]), "bones must be"),
        (dict(root_heading=np.zeros((3, 2))), "root_heading must have shape"),
        (dict(root_heading=np.zeros((12, 3))), "root_heading must have shape"),
        (dict(frame_time=-0.01), "frame_time must be a number of seconds"),
        (dict(frame_time=float("nan")),
         "frame_time must be a number of seconds"),
        (dict(frame_time=float("inf")),
         "frame_time must be a number of seconds"),
    ])
    def test_an_inconsistent_view_raises(self, changes, message):
        view = make_array_view(n_frames=12)
        with pytest.raises(ValueError, match=message):
            dataclasses.replace(view, **changes)

    def test_a_view_without_pairs_or_heading_builds(self):
        view = dataclasses.replace(
            make_array_view(), root_heading=None,
            lr_pairs=np.empty((0, 2), dtype=np.intp))
        assert view.root_heading is None
        assert view.lr_pairs.shape == (0, 2)

    def test_a_view_with_one_node_and_no_bones_builds(self):
        view = dataclasses.replace(
            make_array_view(n_frames=3),
            coords=np.zeros((3, 1, 3)), rest_coords=np.zeros((1, 3)),
            node_names=["Hips"], bones=[], bone_chains=[],
            lr_pairs=np.empty((0, 2), dtype=np.intp))
        assert view.bones == []

    def test_an_unset_frame_time_is_accepted(self, bvh, coords):
        """Bvh.frame_time is 0 when unset, which a Bvh built in memory
        carries. Its rest pose and its stills must still draw: they
        never read the frame time."""
        unset = bvh.copy()
        unset.frame_time = 0
        for frames, clip_frames in [(coords[:1], 0), (coords, slice(None)),
                                    (coords[:1], None)]:
            view = make_scene([unset], [frames], "front", None,
                              clip_frames=clip_frames).views[0]
            assert view.frame_time == 0.0

    def test_a_bvh_built_view_is_checked_too(self, bvh, coords):
        """make_scene goes through the same checks: coords with more
        nodes than the Bvh match neither its names nor its rest pose.
        (Coords with fewer nodes already fail inside the core, which
        indexes them by the Bvh's joints.)"""
        extra = np.concatenate([coords, coords[:, :2]], axis=1)
        with pytest.raises(ValueError, match="node_names has"):
            make_scene([bvh], [extra], "front", None)


class TestViewArraysAreReadOnly:
    """A view's arrays cannot be written through the view. Storage is
    shared between a Scene and the ones derived from it, so a write
    would change several Scenes at once."""

    ARRAYS = ["coords", "rest_coords", "lr_pairs", "root_heading"]

    @pytest.mark.parametrize("name", ARRAYS)
    def test_writing_into_a_views_array_raises(self, name):
        array = getattr(make_array_view(), name)
        with pytest.raises(ValueError, match="read-only"):
            array[...] = 0

    @pytest.mark.parametrize("name", ARRAYS)
    def test_a_bvh_built_view_is_read_only_too(self, bvh, coords, name):
        view = make_scene(
            [bvh], [coords], "front", None, clip_frames=slice(None)).views[0]
        assert not getattr(view, name).flags.writeable

    @pytest.mark.parametrize("name", ARRAYS)
    def test_the_callers_array_keeps_its_flags_and_is_not_copied(self, name):
        source = make_array_view()
        mine = np.array(getattr(source, name))    # a writable copy
        view = dataclasses.replace(source, **{name: mine})
        assert mine.flags.writeable
        assert not getattr(view, name).flags.writeable
        assert np.shares_memory(getattr(view, name), mine)

    @pytest.mark.parametrize("rebuild", [
        copy.copy,
        copy.deepcopy,
        lambda view: pickle.loads(pickle.dumps(view)),
        lambda view: pickle.loads(pickle.dumps(view, protocol=5)),
    ])
    def test_a_copied_or_unpickled_view_is_read_only_too(self, rebuild):
        original = make_array_view()
        rebuilt = rebuild(original)
        for name in self.ARRAYS:
            assert not getattr(rebuilt, name).flags.writeable
            np.testing.assert_array_equal(
                getattr(rebuilt, name), getattr(original, name))
        assert rebuilt.up == original.up

    @pytest.mark.parametrize("operation", [
        lambda scene: scene.subsampled(2),
        lambda scene: scene.spread(1.5),
        lambda scene: scene.offset([np.ones(3), np.ones(3)]),
    ])
    def test_operations_return_read_only_views(self, operation):
        scene = operation(make_array_scene(n_frames=12, n_skeletons=2))
        for view in scene.views:
            for name in self.ARRAYS:
                assert not getattr(view, name).flags.writeable

    def test_a_missing_heading_stays_none(self):
        view = dataclasses.replace(make_array_view(), root_heading=None)
        assert view.root_heading is None

    def test_array_likes_are_accepted(self):
        view = dataclasses.replace(
            make_array_view(), rest_coords=make_array_view().rest_coords.tolist(),
            lr_pairs=[[3, 5]])
        assert view.rest_coords.shape == (9, 3)
        assert view.lr_pairs.shape == (1, 2)


class TestSceneIsCheckedAtConstruction:
    def test_a_scene_needs_a_view(self):
        with pytest.raises(ValueError, match="at least one view"):
            Scene(views=[])

    def test_views_must_share_their_frame_count(self):
        views = [make_array_view(n_frames=12), make_array_view(n_frames=8)]
        with pytest.raises(ValueError, match=r"\[12, 8\]"):
            Scene(views=views)

    def test_views_may_differ_in_frame_time(self):
        """The router only warns on a rate mismatch unless asked to
        resample, so a Scene must accept it."""
        views = [make_array_view(frame_time=1 / 30),
                 make_array_view(frame_time=1 / 60)]
        assert Scene(views=views).frame_time == 1 / 30

    def test_operations_keep_a_scene_valid(self):
        scene = make_array_scene(n_frames=12, n_skeletons=2)
        assert scene.subsampled(5).num_frames == 3
        assert scene.spread("auto").num_frames == 12


# The pybvh modules that know what a Bvh is. bvhplot keeps them behind a
# module boundary: the router and _from_bvh are the Bvh-facing layer, and
# only the router imports _from_bvh; every other module draws or computes
# from the Scene and takes nothing from the core at runtime, except the
# viewport, which may take array kernels.
_CORE_MODULES = {"bvh", "bvhnode", "tools", "analysis", "transforms",
                 "spatial_coord", "batch", "features", "io", "df_to_bvh"}
_ROUTER = "__init__"
_BVH_READER = "_from_bvh"
# Pure data: no plotting library may be imported here.
_PURE_DATA = ["_scene", "_viewport", "_style", "_from_bvh"]
# Array-pure kernels the viewport may take from pybvh.tools: they take
# arrays, never a Bvh.
_VIEWPORT_KERNELS = {"_leftward_units_from_pairs"}


def _bvhplot_modules() -> list[str]:
    """Every top-level module of the package, read from disk: a module
    added later is guarded without anyone remembering to list it.

    bvhplot is a flat package, and ``test_the_package_is_flat`` keeps it
    so: the guards resolve imports relative to the package root and
    would not see into a subpackage."""
    package_dir = pathlib.Path(bvhplot.__file__).parent
    return sorted(path.stem for path in package_dir.glob("*.py"))


# Everything except the router and the Bvh reader.
_BEHIND_THE_BOUNDARY = [name for name in _bvhplot_modules()
                        if name not in (_ROUTER, _BVH_READER)]
# ... of which the viewport alone may take array kernels from the core.
_CORE_FREE = [name for name in _BEHIND_THE_BOUNDARY if name != "_viewport"]


def _is_core_module(dotted: str, level: int) -> bool:
    """Is the module ``from <dots><dotted> import`` / ``import <dotted>``
    names a core module, or the package root (which exports only core
    names such as ``Bvh``)? Level 1 is a bvhplot sibling: never core.
    An explicit ``__init__`` component is the package root spelled out."""
    parts = [p for p in dotted.split(".") if p and p != "__init__"]
    if level == 0:
        return (len(parts) >= 1 and parts[0] == "pybvh"
                and (len(parts) == 1 or parts[1] in _CORE_MODULES))
    if level == 2:
        return not parts or parts[0] in _CORE_MODULES
    return False


def _typing_guard_names(tree: ast.Module) -> set[str]:
    """Which of ``TYPE_CHECKING`` and ``typing`` a module binds exactly
    once, by ``from typing import TYPE_CHECKING`` or ``import typing``.

    Any other binding of either name (assignment, parameter, def, alias,
    except clause) disqualifies it, so a shadowed guard is treated as a
    runtime condition. A star import or a write to an attribute named
    ``TYPE_CHECKING`` anywhere in the module disqualifies both, since
    either can change what the guard evaluates to. Conservative on
    purpose: ``import typing as t`` is not recognised and its block
    counts as runtime."""
    bindings: dict[str, list[bool]] = {"TYPE_CHECKING": [], "typing": []}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and any(
                alias.name == "*" for alias in node.names):
            return set()  # a star import may rebind either name unseen
        if (isinstance(node, ast.Attribute)
                and node.attr == "TYPE_CHECKING"
                and not isinstance(node.ctx, ast.Load)):
            return set()  # typing.TYPE_CHECKING = ... changes its value
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                bound = alias.asname or alias.name
                if bound in bindings:
                    bindings[bound].append(
                        node.level == 0 and node.module == "typing"
                        and alias.name == "TYPE_CHECKING"
                        and alias.asname is None)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".")[0]
                if bound in bindings:
                    bindings[bound].append(
                        alias.name == "typing" and alias.asname is None)
        elif isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            if node.id in bindings:
                bindings[node.id].append(False)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                               ast.ClassDef)):
            if node.name in bindings:
                bindings[node.name].append(False)
        elif isinstance(node, ast.arg) and node.arg in bindings:
            bindings[node.arg].append(False)
        elif isinstance(node, ast.ExceptHandler) and node.name in bindings:
            bindings[node.name].append(False)
    return {name for name, seen in bindings.items()
            if len(seen) == 1 and seen[0]}


def _is_type_checking_test(test: ast.expr, guard_names: set[str]) -> bool:
    """``if TYPE_CHECKING:`` or ``if typing.TYPE_CHECKING:``, with the
    name bound by the typing import alone (:func:`_typing_guard_names`);
    anything else, including ``x.TYPE_CHECKING`` or a shadowed name, is
    an ordinary runtime condition."""
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING" and "TYPE_CHECKING" in guard_names
    return (isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
            and isinstance(test.value, ast.Name) and test.value.id == "typing"
            and "typing" in guard_names)


def _core_imports_in_source(source: str) -> list[tuple[int, str | None, list[str]]]:
    """Every runtime import of a core module in ``source``: (line,
    innermost enclosing function or None, imported names).

    Sees both statement forms (``import pybvh.tools``, ``from ..tools
    import x``) and package-root imports (``from pybvh import Bvh``,
    ``from .. import tools``). Only the body of an ``if TYPE_CHECKING:``
    block is type-only; its ``else`` branch runs and is inspected."""
    tree = ast.parse(source)
    guard_names = _typing_guard_names(tree)
    found: list[tuple[int, str | None, list[str]]] = []

    def visit(nodes, func, type_only):
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visit(node.body, node.name, type_only)
            elif (isinstance(node, ast.If)
                    and _is_type_checking_test(node.test, guard_names)):
                visit(node.body, func, True)
                visit(node.orelse, func, type_only)
            elif type_only:
                continue
            elif isinstance(node, ast.ImportFrom):
                if _is_core_module(node.module or "", node.level):
                    found.append((node.lineno, func,
                                  [a.name for a in node.names]))
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names
                         if _is_core_module(a.name, 0)]
                if names:
                    found.append((node.lineno, func, names))
            else:
                visit(ast.iter_child_nodes(node), func, type_only)

    visit(tree.body, None, False)
    return found


def _sibling_imports_in_source(source: str, sibling: str) -> list[int]:
    """Line of every import of the bvhplot module ``sibling`` in
    ``source``, type-only ones included, in any spelling: ``from
    ._from_bvh import x``, ``from . import _from_bvh``, the absolute
    ``pybvh.bvhplot._from_bvh`` forms and ``from ..bvhplot`` ones."""
    found: list[int] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".")[:3] == ["pybvh", "bvhplot", sibling]
                   for alias in node.names):
                found.append(node.lineno)
        elif isinstance(node, ast.ImportFrom):
            parts = [p for p in (node.module or "").split(".")
                     if p and p != "__init__"]
            if node.level == 0 and parts[:2] == ["pybvh", "bvhplot"]:
                parts = parts[2:]
            elif node.level == 2 and parts[:1] == ["bvhplot"]:
                parts = parts[1:]
            elif node.level != 1:
                continue
            if parts[:1] == [sibling] or (
                    not parts
                    and any(alias.name == sibling for alias in node.names)):
                found.append(node.lineno)
    return found


def _router_imports_in_source(
    source: str, modules: set[str],
) -> list[tuple[int, list[str]]]:
    """Every import statement in ``source`` that reaches the router (the
    bvhplot package root): (line, names).

    The router re-exports what it imports, the Bvh readers included, so
    a module behind the boundary must not take a *name* from it
    (``from . import make_scene``), nor the router itself (``import
    pybvh.bvhplot``, ``from pybvh import bvhplot``), whose attributes
    are those same names. Taking a sibling *module* is fine (``from .
    import _colors``). An explicit ``__init__`` component is the root
    spelled out. A dotted ``import pybvh.bvhplot._scene`` with no
    ``as`` binds the top package, the router with it, and is flagged;
    with ``as`` it binds the sibling alone and passes. ``modules`` are
    the package's module names."""
    found: list[tuple[int, list[str]]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names = []
            for alias in node.names:
                parts = [p for p in alias.name.split(".") if p != "__init__"]
                if parts[:2] == ["pybvh", "bvhplot"] and (
                        len(parts) == 2 or alias.asname is None):
                    names.append(alias.name)
            if names:
                found.append((node.lineno, names))
        elif isinstance(node, ast.ImportFrom):
            parts = [p for p in (node.module or "").split(".")
                     if p and p != "__init__"]
            at_root = ((node.level == 1 and not parts)
                       or (node.level == 0 and parts == ["pybvh", "bvhplot"])
                       or (node.level == 2 and parts == ["bvhplot"]))
            above_root = ((node.level == 0 and parts == ["pybvh"])
                          or (node.level == 2 and not parts))
            if at_root:
                names = [alias.name for alias in node.names
                         if alias.name not in modules]
            elif above_root:
                names = [alias.name for alias in node.names
                         if alias.name == "bvhplot"]
            else:
                names = []
            if names:
                found.append((node.lineno, names))
    return found


def _module_source(module_name: str) -> str:
    package_dir = pathlib.Path(bvhplot.__file__).parent
    return (package_dir / f"{module_name}.py").read_text()


class TestCoreImportGuard:
    """The guard itself, against small sources: a checker that misses a
    spelling proves nothing about the modules it passes."""

    @pytest.mark.parametrize("line", [
        "from ..bvh import Bvh",
        "from ..tools import _compute_forward_at",
        "from .. import tools",
        "from .. import Bvh, analysis",
        "from pybvh.tools import extract_sign",
        "from pybvh import Bvh",
        "import pybvh.tools",
        "import pybvh.analysis as analysis",
        "import pybvh",
        "from pybvh.__init__ import Bvh",
        "from ..__init__ import Bvh",
    ])
    def test_flags_every_spelling_of_a_core_import(self, line):
        found = _core_imports_in_source(line)
        assert [(lineno, func) for lineno, func, _ in found] == [(1, None)]

    @pytest.mark.parametrize("prelude", [
        "import typing\ntyping.TYPE_CHECKING = True\n"
        "if typing.TYPE_CHECKING:\n",
        "from typing import TYPE_CHECKING\nfrom flags import *\n"
        "if TYPE_CHECKING:\n",
    ])
    def test_a_guard_that_may_have_been_overwritten_is_runtime(
            self, prelude):
        found = _core_imports_in_source(prelude + "    import pybvh.tools\n")
        assert [names for _, _, names in found] == [["pybvh.tools"]]

    def test_a_guard_name_rebound_by_an_except_clause_is_runtime(self):
        source = (
            "from typing import TYPE_CHECKING\n"
            "try:\n"
            "    pass\n"
            "except ValueError as TYPE_CHECKING:\n"
            "    if TYPE_CHECKING:\n"
            "        import pybvh.tools\n"
        )
        assert [lineno for lineno, _, _ in
                _core_imports_in_source(source)] == [6]

    @pytest.mark.parametrize("line", [
        "from ._scene import Scene",
        "from . import _colors",
        "from ..bvhplot._scene import Scene",
        "import numpy as np",
        "import matplotlib.pyplot as plt",
        "from typing import TYPE_CHECKING",
    ])
    def test_passes_non_core_imports(self, line):
        assert _core_imports_in_source(line) == []

    def test_type_checking_body_is_type_only_but_its_else_is_not(self):
        source = (
            "from typing import TYPE_CHECKING\n"
            "if TYPE_CHECKING:\n"
            "    from ..bvh import Bvh\n"
            "else:\n"
            "    from ..tools import extract_sign\n"
        )
        assert _core_imports_in_source(source) == [
            (5, None, ["extract_sign"])]

    def test_qualified_type_checking_guard_is_recognised(self):
        source = (
            "import typing\n"
            "if typing.TYPE_CHECKING:\n"
            "    from ..bvh import Bvh\n"
        )
        assert _core_imports_in_source(source) == []

    def test_only_typings_type_checking_is_type_only(self):
        source = (
            "from types import SimpleNamespace\n"
            "flags = SimpleNamespace(TYPE_CHECKING=True)\n"
            "if flags.TYPE_CHECKING:\n"
            "    import pybvh.tools\n"
        )
        assert _core_imports_in_source(source) == [(4, None, ["pybvh.tools"])]

    @pytest.mark.parametrize("source", [
        # bare guard with no typing import behind it
        "if TYPE_CHECKING:\n    import pybvh.tools\n",
        # rebound after the import
        "from typing import TYPE_CHECKING\nTYPE_CHECKING = True\n"
        "if TYPE_CHECKING:\n    import pybvh.tools\n",
        # `typing` is not the typing module
        "from types import SimpleNamespace\n"
        "typing = SimpleNamespace(TYPE_CHECKING=True)\n"
        "if typing.TYPE_CHECKING:\n    import pybvh.tools\n",
        # shadowed by a parameter
        "from typing import TYPE_CHECKING\n"
        "def f(TYPE_CHECKING):\n"
        "    if TYPE_CHECKING:\n        import pybvh.tools\n",
        # aliased import is not recognised, so its block is runtime
        "import typing as t\nif t.TYPE_CHECKING:\n    import pybvh.tools\n",
    ])
    def test_a_shadowed_or_unbound_guard_is_a_runtime_condition(self, source):
        found = _core_imports_in_source(source)
        assert [names for _, _, names in found] == [["pybvh.tools"]]

    def test_reports_the_innermost_enclosing_function(self):
        source = (
            "def outer():\n"
            "    def inner():\n"
            "        import pybvh.tools\n"
            "    from .. import analysis\n"
        )
        assert _core_imports_in_source(source) == [
            (3, "inner", ["pybvh.tools"]), (4, "outer", ["analysis"])]


class TestSiblingImportGuard:
    """The second guard, against small sources."""

    @pytest.mark.parametrize("line", [
        "from ._from_bvh import make_scene",
        "from . import _from_bvh",
        "from . import _colors, _from_bvh",
        "from pybvh.bvhplot._from_bvh import make_scene",
        "from pybvh.bvhplot import _from_bvh",
        "from ..bvhplot._from_bvh import make_scene",
        "from ..bvhplot import _from_bvh",
        "import pybvh.bvhplot._from_bvh",
        "import pybvh.bvhplot._from_bvh as reader",
        "if TYPE_CHECKING:\n    from ._from_bvh import make_scene",
        "def f():\n    from ._from_bvh import make_scene",
    ])
    def test_flags_every_spelling(self, line):
        assert len(_sibling_imports_in_source(line, "_from_bvh")) == 1

    @pytest.mark.parametrize("line", [
        "from ._scene import Scene",
        "from . import _colors",
        "from ._from_bvh_notes import x",
        "from pybvh.bvhplot import Style",
        "import pybvh.bvhplot",
        "from .. import tools",
    ])
    def test_passes_other_imports(self, line):
        assert _sibling_imports_in_source(line, "_from_bvh") == []

    @pytest.mark.parametrize("line", [
        "from . import make_scene",
        "from . import _colors, make_scene",
        "from pybvh.bvhplot import make_scene",
        "from ..bvhplot import Style",
        "from . import *",
        "from .__init__ import make_scene",
        "from pybvh.bvhplot.__init__ import make_scene as reader",
        "from ..bvhplot.__init__ import normalize_input",
    ])
    def test_flags_names_taken_from_the_router(self, line):
        found = _router_imports_in_source(line, {"_colors", "_scene"})
        assert [lineno for lineno, _ in found] == [1]

    @pytest.mark.parametrize("line", [
        "import pybvh.bvhplot",
        "import pybvh.bvhplot as router",
        "import pybvh.bvhplot.__init__ as router",
        "import pybvh.bvhplot._scene",
        "from pybvh import bvhplot",
        "from pybvh import bvhplot as router",
        "from .. import bvhplot",
        "from ..__init__ import bvhplot",
    ])
    def test_flags_the_router_taken_whole(self, line):
        found = _router_imports_in_source(line, {"_colors", "_scene"})
        assert [lineno for lineno, _ in found] == [1]

    @pytest.mark.parametrize("line", [
        "from . import _colors",
        "from . import _colors, _scene",
        "from ._scene import Scene",
        "from pybvh.bvhplot._style import Style",
        "import pybvh.bvhplot._scene as scene",
        "from .. import tools",
        "import numpy as np",
    ])
    def test_passes_sibling_modules_and_other_packages(self, line):
        assert _router_imports_in_source(
            line, {"_colors", "_scene"}) == []

    @pytest.mark.parametrize("line", [
        "from .__init__ import _from_bvh",
        "from pybvh.bvhplot.__init__ import _from_bvh",
    ])
    def test_flags_the_reader_taken_through_an_explicit_init(self, line):
        assert len(_sibling_imports_in_source(line, "_from_bvh")) == 1


class TestSceneIsPureData:
    @pytest.mark.parametrize("module_name", _PURE_DATA)
    def test_no_plotting_imports_in_pure_data_modules(self, module_name):
        """The Scene, the viewport, the Style and the Bvh reader must
        never import a plotting library, by any of its top-level
        package names."""
        tree = ast.parse(_module_source(module_name))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0]
                                for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])

        forbidden = {"matplotlib", "mpl_toolkits", "cv2", "k3d", "vedo",
                     "PIL", "vtk", "vtkmodules"}
        assert not (imported & forbidden), (
            f"{module_name} must stay plotting-free but imports "
            f"{sorted(imported & forbidden)}")

    def test_view_has_no_bvh_field(self):
        """The seam is real only if a view cannot hand a backend a Bvh."""
        names = {f.name for f in dataclasses.fields(SkeletonView)}
        assert "bvh" not in names

    def test_the_package_is_flat(self):
        """The guards below read the package's top-level modules. A
        subpackage would escape them, so adding one must fail here and
        send its author to extend the guards first."""
        package_dir = pathlib.Path(bvhplot.__file__).parent
        nested = sorted(
            str(path.relative_to(package_dir))
            for path in package_dir.rglob("*.py")
            if path.parent != package_dir)
        assert nested == [], (
            f"bvhplot has modules below its top level, which the import "
            f"guards do not inspect: {nested}")

    def test_the_boundary_covers_the_package(self):
        """The module list is read from disk; this pins that it finds the
        modules the guards below are about."""
        assert {"_scene", "_viewport", "_style", "_matplotlib", "_opencv",
                "_k3d", "_vedo", "_vedo_offscreen", "_vedo_capsules",
                "_colors", "_playback"} <= set(_BEHIND_THE_BOUNDARY)
        assert _BVH_READER in _bvhplot_modules()

    @pytest.mark.parametrize("module_name", _CORE_FREE)
    def test_takes_nothing_from_the_core_at_runtime(self, module_name):
        """A backend consumes a Scene and a Style; it never reaches into
        Bvh, tools or analysis, and neither do the Scene's and the
        Style's own modules. Type-only imports are allowed."""
        offenders = _core_imports_in_source(_module_source(module_name))
        assert offenders == [], (
            f"{module_name} imports core modules at runtime: {offenders}")

    @pytest.mark.parametrize("module_name", _BEHIND_THE_BOUNDARY)
    def test_never_reaches_the_router(self, module_name):
        """The router re-exports what it imports, the Bvh readers
        included. A module behind the boundary takes sibling modules
        from the package; it takes neither a name from the router nor
        the router itself."""
        offenders = _router_imports_in_source(
            _module_source(module_name), set(_bvhplot_modules()))
        assert offenders == [], (
            f"{module_name} imports the router or names from it: "
            f"{offenders}")

    @pytest.mark.parametrize("module_name", _BEHIND_THE_BOUNDARY)
    def test_only_the_router_imports_the_bvh_reader(self, module_name):
        """``_from_bvh`` is where a Bvh becomes a Scene. No module but
        the router imports it, not even for types.

        This is a guard on import statements, not proof that nothing
        else consumes a Bvh: a function handed one reads it without
        importing anything, and ``importlib.import_module`` is not an
        import statement. The router hands backends a Scene and a
        Style, never a Bvh."""
        offenders = _sibling_imports_in_source(
            _module_source(module_name), _BVH_READER)
        assert offenders == [], (
            f"{module_name} imports {_BVH_READER} at lines {offenders}")

    def test_viewport_takes_only_array_kernels_from_the_core(self):
        """The viewport is computed at draw time from a view: the only
        things it may import from the core are kernels that take arrays,
        never a Bvh."""
        offenders = [
            (line, func, names)
            for line, func, names in _core_imports_in_source(
                _module_source("_viewport"))
            if not set(names) <= _VIEWPORT_KERNELS]
        assert offenders == [], (
            f"_viewport imports more than array kernels: {offenders}")
