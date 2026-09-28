"""Tests for the Scene/SkeletonView container (bvhplot Phase 0)."""
from __future__ import annotations

import numpy as np
import pytest

from pybvh import read_bvh_file
from pybvh.analysis import root_trajectory
from pybvh.bvhplot._common import (
    Scene,
    SkeletonView,
    make_scene,
    get_skeleton_lines,
    get_bone_chains,
    compute_unified_limits,
    get_camera_angles,
)
from pybvh.tools import _resolve_lr_pairs
from synthetic_bvh import make_nameless_lr_bvh

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
        assert view.bvh is bvh
        assert view.coords is coords
        assert view.label is None

    def test_view_fields_match_helpers(self, bvh, coords):
        """Scene assembly must agree with the individual helpers it wraps."""
        scene = make_scene([bvh], [coords], "front", None)
        view = scene.views[0]

        assert view.bones == get_skeleton_lines(bvh)

        center, half_span = compute_unified_limits([coords])
        np.testing.assert_allclose(view.center, center)
        assert view.half_span == half_span

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
        np.testing.assert_allclose(view.up_vector, bvh.up_axis.vector)
        assert view.forward_axis == bvh.forward_at(0)

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
        view = make_scene([bvh], [coords], "front", None).views[0]
        np.testing.assert_allclose(
            view.root_heading, root_trajectory(bvh)[:, 2:4])

    def test_root_heading_follows_truncated_coords(self, bvh, coords):
        short = coords[:40]
        view = make_scene([bvh], [short], "front", None).views[0]
        assert view.root_heading.shape == (40, 2)
        np.testing.assert_allclose(
            view.root_heading, root_trajectory(bvh)[:40, 2:4])

    def test_root_heading_follows_padded_coords(self, bvh, coords):
        extra = 7
        padded = np.concatenate(
            [coords, np.repeat(coords[-1:], extra, axis=0)], axis=0)
        view = make_scene([bvh], [padded], "front", None).views[0]
        heading = root_trajectory(bvh)[:, 2:4]
        assert view.root_heading.shape == (coords.shape[0] + extra, 2)
        np.testing.assert_allclose(view.root_heading[:-extra], heading)
        np.testing.assert_allclose(
            view.root_heading[-extra:], np.repeat(heading[-1:], extra, axis=0))

    def test_root_heading_for_a_single_frame(self, bvh, coords):
        one = coords[-1:]
        view = make_scene([bvh], [one], "front", None,
                          frame_index=-1).views[0]
        assert view.root_heading.shape == (1, 2)
        np.testing.assert_allclose(
            view.root_heading[0], root_trajectory(bvh)[-1, 2:4])

    def test_root_heading_none_for_caller_supplied_coords(self, bvh, coords):
        view = make_scene([bvh], [coords], "front", None,
                          coords_from_clip=False).views[0]
        assert view.root_heading is None

    def test_scene_frame_time_is_the_first_views(self, bvh, coords):
        scene = make_scene([bvh, bvh], [coords, coords], "front", None)
        assert scene.frame_time == bvh.frame_time


class TestSceneMethods:
    def test_unified_box_covers_all_views(self, bvh, coords):
        shifted = coords + np.array([100.0, 0.0, 0.0])
        scene = make_scene([bvh, bvh], [coords, shifted], "front", None)
        center, half_span = scene.unified_box()
        expected_center, expected_half = compute_unified_limits(
            [coords, shifted])
        np.testing.assert_allclose(center, expected_center)
        assert half_span == expected_half

    def test_replace_coords_swaps_only_coords(self, bvh, coords):
        scene = make_scene([bvh], [coords], "front", ["lbl"])
        new_coords = coords + 5.0
        replaced = scene.replace_coords([new_coords])
        assert replaced is not scene
        assert replaced.views[0].coords is new_coords
        # everything else preserved
        assert replaced.views[0].label == "lbl"
        assert replaced.views[0].azimuth == scene.views[0].azimuth
        np.testing.assert_allclose(
            replaced.views[0].center, scene.views[0].center)
        # original untouched
        assert scene.views[0].coords is coords

    def test_replace_coords_length_mismatch_raises(self, bvh, coords):
        scene = make_scene([bvh], [coords], "front", None)
        with pytest.raises(ValueError, match="coord arrays"):
            scene.replace_coords([coords, coords])

    def test_views_are_frozen(self, bvh, coords):
        scene = make_scene([bvh], [coords], "front", None)
        with pytest.raises(Exception):
            scene.views[0].half_span = 1.0  # type: ignore[misc]


class TestSceneIsPureData:
    def test_no_plotting_imports_in_common(self):
        """_common (Scene's home) must never import a plotting library."""
        import ast
        import pybvh.bvhplot._common as common

        tree = ast.parse(open(common.__file__).read())
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0]
                                for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])

        forbidden = {"matplotlib", "cv2", "k3d", "vedo", "PIL", "vtk"}
        assert not (imported & forbidden), (
            f"_common.py must stay plotting-free but imports "
            f"{sorted(imported & forbidden)}")
