"""Tests for the Scene/SkeletonView container (bvhplot Phase 0)."""
from __future__ import annotations

import numpy as np
import pytest

from pybvh import read_bvh_file
from pybvh.bvhplot._common import (
    Scene,
    SkeletonView,
    make_scene,
    get_skeleton_lines,
    compute_unified_limits,
    get_camera_angles,
)

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
