"""Tests for the Scene/SkeletonView container (bvhplot Phase 0)."""
from __future__ import annotations

import ast
import dataclasses
import importlib

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

    def test_subsampled_slices_every_frame_indexed_field(self, bvh, coords):
        scene = make_scene([bvh], [coords], "front", ["lbl"])
        step = 4
        sub = scene.subsampled(step)
        view, original = sub.views[0], scene.views[0]
        np.testing.assert_array_equal(view.coords, coords[::step])
        np.testing.assert_array_equal(
            view.root_heading, original.root_heading[::step])
        assert view.frame_time == pytest.approx(original.frame_time * step)
        assert sub.frame_time == view.frame_time
        center, half_span = compute_unified_limits([coords[::step]])
        np.testing.assert_allclose(view.center, center)
        assert view.half_span == half_span
        # untouched by subsampling
        assert view.floor_height == original.floor_height
        assert view.label == "lbl"
        assert view.azimuth == original.azimuth
        assert scene.views[0].coords is coords  # original untouched

    def test_subsampled_rejects_step_below_one(self, bvh, coords):
        scene = make_scene([bvh], [coords], "front", None)
        with pytest.raises(ValueError, match="step"):
            scene.subsampled(0)

    def test_offset_moves_coords_center_and_floor_together(self, bvh, coords):
        scene = make_scene([bvh], [coords], "front", None)
        view = scene.views[0]
        off = np.zeros(3)
        off[view.up_index] = 2.5
        moved = scene.offset([off]).views[0]
        np.testing.assert_allclose(moved.coords, coords + off)
        np.testing.assert_allclose(moved.center, view.center + off)
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
        # the moved view's box moved with it; its floor did not
        np.testing.assert_allclose(
            spread.views[1].center - first.center, diff[0, 0])
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
            scene.views[0].half_span = 1.0  # type: ignore[misc]


# The pybvh modules that know what a Bvh is. A backend draws a Scene; it
# must not take anything from these at runtime, and _common may only do so
# inside the functions that turn a Bvh into a Scene.
_CORE_MODULES = {"bvh", "bvhnode", "tools", "analysis", "transforms",
                 "spatial_coord", "batch", "features", "io", "df_to_bvh"}
_BACKENDS = ["_matplotlib", "_opencv", "_k3d", "_vedo", "_vedo_offscreen",
             "_vedo_capsules", "_colors", "_playback"]
# The Bvh -> Scene adapter functions in _common.
_COMMON_ADAPTERS = {"make_scene", "get_camera_angles",
                    "_camera_angles_and_forward", "get_skeleton_lines",
                    "get_bone_chains"}
# Array-pure kernels a draw-time helper may take from pybvh.tools: they
# take arrays, never a Bvh.
_COMMON_KERNELS = {"_leftward_units_from_pairs"}


def _core_imports(module_name):
    """Every runtime import of a core module: (line, enclosing function,
    imported names). Imports under ``if TYPE_CHECKING:`` are skipped."""
    module = importlib.import_module(f"pybvh.bvhplot.{module_name}")
    tree = ast.parse(open(module.__file__).read())
    found = []

    def visit(node, func, type_checking):
        for child in ast.iter_child_nodes(node):
            f, tc = func, type_checking
            if isinstance(child, ast.FunctionDef):
                f = child.name
            if (isinstance(child, ast.If) and isinstance(child.test, ast.Name)
                    and child.test.id == "TYPE_CHECKING"):
                tc = True
            if isinstance(child, ast.ImportFrom) and not tc:
                module = child.module or ""
                top = (module.split(".")[0] if child.level == 0
                       else module.split(".")[0])
                is_core = ((child.level >= 2 and top in _CORE_MODULES)
                           or module.startswith("pybvh.")
                           and module.split(".")[1] in _CORE_MODULES)
                if is_core:
                    found.append((child.lineno, f,
                                  [a.name for a in child.names]))
            visit(child, f, tc)

    visit(tree, None, False)
    return found


class TestSceneIsPureData:
    def test_no_plotting_imports_in_common(self):
        """_common (Scene's home) must never import a plotting library."""
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

    def test_view_has_no_bvh_field(self):
        """The seam is real only if a view cannot hand a backend a Bvh."""
        names = {f.name for f in dataclasses.fields(SkeletonView)}
        assert "bvh" not in names

    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_backends_take_nothing_from_the_core_at_runtime(self, backend):
        """A backend consumes a Scene and a Style; it never reaches into
        Bvh, tools or analysis. Type-only imports are allowed."""
        offenders = _core_imports(backend)
        assert offenders == [], (
            f"{backend} imports core modules at runtime: {offenders}")

    def test_common_reads_bvh_only_in_its_adapter_functions(self):
        """Inside _common, only the Bvh -> Scene adapters may import core
        modules; a helper called at draw time may take array kernels only."""
        offenders = [
            (line, func, names)
            for line, func, names in _core_imports("_common")
            if func not in _COMMON_ADAPTERS
            and not set(names) <= _COMMON_KERNELS]
        assert offenders == [], (
            f"_common reaches into the core outside its adapters: {offenders}")
