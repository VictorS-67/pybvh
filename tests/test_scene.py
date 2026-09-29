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
    make_scene,
    get_skeleton_lines,
    get_bone_chains,
    get_camera_angles,
)
from pybvh.bvhplot._scene import Scene, SkeletonView, compute_unified_limits
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
    def test_unified_box_covers_all_views(self, bvh, coords):
        shifted = coords + np.array([100.0, 0.0, 0.0])
        scene = make_scene([bvh, bvh], [coords, shifted], "front", None)
        center, half_span = scene.unified_box()
        expected_center, expected_half = compute_unified_limits(
            [coords, shifted])
        np.testing.assert_allclose(center, expected_center)
        assert half_span == expected_half

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
# must not take anything from these at runtime, the Scene's own module
# must not either, and _common may only do so inside the functions that
# turn a Bvh into a Scene.
_CORE_MODULES = {"bvh", "bvhnode", "tools", "analysis", "transforms",
                 "spatial_coord", "batch", "features", "io", "df_to_bvh"}
_BACKENDS = ["_matplotlib", "_opencv", "_k3d", "_vedo", "_vedo_offscreen",
             "_vedo_capsules", "_colors", "_playback"]
# Pure data: no plotting library may be imported here.
_PURE_DATA = ["_common", "_scene", "_viewport", "_style"]
# Modules that take nothing from the core at runtime.
_CORE_FREE = _BACKENDS + ["_scene", "_style"]
# The Bvh -> Scene adapter functions in _common.
_COMMON_ADAPTERS = {"make_scene", "get_camera_angles",
                    "_camera_angles_and_forward", "get_skeleton_lines",
                    "get_bone_chains"}
# Array-pure kernels the viewport may take from pybvh.tools: they take
# arrays, never a Bvh.
_VIEWPORT_KERNELS = {"_leftward_units_from_pairs"}


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

    Any other binding of either name (assignment, parameter, def, alias)
    disqualifies it, so a shadowed guard is treated as a runtime
    condition. Conservative on purpose: ``import typing as t`` is not
    recognised and its block counts as runtime."""
    bindings: dict[str, list[bool]] = {"TYPE_CHECKING": [], "typing": []}
    for node in ast.walk(tree):
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


def _core_imports(module_name):
    """:func:`_core_imports_in_source` over a bvhplot module's file."""
    module = importlib.import_module(f"pybvh.bvhplot.{module_name}")
    with open(module.__file__) as f:
        return _core_imports_in_source(f.read())


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

    @pytest.mark.parametrize("line", [
        "from ._common import Scene",
        "from . import _colors",
        "from ..bvhplot._common import Scene",
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


class TestSceneIsPureData:
    @pytest.mark.parametrize("module_name", _PURE_DATA)
    def test_no_plotting_imports_in_pure_data_modules(self, module_name):
        """The Scene's home and the shared helpers must never import a
        plotting library."""
        module = importlib.import_module(f"pybvh.bvhplot.{module_name}")

        tree = ast.parse(open(module.__file__).read())
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0]
                                for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])

        forbidden = {"matplotlib", "cv2", "k3d", "vedo", "PIL", "vtk"}
        assert not (imported & forbidden), (
            f"{module_name} must stay plotting-free but imports "
            f"{sorted(imported & forbidden)}")

    def test_view_has_no_bvh_field(self):
        """The seam is real only if a view cannot hand a backend a Bvh."""
        names = {f.name for f in dataclasses.fields(SkeletonView)}
        assert "bvh" not in names

    @pytest.mark.parametrize("backend", _CORE_FREE)
    def test_backends_take_nothing_from_the_core_at_runtime(self, backend):
        """A backend consumes a Scene and a Style; it never reaches into
        Bvh, tools or analysis, and neither does the Scene's own module.
        Type-only imports are allowed."""
        offenders = _core_imports(backend)
        assert offenders == [], (
            f"{backend} imports core modules at runtime: {offenders}")

    def test_viewport_takes_only_array_kernels_from_the_core(self):
        """The viewport is computed at draw time from a view: the only
        things it may import from the core are kernels that take arrays,
        never a Bvh."""
        offenders = [
            (line, func, names)
            for line, func, names in _core_imports("_viewport")
            if not set(names) <= _VIEWPORT_KERNELS]
        assert offenders == [], (
            f"_viewport imports more than array kernels: {offenders}")

    def test_common_imports_the_core_only_in_its_adapter_functions(self):
        """Inside _common, only the Bvh -> Scene adapters may import core
        modules.

        This is an import guard, not proof that only the adapters consume
        a Bvh: a function handed a Bvh (``normalize_input``) reads it
        without importing anything, and the guard cannot see that."""
        offenders = [
            (line, func, names)
            for line, func, names in _core_imports("_common")
            if func not in _COMMON_ADAPTERS]
        assert offenders == [], (
            f"_common reaches into the core outside its adapters: {offenders}")
