"""Tests for the offscreen vedo backend (Phase 4)."""
from __future__ import annotations

import numpy as np
import pytest

vedo = pytest.importorskip("vedo")

from pybvh import read_bvh_file, bvhplot
from pybvh.bvhplot import Style

BVH_PATH = "bvh_data/cmu_12_01_walk.bvh"


@pytest.fixture(scope="module")
def bvh():
    return read_bvh_file(BVH_PATH)


@pytest.fixture(scope="module")
def bvh_hands():
    """A rig with full hands — the case capsule sizing has to survive."""
    return read_bvh_file("bvh_data/bvh_test3.bvh")


class TestFrameVedo:
    def test_returns_rgb_array(self, bvh):
        img = bvhplot.frame(bvh, 260, backend="vedo",
                            resolution=(400, 360))
        assert isinstance(img, np.ndarray)
        assert img.ndim == 3 and img.shape[2] == 3
        assert img.dtype == np.uint8

    def test_writes_filepath(self, bvh, tmp_path):
        out = tmp_path / "shot.png"
        img = bvhplot.frame(bvh, 260, backend="vedo",
                            resolution=(300, 280), filepath=out)
        assert out.exists() and out.stat().st_size > 0
        assert isinstance(img, np.ndarray)

    def test_shadow_darkens_floor(self, bvh):
        """With Style.shadow the floor must contain gray shadow pixels
        that the shadowless render lacks."""
        with_shadow = bvhplot.frame(
            bvh, 260, backend="vedo", resolution=(400, 360),
            style=Style("paper", shadow=True))
        without = bvhplot.frame(
            bvh, 260, backend="vedo", resolution=(400, 360),
            style=Style("paper", shadow=False))
        assert not np.array_equal(with_shadow, without)
        # shadowed image is darker overall (gray casts on a light floor)
        assert with_shadow.mean() < without.mean()

    def test_unknown_backend_raises(self, bvh):
        with pytest.raises(ValueError, match="backend"):
            bvhplot.frame(bvh, backend="opencv")

    def test_camera_applied_under_notebook_backend(self, bvh):
        """Regression: inside Jupyter, vedo auto-selects its '2d'
        display backend, whose show() ignores camera= — renders came
        out at VTK's default birdview. The offscreen renderer must
        force the plain VTK backend and restore the user's setting."""
        reference = bvhplot.frame(bvh, 260, backend="vedo",
                                  resolution=(300, 280))
        saved = vedo.settings.default_backend
        vedo.settings.default_backend = "2d"   # simulate a notebook kernel
        try:
            img = bvhplot.frame(bvh, 260, backend="vedo",
                                resolution=(300, 280))
            assert vedo.settings.default_backend == "2d", \
                "user's backend setting must be restored after the render"
        finally:
            vedo.settings.default_backend = saved
        assert np.array_equal(img, reference)


class TestRenderVedo:
    def test_gif(self, bvh, tmp_path):
        path = bvhplot.render(
            bvh[0:6], tmp_path / "caps.gif", backend="vedo",
            resolution=(300, 280))
        assert path.exists() and path.stat().st_size > 0

    def test_mp4(self, bvh, tmp_path):
        pytest.importorskip("cv2")
        path = bvhplot.render(
            bvh[0:6], tmp_path / "caps.mp4", backend="vedo",
            resolution=(300, 280))
        assert path.exists() and path.stat().st_size > 0

    def test_mp4_codec_plumbs_through(self, bvh, tmp_path):
        """codec= reaches the vedo backend's video sink."""
        cv2 = pytest.importorskip("cv2")
        path = bvhplot.render(
            bvh[0:4], tmp_path / "caps_m4.mp4", backend="vedo",
            resolution=(300, 280), codec="mpeg4")
        cap = cv2.VideoCapture(str(path))
        fcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        cap.release()
        four = "".join(chr((fcc >> 8 * i) & 0xFF) for i in range(4))
        assert four in ("FMP4", "mp4v", "XVID")

    def test_unsupported_extension_raises(self, bvh, tmp_path):
        with pytest.raises(ValueError, match="vedo backend"):
            bvhplot.render(bvh[0:4], tmp_path / "x.html", backend="vedo")

    def test_unsupported_features_raise(self, bvh, tmp_path):
        with pytest.raises(ValueError, match="ghost"):
            bvhplot.render(bvh[0:4], tmp_path / "x.mp4", backend="vedo",
                           ghost=2)
        with pytest.raises(ValueError, match="turntable"):
            bvhplot.render(bvh[0:4], tmp_path / "x.mp4", backend="vedo",
                           camera="turntable")


class TestCapsuleSizing:
    """Geometry-only capsule radii: four local rules, no joint names."""

    @staticmethod
    def _clear(pose, bones):
        from pybvh.bvhplot._vedo_capsules import crowding_clearance
        return crowding_clearance(np.asarray(pose, dtype=float), bones)

    def test_parallel_side_by_side_bones_crowd(self):
        """Two bones running together, a hand's width apart: crowded."""
        pose = [[0, 0, 0], [0, 0, 4], [1, 0, 0], [1, 0, 4]]
        clear = self._clear(pose, [(0, 1), (2, 3)])
        assert np.allclose(clear, [1.0, 1.0])

    def test_diverging_branches_do_not_crowd(self):
        """Antiparallel bones (the two clavicles leaving a spine) are
        not crowding, though an unsigned parallelism test would say so."""
        pose = [[0, 0, 0], [-4, 0, 0], [0.01, 0, 0], [4, 0, 0]]
        clear = self._clear(pose, [(0, 1), (2, 3)])
        assert np.isinf(clear).all()

    def test_chain_continuation_does_not_crowd(self):
        """Collinear bones stacked end to end (a neck) are not
        side-by-side: this is the case a raw nearest-bone distance
        gets wrong, thinning the neck."""
        pose = [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]]
        clear = self._clear(pose, [(0, 1), (2, 3)])
        assert np.isinf(clear).all()

    def test_crowding_cap_propagates_down_the_chain(self):
        """A child never gets more room than its parent had, so a
        fingertip that splays apart stays as slim as its knuckle."""
        from pybvh.bvhplot._vedo_capsules import adaptive_radii
        # bones 0-1 and 2-3 run side by side (crowded); 1-4 continues
        # from the first, alone in space (uncrowded on its own).
        pose = np.array([[0., 0, 0], [0, 0, 4], [0.5, 0, 0], [0.5, 0, 4],
                         [0, 0, 9]])
        bones = [(0, 1), (2, 3), (1, 4)]
        radii, _ = adaptive_radii(pose, bones, r_base=1.0)
        assert radii[(1, 4)] <= radii[(0, 1)] + 1e-9

    def test_crowded_runs_taper_from_base_to_tip(self, bvh_hands):
        """A finger is thickest at the knuckle. Untapered, the caps grow
        distally — splayed tips have more room than packed metacarpals —
        which renders a hand thin at the wrist and fattest at the tips."""
        from pybvh.bvhplot._vedo_capsules import adaptive_radii
        from pybvh.bvhplot import get_skeleton_lines
        bones = get_skeleton_lines(bvh_hands)
        rest = bvh_hands.rest_pose_positions()
        radii, _ = adaptive_radii(rest, bones, 1.0, rest)
        idx = bvh_hands.node_index
        chain = ["RightHandPinky", "RightHandPinky1", "RightHandPinky2",
                 "RightHandPinky3"]
        along = [radii[(idx[a], idx[b])]
                 for a, b in zip(chain, chain[1:])]
        assert all(later < earlier
                   for earlier, later in zip(along, along[1:])), along

    def test_taper_cannot_touch_an_uncrowded_chain(self, monkeypatch):
        """The taper rides on the crowding cap, which is inf when nothing
        crowds — so however hard it is set, an isolated chain is unmoved."""
        from pybvh.bvhplot import _vedo_capsules as caps
        pose = np.array([[0., 0, 0], [0, 0, 3], [0, 0, 6], [0, 0, 9]])
        bones = [(0, 1), (1, 2), (2, 3)]
        monkeypatch.setattr(caps, "CHAIN_TAPER", 1.0)
        untapered, _ = caps.adaptive_radii(pose, bones, 1.0)
        monkeypatch.setattr(caps, "CHAIN_TAPER", 0.2)
        tapered, _ = caps.adaptive_radii(pose, bones, 1.0)
        assert untapered == tapered

    def test_shipped_taper_leaves_a_fingerless_rig_alone(self, monkeypatch):
        """Scoping in practice: the taper compounds along a chain, so a
        harsh value would eventually bite even the loosely-crowded arms of
        a plain rig. At the shipped value a fingerless skeleton is
        untouched — the effect stays where hands are."""
        from pybvh import read_bvh_file
        from pybvh.bvhplot import _vedo_capsules as caps
        from pybvh.bvhplot import get_skeleton_lines
        plain = read_bvh_file("bvh_data/bvh_test1.bvh")
        bones = get_skeleton_lines(plain)
        rest = plain.rest_pose_positions()
        shipped, _ = caps.adaptive_radii(rest, bones, 1.0, rest)
        monkeypatch.setattr(caps, "CHAIN_TAPER", 1.0)
        no_taper, _ = caps.adaptive_radii(rest, bones, 1.0, rest)
        assert shipped == no_taper

    def test_long_bone_is_exempt_from_the_stub_cap(self, bvh_hands):
        """Regression: the forearm is a hub joining one thick bone to
        five thin metacarpals. Sizing it from the thinnest neighbour
        collapsed it to finger width."""
        from pybvh.bvhplot._vedo_capsules import adaptive_radii, CapsuleSkeleton
        from pybvh.bvhplot import get_skeleton_lines
        bones = get_skeleton_lines(bvh_hands)
        rest = bvh_hands.rest_pose_positions()
        r_base = CapsuleSkeleton.base_radius(
            float(np.linalg.norm(rest.max(0) - rest.min(0))) / 2, 3.0)
        radii, _ = adaptive_radii(rest, bones, r_base, rest)
        idx = bvh_hands.node_index
        forearm = radii[(idx["RightForeArm"], idx["RightHand"])]
        upper = radii[(idx["RightArm"], idx["RightForeArm"])]
        assert forearm == pytest.approx(upper, rel=0.25)

    def test_fingers_are_thinned_without_naming_them(self, bvh_hands):
        from pybvh.bvhplot._vedo_capsules import adaptive_radii, CapsuleSkeleton
        from pybvh.bvhplot import get_skeleton_lines
        bones = get_skeleton_lines(bvh_hands)
        rest = bvh_hands.rest_pose_positions()
        r_base = CapsuleSkeleton.base_radius(
            float(np.linalg.norm(rest.max(0) - rest.min(0))) / 2, 3.0)
        radii, _ = adaptive_radii(rest, bones, r_base, rest)
        idx = bvh_hands.node_index
        upper = radii[(idx["RightArm"], idx["RightForeArm"])]
        finger = radii[(idx["RightHandPinky1"], idx["RightHandPinky2"])]
        assert finger < upper / 3

    def test_short_isolated_bones_keep_full_radius(self):
        """The neck bug: short links with nothing beside them are not
        thinned, so a two-link neck matches the spine below it."""
        from pybvh.bvhplot._vedo_capsules import adaptive_radii
        from pybvh import read_bvh_file
        from pybvh.bvhplot import get_skeleton_lines
        neck_rig = read_bvh_file("bvh_data/bvh_test1.bvh")
        bones = get_skeleton_lines(neck_rig)
        rest = neck_rig.rest_pose_positions()
        radii, _ = adaptive_radii(rest, bones, 1.0, rest)
        idx = neck_rig.node_index
        neck = radii[(idx["Neck"], idx["Neck1"])]
        spine = radii[(idx["Spine2"], idx["Spine3"])]
        assert neck == pytest.approx(spine, rel=0.1)

    def test_joint_radius_is_the_min_of_its_bones(self):
        """At a hub the mean would bulge a sphere past the thin tubes."""
        from pybvh.bvhplot._vedo_capsules import adaptive_radii
        pose = np.array([[0., 0, 0], [0, 0, 4], [0.4, 0, 4], [0.4, 0, 8],
                         [0, 0, 8]])
        bones = [(0, 1), (1, 4), (2, 3)]
        radii, joints = adaptive_radii(pose, bones, r_base=1.0)
        touching = [r for b, r in radii.items() if 1 in b]
        assert joints[1] == pytest.approx(min(touching))

    def test_empty_and_degenerate_skeletons(self):
        from pybvh.bvhplot._vedo_capsules import adaptive_radii
        pose = np.zeros((3, 3))
        radii, joints = adaptive_radii(pose, [], r_base=1.0)
        assert radii == {} and joints.shape == (3,)
        radii, _ = adaptive_radii(pose, [(0, 1), (1, 2)], r_base=1.0)
        assert all(np.isfinite(r) and r > 0 for r in radii.values())


class TestFloorAgreesAcrossBackends:
    """Every backend that draws a ground plane takes it from the scene.

    Three floor rules used to live in three files — ``Bvh.floor_height``
    for matplotlib/OpenCV, a hand-rolled "min over all node centres" in
    the vedo viewer, and the bounding-box bottom in k3d. Only k3d keeps
    its own rule, and for a stated reason: it snaps its trail to k3d's
    own cubic grid, and draws no ``Style.floor`` plane at all.
    """

    @staticmethod
    def _scene(floor_height=None):
        import dataclasses
        from pybvh.bvhplot._from_bvh import make_scene

        bvh = read_bvh_file(BVH_PATH)
        scene = make_scene([bvh], [bvh.node_positions()[:40]], "front", None)
        if floor_height is not None:
            scene = dataclasses.replace(
                scene, views=[dataclasses.replace(
                    scene.views[0], floor_height=floor_height)])
        return scene

    @staticmethod
    def _plane_height(plotter, up_index):
        planes = [o for o in plotter.objects
                  if type(o).__name__ == "Plane"]
        assert len(planes) == 1, f"expected one floor plane, got {len(planes)}"
        return float(planes[0].pos()[up_index])

    def _offscreen_floor(self, scene):
        from pybvh.bvhplot._vedo_offscreen import _build_offscreen

        plt_, _, _ = _build_offscreen(scene, Style("paper"), (200, 200))
        height = self._plane_height(plt_, scene.views[0].up_index)
        plt_.close()
        return height

    def _viewer_floor(self, scene, monkeypatch):
        from pybvh.bvhplot import _vedo

        monkeypatch.setattr(_vedo, "_FORCE_OFFSCREEN", True)
        player = _vedo._VedoPlayer(scene, Style("paper"), 30.0, quality="high")
        height = self._plane_height(player.plt, scene.views[0].up_index)
        player.plt.close()
        return height

    @staticmethod
    def _matplotlib_floor(scene):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from pybvh.bvhplot._matplotlib import frame_mpl

        fig, ax = frame_mpl(scene, Style("paper"), show=False)
        floors = [c for c in ax.collections if c.get_zorder() == 0.5]
        assert len(floors) == 1
        # `_vec` (private, set when the collection is built) is the only
        # handle on world-space vertices; a matplotlib rename fails this
        # test loudly rather than silently weakening it.
        vertices = np.asarray(floors[0]._vec[:3].T)
        height = float(np.unique(vertices[:, scene.views[0].up_index])[0])
        plt.close(fig)
        return height

    def test_all_floor_drawing_backends_sit_at_the_scene_ground(
            self, monkeypatch):
        scene = self._scene()
        view = scene.views[0]
        from pybvh.bvhplot._viewport import make_viewport
        tolerance = 0.005 * make_viewport([view]).half_span  # z-fighting nudges only
        for name, height in (
                ("vedo offscreen", self._offscreen_floor(scene)),
                ("vedo viewer", self._viewer_floor(scene, monkeypatch)),
                ("matplotlib", self._matplotlib_floor(scene))):
            assert abs(height - view.floor_height) < tolerance, name

    def test_no_backend_re_derives_its_own_floor(self, monkeypatch):
        """Move the scene ground and every plane must move with it — the
        property a locally re-derived floor would break."""
        base = self._scene()
        shift = 5.0 * base.views[0].up_sign
        moved = self._scene(floor_height=base.views[0].floor_height - shift)

        for name, before, after in (
                ("vedo offscreen",
                 self._offscreen_floor(base), self._offscreen_floor(moved)),
                ("vedo viewer",
                 self._viewer_floor(base, monkeypatch),
                 self._viewer_floor(moved, monkeypatch)),
                ("matplotlib",
                 self._matplotlib_floor(base), self._matplotlib_floor(moved))):
            assert before - after == pytest.approx(shift, abs=1e-6), name
