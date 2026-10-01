"""Tests for the offscreen vedo backend (Phase 4)."""
from __future__ import annotations

import numpy as np
import pytest

vedo = pytest.importorskip("vedo")

from pybvh import bvhplot, read_bvh_file
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

    def test_a_clip_of_coords_renders_like_its_first_frame(self, bvh):
        """frame() draws only the first row of an (F, N, 3) array, so
        the other rows must not reach the camera or the floor."""
        coords = bvh.node_positions()
        from_clip = bvhplot.frame(bvh, coords=coords, backend="vedo",
                                  resolution=(300, 280))
        from_pose = bvhplot.frame(bvh, coords=coords[0], backend="vedo",
                                  resolution=(300, 280))
        assert np.array_equal(from_clip, from_pose)

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
        from pybvh.bvhplot import get_skeleton_lines
        from pybvh.bvhplot._vedo_capsules import adaptive_radii
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
        from pybvh.bvhplot import get_skeleton_lines
        from pybvh.bvhplot._vedo_capsules import adaptive_radii, base_radius
        bones = get_skeleton_lines(bvh_hands)
        rest = bvh_hands.rest_pose_positions()
        # The body size: this rest pose stands along y.
        r_base = base_radius(float(np.ptp(rest[:, 1])), 3.0)
        radii, _ = adaptive_radii(rest, bones, r_base, rest)
        idx = bvh_hands.node_index
        forearm = radii[(idx["RightForeArm"], idx["RightHand"])]
        upper = radii[(idx["RightArm"], idx["RightForeArm"])]
        assert forearm == pytest.approx(upper, rel=0.25)

    def test_fingers_are_thinned_without_naming_them(self, bvh_hands):
        from pybvh.bvhplot import get_skeleton_lines
        from pybvh.bvhplot._vedo_capsules import adaptive_radii, base_radius
        bones = get_skeleton_lines(bvh_hands)
        rest = bvh_hands.rest_pose_positions()
        # The body size: this rest pose stands along y.
        r_base = base_radius(float(np.ptp(rest[:, 1])), 3.0)
        radii, _ = adaptive_radii(rest, bones, r_base, rest)
        idx = bvh_hands.node_index
        upper = radii[(idx["RightArm"], idx["RightForeArm"])]
        finger = radii[(idx["RightHandPinky1"], idx["RightHandPinky2"])]
        assert finger < upper / 3

    def test_short_isolated_bones_keep_full_radius(self):
        """The neck bug: short links with nothing beside them are not
        thinned, so a two-link neck matches the spine below it."""
        from pybvh import read_bvh_file
        from pybvh.bvhplot import get_skeleton_lines
        from pybvh.bvhplot._vedo_capsules import adaptive_radii
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


class TestCapsuleShading:
    """VTK shades a capsule from the normals it reads off the mesh, so
    those must follow each bone to its pose."""

    def test_bone_normals_point_out_of_each_posed_bone(self):
        """A bone along x, one along y and one along z (the canonical
        tube's own axis, the one case the unrotated normals got right),
        plus one pointing back along -x. Every normal is of unit length;
        on the tube's wall it is orthogonal to the bone and points away
        from its axis, on the end caps it points along the bone, out of
        the tube."""
        from synthetic_scene import make_bare_view
        from vtk.util.numpy_support import vtk_to_numpy

        from pybvh.bvhplot._vedo_capsules import CapsuleSkeleton

        pose = np.array([[0.0, 0.0, 0.0],
                         [1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0],
                         [0.0, 0.0, 1.0],
                         [-0.7, 0.0, 0.0]])
        bones = [(0, 1), (0, 2), (0, 3), (0, 4)]
        view = make_bare_view(pose[np.newaxis], pose, bones)
        capsule = CapsuleSkeleton(
            view, 1.0, [(200, 100, 50)] * len(bones),
            np.full((len(pose), 3), 128, dtype=np.uint8))
        capsule.update(pose)

        mesh = capsule.bones_mesh.dataset
        normals = vtk_to_numpy(mesh.GetPointData().GetNormals())
        normals = normals.reshape(len(bones), -1, 3)
        vertices = capsule.bones_mesh.vertices.reshape(len(bones), -1, 3)
        for k, (parent, child) in enumerate(bones):
            start, end = pose[parent], pose[child]
            length = np.linalg.norm(end - start)
            direction = (end - start) / length
            along = (vertices[k] - start) @ direction
            radial = (vertices[k] - start) - along[:, None] * direction
            n = normals[k]
            np.testing.assert_allclose(
                np.linalg.norm(n, axis=1), 1.0, atol=1e-5)

            cosine = n @ direction
            on_wall = np.abs(cosine) < 1e-5
            on_cap = np.abs(np.abs(cosine) - 1.0) < 1e-5
            assert (on_wall | on_cap).all(), f"bone {k}"
            assert on_wall.any() and on_cap.any(), f"bone {k}"
            outward = radial[on_wall] / np.linalg.norm(
                radial[on_wall], axis=1, keepdims=True)
            np.testing.assert_allclose(n[on_wall], outward, atol=1e-5)
            at_end = along[on_cap] > length / 2
            np.testing.assert_array_equal(cosine[on_cap] > 0, at_end)

    def test_a_second_pose_is_shaded_like_a_fresh_one(self):
        """Playback updates a skeleton whose first pose VTK has already
        drawn: the new normals must reach the renderer, not only the
        mesh. Two bones upright, then turned sideways, must render as
        a skeleton built sideways, whose first pose that is."""
        from synthetic_scene import make_bare_view

        from pybvh.bvhplot._scene import Scene
        from pybvh.bvhplot._vedo_offscreen import _build_offscreen, _vtk_backend

        upright = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                            [0.3, 0.0, 0.0], [0.3, 1.0, 0.0]])
        # Same bone lengths, since frame 0 sizes the capsules.
        sideways = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                             [0.3, 0.0, 0.0], [0.3, 0.0, 1.0]])

        def render(first_pose, then=None):
            """The skeleton rendered in its first pose, and again after
            an update to *then*. Both clips hold the same two poses,
            so the camera, framed on the whole clip, is the same."""
            second_pose = sideways if first_pose is upright else upright
            view = make_bare_view(np.stack([first_pose, second_pose]),
                                  upright, [(0, 1), (2, 3)])
            with _vtk_backend():
                plt, [capsule], camera = _build_offscreen(
                    Scene(views=[view]), Style("paper", floor=None),
                    (240, 240))
                try:
                    plt.show(camera=camera, interactive=False)
                    images = [plt.screenshot(asarray=True)]
                    if then is not None:
                        capsule.update(then)
                        plt.render()
                        images.append(plt.screenshot(asarray=True))
                finally:
                    plt.close()
            return [np.asarray(image).astype(int) for image in images]

        first, then_sideways = render(upright, then=sideways)
        [built_sideways] = render(sideways)
        assert np.abs(first - built_sideways).max() > 50
        np.testing.assert_allclose(then_sideways, built_sideways, atol=2)


class TestLabels:
    def test_each_label_is_drawn_in_its_skeletons_color(self, bvh):
        """The labels were handed to vedo as "rgb(r,g,b)" strings,
        which vedo reads as black."""
        img = bvhplot.frame([bvh, bvh.mirror()], 0, backend="vedo",
                            labels=["walk", "mirror"],
                            resolution=(400, 360))
        header = img[:60].astype(int)   # the labels sit at the top left
        for rgb in [(50, 120, 255), (220, 50, 50)]:   # the palette's first two
            close = np.abs(header - rgb).max(axis=-1) <= 3
            assert close.sum() > 10


class TestStyleColorsReachVedoParsed:
    """The background is read by matplotlib's parser, as in the other
    backends: vedo's own crashed on short hex ("#fff") and read
    matplotlib-only names such as "C0" as gray."""

    @pytest.mark.parametrize("background, expected", [
        ("#fff", (255, 255, 255)),
        ("C0", (31, 119, 180)),   # matplotlib's first cycle color
    ])
    def test_background(self, bvh, background, expected):
        img = bvhplot.frame(bvh, 0, backend="vedo",
                            style=Style("paper", background=background,
                                        floor=None),
                            resolution=(200, 180))
        np.testing.assert_allclose(img[0, 0], expected, atol=1)
