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
    """The G look: plump body capsules, slender articulated hands."""

    def test_hand_bone_set_starts_at_wrist(self, bvh):
        from pybvh.bvhplot._vedo_capsules import hand_bone_set
        from pybvh.bvhplot import get_skeleton_lines
        bones = get_skeleton_lines(bvh)
        hand = hand_bone_set(bvh, bones)
        names = {idx: n for n, idx in bvh.node_index.items()}
        assert hand, "walk skeleton has named hand joints"
        # every hand bone's parent is a hand joint...
        assert all("hand" in names[p].lower() for p, _c in hand)
        # ...and the forearm bone ENDING at the wrist is a body bone
        forearm = [(p, c) for p, c in bones
                   if "hand" in names[c].lower()
                   and "hand" not in names[p].lower()]
        assert forearm and not (set(forearm) & hand)

    def test_hand_bones_render_slimmer(self):
        from pybvh.bvhplot._vedo_capsules import adaptive_radii
        frame0 = np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]])
        bones = [(0, 1), (1, 2)]          # two bones, identical length
        body, _ = adaptive_radii(frame0, bones, r_base=1.0)
        mixed, _ = adaptive_radii(frame0, bones, r_base=1.0,
                                  hand_bones=frozenset({(1, 2)}))
        assert mixed[(0, 1)] == body[(0, 1)]
        assert mixed[(1, 2)] == pytest.approx(body[(1, 2)] * 0.5)

    def test_no_named_hands_falls_back_to_body_sizing(self, bvh):
        from pybvh.bvhplot._vedo_capsules import hand_bone_set
        from pybvh.bvhplot import get_skeleton_lines
        sub = bvh.extract_joints(
            ["Hips", "LowerBack", "Spine", "Spine1", "Neck", "Neck1",
             "Head"])
        assert hand_bone_set(sub, get_skeleton_lines(sub)) == frozenset()
