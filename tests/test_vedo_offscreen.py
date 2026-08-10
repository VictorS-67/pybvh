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
