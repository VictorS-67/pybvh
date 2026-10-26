"""The options of frame() each backend reads, and what frame() does with
an option the backend it runs does not read: raise, since frame() has no
auto and its backend is always named (``"matplotlib"`` by default)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from pybvh import bvhplot, read_bvh_file

BVH_DIR = Path(__file__).parent.parent / "bvh_data"


def _axes():
    """A 3D axes on a figure of its own."""
    return plt.figure().add_subplot(projection="3d")


def _given(option):
    """A value of *option* for frame(), other than None."""
    values = {"figsize": (4, 3), "show": True, "resolution": (400, 300)}
    return {option: _axes() if option == "ax" else values[option]}


@pytest.fixture(scope="module")
def walk():
    return read_bvh_file(BVH_DIR / "cmu_12_01_walk.bvh")


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


class TestANamedBackend:
    def test_refuses_an_option_it_does_not_read(self, walk, tmp_path):
        path = tmp_path / "x.png"
        with pytest.raises(ValueError) as raised:
            bvhplot.frame(walk, backend="vedo", figsize=(4, 3), filepath=path)
        assert str(raised.value) == (
            "frame(backend='vedo') does not read figsize. Leave it unset, "
            "or use backend='matplotlib', which reads it."
        )
        assert not path.exists()

    def test_refuses_an_option_it_does_not_read_even_at_its_default(self, walk):
        """Given means not None, and matplotlib, the default backend, is
        named: resolution at its vedo default is given."""
        with pytest.raises(ValueError) as raised:
            bvhplot.frame(walk, resolution=(1100, 1000))
        assert str(raised.value) == (
            "frame(backend='matplotlib') does not read resolution. Leave it "
            "unset, or use backend='vedo', which reads it."
        )

    @pytest.mark.parametrize(
        "backend, option",
        [
            ("vedo", "figsize"),
            ("vedo", "show"),
            ("vedo", "ax"),
            ("matplotlib", "resolution"),
        ],
    )
    def test_refuses_each_option_it_does_not_read(self, walk, tmp_path, backend, option):
        path = tmp_path / "x.png"
        with pytest.raises(
            ValueError, match=rf"^frame\(backend='{backend}'\) does not read {option}\."
        ):
            bvhplot.frame(walk, backend=backend, filepath=path, **_given(option))
        assert not path.exists()

    def test_names_every_option_it_does_not_read(self, walk):
        with pytest.raises(ValueError) as raised:
            bvhplot.frame(walk, backend="vedo", figsize=(4, 3), show=False, ax=_axes())
        assert str(raised.value) == (
            "frame(backend='vedo') does not read ax, figsize or show. Leave them "
            "unset, or use backend='matplotlib', which reads them."
        )
