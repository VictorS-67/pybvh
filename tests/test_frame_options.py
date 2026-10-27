"""The options of frame() each backend reads, and what frame() does with
an option the backend it runs does not read: raise, since frame() has no
auto and its backend is always named (``"matplotlib"`` by default)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pybvh import bvhplot
from pybvh.bvhplot import _backends

# The package each backend draws with, skipped without it.
PACKAGE = {"matplotlib": "matplotlib", "vedo": "vedo"}
# Every option each backend declares it reads, as the rule reads the
# declarations: each must have its check in EFFECTS below.
DECLARED = [
    (backend, option)
    for backend in sorted(PACKAGE)
    for option in sorted(_backends._reads("frame", backend))
]
# A small picture for vedo, so that each still is quick.
SMALL = (160, 120)


def _axes():
    """A 3D axes on a figure of its own."""
    return plt.figure().add_subplot(projection="3d")


def _a_value_of(option):
    """A value of *option* for frame(), other than None."""
    values = {"figsize": (4, 3), "show": True, "resolution": (400, 300)}
    return {option: _axes() if option == "ax" else values[option]}


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _picture(drawn):
    """The pixels of what ``frame`` returned: vedo's image, or
    matplotlib's figure drawn on its canvas."""
    if isinstance(drawn, np.ndarray):
        return drawn
    fig, _ = drawn
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba()).copy()


def _figsize_sizes_the_figure(walk, pair, monkeypatch):
    fig, _ = bvhplot.frame(walk, figsize=(4, 3))
    assert tuple(fig.get_size_inches()) == (4, 3)
    fig, _ = bvhplot.frame(walk)
    assert tuple(fig.get_size_inches()) != (4, 3)


def _show_shows_the_figure(walk, pair, monkeypatch):
    shown = []
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: shown.append(True))
    bvhplot.frame(walk)
    assert shown == []
    bvhplot.frame(walk, show=True)
    assert shown == [True]


def _ax_is_drawn_on(walk, pair, monkeypatch):
    ax = _axes()
    empty = _picture((ax.figure, ax))
    fig, returned = bvhplot.frame(walk, ax=ax)
    assert returned is ax and fig is ax.figure
    assert not np.array_equal(_picture((fig, ax)), empty)


def _resolution_sizes_the_image(walk, pair, monkeypatch):
    assert bvhplot.frame(walk, backend="vedo", resolution=SMALL).shape == (120, 160, 3)
    # Left unset, the documented default: 1100 wide and 1000 tall.
    assert bvhplot.frame(walk, backend="vedo").shape == (1000, 1100, 3)


def _spacing_spreads_the_skeletons(walk, pair, monkeypatch):
    given = bvhplot.frame(pair, backend="vedo", resolution=SMALL, spacing=1.0)
    assert not np.array_equal(given, bvhplot.frame(pair, backend="vedo", resolution=SMALL))


def _match_size_scales_the_skeletons(walk, pair, monkeypatch):
    given = bvhplot.frame(pair, backend="vedo", resolution=SMALL, match_size=True)
    assert not np.array_equal(given, bvhplot.frame(pair, backend="vedo", resolution=SMALL))


# What each declared option is checked to do, through frame().
EFFECTS = {
    ("matplotlib", "figsize"): _figsize_sizes_the_figure,
    ("matplotlib", "show"): _show_shows_the_figure,
    ("matplotlib", "ax"): _ax_is_drawn_on,
    ("vedo", "resolution"): _resolution_sizes_the_image,
    ("vedo", "spacing"): _spacing_spreads_the_skeletons,
    ("vedo", "match_size"): _match_size_scales_the_skeletons,
}


class TestEachBackendReadsWhatItDeclares:
    @pytest.mark.parametrize("backend, option", DECLARED)
    def test_an_option_declared_takes_effect(self, walk, pair, monkeypatch, backend, option):
        """Named, so the rule lets every declared option through. An
        option added to a declaration fails here until EFFECTS checks
        what it does."""
        pytest.importorskip(PACKAGE[backend])
        EFFECTS[backend, option](walk, pair, monkeypatch)

    @pytest.mark.parametrize("option", [{"spacing": 1.0}, {"match_size": True}], ids=str)
    def test_matplotlib_accepts_spacing_and_match_size_and_draws_the_same(self, pair, option):
        """Each skeleton in its own panel is the arrangement they ask
        for."""
        given = _picture(bvhplot.frame(pair, **option))
        assert np.array_equal(given, _picture(bvhplot.frame(pair)))


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
            bvhplot.frame(walk, backend=backend, filepath=path, **_a_value_of(option))
        assert not path.exists()

    def test_names_every_option_it_does_not_read(self, walk):
        with pytest.raises(ValueError) as raised:
            bvhplot.frame(walk, backend="vedo", figsize=(4, 3), show=False, ax=_axes())
        assert str(raised.value) == (
            "frame(backend='vedo') does not read ax, figsize or show. Leave them "
            "unset, or use backend='matplotlib', which reads them."
        )
