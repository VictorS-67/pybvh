"""The Style values sequence(), trajectory() and rest_pose() draw, which
draw with matplotlib alone, and what each does with a Style field the
caller set that it does not draw: raise, since nothing else could draw
it."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
import style_declarations

from pybvh import bvhplot


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _picture(drawn):
    """The pixels of the figure a call returned, drawn on its canvas."""
    fig, _ = drawn
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba()).copy()


class TestEachCallDrawsTheStyleItDeclares:
    """Drawn with two values of a Style field, a call that declares it
    draws another picture, and one that accepts it without drawing it the
    same (see ``tests/style_declarations.py``)."""

    @pytest.mark.parametrize(
        "case",
        style_declarations.declared("trajectory", ["matplotlib"])
        + style_declarations.accepted("trajectory", ["matplotlib"]),
        ids=str,
    )
    def test_trajectory(self, walk, case):
        def picture(style):
            return _picture(bvhplot.trajectory(walk, style=style))

        assert np.array_equal(picture(case.one), picture(case.other)) != case.differ


class TestTheStyleRefused:
    """A Style field the caller set that the call does not draw raises,
    naming the field and, where the call has another layout that draws
    it, that layout."""

    def test_trajectory_refuses_axes(self, walk):
        """A trajectory keeps its axes, ticks and grid by design."""
        with pytest.raises(ValueError) as raised:
            bvhplot.trajectory(walk, style=bvhplot.Style("paper", axes="full"))
        assert str(raised.value) == (
            "trajectory() does not draw the Style's axes='full'. Leave it out of the Style."
        )


class TestTheStyleAccepted:
    @pytest.mark.parametrize("preset", style_declarations.PRESETS)
    @pytest.mark.parametrize("call", ["sequence", "trajectory", "rest_pose"])
    def test_every_preset(self, walk, call, preset):
        """A preset fills every field, drawn by the call or not."""
        for style in (preset, bvhplot.Style(preset)):
            assert _picture(getattr(bvhplot, call)(walk, style=style)).size

    def test_trajectory_takes_a_style_built_for_frame(self, walk):
        """pybvh-qualities' gallery draws its stills in one grey with this
        Style; a trajectory, which holds no bone or joint, takes it and
        draws what it draws of the paper preset."""
        style = bvhplot.Style(
            "paper", color_mode="single", bone_color=(0.62, 0.62, 0.62), joint_color="#8A8A8A"
        )
        bvhplot.frame(walk, style=style)
        drawn = _picture(bvhplot.trajectory(walk, style=style))
        assert np.array_equal(drawn, _picture(bvhplot.trajectory(walk, style="paper")))
