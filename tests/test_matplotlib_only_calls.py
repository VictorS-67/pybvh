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

LAYOUTS = ["offset", "overlay"]


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

    @pytest.mark.parametrize(
        "case",
        style_declarations.declared("sequence", LAYOUTS)
        + style_declarations.accepted("sequence", LAYOUTS),
        ids=str,
    )
    def test_sequence(self, walk, case):
        """Each layout stands where another call's backend stands."""

        def picture(style):
            return _picture(bvhplot.sequence(walk, n_poses=3, layout=case.backend, style=style))

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

    def test_sequence_offset_refuses_a_perspective(self, walk):
        """The offset layout is orthographic by design: perspective would
        shrink the far poses and break the left-to-right read."""
        with pytest.raises(ValueError) as raised:
            bvhplot.sequence(walk, style=bvhplot.Style("paper", projection="persp"))
        assert str(raised.value) == (
            "sequence(layout='offset') does not draw the Style's projection='persp'. "
            "Leave it out of the Style, or use layout='overlay', which draws it."
        )

    def test_sequence_offers_no_layout_where_none_draws_the_field(self, walk):
        with pytest.raises(ValueError) as raised:
            bvhplot.sequence(walk, layout="overlay", style=bvhplot.Style("paper", shadow=False))
        assert str(raised.value) == (
            "sequence(layout='overlay') does not draw the Style's shadow. "
            "Leave it out of the Style."
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
