"""The options of play() each backend reads, and what play() does with an
option the backend it runs does not read: raise when the backend is
named, warn under auto, whose choice depends on the session."""

from __future__ import annotations

import base64
import dataclasses
import io
import re
import warnings

import numpy as np
import pytest
import style_declarations
from PIL import Image

from pybvh import bvhplot
from pybvh.bvhplot import _backends
from pybvh.bvhplot._style import STYLE_VALUES, resolve_style

# Every package a play backend imports, so no import check stands in the
# way of the rule.
EVERYTHING = {"k3d", "vedo", "cv2"}

# The Style fields and values each play backend does not draw: the
# projections it lacks; axes on the viewers (k3d always shows its grid
# box, vedo has none), while OpenCV draws "full" as its axis indicator;
# the vedo viewer's joint fields, its joint being the capsules' sphere at
# both qualities; a shadow, which only the vedo renderer casts.
REFUSED = [
    ("k3d", "projection", "ortho"),
    ("k3d", "axes", "off"),
    ("k3d", "axes", "full"),
    ("k3d", "shadow", True),
    ("vedo", "projection", "ortho"),
    ("vedo", "axes", "full"),
    ("vedo", "joint_markers", True),
    ("vedo", "joint_markers", False),
    ("vedo", "joint_size", 20.0),
    ("vedo", "joint_color", "red"),
    ("vedo", "shadow", True),
    ("opencv", "projection", "persp"),
    ("opencv", "shadow", True),
    ("matplotlib", "shadow", True),
]
# How a message names each field: one of a fixed set of values with its
# value, any other by its name alone.
SHOWN = {
    ("projection", "ortho"): "projection='ortho'",
    ("projection", "persp"): "projection='persp'",
    ("axes", "off"): "axes='off'",
    ("axes", "full"): "axes='full'",
    ("joint_markers", True): "joint_markers=True",
    ("joint_markers", False): "joint_markers=False",
    ("joint_size", 20.0): "joint_size",
    ("joint_color", "red"): "joint_color",
    ("shadow", True): "shadow=True",
}

# How to have auto choose each backend, in a session the test builds.
AUTO_CHOOSES = {
    "k3d": lambda environment: environment(notebook=True, importable=EVERYTHING),
    "vedo": lambda environment: environment(display=True, importable=EVERYTHING),
    "opencv": lambda environment: environment(notebook=True, importable={"cv2"}),
    "matplotlib": lambda environment: environment(importable=EVERYTHING),
}


def _refused_as(field, value):
    """A field of a fixed set of values is refused at a value, any
    other at every value."""
    return (field, value) if field in STYLE_VALUES else (field, None)


def _every_field_and_value():
    """Each Style field once, at the paper preset's value, and each
    value of a field of a fixed set of values."""
    paper = bvhplot.Style("paper")
    for field in sorted(f.name for f in dataclasses.fields(paper) if f.compare):
        values = sorted(STYLE_VALUES[field], key=repr) if field in STYLE_VALUES else []
        for value in values or [getattr(paper, field)]:
            yield field, value


class TestPlayChoice:
    """The rule at the backend choice, for every play backend, with
    environments the test builds."""

    def test_a_named_backend_refuses_an_option_it_does_not_read(self, environment):
        with pytest.raises(ValueError) as raised:
            _backends.choose_play_backend(
                "k3d", environment(notebook=True, importable=EVERYTHING), given={"quality"}
            )
        assert str(raised.value) == (
            "play(backend='k3d') does not read quality. Leave it unset, or use "
            "backend='vedo', which reads it."
        )

    def test_the_notebook_video_is_named_opencv_when_refusing(self, environment):
        """It runs as "opencv_notebook", a name no caller writes."""
        with pytest.raises(ValueError) as raised:
            _backends.choose_play_backend(
                "opencv", environment(notebook=True, importable=EVERYTHING), given={"quality"}
            )
        assert str(raised.value) == (
            "play(backend='opencv') does not read quality. Leave it unset, or use "
            "backend='vedo', which reads it."
        )

    def test_the_notebook_video_is_named_opencv_when_offered(self, environment):
        with pytest.raises(ValueError) as raised:
            _backends.choose_play_backend(
                "vedo", environment(notebook=True, importable=EVERYTHING), given={"resolution"}
            )
        assert str(raised.value) == (
            "play(backend='vedo') does not read resolution. Leave it unset, or use "
            "backend='opencv', which reads it."
        )

    @pytest.mark.parametrize(
        "named, given",
        [
            ("k3d", {"spacing", "match_size"}),
            ("vedo", {"quality", "spacing", "match_size"}),
            ("opencv", {"resolution", "frame_counter", "spacing", "match_size"}),
            ("matplotlib", {"spacing", "match_size"}),
        ],
    )
    def test_a_named_backend_takes_what_it_reads_and_the_arrangement(
        self, environment, named, given
    ):
        """A panel backend accepts spacing and match_size: each skeleton
        in its own panel is the arrangement they ask for."""
        choice = _backends.choose_play_backend(
            named, environment(notebook=True, importable=EVERYTHING), given=given
        )
        assert choice.warnings == ()

    def test_a_named_backend_names_every_option_it_does_not_read(self, environment):
        """No backend reads quality and resolution both, so none is
        offered."""
        with pytest.raises(ValueError) as raised:
            _backends.choose_play_backend(
                "matplotlib",
                environment(notebook=True, importable=EVERYTHING),
                given={"resolution", "quality", "frame_counter"},
            )
        assert str(raised.value) == (
            "play(backend='matplotlib') does not read frame_counter, quality or "
            "resolution. Leave them unset."
        )

    def test_outside_a_notebook_the_notebook_video_is_not_offered(self, environment):
        """It plays only in a notebook, so naming it in a script would
        raise."""
        with pytest.raises(ValueError) as raised:
            _backends.choose_play_backend(
                "matplotlib",
                environment(display=True, importable=EVERYTHING),
                given={"frame_counter"},
            )
        assert str(raised.value) == (
            "play(backend='matplotlib') does not read frame_counter. Leave it unset."
        )

    def test_auto_warns_naming_the_option_and_the_backend_chosen(self, environment):
        choice = _backends.choose_play_backend(
            "auto", environment(notebook=True, importable=EVERYTHING), given={"resolution"}
        )
        assert choice.backend == "k3d"
        assert choice.warnings == (
            "play() chose backend='k3d', which does not read resolution, so it is "
            "ignored. To draw it, use backend='opencv', which reads it.",
        )

    def test_auto_names_the_notebook_video_opencv_after_its_fallback_warning(self, environment):
        choice = _backends.choose_play_backend(
            "auto", environment(notebook=True, importable={"cv2"}), given={"quality"}
        )
        assert choice.backend == "opencv_notebook"
        assert choice.warnings == (
            "No interactive backend (k3d, vedo) found. Install with: pip install "
            "pybvh[interactive]",
            "play() chose backend='opencv', which does not read quality, so it is "
            "ignored. To draw it, use backend='vedo', which reads it.",
        )

    def test_auto_names_every_option_ignored_when_no_backend_reads_them_all(self, environment):
        """Headless, auto plays in matplotlib's window, which reads
        neither."""
        choice = _backends.choose_play_backend(
            "auto", environment(importable=EVERYTHING), given={"quality", "resolution"}
        )
        assert choice.backend == "matplotlib"
        assert choice.warnings == (
            "play() chose backend='matplotlib', which does not read quality or resolution, "
            "so they are ignored.",
        )

    def test_auto_in_a_script_offers_no_notebook_video(self, environment):
        choice = _backends.choose_play_backend(
            "auto", environment(display=True, importable=EVERYTHING), given={"resolution"}
        )
        assert choice.backend == "vedo"
        assert choice.warnings == (
            "play() chose backend='vedo', which does not read resolution, so it is ignored.",
        )


class TestPlayChoiceAndTheStyle:
    """The rule for a Style field the caller set, at the backend choice,
    for every play backend: the k3d widget and the vedo window cannot
    run in the suite."""

    def test_a_named_backend_refuses_a_field_it_does_not_draw(self, environment):
        with pytest.raises(ValueError) as raised:
            _backends.choose_play_backend(
                "vedo",
                environment(notebook=True, importable=EVERYTHING),
                style=bvhplot.Style("paper", joint_color="red"),
            )
        assert str(raised.value) == (
            "play(backend='vedo') does not draw the Style's joint_color. Leave it out of "
            "the Style, or use backend='k3d', 'matplotlib' or 'opencv', which draw it."
        )

    @pytest.mark.parametrize("named, field, value", REFUSED, ids=str)
    def test_a_named_backend_refuses_each_field_it_does_not_draw(
        self, environment, named, field, value
    ):
        with pytest.raises(ValueError, match=rf"^play\(backend='{named}'\) does not draw "):
            _backends.choose_play_backend(
                named,
                environment(notebook=True, importable=EVERYTHING),
                style=bvhplot.Style("paper", **{field: value}),
            )

    @pytest.mark.parametrize("named, field, value", REFUSED, ids=str)
    def test_auto_warns_of_each_field_its_choice_does_not_draw(
        self, environment, named, field, value
    ):
        choice = _backends.choose_play_backend(
            "auto", AUTO_CHOOSES[named](environment), style=bvhplot.Style("paper", **{field: value})
        )
        assert _backends._PLAY_NAMES.get(choice.backend, choice.backend) == named
        assert choice.warnings[-1].startswith(
            f"play() chose backend='{named}', which does not draw the Style's "
            f"{SHOWN[field, value]}, "
            "so it is ignored."
        )

    @pytest.mark.parametrize("named", sorted(AUTO_CHOOSES))
    def test_a_named_backend_takes_every_other_field_without_a_word(self, environment, named):
        """Each field set alone, at the paper preset's value, or at each
        value of a field of a fixed set of values."""
        refused = [
            _refused_as(field, value) for backend, field, value in REFUSED if backend == named
        ]
        for field, value in _every_field_and_value():
            if _refused_as(field, value) in refused:
                continue
            choice = _backends.choose_play_backend(
                named,
                environment(notebook=True, importable=EVERYTHING),
                style=bvhplot.Style("paper", **{field: value}),
            )
            assert choice.warnings == (), (field, value)

    def test_auto_warns_offering_the_backends_that_draw_the_field(self, environment):
        choice = _backends.choose_play_backend(
            "auto",
            environment(notebook=True, importable=EVERYTHING),
            style=bvhplot.Style("paper", projection="ortho"),
        )
        assert choice.backend == "k3d"
        assert choice.warnings == (
            "play() chose backend='k3d', which does not draw the Style's "
            "projection='ortho', so it is ignored. To draw it, use "
            "backend='matplotlib' or 'opencv', which draw it.",
        )

    def test_auto_warns_of_a_joint_color_given_as_a_list(self, environment):
        """A color matplotlib parses from a list, which no set can hold,
        is ignored by the vedo window as any other joint color."""
        choice = _backends.choose_play_backend(
            "auto",
            AUTO_CHOOSES["vedo"](environment),
            style=bvhplot.Style("paper", joint_color=[1.0, 0.0, 0.0]),
        )
        assert choice.backend == "vedo"
        assert choice.warnings == (
            "play() chose backend='vedo', which does not draw the Style's joint_color, "
            "so it is ignored. To draw it, use backend='k3d' or 'matplotlib', which draw it.",
        )

    @pytest.mark.parametrize("named", sorted(AUTO_CHOOSES))
    @pytest.mark.parametrize("preset", style_declarations.PRESETS)
    def test_accepts_every_preset(self, environment, named, preset):
        """A preset's own values are never refused, though no play
        backend draws the paper preset's shadow."""
        for style in (preset, bvhplot.Style(preset)):
            choice = _backends.choose_play_backend(
                named,
                environment(notebook=True, importable=EVERYTHING),
                style=resolve_style(style),
            )
            assert choice.warnings == ()

    @pytest.mark.parametrize("named", sorted(AUTO_CHOOSES))
    def test_accepts_ghost_spacing_and_dpi(self, environment, named):
        """play draws no ghosts, and a picture whose pixels are set
        otherwise has no dpi."""
        choice = _backends.choose_play_backend(
            named,
            environment(notebook=True, importable=EVERYTHING),
            style=bvhplot.Style("paper", ghost_spacing=0.5, dpi=300),
        )
        assert choice.warnings == ()


@pytest.fixture
def windowed(monkeypatch):
    """The options each play() call handed matplotlib's window, which
    is stubbed out so that none opens."""
    import pybvh.bvhplot._matplotlib as mpl_backend

    calls = []
    monkeypatch.setattr(mpl_backend, "play_mpl", lambda *args, **kwargs: calls.append(kwargs))
    return calls


@pytest.fixture
def shown(monkeypatch):
    """Each object play() hands a notebook to display."""
    ipython_display = pytest.importorskip("IPython.display")
    objects = []
    monkeypatch.setattr(ipython_display, "display", objects.append)
    return objects


def _play_warnings(call):
    """The messages of the warnings *call* raises that the rule wrote,
    each with the file it names."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        call()
    return [(str(w.message), w.filename) for w in caught if "play()" in str(w.message)]


class TestThroughPlay:
    """The rule through play() itself, on the paths that open no window,
    in sessions the test builds."""

    def test_a_named_backend_refuses_before_playing(self, walk, environment, windowed):
        environment()
        with pytest.raises(ValueError) as raised:
            bvhplot.play(walk, backend="matplotlib", quality="fast")
        assert str(raised.value) == (
            "play(backend='matplotlib') does not read quality. Leave it unset, or use "
            "backend='vedo', which reads it."
        )
        assert windowed == []

    def test_the_notebook_video_refuses_an_option_before_asking_for_opencv(self, walk, environment):
        """As render() does: the caller learns the options cannot run
        together before being told to install."""
        environment(notebook=True)
        with pytest.raises(ValueError) as raised:
            bvhplot.play(walk, backend="opencv", quality="fast")
        assert str(raised.value) == (
            "play(backend='opencv') does not read quality. Leave it unset, or use "
            "backend='vedo', which reads it."
        )

    def test_the_notebook_video_without_opencv_raises_naming_the_install(self, walk, environment):
        environment(notebook=True)
        with pytest.raises(ImportError) as raised:
            bvhplot.play(walk, backend="opencv")
        assert str(raised.value) == (
            "The opencv play backend requires opencv-python. "
            "Install with: pip install pybvh[opencv]"
        )

    def test_the_notebook_video_in_a_script_raises_before_asking_for_opencv(
        self, walk, environment
    ):
        """No install would let it play there."""
        environment(display=True)
        with pytest.raises(ValueError, match=r"^backend='opencv' plays an inline video"):
            bvhplot.play(walk, backend="opencv")

    def test_given_means_not_none(self, walk, environment, windowed):
        """frame_counter=False is its default, given all the same."""
        environment(notebook=True)
        with pytest.raises(
            ValueError, match=r"^play\(backend='matplotlib'\) does not read frame_counter\."
        ):
            bvhplot.play(walk, backend="matplotlib", frame_counter=False)

    def test_auto_warns_at_the_users_line_and_plays(self, walk, environment, windowed):
        """Headless, auto plays in matplotlib's window, which has no
        quality to set."""
        environment(importable={"vedo"})
        assert _play_warnings(lambda: bvhplot.play(walk, quality="fast")) == [
            (
                "play() chose backend='matplotlib', which does not read quality, so it is "
                "ignored. To draw it, use backend='vedo', which reads it.",
                __file__,
            )
        ]
        assert len(windowed) == 1

    def test_the_arrangement_is_accepted_by_a_panel_backend(self, walk, environment, windowed):
        environment()
        bvhplot.play([walk, walk], backend="matplotlib", spacing=1.0, match_size=True)
        assert len(windowed) == 1


class TestThroughPlayAndTheStyle:
    """The rule for a Style field through play() itself, on the paths
    that open no window."""

    def test_a_named_backend_refuses_before_playing(self, walk, environment, windowed):
        environment()
        with pytest.raises(ValueError) as raised:
            bvhplot.play(walk, backend="matplotlib", style=bvhplot.Style("paper", shadow=True))
        assert str(raised.value) == (
            "play(backend='matplotlib') does not draw the Style's shadow=True. "
            "Leave it out of the Style."
        )
        assert windowed == []

    def test_auto_warns_at_the_users_line_and_plays(self, walk, environment, windowed):
        """Headless, auto plays in matplotlib's window, which draws no
        shadow, as no play backend does."""
        environment(importable={"vedo"})
        style = bvhplot.Style("paper", shadow=True)
        assert _play_warnings(lambda: bvhplot.play(walk, style=style)) == [
            (
                "play() chose backend='matplotlib', which does not draw the Style's "
                "shadow=True, so it is ignored.",
                __file__,
            )
        ]
        assert len(windowed) == 1


def _jshtml_frames(page):
    """Every frame of matplotlib's notebook player, a page holding each
    as a PNG, as ``(H, W, 3)`` uint8 RGB arrays."""
    pngs = re.findall(r'frames\[\d+\] = "data:image/png;base64,([^"]+)"', page)
    assert pngs
    return [
        np.asarray(Image.open(io.BytesIO(base64.b64decode(png))).convert("RGB")) for png in pngs
    ]


class TestEachBackendDrawsTheStyleItDeclares:
    """Played with two values of a Style field, matplotlib's notebook
    player and the notebook video draw other frames where the backend
    declares the field drawn, and the same where it accepts the field
    without drawing it (see ``tests/style_declarations.py``). The one
    frame of the walk shows a field."""

    @pytest.mark.parametrize(
        "case",
        style_declarations.cases("play", ["matplotlib", "opencv_notebook"]),
        ids=str,
    )
    def test_a_field_declared_changes_the_frames_and_one_accepted_does_not(
        self, walk, environment, shown, handed_to_writer, same_frames, case
    ):
        if case.backend == "opencv_notebook":
            pytest.importorskip("cv2")
        environment(notebook=True, importable={"cv2"})

        def frames(style):
            if case.backend == "matplotlib":
                bvhplot.play(walk[:1], backend="matplotlib", fps=10, style=style)
                return _jshtml_frames(shown[-1].data)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # the MPEG-4 Part 2 fallback
                bvhplot.play(walk[:1], backend="opencv", fps=10, resolution=(96, 72), style=style)
            return handed_to_writer[-1]

        assert same_frames(frames(case.one), frames(case.other)) != case.differ


class TestTheNotebookVideo:
    """The one play backend whose frames a test can read: the OpenCV
    video in a notebook, played for real with its display captured."""

    @pytest.fixture
    def played(self, walk, environment, shown, handed_to_writer):
        """The frames the notebook video of the walk (or of *clips*) is
        drawn as, with the options given; the session lacks k3d, so auto
        chooses it."""
        pytest.importorskip("cv2")
        environment(notebook=True, importable={"cv2"})

        def play(clips=None, **options):
            with warnings.catch_warnings():
                # the MPEG-4 Part 2 fallback, and the rule's own warnings,
                # which other tests check
                warnings.simplefilter("ignore")
                bvhplot.play(walk if clips is None else clips, fps=10, **options)
            return handed_to_writer[-1]

        return play

    def test_it_reads_the_resolution_given(self, played):
        assert played(resolution=(64, 48))[0].shape == (48, 64, 3)

    def test_its_resolution_defaults_to_960_by_540(self, played):
        assert played()[0].shape == (540, 960, 3)

    def test_it_reads_the_frame_counter(self, played, same_frames):
        without = played(resolution=(96, 72))
        given = played(resolution=(96, 72), frame_counter=True)
        assert len(given) == len(without)
        assert not same_frames(given, without)

    def test_quality_leaves_its_frames_alone(self, played, same_frames):
        """Under auto it warns, and the video is the same."""
        assert same_frames(played(resolution=(96, 72), quality="fast"), played(resolution=(96, 72)))

    @pytest.mark.parametrize("option", [{"spacing": 1.0}, {"match_size": True}], ids=str)
    def test_the_arrangement_leaves_its_frames_alone(self, played, pair, same_frames, option):
        """Each skeleton in its own panel is the arrangement asked for;
        the second is twice the first's size, so a matched size would
        show."""
        assert same_frames(
            played(pair, resolution=(96, 72), **option), played(pair, resolution=(96, 72))
        )

    def test_auto_reads_its_options_without_a_warning(self, walk, environment, shown):
        pytest.importorskip("cv2")
        environment(notebook=True, importable={"cv2"})
        assert (
            _play_warnings(
                lambda: bvhplot.play(walk, fps=10, resolution=(64, 48), frame_counter=True)
            )
            == []
        )
        assert len(shown) == 1
