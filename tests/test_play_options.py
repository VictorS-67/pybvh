"""The options of play() each backend reads, and what play() does with an
option the backend it runs does not read: raise when the backend is
named, warn under auto, whose choice depends on the session."""

from __future__ import annotations

import warnings

import pytest

from pybvh import bvhplot
from pybvh.bvhplot import _backends

# Every package a play backend imports, so no import check stands in the
# way of the rule.
EVERYTHING = {"k3d", "vedo", "cv2"}


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
