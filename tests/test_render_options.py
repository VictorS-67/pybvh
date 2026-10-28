"""The options of render() each backend reads and the Style values it
draws, and what render() does with an option the backend it runs does
not read, or a Style field the caller set that it does not draw: raise
when the backend is named, warn under auto. No option reaches auto's
warning, since auto chooses OpenCV or matplotlib, which read the same
options; a Style field does (OpenCV draws no "persp", neither a shadow)."""

from __future__ import annotations

import pytest
import style_declarations

from pybvh import bvhplot
from pybvh.bvhplot import _backends

# Each option of render() that some backend does not read, as given.
OPTIONS = {
    "follow": {"follow": True},
    "turntable": {"camera": "turntable"},
    "ghost": {"ghost": 2},
    "trajectory": {"trajectory": True},
    "frame_counter": {"frame_counter": True},
    "spacing": {"spacing": 1.0},
    "match_size": {"match_size": True},
}
FPS = 10
# The package each backend draws with, skipped without it.
PACKAGE = {"opencv": "cv2", "matplotlib": "matplotlib", "vedo": "vedo"}
# Every option each backend declares it reads, as the rule reads the
# declarations: each must have its value in OPTIONS.
DECLARED = [
    (backend, option)
    for backend in sorted(PACKAGE)
    for option in sorted(_backends._reads("render", backend))
]


@pytest.fixture
def drawn(walk, pair, tmp_path, apng_frames):
    """The frames ``render`` writes to an ``.apng`` of the walk (of the
    pair, for ``spacing`` and ``match_size``) on *backend*, with the
    options given, read back."""

    def draw(backend, option, **options):
        clips = pair if option in ("spacing", "match_size") else walk
        path = tmp_path / f"{backend}-{option}-{len(options)}.apng"
        bvhplot.render(clips, path, backend=backend, resolution=(96, 72), fps=FPS, **options)
        return apng_frames(path, FPS)

    return draw


class TestEachBackendReadsWhatItDeclares:
    """Rendered with and without an option, a backend that reads it
    writes other frames, and one that does not writes the same."""

    @pytest.mark.parametrize("backend, option", DECLARED)
    def test_an_option_declared_changes_the_frames(self, drawn, same_frames, backend, option):
        """Named, so the rule lets every declared option through."""
        pytest.importorskip(PACKAGE[backend])
        without = drawn(backend, option)
        given = drawn(backend, option, **OPTIONS[option])
        assert len(given) == len(without)
        assert not same_frames(given, without)

    @pytest.mark.parametrize(
        "backend, option",
        [
            ("opencv", "spacing"),
            ("opencv", "match_size"),
            ("matplotlib", "spacing"),
            ("matplotlib", "match_size"),
        ],
    )
    def test_an_option_a_panel_backend_does_not_read_leaves_the_frames_alone(
        self, drawn, same_frames, backend, option
    ):
        """Named, it is accepted: each skeleton in its own panel is the
        arrangement it asks for."""
        pytest.importorskip(PACKAGE[backend])
        assert same_frames(drawn(backend, option, **OPTIONS[option]), drawn(backend, option))


class TestEachBackendDrawsTheStyleItDeclares:
    """Rendered with two values of a Style field, a backend that declares
    it draws other frames, and one that accepts it without drawing it the
    same (see ``tests/style_declarations.py``)."""

    @pytest.mark.parametrize(
        "case",
        style_declarations.declared("render", sorted(PACKAGE))
        + style_declarations.accepted("render", sorted(PACKAGE)),
        ids=str,
    )
    def test_a_field_declared_changes_the_frames_and_one_accepted_does_not(
        self, walk, tmp_path, apng_frames, same_frames, case
    ):
        """Named, so the rule lets every declared field through. One
        frame shows a field (vedo's renderer takes about 2 s a frame),
        but ghost_spacing, where the backend reads ghosts: on the walk's
        four frames, with two ghosts."""
        pytest.importorskip(PACKAGE[case.backend])
        clip, options = walk[:1], {}
        if case.field == "ghost_spacing" and "ghost" in _backends._reads("render", case.backend):
            clip, options = walk, {"ghost": 2}

        def frames(style, name):
            path = tmp_path / f"{name}.apng"
            bvhplot.render(
                clip,
                path,
                backend=case.backend,
                resolution=(96, 72),
                fps=FPS,
                style=style,
                **options,
            )
            return apng_frames(path, FPS)

        assert same_frames(frames(case.one, "one"), frames(case.other, "other")) != case.differ


class TestANamedBackend:
    def test_refuses_an_option_it_does_not_read_even_at_its_default(self, walk, tmp_path):
        """Given means not None: frame_counter=False is given."""
        path = tmp_path / "x.apng"
        with pytest.raises(ValueError) as raised:
            bvhplot.render(walk, path, backend="vedo", frame_counter=False)
        assert str(raised.value) == (
            "render(backend='vedo') does not read frame_counter. Leave it "
            "unset, or use backend='matplotlib' or 'opencv', which read it."
        )
        assert not path.exists()

    @pytest.mark.parametrize(
        "backend, option, shown",
        [
            ("vedo", {"follow": True}, "follow"),
            ("vedo", {"camera": "turntable"}, "camera='turntable'"),
            ("vedo", {"ghost": 2}, "ghost"),
            ("vedo", {"trajectory": True}, "trajectory"),
            ("vedo", {"frame_counter": True}, "frame_counter"),
        ],
        ids=lambda value: "-".join(map(str, value)) if isinstance(value, dict) else value,
    )
    def test_refuses_each_option_it_does_not_read(self, walk, tmp_path, backend, option, shown):
        path = tmp_path / "x.apng"
        with pytest.raises(
            ValueError, match=rf"^render\(backend='{backend}'\) does not read {shown}\."
        ):
            bvhplot.render(walk, path, backend=backend, **option)
        assert not path.exists()

    def test_names_every_option_it_does_not_read_and_the_backends_reading_them(
        self, walk, tmp_path
    ):
        with pytest.raises(ValueError) as raised:
            bvhplot.render(
                walk, tmp_path / "x.apng", backend="vedo", camera="turntable", ghost=1, follow=False
            )
        assert str(raised.value) == (
            "render(backend='vedo') does not read follow, ghost or camera='turntable'. "
            "Leave them unset, or use backend='matplotlib' or 'opencv', which read them."
        )


class TestANamedBackendAndTheStyle:
    """A Style field the caller set that the named backend does not draw
    raises, as an option it does not read does; a preset's own values are
    never refused."""

    def test_refuses_a_value_set_that_it_does_not_draw(self, walk, tmp_path):
        pytest.importorskip("cv2")
        path = tmp_path / "x.apng"
        with pytest.raises(ValueError) as raised:
            bvhplot.render(
                walk, path, backend="opencv", style=bvhplot.Style("paper", projection="persp")
            )
        assert str(raised.value) == (
            "render(backend='opencv') does not draw the Style's projection='persp'. "
            "Leave it out of the Style, or use backend='matplotlib' or 'vedo', which draw it."
        )
        assert not path.exists()

    @pytest.mark.parametrize("backend", sorted(PACKAGE))
    @pytest.mark.parametrize("preset", ["paper", "debug", "dark"])
    def test_accepts_every_preset(self, walk, tmp_path, backend, preset):
        """A preset fills every field, drawn by the backend or not."""
        pytest.importorskip(PACKAGE[backend])
        for style in (preset, bvhplot.Style(preset)):
            path = tmp_path / f"{backend}-{preset}.apng"
            bvhplot.render(walk, path, backend=backend, resolution=(96, 72), style=style)
            assert path.exists()

    def test_opencv_accepts_the_projection_it_draws(self, walk, tmp_path):
        pytest.importorskip("cv2")
        path = tmp_path / "x.apng"
        style = bvhplot.Style("paper", projection="ortho")
        bvhplot.render(walk, path, backend="opencv", resolution=(96, 72), style=style)
        assert path.exists()

    def test_a_field_set_before_a_replace_is_still_refused(self, walk, tmp_path):
        """replace adds the fields it changes to those set before."""
        pytest.importorskip("cv2")
        style = bvhplot.Style("paper", projection="persp").replace(floor=None)
        with pytest.raises(ValueError, match=r"does not draw the Style's projection='persp'\."):
            bvhplot.render(walk, tmp_path / "x.apng", backend="opencv", style=style)

    def test_a_field_set_by_a_replace_is_refused(self, walk, tmp_path):
        pytest.importorskip("cv2")
        style = bvhplot.Style("paper", floor=None).replace(shadow=True)
        with pytest.raises(ValueError, match=r"does not draw the Style's shadow\."):
            bvhplot.render(walk, tmp_path / "x.apng", backend="opencv", style=style)


class TestAutoAndTheStyle:
    """Under auto, a Style field the caller set that the backend chosen
    does not draw warns, naming the backend chosen, and the file is
    written."""

    def test_opencv_warns_of_a_projection_it_does_not_draw(self, walk, tmp_path):
        pytest.importorskip("cv2")
        path = tmp_path / "x.avi"
        with pytest.warns(UserWarning) as warned:
            bvhplot.render(walk, path, resolution=(96, 72), style=bvhplot.Style(projection="persp"))
        assert (
            "render() chose backend='opencv', which does not draw the Style's "
            "projection='persp', so it is ignored. To draw it, use backend='matplotlib' "
            "or 'vedo', which draw it."
        ) in [str(warning.message) for warning in warned]
        assert path.exists()

    def test_matplotlib_warns_of_a_shadow_it_does_not_draw(self, walk, tmp_path):
        path = tmp_path / "x.apng"
        with pytest.warns(UserWarning) as warned:
            bvhplot.render(walk, path, resolution=(96, 72), style=bvhplot.Style(shadow=False))
        assert [str(warning.message) for warning in warned] == [
            "render() chose backend='matplotlib', which does not draw the Style's shadow, "
            "so it is ignored. To draw it, use backend='vedo', which draws it."
        ]
        assert path.exists()
