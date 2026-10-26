"""The options of render() each backend reads, and what render() does
with an option the backend it runs does not read: raise when the backend
is named, warn under auto."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pybvh import bvhplot, read_bvh_file

BVH_DIR = Path(__file__).parent.parent / "bvh_data"

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


@pytest.fixture(scope="module")
def walk():
    """Four frames of a walk spread over 3 seconds, as it turns, so a
    camera that follows, a turntable, ghosts and the root's trail each
    change the picture."""
    return read_bvh_file(BVH_DIR / "cmu_12_01_walk.bvh")[0:400:100]


@pytest.fixture(scope="module")
def pair(walk):
    """The walk beside itself drawn twice as large, so that a spacing
    and a matched size each change a scene holding both."""
    return [walk, walk.scale(2.0)]


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


def _same(frames, others):
    return len(frames) == len(others) and all(
        np.array_equal(a, b) for a, b in zip(frames, others, strict=True)
    )


class TestEachBackendReadsWhatItDeclares:
    """Rendered with and without an option, a backend that reads it
    writes other frames, and one that does not writes the same."""

    @pytest.mark.parametrize(
        "backend, option",
        [
            *[
                ("opencv", option)
                for option in ("follow", "turntable", "ghost", "trajectory", "frame_counter")
            ],
            *[("matplotlib", option) for option in ("follow", "turntable", "ghost", "trajectory")],
            ("vedo", "spacing"),
            ("vedo", "match_size"),
        ],
    )
    def test_an_option_read_changes_the_frames(self, drawn, backend, option):
        pytest.importorskip(PACKAGE[backend])
        without = drawn(backend, option)
        given = drawn(backend, option, **OPTIONS[option])
        assert len(given) == len(without)
        assert not _same(given, without)

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
        self, drawn, backend, option
    ):
        """Named, it is accepted: each skeleton in its own panel is the
        arrangement it asks for."""
        pytest.importorskip(PACKAGE[backend])
        assert _same(drawn(backend, option, **OPTIONS[option]), drawn(backend, option))

    def test_matplotlib_draws_no_frame_counter(self, drawn):
        """Reached under auto, which gives an animated image to
        matplotlib and warns rather than refuse."""
        with pytest.warns(UserWarning, match="frame_counter"):
            given = drawn("auto", "frame_counter", frame_counter=True)
        assert _same(given, drawn("matplotlib", "frame_counter"))


class TestANamedBackend:
    def test_refuses_an_option_it_does_not_read_even_at_its_default(self, walk, tmp_path):
        """Given means not None: frame_counter=False is given."""
        path = tmp_path / "x.apng"
        with pytest.raises(ValueError) as raised:
            bvhplot.render(walk, path, backend="vedo", frame_counter=False)
        assert str(raised.value) == (
            "render(backend='vedo') does not read frame_counter. Leave it "
            "unset, or use backend='opencv', which reads it."
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
            ("matplotlib", {"frame_counter": True}, "frame_counter"),
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


class TestUnderAuto:
    def test_an_option_the_backend_chosen_does_not_read_warns_and_the_file_is_written(
        self, walk, tmp_path
    ):
        """Auto sends an animated image to matplotlib, which draws no
        frame counter."""
        with pytest.warns(UserWarning) as record:
            path = bvhplot.render(
                walk, tmp_path / "x.apng", resolution=(64, 48), frame_counter=True
            )
        [message] = [str(w.message) for w in record if "frame_counter" in str(w.message)]
        assert message == (
            "render() chose backend='matplotlib', which does not read frame_counter, "
            "so it is ignored. To draw it, use backend='opencv', which reads it."
        )
        assert path.exists()

    def test_the_warning_names_the_users_line(self, walk, tmp_path):
        with pytest.warns(UserWarning, match="frame_counter") as record:
            bvhplot.render(walk, tmp_path / "x.apng", resolution=(64, 48), frame_counter=False)
        [warning] = [w for w in record if "frame_counter" in str(w.message)]
        assert warning.filename == __file__
