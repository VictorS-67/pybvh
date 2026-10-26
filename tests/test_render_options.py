"""The options of render() each backend reads, and what render() does
with an option the backend it runs does not read: raise when the backend
is named, warn under auto."""

from __future__ import annotations

from pathlib import Path

import pytest

from pybvh import bvhplot, read_bvh_file

BVH_DIR = Path(__file__).parent.parent / "bvh_data"


@pytest.fixture(scope="module")
def walk():
    """A few frames of a walk that turns, so a camera that follows, a
    turntable, ghosts and the root's trail each change the picture."""
    return read_bvh_file(BVH_DIR / "cmu_12_01_walk.bvh")[0:40:8]


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
