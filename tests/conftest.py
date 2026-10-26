"""Shared fixtures for the pybvh test suite."""

from __future__ import annotations

import shutil
from dataclasses import dataclass

import numpy as np
import pytest

import pybvh.analysis
import pybvh.bvh
from pybvh.bvhplot import _backends, _video


@dataclass
class Computations:
    """How many times each derived value of a clip was computed.

    Attributes
    ----------
    fk : int
        Full-clip forward kinematics (``frames_to_node_positions`` called
        by ``Bvh`` with a 2-D ``root_pos``). The single-frame FK that the
        world-up inference runs is not counted.
    world_up : int
        Every call of ``_infer_world_up`` made by ``Bvh``.
    floor_height : int
        Every call of ``analysis._compute_floor_height``.
    rest_pose : int
        Every call of ``_build_rest_pose`` made by ``Bvh``.
    report : int
        Every call of ``_build_orientation_report`` made by ``Bvh``: each
        computation of the orientation report, the rest-pose axes in it.
    """

    fk: int = 0
    world_up: int = 0
    floor_height: int = 0
    rest_pose: int = 0
    report: int = 0


@pytest.fixture
def computations(monkeypatch):
    """Count the computations of a clip's derived values (ADR 0003).

    Tests read the counts to show that a value is computed once, computed
    again after the change that should clear it, and left alone by the
    changes that should not.
    """
    counts = Computations()

    real_fk = pybvh.bvh.frames_to_node_positions
    real_infer = pybvh.bvh._infer_world_up
    real_floor = pybvh.analysis._compute_floor_height
    real_rest = pybvh.bvh._build_rest_pose
    real_report = pybvh.bvh._build_orientation_report

    def counted_fk(*args, **kwargs):
        if np.ndim(kwargs.get("root_pos")) == 2:
            counts.fk += 1
        return real_fk(*args, **kwargs)

    def counted_infer(*args, **kwargs):
        counts.world_up += 1
        return real_infer(*args, **kwargs)

    def counted_floor(*args, **kwargs):
        counts.floor_height += 1
        return real_floor(*args, **kwargs)

    def counted_rest(*args, **kwargs):
        counts.rest_pose += 1
        return real_rest(*args, **kwargs)

    def counted_report(*args, **kwargs):
        counts.report += 1
        return real_report(*args, **kwargs)

    monkeypatch.setattr(pybvh.bvh, "frames_to_node_positions", counted_fk)
    monkeypatch.setattr(pybvh.bvh, "_infer_world_up", counted_infer)
    monkeypatch.setattr(pybvh.analysis, "_compute_floor_height", counted_floor)
    monkeypatch.setattr(pybvh.bvh, "_build_rest_pose", counted_rest)
    monkeypatch.setattr(pybvh.bvh, "_build_orientation_report", counted_report)
    return counts


@pytest.fixture
def environment(monkeypatch):
    """Replace the machine bvhplot reads with one the test builds.

    Call it with what the machine offers: ``environment(notebook=True,
    importable={"cv2"})`` makes every later bvhplot call run in a notebook
    where only OpenCV imports and ffmpeg is not on PATH. Returns the value
    built.
    """

    def build(*, notebook=False, display=False, importable=(), ffmpeg=False):
        built = _backends.Environment(
            in_notebook=notebook,
            has_display=display,
            importable=frozenset(importable),
            has_ffmpeg=ffmpeg,
        )
        monkeypatch.setattr(_backends, "read_environment", lambda: built)
        return built

    return build


@pytest.fixture
def ffmpeg_on_path():
    """Skip the test unless the ffmpeg executable is on PATH: an H.264
    file needs the real encoder, whatever environment the test builds."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("needs the ffmpeg executable")


@pytest.fixture
def handed_to_writer(monkeypatch):
    """The frames each backend hands to the video module, one list per
    file written, recorded as they pass through to the file."""
    written = []
    write = _video.Writer.write

    def recording(self, frames):
        kept = []
        written.append(kept)

        def passing(frames):
            for frame in frames:
                kept.append(frame.copy())
                yield frame

        return write(self, passing(frames))

    monkeypatch.setattr(_video.Writer, "write", recording)
    return written


@pytest.fixture
def apng_frames():
    """Read back every frame of an ``.apng``: ``apng_frames(path, fps)``
    gives the frames of the file at *path*, written at *fps*, as
    ``(H, W, 3)`` uint8 RGB arrays. Pillow writes a run of identical
    frames as one frame shown for the run's summed duration, so each is
    repeated for its duration over the frame's (``round(1000 / fps)``
    milliseconds)."""
    from PIL import Image, ImageSequence

    def read(path, fps):
        shown_ms = round(1000 / fps)
        frames = []
        with Image.open(path) as apng:
            for image in ImageSequence.Iterator(apng):
                duration = image.info.get("duration", shown_ms)
                repeats, remainder = divmod(duration, shown_ms)
                assert remainder == 0, f"a frame of {path} lasts {duration} ms"
                frames.extend([np.asarray(image.convert("RGB"))] * int(repeats))
        return frames

    return read
