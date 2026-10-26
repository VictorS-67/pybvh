"""Writing an animation's frames to a file.

A backend that produces frames draws them at the :class:`VideoTarget`'s
resolution and hands them to the :class:`Writer` the router chose with
:func:`writer_for`. This module owns the table of the extensions written
from frames and their writers, and decides from the machine's
:class:`~._backends.Environment` how a video container is encoded: H.264
through the system ``ffmpeg`` executable, or OpenCV's MPEG-4 Part 2.
Frames are ``(H, W, 3)`` uint8 RGB arrays, every one the size of the
first.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import chain
from pathlib import Path

import numpy as np
import numpy.typing as npt

from ._backends import Environment, install_hint

# Valid values of render(codec=).
CODECS = frozenset({"auto", "h264", "mpeg4"})

# The format of each extension written from frames: a video container,
# encoded as the codec and the machine allow, or the name of the Pillow
# format an animated image is saved in.
_VIDEO = "video"
_FORMATS = {
    ".mp4": _VIDEO,
    ".mov": _VIDEO,
    ".avi": _VIDEO,
    ".gif": "GIF",
    ".webp": "WEBP",
    ".apng": "PNG",  # an animated PNG: Pillow writes one from several frames
}
VIDEO_CONTAINERS = frozenset(ext for ext, fmt in _FORMATS.items() if fmt == _VIDEO)


@dataclass(frozen=True)
class VideoTarget:
    """Where an animation is written: the path, the rate written, the
    resolution and the codec.

    Attributes
    ----------
    path : Path
        The file written, whose extension says its format; never changed.
    fps : float
        Frames per second asked, fractional rates included; the router
        has checked it is a positive number (``_resolve_fps``). The file
        is written at :attr:`written_fps`.
    resolution : (int, int)
        ``(width, height)`` in pixels the backend draws at, each a
        positive number.
    codec : str
        ``"auto"``, ``"h264"`` or ``"mpeg4"``, as
        :func:`pybvh.bvhplot.render` documents; anything but ``"auto"``
        only for a video container (``.mp4``, ``.mov``, ``.avi``).

    Raises
    ------
    ValueError
        On a resolution that is not two positive numbers, an unknown
        codec, or a codec named for a file that is not a video container.
    """

    path: Path
    fps: float
    resolution: tuple[int, int]
    codec: str = "auto"

    def __post_init__(self) -> None:
        sizes = tuple(self.resolution) if isinstance(self.resolution, (list, tuple)) else ()
        if not (len(sizes) == 2 and all(_is_positive_number(size) for size in sizes)):
            raise ValueError(
                f"resolution must be two positive numbers (width, height) in "
                f"pixels, got {self.resolution!r}."
            )
        object.__setattr__(self, "resolution", sizes)  # a list given is kept as a tuple
        if self.codec not in CODECS:
            raise ValueError(f"Unknown codec {self.codec!r}. Choose from: {sorted(CODECS)}")
        if self.codec != "auto" and self.suffix not in VIDEO_CONTAINERS:
            raise ValueError(
                f"codec= applies to video containers "
                f"({sorted(VIDEO_CONTAINERS)}), not {self.path.suffix!r}."
            )

    @property
    def suffix(self) -> str:
        """The path's extension, in lower case."""
        return self.path.suffix.lower()

    @property
    def written_fps(self) -> float:
        """The rate the file is written at: the rate its frames play at.

        A video container is written at :attr:`fps` as given, since
        ffmpeg and OpenCV take fractional rates. An animated image (GIF,
        WebP, APNG) stores each frame's duration in whole milliseconds,
        ``1000 / fps`` rounded to the nearest: 42 ms at 24 fps, so 23.81
        fps; a GIF then stores it truncated to 10 ms, a limit of the
        format not counted here.
        """
        if _FORMATS.get(self.suffix) == _VIDEO:
            return self.fps
        return 1000.0 / round(1000.0 / self.fps)


def _is_positive_number(value: object) -> bool:
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and bool(value > 0)
    )


_FFMPEG = "ffmpeg"  # H.264 through the system ffmpeg
_OPENCV = "opencv"  # MPEG-4 Part 2 through OpenCV's VideoWriter
_PILLOW = "pillow"  # an image sequence through Pillow


@dataclass(frozen=True)
class Writer:
    """How this machine writes a target, and the warnings the choice owes
    the caller (each a message, warned by the router at the user's
    line). Built by :func:`writer_for`."""

    target: VideoTarget
    encoder: str
    warnings: tuple[str, ...] = ()

    def write(self, frames: Iterable[npt.NDArray[np.uint8]]) -> Path:
        """Write *frames* to the target's path and return the path.

        The file is sized from the first frame, not from the target's
        resolution: vedo may deliver screenshots larger than asked
        (HiDPI scaling).
        """
        path, fps = self.target.path, self.target.written_fps
        frames = iter(frames)
        first = next(frames)
        frames = chain([first], frames)
        if self.encoder == _PILLOW:
            _save_with_pillow(path, frames, fps, _FORMATS[self.target.suffix])
            return path
        h, w = first.shape[:2]
        sink_type = _FfmpegSink if self.encoder == _FFMPEG else _OpenCvSink
        sink = sink_type(path, fps, (w, h))
        try:
            for frame in frames:
                sink.write(frame)
        except BaseException:
            sink.abort()
            raise
        sink.release()
        return path


_NOT_IN_BROWSERS = (
    "ffmpeg is not on PATH, so the video is written as MPEG-4 Part 2, which "
    "plays in desktop players such as VLC but not in browsers, VSCode or "
    f"notebooks. {install_hint('ffmpeg')} to write H.264."
)


def writer_for(target: VideoTarget, environment: Environment, subject: str) -> Writer:
    """The writer of *target* on the machine *environment* describes, for
    the backend *subject* names in an error (``"The OpenCV backend"``).

    A video container is H.264 through ffmpeg when ffmpeg is on PATH and
    the codec is ``"auto"`` or ``"h264"``, else MPEG-4 Part 2 through
    OpenCV; under ``"auto"`` that fallback warns that the file will not
    play in browsers. The path is never changed.

    Raises
    ------
    ValueError
        When *target*'s extension is not written from frames.
    RuntimeError
        On ``codec="h264"`` without ffmpeg.
    ImportError
        When MPEG-4 Part 2 is needed and OpenCV does not import.
    """
    ext = target.suffix
    fmt = _FORMATS.get(ext)
    if fmt is None:
        raise ValueError(
            f"{subject} cannot write {ext!r} files. "
            f"Supported extensions: {sorted(_FORMATS)}. "
            f"Use backend='matplotlib' for other formats."
        )
    if fmt != _VIDEO:
        return Writer(target, _PILLOW)
    if target.codec == "h264" and not environment.has_ffmpeg:
        raise RuntimeError(
            "codec='h264' requires the ffmpeg executable on PATH — "
            f"OpenCV cannot encode H.264 itself. {install_hint('ffmpeg')}, or use "
            "codec='mpeg4' (plays in desktop players such as VLC, but "
            "not in browsers or VSCode)."
        )
    if environment.has_ffmpeg and target.codec != "mpeg4":
        return Writer(target, _FFMPEG)
    if "cv2" not in environment.importable:
        if target.codec == "mpeg4":
            raise ImportError(
                f"codec='mpeg4' is written by OpenCV, which requires "
                f"opencv-python. {install_hint('cv2')}"
            )
        raise ImportError(
            f"{subject} writes {ext} files as H.264 through the ffmpeg "
            f"executable or as MPEG-4 Part 2 through opencv-python, and "
            f"neither is available. {install_hint('ffmpeg')}, or OpenCV "
            f"({install_hint('cv2')}), or render to .gif instead."
        )
    warned = () if target.codec == "mpeg4" else (_NOT_IN_BROWSERS,)
    return Writer(target, _OPENCV, warned)


class _FfmpegSink:
    """H.264 video sink: frames piped to the system ``ffmpeg``.

    Encodes libx264 + yuv420p + faststart, the combination that plays in
    browsers, VSCode, and notebook embeds. Odd frame dimensions are
    padded by one pixel (yuv420p requires even sizes). Frames are piped
    as BGR, not RGB: ffmpeg rounds the two conversions to yuv420p a few
    levels apart, and BGR is the conversion every H.264 file pybvh has
    written went through, so a frame drawn the same encodes the same.
    """

    def __init__(self, filepath: Path, fps: float, resolution: tuple[int, int]) -> None:
        import subprocess

        w, h = resolution
        self._proc = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{w}x{h}",
                "-r",
                f"{fps}",
                "-i",
                "-",
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(filepath),
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def write(self, frame: npt.NDArray[np.uint8]) -> None:
        assert self._proc.stdin is not None
        try:
            self._proc.stdin.write(frame[:, :, ::-1].tobytes())
        except BrokenPipeError:
            self.release()  # ffmpeg has exited: raise its own error
            raise

    def release(self) -> None:
        """Close the pipes, wait for ffmpeg to finish the file, and raise
        its error if it failed."""
        _, err = self._proc.communicate()
        if self._proc.returncode != 0:
            message = err.decode(errors="replace").strip()[:500]
            raise RuntimeError(
                f"ffmpeg exited with code {self._proc.returncode} while encoding: {message}"
            )

    def abort(self) -> None:
        """Stop ffmpeg without finishing the file, and close the pipes."""
        self._proc.kill()
        self._proc.communicate()


class _OpenCvSink:
    """MPEG-4 Part 2 video sink: OpenCV's ``VideoWriter``, ``mp4v`` and
    else ``XVID``. OpenCV builds ship without an H.264 encoder (patent
    licensing), which is why H.264 needs the external binary."""

    def __init__(self, filepath: Path, fps: float, resolution: tuple[int, int]) -> None:
        import cv2

        codecs = ["mp4v", "XVID"]
        for fourcc_name in codecs:
            fourcc = cv2.VideoWriter_fourcc(*fourcc_name)  # type: ignore[attr-defined]
            self._writer = cv2.VideoWriter(str(filepath), fourcc, fps, resolution)
            if self._writer.isOpened():
                return
        raise RuntimeError(
            f"Could not open video writer for {filepath}. "
            f"Tried codecs: {codecs}. Ensure OpenCV has video codec support."
        )

    def write(self, frame: npt.NDArray[np.uint8]) -> None:
        self._writer.write(np.ascontiguousarray(frame[:, :, ::-1]))

    def release(self) -> None:
        self._writer.release()

    def abort(self) -> None:
        """Close the file as written so far."""
        self._writer.release()


def _save_with_pillow(
    path: Path, frames: Iterable[npt.NDArray[np.uint8]], fps: float, pillow_format: str
) -> None:
    """Save *frames* as one animated image in Pillow's *pillow_format*, looping,
    each frame shown for ``1000 / fps`` milliseconds, *fps* being a
    :attr:`VideoTarget.written_fps` (a GIF stores them further truncated
    to 10 ms).

    Pillow stores a run of identical frames as one frame shown for the
    run's summed duration, a whole number of frame durations; a file
    whose frames are all identical is a still image. WebP is Pillow's
    default lossy encoding; PNG is lossless.
    """
    from PIL import Image

    images = (Image.fromarray(frame) for frame in frames)
    first = next(images)
    # Pillow documents append_images as a list, and its PNG writer reads
    # the frames twice; its GIF writer reads them once, keeping each frame
    # only in its palette, so a GIF's RGB frames are not all held at once.
    rest = list(images) if pillow_format == "PNG" else images
    first.save(
        path,
        format=pillow_format,
        save_all=True,
        append_images=rest,
        duration=round(1000.0 / fps),
        loop=0,
    )
