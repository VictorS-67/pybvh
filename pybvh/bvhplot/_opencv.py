"""OpenCV fast render backend.

Renders skeleton animations to video files using orthographic 2D
projection and OpenCV drawing primitives. Orders of magnitude faster
than matplotlib for video export.

Requires ``opencv-python >= 4.5``.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pathlib import Path
from typing import TYPE_CHECKING

from ._style import (
    GHOST_WIDTH_FACTOR,
    Style,
    TRACE_BLEND,
    TRACE_COLOR,
    ghost_schedule,
    PALETTE_RGB,
)
from ._viewport import Viewport, floor_trace_points, panel_viewports
from ._scene import Scene, SkeletonView
from ._colors import bone_colors_255, floor_palette, node_colors_255

if TYPE_CHECKING:
    from collections.abc import Iterator

# RGB is canonical in _style; the channel flip for OpenCV's BGR
# drawing API happens here, at this backend's border.
PALETTE_BGR = [(b, g, r) for (r, g, b) in PALETTE_RGB]


def _to_bgr(color: object) -> tuple[int, int, int]:
    """Any matplotlib-parseable color -> OpenCV BGR uint8 tuple."""
    from matplotlib.colors import to_rgb
    r, g, b = to_rgb(color)  # type: ignore[arg-type]
    return (int(b * 255), int(g * 255), int(r * 255))


def _blend_bgr(
    fg: tuple[int, int, int],
    bg: tuple[int, int, int],
    alpha: float,
) -> tuple[int, int, int]:
    """Emulate alpha compositing (cv2 has none) by pre-blending."""
    return tuple(int(f * alpha + b * (1.0 - alpha))
                 for f, b in zip(fg, bg))  # type: ignore[return-value]


def _draw_floor_opencv(
    img: npt.NDArray[np.uint8],
    view: SkeletonView,
    style: Style,
    viewport: Viewport,
    frame_idx: int,
    panel_w: int,
    h: int,
    x_offset: int,
    bg_bgr: tuple[int, int, int],
    px_scale: float = 1.0,
) -> None:
    """Project and draw the ground plane into one panel.

    Mirrors the matplotlib floor geometry (same extents, same grays,
    ``floor_alpha`` emulated by pre-blending toward the background).
    """
    import cv2

    up = view.up_index
    ground = [i for i in range(3) if i != up]
    ext = view.half_span * 1.8
    y = view.floor_height
    palette = floor_palette(style)

    def project(world_pts: npt.NDArray[np.float64]) -> npt.NDArray[np.int32]:
        pts = viewport.project(world_pts, (panel_w, h), frame_idx)
        pts[:, 0] += x_offset
        return pts

    def quad(c0: float, c1: float, s: float) -> npt.NDArray[np.float64]:
        pts = np.zeros((4, 3))
        for k, (d0, d1) in enumerate(((-s, -s), (s, -s), (s, s), (-s, s))):
            pts[k, ground[0]] = c0 + d0
            pts[k, ground[1]] = c1 + d1
            pts[k, up] = y
        return pts

    c0 = float(view.center[ground[0]])
    c1 = float(view.center[ground[1]])

    if style.floor == "solid":
        face = _blend_bgr(_to_bgr(palette["face"]), bg_bgr,
                          style.floor_alpha)
        cv2.fillPoly(img, [project(quad(c0, c1, ext))], face,
                     lineType=cv2.LINE_AA)
    elif style.floor == "checker":
        n = 8
        s = ext / n
        shades = [
            _blend_bgr(_to_bgr(shade), bg_bgr, style.floor_alpha)
            for shade in palette["checker"]]
        for i in range(-n, n):
            for j in range(-n, n):
                sq = quad(c0 + (i + 0.5) * s, c1 + (j + 0.5) * s, s / 2)
                cv2.fillPoly(img, [project(sq)], shades[(i + j) % 2],
                             lineType=cv2.LINE_AA)
    elif style.floor == "grid":
        color = _blend_bgr(_to_bgr(palette["grid"]), bg_bgr, 0.8)
        n = 10
        s = ext / n
        for i in range(-n, n + 1):
            for horiz in (True, False):
                seg = np.zeros((2, 3))
                if horiz:
                    seg[0, ground[0]], seg[1, ground[0]] = c0 - ext, c0 + ext
                    seg[:, ground[1]] = c1 + i * s
                else:
                    seg[:, ground[0]] = c0 + i * s
                    seg[0, ground[1]], seg[1, ground[1]] = c1 - ext, c1 + ext
                seg[:, up] = y
                p = project(seg)
                cv2.line(img, tuple(p[0]), tuple(p[1]), color,
                         max(1, int(px_scale + 0.5)), cv2.LINE_AA)


# Extensions this backend can actually write: video containers via
# cv2.VideoWriter, plus GIF via a dedicated Pillow path.
_OPENCV_EXTENSIONS = {'.mp4', '.mov', '.avi', '.gif'}

# Valid values for render(codec=); shared by the OpenCV and vedo sinks.
VIDEO_CODECS = {"auto", "h264", "mpeg4"}


class _ViewDrawContext:
    """Frame-invariant drawing data for one view, computed once per
    render instead of once per frame (chain classification triggers
    rest-pose FK + foot detection — far too heavy for a frame loop)."""

    def __init__(
        self,
        view: SkeletonView,
        style: Style,
        view_index: int,
        n_skeletons: int,
        bg_bgr: tuple[int, int, int],
        ghost: int,
        trajectory: bool,
    ) -> None:
        bone_rgb = bone_colors_255(view, style, view_index, n_skeletons)
        self.bone_bgr = [(b, g, r) for (r, g, b) in bone_rgb]
        # node -> BGR dot color for the markers-off (legacy) look
        node_rgb = node_colors_255(view, style, view_index, n_skeletons,
                                   bone_rgb)
        self.node_bgr = node_rgb[:, ::-1]
        self.joint_bgr = _to_bgr(style.joint_color)
        self.label_bgr = (self.bone_bgr[0] if self.bone_bgr
                          else (0, 0, 0))
        if ghost > 0:
            self.ghost_lag, weights = ghost_schedule(
                style, view.frame_time, ghost)
            self.ghost_bgr = [
                [_blend_bgr(c, bg_bgr, float(w)) for c in self.bone_bgr]
                for w in weights]
        if trajectory:
            # Full floored path once; per frame we slice a view of it.
            self.trace_path = floor_trace_points(view)
            self.trace_bgr = _blend_bgr(
                _to_bgr(TRACE_COLOR), bg_bgr, TRACE_BLEND)


def _draw_skeletons_on_frame(
    img: npt.NDArray[np.uint8],
    frame_idx: int,
    scene: Scene,
    style: Style,
    viewports: list[Viewport],
    contexts: list[_ViewDrawContext],
    panel_w: int,
    h: int,
    bg_bgr: tuple[int, int, int],
    px_scale: float = 1.0,
    ghost: int = 0,
    trajectory: bool = False,
) -> None:
    """Draw all skeletons for one frame onto *img* (mutates in place).

    Each skeleton is projected through its own viewport, so skeletons
    with different forward/up axes all render correctly side by side.
    In multi-panel mode every view draws into its own panel-sized
    buffer that is then blitted into place — cv2 primitives have no
    clip rectangle, and an unclipped floor quad (1.8x half_span, wider
    than a panel) would otherwise paint over the neighboring panel.

    The viewport frames the drawing to the clip's motion and holds one
    scale across frames even when the camera rotates (see
    :meth:`~._viewport.Viewport.project`).

    Joint dots are always drawn in this backend — at raster resolution
    they are load-bearing for line joins (matplotlib omits them when
    ``style.joint_markers`` is off). With markers off they are small
    and bone-colored (the pre-0.9.0 look); with markers on they use
    ``style.joint_color``.

    ``ghost`` faded trailing poses draw behind the live skeleton
    (oldest first); ``trajectory`` draws the root trace on the floor up
    to the current frame (solid thin line — cv2 has no dashes).
    """
    import cv2

    n_skeletons = scene.num_skeletons
    # Primitive sizes scale with the drawing resolution (1080p is the
    # 1:1 anchor: bone_width 3.0 -> 3 px there, twice that at 4K, and
    # supersampled drawing surfaces scale up with them).
    thickness = max(1, int(style.bone_width * px_scale + 0.5))
    thin = max(1, int(px_scale + 0.5))

    for s, view in enumerate(scene.views):
        ctx = contexts[s]
        frame_data = view.coords[frame_idx]
        viewport = viewports[s]
        view_matrix = viewport.view_matrix(frame_idx)

        if n_skeletons > 1:
            # Contiguous per-panel canvas: clips every primitive to the
            # panel, then blits.
            canvas = np.empty((h, panel_w, 3), dtype=np.uint8)
            canvas[:] = bg_bgr
        else:
            canvas = img

        if style.floor is not None:
            _draw_floor_opencv(
                canvas, view, style, viewport, frame_idx, panel_w, h, 0,
                bg_bgr, px_scale=px_scale)

        def project(world_pts):
            return viewport.project(world_pts, (panel_w, h), frame_idx)

        if trajectory and frame_idx >= 1:
            path = ctx.trace_path[:frame_idx + 1]
            cv2.polylines(canvas, [project(path)], False, ctx.trace_bgr,
                          thin, cv2.LINE_AA)

        # Painter's order: cv2 has no depth buffer, so bones draw
        # far-to-near along the camera direction (view_matrix row 2
        # points toward the viewer). Recomputed per frame — under
        # follow/turntable the view matrix changes every frame.
        bones_arr = np.asarray(view.bones, dtype=int)

        def painter_order(pose):
            return np.argsort(pose[bones_arr].mean(axis=1)
                              @ view_matrix[2])

        if ghost > 0:
            ghost_thickness = max(
                1, int(style.bone_width * GHOST_WIDTH_FACTOR
                       * px_scale + 0.5))
            for j in reversed(range(ghost)):     # oldest first
                gf = frame_idx - (j + 1) * ctx.ghost_lag
                if gf < 0:
                    continue
                gpts = project(view.coords[gf])
                for b in painter_order(view.coords[gf]):
                    p_idx, c_idx = view.bones[b]
                    cv2.line(canvas, tuple(gpts[p_idx]),
                             tuple(gpts[c_idx]), ctx.ghost_bgr[j][b],
                             ghost_thickness, cv2.LINE_AA)

        pts_2d = project(frame_data)

        for b in painter_order(frame_data):
            p_idx, c_idx = view.bones[b]
            pt1 = (int(pts_2d[p_idx, 0]), int(pts_2d[p_idx, 1]))
            pt2 = (int(pts_2d[c_idx, 0]), int(pts_2d[c_idx, 1]))
            cv2.line(canvas, pt1, pt2, ctx.bone_bgr[b], thickness,
                     cv2.LINE_AA)

        if style.joint_markers:
            for pt in pts_2d:
                cv2.circle(canvas, (int(pt[0]), int(pt[1])),
                           thickness + 2, ctx.joint_bgr, -1, cv2.LINE_AA)
        else:
            for j, pt in enumerate(pts_2d):
                cv2.circle(canvas, (int(pt[0]), int(pt[1])),
                           thickness + 1,
                           tuple(int(c) for c in ctx.node_bgr[j]), -1,
                           cv2.LINE_AA)

        if view.label is not None:
            cv2.putText(
                canvas, view.label,
                (int(15 * px_scale), int(35 * px_scale)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8 * px_scale, ctx.label_bgr,
                max(1, int(2 * px_scale + 0.5)), cv2.LINE_AA)

        if n_skeletons > 1:
            x0 = s * panel_w
            img[:, x0:x0 + panel_w] = canvas

    if n_skeletons > 1:
        for s in range(1, n_skeletons):
            x = s * panel_w
            cv2.line(img, (x, 0), (x, h), (200, 200, 200), thin)


def _generate_frames(
    scene: Scene,
    style: Style,
    resolution: tuple[int, int],
    *,
    motion: str = "fixed",
    frame_counter: bool = False,
    ghost: int = 0,
    trajectory: bool = False,
) -> Iterator[npt.NDArray[np.uint8]]:
    """Yield one rendered BGR image ``(H, W, 3)`` per animation frame.

    Shared by the video and GIF paths of :func:`render_opencv` — only
    the sink (``cv2.VideoWriter`` vs Pillow) differs between them.

    Parameters
    ----------
    scene : Scene
        Prepared visualization (per-view cameras; each panel is framed
        by its own viewport).
    style : Style
        Visual styling; ``style.axes == "full"`` draws the per-panel
        axis indicator (the pre-0.9.0 ``show_axis``).
    resolution : (int, int)
        ``(width, height)`` in pixels.
    motion : str, optional
        How each panel's camera moves, handed to the viewport
        untouched (:func:`~._viewport.make_viewport`).
    frame_counter : bool, optional
        Draw a ``Frame f/F`` counter in the bottom-right corner.
        Default ``False`` (opt-in — publication output never stamps
        text).
    """
    import cv2

    w, h = resolution
    bg_bgr = _to_bgr(style.background)

    # Supersampling: draw at style.supersample x the target resolution
    # and downsample with INTER_AREA — cheap, dramatically better
    # anti-aliasing. Primitive sizes scale with the drawing surface
    # (1080p = the 1:1 anchor), so a 4K export looks better than a
    # 720p one instead of thinner.
    ss = style.supersample
    draw_w, draw_h = w * ss, h * ss
    px_scale = draw_h / 1080.0
    num_frames = scene.num_frames
    n_skeletons = scene.num_skeletons
    panel_w = draw_w // n_skeletons if n_skeletons > 1 else draw_w

    # One viewport per panel: the box the clip sweeps, the camera's
    # schedule, and one projection scale for the whole clip.
    # Orthographic whatever the style asks: this backend has no
    # perspective projection, and says so to the viewport.
    viewports = panel_viewports(
        scene.views, framing="clip", motion=motion, projection="ortho")

    contexts = [
        _ViewDrawContext(v, style, s, n_skeletons, bg_bgr, ghost,
                         trajectory)
        for s, v in enumerate(scene.views)]

    for f in range(num_frames):
        img = np.empty((draw_h, draw_w, 3), dtype=np.uint8)
        img[:] = bg_bgr

        _draw_skeletons_on_frame(
            img, f, scene, style, viewports, contexts, panel_w,
            draw_h, bg_bgr,
            px_scale=px_scale,
            ghost=ghost, trajectory=trajectory)

        if ss > 1:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

        # Text and the axis indicator stamp AFTER the downsample so
        # they stay crisp at the output resolution.
        if frame_counter:
            fc_text = f"Frame {f}/{num_frames - 1}"
            fc_x = max(5, w - 200)
            cv2.putText(
                img, fc_text,
                (fc_x, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1,
                cv2.LINE_AA)

        if style.axes == "full":
            for s, viewport in enumerate(viewports):
                _draw_axis_indicator(
                    img, viewport.view_matrix(f), viewport.up_axis,
                    panel_w // ss, h, panel_idx=s)

        yield img


def render_opencv(
    scene: Scene,
    style: Style,
    filepath: Path,
    fps: float,
    resolution: tuple[int, int],
    *,
    motion: str = "fixed",
    frame_counter: bool = False,
    ghost: int = 0,
    trajectory: bool = False,
    codec: str = "auto",
) -> Path:
    """Render skeleton animation to a video or GIF file using OpenCV.

    Each panel uses its own bounding box and camera so that mixed-up-axis
    side-by-side comparisons render correctly. *motion* decides how
    each panel's camera moves.

    Frame generation is shared with the GIF path via
    :func:`_generate_frames`; this function only picks the sink.

    Parameters
    ----------
    scene : Scene
        Prepared visualization (per-view cameras; each panel is framed
        by its own viewport).
    style : Style
        Visual styling (colors, floor, background; ``axes="full"``
        draws the per-panel axis indicator).
    filepath : Path
        Output file path. Must end in one of ``.mp4``, ``.mov``,
        ``.avi``, or ``.gif``.
    fps : float
        Frames per second.
    resolution : (int, int)
        ``(width, height)`` in pixels.
    motion : str, optional
        ``"fixed"`` (default), ``"turntable"`` or ``"follow"``, handed
        to the viewport untouched.

    Returns
    -------
    Path
        The path to the written video file.

    Raises
    ------
    ValueError
        If the file extension is not one this backend can write
        (``.html``/``.webp``/``.apng`` need the matplotlib backend).
    """
    ext = filepath.suffix.lower()
    if ext not in _OPENCV_EXTENSIONS:
        raise ValueError(
            f"The OpenCV backend cannot write {ext!r} files. "
            f"Supported extensions: {sorted(_OPENCV_EXTENSIONS)}. "
            f"Use backend='matplotlib' for other formats.")

    frames = _generate_frames(
        scene, style, resolution, motion=motion,
        frame_counter=frame_counter, ghost=ghost, trajectory=trajectory)

    # Pillow sink for GIF output (cv2.VideoWriter doesn't support GIF).
    if ext == '.gif':
        return _render_gif(frames, filepath, fps)

    writer = _open_writer(filepath, fps, resolution, codec)
    for img in frames:
        writer.write(img)  # type: ignore[attr-defined]
    writer.release()  # type: ignore[attr-defined]
    return filepath


class _FfmpegPipeWriter:
    """H.264 video sink: BGR frames piped to the system ``ffmpeg``.

    Duck-types ``cv2.VideoWriter`` (``write``/``release``) so the two
    sinks are interchangeable downstream. Encodes libx264 + yuv420p +
    faststart — the combination that plays in browsers, VSCode, and
    notebook embeds. Odd frame dimensions are padded by one pixel
    (yuv420p requires even sizes).
    """

    def __init__(
        self,
        filepath: Path,
        fps: float,
        resolution: tuple[int, int],
    ) -> None:
        import subprocess

        w, h = resolution
        self._frame_bytes = w * h * 3
        self._proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
             "-r", f"{fps}", "-i", "-",
             "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
             "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(filepath)],
            stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame: npt.NDArray[np.uint8]) -> None:
        assert self._proc.stdin is not None
        self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def release(self) -> None:
        assert self._proc.stdin is not None and self._proc.stderr is not None
        self._proc.stdin.close()
        err = self._proc.stderr.read().decode(errors="replace")
        code = self._proc.wait()
        if code != 0:
            raise RuntimeError(
                f"ffmpeg exited with code {code} while encoding: "
                f"{err.strip()[:500]}")


def _open_writer(
    filepath: Path,
    fps: float,
    resolution: tuple[int, int],
    codec: str = "auto",
) -> object:
    """Open the video sink for *codec*.

    ``"auto"``: H.264 via the system ``ffmpeg`` when one is on PATH,
    else OpenCV's MPEG-4 Part 2 (``mp4v``). ``"h264"``: require ffmpeg,
    raise with an install hint otherwise. ``"mpeg4"``: always the
    OpenCV writer. OpenCV builds ship without an H.264 encoder (patent
    licensing), which is why H.264 needs the external binary.
    """
    import shutil

    if codec not in VIDEO_CODECS:
        raise ValueError(
            f"Unknown codec {codec!r}. Choose from: {sorted(VIDEO_CODECS)}")
    have_ffmpeg = shutil.which("ffmpeg") is not None
    if codec == "h264" and not have_ffmpeg:
        raise RuntimeError(
            "codec='h264' requires the ffmpeg executable on PATH — "
            "OpenCV cannot encode H.264 itself. Install ffmpeg (e.g. "
            "apt install ffmpeg / conda install ffmpeg), or use "
            "codec='mpeg4' (plays in desktop players such as VLC, but "
            "not in browsers or VSCode).")
    if have_ffmpeg and codec in ("auto", "h264"):
        return _FfmpegPipeWriter(filepath, fps, resolution)

    import cv2

    codecs = ['mp4v', 'XVID']
    for fourcc_name in codecs:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_name)  # type: ignore[attr-defined]
        writer = cv2.VideoWriter(str(filepath), fourcc, fps, resolution)
        if writer.isOpened():
            return writer

    raise RuntimeError(
        f"Could not open video writer for {filepath}. "
        f"Tried codecs: {codecs}. Ensure OpenCV has video codec support.")


def _render_gif(
    frames: Iterator[npt.NDArray[np.uint8]],
    filepath: Path,
    fps: float,
) -> Path:
    """Pillow sink for GIF output (cv2.VideoWriter doesn't support GIF).

    Consumes the BGR frames from :func:`_generate_frames`, converting
    each to RGB for Pillow.
    """
    from PIL import Image

    duration_ms = int(1000.0 / fps)
    pil_frames = (Image.fromarray(img[:, :, ::-1]) for img in frames)
    first_frame = next(pil_frames)
    first_frame.save(
        filepath,
        save_all=True,
        append_images=pil_frames,
        duration=duration_ms,
        loop=0)

    return filepath


def _draw_axis_indicator(
    img: npt.NDArray[np.uint8],
    view_matrix: npt.NDArray[np.float64],
    up_axis: str,
    panel_w: int,
    h: int,
    panel_idx: int = 0,
) -> None:
    """Draw a small 3D axis indicator in the bottom-left corner of a panel.

    For side-by-side renders, one indicator is drawn per panel so that
    each skeleton's own camera orientation is visible.
    """
    import cv2

    x_offset = panel_idx * panel_w
    origin = np.array([x_offset + 50, h - 50])
    axis_len = 30

    axis_colors = {
        'x': (50, 50, 220),    # red
        'y': (50, 180, 50),    # green
        'z': (220, 120, 50),   # blue
    }

    for i, axis_name in enumerate('xyz'):
        direction_3d = np.zeros(3)
        direction_3d[i] = 1.0
        projected = view_matrix @ direction_3d
        end = origin + np.array([projected[0], -projected[1]]) * axis_len
        end = end.astype(int)

        cv2.line(img, tuple(origin), tuple(end),
                 axis_colors[axis_name], 2, cv2.LINE_AA)
        cv2.putText(img, axis_name, tuple(end + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    axis_colors[axis_name], 1, cv2.LINE_AA)
