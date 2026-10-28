"""OpenCV fast render backend.

Renders skeleton animations as frames using orthographic 2D
projection and OpenCV drawing primitives, written by the video module.
Orders of magnitude faster than matplotlib for video export.

Requires ``opencv-python >= 4.5``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ._colors import bone_colors_255, floor_palette, node_colors_255, skeleton_color_255
from ._scene import Scene, SkeletonView
from ._style import (
    COUNTER_BASELINE_PX,
    COUNTER_RGB,
    GHOST_WIDTH_FACTOR,
    PALETTE_RGB,
    TRACE_BLEND,
    TRACE_COLOR,
    DrawnStyle,
    Style,
    bone_width_px,
    counter_left_px,
    frame_counter_text,
    ghost_schedule,
    joint_radius_px,
)
from ._viewport import FLOOR_CHECKER_SQUARES, FLOOR_GRID_CELLS, Viewport, panel_viewports

if TYPE_CHECKING:
    from collections.abc import Iterator

    from matplotlib.typing import ColorType

    from ._animation_options import AnimationOptions
    from ._video import Writer

# RGB is canonical in _style; the channel flip for OpenCV's BGR
# drawing API happens here, at this backend's border.
PALETTE_BGR = [(b, g, r) for (r, g, b) in PALETTE_RGB]


def _to_bgr(color: ColorType) -> tuple[int, int, int]:
    """Any matplotlib-parseable color -> OpenCV BGR uint8 tuple."""
    from matplotlib.colors import to_rgb

    r, g, b = to_rgb(color)
    return (int(b * 255), int(g * 255), int(r * 255))


def _blend_bgr(
    fg: tuple[int, int, int],
    bg: tuple[int, int, int],
    alpha: float,
) -> tuple[int, int, int]:
    """Emulate alpha compositing (cv2 has none) by pre-blending."""
    return tuple(int(f * alpha + b * (1.0 - alpha)) for f, b in zip(fg, bg, strict=True))  # type: ignore[return-value]


def _draw_floor_opencv(
    img: npt.NDArray[np.uint8],
    style: Style,
    viewport: Viewport,
    frame_idx: int,
    panel_w: int,
    h: int,
    x_offset: int,
    bg_bgr: tuple[int, int, int],
    px_scale: float = 1.0,
) -> None:
    """Project and draw the viewport's ground plane into one panel.

    The plane is where the viewport puts it, the same one matplotlib
    draws (same grays too, ``floor_alpha`` emulated by pre-blending
    toward the background). Painter's order, floor first, so no nudge
    below the ground is needed.
    """
    import cv2

    up = viewport.up_index
    ground = list(viewport.ground_indices)
    ext = viewport.floor_reach
    y = viewport.floor_height
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

    c0 = float(viewport.center[ground[0]])
    c1 = float(viewport.center[ground[1]])

    if style.floor == "solid":
        face = _blend_bgr(_to_bgr(palette["face"]), bg_bgr, style.floor_alpha)
        cv2.fillPoly(img, [project(quad(c0, c1, ext))], face, lineType=cv2.LINE_AA)  # type: ignore[list-item]  # mypy reads a list of MatLike as Mat
    elif style.floor == "checker":
        n = FLOOR_CHECKER_SQUARES // 2  # squares on each side of the centre
        s = ext / n
        shades = [
            _blend_bgr(_to_bgr(shade), bg_bgr, style.floor_alpha) for shade in palette["checker"]
        ]
        for i in range(-n, n):
            for j in range(-n, n):
                sq = quad(c0 + (i + 0.5) * s, c1 + (j + 0.5) * s, s / 2)
                cv2.fillPoly(img, [project(sq)], shades[(i + j) % 2], lineType=cv2.LINE_AA)  # type: ignore[list-item]  # mypy reads a list of MatLike as Mat
    elif style.floor == "grid":
        color = _blend_bgr(_to_bgr(palette["grid"]), bg_bgr, style.floor_alpha)
        n = FLOOR_GRID_CELLS // 2  # cells on each side of the centre
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
                cv2.line(
                    img, tuple(p[0]), tuple(p[1]), color, max(1, int(px_scale + 0.5)), cv2.LINE_AA
                )


class _ViewDrawContext:
    """Frame-invariant drawing data for one view, computed once per
    render instead of once per frame (chain classification triggers
    rest-pose FK + foot detection — far too heavy for a frame loop)."""

    def __init__(
        self,
        view: SkeletonView,
        viewport: Viewport,
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
        node_rgb = node_colors_255(view, style, view_index, n_skeletons, bone_rgb)
        self.node_bgr = node_rgb[:, ::-1]
        self.joint_bgr = _to_bgr(style.joint_color)
        # The skeleton's one color, the spine color under chain colors,
        # as the vedo backends color a label.
        r, g, b = skeleton_color_255(style, view_index, n_skeletons)
        self.label_bgr = (b, g, r)
        if ghost > 0:
            self.ghost_lag, weights = ghost_schedule(style, view.frame_time, ghost)
            self.ghost_bgr = [
                [_blend_bgr(c, bg_bgr, float(w)) for c in self.bone_bgr] for w in weights
            ]
        if trajectory:
            # Full floored path once; per frame we slice a view of it.
            self.trace_path = viewport.ground_path(view.coords[:, 0])
            self.trace_bgr = _blend_bgr(_to_bgr(TRACE_COLOR), bg_bgr, TRACE_BLEND)


def _painter_order(
    pose: npt.NDArray[np.floating],
    bones: npt.NDArray[np.integer],
    view_matrix: npt.NDArray[np.floating],
) -> npt.NDArray[np.intp]:
    """Bone indices of *pose* ordered far-to-near from the camera.

    *pose* is ``(N, 3)`` world coordinates, *bones* ``(B, 2)`` parent
    and child node indices, *view_matrix* the ``(3, 3)`` world-to-view
    rotation; the result is ``(B,)`` indices into *bones*.

    cv2 has no depth buffer, so bones are drawn in painter's order:
    sorted by their midpoint's depth along row 2 of *view_matrix*,
    which points toward the viewer. The caller passes the frame's own
    view matrix, since under follow/turntable it changes every frame.
    """
    return np.argsort(pose[bones].mean(axis=1) @ view_matrix[2])


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
    clip rectangle, and an unclipped floor quad (wider than a panel)
    would otherwise paint over the neighboring panel.

    The viewport frames the drawing to the clip's motion and holds one
    scale across frames even when the camera rotates (see
    :meth:`~._viewport.Viewport.project`).

    Joint dots are always drawn in this backend — at raster resolution
    they are load-bearing for line joins (matplotlib omits them when
    ``style.joint_markers`` is off). With markers off they are
    bone-colored, one pixel wider in radius than the bones are wide
    (the pre-0.9.0 look): they are how the bones are joined, and do not
    read ``style.joint_size``. With markers on they are discs of
    ``style.joint_color``, sized from ``style.joint_size`` through the
    same pixel scale as the bones (:func:`~._style.joint_radius_px`).

    ``ghost`` faded trailing poses draw behind the live skeleton
    (oldest first); ``trajectory`` draws the root trace on the floor up
    to the current frame (solid thin line — cv2 has no dashes). Both
    reach back only to the start of the frame's pass
    (:meth:`~._scene.Scene.pass_start`), so a looped Scene draws every
    pass as the first.
    """
    import cv2

    n_skeletons = scene.num_skeletons
    pass_start = scene.pass_start(frame_idx)
    # Primitive sizes scale with the drawing resolution (1080p is the
    # 1:1 anchor: bone_width 3.0 -> 3 px there, twice that at 4K, and
    # supersampled drawing surfaces scale up with them).
    thickness = bone_width_px(style.bone_width, px_scale)
    joint_radius = joint_radius_px(style.joint_size, px_scale)
    thin = max(1, int(px_scale + 0.5))
    panel_size = (panel_w, h)

    for s, view in enumerate(scene.views):
        ctx = contexts[s]
        frame_data = view.coords[frame_idx]
        viewport = viewports[s]
        view_matrix = viewport.view_matrix(frame_idx)

        canvas: npt.NDArray[np.uint8]
        if n_skeletons > 1:
            # Contiguous per-panel canvas: clips every primitive to the
            # panel, then blits.
            canvas = np.empty((h, panel_w, 3), dtype=np.uint8)
            canvas[:] = bg_bgr
        else:
            canvas = img

        if style.floor is not None:
            _draw_floor_opencv(
                canvas, style, viewport, frame_idx, panel_w, h, 0, bg_bgr, px_scale=px_scale
            )

        if trajectory and frame_idx > pass_start:
            path = ctx.trace_path[pass_start : frame_idx + 1]
            path_2d = viewport.project(path, panel_size, frame_idx)
            cv2.polylines(canvas, [path_2d], False, ctx.trace_bgr, thin, cv2.LINE_AA)  # type: ignore[list-item]  # mypy reads a list of MatLike as Mat

        bones_arr = np.asarray(view.bones, dtype=int)

        if ghost > 0:
            ghost_thickness = max(1, int(style.bone_width * GHOST_WIDTH_FACTOR * px_scale + 0.5))
            for j in reversed(range(ghost)):  # oldest first
                gf = frame_idx - (j + 1) * ctx.ghost_lag
                if gf < pass_start:
                    continue
                gpts = viewport.project(view.coords[gf], panel_size, frame_idx)
                for b in _painter_order(view.coords[gf], bones_arr, view_matrix):
                    p_idx, c_idx = view.bones[b]
                    cv2.line(
                        canvas,
                        tuple(gpts[p_idx]),
                        tuple(gpts[c_idx]),
                        ctx.ghost_bgr[j][b],
                        ghost_thickness,
                        cv2.LINE_AA,
                    )

        pts_2d = viewport.project(frame_data, panel_size, frame_idx)

        for b in _painter_order(frame_data, bones_arr, view_matrix):
            p_idx, c_idx = view.bones[b]
            pt1 = (int(pts_2d[p_idx, 0]), int(pts_2d[p_idx, 1]))
            pt2 = (int(pts_2d[c_idx, 0]), int(pts_2d[c_idx, 1]))
            cv2.line(canvas, pt1, pt2, ctx.bone_bgr[b], thickness, cv2.LINE_AA)

        if style.joint_markers:
            for pt in pts_2d:
                cv2.circle(
                    canvas,
                    (int(pt[0]), int(pt[1])),
                    joint_radius,
                    ctx.joint_bgr,
                    -1,
                    cv2.LINE_AA,
                )
        else:
            for j, pt in enumerate(pts_2d):
                cv2.circle(
                    canvas,
                    (int(pt[0]), int(pt[1])),
                    thickness + 1,
                    tuple(int(c) for c in ctx.node_bgr[j]),
                    -1,
                    cv2.LINE_AA,
                )

        if view.label is not None:
            cv2.putText(
                canvas,
                view.label,
                (int(15 * px_scale), int(35 * px_scale)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8 * px_scale,
                ctx.label_bgr,
                max(1, int(2 * px_scale + 0.5)),
                cv2.LINE_AA,
            )

        if n_skeletons > 1:
            x0 = s * panel_w
            img[:, x0 : x0 + panel_w] = canvas

    if n_skeletons > 1:
        for s in range(1, n_skeletons):
            x = s * panel_w
            cv2.line(img, (x, 0), (x, h), (200, 200, 200), thin)


def _generate_frames(
    scene: Scene,
    style: Style,
    resolution: tuple[int, int],
    animation_options: AnimationOptions,
    *,
    fps: float | None = None,
) -> Iterator[npt.NDArray[np.uint8]]:
    """Yield one rendered BGR image ``(H, W, 3)`` per animation frame.

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
    animation_options : AnimationOptions
        What is drawn besides the skeletons, every field read: the
        camera's schedule, handed to the viewport untouched
        (:func:`~._viewport.make_viewport`); the ghosts, spaced by
        ``style.ghost_spacing`` seconds of clip time; each root's trace
        on the floor, growing with playback; and the ``Frame f/F``
        counter in the bottom-right corner.
    fps : float, optional
        The rate the video plays at, handed to the viewport, which
        times a follow camera by it on a clip whose rate is unset.
    """
    import cv2

    w, h = resolution
    bg_bgr = _to_bgr(style.background)
    ghost, trajectory = animation_options.ghost, animation_options.trajectory

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
        scene.views,
        framing="clip",
        schedule=animation_options.schedule,
        projection="ortho",
        fps=fps,
    )

    contexts = [
        _ViewDrawContext(v, viewports[s], style, s, n_skeletons, bg_bgr, ghost, trajectory)
        for s, v in enumerate(scene.views)
    ]

    for f in range(num_frames):
        img = np.empty((draw_h, draw_w, 3), dtype=np.uint8)
        img[:] = bg_bgr

        _draw_skeletons_on_frame(
            img,
            f,
            scene,
            style,
            viewports,
            contexts,
            panel_w,
            draw_h,
            bg_bgr,
            px_scale=px_scale,
            ghost=ghost,
            trajectory=trajectory,
        )

        if ss > 1:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)  # type: ignore[assignment]  # stub says any MatLike; resize keeps the uint8 it is given

        # Text and the axis indicator stamp AFTER the downsample so
        # they stay crisp at the output resolution.
        if animation_options.frame_counter:
            cv2.putText(
                img,
                frame_counter_text(scene, f),
                (counter_left_px(w), h - COUNTER_BASELINE_PX),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                COUNTER_RGB[::-1],  # BGR, this backend's drawing order
                1,
                cv2.LINE_AA,
            )

        if style.axes == "full":
            for s, viewport in enumerate(viewports):
                _draw_axis_indicator(
                    img, viewport.view_matrix(f), viewport.up_letter, panel_w // ss, h, panel_idx=s
                )

        yield img


# The options of render() this backend reads (ADR 0006).
RENDER_READS = frozenset({"follow", "turntable", "ghost", "trajectory", "frame_counter"})
# The Style values render() draws on this backend: orthographic only, no
# shadow, and no dpi (the resolution sets the pixels). axes="full" it
# draws in a reduced form of its own, as asked all the same: a small axis
# indicator in a panel's corner (ADR 0007). joint_markers=False it draws
# exactly, no markers; the bone-colored dot at each joint is how it joins
# its bones.
RENDER_DRAWS = DrawnStyle(
    {
        "bone_width",
        "bone_color",
        "color_mode",
        "chain_colors",
        "joint_markers",
        "joint_size",
        "joint_color",
        "floor",
        "floor_alpha",
        "background",
        "axes",
        "supersample",
        "ghost_spacing",
    },
    projection={"ortho"},
    shadow={False},
)
# The options of play() its notebook video, drawn by render_opencv, reads.
PLAY_READS = frozenset({"resolution", "frame_counter"})
# The Style values its notebook video draws: render()'s, but the ghosts'
# spacing, since play draws no ghosts.
PLAY_DRAWS = RENDER_DRAWS.derive(drop={"ghost_spacing"})


def render_opencv(
    scene: Scene, style: Style, writer: Writer, animation_options: AnimationOptions
) -> Path:
    """Render skeleton animation with OpenCV and hand the frames to
    *writer*, which writes the file.

    Each panel uses its own bounding box and camera so that mixed-up-axis
    side-by-side comparisons render correctly.

    Parameters
    ----------
    scene : Scene
        Prepared visualization (per-view cameras; each panel is framed
        by its own viewport).
    style : Style
        Visual styling (colors, floor, background; ``axes="full"``
        draws the per-panel axis indicator).
    writer : Writer
        The video module's writer of the target: frames are drawn at
        its target's resolution and rate.
    animation_options : AnimationOptions
        What is drawn besides the skeletons, every field read: the
        camera's schedule, ghosts, the root trail and the frame
        counter, handed whole to :func:`_generate_frames`.

    Returns
    -------
    Path
        The path to the written file.
    """
    target = writer.target
    frames = _generate_frames(scene, style, target.resolution, animation_options, fps=target.fps)
    # BGR is this backend's drawing order; the video module takes RGB.
    return writer.write(img[:, :, ::-1] for img in frames)


def _draw_axis_indicator(
    img: npt.NDArray[np.uint8],
    view_matrix: npt.NDArray[np.float64],
    up_letter: str,
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
        "x": (50, 50, 220),  # red
        "y": (50, 180, 50),  # green
        "z": (220, 120, 50),  # blue
    }

    for i, axis_name in enumerate("xyz"):
        direction_3d = np.zeros(3)
        direction_3d[i] = 1.0
        projected = view_matrix @ direction_3d
        end = origin + np.array([projected[0], -projected[1]]) * axis_len
        end = end.astype(int)

        cv2.line(img, tuple(origin), tuple(end), axis_colors[axis_name], 2, cv2.LINE_AA)
        cv2.putText(
            img,
            axis_name,
            tuple(end + 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            axis_colors[axis_name],
            1,
            cv2.LINE_AA,
        )
