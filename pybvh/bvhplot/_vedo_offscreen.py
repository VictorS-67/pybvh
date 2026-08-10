"""Offscreen vedo renderer: shadowed capsule-skeleton stills and videos.

The publication tier: real diffuse lighting (unlike the viewer's flat
ambient) and **projected** per-mesh shadows via ``mesh.add_shadow()``,
wrapped in :func:`_attach_projected_shadow`. The renderer-level VTK
shadow-map pass (``Plotter.add_shadows()``) is deliberately not used —
it casts nothing offscreen and tints the floor in the supported vedo
versions; see ``docs/adr/0001-vedo-projected-shadows.md``. Projected
shadows are hard-edged parallel projections (no soft penumbra, no
self-shadowing); for raytraced softness use a Blender pipeline
(pybvh-blender).

Headless-safe: no display or interactor is needed.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pathlib import Path
from typing import TYPE_CHECKING

from ._common import (
    Scene,
    Style,
    bone_colors_for_view,
    build_view_matrix,
)
from ._vedo_capsules import CapsuleSkeleton

if TYPE_CHECKING:
    pass

# Opaque same-gray shadows: the merged bone and joint meshes each cast
# one, and opaque identical grays overlap invisibly (translucent
# shadows would darken where the two projections cross).
_SHADOW_GRAY = (0.72, 0.72, 0.72)


def _attach_projected_shadow(
    mesh: object,
    up_axis: str,
    shadow_height: float,
) -> None:
    """Attach a projected (flattened-copy) shadow to a vedo mesh.

    Isolates vedo's confusably-named ``mesh.add_shadow()`` (works) from
    ``Plotter.add_shadows()`` (the broken shadow-map pass) — the two
    never appear side by side in pybvh code outside this module.
    """
    mesh.add_shadow(  # type: ignore[attr-defined]
        up_axis, shadow_height, c=_SHADOW_GRAY, alpha=1)


def _chain_rgb_for_view(scene: Scene, style: Style, s: int):
    """Per-bone (0-255) RGB for CapsuleSkeleton, honoring color modes."""
    from matplotlib.colors import to_rgb

    colors = bone_colors_for_view(
        scene.views[s], style, s, scene.num_skeletons)
    rgb = [tuple(int(c * 255) for c in to_rgb(col)) for col in colors]
    spine = tuple(int(c * 255) for c in to_rgb(
        style.chain_colors.get("spine", "#3A3F4A")))
    return rgb, spine


def _build_offscreen(
    scene: Scene,
    style: Style,
    resolution: tuple[int, int],
):
    """Build the shadowed offscreen scene posed to frame 0.

    Returns ``(plotter, capsules, camera_dict)``. All skeletons share
    one scene (single-scene backend, like the viewer): unified bounding
    box, camera from the first view, one floor at the lowest per-view
    floor height.
    """
    from vedo import Plane, Plotter  # type: ignore[import-untyped]

    center, half_span = scene.unified_box()
    view0 = scene.views[0]
    up_idx = view0.up_index
    floor_height = min(v.floor_height for v in scene.views)

    plt = Plotter(offscreen=True, size=resolution, bg=style.background)

    r_base = half_span * 0.013 * (style.bone_width / 3.0)
    floor_height_low = floor_height - 0.004 * half_span
    shadow_height = floor_height - 0.002 * half_span

    capsules: list[CapsuleSkeleton] = []
    for s, view in enumerate(scene.views):
        chain_rgb, spine_rgb = _chain_rgb_for_view(scene, style, s)
        capsule = CapsuleSkeleton(
            view, r_base, "#AAAAAA",
            chain_rgb=chain_rgb, spine_rgb=spine_rgb,
            flat_lighting=False)
        capsule.update(view.coords[0])
        for mesh in capsule.actors:
            # Shadows must exist BEFORE the mesh joins the plotter —
            # vedo registers a mesh's shadow sub-objects at add time.
            if style.shadow and style.floor is not None:
                _attach_projected_shadow(
                    mesh, view0.up_axis, shadow_height)
            plt += mesh
        capsules.append(capsule)

    if style.floor is not None:
        # Flat-shaded solid plane: under real lights the plane picks up
        # a tint in this stack, and a flat floor is the paper look.
        normal = [0.0, 0.0, 0.0]
        normal[up_idx] = 1.0
        floor_pos = center.copy()
        floor_pos[up_idx] = floor_height_low
        dark = style.background not in ("white", "#FFFFFF", "#ffffff")
        floor = Plane(pos=tuple(floor_pos), normal=tuple(normal),
                      s=(half_span * 4, half_span * 4))
        floor.c('#2A2E36' if dark else '#EDEDF1').lighting('off')
        plt += floor

    view_mat = build_view_matrix(
        view0.azimuth, view0.elevation, view0.up_axis)
    camera = dict(
        position=(center + view_mat[2] * half_span * 4.0).tolist(),
        focal_point=center.tolist(),
        viewup=view_mat[1].tolist(),
    )
    return plt, capsules, camera


def frame_vedo(
    scene: Scene,
    style: Style,
    *,
    resolution: tuple[int, int] = (1100, 1000),
    filepath: str | Path | None = None,
) -> npt.NDArray[np.uint8]:
    """Render one shadowed capsule-skeleton still.

    Returns the image as an ``(H, W, 3)`` uint8 RGB array (pybvh's
    framework-agnostic contract); optionally also writes *filepath*.
    """
    plt, _capsules, camera = _build_offscreen(scene, style, resolution)
    try:
        plt.show(camera=camera, interactive=False)
        img = np.asarray(plt.screenshot(asarray=True))
        if filepath is not None:
            from PIL import Image
            Image.fromarray(img).save(filepath)
    finally:
        plt.close()
    return img


def render_vedo(
    scene: Scene,
    style: Style,
    filepath: Path,
    fps: float,
    resolution: tuple[int, int] = (1100, 1000),
) -> Path:
    """Render a shadowed capsule-skeleton video (.mp4/.mov/.avi/.gif).

    Per-frame screenshots feed the same sinks the OpenCV backend uses:
    ``cv2.VideoWriter`` for video containers (requires opencv-python),
    Pillow for GIF. Camera is fixed; ``follow``/``turntable``/ghosting
    are not supported on this backend.
    """
    ext = filepath.suffix.lower()
    if ext not in {'.mp4', '.mov', '.avi', '.gif'}:
        raise ValueError(
            f"The vedo backend cannot write {ext!r} files. Supported: "
            f".mp4, .mov, .avi, .gif. Use backend='matplotlib' for "
            f"other formats.")

    plt, capsules, camera = _build_offscreen(scene, style, resolution)
    try:
        plt.show(camera=camera, interactive=False)

        def frames_bgr():
            for f in range(scene.num_frames):
                for capsule, view in zip(capsules, scene.views):
                    capsule.update(view.coords[f])
                    if style.shadow and style.floor is not None:
                        for mesh in capsule.actors:
                            mesh.update_shadows()
                plt.render()
                rgb = np.asarray(plt.screenshot(asarray=True))
                yield rgb[:, :, ::-1]

        if ext == '.gif':
            from ._opencv import _render_gif
            return _render_gif(frames_bgr(), filepath, fps)

        from ._opencv import _open_writer
        frame_iter = frames_bgr()
        first = next(frame_iter)
        # vedo may deliver a screenshot size differing from the request
        # (HiDPI scaling); size the writer from the actual frames.
        h, w = first.shape[:2]
        writer = _open_writer(filepath, fps, (w, h))
        writer.write(np.ascontiguousarray(first))  # type: ignore[attr-defined]
        for img in frame_iter:
            writer.write(np.ascontiguousarray(img))  # type: ignore[attr-defined]
        writer.release()  # type: ignore[attr-defined]
        return filepath
    finally:
        plt.close()
