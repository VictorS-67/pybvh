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

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from ._style import Style
from ._viewport import make_viewport
from ._scene import Scene
from ._colors import bone_colors_255, floor_palette, node_colors_255
from ._vedo_capsules import CapsuleSkeleton, floor_placement, shadow_height

if TYPE_CHECKING:
    pass

# Opaque same-gray shadows: the merged bone and joint meshes each cast
# one, and opaque identical grays overlap invisibly (translucent
# shadows would darken where the two projections cross).
_SHADOW_GRAY = (0.72, 0.72, 0.72)


@contextmanager
def _vtk_backend():
    """Force vedo's plain-VTK backend while rendering offscreen.

    Inside a Jupyter kernel vedo auto-selects its notebook display
    backend (``"2d"``), and in that mode ``Plotter.show()`` silently
    ignores the ``camera=`` argument — every render comes out at VTK's
    default downward-looking camera (a birdview for z-up scenes).
    Offscreen rendering must not depend on the calling environment, so
    the plain backend is forced for the duration of the render and the
    user's setting is restored afterwards.
    """
    import vedo  # type: ignore[import-untyped]

    saved = vedo.settings.default_backend
    vedo.settings.default_backend = "vtk"
    try:
        yield
    finally:
        vedo.settings.default_backend = saved


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


def _build_offscreen(
    scene: Scene,
    style: Style,
    resolution: tuple[int, int],
):
    """Build the shadowed offscreen scene posed to frame 0.

    Returns ``(plotter, capsules, camera_dict)``. All skeletons share
    one scene (single-scene backend, like the viewer): one viewport of
    all the views, which supplies the cube, the camera and the floor.
    """
    from vedo import Plane, Plotter  # type: ignore[import-untyped]

    # vedo draws in perspective whatever the style asks.
    viewport = make_viewport(scene.views, projection="persp")
    center, half_span = viewport.center, viewport.half_span
    view0 = scene.views[0]

    plt = Plotter(offscreen=True, size=resolution, bg=style.background)

    r_base = CapsuleSkeleton.base_radius(half_span, style.bone_width)

    capsules: list[CapsuleSkeleton] = []
    for s, view in enumerate(scene.views):
        bone_rgb = bone_colors_255(view, style, s, scene.num_skeletons)
        joint_rgb = node_colors_255(
            view, style, s, scene.num_skeletons, bone_rgb)
        capsule = CapsuleSkeleton(
            view, r_base, bone_rgb, joint_rgb, flat_lighting=False)
        capsule.update(view.coords[0])
        for mesh in capsule.actors:
            # Shadows must exist BEFORE the mesh joins the plotter —
            # vedo registers a mesh's shadow sub-objects at add time.
            if style.shadow and style.floor is not None:
                _attach_projected_shadow(
                    mesh, viewport.up_axis, shadow_height(viewport))
            plt += mesh
        capsules.append(capsule)

    if style.floor is not None:
        # Flat-shaded solid plane: under real lights the plane picks up
        # a tint in this stack, and a flat floor is the paper look.
        position, normal, side = floor_placement(viewport)
        floor = Plane(pos=tuple(position), normal=tuple(normal),
                      s=(side, side))
        floor.c(floor_palette(style)["face"]).lighting('off')
        plt += floor

    if scene.labels is not None:
        from vedo import Text2D  # type: ignore[import-untyped]

        for s, view in enumerate(scene.views):
            if view.label is None:
                continue
            r, g, b = bone_colors_255(
                view, style, s, scene.num_skeletons)[0]
            plt += Text2D(
                view.label, pos=(0.03, 0.95 - s * 0.05),
                c=f"rgb({r},{g},{b})", s=1.2, font='Calco')

    eye, target, up = viewport.camera()
    camera = dict(
        position=eye.tolist(),
        focal_point=target.tolist(),
        viewup=up.tolist(),
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
    with _vtk_backend():
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
    codec: str = "auto",
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
    if ext != '.gif':
        # Guard BEFORE rendering every frame: the video sink is
        # cv2-based, and vedo-only installs would otherwise crash deep
        # in the writer with a raw ModuleNotFoundError.
        try:
            import cv2  # noqa: F401
        except ImportError:
            raise ImportError(
                f"Writing {ext} via the vedo backend requires "
                f"opencv-python. Install with: pip install "
                f"pybvh[opencv], or render to .gif instead.")

    with _vtk_backend():
        return _render_vedo_frames(scene, style, filepath, fps, resolution,
                                   codec)


def _render_vedo_frames(
    scene: Scene,
    style: Style,
    filepath: Path,
    fps: float,
    resolution: tuple[int, int],
    codec: str,
) -> Path:
    """The render loop of :func:`render_vedo`, run under ``_vtk_backend``."""
    ext = filepath.suffix.lower()
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
        writer = _open_writer(filepath, fps, (w, h), codec)
        writer.write(np.ascontiguousarray(first))  # type: ignore[attr-defined]
        for img in frame_iter:
            writer.write(np.ascontiguousarray(img))  # type: ignore[attr-defined]
        writer.release()  # type: ignore[attr-defined]
        return filepath
    finally:
        plt.close()
