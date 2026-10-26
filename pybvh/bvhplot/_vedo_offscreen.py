"""Offscreen vedo renderer: shadowed capsule-skeleton stills and videos.

The publication tier: the viewer's shaded capsules plus **projected**
per-mesh shadows via ``mesh.add_shadow()``,
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

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ._colors import (
    bone_colors_255,
    floor_palette,
    node_colors_255,
    skeleton_color_255,
)
from ._scene import Scene
from ._style import Style
from ._vedo_capsules import (
    CapsuleSkeleton,
    floor_placement,
    shadow_height,
    vedo_color,
    vedo_rgb,
)
from ._viewport import make_viewport

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ._video import Writer

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
    import vedo

    saved = vedo.settings.default_backend
    vedo.settings.default_backend = "vtk"
    try:
        yield
    finally:
        vedo.settings.default_backend = saved


def _attach_projected_shadow(
    mesh: object,
    up_letter: str,
    shadow_height: float,
) -> None:
    """Attach a projected (flattened-copy) shadow to a vedo mesh.

    Isolates vedo's confusably-named ``mesh.add_shadow()`` (works) from
    ``Plotter.add_shadows()`` (the broken shadow-map pass) — the two
    never appear side by side in pybvh code outside this module.
    """
    mesh.add_shadow(  # type: ignore[attr-defined]
        up_letter, shadow_height, c=_SHADOW_GRAY, alpha=1
    )


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
    from vedo import Plane, Plotter

    # vedo draws in perspective whatever the style asks.
    viewport = make_viewport(scene.views, projection="persp")

    plt = Plotter(offscreen=True, size=resolution, bg=vedo_color(style.background))

    capsules: list[CapsuleSkeleton] = []
    for s, view in enumerate(scene.views):
        bone_rgb = bone_colors_255(view, style, s, scene.num_skeletons)
        joint_rgb = node_colors_255(view, style, s, scene.num_skeletons, bone_rgb)
        capsule = CapsuleSkeleton(view, style.bone_width, bone_rgb, joint_rgb)
        capsule.update(view.coords[0])
        for mesh in capsule.actors:
            # Shadows must exist BEFORE the mesh joins the plotter —
            # vedo registers a mesh's shadow sub-objects at add time.
            if style.shadow and style.floor is not None:
                _attach_projected_shadow(mesh, viewport.up_letter, shadow_height(viewport))
            plt += mesh
        capsules.append(capsule)

    if style.floor is not None:
        # Flat-shaded solid plane: under real lights the plane picks up
        # a tint in this stack, and a flat floor is the paper look.
        position, normal, side = floor_placement(viewport)
        floor = Plane(pos=tuple(position), normal=tuple(normal), s=(side, side))
        floor.c(vedo_color(floor_palette(style)["face"])).lighting("off")
        plt += floor

    if scene.labels is not None:
        from vedo import Text2D

        for s, view in enumerate(scene.views):
            if view.label is None:
                continue
            color = skeleton_color_255(style, s, scene.num_skeletons)
            plt += Text2D(
                view.label, pos=(0.03, 0.95 - s * 0.05), c=vedo_rgb(color), s=1.2, font="Calco"
            )

    # The view angle is read from VTK, not written down here, so the fit
    # follows whatever vedo sets it to.
    width, height = resolution
    eye, target, up = viewport.camera(view_angle=plt.camera.GetViewAngle(), aspect=width / height)
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


# The options of render() this backend reads (ADR 0006): it draws every
# skeleton in one scene, arranged by the router.
RENDER_READS = frozenset({"spacing", "match_size"})


def render_vedo(scene: Scene, style: Style, writer: Writer) -> Path:
    """Render a shadowed capsule-skeleton animation and hand the frames
    to *writer*, which writes the file.

    Frames are screenshots at the writer's target resolution (vedo may
    deliver them larger, under HiDPI scaling). It is handed no
    :class:`~._animation_options.AnimationOptions`: the camera is
    fixed, and no ghost, trail or frame counter is drawn, as
    ``RENDER_READS`` declares.
    """
    with _vtk_backend():
        plt, capsules, camera = _build_offscreen(scene, style, writer.target.resolution)
        try:
            plt.show(camera=camera, interactive=False)

            def frames() -> Iterator[npt.NDArray[np.uint8]]:
                for f in range(scene.num_frames):
                    for capsule, view in zip(capsules, scene.views, strict=True):
                        capsule.update(view.coords[f])
                        if style.shadow and style.floor is not None:
                            for mesh in capsule.actors:
                                mesh.update_shadows()
                    plt.render()
                    yield np.asarray(plt.screenshot(asarray=True))

            return writer.write(frames())
        finally:
            plt.close()
