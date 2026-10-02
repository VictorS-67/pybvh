"""Headless smoke tests for the vedo viewer shell (viewer cluster).

The playback *semantics* are covered in test_playback.py against the
pure PlaybackClock; these tests only prove the rendering/UI shell
constructs, delegates to the clock, and takes clean screenshots —
offscreen, no display needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from pybvh import read_bvh_file
from pybvh.bvhplot import _vedo
from pybvh.bvhplot._from_bvh import make_scene
from pybvh.bvhplot._style import Style

vedo = pytest.importorskip("vedo")

BVH_PATH = "bvh_data/cmu_12_01_walk.bvh"


@pytest.fixture(scope="module")
def scene():
    bvh = read_bvh_file(BVH_PATH)
    coords = bvh.node_positions()[:40]
    return make_scene([bvh], [coords], "front", None)


@pytest.fixture
def player(scene, monkeypatch):
    monkeypatch.setattr(_vedo, "_FORCE_OFFSCREEN", True)
    p = _vedo._VedoPlayer(scene, Style("paper"), 30.0, quality="high")
    yield p
    p.plt.close()


class TestCamera:
    """The viewer's camera is the viewport's, and stays so."""

    @staticmethod
    def _walk_toward_the_camera():
        import dataclasses

        from synthetic_scene import make_array_view

        from pybvh.bvhplot._scene import Scene

        view = make_array_view(n_frames=24)
        # twenty times the stride: the walk covers several body lengths
        far = view.coords.copy()
        far[..., 2] += 20 * (view.coords[:, :1, 2] - view.coords[:1, :1, 2])
        return Scene(views=[dataclasses.replace(view, coords=far)])

    def test_kept_through_show_and_reset(self, scene, monkeypatch):
        """The viewport's camera, fitted to VTK's 30 degree view angle,
        the window's aspect ratio and the band the controls leave free.
        VTK used to refit the distance and the target to everything in
        the scene, floor plane included."""
        from pybvh.bvhplot._vedo_offscreen import _vtk_backend

        monkeypatch.setattr(_vedo, "_FORCE_OFFSCREEN", True)
        with _vtk_backend():
            p = _vedo._VedoPlayer(scene, Style("paper"), 30.0, quality="high")
            try:
                assert p.plt.camera.GetViewAngle() == 30.0
                assert tuple(p.plt.window.GetSize()) == (1400, 900)
                eye, target, up = p.viewport.camera(
                    view_angle=30.0, aspect=1400 / 900, band=_vedo._FIGURE_BAND
                )

                def check():
                    camera = p.plt.camera
                    np.testing.assert_allclose(camera.GetPosition(), eye)
                    np.testing.assert_allclose(camera.GetFocalPoint(), target)
                    np.testing.assert_allclose(camera.GetViewUp(), up, atol=1e-12)

                check()
                p.show()
                check()
                p.plt.camera.SetPosition(*(eye * 3.0))
                p._on_reset_camera()
                check()
            finally:
                p.plt.close()

    def test_reset_refits_to_a_resized_window(self, scene, monkeypatch):
        """A window resized after opening, reset: the camera is the
        viewport's fit at the window's new aspect ratio, not the one it
        opened with."""
        from pybvh.bvhplot._vedo_offscreen import _vtk_backend

        monkeypatch.setattr(_vedo, "_FORCE_OFFSCREEN", True)
        with _vtk_backend():
            p = _vedo._VedoPlayer(scene, Style("paper"), 30.0, quality="high")
            try:
                p.show()
                opened = p.plt.camera.GetPosition()
                p.plt.window.SetSize(600, 1000)
                assert tuple(p.plt.window.GetSize()) == (600, 1000)
                p._on_reset_camera()
                eye, target, up = p.viewport.camera(
                    view_angle=30.0, aspect=600 / 1000, band=_vedo._FIGURE_BAND
                )
                camera = p.plt.camera
                assert not np.allclose(opened, eye)
                np.testing.assert_allclose(camera.GetPosition(), eye)
                np.testing.assert_allclose(camera.GetFocalPoint(), target)
                np.testing.assert_allclose(camera.GetViewUp(), up, atol=1e-12)
            finally:
                p.plt.close()

    def test_a_clip_in_place_opens_between_the_controls(self, monkeypatch):
        """Every coordinate, projected by VTK's own camera, lands above
        the frame slider (the top of the transport bar) and below the
        top row of buttons: the feet used to reach behind the bar."""
        from pybvh.bvhplot._vedo_offscreen import _vtk_backend

        bvh = read_bvh_file("bvh_data/bvh_example.bvh")
        coords = bvh.node_positions()
        in_place = make_scene([bvh], [coords], "front", None)
        monkeypatch.setattr(_vedo, "_FORCE_OFFSCREEN", True)
        with _vtk_backend():
            p = _vedo._VedoPlayer(in_place, Style("paper"), 30.0, quality="high")
            try:
                p.show()
                width, height = p.plt.window.GetSize()
                camera = p.plt.camera
                to_picture = camera.GetCompositeProjectionTransformMatrix(width / height, -1.0, 1.0)
                matrix = np.array(
                    [[to_picture.GetElement(i, j) for j in range(4)] for i in range(4)]
                )
                slider = p.slider.GetRepresentation()
                slider_line = slider.GetPoint1Coordinate().GetValue()[1]
                top_row = max(y0 for _, y0, _, _, _ in p._buttons)
            finally:
                p.plt.close()
        points = coords.reshape(-1, 3)
        clip = np.c_[points, np.ones(len(points))] @ matrix.T
        across = clip[:, 0] / clip[:, 3]
        heights = 0.5 + 0.5 * clip[:, 1] / clip[:, 3]
        assert heights.min() > slider_line
        assert heights.max() < top_row
        assert np.abs(across).max() < 1.0

    @pytest.mark.parametrize("style", ["paper", "debug"])
    def test_the_skeleton_is_never_clipped_in_depth(self, monkeypatch, style):
        """VTK fits the clipping planes to what is in the scene at the
        moment. A skeleton that walks toward the camera must still be
        inside them on the last frame, and back on the first one after
        a reset there. Without a floor (debug) nothing else widens the
        planes, so that is the case that fails when they are not
        refitted."""
        from pybvh.bvhplot._vedo_offscreen import _vtk_backend

        walk = self._walk_toward_the_camera()

        def depths_inside(p, frame):
            camera = p.plt.camera
            eye = np.array(camera.GetPosition())
            direction = np.array(camera.GetDirectionOfProjection())
            depth = (walk.views[0].coords[frame] - eye) @ direction
            near, far_plane = camera.GetClippingRange()
            return near <= depth.min() and depth.max() <= far_plane

        monkeypatch.setattr(_vedo, "_FORCE_OFFSCREEN", True)
        with _vtk_backend():
            p = _vedo._VedoPlayer(walk, Style(style), 30.0, quality="high")
            try:
                p.show()
                assert depths_inside(p, 0)
                p._update_frame(23)
                assert depths_inside(p, 23)
                p._on_reset_camera()
                assert depths_inside(p, 23)
                p._update_frame(0)
                assert depths_inside(p, 0)
            finally:
                p.plt.close()


class TestPlayerShell:
    def test_constructs_with_chain_colors(self, player):
        # merged bone mesh exists and carries per-point colors
        capsule = player._capsules[0]
        assert capsule is not None and capsule.bones_mesh is not None
        assert capsule.bones_mesh.pointcolors is not None
        assert len(capsule.bones_mesh.pointcolors) > 0

    def test_delegates_to_clock(self, player):
        assert player.clock.playing
        player._toggle_play()
        assert not player.clock.playing
        player._jump_to(10)
        assert player.clock.frame == 10
        player._on_next()
        assert player.clock.frame == 11

    def test_clean_screenshot_hides_and_restores_ui(self, player, tmp_path):
        out = tmp_path / "shot.png"
        visible_before = [a.actor.GetVisibility() for a in player._ui_actors]
        fname = player.screenshot(str(out), scale=1)
        assert fname == str(out)
        assert out.exists() and out.stat().st_size > 0
        visible_after = [a.actor.GetVisibility() for a in player._ui_actors]
        assert visible_before == visible_after

    def test_fps_switch_resamples(self, player):
        idx15 = player.clock.fps_presets.index(15)
        player._set_fps(idx15)
        assert player.clock.target_fps == 15
        assert player.num_frames == len(player.coords_list[0])
        assert player.clock.frame == 0


class TestShading:
    """The viewer lights its capsules as the offscreen renderer does,
    under one headlight that stays at the camera."""

    def test_capsules_take_diffuse_light(self, player):
        for actor in player._capsules[0].actors:
            prop = actor.actor.GetProperty()
            assert prop.GetAmbient() == pytest.approx(0.2)
            assert prop.GetDiffuse() == pytest.approx(0.8)
            assert prop.GetSpecular() == pytest.approx(0.1)

    def test_one_headlight_stays_at_the_camera(self, player):
        """Wherever the camera goes, by the mouse or by the reset key,
        the side of a capsule facing it is the lit one."""

        def lights():
            collection = player.plt.renderer.GetLights()
            return [collection.GetItemAsObject(i) for i in range(collection.GetNumberOfItems())]

        camera = player.plt.camera
        for move in (
            lambda: camera.Azimuth(90),
            lambda: camera.Elevation(40),
            player._on_reset_camera,
        ):
            move()
            player._update_frame(0)
            [light] = lights()
            assert light.LightTypeIsHeadlight() and light.GetSwitch()
            np.testing.assert_allclose(light.GetPosition(), camera.GetPosition())
            np.testing.assert_allclose(light.GetFocalPoint(), camera.GetFocalPoint())


class TestPlayerDarkStyle:
    def test_dark_background(self, scene, monkeypatch, tmp_path):
        monkeypatch.setattr(_vedo, "_FORCE_OFFSCREEN", True)
        p = _vedo._VedoPlayer(scene, Style("dark"), 30.0, quality="high")
        try:
            out = tmp_path / "dark.png"
            p.screenshot(str(out), scale=1)
            assert out.exists()
        finally:
            p.plt.close()


class TestGridFloor:
    """The grid floor ("grid", and "checker" which falls back to it) is
    built flat in vedo's XY plane and turned to face up. vedo turns
    about the world origin, so the grid must be turned first and moved
    after: placed first, it swung away from under the skeleton on any
    clip not centred on the origin."""

    @staticmethod
    def _scene_with_up(up):
        import dataclasses

        from synthetic_scene import make_array_view

        from pybvh.bvhplot._scene import Scene

        view = make_array_view(n_frames=12)
        if up == "+y":
            return Scene(views=[view])
        order = {"+z": [0, 2, 1], "+x": [1, 0, 2]}[up]
        forward = {"+z": "+y", "+x": "+z"}[up]
        heights = view.coords[..., order][..., "xyz".index(up[1])]
        return Scene(
            views=[
                dataclasses.replace(
                    view,
                    coords=view.coords[..., order],
                    rest_coords=view.rest_coords[..., order],
                    rest_up=up,
                    up=up,
                    forward_axis=forward,
                    root_heading=None,
                    floor_height=float(heights.min()),
                )
            ]
        )

    @pytest.mark.parametrize("up", ["+y", "+z", "+x"])
    @pytest.mark.parametrize("kind", ["grid", "checker"])
    def test_the_grid_lies_where_the_plane_goes(self, up, kind, monkeypatch):
        from pybvh.bvhplot._vedo_capsules import FLOOR_EPSILON

        monkeypatch.setattr(_vedo, "_FORCE_OFFSCREEN", True)
        p = _vedo._VedoPlayer(
            self._scene_with_up(up), Style("paper", floor=kind), 30.0, quality="high"
        )
        try:
            grids = [o for o in p.plt.objects if type(o).__name__ == "Grid"]
            assert len(grids) == 1
            vertices = np.asarray(grids[0].vertices)
            viewport = p.viewport
            # the scene is not centred on the origin, or this proves nothing
            assert np.abs(viewport.center).max() > 0.2
            expected = viewport.floor_quad()
            for axis in viewport.ground_axes:
                assert vertices[:, axis].min() == pytest.approx(expected[:, axis].min(), rel=1e-5)
                assert vertices[:, axis].max() == pytest.approx(expected[:, axis].max(), rel=1e-5)
            np.testing.assert_allclose(
                vertices[:, viewport.up_index],
                viewport.below_floor(FLOOR_EPSILON * viewport.half_span),
                rtol=1e-5,
                atol=1e-6,
            )
        finally:
            p.plt.close()


def _rendered(player, tmp_path):
    """The viewer's current frame as an (H, W, 3) uint8 image."""
    from PIL import Image

    path = player.screenshot(str(tmp_path / "shot.png"), scale=1)
    return np.asarray(Image.open(path).convert("RGB"))


def _pixels_of(image, rgb, tol=3):
    """How many pixels of *image* are *rgb* as the capsules shade it.

    Ambient and diffuse light scale a capsule's color, from the ambient
    0.2 in shadow to 1 lit head on (a flat-lit line is always 1), and
    the specular highlight adds gray: a pixel is ``shade * rgb + gray``
    with ``shade`` in [0.2, 1] and ``gray >= 0``, within *tol* per
    channel. That is *rgb*'s hue, but not every color of that hue: a
    fully saturated one (a scalar map's) needs negative gray. Black,
    the background and the floor have no shade of it."""
    color = np.asarray(rgb, dtype=float)
    pixels = image.reshape(-1, 3).astype(float)
    chroma = pixels.max(axis=-1) - pixels.min(axis=-1)
    shade = chroma / (color.max() - color.min())
    gray = pixels.min(axis=-1) - shade * color.min()
    model = shade[:, None] * color + gray[:, None]
    fits = np.abs(pixels - model).max(axis=-1) <= tol
    lit = (shade >= 0.2) & (shade <= 1 + tol / 255) & (gray >= -tol)
    return int((fits & lit).sum())


class TestColors:
    """The viewer draws the colors the style resolves to, the ones the
    offscreen renderer draws. vedo read the "rgb(r,g,b)" strings the
    viewer used to pass as black, and without per-vertex colors the
    capsules fell back to a scalar map over the tube radius."""

    BLUE, RED = (50, 120, 255), (220, 50, 50)  # the palette's first two

    @staticmethod
    def _pair(labels):
        walk = read_bvh_file(BVH_PATH)
        mirror = walk.mirror()
        coords = [b.node_positions()[:10] for b in (walk, mirror)]
        return make_scene([walk, mirror], coords, "front", labels).spread(40)

    @pytest.fixture(scope="class")
    def pair(self):
        """Unlabelled, so that only the skeletons draw palette colors."""
        return self._pair(None)

    @pytest.fixture(scope="class")
    def labelled_pair(self):
        return self._pair(["walk", "mirror"])

    @pytest.mark.parametrize("quality", ["high", "fast"])
    def test_each_skeleton_of_a_pair_has_its_palette_color(
        self, pair, quality, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(_vedo, "_FORCE_OFFSCREEN", True)
        p = _vedo._VedoPlayer(pair, Style("paper", floor=None), 30.0, quality=quality)
        try:
            image = _rendered(p, tmp_path)
        finally:
            p.plt.close()
        # fast mode draws one-pixel lines: a few dozen pixels is a skeleton
        assert _pixels_of(image, self.BLUE) > 50
        assert _pixels_of(image, self.RED) > 50

    def test_the_debug_style_draws_its_bone_color(self, monkeypatch, tmp_path):
        """A single skeleton used to be drawn in a fixed amber whatever
        the style said; render(backend="vedo") draws bone_color."""
        walk = read_bvh_file(BVH_PATH)
        scene = make_scene([walk], [walk.node_positions()[:10]], "front", None)
        debug_blue = (25, 51, 204)  # bone_color (0.1, 0.2, 0.8)
        monkeypatch.setattr(_vedo, "_FORCE_OFFSCREEN", True)
        p = _vedo._VedoPlayer(scene, Style("debug"), 30.0, quality="high")
        try:
            image = _rendered(p, tmp_path)
        finally:
            p.plt.close()
        assert _pixels_of(image, debug_blue) > 1000

    def test_labels_and_trails_carry_their_skeletons_color(self, labelled_pair, monkeypatch):
        import vedo

        monkeypatch.setattr(_vedo, "_FORCE_OFFSCREEN", True)
        p = _vedo._VedoPlayer(labelled_pair, Style("paper"), 30.0, quality="high")
        try:
            labels = {o.text(): o for o in p.plt.objects if isinstance(o, vedo.Text2D)}
            for name, trail, rgb in zip(["walk", "mirror"], p._trail_actors, [self.BLUE, self.RED]):
                expected = np.asarray(rgb) / 255
                np.testing.assert_allclose(labels[name].properties.GetColor(), expected)
                np.testing.assert_allclose(trail.properties.GetColor(), expected)
        finally:
            p.plt.close()


def _rows(colors):
    """The distinct RGB rows of a vedo color array (alpha dropped)."""
    return {tuple(int(c) for c in row[:3]) for row in colors}


class TestColorModes:
    """Each color mode reaches the viewer's actors, read off their
    per-cell and per-point color arrays."""

    BLUE, RED = TestColors.BLUE, TestColors.RED

    @staticmethod
    def _player(scene, style, quality, monkeypatch):
        monkeypatch.setattr(_vedo, "_FORCE_OFFSCREEN", True)
        return _vedo._VedoPlayer(scene, style, 30.0, quality=quality)

    @staticmethod
    def _chain_rgb(style, view):
        from pybvh.bvhplot._colors import rgb255

        return [rgb255(style.chain_colors[chain]) for chain in view.bone_chains]

    def test_fast_quality_draws_chains_for_one_skeleton(self, scene, monkeypatch):
        """Fast quality drew one flat color, black, before."""
        style = Style("paper")
        p = self._player(scene, style, "fast", monkeypatch)
        try:
            bones = p._lines_actors[0].cellcolors[:, :3]
            expected = self._chain_rgb(style, scene.views[0])
            np.testing.assert_array_equal(bones, expected)
            assert len(set(map(tuple, expected))) == 5
        finally:
            p.plt.close()

    def test_forced_chains_color_every_skeleton_of_a_pair(self, monkeypatch):
        pair = TestColors._pair(None)
        style = Style("paper", color_mode="chains")
        p = self._player(pair, style, "high", monkeypatch)
        try:
            for capsule, view in zip(p._capsules, pair.views):
                assert _rows(capsule.bones_mesh.pointcolors) == set(self._chain_rgb(style, view))
        finally:
            p.plt.close()

    def test_explicit_skeleton_mode_draws_one_skeleton_in_the_palette(self, scene, monkeypatch):
        p = self._player(scene, Style("paper", color_mode="skeleton"), "high", monkeypatch)
        try:
            capsule = p._capsules[0]
            assert _rows(capsule.bones_mesh.pointcolors) == {self.BLUE}
            # The root joint is nobody's child and takes the spine
            # color in every mode (node_colors_255).
            assert _rows(capsule.joints_mesh.pointcolors) == {self.BLUE, (58, 63, 74)}
        finally:
            p.plt.close()

    def test_under_chains_label_and_trail_take_the_spine_color(self, monkeypatch):
        """Not the first bone's color, which depends on the order the
        file lists the root's children in."""
        import vedo

        from pybvh.bvhplot._colors import rgb255

        walk = read_bvh_file(BVH_PATH)
        scene = make_scene([walk], [walk.node_positions()[:10]], "front", ["walk"])
        style = Style("paper")
        p = self._player(scene, style, "high", monkeypatch)
        try:
            spine = np.asarray(rgb255(style.chain_colors["spine"])) / 255
            label = next(
                o for o in p.plt.objects if isinstance(o, vedo.Text2D) and o.text() == "walk"
            )
            np.testing.assert_allclose(label.properties.GetColor(), spine)
            np.testing.assert_allclose(p._trail_actors[0].properties.GetColor(), spine)
        finally:
            p.plt.close()


class TestStyleColorsReachVedoParsed:
    """Every color a Style supplies is read by matplotlib's parser, as
    in the other backends, and reaches vedo as floats. vedo's own
    parser crashed on short hex ("#fff") and read matplotlib-only
    names such as "C0" as gray."""

    @pytest.mark.parametrize(
        "background, expected",
        [
            ("#fff", (255, 255, 255)),
            ("C0", (31, 119, 180)),  # matplotlib's first cycle color
        ],
    )
    @pytest.mark.parametrize("quality", ["high", "fast"])
    def test_background(self, scene, background, expected, quality, monkeypatch):
        monkeypatch.setattr(_vedo, "_FORCE_OFFSCREEN", True)
        p = _vedo._VedoPlayer(scene, Style("paper", background=background), 30.0, quality=quality)
        try:
            np.testing.assert_allclose(p.plt.renderer.GetBackground(), np.asarray(expected) / 255)
        finally:
            p.plt.close()

    @pytest.mark.parametrize("floor, key", [("solid", "face"), ("grid", "grid")])
    def test_floor(self, scene, floor, key, monkeypatch):
        from pybvh.bvhplot._colors import floor_palette, rgb255

        monkeypatch.setattr(_vedo, "_FORCE_OFFSCREEN", True)
        style = Style("paper", floor=floor)
        p = _vedo._VedoPlayer(scene, style, 30.0, quality="high")
        try:
            plane = next(o for o in p.plt.objects if isinstance(o, (vedo.Plane, vedo.Grid)))
            np.testing.assert_allclose(
                plane.properties.GetColor(), np.asarray(rgb255(floor_palette(style)[key])) / 255
            )
        finally:
            p.plt.close()


def _text_box(text2d, renderer, dpi):
    """The pixel box VTK draws a Text2D's text and background in.

    The text renderer's own bounding box of the string, around the
    position the actor computes for this window: the quad VTK textures
    the text onto, padding and background included."""
    import vtk

    corners = [0, 0, 0, 0]
    vtk.vtkTextRenderer.GetInstance().GetBoundingBox(
        text2d.mapper.GetTextProperty(), text2d.mapper.GetInput(), corners, dpi
    )
    # .actor is the VTK actor on every vedo the extra allows: the Text2D
    # itself until vedo 2026, which made Text2D wrap one instead.
    x, y = text2d.actor.GetPositionCoordinate().GetComputedDisplayValue(renderer)
    return (x + corners[0], x + corners[1], y + corners[2], y + corners[3])


def _slider_box(slider, renderer):
    """The pixel box of the frame slider's drawn geometry.

    An offscreen plotter has no interactor to enable the widget, so its
    representation is built here, for this renderer, as a window
    would."""
    import vtk

    representation = slider.GetRepresentation()
    representation.SetRenderer(renderer)
    representation.BuildRepresentation()
    actors = vtk.vtkPropCollection()
    representation.GetActors2D(actors)
    boxes = []
    for i in range(actors.GetNumberOfItems()):
        mapper = actors.GetItemAsObject(i).GetMapper()
        if isinstance(mapper, vtk.vtkPolyDataMapper2D):
            mapper.GetInputAlgorithm().Update()
            # Cell by cell: the parts share one point array, some of
            # whose points no part of the slider uses or sets.
            drawn = mapper.GetInput()
            for cell in range(drawn.GetNumberOfCells()):
                x0, x1, y0, y1, _, _ = drawn.GetCell(cell).GetBounds()
                boxes.append((x0, x1, y0, y1))
    assert boxes, "the slider drew nothing"
    x0s, x1s, y0s, y1s = zip(*boxes)
    return (min(x0s), max(x1s), min(y0s), max(y1s))


def _overlap(a, b):
    """Whether two (x0, x1, y0, y1) boxes share any area."""
    return min(a[1], b[1]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[2], b[2])


class TestSkeletonLabels:
    """Each skeleton's label is readable whatever its length: drawn
    clear of the other labels and of every control. Labels used to sit
    0.15 of the window's width apart on one line, which holds about 13
    characters."""

    LABELS = ["cmu_12_01_walk (original)", "cmu_12_01_walk (mirrored)", "cmu_12_01_walk third clip"]

    @staticmethod
    def _scene(labels):
        walk = read_bvh_file(BVH_PATH)
        clips = [walk, walk.mirror(), walk, walk.mirror()][: len(labels)]
        coords = [b.node_positions()[:10] for b in clips]
        return make_scene(clips, coords, "front", labels).spread("auto")

    @staticmethod
    def _drawn(scene, window_size, monkeypatch):
        """The pixel boxes of the labels (by text), of Reset Cam's text,
        and of every control, in the viewer rendered at *window_size*."""
        from pybvh.bvhplot._vedo_offscreen import _vtk_backend

        monkeypatch.setattr(_vedo, "_FORCE_OFFSCREEN", True)
        with _vtk_backend():
            p = _vedo._VedoPlayer(scene, Style("paper"), 30.0, quality="high")
            try:
                p.plt.window.SetSize(*window_size)
                p.show()
                renderer = p.plt.renderer
                dpi = p.plt.window.GetDPI()
                width, height = p.plt.window.GetSize()
                assert (width, height) == window_size
                labels = {
                    o.text(): _text_box(o, renderer, dpi)
                    for o in p.plt.objects
                    if isinstance(o, vedo.Text2D) and o.text() in scene.labels
                }
                reset = _text_box(p.reset_btn, renderer, dpi)
                controls = [_text_box(t, renderer, dpi) for t in p._ui_actors if t.text()]
                controls += [
                    (x0 * width, (x0 + w) * width, y0 * height, (y0 + h) * height)
                    for x0, y0, w, h, _ in p._buttons
                ]
                controls.append(_slider_box(p.slider, renderer))
            finally:
                p.plt.close()
        return labels, reset, controls

    @pytest.mark.parametrize("window_size", [(1400, 900), (1400, 650)])
    def test_long_labels_clear_each_other_and_the_controls(self, window_size, monkeypatch):
        """At the default window and at one short enough that the left
        panel's rows are about to meet, 39 pixels apart for its buttons'
        38-pixel boxes."""
        assert {len(label) for label in self.LABELS} == {25}
        drawn, _, controls = self._drawn(self._scene(self.LABELS), window_size, monkeypatch)
        labels = [drawn[label] for label in self.LABELS]
        for i, label in enumerate(labels):
            for other in labels[i + 1 :]:
                assert not _overlap(label, other), (label, other)
            for control in controls:
                assert not _overlap(label, control), (label, control)

    def test_unlabelled_skeletons_leave_no_empty_rows(self, monkeypatch):
        """Shown labels stack from the panel down, each within one line
        of the one above, Reset Cam's text first: a skeleton without a
        label used to keep its empty row."""
        drawn, reset, _ = self._drawn(
            self._scene([None, "second", None, "fourth"]), (1400, 900), monkeypatch
        )
        above = reset
        for name in ["second", "fourth"]:
            box = drawn[name]
            line = box[3] - box[2]
            assert 0 < above[2] - box[3] < line, (name, above, box)
            above = box
