"""Tests for the Viewport: the geometry of a picture, computed once.

Everything here runs on Scenes built from plain arrays
(``tests/synthetic_scene.py``): the viewport is a function of views,
and no Bvh is needed to say what it should compute.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from pybvh.bvhplot._scene import compute_unified_limits
from pybvh.bvhplot._viewport import (
    EYE_DISTANCE,
    FIT_FRACTION,
    FLOOR_EXTENT,
    FLOOR_INSET,
    FRAMING_MARGIN,
    Viewport,
    build_view_matrix,
    compute_follow_azimuths,
    framing_bounds,
    make_viewport,
    panel_viewports,
    turntable_azimuths,
)
from synthetic_scene import make_array_scene, make_array_view


@pytest.fixture
def view():
    """A stick person walking along +z, y up, 24 frames."""
    return make_array_view(n_frames=24)


def _negative_up(view):
    """The same motion in a world whose up axis is -y."""
    flipped = view.coords * np.array([1.0, -1.0, 1.0])
    return dataclasses.replace(
        view, coords=flipped, up="-y",
        rest_coords=view.rest_coords * np.array([1.0, -1.0, 1.0]),
        floor_height=float(flipped[..., 1].max()))


class TestStillFraming:
    def test_the_box_is_the_cube_around_the_coords(self, view):
        viewport = make_viewport([view], framing="still", include_floor=False)
        center, half_span = compute_unified_limits([view.coords])
        np.testing.assert_array_equal(viewport.center, center)
        assert viewport.half_span == half_span
        np.testing.assert_array_equal(viewport.lo, center - half_span)
        np.testing.assert_array_equal(viewport.hi, center + half_span)

    def test_a_floor_below_the_cube_pulls_the_box_down_to_it(self, view):
        low = dataclasses.replace(view, floor_height=-5.0)
        viewport = make_viewport([low], framing="still", include_floor=True)
        up = viewport.up_index
        inset = FLOOR_INSET * viewport.half_span
        assert viewport.lo[up] == pytest.approx(-5.0 - inset)
        # the cube keeps its size, and the scale cube does not move
        np.testing.assert_allclose(
            viewport.hi - viewport.lo, 2 * viewport.half_span)
        center, _ = compute_unified_limits([view.coords])
        np.testing.assert_array_equal(viewport.center, center)

    def test_a_floor_well_inside_the_cube_moves_nothing(self, view):
        center, half_span = compute_unified_limits([view.coords])
        inside = dataclasses.replace(
            view, floor_height=float(center[1]))
        viewport = make_viewport([inside], framing="still")
        np.testing.assert_array_equal(viewport.lo, center - half_span)

    def test_the_shift_follows_a_negative_up_axis(self, view):
        negative = dataclasses.replace(_negative_up(view), floor_height=5.0)
        viewport = make_viewport([negative], framing="still")
        up = viewport.up_index
        inset = FLOOR_INSET * viewport.half_span
        # the ground is at the coordinate maximum: the box moves up to it
        assert viewport.hi[up] == pytest.approx(5.0 + inset)

    def test_without_a_floor_the_box_ignores_it(self, view):
        low = dataclasses.replace(view, floor_height=-5.0)
        viewport = make_viewport([low], framing="still", include_floor=False)
        assert viewport.lo[viewport.up_index] > -5.0


class TestClipFraming:
    @pytest.mark.parametrize("motion, rotating", [
        ("fixed", False), ("turntable", True), ("follow", True)])
    def test_the_box_is_the_one_the_motion_sweeps(
            self, view, motion, rotating):
        viewport = make_viewport([view], framing="clip", motion=motion)
        lo, hi = framing_bounds(view, rotating=rotating)
        np.testing.assert_array_equal(viewport.lo, lo)
        np.testing.assert_array_equal(viewport.hi, hi)

    def test_each_axis_gets_its_own_extent(self, view):
        viewport = make_viewport([view], framing="clip")
        spans = viewport.hi - viewport.lo
        assert spans.min() < 0.8 * spans.max()

    def test_the_floor_is_inside_the_box(self, view):
        low = dataclasses.replace(view, floor_height=-5.0)
        viewport = make_viewport([low], framing="clip")
        assert viewport.lo[viewport.up_index] < -5.0

    def test_without_a_floor_the_box_hugs_the_coords(self, view):
        low = dataclasses.replace(view, floor_height=-5.0)
        viewport = make_viewport([low], framing="clip", include_floor=False)
        points = view.coords.reshape(-1, 3)
        raw_lo, raw_hi = points.min(axis=0), points.max(axis=0)
        # the margin is a fraction of the longest raw span
        pad = FRAMING_MARGIN * float((raw_hi - raw_lo).max())
        np.testing.assert_allclose(viewport.lo, raw_lo - pad, rtol=1e-12)
        np.testing.assert_allclose(viewport.hi, raw_hi + pad, rtol=1e-12)

    def test_the_margin_is_a_fraction_of_the_longest_raw_span(self, view):
        viewport = make_viewport([view], framing="clip")
        points = view.coords.reshape(-1, 3)
        raw_lo, raw_hi = points.min(axis=0), points.max(axis=0)
        raw_lo[1] = min(raw_lo[1], view.floor_height)
        pad = FRAMING_MARGIN * float((raw_hi - raw_lo).max())
        np.testing.assert_allclose(viewport.lo, raw_lo - pad, rtol=1e-12)
        np.testing.assert_allclose(viewport.hi, raw_hi + pad, rtol=1e-12)

    def test_a_floor_above_the_motion_is_in_the_box_too(self, view):
        """The box contains the ground wherever it is, even on the far
        side of the motion."""
        high = dataclasses.replace(view, floor_height=9.0)
        viewport = make_viewport([high], framing="clip")
        assert viewport.hi[1] > 9.0
        assert viewport.lo[1] < float(view.coords[..., 1].min())

    def test_a_negative_up_axis_frames_the_same_motion(self, view):
        negative = _negative_up(view)
        viewport = make_viewport([negative], framing="clip")
        mirrored = make_viewport([view], framing="clip")
        np.testing.assert_allclose(viewport.lo[[0, 2]], mirrored.lo[[0, 2]])
        np.testing.assert_allclose(viewport.hi[[0, 2]], mirrored.hi[[0, 2]])
        np.testing.assert_allclose(viewport.lo[1], -mirrored.hi[1])
        np.testing.assert_allclose(viewport.hi[1], -mirrored.lo[1])

    def test_a_rotating_camera_squares_the_ground_off(self, view):
        viewport = make_viewport([view], framing="clip", motion="turntable")
        first, second = viewport.ground_axes
        spans = viewport.hi - viewport.lo
        assert spans[first] == pytest.approx(spans[second])

    def test_the_cube_is_still_the_size_scale(self, view):
        viewport = make_viewport([view], framing="clip")
        center, half_span = compute_unified_limits([view.coords])
        np.testing.assert_array_equal(viewport.center, center)
        assert viewport.half_span == half_span


class TestSchedule:
    def test_a_fixed_camera_has_no_schedule(self, view):
        viewport = make_viewport([view], motion="fixed")
        assert viewport.azimuths is None
        assert not viewport.rotating
        assert viewport.azimuth_at(7) == view.azimuth

    def test_turntable_orbits_once(self, view):
        viewport = make_viewport([view], motion="turntable")
        np.testing.assert_array_equal(
            viewport.azimuths, turntable_azimuths(view.azimuth, 24))
        assert viewport.rotating
        assert viewport.azimuth_at(6) == pytest.approx(view.azimuth + 90.0)

    def test_follow_tracks_the_facing(self, view):
        turning = _turning_view(view)
        viewport = make_viewport([turning], motion="follow")
        np.testing.assert_array_equal(
            viewport.azimuths,
            compute_follow_azimuths(turning, turning.azimuth))
        assert viewport.rotating

    def test_a_schedule_that_never_changes_is_a_fixed_camera(self, view):
        """Follow on a rig with no left/right pairs has nothing to
        track; a turntable over one frame has nowhere to go."""
        no_pairs = dataclasses.replace(
            view, lr_pairs=np.empty((0, 2), dtype=np.intp))
        assert make_viewport([no_pairs], motion="follow").azimuths is None
        one_frame = dataclasses.replace(
            view, coords=view.coords[:1], root_heading=view.root_heading[:1])
        viewport = make_viewport([one_frame], motion="turntable")
        assert viewport.azimuths is None
        assert not viewport.rotating

    def test_a_character_that_never_turns_is_a_fixed_camera_too(self, view):
        """The rule is about the schedule, not about why it is flat:
        this rig has its pairs, and glides straight ahead in one pose."""
        assert len(view.lr_pairs) > 0
        travel = view.coords[:, :1] - view.coords[:1, :1]
        gliding = dataclasses.replace(
            view, coords=view.coords[:1] + travel)
        viewport = make_viewport([gliding], framing="clip", motion="follow")
        assert viewport.azimuths is None
        fixed = make_viewport([gliding], framing="clip", motion="fixed")
        np.testing.assert_array_equal(viewport.lo, fixed.lo)

    def test_a_fixed_camera_is_framed_as_one(self, view):
        """With nothing to rotate, the ground is not squared off."""
        no_pairs = dataclasses.replace(
            view, lr_pairs=np.empty((0, 2), dtype=np.intp))
        followed = make_viewport([no_pairs], framing="clip", motion="follow")
        fixed = make_viewport([no_pairs], framing="clip", motion="fixed")
        np.testing.assert_array_equal(followed.lo, fixed.lo)
        np.testing.assert_array_equal(followed.hi, fixed.hi)


def _turning_view(view):
    """The walker, turning a quarter turn about the up axis as it goes."""
    num_frames = view.coords.shape[0]
    angles = np.linspace(0.0, np.pi / 2, num_frames)
    cos, sin = np.cos(angles), np.sin(angles)
    root = view.coords[:, :1, :]
    local = view.coords - root
    turned = local.copy()
    turned[..., 0] = cos[:, None] * local[..., 0] + sin[:, None] * local[..., 2]
    turned[..., 2] = -sin[:, None] * local[..., 0] + cos[:, None] * local[..., 2]
    return dataclasses.replace(view, coords=turned + root)


class TestCamera:
    def test_view_matrix_is_the_schedules(self, view):
        viewport = make_viewport([view], motion="turntable")
        for frame in (0, 5, 23):
            np.testing.assert_array_equal(
                viewport.view_matrix(frame),
                build_view_matrix(viewport.azimuth_at(frame),
                                  view.elevation, view.up_axis))

    def test_a_fixed_camera_has_one_matrix_for_every_frame(self, view):
        viewport = make_viewport([view])
        np.testing.assert_array_equal(
            viewport.view_matrix(17), viewport.view_matrix(0))

    def test_the_eye_stands_back_from_the_cubes_centre(self, view):
        viewport = make_viewport([view])
        camera = viewport.camera()
        np.testing.assert_array_equal(camera.target, viewport.center)
        offset = camera.eye - camera.target
        assert np.linalg.norm(offset) == pytest.approx(
            EYE_DISTANCE * viewport.half_span)
        matrix = viewport.view_matrix()
        np.testing.assert_allclose(
            offset / np.linalg.norm(offset), matrix[2], atol=1e-12)
        np.testing.assert_array_equal(camera.up, matrix[1])

    def test_the_camera_does_not_depend_on_the_framing(self, view):
        still = make_viewport([view], framing="still").camera()
        clip = make_viewport([view], framing="clip").camera()
        np.testing.assert_array_equal(still.eye, clip.eye)
        np.testing.assert_array_equal(still.target, clip.target)

    def test_the_camera_orbits_with_the_schedule(self, view):
        viewport = make_viewport([view], motion="turntable")
        start, quarter = viewport.camera(0), viewport.camera(6)
        assert not np.allclose(start.eye, quarter.eye)
        np.testing.assert_array_equal(start.target, quarter.target)

    def test_camera_arrays_are_the_callers_to_change(self, view):
        viewport = make_viewport([view])
        viewport.camera().target[:] = 99.0
        viewport.camera().up[:] = 99.0
        assert not np.any(viewport.center == 99.0)
        assert not np.any(viewport.view_matrix() == 99.0)


class TestFloor:
    @pytest.mark.parametrize("framing", ["still", "clip"])
    def test_the_plane_sits_exactly_at_the_ground(self, view, framing):
        viewport = make_viewport([view], framing=framing)
        quad = viewport.floor_quad()
        assert quad.shape == (4, 3)
        np.testing.assert_array_equal(
            quad[:, viewport.up_index], view.floor_height)

    @pytest.mark.parametrize("framing, motion", [
        ("still", "fixed"), ("clip", "fixed"), ("clip", "turntable")])
    def test_a_square_of_the_stated_extent_around_the_centre(
            self, view, framing, motion):
        viewport = make_viewport([view], framing=framing, motion=motion)
        quad = viewport.floor_quad()
        ground = list(viewport.ground_axes)
        sides = quad[2, ground] - quad[0, ground]
        np.testing.assert_allclose(
            sides, 2 * FLOOR_EXTENT * viewport.half_span)
        np.testing.assert_allclose(
            (quad[0, ground] + quad[2, ground]) / 2,
            viewport.center[ground])
        # ... which is where the framing box is centred on the ground
        np.testing.assert_allclose(
            (viewport.lo + viewport.hi)[ground] / 2,
            viewport.center[ground], atol=1e-12)

    @pytest.mark.parametrize("up, first, second", [
        ("+y", 0, 2), ("-y", 0, 2), ("+z", 0, 1), ("-z", 0, 1),
        ("+x", 1, 2), ("-x", 1, 2)])
    def test_corner_order_on_every_up_axis(self, view, up, first, second):
        """(lo, lo), (hi, lo), (hi, hi), (lo, hi) over the ground axes
        in x, y, z order, whatever the up axis and its sign."""
        forward = "+x" if up[1] != "x" else "+y"
        turned = dataclasses.replace(
            view, up=up, forward_axis=forward, floor_height=-0.25)
        viewport = make_viewport([turned], include_floor=False)
        assert viewport.ground_axes == (first, second)
        quad = viewport.floor_quad()
        reach = FLOOR_EXTENT * viewport.half_span
        a_lo = viewport.center[first] - reach
        a_hi = viewport.center[first] + reach
        b_lo = viewport.center[second] - reach
        b_hi = viewport.center[second] + reach
        np.testing.assert_allclose(quad[:, first], [a_lo, a_hi, a_hi, a_lo])
        np.testing.assert_allclose(quad[:, second], [b_lo, b_lo, b_hi, b_hi])
        np.testing.assert_array_equal(quad[:, viewport.up_index], -0.25)

    def test_corners_run_around_the_square(self, view):
        quad = make_viewport([view]).floor_quad()
        edges = np.roll(quad, -1, axis=0) - quad
        lengths = np.linalg.norm(edges, axis=1)
        np.testing.assert_allclose(lengths, lengths[0])
        np.testing.assert_allclose(
            np.linalg.norm(quad[2] - quad[0]), lengths[0] * np.sqrt(2))

    def test_clipped_to_the_framing_box(self, view):
        viewport = make_viewport([view], framing="clip")
        quad = viewport.floor_quad(clip_to_box=True)
        ground = list(viewport.ground_axes)
        np.testing.assert_array_equal(quad[0, ground], viewport.lo[ground])
        np.testing.assert_array_equal(quad[2, ground], viewport.hi[ground])
        np.testing.assert_array_equal(
            quad[:, viewport.up_index], view.floor_height)

    def test_a_negative_up_axis_keeps_the_plane_at_the_ground(self, view):
        negative = _negative_up(view)
        viewport = make_viewport([negative])
        quad = viewport.floor_quad()
        assert np.all(quad[:, 1] == negative.floor_height)
        assert negative.floor_height >= negative.coords[..., 1].max()
        assert viewport.below_floor(1.0) > viewport.floor_height

    def test_ground_path_drops_points_onto_the_plane(self, view):
        viewport = make_viewport([view])
        root = view.coords[:, 0]
        path = viewport.ground_path(root)
        np.testing.assert_array_equal(path[:, 1], view.floor_height)
        np.testing.assert_array_equal(path[:, [0, 2]], root[:, [0, 2]])
        assert path.flags.writeable            # a new array, the caller's
        assert not np.shares_memory(path, root)


class TestEnclosingCube:
    def test_the_smallest_cube_around_the_framing_box(self, view):
        viewport = make_viewport([view], framing="clip")
        lo, hi = viewport.enclosing_cube()
        sides = hi - lo
        np.testing.assert_allclose(sides, sides[0])
        assert sides[0] == pytest.approx(
            float((viewport.hi - viewport.lo).max()))
        np.testing.assert_allclose(
            (lo + hi) / 2, (viewport.lo + viewport.hi) / 2)
        assert np.all(lo <= viewport.lo + 1e-12)
        assert np.all(hi >= viewport.hi - 1e-12)

    def test_a_still_is_its_own_enclosing_cube(self, view):
        viewport = make_viewport([view], framing="still")
        lo, hi = viewport.enclosing_cube()
        np.testing.assert_allclose(lo, viewport.lo)
        np.testing.assert_allclose(hi, viewport.hi)


class TestProjection:
    RESOLUTION = (640, 480)

    def test_the_boxs_centre_lands_mid_panel(self, view):
        viewport = make_viewport([view], framing="clip")
        middle = ((viewport.lo + viewport.hi) / 2)[np.newaxis]
        pixel = viewport.project(middle, self.RESOLUTION)[0]
        assert pixel.tolist() == [320, 240]

    def test_the_box_fits_the_stated_fraction_of_the_panel(self, view):
        from pybvh.bvhplot._viewport import box_corners
        viewport = make_viewport([view], framing="clip")
        pixels = viewport.project(
            box_corners(viewport.lo, viewport.hi), self.RESOLUTION)
        width = pixels[:, 0].max() - pixels[:, 0].min()
        height = pixels[:, 1].max() - pixels[:, 1].min()
        fill = max(width / 640, height / 480)
        assert fill == pytest.approx(FIT_FRACTION, abs=0.01)
        assert width <= 640 * FIT_FRACTION + 1
        assert height <= 480 * FIT_FRACTION + 1

    def test_an_orbit_holds_one_scale(self, view):
        """Two points a fixed distance apart along the up axis project
        to the same pixel distance on every frame of a turntable."""
        viewport = make_viewport([view], framing="clip", motion="turntable")
        middle = (viewport.lo + viewport.hi) / 2
        pair = np.stack([middle, middle + viewport.up_vector])
        lengths = []
        for frame in range(24):
            pixels = viewport.project(pair, (1920, 1080), frame)
            lengths.append(np.linalg.norm(pixels[1] - pixels[0]))
        assert max(lengths) - min(lengths) <= 2.0      # integer pixels

    def test_the_whole_orbit_stays_in_the_panel(self, view):
        viewport = make_viewport([view], framing="clip", motion="turntable")
        for frame in range(24):
            pixels = viewport.project(view.coords[frame], self.RESOLUTION, frame)
            assert pixels[:, 0].min() >= 0 and pixels[:, 0].max() < 640
            assert pixels[:, 1].min() >= 0 and pixels[:, 1].max() < 480


class TestSeveralViews:
    """k3d and vedo draw every skeleton into one scene: one viewport."""

    def test_the_cube_covers_every_view(self):
        scene = make_array_scene(n_frames=12, n_skeletons=3).spread(2.0)
        viewport = make_viewport(scene.views)
        center, half_span = compute_unified_limits(
            [v.coords for v in scene.views])
        np.testing.assert_array_equal(viewport.center, center)
        assert viewport.half_span == half_span
        for v in scene.views:
            assert np.all(v.coords >= viewport.lo - 1e-9)
            assert np.all(v.coords <= viewport.hi + 1e-9)

    def test_the_floor_is_the_lowest_ground(self):
        views = make_array_scene(n_frames=12, n_skeletons=2).views
        views = [dataclasses.replace(views[0], floor_height=0.3),
                 dataclasses.replace(views[1], floor_height=-0.2)]
        assert make_viewport(views).floor_height == -0.2

    def test_the_lowest_ground_of_a_negative_up_axis_is_the_maximum(self):
        views = [_negative_up(v) for v in
                 make_array_scene(n_frames=12, n_skeletons=2).views]
        views = [dataclasses.replace(views[0], floor_height=0.3),
                 dataclasses.replace(views[1], floor_height=-0.2)]
        assert make_viewport(views).floor_height == 0.3

    def test_the_camera_is_the_first_views(self):
        views = make_array_scene(n_frames=12, n_skeletons=2).views
        views = [dataclasses.replace(views[0], azimuth=33.0, elevation=12.0),
                 dataclasses.replace(views[1], azimuth=-70.0, elevation=5.0)]
        viewport = make_viewport(views)
        assert (viewport.azimuth, viewport.elevation) == (33.0, 12.0)

    def test_the_clip_box_covers_every_view(self):
        scene = make_array_scene(n_frames=12, n_skeletons=2).spread(2.0)
        viewport = make_viewport(scene.views, framing="clip")
        for v in scene.views:
            assert np.all(v.coords >= viewport.lo)
            assert np.all(v.coords <= viewport.hi)

    def test_panel_viewports_frames_each_view_on_its_own(self):
        scene = make_array_scene(n_frames=12, n_skeletons=2).spread(2.0)
        viewports = panel_viewports(scene.views, framing="clip")
        assert len(viewports) == 2
        for viewport, v in zip(viewports, scene.views):
            alone = make_viewport([v], framing="clip")
            np.testing.assert_array_equal(viewport.lo, alone.lo)
            np.testing.assert_array_equal(viewport.hi, alone.hi)


class TestViewportValue:
    def test_is_frozen(self, view):
        viewport = make_viewport([view])
        with pytest.raises(dataclasses.FrozenInstanceError):
            viewport.half_span = 1.0  # type: ignore[misc]

    def test_carries_the_up_axis_and_projection(self, view):
        viewport = make_viewport([view], projection="ortho")
        assert isinstance(viewport, Viewport)
        assert viewport.up == "+y"
        assert viewport.up_axis == "y"
        assert viewport.ground_axes == (0, 2)
        assert viewport.projection == "ortho"

    @pytest.mark.parametrize("options, message", [
        (dict(framing="cube"), "Unknown framing"),
        (dict(motion="orbit"), "Unknown motion"),
    ])
    def test_rejects_unknown_options(self, view, options, message):
        with pytest.raises(ValueError, match=message):
            make_viewport([view], **options)

    def test_needs_a_view(self):
        with pytest.raises(ValueError, match="at least one view"):
            make_viewport([])
