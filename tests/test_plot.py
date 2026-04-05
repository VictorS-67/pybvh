"""Tests for the pybvh.bvhplot visualization module."""
from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from pybvh import read_bvh_file, bvhplot
from pybvh.bvhplot._common import (
    get_skeleton_lines,
    normalize_input,
    compute_unified_limits,
    get_camera_angles,
    build_view_matrix,
    ortho_project,
    align_frame_counts,
)

BVH_DIR = Path(__file__).parent.parent / "bvh_data"


@pytest.fixture
def bvh_example():
    return read_bvh_file(BVH_DIR / "bvh_example.bvh")


@pytest.fixture
def bvh_test1():
    return read_bvh_file(BVH_DIR / "bvh_test1.bvh")


# ===================================================================
# _common.py tests
# ===================================================================


class TestGetSkeletonLines:
    def test_returns_correct_count(self, bvh_example):
        lines = get_skeleton_lines(bvh_example)
        # One bone per non-root node
        assert len(lines) == len(bvh_example.nodes) - 1

    def test_parent_child_indices_valid(self, bvh_example):
        lines = get_skeleton_lines(bvh_example)
        n_nodes = len(bvh_example.nodes)
        for p_idx, c_idx in lines:
            assert 0 <= p_idx < n_nodes
            assert 0 <= c_idx < n_nodes
            assert p_idx != c_idx

    def test_all_children_represented(self, bvh_example):
        lines = get_skeleton_lines(bvh_example)
        child_indices = {c for _, c in lines}
        # Every non-root node should appear as a child
        assert len(child_indices) == len(bvh_example.nodes) - 1
        # Root (index 0) should not be a child
        assert 0 not in child_indices


class TestNormalizeInput:
    def test_single_bvh_all_frames(self, bvh_example):
        bvh_list, coords_list = normalize_input(bvh_example, None, "world")
        assert len(bvh_list) == 1
        assert coords_list[0].ndim == 3
        assert coords_list[0].shape[0] == bvh_example.frame_count
        assert coords_list[0].shape[1] == len(bvh_example.nodes)

    def test_single_bvh_one_frame(self, bvh_example):
        bvh_list, coords_list = normalize_input(bvh_example, 0, "world")
        assert coords_list[0].shape == (1, len(bvh_example.nodes), 3)

    def test_list_of_bvh(self, bvh_example):
        bvh_list, coords_list = normalize_input(
            [bvh_example, bvh_example], None, "world")
        assert len(bvh_list) == 2
        assert len(coords_list) == 2

    def test_precomputed_array_2d(self, bvh_example):
        coords = bvh_example.get_spatial_coord(frame_num=0)
        _, coords_list = normalize_input(bvh_example, coords, "world")
        assert coords_list[0].shape == (1, len(bvh_example.nodes), 3)

    def test_precomputed_array_3d(self, bvh_example):
        coords = bvh_example.get_spatial_coord()
        _, coords_list = normalize_input(bvh_example, coords, "world")
        assert coords_list[0].shape == coords.shape

    def test_precomputed_array_with_list_raises(self, bvh_example):
        coords = bvh_example.get_spatial_coord()
        with pytest.raises(ValueError, match="single Bvh"):
            normalize_input([bvh_example, bvh_example], coords, "world")

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="At least one"):
            normalize_input([], None, "world")


class TestComputeUnifiedLimits:
    def test_returns_center_and_span(self, bvh_example):
        coords = bvh_example.get_spatial_coord()
        center, half_span = compute_unified_limits([coords])
        assert center.shape == (3,)
        assert half_span > 0

    def test_multi_skeleton_encompasses_all(self, bvh_example):
        coords = bvh_example.get_spatial_coord()
        # Offset a copy
        coords2 = coords.copy()
        coords2[:, :, 0] += 100.0
        center, half_span = compute_unified_limits([coords, coords2])
        # Center should be roughly between the two
        assert center[0] > coords[:, :, 0].mean()
        assert center[0] < coords2[:, :, 0].mean()

    def test_equal_aspect_ratio(self, bvh_example):
        coords = bvh_example.get_spatial_coord()
        center, half_span = compute_unified_limits([coords])
        # half_span is a scalar (cubic bounding box)
        assert isinstance(half_span, float)


class TestAlignFrameCounts:
    def test_single_item_unchanged(self):
        coords = [np.zeros((10, 5, 3))]
        result = align_frame_counts(coords)
        assert result[0].shape[0] == 10

    def test_truncates_to_shortest(self):
        c1 = np.zeros((100, 5, 3))
        c2 = np.zeros((50, 5, 3))
        c3 = np.zeros((75, 5, 3))
        result = align_frame_counts([c1, c2, c3])
        assert all(c.shape[0] == 50 for c in result)

    def test_equal_lengths_unchanged(self):
        c1 = np.ones((20, 5, 3))
        c2 = np.ones((20, 5, 3)) * 2
        result = align_frame_counts([c1, c2])
        assert result[0].shape[0] == 20
        assert result[1][0, 0, 0] == 2.0  # data preserved


class TestGetCameraAngles:
    def test_front_returns_tuple(self, bvh_example):
        frame = bvh_example.get_spatial_coord(frame_num=0)
        azim, elev, up = get_camera_angles(bvh_example, frame, "front")
        assert isinstance(azim, float)
        assert isinstance(elev, float)
        assert up in ('x', 'y', 'z')

    def test_side_differs_from_front(self, bvh_example):
        frame = bvh_example.get_spatial_coord(frame_num=0)
        azim_f, _, _ = get_camera_angles(bvh_example, frame, "front")
        azim_s, _, _ = get_camera_angles(bvh_example, frame, "side")
        assert abs(azim_s - azim_f) == pytest.approx(90.0)

    def test_top_has_high_elevation(self, bvh_example):
        frame = bvh_example.get_spatial_coord(frame_num=0)
        _, elev, _ = get_camera_angles(bvh_example, frame, "top")
        assert elev == pytest.approx(90.0)

    def test_custom_tuple(self, bvh_example):
        frame = bvh_example.get_spatial_coord(frame_num=0)
        azim, elev, _ = get_camera_angles(bvh_example, frame, (45.0, 30.0))
        assert azim == pytest.approx(45.0)
        assert elev == pytest.approx(30.0)

    def test_unknown_preset_raises(self, bvh_example):
        frame = bvh_example.get_spatial_coord(frame_num=0)
        with pytest.raises(ValueError, match="Unknown camera"):
            get_camera_angles(bvh_example, frame, "below")


class TestOrthoProject:
    def test_output_shape(self):
        coords = np.array([[0, 0, 0], [1, 1, 1], [2, 0, 0]], dtype=np.float64)
        view = build_view_matrix(0, 0, 'y')
        center = np.array([1.0, 0.5, 0.5])
        pixels = ortho_project(coords, view, center, 2.0, (640, 480))
        assert pixels.shape == (3, 2)
        assert pixels.dtype == np.int32

    def test_center_projects_to_image_center(self):
        center = np.array([5.0, 5.0, 5.0])
        coords = center.reshape(1, 3)
        view = build_view_matrix(0, 0, 'y')
        pixels = ortho_project(coords, view, center, 2.0, (640, 480))
        assert abs(pixels[0, 0] - 320) <= 1
        assert abs(pixels[0, 1] - 240) <= 1

    def test_different_resolutions(self):
        coords = np.zeros((1, 3), dtype=np.float64)
        view = build_view_matrix(0, 0, 'y')
        center = np.zeros(3)
        p1 = ortho_project(coords, view, center, 1.0, (100, 100))
        p2 = ortho_project(coords, view, center, 1.0, (200, 200))
        # Center point should be at the center of each resolution
        assert abs(p1[0, 0] - 50) <= 1
        assert abs(p2[0, 0] - 100) <= 1


class TestBuildViewMatrix:
    def test_identity_like_at_zero(self):
        view = build_view_matrix(0, 0, 'y')
        assert view.shape == (3, 3)
        # Should be close to identity (Y-up, no rotation)
        assert np.allclose(view, np.eye(3), atol=1e-10)

    def test_orthogonal(self):
        for azim, elev in [(30, 20), (90, 45), (-45, 60)]:
            view = build_view_matrix(azim, elev, 'y')
            # Columns should be orthonormal
            assert np.allclose(view @ view.T, np.eye(3), atol=1e-10)

    def test_different_up_axes(self):
        for up in ('x', 'y', 'z'):
            view = build_view_matrix(0, 0, up)
            assert view.shape == (3, 3)
            assert np.allclose(view @ view.T, np.eye(3), atol=1e-10)

    def test_up_axis_points_up_on_screen(self):
        """Row 1 (view-up) should have its largest component along the up axis."""
        for up, idx in [('x', 0), ('y', 1), ('z', 2)]:
            view = build_view_matrix(0, 20, up)
            # Row 1 = up direction. The up_axis component should be the largest.
            assert abs(view[1, idx]) == max(abs(view[1, :]))

    def test_matches_matplotlib_right_direction(self, bvh_example):
        """OpenCV 'right' direction should match matplotlib for all up-axes."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import proj3d

        for up in ('y', 'z'):
            for azim in (0, 45, 90, 180):
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.view_init(elev=20, azim=azim, vertical_axis=up)
                fig.canvas.draw()

                # Matplotlib right direction: project unit vectors, take screen-x
                mpl_right = np.zeros(3)
                origin = np.array(
                    proj3d.proj_transform(0, 0, 0, ax.get_proj()))
                for i in range(3):
                    v = np.zeros(3); v[i] = 1.0
                    p = np.array(
                        proj3d.proj_transform(*v, ax.get_proj()))
                    mpl_right[i] = p[0] - origin[0]
                plt.close()

                # OpenCV right direction: row 0 of view matrix
                vm = build_view_matrix(azim, 20, up)
                cv_right = vm[0, :]

                mpl_right /= np.linalg.norm(mpl_right)
                cv_right /= np.linalg.norm(cv_right)
                dot = np.dot(mpl_right, cv_right)
                assert dot > 0.99, (
                    f"Right direction mismatch: up={up}, azim={azim}, "
                    f"dot={dot:.4f}")


class TestFrontViewSemantics:
    """Verify that camera='front' shows the skeleton's chest/face."""

    def test_front_view_toes_toward_viewer(self, bvh_example):
        """In front view, the forward axis should point toward the viewer
        (positive w component in view space)."""
        from pybvh.tools import get_forw_up_axis
        frame = bvh_example.get_spatial_coord(frame_num=0)
        dirs = get_forw_up_axis(bvh_example, frame)
        azim, elev, up = get_camera_angles(bvh_example, frame, "front")

        vm = build_view_matrix(azim, elev, up)
        fwd_vec = np.zeros(3)
        fwd_idx = {'x': 0, 'y': 1, 'z': 2}[dirs['forward'][1]]
        fwd_sign = 1.0 if dirs['forward'][0] == '+' else -1.0
        fwd_vec[fwd_idx] = fwd_sign

        # Row 2 (w) points toward viewer. Positive w = toward viewer.
        fwd_w = (vm @ fwd_vec)[2]
        assert fwd_w > 0, (
            f"Forward axis should point toward viewer (w>0) in front view, "
            f"got w={fwd_w:.3f}")

    def test_front_view_right_hand_rule(self, bvh_example):
        """The view matrix should preserve right-handedness: det > 0."""
        frame = bvh_example.get_spatial_coord(frame_num=0)
        azim, elev, up = get_camera_angles(bvh_example, frame, "front")
        vm = build_view_matrix(azim, elev, up)
        assert np.linalg.det(vm) > 0, (
            f"View matrix should be right-handed (det>0), "
            f"got det={np.linalg.det(vm):.3f}")

    def test_side_view_perpendicular_to_front(self, bvh_example):
        """Side view should look 90 degrees from front along the forward axis."""
        from pybvh.tools import get_forw_up_axis
        frame = bvh_example.get_spatial_coord(frame_num=0)
        dirs = get_forw_up_axis(bvh_example, frame)
        azim_f, elev, up = get_camera_angles(bvh_example, frame, "front")
        azim_s, _, _ = get_camera_angles(bvh_example, frame, "side")

        vm_f = build_view_matrix(azim_f, elev, up)
        vm_s = build_view_matrix(azim_s, elev, up)

        # Forward axis: in front view mostly depth (w), in side view mostly
        # screen-right or screen-left (u).
        fwd_vec = np.zeros(3)
        fwd_idx = {'x': 0, 'y': 1, 'z': 2}[dirs['forward'][1]]
        fwd_vec[fwd_idx] = 1.0

        fwd_in_front = vm_f @ fwd_vec
        fwd_in_side = vm_s @ fwd_vec
        # In front view, forward is mostly in w (depth)
        assert abs(fwd_in_front[2]) > abs(fwd_in_front[0]), (
            "Forward should be mostly depth in front view")
        # In side view, forward is mostly in u (screen horizontal)
        assert abs(fwd_in_side[0]) > abs(fwd_in_side[2]), (
            "Forward should be mostly horizontal in side view")

    def test_backends_agree_on_front(self, bvh_example):
        """Matplotlib and OpenCV should show the same side of the skeleton."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import proj3d

        rest = bvh_example.get_rest_pose(mode='coordinates')
        azim, elev, up = get_camera_angles(bvh_example, rest, "front")
        idx = bvh_example.node_index

        # Find a left/right pair with different positions
        lp = rp = None
        for n in bvh_example.nodes:
            if not n.is_end_site() and 'Left' in n.name:
                rn = n.name.replace('Left', 'Right')
                if rn in idx:
                    l_pos = rest[idx[n.name]]
                    r_pos = rest[idx[rn]]
                    if np.linalg.norm(l_pos - r_pos) > 1.0:
                        lp, rp = n.name, rn
                        break

        assert lp is not None, "Need a left/right pair for this test"

        # Matplotlib: check screen-x order
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=elev, azim=azim, vertical_axis=up)
        fig.canvas.draw()
        lx_mpl, _, _ = proj3d.proj_transform(*rest[idx[lp]], ax.get_proj())
        rx_mpl, _, _ = proj3d.proj_transform(*rest[idx[rp]], ax.get_proj())
        plt.close()
        mpl_left_is_left = lx_mpl < rx_mpl

        # OpenCV: check screen-x order via view matrix
        vm = build_view_matrix(azim, elev, up)
        lx_cv = (vm @ rest[idx[lp]])[0]
        rx_cv = (vm @ rest[idx[rp]])[0]
        cv_left_is_left = lx_cv < rx_cv

        assert mpl_left_is_left == cv_left_is_left, (
            f"Backends disagree on left/right: "
            f"mpl Left<Right={mpl_left_is_left}, "
            f"cv Left<Right={cv_left_is_left}")


# ===================================================================
# Public API tests (matplotlib backend)
# ===================================================================


class TestFrame:
    def test_single_frame_returns_fig_ax(self, bvh_example):
        import matplotlib
        matplotlib.use('Agg')  # non-interactive for CI
        fig, ax = bvhplot.frame(bvh_example, 0, show=False)
        assert fig is not None
        assert ax is not None

    def test_from_array(self, bvh_example):
        import matplotlib
        matplotlib.use('Agg')
        coords = bvh_example.get_spatial_coord(frame_num=0)
        fig, ax = bvhplot.frame(bvh_example, coords, show=False)
        assert fig is not None

    def test_side_by_side_returns_list(self, bvh_example):
        import matplotlib
        matplotlib.use('Agg')
        fig, axs = bvhplot.frame(
            [bvh_example, bvh_example], 0,
            labels=["A", "B"], show=False)
        assert isinstance(axs, list)
        assert len(axs) == 2

    def test_centered_modes(self, bvh_example):
        import matplotlib
        matplotlib.use('Agg')
        for mode in ("world", "skeleton", "first"):
            fig, ax = bvhplot.frame(bvh_example, 0, centered=mode, show=False)
            assert fig is not None


class TestTrajectory:
    def test_returns_fig_ax(self, bvh_example):
        import matplotlib
        matplotlib.use('Agg')
        fig, ax = bvhplot.trajectory(bvh_example, show=False)
        assert fig is not None
        assert ax is not None

    def test_multi_skeleton(self, bvh_example):
        import matplotlib
        matplotlib.use('Agg')
        fig, ax = bvhplot.trajectory(
            [bvh_example, bvh_example],
            labels=["A", "B"], show=False)
        assert ax.get_legend() is not None


class TestRenderMatplotlib:
    def test_render_creates_file(self, bvh_example, tmp_path):
        import matplotlib
        matplotlib.use('Agg')
        # Use only first 5 frames for speed
        bvh_short = bvh_example.slice_frames(0, 5)
        path = bvhplot.render(
            bvh_short, tmp_path / "test.gif",
            backend="matplotlib")
        assert path.exists()
        assert path.stat().st_size > 0

    def test_render_html(self, bvh_example, tmp_path):
        import matplotlib
        matplotlib.use('Agg')
        bvh_short = bvh_example.slice_frames(0, 3)
        path = bvhplot.render(
            bvh_short, tmp_path / "test.html",
            backend="matplotlib")
        assert path.exists()
        assert path.stat().st_size > 0


# ===================================================================
# OpenCV backend tests
# ===================================================================


class TestRenderOpenCV:
    @pytest.fixture(autouse=True)
    def _skip_if_no_cv2(self):
        pytest.importorskip("cv2")

    def test_creates_file(self, bvh_example, tmp_path):
        path = bvhplot.render(
            bvh_example, tmp_path / "out.mp4", backend="opencv",
            resolution=(320, 240))
        assert path.exists()
        assert path.stat().st_size > 0

    def test_frame_count(self, bvh_example, tmp_path):
        import cv2
        path = bvhplot.render(
            bvh_example, tmp_path / "out.mp4", backend="opencv",
            resolution=(320, 240))
        cap = cv2.VideoCapture(str(path))
        fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        assert fc == bvh_example.frame_count

    def test_resolution(self, bvh_example, tmp_path):
        import cv2
        path = bvhplot.render(
            bvh_example, tmp_path / "out.mp4", backend="opencv",
            resolution=(640, 480))
        cap = cv2.VideoCapture(str(path))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        assert (w, h) == (640, 480)

    def test_side_by_side(self, bvh_example, tmp_path):
        path = bvhplot.render(
            [bvh_example, bvh_example], tmp_path / "cmp.mp4",
            backend="opencv", resolution=(640, 240),
            labels=["A", "B"])
        assert path.exists()

    def test_camera_presets(self, bvh_example, tmp_path):
        for cam in ["front", "side", "top", (45, 30)]:
            path = bvhplot.render(
                bvh_example, tmp_path / "cam.mp4",
                backend="opencv", resolution=(320, 240),
                camera=cam)
            assert path.exists()

    def test_gif_output(self, bvh_example, tmp_path):
        bvh_short = bvh_example.slice_frames(0, 5)
        path = bvhplot.render(
            bvh_short, tmp_path / "out.gif",
            backend="opencv", resolution=(320, 240))
        assert path.exists()
        assert path.suffix == '.gif'
        assert path.stat().st_size > 0

    def test_show_axis(self, bvh_example, tmp_path):
        path = bvhplot.render(
            bvh_example, tmp_path / "axis.mp4",
            backend="opencv", resolution=(320, 240),
            show_axis=True)
        assert path.exists()

    def test_auto_backend_selects_opencv(self, bvh_example, tmp_path):
        """When cv2 is available, auto backend should select opencv."""
        path = bvhplot.render(
            bvh_example, tmp_path / "auto.mp4",
            backend="auto", resolution=(320, 240))
        assert path.exists()


# ===================================================================
# get_forw_up_axis re-export test
# ===================================================================


def test_get_forw_up_axis_accessible_from_plot():
    """Ensure get_forw_up_axis is accessible via pybvh.bvhplot."""
    from pybvh.tools import get_forw_up_axis as original
    assert bvhplot.get_forw_up_axis is original
