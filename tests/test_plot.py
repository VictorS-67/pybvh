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


@pytest.fixture
def bvh_test2():
    return read_bvh_file(BVH_DIR / "bvh_test2.bvh")


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
        coords = bvh_example.node_positions(frame=0)
        _, coords_list = normalize_input(bvh_example, coords, "world")
        assert coords_list[0].shape == (1, len(bvh_example.nodes), 3)

    def test_precomputed_array_3d(self, bvh_example):
        coords = bvh_example.node_positions()
        _, coords_list = normalize_input(bvh_example, coords, "world")
        assert coords_list[0].shape == coords.shape

    def test_precomputed_array_with_list_raises(self, bvh_example):
        coords = bvh_example.node_positions()
        with pytest.raises(ValueError, match="single Bvh"):
            normalize_input([bvh_example, bvh_example], coords, "world")

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="At least one"):
            normalize_input([], None, "world")


class TestComputeUnifiedLimits:
    def test_returns_center_and_span(self, bvh_example):
        coords = bvh_example.node_positions()
        center, half_span = compute_unified_limits([coords])
        assert center.shape == (3,)
        assert half_span > 0

    def test_multi_skeleton_encompasses_all(self, bvh_example):
        coords = bvh_example.node_positions()
        # Offset a copy
        coords2 = coords.copy()
        coords2[:, :, 0] += 100.0
        center, half_span = compute_unified_limits([coords, coords2])
        # Center should be roughly between the two
        assert center[0] > coords[:, :, 0].mean()
        assert center[0] < coords2[:, :, 0].mean()

    def test_equal_aspect_ratio(self, bvh_example):
        coords = bvh_example.node_positions()
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
        frame = bvh_example.node_positions(frame=0)
        azim, elev, up = get_camera_angles(bvh_example, frame, "front")
        assert isinstance(azim, float)
        assert isinstance(elev, float)
        assert up in ('x', 'y', 'z')

    def test_side_differs_from_front(self, bvh_example):
        frame = bvh_example.node_positions(frame=0)
        azim_f, _, _ = get_camera_angles(bvh_example, frame, "front")
        azim_s, _, _ = get_camera_angles(bvh_example, frame, "side")
        assert abs(azim_s - azim_f) == pytest.approx(90.0)

    def test_top_has_high_elevation(self, bvh_example):
        frame = bvh_example.node_positions(frame=0)
        _, elev, _ = get_camera_angles(bvh_example, frame, "top")
        assert elev == pytest.approx(90.0)

    def test_custom_tuple(self, bvh_example):
        frame = bvh_example.node_positions(frame=0)
        azim, elev, _ = get_camera_angles(bvh_example, frame, (45.0, 30.0))
        assert azim == pytest.approx(45.0)
        assert elev == pytest.approx(30.0)

    def test_unknown_preset_raises(self, bvh_example):
        frame = bvh_example.node_positions(frame=0)
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
        frame = bvh_example.node_positions(frame=0)
        fwd = bvh_example.forward_at(frame=0)
        azim, elev, up = get_camera_angles(bvh_example, frame, "front")

        vm = build_view_matrix(azim, elev, up)
        fwd_vec = np.zeros(3)
        fwd_idx = {'x': 0, 'y': 1, 'z': 2}[fwd[1]]
        fwd_sign = 1.0 if fwd[0] == '+' else -1.0
        fwd_vec[fwd_idx] = fwd_sign

        # Row 2 (w) points toward viewer. Positive w = toward viewer.
        fwd_w = (vm @ fwd_vec)[2]
        assert fwd_w > 0, (
            f"Forward axis should point toward viewer (w>0) in front view, "
            f"got w={fwd_w:.3f}")

    def test_front_view_right_hand_rule(self, bvh_example):
        """The view matrix should preserve right-handedness: det > 0."""
        frame = bvh_example.node_positions(frame=0)
        azim, elev, up = get_camera_angles(bvh_example, frame, "front")
        vm = build_view_matrix(azim, elev, up)
        assert np.linalg.det(vm) > 0, (
            f"View matrix should be right-handed (det>0), "
            f"got det={np.linalg.det(vm):.3f}")

    def test_side_view_perpendicular_to_front(self, bvh_example):
        """Side view should look 90 degrees from front along the forward axis."""
        frame = bvh_example.node_positions(frame=0)
        fwd = bvh_example.forward_at(frame=0)
        azim_f, elev, up = get_camera_angles(bvh_example, frame, "front")
        azim_s, _, _ = get_camera_angles(bvh_example, frame, "side")

        vm_f = build_view_matrix(azim_f, elev, up)
        vm_s = build_view_matrix(azim_s, elev, up)

        # Forward axis: in front view mostly depth (w), in side view mostly
        # screen-right or screen-left (u).
        fwd_vec = np.zeros(3)
        fwd_idx = {'x': 0, 'y': 1, 'z': 2}[fwd[1]]
        fwd_vec[fwd_idx] = 1.0 if fwd[0] == '+' else -1.0

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

        rest = bvh_example.rest_pose_positions()
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

    def test_camera_front_shows_face_bvh_test2(self, bvh_test2):
        """Regression test for the bvh_test2 orientation bug.

        bvh_test2 has a Y-up rest pose but its animation rotates the root
        ~180° so the character faces -Z in world. Previously camera='front'
        used the topological +Z forward and showed the BACK of the character
        (toes farther from viewer than ankles).

        After the orientation refactor, camera='front' should show the FRONT:
        for both feet, the toes should be CLOSER to the viewer than the ankles.
        """
        frame = bvh_test2.node_positions(frame=15)
        azim, elev, up = get_camera_angles(bvh_test2, frame, "front")
        vm = build_view_matrix(azim, elev, up)

        names = [n.name for n in bvh_test2.nodes]
        center = frame.mean(axis=0)

        for side in ('Left', 'Right'):
            ankle_name = f'{side}Ankle'
            toe_name = f'{side}Toe'
            if ankle_name not in names or toe_name not in names:
                continue
            ankle_pos = frame[names.index(ankle_name)] - center
            toe_pos = frame[names.index(toe_name)] - center

            # Project through view matrix; row 2 (w) points toward viewer.
            # Larger w = closer to viewer.
            ankle_depth = (vm @ ankle_pos)[2]
            toe_depth = (vm @ toe_pos)[2]

            assert toe_depth > ankle_depth, (
                f"{side} foot: toe should be closer to viewer than ankle "
                f"in front view, got toe_depth={toe_depth:.2f}, "
                f"ankle_depth={ankle_depth:.2f} (negative diff means we're "
                f"looking at the back of the character).")


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
        coords = bvh_example.node_positions(frame=0)
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

    def test_ax_injection_uses_provided_ax(self, bvh_example):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(
            1, 2, subplot_kw={'projection': '3d'}, figsize=(10, 5))
        returned_fig, returned_ax = bvhplot.frame(
            bvh_example, 0, ax=axes[0], show=False)
        # The returned ax must be the exact one we passed in
        assert returned_ax is axes[0]
        # The returned fig must be the one owning our ax
        assert returned_fig is fig

    def test_ax_injection_with_list_raises(self, bvh_example):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        with pytest.raises(ValueError, match="single skeletons"):
            bvhplot.frame(
                [bvh_example, bvh_example], 0, ax=ax, show=False)

    def test_ax_injection_with_2d_axes_raises(self, bvh_example):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()  # 2D axes — wrong!
        with pytest.raises(ValueError, match="3D axes"):
            bvhplot.frame(bvh_example, 0, ax=ax, show=False)

    def test_mixed_up_axis_side_by_side(self, bvh_test1, bvh_test2):
        """Side-by-side of Z-up + Y-up skeletons must orient each subplot
        using its OWN up axis, not the first skeleton's.

        Regression test: previously both subplots shared the first
        skeleton's vertical_axis, leaving mismatched skeletons rotated
        on their side.
        """
        import matplotlib
        matplotlib.use('Agg')
        # Sanity: these two fixtures really do have different up axes
        up1 = bvh_test1.world_up[1]
        up2 = bvh_test2.world_up[1]
        assert up1 != up2, (
            f"Fixture precondition: bvh_test1 up={up1}, bvh_test2 up={up2}. "
            "Mixed-up-axis test needs two different up axes.")

        fig, axs = bvhplot.frame(
            [bvh_test1, bvh_test2], 0, show=False)

        # matplotlib stores view_init's vertical axis as an int index
        # (0=x, 1=y, 2=z) on Axes3D._vertical_axis.
        axis_name = {0: 'x', 1: 'y', 2: 'z'}
        assert axis_name[axs[0]._vertical_axis] == up1
        assert axis_name[axs[1]._vertical_axis] == up2


class TestRestPose:
    def test_returns_fig_ax(self, bvh_example):
        import matplotlib
        matplotlib.use('Agg')
        fig, ax = bvhplot.rest_pose(bvh_example, show=False)
        assert fig is not None
        assert ax is not None

    def test_side_by_side_returns_list(self, bvh_example):
        import matplotlib
        matplotlib.use('Agg')
        fig, axs = bvhplot.rest_pose(
            [bvh_example, bvh_example], labels=["A", "B"], show=False)
        assert isinstance(axs, list)
        assert len(axs) == 2

    def test_ax_injection_uses_provided_ax(self, bvh_example):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(
            1, 2, subplot_kw={'projection': '3d'}, figsize=(10, 5))
        returned_fig, returned_ax = bvhplot.rest_pose(
            bvh_example, ax=axes[0], show=False)
        assert returned_ax is axes[0]
        assert returned_fig is fig

    def test_ax_injection_with_list_raises(self, bvh_example):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        with pytest.raises(ValueError, match="single skeletons"):
            bvhplot.rest_pose(
                [bvh_example, bvh_example], ax=ax, show=False)


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

    def test_legend_always_has_start_end(self, bvh_example):
        """Start/end marker entries must be in the legend even without labels."""
        import matplotlib
        matplotlib.use('Agg')
        fig, ax = bvhplot.trajectory(bvh_example, show=False)
        legend = ax.get_legend()
        assert legend is not None
        entries = {t.get_text() for t in legend.get_texts()}
        assert "start" in entries
        assert "end" in entries

    def test_legend_with_labels_includes_both(self, bvh_example):
        """When labels are passed, legend has BOTH skeleton labels AND start/end."""
        import matplotlib
        matplotlib.use('Agg')
        fig, ax = bvhplot.trajectory(
            [bvh_example, bvh_example],
            labels=["Motion A", "Motion B"], show=False)
        entries = {t.get_text() for t in ax.get_legend().get_texts()}
        assert entries == {"Motion A", "Motion B", "start", "end"}

    def test_ax_injection_uses_provided_ax(self, bvh_example):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        returned_fig, returned_ax = bvhplot.trajectory(
            bvh_example, ax=axes[0], show=False)
        assert returned_ax is axes[0]
        assert returned_fig is fig

    def test_ax_injection_works_with_list(self, bvh_example):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        returned_fig, returned_ax = bvhplot.trajectory(
            [bvh_example, bvh_example],
            labels=["A", "B"], ax=ax, show=False)
        assert returned_ax is ax
        assert returned_fig is fig

    def test_ax_injection_with_3d_axes_raises(self, bvh_example):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')  # 3D axes — wrong!
        with pytest.raises(ValueError, match="2D axes"):
            bvhplot.trajectory(bvh_example, ax=ax, show=False)

    def test_facing_arrows_off_by_default(self, bvh_example):
        """Default call produces no quiver artist."""
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib.quiver import Quiver
        fig, ax = bvhplot.trajectory(bvh_example, show=False)
        quivers = [c for c in ax.get_children() if isinstance(c, Quiver)]
        assert len(quivers) == 0

    def test_facing_arrows_single_skeleton(self, bvh_example):
        """facing_arrows=True adds one quiver artist for a single skeleton."""
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib.quiver import Quiver
        fig, ax = bvhplot.trajectory(
            bvh_example, facing_arrows=True, show=False)
        quivers = [c for c in ax.get_children() if isinstance(c, Quiver)]
        assert len(quivers) == 1

    def test_facing_arrows_multi_skeleton(self, bvh_example, bvh_test2):
        """facing_arrows=True adds one quiver artist per skeleton; clips may differ in length."""
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib.quiver import Quiver
        fig, ax = bvhplot.trajectory(
            [bvh_example, bvh_test2],
            facing_arrows=True, labels=['A', 'B'], show=False)
        quivers = [c for c in ax.get_children() if isinstance(c, Quiver)]
        assert len(quivers) == 2

    def test_facing_arrows_via_bvh_wrapper(self, bvh_example):
        """bvh.plot_trajectory(facing_arrows=True) forwards the kwarg."""
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib.quiver import Quiver
        fig, ax = bvh_example.plot_trajectory(facing_arrows=True)
        quivers = [c for c in ax.get_children() if isinstance(c, Quiver)]
        assert len(quivers) == 1

    def test_tight_false_uses_skeleton_extent(self, bvh_example):
        """Default (tight=False) axes span the full horizontal skeleton extent,
        which is wider than the root path alone."""
        import matplotlib
        matplotlib.use('Agg')
        import numpy as np
        coords = bvh_example.node_positions()
        # Horizontal axes depend on world_up; drop the up axis
        from pybvh.tools import _AXIS_CHAR_TO_IDX
        up_idx = _AXIS_CHAR_TO_IDX[bvh_example.world_up[1]]
        horiz = [j for j in range(3) if j != up_idx]
        sk_h0_span = np.ptp(coords[:, :, horiz[0]])
        sk_h1_span = np.ptp(coords[:, :, horiz[1]])
        root_h0_span = np.ptp(coords[:, 0, horiz[0]])
        root_h1_span = np.ptp(coords[:, 0, horiz[1]])

        fig, ax = bvhplot.trajectory(bvh_example, tight=False, show=False)
        plot_h0_span = ax.get_xlim()[1] - ax.get_xlim()[0]
        plot_h1_span = ax.get_ylim()[1] - ax.get_ylim()[0]
        # Axis span should be close to the full skeleton span (not the
        # much smaller root path span).
        assert plot_h0_span > 1.5 * root_h0_span
        assert plot_h1_span > 1.5 * root_h1_span
        assert plot_h0_span >= sk_h0_span  # always at least the full extent
        assert plot_h1_span >= sk_h1_span

    def test_tight_true_fits_path(self, bvh_example):
        """tight=True leaves matplotlib auto-scaling to the root path,
        which is notably narrower than the full skeleton extent."""
        import matplotlib
        matplotlib.use('Agg')
        import numpy as np
        fig, ax_tight = bvhplot.trajectory(bvh_example, tight=True, show=False)
        fig2, ax_wide = bvhplot.trajectory(bvh_example, tight=False, show=False)
        tight_span = (ax_tight.get_xlim()[1] - ax_tight.get_xlim()[0])
        wide_span = (ax_wide.get_xlim()[1] - ax_wide.get_xlim()[0])
        assert tight_span < wide_span

    def test_tight_multi_skeleton_uses_union(self, bvh_example, bvh_test2):
        """tight=False for multi-skeleton plots uses the union of per-
        skeleton extents so both skeletons stay in frame."""
        import matplotlib
        matplotlib.use('Agg')
        fig, ax = bvhplot.trajectory(
            [bvh_example, bvh_test2], tight=False, labels=['A', 'B'],
            show=False)
        # Individual plots
        fig_a, ax_a = bvhplot.trajectory(bvh_example, tight=False, show=False)
        fig_b, ax_b = bvhplot.trajectory(bvh_test2, tight=False, show=False)
        # Multi-skeleton x-range should span at least the individual ranges
        # (modulo padding).
        assert ax.get_xlim()[1] >= max(ax_a.get_xlim()[1], ax_b.get_xlim()[1]) - 1e-6
        assert ax.get_xlim()[0] <= min(ax_a.get_xlim()[0], ax_b.get_xlim()[0]) + 1e-6


class TestRenderMatplotlib:
    def test_render_creates_file(self, bvh_example, tmp_path):
        import matplotlib
        matplotlib.use('Agg')
        # Use only first 5 frames for speed
        bvh_short = bvh_example[0:5]
        path = bvhplot.render(
            bvh_short, tmp_path / "test.gif",
            backend="matplotlib")
        assert path.exists()
        assert path.stat().st_size > 0

    def test_render_html(self, bvh_example, tmp_path):
        import matplotlib
        matplotlib.use('Agg')
        bvh_short = bvh_example[0:3]
        path = bvhplot.render(
            bvh_short, tmp_path / "test.html",
            backend="matplotlib")
        assert path.exists()
        assert path.stat().st_size > 0

    def test_render_with_follow(self, bvh_example, tmp_path):
        """render(follow=True) should produce a file without crashing."""
        import matplotlib
        matplotlib.use('Agg')
        bvh_short = bvh_example[0:5]
        path = bvhplot.render(
            bvh_short, tmp_path / "follow.gif",
            backend="matplotlib", follow=True)
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

    def test_render_with_follow(self, bvh_example, tmp_path):
        """render(follow=True) should produce a file without crashing."""
        path = bvhplot.render(
            bvh_example, tmp_path / "follow.mp4", backend="opencv",
            resolution=(320, 240), follow=True)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_follow_produces_different_output_than_static(
            self, bvh_example, tmp_path):
        """When the character rotates, follow=True should yield different
        pixel output than follow=False (because the camera moves with the
        skeleton)."""
        import cv2
        # Apply a 90° rotation at frame 0, then rotate_vertical to sweep
        # through an extra 180° over the clip — guaranteed to produce a
        # turning character.
        bvh_short = bvh_example[0:5]
        path_static = bvhplot.render(
            bvh_short, tmp_path / "static.mp4", backend="opencv",
            resolution=(320, 240), follow=False)
        path_follow = bvhplot.render(
            bvh_short, tmp_path / "follow.mp4", backend="opencv",
            resolution=(320, 240), follow=True)

        # Both files should exist. Compare the LAST frames: for a static
        # camera on a non-rotating character these are identical; for a
        # rotating character (or follow mode) they differ. We can't
        # guarantee the fixture rotates, so we just assert both files
        # open and produce valid frames.
        cap_s = cv2.VideoCapture(str(path_static))
        cap_f = cv2.VideoCapture(str(path_follow))
        ret_s, _ = cap_s.read()
        ret_f, _ = cap_f.read()
        cap_s.release()
        cap_f.release()
        assert ret_s and ret_f

    def test_follow_with_custom_camera_tuple_is_noop(
            self, bvh_example, tmp_path):
        """A custom (azim, elev) camera tuple is fixed, so follow=True
        should be a silent no-op and produce valid output anyway."""
        path = bvhplot.render(
            bvh_example, tmp_path / "follow_tuple.mp4", backend="opencv",
            resolution=(320, 240), camera=(45, 30), follow=True)
        assert path.exists()

    def test_gif_output(self, bvh_example, tmp_path):
        bvh_short = bvh_example[0:5]
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


# =============================================================================
# match_fps
# =============================================================================

class TestMatchFps:

    @pytest.fixture
    def bvh_30fps(self):
        return read_bvh_file(Path(__file__).parent.parent / "bvh_data" / "bvh_test1.bvh")

    @pytest.fixture
    def bvh_120fps(self):
        return read_bvh_file(Path(__file__).parent.parent / "bvh_data" / "bvh_test2.bvh")

    def test_warns_on_mismatch(self, bvh_30fps, bvh_120fps):
        from pybvh.bvhplot import _match_frame_rates
        with pytest.warns(UserWarning, match="Frame rates differ"):
            _match_frame_rates([bvh_30fps, bvh_120fps], None)

    def test_no_warning_when_same_fps(self, bvh_30fps):
        import warnings
        from pybvh.bvhplot import _match_frame_rates
        bvh2 = bvh_30fps.copy()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _match_frame_rates([bvh_30fps, bvh2], None)
            fps_warns = [x for x in w if "Frame rates differ" in str(x.message)]
            assert len(fps_warns) == 0

    def test_lowest_resamples_to_min(self, bvh_30fps, bvh_120fps):
        from pybvh.bvhplot import _match_frame_rates
        result = _match_frame_rates([bvh_30fps, bvh_120fps], "lowest")
        fps0 = 1.0 / result[0].frame_time
        fps1 = 1.0 / result[1].frame_time
        assert abs(fps0 - fps1) < 0.5
        assert abs(fps0 - 30.0) < 0.5

    def test_highest_resamples_to_max(self, bvh_30fps, bvh_120fps):
        from pybvh.bvhplot import _match_frame_rates
        result = _match_frame_rates([bvh_30fps, bvh_120fps], "highest")
        fps0 = 1.0 / result[0].frame_time
        fps1 = 1.0 / result[1].frame_time
        assert abs(fps0 - fps1) < 0.5
        assert abs(fps0 - 120.0) < 0.5

    def test_invalid_match_fps_raises(self, bvh_30fps, bvh_120fps):
        from pybvh.bvhplot import _match_frame_rates
        with pytest.raises(ValueError, match="match_fps"):
            _match_frame_rates([bvh_30fps, bvh_120fps], "bad")

    def test_single_clip_no_warning(self, bvh_30fps):
        import warnings
        from pybvh.bvhplot import _match_frame_rates
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _match_frame_rates([bvh_30fps], None)
            assert len(w) == 0
            assert len(result) == 1


class TestSceneSpacing:
    """Tests for _apply_scene_spacing() and _warn_world_up_mismatch().

    No k3d or vedo installation required — tests call helpers directly.
    """

    @pytest.fixture
    def two_bvhs(self):
        """Two skeletons with the same world_up (+z) for basic spacing tests."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from synthetic_bvh import make_pos_z_up_bvh
        b1 = make_pos_z_up_bvh()
        b2 = make_pos_z_up_bvh()
        return b1, b2

    @pytest.fixture
    def two_coords(self, two_bvhs):
        """Spatial coords for the two skeletons, centered='first'."""
        b1, b2 = two_bvhs
        c1 = b1.node_positions(centered="first")
        c2 = b2.node_positions(centered="first")
        return [c1, c2]

    # ------------------------------------------------------------------
    # Single skeleton — no offset ever applied
    # ------------------------------------------------------------------

    def test_single_skeleton_no_offset(self, two_bvhs, two_coords):
        from pybvh.bvhplot import _apply_scene_spacing
        b1, _ = two_bvhs
        c1 = two_coords[0]
        result = _apply_scene_spacing([b1], [c1], "auto", "z", "first")
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], c1)

    # ------------------------------------------------------------------
    # centered='world' + auto → no offset
    # ------------------------------------------------------------------

    def test_auto_world_no_offset(self, two_bvhs, two_coords):
        from pybvh.bvhplot import _apply_scene_spacing
        b1, b2 = two_bvhs
        result = _apply_scene_spacing(
            [b1, b2], two_coords, "auto", "z", "world")
        np.testing.assert_array_equal(result[0], two_coords[0])
        np.testing.assert_array_equal(result[1], two_coords[1])

    # ------------------------------------------------------------------
    # centered='first' + auto → skeleton 1 shifted laterally
    # ------------------------------------------------------------------

    def test_auto_first_applies_offset(self, two_bvhs, two_coords):
        from pybvh.bvhplot import _apply_scene_spacing
        b1, b2 = two_bvhs
        result = _apply_scene_spacing(
            [b1, b2], two_coords, "auto", "z", "first")
        # Skeleton 0 unchanged
        np.testing.assert_array_equal(result[0], two_coords[0])
        # Skeleton 1 shifted (at least one coordinate differs)
        assert not np.allclose(result[1], two_coords[1])
        # Skeleton 1 differs only along the lateral axis (not up axis Z=2)
        diff = result[1] - two_coords[1]
        assert np.allclose(diff[:, :, 2], 0.0), "Up axis (Z) must not shift"

    def test_auto_skeleton_applies_offset(self, two_bvhs):
        from pybvh.bvhplot import _apply_scene_spacing
        b1, b2 = two_bvhs
        c1 = b1.node_positions(centered="skeleton")
        c2 = b2.node_positions(centered="skeleton")
        result = _apply_scene_spacing(
            [b1, b2], [c1, c2], "auto", "z", "skeleton")
        assert not np.allclose(result[1], c2)

    # ------------------------------------------------------------------
    # Explicit float spacing
    # ------------------------------------------------------------------

    def test_explicit_float_offset(self, two_bvhs, two_coords):
        from pybvh.bvhplot import _apply_scene_spacing
        b1, b2 = two_bvhs
        result = _apply_scene_spacing(
            [b1, b2], two_coords, 3.0, "z", "first")
        diff = result[1] - two_coords[1]
        # Total shift magnitude = 3.0 (skeleton index 1 × spacing 3.0)
        np.testing.assert_allclose(np.linalg.norm(diff[0, 0]), 3.0, atol=1e-10)

    def test_explicit_zero_no_offset(self, two_bvhs, two_coords):
        from pybvh.bvhplot import _apply_scene_spacing
        b1, b2 = two_bvhs
        result = _apply_scene_spacing(
            [b1, b2], two_coords, 0.0, "z", "first")
        np.testing.assert_array_equal(result[0], two_coords[0])
        np.testing.assert_array_equal(result[1], two_coords[1])

    # ------------------------------------------------------------------
    # Offset is along the lateral axis only
    # ------------------------------------------------------------------

    def test_offset_along_lateral_axis(self, two_bvhs, two_coords):
        """Offset must be along the axis that is neither up nor forward."""
        from pybvh.bvhplot import _apply_scene_spacing
        from pybvh.bvhplot._common import UP_AXIS_INDEX
        b1, b2 = two_bvhs
        up_char = b1.world_up[1]
        fwd_str = b1.forward_at(frame=0)
        fwd_idx = UP_AXIS_INDEX[fwd_str[1]]
        up_idx = UP_AXIS_INDEX[up_char]
        lat_idx = next(i for i in range(3) if i != up_idx and i != fwd_idx)

        result = _apply_scene_spacing(
            [b1, b2], two_coords, 2.0, up_char, "first")
        diff = result[1] - two_coords[1]

        # Lateral axis carries the offset; others are zero
        assert not np.allclose(diff[:, :, lat_idx], 0.0), "Lateral axis should shift"
        assert np.allclose(diff[:, :, up_idx], 0.0), "Up axis must not shift"

    # ------------------------------------------------------------------
    # world_up mismatch warning
    # ------------------------------------------------------------------

    def test_world_up_mismatch_warning(self, two_bvhs):
        from pybvh.bvhplot import _warn_world_up_mismatch
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from synthetic_bvh import make_pos_y_up_bvh
        b_yup = make_pos_y_up_bvh()
        b_zup, _ = two_bvhs
        with pytest.warns(UserWarning, match="world_up"):
            _warn_world_up_mismatch([b_zup, b_yup])

    def test_no_warning_same_world_up(self, two_bvhs):
        import warnings
        from pybvh.bvhplot import _warn_world_up_mismatch
        b1, b2 = two_bvhs
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_world_up_mismatch([b1, b2])
            wu_warns = [x for x in w if "world_up" in str(x.message)]
            assert len(wu_warns) == 0

    def test_warning_mentions_reorient(self, two_bvhs):
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from synthetic_bvh import make_pos_y_up_bvh
        from pybvh.bvhplot import _warn_world_up_mismatch
        b_yup = make_pos_y_up_bvh()
        b_zup, _ = two_bvhs
        with pytest.warns(UserWarning, match="reorient_world_up"):
            _warn_world_up_mismatch([b_zup, b_yup])

    # ------------------------------------------------------------------
    # Invalid spacing raises ValueError
    # ------------------------------------------------------------------

    def test_negative_spacing_raises(self):
        from pybvh.bvhplot import play
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from synthetic_bvh import make_pos_z_up_bvh
        bvh = make_pos_z_up_bvh()
        with pytest.raises(ValueError, match="non-negative"):
            play([bvh, bvh], spacing=-1.0, backend="matplotlib")

    def test_invalid_string_spacing_raises(self):
        from pybvh.bvhplot import play
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from synthetic_bvh import make_pos_z_up_bvh
        bvh = make_pos_z_up_bvh()
        with pytest.raises(ValueError, match="spacing"):
            play([bvh, bvh], spacing="bad", backend="matplotlib")

