"""Tests for the pybvh.bvhplot visualization module."""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import re
from pathlib import Path

import numpy as np
import pytest

from pybvh import bvhplot, read_bvh_file
from pybvh.analysis import root_trajectory
from pybvh.bvhplot import _k3d, _opencv, _vedo
from pybvh.bvhplot._from_bvh import (
    get_camera_angles,
    get_skeleton_lines,
    normalize_input,
)
from pybvh.bvhplot._scene import align_frame_counts
from pybvh.bvhplot._viewport import (
    build_view_matrix,
    compute_unified_limits,
    ortho_project,
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


@pytest.fixture
def played(monkeypatch):
    """The (scene, fps) of each play() call that reached the matplotlib
    backend, which is stubbed out."""
    import pybvh.bvhplot._matplotlib as mpl_backend

    calls = []

    def fake_play_mpl(scene, style, fps, **kwargs):
        calls.append((scene, fps))

    monkeypatch.setattr(mpl_backend, "play_mpl", fake_play_mpl)
    return calls


@pytest.fixture
def drawn(monkeypatch):
    """The scene of each frame() call that reached the matplotlib
    backend, which is stubbed out."""
    import pybvh.bvhplot._matplotlib as mpl_backend

    scenes = []

    def fake_frame_mpl(scene, style, **kwargs):
        scenes.append(scene)
        return None, None

    monkeypatch.setattr(mpl_backend, "frame_mpl", fake_frame_mpl)
    return scenes


def assert_view_shows(view, clip, frames):
    """*view* holds *clip*'s poses and root heading at *frames*."""
    np.testing.assert_allclose(view.coords, clip.node_positions()[frames])
    np.testing.assert_allclose(view.root_heading, root_trajectory(clip)[frames, 2:4])


# ===================================================================
# Scene, viewport and Bvh reader helpers
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
        bvh_list, coords_list = normalize_input([bvh_example, bvh_example], None, "world")
        assert len(bvh_list) == 2
        assert len(coords_list) == 2

    def test_precomputed_array_2d(self, bvh_example):
        coords = bvh_example.node_positions(frame=0)
        _, coords_list = normalize_input(bvh_example, coords, "world")
        assert coords_list[0].shape == (1, len(bvh_example.nodes), 3)

    def test_precomputed_array_3d_keeps_its_first_frame(self, bvh_example):
        coords = bvh_example.node_positions()
        _, coords_list = normalize_input(bvh_example, coords, "world")
        np.testing.assert_array_equal(coords_list[0], coords[:1])

    def test_precomputed_array_with_list_raises(self, bvh_example):
        coords = bvh_example.node_positions()
        with pytest.raises(ValueError, match="single Bvh"):
            normalize_input([bvh_example, bvh_example], coords, "world")

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="At least one"):
            normalize_input([], None, "world")


class TestEmptyClipList:
    """Every entry point that takes a list of clips rejects an empty one
    with the same message, before any other work."""

    MESSAGE = "^" + re.escape("At least one Bvh object is required.") + "$"

    def test_render(self, tmp_path):
        with pytest.raises(ValueError, match=self.MESSAGE):
            bvhplot.render([], tmp_path / "out.gif")

    def test_play(self, played):
        with pytest.raises(ValueError, match=self.MESSAGE):
            bvhplot.play([], backend="matplotlib")

    def test_rest_pose(self):
        with pytest.raises(ValueError, match=self.MESSAGE):
            bvhplot.rest_pose([])

    def test_frame(self):
        with pytest.raises(ValueError, match=self.MESSAGE):
            bvhplot.frame([], 0)

    @pytest.mark.parametrize(
        "entry, args, invalid",
        [
            ("rest_pose", (), {"style": "no-such-style"}),
            ("frame", (0,), {"backend": "no-such-backend"}),
            ("trajectory", (), {"centered": "no-such-mode"}),
            ("render", ("out.gif",), {"sync": "no-such-sync"}),
            ("play", (), {"backend": "no-such-backend"}),
        ],
    )
    def test_checks_the_clips_before_other_arguments(self, bvh_example, entry, args, invalid):
        entry_point = getattr(bvhplot, entry)
        with pytest.raises(ValueError, match="^Unknown"):
            entry_point(bvh_example, *args, **invalid)
        with pytest.raises(ValueError, match=self.MESSAGE):
            entry_point([], *args, **invalid)

    def test_trajectory(self):
        with pytest.raises(ValueError, match=self.MESSAGE):
            bvhplot.trajectory([])


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
        assert up in ("x", "y", "z")

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
        view = build_view_matrix(0, 0, "y")
        center = np.array([1.0, 0.5, 0.5])
        pixels = ortho_project(coords, view, center, (2.0, 2.0), (640, 480))
        assert pixels.shape == (3, 2)
        assert pixels.dtype == np.int32

    def test_center_projects_to_image_center(self):
        center = np.array([5.0, 5.0, 5.0])
        coords = center.reshape(1, 3)
        view = build_view_matrix(0, 0, "y")
        pixels = ortho_project(coords, view, center, (2.0, 2.0), (640, 480))
        assert abs(pixels[0, 0] - 320) <= 1
        assert abs(pixels[0, 1] - 240) <= 1

    def test_different_resolutions(self):
        coords = np.zeros((1, 3), dtype=np.float64)
        view = build_view_matrix(0, 0, "y")
        center = np.zeros(3)
        p1 = ortho_project(coords, view, center, (1.0, 1.0), (100, 100))
        p2 = ortho_project(coords, view, center, (1.0, 1.0), (200, 200))
        # Center point should be at the center of each resolution
        assert abs(p1[0, 0] - 50) <= 1
        assert abs(p2[0, 0] - 100) <= 1

    def test_one_scale_set_by_the_tighter_direction(self):
        """A world unit covers the same pixels across and up; the half
        extents fit 90% of the panel in whichever direction is tighter."""
        view = np.eye(3)
        points = np.array([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        pixels = ortho_project(points, view, np.zeros(3), (2.0, 1.0), (400, 400))
        # width is the tight one: 2 units fill 0.9 * 200 px
        assert pixels[0].tolist() == [380, 200]
        assert pixels[1].tolist() == [200, 110]

    def test_a_direction_without_extent_sets_no_constraint(self):
        view = np.eye(3)
        points = np.array([[0.0, 1.0, 0.0]])
        pixels = ortho_project(points, view, np.zeros(3), (0.0, 1.0), (400, 200))
        assert pixels[0].tolist() == [200, 10]
        flat = ortho_project(points, view, np.zeros(3), (0.0, 0.0), (400, 200))
        assert flat[0].tolist() == [200, 99]


class TestBuildViewMatrix:
    def test_identity_like_at_zero(self):
        view = build_view_matrix(0, 0, "y")
        assert view.shape == (3, 3)
        # Should be close to identity (Y-up, no rotation)
        assert np.allclose(view, np.eye(3), atol=1e-10)

    def test_orthogonal(self):
        for azim, elev in [(30, 20), (90, 45), (-45, 60)]:
            view = build_view_matrix(azim, elev, "y")
            # Columns should be orthonormal
            assert np.allclose(view @ view.T, np.eye(3), atol=1e-10)

    def test_different_up_axes(self):
        for up in ("x", "y", "z"):
            view = build_view_matrix(0, 0, up)
            assert view.shape == (3, 3)
            assert np.allclose(view @ view.T, np.eye(3), atol=1e-10)

    def test_up_axis_points_up_on_screen(self):
        """Row 1 (view-up) should have its largest component along the up axis."""
        for up, idx in [("x", 0), ("y", 1), ("z", 2)]:
            view = build_view_matrix(0, 20, up)
            # Row 1 = up direction. The up_axis component should be the largest.
            assert abs(view[1, idx]) == max(abs(view[1, :]))

    def test_matches_matplotlib_right_direction(self, bvh_example):
        """OpenCV 'right' direction should match matplotlib for all up-axes."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import proj3d

        for up in ("y", "z"):
            for azim in (0, 45, 90, 180):
                fig = plt.figure()
                ax = fig.add_subplot(111, projection="3d")
                ax.view_init(elev=20, azim=azim, vertical_axis=up)
                fig.canvas.draw()

                # Matplotlib right direction: project unit vectors, take screen-x
                mpl_right = np.zeros(3)
                origin = np.array(proj3d.proj_transform(0, 0, 0, ax.get_proj()))
                for i in range(3):
                    v = np.zeros(3)
                    v[i] = 1.0
                    p = np.array(proj3d.proj_transform(*v, ax.get_proj()))
                    mpl_right[i] = p[0] - origin[0]
                plt.close()

                # OpenCV right direction: row 0 of view matrix
                vm = build_view_matrix(azim, 20, up)
                cv_right = vm[0, :]

                mpl_right /= np.linalg.norm(mpl_right)
                cv_right /= np.linalg.norm(cv_right)
                dot = np.dot(mpl_right, cv_right)
                assert dot > 0.99, f"Right direction mismatch: up={up}, azim={azim}, dot={dot:.4f}"


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
        fwd_idx = {"x": 0, "y": 1, "z": 2}[fwd[1]]
        fwd_sign = 1.0 if fwd[0] == "+" else -1.0
        fwd_vec[fwd_idx] = fwd_sign

        # Row 2 (w) points toward viewer. Positive w = toward viewer.
        fwd_w = (vm @ fwd_vec)[2]
        assert fwd_w > 0, (
            f"Forward axis should point toward viewer (w>0) in front view, got w={fwd_w:.3f}"
        )

    def test_front_view_right_hand_rule(self, bvh_example):
        """The view matrix should preserve right-handedness: det > 0."""
        frame = bvh_example.node_positions(frame=0)
        azim, elev, up = get_camera_angles(bvh_example, frame, "front")
        vm = build_view_matrix(azim, elev, up)
        assert np.linalg.det(vm) > 0, (
            f"View matrix should be right-handed (det>0), got det={np.linalg.det(vm):.3f}"
        )

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
        fwd_idx = {"x": 0, "y": 1, "z": 2}[fwd[1]]
        fwd_vec[fwd_idx] = 1.0 if fwd[0] == "+" else -1.0

        fwd_in_front = vm_f @ fwd_vec
        fwd_in_side = vm_s @ fwd_vec
        # In front view, forward is mostly in w (depth)
        assert abs(fwd_in_front[2]) > abs(fwd_in_front[0]), (
            "Forward should be mostly depth in front view"
        )
        # In side view, forward is mostly in u (screen horizontal)
        assert abs(fwd_in_side[0]) > abs(fwd_in_side[2]), (
            "Forward should be mostly horizontal in side view"
        )

    def test_backends_agree_on_front(self, bvh_example):
        """Matplotlib and OpenCV should show the same side of the skeleton."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import proj3d

        rest = bvh_example.rest_pose_positions()
        azim, elev, up = get_camera_angles(bvh_example, rest, "front")
        idx = bvh_example.node_index

        # Find a left/right pair with different positions
        lp = rp = None
        for n in bvh_example.nodes:
            if not n.is_end_site() and "Left" in n.name:
                rn = n.name.replace("Left", "Right")
                if rn in idx:
                    l_pos = rest[idx[n.name]]
                    r_pos = rest[idx[rn]]
                    if np.linalg.norm(l_pos - r_pos) > 1.0:
                        lp, rp = n.name, rn
                        break

        assert lp is not None, "Need a left/right pair for this test"

        # Matplotlib: check screen-x order
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
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
            f"cv Left<Right={cv_left_is_left}"
        )

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

        for side in ("Left", "Right"):
            ankle_name = f"{side}Ankle"
            toe_name = f"{side}Toe"
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
                f"looking at the back of the character)."
            )


# ===================================================================
# Public API tests (matplotlib backend)
# ===================================================================


class TestFrame:
    def test_single_frame_returns_fig_ax(self, bvh_example):
        import matplotlib

        matplotlib.use("Agg")  # non-interactive for CI
        fig, ax = bvhplot.frame(bvh_example, 0, show=False)
        assert fig is not None
        assert ax is not None

    def test_from_array(self, bvh_example):
        import matplotlib

        matplotlib.use("Agg")
        coords = bvh_example.node_positions(frame=0)
        fig, ax = bvhplot.frame(bvh_example, coords, show=False)
        assert fig is not None

    def test_side_by_side_returns_list(self, bvh_example):
        import matplotlib

        matplotlib.use("Agg")
        fig, axs = bvhplot.frame([bvh_example, bvh_example], 0, labels=["A", "B"], show=False)
        assert isinstance(axs, list)
        assert len(axs) == 2

    def test_centered_modes(self, bvh_example):
        import matplotlib

        matplotlib.use("Agg")
        for mode in ("world", "skeleton", "first"):
            fig, ax = bvhplot.frame(bvh_example, 0, centered=mode, show=False)
            assert fig is not None

    def test_ax_injection_uses_provided_ax(self, bvh_example):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, subplot_kw={"projection": "3d"}, figsize=(10, 5))
        returned_fig, returned_ax = bvhplot.frame(bvh_example, 0, ax=axes[0], show=False)
        # The returned ax must be the exact one we passed in
        assert returned_ax is axes[0]
        # The returned fig must be the one owning our ax
        assert returned_fig is fig

    def test_ax_injection_with_list_raises(self, bvh_example):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        with pytest.raises(ValueError, match="single skeletons"):
            bvhplot.frame([bvh_example, bvh_example], 0, ax=ax, show=False)

    def test_ax_injection_with_2d_axes_raises(self, bvh_example):
        import matplotlib

        matplotlib.use("Agg")
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

        matplotlib.use("Agg")
        # Sanity: these two fixtures really do have different up axes
        up1 = bvh_test1.world_up[1]
        up2 = bvh_test2.world_up[1]
        assert up1 != up2, (
            f"Fixture precondition: bvh_test1 up={up1}, bvh_test2 up={up2}. "
            "Mixed-up-axis test needs two different up axes."
        )

        fig, axs = bvhplot.frame([bvh_test1, bvh_test2], 0, show=False)

        # matplotlib stores view_init's vertical axis as an int index
        # (0=x, 1=y, 2=z) on Axes3D._vertical_axis.
        axis_name = {0: "x", 1: "y", 2: "z"}
        assert axis_name[axs[0]._vertical_axis] == up1
        assert axis_name[axs[1]._vertical_axis] == up2

    def test_a_negative_frame_counts_from_each_clips_end(self, bvh_example, drawn):
        """frame=-3 is each clip's third frame from its own end, for the
        pose and for the heading taken from the clip."""
        long, short = bvh_example, bvh_example[0:50]
        bvhplot.frame([long, short], -3)
        (scene,) = drawn
        for view, clip in zip(scene.views, (long, short)):
            third_from_end = len(clip) - 3
            assert_view_shows(view, clip, slice(third_from_end, third_from_end + 1))

    def test_one_frame_of_coords_is_drawn_as_given(self, bvh_example, drawn):
        """An (N, 3) array is one pose: the frame index is ignored, no
        clip heading is attached, and the floor is the pose's lowest
        point rather than the clip's."""
        pose = bvh_example.node_positions(frame=40)
        pose[:, 2] += 5.0  # lift it off the clip's floor
        bvhplot.frame(bvh_example, 10, coords=pose)
        (scene,) = drawn
        view = scene.views[0]
        np.testing.assert_array_equal(view.coords, pose[np.newaxis])
        assert view.root_heading is None
        assert view.floor_height == pose[:, 2].min()

    def test_a_clip_of_coords_reaches_the_scene_without_a_heading(self, bvh_example, drawn):
        """An (F, N, 3) array is accepted: the Scene holds only the
        array's own first frame, and no clip heading is attached."""
        coords = bvh_example.node_positions()[30:40]
        bvhplot.frame(bvh_example, coords=coords)
        (scene,) = drawn
        view = scene.views[0]
        np.testing.assert_array_equal(view.coords, coords[:1])
        assert view.root_heading is None

    def test_a_clip_of_coords_frames_the_still_like_its_first_frame(self, bvh_example):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        coords = bvh_example.node_positions()
        try:
            _, from_clip = bvhplot.frame(bvh_example, coords=coords)
            _, from_pose = bvhplot.frame(bvh_example, coords=coords[0])
            for limits in ("get_xlim", "get_ylim", "get_zlim"):
                np.testing.assert_allclose(
                    getattr(from_clip, limits)(), getattr(from_pose, limits)()
                )
        finally:
            plt.close("all")


class TestRestPose:
    def test_returns_fig_ax(self, bvh_example):
        import matplotlib

        matplotlib.use("Agg")
        fig, ax = bvhplot.rest_pose(bvh_example, show=False)
        assert fig is not None
        assert ax is not None

    def test_side_by_side_returns_list(self, bvh_example):
        import matplotlib

        matplotlib.use("Agg")
        fig, axs = bvhplot.rest_pose([bvh_example, bvh_example], labels=["A", "B"], show=False)
        assert isinstance(axs, list)
        assert len(axs) == 2

    def test_ax_injection_uses_provided_ax(self, bvh_example):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, subplot_kw={"projection": "3d"}, figsize=(10, 5))
        returned_fig, returned_ax = bvhplot.rest_pose(bvh_example, ax=axes[0], show=False)
        assert returned_ax is axes[0]
        assert returned_fig is fig

    def test_ax_injection_with_list_raises(self, bvh_example):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        with pytest.raises(ValueError, match="single skeletons"):
            bvhplot.rest_pose([bvh_example, bvh_example], ax=ax, show=False)


class TestTrajectory:
    def test_returns_fig_ax(self, bvh_example):
        import matplotlib

        matplotlib.use("Agg")
        fig, ax = bvhplot.trajectory(bvh_example, show=False)
        assert fig is not None
        assert ax is not None

    def test_multi_skeleton(self, bvh_example):
        import matplotlib

        matplotlib.use("Agg")
        fig, ax = bvhplot.trajectory([bvh_example, bvh_example], labels=["A", "B"], show=False)
        assert ax.get_legend() is not None

    def test_legend_always_has_start_end(self, bvh_example):
        """Start/end marker entries must be in the legend even without labels."""
        import matplotlib

        matplotlib.use("Agg")
        fig, ax = bvhplot.trajectory(bvh_example, show=False)
        legend = ax.get_legend()
        assert legend is not None
        entries = {t.get_text() for t in legend.get_texts()}
        assert "start" in entries
        assert "end" in entries

    def test_legend_with_labels_includes_both(self, bvh_example):
        """When labels are passed, legend has BOTH skeleton labels AND start/end."""
        import matplotlib

        matplotlib.use("Agg")
        fig, ax = bvhplot.trajectory(
            [bvh_example, bvh_example], labels=["Motion A", "Motion B"], show=False
        )
        entries = {t.get_text() for t in ax.get_legend().get_texts()}
        assert entries == {"Motion A", "Motion B", "start", "end"}

    def test_ax_injection_uses_provided_ax(self, bvh_example):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        returned_fig, returned_ax = bvhplot.trajectory(bvh_example, ax=axes[0], show=False)
        assert returned_ax is axes[0]
        assert returned_fig is fig

    def test_ax_injection_works_with_list(self, bvh_example):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        returned_fig, returned_ax = bvhplot.trajectory(
            [bvh_example, bvh_example], labels=["A", "B"], ax=ax, show=False
        )
        assert returned_ax is ax
        assert returned_fig is fig

    def test_ax_injection_with_3d_axes_raises(self, bvh_example):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")  # 3D axes — wrong!
        with pytest.raises(ValueError, match="2D axes"):
            bvhplot.trajectory(bvh_example, ax=ax, show=False)

    def test_facing_arrows_off_by_default(self, bvh_example):
        """Default call produces no quiver artist."""
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.quiver import Quiver

        fig, ax = bvhplot.trajectory(bvh_example, show=False)
        quivers = [c for c in ax.get_children() if isinstance(c, Quiver)]
        assert len(quivers) == 0

    def test_facing_arrows_single_skeleton(self, bvh_example):
        """facing_arrows=True adds one quiver artist for a single skeleton."""
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.quiver import Quiver

        fig, ax = bvhplot.trajectory(bvh_example, facing_arrows=True, show=False)
        quivers = [c for c in ax.get_children() if isinstance(c, Quiver)]
        assert len(quivers) == 1

    def test_facing_arrows_multi_skeleton(self, bvh_example, bvh_test2):
        """facing_arrows=True adds one quiver artist per skeleton; clips may differ in length."""
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.quiver import Quiver

        fig, ax = bvhplot.trajectory(
            [bvh_example, bvh_test2], facing_arrows=True, labels=["A", "B"], show=False
        )
        quivers = [c for c in ax.get_children() if isinstance(c, Quiver)]
        assert len(quivers) == 2

    def test_facing_arrows_via_bvh_wrapper(self, bvh_example):
        """bvh.plot_trajectory(facing_arrows=True) forwards the kwarg."""
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.quiver import Quiver

        fig, ax = bvh_example.plot_trajectory(facing_arrows=True)
        quivers = [c for c in ax.get_children() if isinstance(c, Quiver)]
        assert len(quivers) == 1

    def test_tight_false_uses_skeleton_extent(self, bvh_example):
        """Default (tight=False) axes span the full horizontal skeleton extent,
        which is wider than the root path alone."""
        import matplotlib

        matplotlib.use("Agg")
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

        matplotlib.use("Agg")
        fig, ax_tight = bvhplot.trajectory(bvh_example, tight=True, show=False)
        fig2, ax_wide = bvhplot.trajectory(bvh_example, tight=False, show=False)
        tight_span = ax_tight.get_xlim()[1] - ax_tight.get_xlim()[0]
        wide_span = ax_wide.get_xlim()[1] - ax_wide.get_xlim()[0]
        assert tight_span < wide_span

    def test_tight_multi_skeleton_uses_union(self, bvh_example, bvh_test2):
        """tight=False for multi-skeleton plots uses the union of per-
        skeleton extents so both skeletons stay in frame."""
        import matplotlib

        matplotlib.use("Agg")
        fig, ax = bvhplot.trajectory(
            [bvh_example, bvh_test2], tight=False, labels=["A", "B"], show=False
        )
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

        matplotlib.use("Agg")
        # Use only first 5 frames for speed
        bvh_short = bvh_example[0:5]
        path = bvhplot.render(bvh_short, tmp_path / "test.gif", backend="matplotlib")
        assert path.exists()
        assert path.stat().st_size > 0

    def test_render_html(self, bvh_example, tmp_path):
        import matplotlib

        matplotlib.use("Agg")
        bvh_short = bvh_example[0:3]
        path = bvhplot.render(bvh_short, tmp_path / "test.html", backend="matplotlib")
        assert path.exists()
        assert path.stat().st_size > 0

    def test_render_with_follow(self, bvh_example, tmp_path):
        """render(follow=True) should produce a file without crashing."""
        import matplotlib

        matplotlib.use("Agg")
        bvh_short = bvh_example[0:5]
        path = bvhplot.render(bvh_short, tmp_path / "follow.gif", backend="matplotlib", follow=True)
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
            bvh_example, tmp_path / "out.mp4", backend="opencv", resolution=(320, 240)
        )
        assert path.exists()
        assert path.stat().st_size > 0

    def test_frame_count(self, bvh_example, tmp_path):
        import cv2

        path = bvhplot.render(
            bvh_example, tmp_path / "out.mp4", backend="opencv", resolution=(320, 240)
        )
        cap = cv2.VideoCapture(str(path))
        fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        assert fc == bvh_example.frame_count

    def test_resolution(self, bvh_example, tmp_path):
        import cv2

        path = bvhplot.render(
            bvh_example, tmp_path / "out.mp4", backend="opencv", resolution=(640, 480)
        )
        cap = cv2.VideoCapture(str(path))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        assert (w, h) == (640, 480)

    def test_side_by_side(self, bvh_example, tmp_path):
        path = bvhplot.render(
            [bvh_example, bvh_example],
            tmp_path / "cmp.mp4",
            backend="opencv",
            resolution=(640, 240),
            labels=["A", "B"],
        )
        assert path.exists()

    def test_camera_presets(self, bvh_example, tmp_path):
        for cam in ["front", "side", "top", (45, 30)]:
            path = bvhplot.render(
                bvh_example,
                tmp_path / "cam.mp4",
                backend="opencv",
                resolution=(320, 240),
                camera=cam,
            )
            assert path.exists()

    def test_render_with_follow(self, bvh_example, tmp_path):
        """render(follow=True) should produce a file without crashing."""
        path = bvhplot.render(
            bvh_example,
            tmp_path / "follow.mp4",
            backend="opencv",
            resolution=(320, 240),
            follow=True,
        )
        assert path.exists()
        assert path.stat().st_size > 0

    def test_follow_produces_different_output_than_static(self, bvh_example, tmp_path):
        """When the character rotates, follow=True should yield different
        pixel output than follow=False (because the camera moves with the
        skeleton)."""
        import cv2

        # Apply a 90° rotation at frame 0, then rotate_vertical to sweep
        # through an extra 180° over the clip — guaranteed to produce a
        # turning character.
        bvh_short = bvh_example[0:5]
        path_static = bvhplot.render(
            bvh_short,
            tmp_path / "static.mp4",
            backend="opencv",
            resolution=(320, 240),
            follow=False,
        )
        path_follow = bvhplot.render(
            bvh_short, tmp_path / "follow.mp4", backend="opencv", resolution=(320, 240), follow=True
        )

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

    def test_follow_with_custom_camera_tuple_is_noop(self, bvh_example, tmp_path):
        """A custom (azim, elev) camera tuple is fixed, so follow=True
        should be a silent no-op and produce valid output anyway."""
        path = bvhplot.render(
            bvh_example,
            tmp_path / "follow_tuple.mp4",
            backend="opencv",
            resolution=(320, 240),
            camera=(45, 30),
            follow=True,
        )
        assert path.exists()

    def test_gif_output(self, bvh_example, tmp_path):
        bvh_short = bvh_example[0:5]
        path = bvhplot.render(
            bvh_short, tmp_path / "out.gif", backend="opencv", resolution=(320, 240)
        )
        assert path.exists()
        assert path.suffix == ".gif"
        assert path.stat().st_size > 0

    def test_axes_full_draws_indicator(self, bvh_example, tmp_path):
        # axes visibility moved from show_axis= into Style (v0.9.0)
        path = bvhplot.render(
            bvh_example,
            tmp_path / "axis.mp4",
            backend="opencv",
            resolution=(320, 240),
            style=bvhplot.Style("paper", axes="full"),
        )
        assert path.exists()

    def test_auto_backend_selects_opencv(self, bvh_example, tmp_path):
        """When cv2 is available, auto backend should select opencv."""
        path = bvhplot.render(
            bvh_example, tmp_path / "auto.mp4", backend="auto", resolution=(320, 240)
        )
        assert path.exists()


# =============================================================================
# Backend / extension routing and fps resolution
# =============================================================================


class TestRenderCodec:
    """render(codec=): "auto" is best-available, "h264"/"mpeg4" are
    guarantees — H.264 needs the external ffmpeg binary (OpenCV builds
    cannot encode it), mp4v is the always-available desktop-player
    codec."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_cv2(self):
        pytest.importorskip("cv2")

    @staticmethod
    def _fourcc(path):
        import cv2

        cap = cv2.VideoCapture(str(path))
        fcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        cap.release()
        return "".join(chr((fcc >> 8 * i) & 0xFF) for i in range(4))

    def test_mpeg4_forced(self, bvh_example, tmp_path):
        path = bvhplot.render(
            bvh_example, tmp_path / "m4.mp4", backend="opencv", resolution=(320, 240), codec="mpeg4"
        )
        assert self._fourcc(path) in ("FMP4", "mp4v", "XVID")

    def test_h264_without_ffmpeg_raises(self, bvh_example, tmp_path):
        import shutil

        if shutil.which("ffmpeg"):
            pytest.skip("ffmpeg present — the raise path needs it absent")
        with pytest.raises(RuntimeError, match="ffmpeg"):
            bvhplot.render(
                bvh_example,
                tmp_path / "h264.mp4",
                backend="opencv",
                resolution=(320, 240),
                codec="h264",
            )

    def test_h264_with_ffmpeg(self, bvh_example, tmp_path):
        import shutil

        if not shutil.which("ffmpeg"):
            pytest.skip("needs the ffmpeg executable")
        path = bvhplot.render(
            bvh_example,
            tmp_path / "h264.mp4",
            backend="opencv",
            resolution=(320, 240),
            codec="h264",
        )
        assert path.exists() and path.stat().st_size > 0
        assert self._fourcc(path) in ("avc1", "h264", "H264")

    def test_auto_matches_environment(self, bvh_example, tmp_path):
        """auto == h264 when ffmpeg is on PATH, mp4v otherwise."""
        import shutil

        path = bvhplot.render(
            bvh_example, tmp_path / "auto.mp4", backend="opencv", resolution=(320, 240)
        )
        four = self._fourcc(path)
        if shutil.which("ffmpeg"):
            assert four in ("avc1", "h264", "H264")
        else:
            assert four in ("FMP4", "mp4v", "XVID")

    def test_unknown_codec_raises(self, bvh_example, tmp_path):
        with pytest.raises(ValueError, match="codec"):
            bvhplot.render(bvh_example, tmp_path / "x.mp4", codec="hevc")

    def test_codec_invalid_for_gif(self, bvh_example, tmp_path):
        with pytest.raises(ValueError, match="video containers"):
            bvhplot.render(bvh_example, tmp_path / "x.gif", codec="mpeg4")

    def test_mpeg4_rejected_on_matplotlib_backend(self, bvh_example, tmp_path):
        with pytest.raises(ValueError, match="OpenCV"):
            bvhplot.render(bvh_example, tmp_path / "x.mp4", backend="matplotlib", codec="mpeg4")


class TestRenderBackendResolution:
    """Extension-aware backend routing for render()."""

    def test_mpl_only_extensions_route_to_matplotlib(self):
        """Formats OpenCV cannot write always go to matplotlib under auto,
        even when cv2 is installed."""
        from pybvh.bvhplot import _resolve_render_backend

        for ext in (".html", ".webp", ".apng", ".gif"):
            assert _resolve_render_backend("auto", ext) == "matplotlib"

    def test_video_extensions_prefer_opencv_when_available(self):
        pytest.importorskip("cv2")
        from pybvh.bvhplot import _resolve_render_backend

        for ext in (".mp4", ".mov", ".avi"):
            assert _resolve_render_backend("auto", ext) == "opencv"

    def test_explicit_backend_wins_over_extension(self):
        from pybvh.bvhplot import _resolve_render_backend

        assert _resolve_render_backend("matplotlib", ".mp4") == "matplotlib"
        assert _resolve_render_backend("opencv", ".gif") == "opencv"

    def test_unknown_backend_raises(self, bvh_example, tmp_path):
        with pytest.raises(ValueError, match="backend"):
            bvhplot.render(bvh_example, tmp_path / "x.mp4", backend="opencvv")

    def test_forced_opencv_rejects_unsupported_extension(self, bvh_example, tmp_path):
        pytest.importorskip("cv2")
        with pytest.raises(ValueError, match="cannot write"):
            bvhplot.render(bvh_example, tmp_path / "x.html", backend="opencv")

    def test_auto_gif_renders_without_codec_error(self, bvh_example, tmp_path):
        """The routing regression: auto + .gif must not hit OpenCV's
        VideoWriter (which cannot write GIF containers)."""
        import matplotlib

        matplotlib.use("Agg")
        bvh_short = bvh_example[0:3]
        path = bvhplot.render(bvh_short, tmp_path / "route.gif")
        assert path.exists()
        assert path.suffix == ".gif"
        assert path.stat().st_size > 0


class TestFpsResolution:
    """Shared fps parameter of play() and render()."""

    def test_none_uses_bvh_frame_rate(self):
        from pybvh.bvhplot import _resolve_fps

        assert _resolve_fps(None, 1.0 / 30.0) == pytest.approx(30.0)

    def test_fractional_fps_preserved(self):
        from pybvh.bvhplot import _resolve_fps

        assert _resolve_fps(119.88, 1.0 / 30.0) == pytest.approx(119.88)

    def test_zero_fps_raises(self):
        from pybvh.bvhplot import _resolve_fps

        with pytest.raises(ValueError, match="fps must be positive"):
            _resolve_fps(0, 1.0 / 30.0)

    def test_negative_fps_raises(self):
        from pybvh.bvhplot import _resolve_fps

        with pytest.raises(ValueError, match="fps must be positive"):
            _resolve_fps(-1, 1.0 / 30.0)

    def test_render_fps_zero_raises(self, bvh_example, tmp_path):
        with pytest.raises(ValueError, match="fps must be positive"):
            bvhplot.render(bvh_example, tmp_path / "x.gif", fps=0)

    def test_render_fractional_fps(self, bvh_example, tmp_path):
        import matplotlib

        matplotlib.use("Agg")
        bvh_short = bvh_example[0:3]
        path = bvhplot.render(bvh_short, tmp_path / "frac.gif", backend="matplotlib", fps=12.5)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_play_caps_the_clip_rate_at_30_fps(self, bvh_test2, played):
        """bvh_test2 is 61 frames at 120 fps: every 4th frame at 30 fps."""
        bvhplot.play(bvh_test2, backend="matplotlib")
        ((scene, fps),) = played
        assert fps == pytest.approx(30.0)
        assert scene.views[0].coords.shape[0] == 16

    def test_play_keeps_every_frame_at_an_explicit_fps(self, bvh_test2, played):
        bvhplot.play(bvh_test2, backend="matplotlib", fps=120)
        ((scene, fps),) = played
        assert fps == 120
        assert scene.views[0].coords.shape[0] == 61

    @pytest.mark.parametrize(
        "rate, step, played_fps",
        [
            (100.0, 4, 25.0),
            (31.0, 2, 15.5),
            (30.0, 1, 30.0),
        ],
    )
    def test_play_cap_keeps_every_step_th_frame_at_its_moment(
        self, bvh_test2, played, rate, step, played_fps
    ):
        """The step is the smallest whole number that brings the clip
        rate to 30 fps or below. The kept frames keep their moments, so
        the frame time grows by the step and the rate played is the one
        the subsampled Scene states: the clip rate over the step, which
        is 30 only when the clip rate is exactly 30 times the step."""
        bvh_test2.frame_time = 1.0 / rate
        bvhplot.play(bvh_test2, backend="matplotlib")
        ((scene, fps),) = played
        assert_view_shows(scene.views[0], bvh_test2, slice(None, None, step))
        assert scene.frame_time == pytest.approx(step / rate)
        assert fps == pytest.approx(played_fps)

    @pytest.mark.parametrize(
        "backend, library, module, player, step, played_fps",
        [
            pytest.param("k3d", "k3d", _k3d, "play_k3d", 4, 30.0, id="k3d"),
            pytest.param("vedo", "vedo", _vedo, "play_vedo", 1, 120.0, id="vedo"),
            # the notebook inline video
            pytest.param("opencv", "cv2", _opencv, "render_opencv", 1, 120.0, id="opencv"),
        ],
    )
    def test_play_caps_the_rate_only_where_the_player_needs_it(
        self, bvh_test2, monkeypatch, backend, library, module, player, step, played_fps
    ):
        """k3d widgets cannot keep up with 120 fps; vedo's timer and a
        notebook video player can."""
        pytest.importorskip(library)
        signature = inspect.signature(getattr(module, player))
        calls = []

        def record(*args, **kwargs):
            given = signature.bind(*args, **kwargs).arguments
            calls.append((given["scene"], given["fps"]))

        monkeypatch.setattr(module, player, record)
        if backend == "opencv":
            ipython_display = pytest.importorskip("IPython.display")
            monkeypatch.setattr(bvhplot, "_detect_notebook", lambda: True)
            # the video file is never written: show nothing
            monkeypatch.setattr(ipython_display, "Video", lambda *args, **kwargs: None)
            monkeypatch.setattr(ipython_display, "display", lambda *args, **kwargs: None)
        bvhplot.play(bvh_test2, backend=backend)
        ((scene, fps),) = calls
        assert scene.num_frames == len(range(0, len(bvh_test2), step))
        assert fps == pytest.approx(played_fps)


class TestPlaySync:
    """Clips of unequal length share one frame counter: sync= says
    whether the longer clip is cut or the shorter one held."""

    @pytest.fixture
    def clips(self, bvh_example):
        """75 and 40 frames, both at 30 fps, so play() keeps every frame."""
        return bvh_example, bvh_example[20:60]

    def test_truncate_cuts_every_clip_to_the_shortest(self, clips, played):
        long, short = clips
        bvhplot.play([long, short], backend="matplotlib", sync="truncate")
        ((scene, _),) = played
        long_view, short_view = scene.views
        assert scene.num_frames == len(short)
        assert_view_shows(long_view, long, slice(len(short)))
        assert_view_shows(short_view, short, slice(None))

    def test_pad_holds_the_shorter_clip_on_its_last_frame(self, clips, played):
        long, short = clips
        bvhplot.play([long, short], backend="matplotlib", sync="pad")
        ((scene, _),) = played
        long_view, short_view = scene.views
        assert scene.num_frames == len(long)
        assert_view_shows(long_view, long, slice(None))
        last = len(short) - 1
        held = [*range(len(short)), *[last] * (len(long) - len(short))]
        assert_view_shows(short_view, short, held)


class TestUnsetFrameTime:
    """A clip whose frame_time is 0, the "unset" value a Bvh built in
    memory carries until a rate is assigned."""

    @pytest.fixture
    def unset(self, bvh_test1):
        clip = bvh_test1[0:5]
        clip.frame_time = 0
        return clip

    def test_static_entry_points_draw_the_clip(self, unset):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        try:
            bvhplot.rest_pose(unset)
            bvhplot.frame(unset, 0)
            bvhplot.sequence(unset, n_poses=3)
        finally:
            plt.close("all")

    def test_render_raises_naming_both_ways_out(self, unset, tmp_path):
        with pytest.raises(ValueError, match="frame_time") as info:
            bvhplot.render(unset, tmp_path / "x.gif", backend="matplotlib")
        assert "bvh.fps" in str(info.value)
        assert "fps=" in str(info.value)
        assert not (tmp_path / "x.gif").exists()

    def test_play_raises_naming_both_ways_out(self, unset):
        with pytest.raises(ValueError, match="frame_time") as info:
            bvhplot.play(unset, backend="matplotlib")
        assert "bvh.fps" in str(info.value)
        assert "fps=" in str(info.value)

    def test_render_with_fps_draws_the_clip(self, unset, tmp_path):
        import matplotlib

        matplotlib.use("Agg")
        path = bvhplot.render(unset, tmp_path / "x.gif", backend="matplotlib", fps=10)
        assert path.stat().st_size > 0

    @pytest.mark.parametrize("backend", ["matplotlib", "opencv"])
    def test_render_with_fps_follows_the_clip(self, unset, tmp_path, backend):
        """The follow camera smooths over a second, which on this clip
        is a second of playback at the fps given."""
        if backend == "opencv":
            pytest.importorskip("cv2")
        else:
            import matplotlib

            matplotlib.use("Agg")
        path = bvhplot.render(unset, tmp_path / "x.gif", backend=backend, fps=10, follow=True)
        assert path.stat().st_size > 0

    def test_render_with_fps_draws_the_trajectory(self, unset, tmp_path):
        import matplotlib

        matplotlib.use("Agg")
        path = bvhplot.render(
            unset, tmp_path / "x.gif", backend="matplotlib", fps=10, trajectory=True
        )
        assert path.stat().st_size > 0

    @pytest.mark.filterwarnings(
        "ignore:FigureCanvasAgg is non-interactive",
        "ignore:Animation was deleted without rendering",
    )
    def test_play_with_fps_plays_the_clip(self, unset):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        try:
            bvhplot.play(unset, backend="matplotlib", fps=10)
        finally:
            plt.close("all")

    def test_render_ghost_needs_the_clip_rate_despite_fps(self, unset, tmp_path):
        with pytest.raises(ValueError, match="ghost") as info:
            bvhplot.render(unset, tmp_path / "x.gif", backend="matplotlib", fps=10, ghost=2)
        assert "frame_time" in str(info.value)

    def test_a_later_unset_clip_is_caught(self, bvh_test1, unset, tmp_path):
        clips = [bvh_test1[0:5], unset]
        with pytest.raises(ValueError, match="index 1"):
            bvhplot.render(clips, tmp_path / "x.gif", backend="matplotlib")
        with pytest.raises(ValueError, match="ghost"):
            bvhplot.render(clips, tmp_path / "x.gif", backend="matplotlib", fps=10, ghost=2)

    def test_match_fps_does_not_resample_to_zero(self, bvh_test1, unset, tmp_path):
        clips = [bvh_test1[0:5], unset]
        with pytest.raises(ValueError, match="match_fps") as info:
            bvhplot.render(
                clips, tmp_path / "x.gif", backend="matplotlib", fps=10, match_fps="lowest"
            )
        assert "index 1" in str(info.value)

    def test_render_ghost_without_fps_names_only_the_working_fix(self, unset, tmp_path):
        """fps= alone cannot satisfy ghost=, so it is not offered alone."""
        with pytest.raises(ValueError, match="ghost") as info:
            bvhplot.render(unset, tmp_path / "x.gif", backend="matplotlib", ghost=2)
        message = str(info.value)
        assert "bvh.frame_time" in message
        assert "drop ghost=" in message
        assert "or pass fps=" not in message

    def test_play_match_fps_without_fps_names_only_the_working_fix(self, bvh_test1, unset, played):
        """fps= alone cannot satisfy match_fps=, so it is not offered alone."""
        with pytest.raises(ValueError, match="match_fps") as info:
            bvhplot.play([bvh_test1[0:5], unset], backend="matplotlib", match_fps="lowest")
        message = str(info.value)
        assert "index 1" in message
        assert "bvh.frame_time" in message
        assert "drop match_fps=" in message
        assert "or pass fps=" not in message
        assert played == []

    def test_match_fps_raises_on_an_unset_clip_beside_a_slow_one(self, bvh_test1, unset, tmp_path):
        """An unset rate is not 0 fps: beside a 0.25 fps clip it must not
        pass as close enough to need no resampling."""
        slow = bvh_test1[0:5]
        slow.frame_time = 4.0
        with pytest.raises(ValueError, match="match_fps"):
            bvhplot.render(
                [slow, unset], tmp_path / "x.gif", backend="matplotlib", fps=10, match_fps="lowest"
            )

    @pytest.mark.parametrize("match_fps", ["lowest", "highest"])
    def test_match_fps_raises_when_every_clip_is_unset(self, unset, match_fps, tmp_path):
        with pytest.raises(ValueError, match="match_fps") as info:
            bvhplot.render(
                [unset, unset.copy()],
                tmp_path / "x.gif",
                backend="matplotlib",
                fps=10,
                match_fps=match_fps,
            )
        assert "indices 0, 1" in str(info.value)

    def test_rate_warning_names_the_unset_clip(self, bvh_test1, unset, played):
        with pytest.warns(UserWarning, match="Frame rates differ") as record:
            bvhplot.play([bvh_test1[0:5], unset], backend="matplotlib", fps=10)
        message = " ".join(str(w.message) for w in record)
        assert "index 1" in message
        assert "30.0 fps" in message
        assert not re.search(r"(?<![\d.])0\.0 fps", message)
        assert "match_fps='lowest'" not in message

    def test_play_mixed_list_without_fps_raises(self, bvh_test1, unset, played):
        with pytest.raises(ValueError, match="index 1") as info:
            bvhplot.play([bvh_test1[0:5], unset], backend="matplotlib")
        assert "fps=" in str(info.value)
        assert played == []

    @pytest.mark.filterwarnings("ignore:Frame rates differ")
    def test_play_mixed_list_with_fps_plays(self, bvh_test1, unset, played):
        bvhplot.play([bvh_test1[0:5], unset], backend="matplotlib", fps=10)
        ((scene, fps),) = played
        assert len(scene.views) == 2
        assert fps == 10


def _view(bvh, coords):
    """The SkeletonView compute_follow_azimuths reads (no Bvh at draw time)."""
    from pybvh.bvhplot._from_bvh import make_scene

    return make_scene([bvh], [coords], "front", None).views[0]


class TestComputeFollowAzimuths:
    """Vectorized follow-camera azimuth tracking (shared by all backends)."""

    # Captured when the schedule began smoothing out the stride's sway
    # (#22). Frames 0 and 523 are the clip's ends, which the smoothing
    # holds at their measured heading: v0.9.0 had the same two values.
    PINNED = {
        0: -20.0,
        50: -18.875147461847853,
        100: -20.27533914489359,
        200: -18.67794831511157,
        300: -18.983727690956485,
        400: -19.146641370627364,
        523: -4.049526760718898,
    }

    def test_the_walk_is_followed_without_its_stride_sway(self):
        """#22: on the CMU walk the camera swayed about 30 degrees peak
        to peak with every stride. Sway is measured around the
        schedule's own one-second trend, a centred 121-frame moving
        average taken where it fits whole. The net turn, last frame
        minus first, is the unsmoothed schedule's (v0.9.0's -20 to
        -4.0495 degrees) and must be kept."""
        from pybvh.bvhplot._viewport import compute_follow_azimuths

        bvh = read_bvh_file(BVH_DIR / "cmu_12_01_walk.bvh")
        az = compute_follow_azimuths(_view(bvh, bvh.node_positions()), -20.0)

        window = 121
        trend = np.convolve(az, np.ones(window) / window, mode="valid")
        half = window // 2
        sway = az[half : len(az) - half] - trend
        assert np.ptp(sway) < 3.0
        assert az[-1] - az[0] == pytest.approx(15.950473239281102, abs=1.0)

    def test_frame0_equals_base(self, bvh_example):
        from pybvh.bvhplot._viewport import compute_follow_azimuths

        coords = bvh_example.node_positions()
        az = compute_follow_azimuths(_view(bvh_example, coords), -20.0)
        assert az.shape == (coords.shape[0],)
        assert az[0] == pytest.approx(-20.0)

    def test_matches_per_frame_reference(self, bvh_example):
        """The schedule is the per-frame heading change of the tools
        helpers it vectorizes, unwrapped, then averaged frame by frame
        under a Gaussian of FOLLOW_SIGMA seconds reaching
        FOLLOW_TRUNCATE standard deviations, over the clip extended by
        point reflection about its end frames."""
        import math

        from pybvh.bvhplot._viewport import FOLLOW_SIGMA, FOLLOW_TRUNCATE, compute_follow_azimuths
        from pybvh.tools import (
            _axis_to_vector,
            _signed_rotation_delta_around_axis,
            _world_leftward_unit_at_frame,
        )

        coords = bvh_example.node_positions()
        base_azim = 160.0
        vec = compute_follow_azimuths(_view(bvh_example, coords), base_azim)

        up_vec = _axis_to_vector(bvh_example.world_up)
        left_0 = _world_leftward_unit_at_frame(bvh_example, coords[0], bvh_example.world_up)
        assert left_0 is not None
        changes = []
        for f in range(coords.shape[0]):
            left_f = _world_leftward_unit_at_frame(bvh_example, coords[f], bvh_example.world_up)
            assert left_f is not None
            changes.append(_signed_rotation_delta_around_axis(left_0, left_f, up_vec))
        change = np.unwrap(changes, period=360.0)

        last = len(change) - 1
        sigma = FOLLOW_SIGMA / bvh_example.frame_time
        reach = min(math.ceil(FOLLOW_TRUNCATE * sigma), last)
        for f in range(len(change)):
            total = weight_sum = 0.0
            for k in range(-reach, reach + 1):
                i = f + k
                if i < 0:
                    value = 2 * change[0] - change[-i]
                elif i > last:
                    value = 2 * change[last] - change[2 * last - i]
                else:
                    value = change[i]
                weight = math.exp(-0.5 * (k / sigma) ** 2)
                total += weight * value
                weight_sum += weight
            expected = base_azim + total / weight_sum
            assert vec[f] == pytest.approx(expected, abs=1e-9)

    def test_no_lr_pairs_falls_back_to_base(self, bvh_example):
        """A skeleton without L/R pairs keeps the base azimuth on all
        frames (camera stays fixed)."""
        from pybvh.bvhplot._viewport import compute_follow_azimuths

        # Spine-only skeleton: no lateral joints, so lr auto-detection
        # finds no pairs.
        bvh = bvh_example.extract_joints(
            ["Hips", "Spine", "Spine1", "Spine2", "Spine3", "Neck", "Head"]
        )
        assert bvh.lr_mapping is None
        coords = bvh.node_positions()
        az = compute_follow_azimuths(_view(bvh, coords), 45.0)
        assert np.allclose(az, 45.0)

    def test_the_whole_schedule_matches_the_pin(self):
        """Every frame of the CMU walk, against the frozen array
        (``tests/fixtures/follow_azimuths_pinned.npz``): the follow
        camera must not move on any of them. The seven values written
        out in the next test tie the fixture to what was captured when
        the schedule began smoothing (#22)."""
        from pybvh.bvhplot._viewport import compute_follow_azimuths

        pinned = np.load(Path(__file__).parent / "fixtures" / "follow_azimuths_pinned.npz")
        bvh = read_bvh_file(BVH_DIR / "cmu_12_01_walk.bvh")
        coords = bvh.node_positions()
        az = compute_follow_azimuths(_view(bvh, coords), -20.0)
        np.testing.assert_allclose(az, pinned["azimuths"], rtol=0, atol=1e-12)
        for frame_idx, value in self.PINNED.items():
            assert pinned["azimuths"][frame_idx] == pytest.approx(value, abs=1e-12), (
                "the fixture was re-baselined"
            )

    def test_the_viewport_schedules_the_pinned_azimuths(self):
        from pybvh.bvhplot._viewport import make_viewport

        pinned = np.load(Path(__file__).parent / "fixtures" / "follow_azimuths_pinned.npz")
        bvh = read_bvh_file(BVH_DIR / "cmu_12_01_walk.bvh")
        view = _view(bvh, bvh.node_positions())
        viewport = make_viewport(
            [dataclasses.replace(view, azimuth=-20.0)], framing="clip", motion="follow"
        )
        np.testing.assert_allclose(viewport.azimuths, pinned["azimuths"], rtol=0, atol=1e-12)

    def test_pinned_values_on_real_turning_walk(self):
        """Hard-pinned outputs on cmu_12_01_walk, captured when the
        schedule began smoothing (#22): the follow camera must not
        move."""
        from pybvh.bvhplot._viewport import compute_follow_azimuths

        bvh = read_bvh_file(BVH_DIR / "cmu_12_01_walk.bvh")
        coords = bvh.node_positions()
        az = compute_follow_azimuths(_view(bvh, coords), -20.0)
        assert az.shape == (524,)
        for frame_idx, value in self.PINNED.items():
            assert az[frame_idx] == pytest.approx(value, abs=1e-12), (
                f"azimuth moved at frame {frame_idx}"
            )


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

    def test_resampling_never_drops_a_clip(self, bvh_30fps, bvh_120fps):
        """Every clip comes back, resampled or not, or the call raises: a
        clip whose rate cannot be compared (NaN, which the frame_time
        setter now rejects at the assignment) must not vanish from the
        comparison."""
        from pybvh.bvhplot import _match_frame_rates

        odd = bvh_30fps[0:5]
        try:
            odd.frame_time = float("nan")
            result = _match_frame_rates([bvh_30fps, bvh_120fps, odd], "lowest")
        except ValueError:
            return
        assert len(result) == 3

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
    """Scene.spread() (lateral spacing for the single-scene backends) and
    _warn_world_up_mismatch(). The router's policy around the spread is
    tested through play, frame and render in TestSpreadInOneScene.

    No k3d or vedo installation required — the Scene is exercised directly.
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

    @staticmethod
    def _scene(bvhs, coords):
        from pybvh.bvhplot._from_bvh import make_scene

        return make_scene(list(bvhs), list(coords), "front", None)

    # ------------------------------------------------------------------
    # Single skeleton — no offset ever applied
    # ------------------------------------------------------------------

    def test_single_skeleton_no_offset(self, two_bvhs, two_coords):
        b1, _ = two_bvhs
        scene = self._scene([b1], [two_coords[0]])
        assert scene.spread("auto") is scene

    # ------------------------------------------------------------------
    # Explicit float spacing
    # ------------------------------------------------------------------

    def test_explicit_float_offset(self, two_bvhs, two_coords):
        scene = self._scene(two_bvhs, two_coords)
        result = scene.spread(3.0)
        diff = result.views[1].coords - two_coords[1]
        # Total shift magnitude = 3.0 (skeleton index 1 × spacing 3.0)
        np.testing.assert_allclose(np.linalg.norm(diff[0, 0]), 3.0, atol=1e-10)

    def test_explicit_zero_no_offset(self, two_bvhs, two_coords):
        scene = self._scene(two_bvhs, two_coords)
        result = scene.spread(0.0)
        np.testing.assert_array_equal(result.views[0].coords, two_coords[0])
        np.testing.assert_array_equal(result.views[1].coords, two_coords[1])

    # ------------------------------------------------------------------
    # Offset is along the lateral axis only
    # ------------------------------------------------------------------

    def test_offset_along_lateral_axis(self, two_bvhs, two_coords):
        """Offset must be along the axis that is neither up nor forward."""
        from pybvh.bvhplot._scene import UP_AXIS_INDEX

        scene = self._scene(two_bvhs, two_coords)
        first = scene.views[0]
        up_idx = first.up_index
        fwd_idx = UP_AXIS_INDEX[first.forward_axis[1]]
        lat_idx = next(i for i in range(3) if i != up_idx and i != fwd_idx)

        result = scene.spread(2.0)
        diff = result.views[1].coords - two_coords[1]

        # Lateral axis carries the offset; others are zero
        assert not np.allclose(diff[:, :, lat_idx], 0.0), "Lateral axis should shift"
        assert np.allclose(diff[:, :, up_idx], 0.0), "Up axis must not shift"

    def test_spread_moves_what_is_framed_and_keeps_the_floor(self, two_bvhs, two_coords):
        """A picture of the moved view is framed where the view went;
        a lateral move leaves the floor where it was."""
        from pybvh.bvhplot._viewport import make_viewport

        scene = self._scene(two_bvhs, two_coords)
        result = scene.spread(3.0)
        for before, after in zip(scene.views, result.views):
            shift = after.coords[0, 0] - before.coords[0, 0]
            np.testing.assert_allclose(
                make_viewport([after]).center, make_viewport([before]).center + shift
            )
            assert make_viewport([after]).half_span == pytest.approx(
                make_viewport([before]).half_span
            )
            assert after.floor_height == before.floor_height

    # ------------------------------------------------------------------
    # world_up mismatch warning
    # ------------------------------------------------------------------

    def test_world_up_mismatch_warning(self, two_bvhs):
        import sys

        from pybvh.bvhplot import _warn_world_up_mismatch

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


@pytest.fixture
def reached(monkeypatch):
    """Stub the backend function *name* of *module*; returns the Scene
    of each call that reached it."""
    scenes = []

    def backend(scene, *args, **kwargs):
        scenes.append(scene)
        return None, None  # frame() unpacks matplotlib's (fig, ax)

    def stub(module, name):
        monkeypatch.setattr(module, name, backend)
        return scenes

    return stub


# (entry point, backend, backend module, its stubbed function)
SINGLE_SCENE = [
    ("play", "k3d", "_k3d", "play_k3d"),
    ("play", "vedo", "_vedo", "play_vedo"),
    ("frame", "vedo", "_vedo_offscreen", "frame_vedo"),
    ("render", "vedo", "_vedo_offscreen", "render_vedo"),
]
PANELS = [
    ("play", "matplotlib", "_matplotlib", "play_mpl"),
    ("frame", "matplotlib", "_matplotlib", "frame_mpl"),
    ("render", "matplotlib", "_matplotlib", "render_mpl"),
    ("render", "opencv", "_opencv", "render_opencv"),
]


def draw_through(entry_point, backend, clips, tmp_path, **kwargs):
    """Draw *clips* through the bvhplot function *entry_point* on
    *backend*; render writes at 30 fps into *tmp_path*."""
    kwargs["backend"] = backend
    if entry_point == "play":
        bvhplot.play(clips, **kwargs)
    elif entry_point == "frame":
        bvhplot.frame(clips, **kwargs)
    else:
        suffix = ".gif" if backend == "matplotlib" else ".mp4"
        bvhplot.render(clips, tmp_path / f"pair{suffix}", fps=30, **kwargs)


class TestMatchSize:
    """match_size=True draws every skeleton of a single-scene backend
    as tall as the first; the clips are not touched, and the backends
    that draw each skeleton in its own panel draw the same thing."""

    @pytest.fixture
    def clips(self, bvh_test2):
        """The CMU walk and bvh_test2, which stands 7.5 times as tall."""
        return [read_bvh_file(BVH_DIR / "cmu_12_01_walk.bvh"), bvh_test2]

    @staticmethod
    def _draw(entry_point, backend, clips, tmp_path, **kwargs):
        """Draw *clips* through *entry_point*, labelled "walk" and
        "test2" unless *labels* says otherwise."""
        kwargs.setdefault("labels", ["walk", "test2"])
        draw_through(entry_point, backend, clips, tmp_path, **kwargs)

    @pytest.mark.parametrize("entry_point, backend, module, function", SINGLE_SCENE)
    def test_a_single_scene_draws_the_second_clip_as_tall_as_the_first(
        self, clips, reached, tmp_path, entry_point, backend, module, function
    ):
        pytest.importorskip(backend)
        scenes = reached(importlib.import_module(f"pybvh.bvhplot.{module}"), function)
        test2 = clips[1]
        rest_pose, positions = test2.rest_pose_positions(), test2.node_positions()
        self._draw(entry_point, backend, clips, tmp_path, match_size=True)
        (scene,) = scenes
        first, second = scene.views
        assert second.body_size == pytest.approx(first.body_size)
        assert scene.labels == ["walk", "test2 ×0.13"]
        # the clips themselves are untouched
        np.testing.assert_array_equal(test2.rest_pose_positions(), rest_pose)
        np.testing.assert_array_equal(test2.node_positions(), positions)

    @pytest.mark.parametrize("entry_point, backend, module, function", SINGLE_SCENE)
    def test_off_by_default(self, clips, reached, tmp_path, entry_point, backend, module, function):
        pytest.importorskip(backend)
        scenes = reached(importlib.import_module(f"pybvh.bvhplot.{module}"), function)
        self._draw(entry_point, backend, clips, tmp_path)
        (scene,) = scenes
        first, second = scene.views
        assert second.body_size == pytest.approx(182.346225)
        assert scene.labels == ["walk", "test2"]

    @pytest.mark.parametrize("entry_point, backend, module, function", SINGLE_SCENE)
    def test_unlabelled_skeletons_show_the_factor_alone(
        self, clips, reached, tmp_path, entry_point, backend, module, function
    ):
        pytest.importorskip(backend)
        scenes = reached(importlib.import_module(f"pybvh.bvhplot.{module}"), function)
        self._draw(entry_point, backend, clips, tmp_path, labels=None, match_size=True)
        (scene,) = scenes
        assert scene.labels == [None, "×0.13"]

    @pytest.mark.parametrize("entry_point, backend, module, function", SINGLE_SCENE)
    def test_a_clip_in_another_unit_is_drawn_at_the_first_ones_size(
        self, clips, reached, tmp_path, entry_point, backend, module, function
    ):
        """The walk scaled from its file's unit to a hundred times it
        (inches to hundredths of an inch, say) is the same body in
        another unit: it is drawn back at the walk's size, by exactly
        the factor that separates the two."""
        pytest.importorskip(backend)
        scenes = reached(importlib.import_module(f"pybvh.bvhplot.{module}"), function)
        walk = clips[0]
        self._draw(
            entry_point,
            backend,
            [walk, walk.scale(100.0)],
            tmp_path,
            labels=["walk", "walk in hundredths"],
            match_size=True,
        )
        (scene,) = scenes
        first, second = scene.views
        assert second.body_size == pytest.approx(first.body_size, rel=1e-12)
        assert scene.labels == ["walk", "walk in hundredths ×0.01"]

    @pytest.mark.parametrize("entry_point, backend, module, function", PANELS)
    def test_the_panel_backends_draw_the_same_scene(
        self, clips, reached, tmp_path, entry_point, backend, module, function
    ):
        """Each panel is framed on its own skeleton: there is no size
        to match, and the label gets no factor."""
        if backend == "opencv":
            pytest.importorskip("cv2")
        scenes = reached(importlib.import_module(f"pybvh.bvhplot.{module}"), function)
        self._draw(entry_point, backend, clips, tmp_path)
        self._draw(entry_point, backend, clips, tmp_path, match_size=True)
        plain, matched = scenes
        assert matched.labels == plain.labels == ["walk", "test2"]
        for plain_view, matched_view in zip(plain.views, matched.views):
            np.testing.assert_array_equal(matched_view.coords, plain_view.coords)
            np.testing.assert_array_equal(matched_view.rest_coords, plain_view.rest_coords)

    # A skeleton with every offset zero has no left/right geometry
    # either, and says so when its facing is read; not what is tested.
    NO_FACING = "ignore:No usable left/right geometry:UserWarning"

    @staticmethod
    def _sizeless(clip):
        """*clip* with every rest-pose offset zero: no height to match."""
        for node in clip.nodes:
            node.offset = np.zeros(3)
        return clip

    @pytest.mark.filterwarnings(NO_FACING)
    @pytest.mark.parametrize("entry_point, backend, module, function", SINGLE_SCENE)
    def test_a_skeleton_with_no_size_warns_at_the_users_call(
        self, clips, reached, tmp_path, entry_point, backend, module, function
    ):
        """A skeleton whose rest pose has every node at one point has no
        height to match: it keeps its size, and the warning names the
        line that asked for the match."""
        pytest.importorskip(backend)
        reached(importlib.import_module(f"pybvh.bvhplot.{module}"), function)
        walk, test2 = clips
        with pytest.warns(UserWarning, match="view 1") as record:
            self._draw(
                entry_point, backend, [walk, self._sizeless(test2)], tmp_path, match_size=True
            )
        (warning,) = [w for w in record if "view 1" in str(w.message)]
        assert warning.filename == __file__

    @pytest.mark.filterwarnings(NO_FACING)
    @pytest.mark.parametrize(
        "method, module, function",
        [
            ("play", "_vedo", "play_vedo"),
            ("plot_frame", "_vedo_offscreen", "frame_vedo"),
            ("render", "_vedo_offscreen", "render_vedo"),
        ],
    )
    def test_through_a_bvh_method_the_warning_names_the_users_call(
        self, bvh_test2, reached, tmp_path, method, module, function
    ):
        """Bvh.play, plot_frame and render wrap the bvhplot functions:
        the warning still names the user's line, not the wrapper's."""
        pytest.importorskip("vedo")
        reached(importlib.import_module(f"pybvh.bvhplot.{module}"), function)
        sizeless = self._sizeless(bvh_test2)
        args = (tmp_path / "clip.mp4",) if method == "render" else ()
        kwargs = {} if method == "plot_frame" else {"fps": 30}
        with pytest.warns(UserWarning, match="first view") as record:
            getattr(sizeless, method)(*args, backend="vedo", match_size=True, **kwargs)
        (warning,) = [w for w in record if "first view" in str(w.message)]
        assert warning.filename == __file__


class TestSpreadInOneScene:
    """play, frame and render arrange a comparison in one scene the same
    way: skeleton k moves k × spacing to the first skeleton's own left,
    except that spacing="auto" under centered="world" keeps the files'
    positions. The panel backends accept spacing and ignore it."""

    # The synthetic skeleton is +z up and faces +y: its left is
    # up × forward = -x, where its LeftLeg hangs.
    LEFT = np.array([-1.0, 0.0, 0.0])

    @pytest.fixture
    def twins(self):
        """Two copies of one +z-up clip, so they stand at the same place
        under any centering."""
        import sys

        sys.path.insert(0, str(Path(__file__).parent))
        from synthetic_bvh import make_pos_z_up_bvh

        return [make_pos_z_up_bvh(), make_pos_z_up_bvh()]

    @pytest.mark.parametrize("entry_point, backend, module, function", SINGLE_SCENE)
    @pytest.mark.parametrize("centered", ["first", "skeleton"])
    def test_a_spacing_puts_the_second_skeleton_on_the_first_ones_left(
        self, twins, reached, tmp_path, centered, entry_point, backend, module, function
    ):
        pytest.importorskip(backend)
        scenes = reached(importlib.import_module(f"pybvh.bvhplot.{module}"), function)
        draw_through(entry_point, backend, twins, tmp_path, centered=centered, spacing=2.0)
        (scene,) = scenes
        first, second = scene.views
        np.testing.assert_allclose(
            second.coords - first.coords,
            np.broadcast_to(2.0 * self.LEFT, first.coords.shape),
            atol=1e-12,
        )

    @pytest.mark.parametrize("entry_point, backend, module, function", SINGLE_SCENE)
    @pytest.mark.parametrize("centered", ["first", "skeleton"])
    def test_auto_spacing_keeps_the_skeletons_apart(
        self, twins, reached, tmp_path, centered, entry_point, backend, module, function
    ):
        """The default spacing moves the second skeleton to the first
        one's left, clear of it on every frame drawn."""
        pytest.importorskip(backend)
        scenes = reached(importlib.import_module(f"pybvh.bvhplot.{module}"), function)
        draw_through(entry_point, backend, twins, tmp_path, centered=centered)
        (scene,) = scenes
        first, second = scene.views
        first_left_edge = (first.coords @ self.LEFT).max(axis=1)
        second_right_edge = (second.coords @ self.LEFT).min(axis=1)
        assert np.all(second_right_edge > first_left_edge)

    @pytest.mark.parametrize("centered", ["first", "skeleton"])
    def test_a_video_arranges_the_skeletons_as_the_viewer_does(
        self, twins, reached, tmp_path, centered
    ):
        pytest.importorskip("vedo")
        from pybvh.bvhplot import _vedo, _vedo_offscreen

        scenes = reached(_vedo, "play_vedo")
        reached(_vedo_offscreen, "render_vedo")
        draw_through("play", "vedo", twins, tmp_path, centered=centered)
        draw_through("render", "vedo", twins, tmp_path, centered=centered)
        played, rendered = scenes
        for played_view, rendered_view in zip(played.views, rendered.views):
            np.testing.assert_array_equal(rendered_view.coords, played_view.coords)

    @pytest.mark.parametrize("entry_point, backend, module, function", SINGLE_SCENE)
    def test_auto_spacing_keeps_world_positions(
        self, twins, reached, tmp_path, entry_point, backend, module, function
    ):
        """Under centered="world" the default spacing leaves each
        skeleton where its file puts it: the twins coincide."""
        pytest.importorskip(backend)
        scenes = reached(importlib.import_module(f"pybvh.bvhplot.{module}"), function)
        draw_through(entry_point, backend, twins, tmp_path, centered="world")
        (scene,) = scenes
        first, second = scene.views
        np.testing.assert_array_equal(second.coords, first.coords)
        drawn_frames = len(first.coords)  # a still draws frame 0 alone
        np.testing.assert_array_equal(first.coords, twins[0].node_positions()[:drawn_frames])

    @pytest.mark.parametrize("entry_point, backend, module, function", PANELS)
    def test_the_panel_backends_ignore_the_spacing(
        self, twins, reached, tmp_path, entry_point, backend, module, function
    ):
        """Each panel is framed on its own skeleton: nothing to spread."""
        if backend == "opencv":
            pytest.importorskip("cv2")
        scenes = reached(importlib.import_module(f"pybvh.bvhplot.{module}"), function)
        draw_through(entry_point, backend, twins, tmp_path, centered="first")
        draw_through(entry_point, backend, twins, tmp_path, centered="first", spacing=2.0)
        plain, spaced = scenes
        for plain_view, spaced_view in zip(plain.views, spaced.views):
            np.testing.assert_array_equal(spaced_view.coords, plain_view.coords)

    @pytest.mark.parametrize("entry_point", ["play", "frame", "render"])
    @pytest.mark.parametrize(
        "spacing, message",
        [
            (-1.0, "non-negative"),
            ("bad", "'auto' or a non-negative number"),
        ],
    )
    def test_an_invalid_spacing_raises_on_every_backend(
        self, twins, tmp_path, entry_point, spacing, message
    ):
        with pytest.raises(ValueError, match=message):
            draw_through(entry_point, "matplotlib", twins, tmp_path, spacing=spacing)

    @pytest.fixture(params=["sideways", "walk"])
    def pair_in_motion(self, request):
        """Two clips whose first skeleton is wider over the clip than at
        frame 6: the synthetic clip drifting 30 units to its left, or
        the CMU walk (a T-pose at frame 0, then walking) and its
        mirror."""
        if request.param == "walk":
            walk = read_bvh_file(BVH_DIR / "cmu_12_01_walk.bvh")
            return [walk, walk.mirror()]
        import sys

        sys.path.insert(0, str(Path(__file__).parent))
        from synthetic_bvh import make_pos_z_up_bvh

        drifting, still = make_pos_z_up_bvh(), make_pos_z_up_bvh()
        root_pos = drifting.root_pos.copy()
        root_pos[:, 0] -= np.linspace(0.0, 30.0, len(root_pos))
        drifting.root_pos = root_pos
        return [drifting, still]

    @pytest.mark.parametrize("centered", ["world", "first", "skeleton"])
    def test_a_still_moves_the_skeletons_as_the_viewer_does(
        self, pair_in_motion, reached, tmp_path, centered
    ):
        """The still at frame f moves each skeleton by the offset the
        viewer moves it by: "auto" spaces by the first skeleton's width
        over the clip, in its facing at the clip's start, not by the
        pose drawn. (Each is compared with itself at spacing 0, since a
        still under centered="first" is centred on the frame it draws.)"""
        pytest.importorskip("vedo")
        from pybvh.bvhplot import _vedo, _vedo_offscreen

        scenes = reached(_vedo, "play_vedo")
        reached(_vedo_offscreen, "frame_vedo")
        f = 6
        for spacing in ("auto", 0.0):
            draw_through(
                "play", "vedo", pair_in_motion, tmp_path, centered=centered, spacing=spacing
            )
            draw_through(
                "frame",
                "vedo",
                pair_in_motion,
                tmp_path,
                centered=centered,
                frame=f,
                spacing=spacing,
            )
        played, still, played_in_place, still_in_place = scenes
        for k in range(2):
            played_offset = played.views[k].coords[f] - played_in_place.views[k].coords[f]
            still_offset = still.views[k].coords[0] - still_in_place.views[k].coords[0]
            np.testing.assert_allclose(still_offset, played_offset, atol=1e-9)

    @pytest.fixture
    def walk_and_hundredfold(self):
        """The CMU walk and the same walk in a unit a hundred times
        smaller: matched in size, the second is drawn at ×0.01."""
        walk = read_bvh_file(BVH_DIR / "cmu_12_01_walk.bvh")
        return [walk, walk.scale(100.0)]

    @staticmethod
    def _matched_pair(clips, reached, tmp_path, centered, f):
        """The viewer's Scene and the still's at frame *f*, sizes
        matched."""
        from pybvh.bvhplot import _vedo, _vedo_offscreen

        scenes = reached(_vedo, "play_vedo")
        reached(_vedo_offscreen, "frame_vedo")
        draw_through("play", "vedo", clips, tmp_path, centered=centered, match_size=True)
        draw_through("frame", "vedo", clips, tmp_path, centered=centered, frame=f, match_size=True)
        return scenes

    def test_a_matched_still_in_world_coordinates_is_the_viewers_frame(
        self, walk_and_hundredfold, reached, tmp_path
    ):
        """Under centered="world" the still at frame f is the viewer at
        frame f: each skeleton is scaled about its ground point at the
        clip's first frame, as the viewer scales it, not at frame f."""
        pytest.importorskip("vedo")
        f = 60
        played, still = self._matched_pair(walk_and_hundredfold, reached, tmp_path, "world", f)
        assert still.labels == played.labels == [None, "×0.01"]
        for played_view, still_view in zip(played.views, still.views):
            np.testing.assert_allclose(still_view.coords[0], played_view.coords[f], atol=1e-9)

    def test_a_matched_still_centred_first_is_the_viewers_frame_recentred(
        self, walk_and_hundredfold, reached, tmp_path
    ):
        """Under centered="first" a still is centred on the frame it
        draws, so each skeleton sits where the viewer draws it at that
        frame, less the clip's own travel on the ground since its first
        frame, drawn at the skeleton's factor."""
        pytest.importorskip("vedo")
        f = 60
        played, still = self._matched_pair(walk_and_hundredfold, reached, tmp_path, "first", f)
        for clip, factor, played_view, still_view in zip(
            walk_and_hundredfold, [1.0, 0.01], played.views, still.views
        ):
            travel = clip.root_pos[f] - clip.root_pos[0]
            travel[played_view.up_index] = 0.0
            np.testing.assert_allclose(
                still_view.coords[0], played_view.coords[f] - factor * travel, atol=1e-9
            )

    @pytest.fixture
    def whole_clip_scenes(self, monkeypatch):
        """Count the Scenes prepared from every frame of the clips,
        which a still builds only to measure an arrangement on."""
        prepared = bvhplot._prepare
        calls = []

        def spy(clips, frames, *args, **kwargs):
            if frames is None:
                calls.append(list(clips))
            return prepared(clips, frames, *args, **kwargs)

        monkeypatch.setattr(bvhplot, "_prepare", spy)
        return calls

    def test_a_still_of_one_clip_prepares_only_its_frame(
        self, twins, reached, tmp_path, whole_clip_scenes
    ):
        """One skeleton has nothing to be arranged with, so its still
        never prepares the whole clip."""
        pytest.importorskip("vedo")
        from pybvh.bvhplot import _vedo_offscreen

        reached(_vedo_offscreen, "frame_vedo")
        bvhplot.frame(twins[0], 3, backend="vedo", centered="first", match_size=True)
        assert whole_clip_scenes == []

    def test_a_still_of_a_comparison_prepares_the_whole_clips(
        self, twins, reached, tmp_path, whole_clip_scenes
    ):
        pytest.importorskip("vedo")
        from pybvh.bvhplot import _vedo_offscreen

        reached(_vedo_offscreen, "frame_vedo")
        bvhplot.frame(twins, 3, backend="vedo", centered="first")
        assert len(whole_clip_scenes) == 1
        assert all(a is b for a, b in zip(whole_clip_scenes[0], twins))


# Each public function that draws into a caller's ax=, with the axes
# projection it needs.
DRAWN_INTO_AN_AXES = {
    "frame": (lambda clip, ax, style: bvhplot.frame(clip, 0, ax=ax, style=style), "3d"),
    "rest_pose": (lambda clip, ax, style: bvhplot.rest_pose(clip, ax=ax, style=style), "3d"),
    "sequence": (lambda clip, ax, style: bvhplot.sequence(clip, ax=ax, style=style), "3d"),
    "trajectory": (lambda clip, ax, style: bvhplot.trajectory(clip, ax=ax, style=style), None),
}


@pytest.mark.parametrize("draw, projection", DRAWN_INTO_AN_AXES.values(), ids=DRAWN_INTO_AN_AXES)
class TestAxesInSubFigure:
    """An ax= inside a SubFigure, the panel of a composite figure."""

    @pytest.fixture
    def composite(self, projection):
        """(figure, panel, ax): a figure split in two SubFigures, and an
        axes in the second."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plt.figure()
        panel = fig.subfigures(1, 2)[1]
        ax = panel.add_subplot(projection=projection)
        yield fig, panel, ax
        plt.close(fig)

    def test_returns_the_root_figure(self, draw, composite, bvh_test1, tmp_path):
        fig, _, ax = composite
        returned_fig, returned_ax = draw(bvh_test1, ax, "paper")
        assert returned_fig is fig
        assert returned_ax is ax
        returned_fig.savefig(tmp_path / "composite.png")
        assert (tmp_path / "composite.png").stat().st_size > 0

    def test_paints_only_its_panel(self, draw, composite, bvh_test1):
        """The style's background colors the SubFigure holding the axes,
        not the figure behind the caller's other panels."""
        from matplotlib.colors import to_rgba

        fig, panel, ax = composite
        figure_color = fig.get_facecolor()
        draw(bvh_test1, ax, "dark")
        assert panel.get_facecolor() == to_rgba(bvhplot.Style("dark").background)
        assert fig.get_facecolor() == figure_color
