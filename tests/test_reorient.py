"""Tests for world_up features: load-time parameter, property validation,
reorient_world_up, reorient_rest_up, reorient_rest_forward, warning toggle.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from synthetic_bvh import (
    make_disagreement_bvh,
    make_neg_z_up_bvh,
    make_pos_y_up_bvh,
    make_pos_y_up_rotating_bvh,
    make_pos_z_up_bvh,
)

from pybvh import Bvh, nodes_from_table, read_bvh_file, rotations, transforms
from pybvh.rotations import _axis_aligned_rotation
from pybvh.tools import _axis_to_vector

BVH_DIR = Path(__file__).parent.parent / "bvh_data"
EXAMPLE = str(BVH_DIR / "bvh_test1.bvh")
TEST3 = str(BVH_DIR / "bvh_test3.bvh")  # disagreement file


def _turn(from_ax, to_ax):
    """The axis-aligned rotation between two signed axis strings."""
    return _axis_aligned_rotation(_axis_to_vector(from_ax), _axis_to_vector(to_ax))


# ========================================================================
#  Feature 2: Property shape validation
# ========================================================================


class TestPropertyValidation:
    def test_root_pos_rejects_1d(self):
        bvh = read_bvh_file(EXAMPLE)
        with pytest.raises(ValueError, match="root_pos"):
            bvh.root_pos = np.zeros(10)

    def test_root_pos_rejects_wrong_columns(self):
        bvh = read_bvh_file(EXAMPLE)
        with pytest.raises(ValueError, match="root_pos"):
            bvh.root_pos = np.zeros((10, 4))

    def test_joint_angles_rejects_2d(self):
        bvh = read_bvh_file(EXAMPLE)
        with pytest.raises(ValueError, match="joint_angles"):
            bvh.joint_angles = np.zeros((10, 3))

    def test_joint_angles_rejects_wrong_last_dim(self):
        bvh = read_bvh_file(EXAMPLE)
        with pytest.raises(ValueError, match="joint_angles"):
            bvh.joint_angles = np.zeros((10, 24, 2))

    def test_valid_shapes_accepted(self):
        bvh = read_bvh_file(EXAMPLE)
        bvh.root_pos = np.zeros((5, 3))
        bvh.joint_angles = np.zeros((5, 24, 3))
        assert bvh.frame_count == 5

    def test_empty_shapes_accepted(self):
        bvh = Bvh()
        assert bvh.root_pos.shape == (0, 3)
        assert bvh.joint_angles.shape == (0, 0, 3)


# ========================================================================
#  Features 1+6: world_up parameter + warning toggle
# ========================================================================


class TestWorldUpParameter:
    def test_read_bvh_file_with_world_up(self):
        bvh = read_bvh_file(EXAMPLE, world_up="+y")
        assert bvh.world_up == "+y"

    def test_read_bvh_file_auto_default(self):
        bvh = read_bvh_file(EXAMPLE)
        assert bvh.world_up in {"+x", "-x", "+y", "-y", "+z", "-z"}

    def test_read_bvh_file_invalid_raises(self):
        with pytest.raises(ValueError):
            read_bvh_file(EXAMPLE, world_up="bad")

    def test_read_bvh_file_override_suppresses_warning(self):
        """Loading bvh_test3 with explicit world_up should not warn."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            read_bvh_file(TEST3, world_up="+z")
            user_warns = [
                x
                for x in w
                if issubclass(x.category, UserWarning) and "world up" in str(x.message).lower()
            ]
            assert len(user_warns) == 0

    def test_read_bvh_file_auto_warns_on_disagreement(self):
        """Loading bvh_test3 without override should warn."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            read_bvh_file(TEST3)
            user_warns = [
                x
                for x in w
                if issubclass(x.category, UserWarning) and "world up" in str(x.message).lower()
            ]
            assert len(user_warns) > 0

    def test_warn_toggle_suppresses(self):
        """warn_on_world_up_disagreement=False suppresses the warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            read_bvh_file(TEST3, warn_on_world_up_disagreement=False)
            user_warns = [
                x
                for x in w
                if issubclass(x.category, UserWarning) and "world up" in str(x.message).lower()
            ]
            assert len(user_warns) == 0

    def test_bvh_init_world_up_param(self):
        bvh = Bvh(world_up="+y")
        assert bvh.world_up == "+y"

    def test_read_bvh_directory_world_up(self):
        from pybvh import read_bvh_directory

        clips = read_bvh_directory(BVH_DIR, world_up="+y")
        assert all(c.world_up == "+y" for c in clips)


class TestWorldUpSetterMovesTheFloor:
    """The scene ground is measured along world up, so setting world up
    rebuilds it (bug 1 of #112: v0.9.0 kept the old axis's floor)."""

    def test_floor_height_after_setting_world_up_is_the_new_axis_floor(self):
        bvh = read_bvh_file(EXAMPLE)
        _ = bvh.floor_height
        bvh.world_up = "+x"
        loaded_with_x_up = read_bvh_file(EXAMPLE, world_up="+x")
        assert bvh.floor_height == pytest.approx(loaded_with_x_up.floor_height)

    def test_clearing_the_override_gives_back_the_guessed_axis_floor(self):
        bvh = read_bvh_file(EXAMPLE, world_up="+x")
        _ = bvh.floor_height
        bvh.world_up = "auto"
        guessed = read_bvh_file(EXAMPLE)
        assert bvh.floor_height == pytest.approx(guessed.floor_height)

    def test_only_the_floor_is_computed_again(self, computations):
        bvh = read_bvh_file(EXAMPLE)
        _ = bvh.floor_height
        before = (computations.fk, computations.world_up, computations.floor_height)
        bvh.world_up = "+x"
        _ = bvh.floor_height
        _ = bvh.node_positions()
        bvh.world_up = "auto"
        _ = bvh.world_up
        assert (computations.fk, computations.world_up, computations.floor_height) == (
            before[0],
            before[1],
            before[2] + 1,
        )


# ========================================================================
#  The load-time warning preference outlives the load
# ========================================================================


def _disagreement_warnings(caught: list[warnings.WarningMessage]) -> int:
    return sum(1 for x in caught if "rest pose suggests world up" in str(x.message).lower())


def _guess_again(bvh):
    """Assign the clip's motion back, which makes it guess its world up
    again from its own first frame, and read the guess."""
    bvh.joint_angles = bvh.joint_angles
    return bvh.world_up


class TestWorldUpWarningPreference:
    """``warn_on_world_up_disagreement`` holds for the clip's lifetime.

    bvh_test3's rest pose and first frame disagree on world up, so every
    fresh guess on it is a chance to warn. A derived clip inherits the
    guess instead of making one, so the tests on derived clips assign
    their motion first (``_guess_again``), which makes them guess.
    """

    @pytest.fixture
    def silenced(self):
        return read_bvh_file(TEST3, warn_on_world_up_disagreement=False)

    def test_slice_stays_silent(self, silenced):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _guess_again(silenced[0:10])
        assert _disagreement_warnings(w) == 0

    def test_copy_stays_silent(self, silenced):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            clone = silenced.copy()
            clone.joint_angles = clone.joint_angles.copy()  # drops the cache
            _ = clone.world_up
        assert _disagreement_warnings(w) == 0

    def test_scale_stays_silent(self, silenced):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _guess_again(silenced.scale(2.0))
        assert _disagreement_warnings(w) == 0

    def test_mirror_stays_silent(self, silenced):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _guess_again(silenced.mirror())
        assert _disagreement_warnings(w) == 0

    def test_inplace_transform_stays_silent(self, silenced):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            silenced.translate_root(np.array([1.0, 0.0, 0.0]), inplace=True)
            _guess_again(silenced)
        assert _disagreement_warnings(w) == 0

    def test_extract_joints_stays_silent(self, silenced):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            spine = silenced.extract_joints(
                ["Hips", "Spine", "Spine1", "Spine2", "Spine3", "Neck", "Head"]
            )
            _guess_again(spine)
        assert _disagreement_warnings(w) == 0

    def test_world_up_inferred_stays_silent(self, silenced):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            assert silenced.world_up_inferred == "+z"
        assert _disagreement_warnings(w) == 0

    def test_concatenation_follows_the_left_clip(self, silenced):
        with pytest.warns(UserWarning, match="Rest pose suggests world up"):
            loud = read_bvh_file(TEST3)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _guess_again(silenced + loud)
            assert _disagreement_warnings(w) == 0
            _guess_again(loud + silenced)
            assert _disagreement_warnings(w) == 1

    def test_clip_rebuilt_from_its_data_starts_from_the_default(self, silenced):
        pd = pytest.importorskip("pandas")
        frame = pd.DataFrame(silenced.to_df_dict())
        with pytest.warns(UserWarning, match="Rest pose suggests world up"):
            Bvh.from_df(silenced.nodes, frame)

    def test_the_constructor_takes_the_readers_name(self, silenced):
        def build(**flag):
            return Bvh(
                nodes=nodes_from_table(silenced.to_node_table()),
                root_pos=silenced.root_pos,
                joint_angles=silenced.joint_angles,
                frame_time=silenced.frame_time,
                **flag,
            )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            build(warn_on_world_up_disagreement=False)
        assert _disagreement_warnings(w) == 0
        with pytest.warns(UserWarning, match="Rest pose suggests world up"):
            build()

    def test_the_warning_names_the_joint_the_guess_reads(self):
        # The guess reads the root's direction to the first joint named
        # head, else neck, chest or spine, not a head-to-hips direction.
        with pytest.warns(UserWarning, match="Rest pose suggests world up") as caught:
            read_bvh_file(TEST3)
        message = next(str(x.message) for x in caught if "Rest pose suggests" in str(x.message))
        assert (
            "the direction from the root to the head joint (else the neck, chest or spine)"
            in message
        )

    def test_read_bvh_directory_loads_silently(self):
        from pybvh import read_bvh_directory

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            clips = read_bvh_directory(BVH_DIR, warn_on_world_up_disagreement=False)
            assert any(c.source_path == TEST3 for c in clips)
            for clip in clips:
                _guess_again(clip[0:10])
        assert _disagreement_warnings(w) == 0

    def test_default_warns_once_per_guess(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bvh = read_bvh_file(TEST3)
            _ = bvh.world_up  # guessed at load: no second guess
            assert _disagreement_warnings(w) == 1
            _ = bvh[0:10].world_up  # inherited
            bvh.translate_root(np.array([1.0, 0.0, 0.0]), inplace=True)
            _ = bvh.world_up  # kept by pybvh's own edits
            assert _disagreement_warnings(w) == 1
            _guess_again(bvh)
            _ = bvh.world_up
            assert _disagreement_warnings(w) == 2

    def test_each_inference_site_keeps_its_own_warning_location(self):
        # Under the "default" filter Python shows a warning once per
        # location, so warnings attributed to one shared line would
        # collapse into the first; each inference site must be its own.
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("default")
            bvh = read_bvh_file(TEST3)
            _guess_again(bvh)
            _ = bvh.world_up_inferred
        locations = {
            (x.filename, x.lineno)
            for x in w
            if "rest pose suggests world up" in str(x.message).lower()
        }
        assert len(locations) == 3


# ========================================================================
#  The rest-up fallback warns when the root rotation contradicts it
# ========================================================================


def _bend_spine_at_first_frame(bvh, degrees=50.0):
    """Bend *bvh*'s spine at frame 0 about the rest pose's x axis, so that
    the head's direction from the root no longer decides world up while
    the root keeps its rotation; the next read of world up guesses again.

    50° leaves the largest component of the head's direction under twice
    the second on bvh_test1 and bvh_test3, whose rest left is along x.
    """
    joints = [node for node in bvh.nodes if not node.is_end_site()]
    spine = [joint.name for joint in joints].index("Spine")
    order = joints[spine].rot_channels
    bend = rotations.axisangle_to_rotmat(np.radians(degrees) * np.array([1.0, 0.0, 0.0]))
    angles = bvh.joint_angles.copy()
    angles[0, spine] = rotations.rotmat_to_euler(
        rotations.euler_to_rotmat(angles[0, spine], order) @ bend, order
    )
    bvh.joint_angles = angles
    return bvh


def _world_up_warnings(caught: list[warnings.WarningMessage]) -> list[str]:
    return [str(x.message) for x in caught if "world up" in str(x.message).lower()]


class TestRestUpFallbackWarning:
    """When frame 0 cannot decide, rest up stands in; if the root rotation
    of frame 0 carries rest up onto another axis, the rest pose may be
    authored in another convention than the motion, and pybvh says so
    while keeping the guess."""

    @staticmethod
    def _load(path, **flag):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return read_bvh_file(path, **flag)

    def test_a_rest_pose_in_another_convention_warns_and_keeps_the_guess(self):
        # bvh_test3's rest pose is +y up, its motion +z up.
        bvh = _bend_spine_at_first_frame(self._load(TEST3))
        with pytest.warns(UserWarning, match="another convention") as caught:
            assert bvh.world_up == "+y"
        assert bvh.orientation_report.world_up_source == "rest_up"
        message = next(m for m in _world_up_warnings(caught) if "another convention" in m)
        assert "'+y'" in message and "'+z'" in message

    def test_a_rest_pose_in_the_motions_convention_stays_silent(self):
        bvh = _bend_spine_at_first_frame(self._load(EXAMPLE))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            assert bvh.world_up == "+z"
        assert bvh.orientation_report.world_up_source == "rest_up"
        assert _world_up_warnings(w) == []

    def test_the_load_flag_silences_it(self):
        bvh = _bend_spine_at_first_frame(self._load(TEST3, warn_on_world_up_disagreement=False))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            assert bvh.world_up == "+y"
        assert _world_up_warnings(w) == []

    def test_world_up_inferred_warns_too(self):
        bvh = _bend_spine_at_first_frame(self._load(TEST3))
        bvh.world_up = "+z"
        with pytest.warns(UserWarning, match="another convention"):
            assert bvh.world_up_inferred == "+y"

    def test_a_root_that_does_not_decide_stays_silent(self):
        # bvh_test3's root turned 35° further about x at frame 0: the head
        # and the carried rest up both lie about halfway between y and z.
        bvh = self._load(TEST3)
        order = bvh.nodes[0].rot_channels
        tilt = rotations.axisangle_to_rotmat(np.radians(35.0) * np.array([1.0, 0.0, 0.0]))
        angles = bvh.joint_angles.copy()
        angles[0, 0] = rotations.rotmat_to_euler(
            tilt @ rotations.euler_to_rotmat(angles[0, 0], order), order
        )
        bvh.joint_angles = angles
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            assert bvh.world_up == "+y"
        assert bvh.orientation_report.world_up_source == "rest_up"
        assert _world_up_warnings(w) == []

    def test_a_mapping_that_changes_the_fallback_warns(self):
        # Every joint renamed J<index>: no up marker, so world up falls
        # back on rest up, read from the spread of the T-pose without a
        # mapping and from the legs with one.
        source = self._load(TEST3)
        table = source.to_node_table()
        renamed = {row["name"]: f"J{index}" for index, row in enumerate(table)}
        for row in table:
            row["name"] = renamed[row["name"]]
        mapping = {renamed[a]: renamed[b] for a, b in source.lr_mapping.items()}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bvh = Bvh(
                nodes=nodes_from_table(table),
                root_pos=source.root_pos,
                joint_angles=source.joint_angles,
                frame_time=source.frame_time,
            )
            assert bvh.world_up == "-x"
        assert _world_up_warnings(w) == []
        bvh.lr_mapping = mapping
        with pytest.warns(UserWarning, match="another convention"):
            assert bvh.world_up == "+y"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bvh.lr_mapping = mapping
            _ = bvh.world_up  # the same guess again: no second warning
        assert _world_up_warnings(w) == []

    def test_a_first_frame_holding_nan_falls_back_on_rest_up(self):
        bvh = self._load(EXAMPLE)
        root_pos = bvh.root_pos.copy()
        root_pos[0, 0] = np.nan
        bvh.root_pos = root_pos
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert bvh.world_up == "+z"
        assert bvh.orientation_report.world_up_source == "rest_up"


def test_the_disagreement_warning_names_both_causes():
    # A rest pose in another convention than the motion, or a first frame
    # that is not upright: on the maintainer's corpus the second is the
    # common one.
    with pytest.warns(UserWarning, match="Rest pose suggests world up") as caught:
        read_bvh_file(TEST3)
    message = next(str(x.message) for x in caught if "Rest pose suggests" in str(x.message))
    assert "another convention" in message
    assert "not upright" in message


# ========================================================================
#  Helper: rotations._axis_aligned_rotation
# ========================================================================


class TestAxisAlignedRotation:
    @pytest.mark.parametrize("about", ["+y", "-y"])
    def test_a_half_turn_about_the_axis_given(self, about):
        R = _axis_aligned_rotation(
            _axis_to_vector("+z"), _axis_to_vector("-z"), about=_axis_to_vector(about)
        )
        npt.assert_array_equal(R, np.diag([-1.0, 1.0, -1.0]))

    @pytest.mark.parametrize(
        "from_ax,to_ax",
        [
            ("+y", "+z"),
            ("+z", "+y"),
            ("+y", "-y"),
            ("+z", "-z"),
            ("+y", "+x"),
            ("-z", "+y"),
            ("+x", "-x"),
            ("-y", "+z"),
            ("+x", "+z"),
            ("-x", "-y"),
        ],
    )
    def test_maps_axis_correctly(self, from_ax, to_ax):
        R = _turn(from_ax, to_ax)
        result = R @ _axis_to_vector(from_ax)
        expected = _axis_to_vector(to_ax)
        npt.assert_allclose(result, expected, atol=1e-14)

    @pytest.mark.parametrize("ax", ["+x", "+y", "+z", "-x", "-y", "-z"])
    def test_identity_for_same_axis(self, ax):
        R = _turn(ax, ax)
        npt.assert_allclose(R, np.eye(3), atol=1e-14)

    @pytest.mark.parametrize(
        "from_ax,to_ax",
        [
            ("+y", "+z"),
            ("+y", "-y"),
            ("+x", "+z"),
        ],
    )
    def test_det_is_one(self, from_ax, to_ax):
        R = _turn(from_ax, to_ax)
        assert abs(np.linalg.det(R) - 1.0) < 1e-14

    @pytest.mark.parametrize(
        "from_ax,to_ax",
        [
            ("+y", "+z"),
            ("+y", "-y"),
            ("-z", "+x"),
        ],
    )
    def test_entries_are_exact_integers(self, from_ax, to_ax):
        R = _turn(from_ax, to_ax)
        npt.assert_array_equal(R, R.astype(int))


# ========================================================================
#  Feature 3: reorient_world_up
# ========================================================================


class TestReorientWorldUp:
    def test_noop_same_axis(self):
        bvh = make_pos_y_up_bvh()
        result = transforms.reorient_world_up(bvh, "+y")
        npt.assert_allclose(result.root_pos, bvh.root_pos)
        npt.assert_allclose(result.joint_angles, bvh.joint_angles)

    def test_metadata_updated(self):
        bvh = make_pos_y_up_bvh()
        result = transforms.reorient_world_up(bvh, "+z")
        assert result.world_up == "+z"

    def test_fk_positions_rotated(self):
        """FK positions in new system = R @ FK positions in old system."""
        bvh = make_pos_y_up_bvh()
        R = _turn("+y", "+z")
        coords_old = bvh.node_positions()
        result = transforms.reorient_world_up(bvh, "+z")
        coords_new = result.node_positions()
        for f in range(coords_old.shape[0]):
            expected = (R @ coords_old[f].T).T
            npt.assert_allclose(coords_new[f], expected, atol=1e-10)

    def test_fk_positions_rotated_with_animation(self):
        """Same test with rotating BVH (non-zero root angles)."""
        bvh = make_pos_y_up_rotating_bvh()
        R = _turn("+y", "+z")
        coords_old = bvh.node_positions()
        result = transforms.reorient_world_up(bvh, "+z")
        coords_new = result.node_positions()
        for f in range(coords_old.shape[0]):
            expected = (R @ coords_old[f].T).T
            npt.assert_allclose(coords_new[f], expected, atol=1e-10)

    def test_roundtrip_fk_exact(self):
        """Y->Z->Y round-trip: FK positions must match original."""
        bvh = make_pos_y_up_rotating_bvh()
        coords_before = bvh.node_positions()
        bvh_z = transforms.reorient_world_up(bvh, "+z")
        bvh_back = transforms.reorient_world_up(bvh_z, "+y")
        coords_after = bvh_back.node_positions()
        npt.assert_allclose(coords_before, coords_after, atol=1e-10)

    def test_180_degree_rotation(self):
        """+Y -> -Y: FK Y and X should be negated."""
        bvh = make_pos_y_up_bvh()
        R = _turn("+y", "-y")
        coords_old = bvh.node_positions()
        result = transforms.reorient_world_up(bvh, "-y")
        coords_new = result.node_positions()
        for f in range(coords_old.shape[0]):
            expected = (R @ coords_old[f].T).T
            npt.assert_allclose(coords_new[f], expected, atol=1e-10)

    def test_negative_axes(self):
        """-Z -> +Y reorientation."""
        bvh = make_neg_z_up_bvh()
        R = _turn("-z", "+y")
        coords_old = bvh.node_positions()
        result = transforms.reorient_world_up(bvh, "+y")
        coords_new = result.node_positions()
        for f in range(coords_old.shape[0]):
            expected = (R @ coords_old[f].T).T
            npt.assert_allclose(coords_new[f], expected, atol=1e-10)

    def test_inplace(self):
        bvh = make_pos_y_up_bvh()
        coords_before = bvh.node_positions()
        R = _turn("+y", "+z")
        result = transforms.reorient_world_up(bvh, "+z", inplace=True)
        assert result is None
        assert bvh.world_up == "+z"
        coords_after = bvh.node_positions()
        for f in range(coords_before.shape[0]):
            expected = (R @ coords_before[f].T).T
            npt.assert_allclose(coords_after[f], expected, atol=1e-10)

    def test_write_read_roundtrip(self, tmp_path):
        """Reorient, write, read back, compare FK."""
        bvh = make_pos_y_up_rotating_bvh()
        reoriented = transforms.reorient_world_up(bvh, "+z")
        coords_expected = reoriented.node_positions()
        p = tmp_path / "reoriented.bvh"
        reoriented.write(str(p), verbose=False)
        bvh2 = read_bvh_file(str(p), world_up="+z")
        coords_loaded = bvh2.node_positions()
        npt.assert_allclose(coords_loaded, coords_expected, atol=1e-4)

    def test_real_file(self):
        """Reorient a real BVH file (+z -> +y)."""
        bvh = read_bvh_file(EXAMPLE)
        R = _turn(bvh.world_up, "+y")
        coords_old = bvh.node_positions()
        result = transforms.reorient_world_up(bvh, "+y")
        coords_new = result.node_positions()
        for f in range(coords_old.shape[0]):
            expected = (R @ coords_old[f].T).T
            npt.assert_allclose(coords_new[f], expected, atol=1e-10)

    def test_bvh_method_wrapper(self):
        bvh = make_pos_y_up_bvh()
        result = bvh.reorient_world_up("+z")
        assert result.world_up == "+z"


def _reorient(bvh, new_up, inplace):
    """Reorient ``bvh``, in place or not, and return the reoriented clip."""
    result = bvh.reorient_world_up(new_up, inplace=inplace)
    return bvh if inplace else result


@pytest.mark.parametrize("inplace", [False, True], ids=["copy", "inplace"])
class TestReorientWorldUpRotatesTheGuess:
    """Rotating the world turns the guessed up with it (round 1 of the
    review of #113): clearing the override after ``reorient_world_up``
    gives the axis the reoriented frames show, not the one from before
    the rotation."""

    def test_clearing_the_override_gives_the_rotated_guess(self, inplace):
        bvh = read_bvh_file(EXAMPLE)  # guesses '+z'
        reoriented = _reorient(bvh, "+y", inplace)
        reoriented.world_up = "auto"
        assert reoriented.world_up == "+y"
        assert reoriented.world_up_inferred == "+y"

    def test_the_floor_follows_the_rotated_guess(self, inplace):
        bvh = read_bvh_file(EXAMPLE)
        reoriented = _reorient(bvh, "+y", inplace)
        floor_along_new_up = reoriented.floor_height
        reoriented.world_up = "auto"
        assert reoriented.floor_height == pytest.approx(floor_along_new_up)

    def test_a_guess_under_an_override_turns_too(self, inplace):
        # The guess '+z' is made at load, then overridden by a wrong '+x';
        # turning '+x' onto '+z' carries the guessed '+z' onto '-x'.
        bvh = read_bvh_file(EXAMPLE)
        bvh.world_up = "+x"
        reoriented = _reorient(bvh, "+z", inplace)
        reoriented.world_up = "auto"
        assert reoriented.world_up == "-x"
        assert reoriented.world_up_inferred == "-x"

    def test_an_override_with_no_guess_yet_guesses_from_the_new_frames(self, inplace):
        bvh = read_bvh_file(EXAMPLE, world_up="+x")  # nothing guessed
        reoriented = _reorient(bvh, "+z", inplace)
        reoriented.world_up = "auto"
        assert reoriented.world_up == "-x"
        assert reoriented.world_up_inferred == "-x"

    def test_the_guess_is_not_made_again(self, inplace, computations):
        bvh = read_bvh_file(EXAMPLE)
        reoriented = _reorient(bvh, "+y", inplace)
        guesses = computations.world_up
        reoriented.world_up = "auto"
        _ = reoriented.world_up
        assert computations.world_up == guesses


# ========================================================================
#  Feature 4: reorient_rest_up
# ========================================================================


class TestReorientRestUp:
    def test_fk_invariance(self):
        """FK positions must be identical before and after."""
        bvh = make_pos_z_up_bvh()
        coords_before = bvh.node_positions()
        result = transforms.reorient_rest_up(bvh, "+y")
        coords_after = result.node_positions()
        npt.assert_allclose(coords_before, coords_after, atol=1e-10)

    def test_fk_invariance_with_rotation(self):
        """Same with rotating BVH."""
        bvh = make_pos_y_up_rotating_bvh()
        coords_before = bvh.node_positions()
        result = transforms.reorient_rest_up(bvh, "+z")
        coords_after = result.node_positions()
        npt.assert_allclose(coords_before, coords_after, atol=1e-10)

    def test_rest_up_changes(self):
        bvh = make_pos_z_up_bvh()
        result = transforms.reorient_rest_up(bvh, "+y")
        assert result.rest_up == "+y"

    def test_root_pos_unchanged(self):
        bvh = make_pos_z_up_bvh()
        original_pos = bvh.root_pos.copy()
        result = transforms.reorient_rest_up(bvh, "+y")
        npt.assert_allclose(result.root_pos, original_pos)

    def test_noop_same_axis(self):
        bvh = make_pos_y_up_bvh()
        result = transforms.reorient_rest_up(bvh, "+y")
        npt.assert_allclose(result.joint_angles, bvh.joint_angles)

    def test_roundtrip_fk(self):
        """Z->Y->Z round-trip: FK positions must match."""
        bvh = make_pos_y_up_rotating_bvh()
        coords_before = bvh.node_positions()
        bvh2 = transforms.reorient_rest_up(bvh, "+z")
        bvh3 = transforms.reorient_rest_up(bvh2, "+y")
        coords_after = bvh3.node_positions()
        npt.assert_allclose(coords_before, coords_after, atol=1e-10)

    def test_real_file(self):
        """Reorient a real BVH file's rest pose."""
        bvh = read_bvh_file(EXAMPLE)
        coords_before = bvh.node_positions()
        result = transforms.reorient_rest_up(bvh, "+y")
        coords_after = result.node_positions()
        npt.assert_allclose(coords_before, coords_after, atol=1e-10)
        assert result.rest_up == "+y"

    def test_inplace(self):
        bvh = make_pos_z_up_bvh()
        coords_before = bvh.node_positions()
        result = transforms.reorient_rest_up(bvh, "+y", inplace=True)
        assert result is None
        coords_after = bvh.node_positions()
        npt.assert_allclose(coords_before, coords_after, atol=1e-10)

    def test_bvh_method_wrapper(self):
        bvh = make_pos_z_up_bvh()
        coords_before = bvh.node_positions()
        result = bvh.reorient_rest_up("+y")
        coords_after = result.node_positions()
        npt.assert_allclose(coords_before, coords_after, atol=1e-10)


# ========================================================================
#  Feature 5: reorient_rest_forward
# ========================================================================


class TestReorientRestForward:
    def test_fk_invariance(self):
        bvh = make_pos_y_up_bvh()
        coords_before = bvh.node_positions()
        # Get current forward, pick a different one
        current_fwd = bvh.rest_forward
        # Pick target that's perpendicular to up and different from current
        candidates = ["+x", "-x", "+z", "-z"]
        target = [c for c in candidates if c != current_fwd and c[1] != bvh.rest_up[1]][0]
        result = transforms.reorient_rest_forward(bvh, target)
        coords_after = result.node_positions()
        npt.assert_allclose(coords_before, coords_after, atol=1e-10)

    def test_fk_invariance_with_rotation(self):
        bvh = make_pos_y_up_rotating_bvh()
        coords_before = bvh.node_positions()
        current_fwd = bvh.rest_forward
        candidates = ["+x", "-x", "+z", "-z"]
        target = [c for c in candidates if c != current_fwd and c[1] != bvh.rest_up[1]][0]
        result = transforms.reorient_rest_forward(bvh, target)
        coords_after = result.node_positions()
        npt.assert_allclose(coords_before, coords_after, atol=1e-10)

    def test_parallel_axis_raises(self):
        bvh = make_pos_y_up_bvh()
        with pytest.raises(ValueError, match="parallel"):
            transforms.reorient_rest_forward(bvh, "+y")

    @pytest.mark.parametrize("target", ["+x", "-x", "+z", "-z"])
    def test_turns_about_rest_up(self, target):
        """On bvh_test3 rest up is +y and world up +z: the rest pose turns
        about its own up, whatever world up is, and the posed positions
        stay where they were."""
        bvh = read_bvh_file(TEST3, warn_on_world_up_disagreement=False)
        assert (bvh.rest_up, bvh.world_up, bvh.rest_forward) == ("+y", "+z", "+z")
        result = transforms.reorient_rest_forward(bvh, target)
        assert result.rest_up == "+y"
        assert result.rest_forward == target
        npt.assert_allclose(result.node_positions(), bvh.node_positions(), atol=1e-9)

    def test_refuses_a_target_along_rest_up_not_world_up(self):
        bvh = read_bvh_file(TEST3, warn_on_world_up_disagreement=False)
        with pytest.raises(ValueError, match="rest_up"):
            transforms.reorient_rest_forward(bvh, "-y")

    def test_inplace(self):
        bvh = make_pos_y_up_bvh()
        current_fwd = bvh.rest_forward
        candidates = ["+x", "-x", "+z", "-z"]
        target = [c for c in candidates if c != current_fwd and c[1] != bvh.rest_up[1]][0]
        coords_before = bvh.node_positions()
        result = transforms.reorient_rest_forward(bvh, target, inplace=True)
        assert result is None
        coords_after = bvh.node_positions()
        npt.assert_allclose(coords_before, coords_after, atol=1e-10)

    def test_bvh_method_wrapper(self):
        bvh = make_pos_y_up_bvh()
        current_fwd = bvh.rest_forward
        candidates = ["+x", "-x", "+z", "-z"]
        target = [c for c in candidates if c != current_fwd and c[1] != bvh.rest_up[1]][0]
        coords_before = bvh.node_positions()
        result = bvh.reorient_rest_forward(target)
        coords_after = result.node_positions()
        npt.assert_allclose(coords_before, coords_after, atol=1e-10)


# ========================================================================
#  Heterogeneous Euler order tests
# ========================================================================


class TestReorientHeterogeneousEulerOrders:
    """Verify reorient functions work when joints have different Euler orders."""

    def test_reorient_world_up_heterogeneous(self):
        from synthetic_bvh import make_heterogeneous_euler_bvh

        bvh = make_heterogeneous_euler_bvh()
        R = _turn(bvh.world_up, "+z")
        coords_old = bvh.node_positions()
        result = transforms.reorient_world_up(bvh, "+z")
        coords_new = result.node_positions()
        for f in range(coords_old.shape[0]):
            expected = (R @ coords_old[f].T).T
            npt.assert_allclose(coords_new[f], expected, atol=1e-10)

    def test_reorient_rest_up_heterogeneous(self):
        from synthetic_bvh import make_heterogeneous_euler_bvh

        bvh = make_heterogeneous_euler_bvh()
        coords_before = bvh.node_positions()
        result = transforms.reorient_rest_up(bvh, "+z")
        coords_after = result.node_positions()
        npt.assert_allclose(coords_before, coords_after, atol=1e-10)

    def test_reorient_world_up_roundtrip_heterogeneous(self):
        from synthetic_bvh import make_heterogeneous_euler_bvh

        bvh = make_heterogeneous_euler_bvh()
        coords_before = bvh.node_positions()
        bvh2 = transforms.reorient_world_up(bvh, "+z")
        bvh3 = transforms.reorient_world_up(bvh2, "+y")
        coords_after = bvh3.node_positions()
        npt.assert_allclose(coords_before, coords_after, atol=1e-10)


# ========================================================================
#  Disagreement fixture tests
# ========================================================================


class TestDisagreementFixture:
    def test_disagreement_bvh_constructed(self):
        bvh = make_disagreement_bvh()
        assert bvh.frame_count > 0
        # Rest-pose topology should be +Z up (offsets along Z)
        rest_up = bvh.rest_up
        assert rest_up[1] == "z"
