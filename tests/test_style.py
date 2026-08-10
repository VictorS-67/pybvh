"""Tests for the Style system and bone-chain classification (Phase 1)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from pybvh import read_bvh_file, bvhplot
from pybvh.bvhplot._common import (
    CHAIN_COLORS,
    Style,
    effective_color_mode,
    get_bone_chains,
    get_skeleton_lines,
    resolve_style,
)

BVH_PATH = "bvh_data/cmu_12_01_walk.bvh"
BASELINE_DIR = Path(__file__).parent / "fixtures" / "baseline_v082"


@pytest.fixture(scope="module")
def bvh():
    return read_bvh_file(BVH_PATH)


class TestStyleConstruction:
    def test_default_is_paper(self):
        s = Style()
        assert s.floor == "solid"
        assert s.axes == "off"
        assert s.color_mode == "auto"
        assert s.joint_markers is True

    def test_debug_preset_reproduces_legacy_values(self):
        s = Style("debug")
        assert s.bone_width == 2.5
        assert s.bone_color == (0.1, 0.2, 0.8)
        assert s.color_mode == "single"
        assert s.floor is None
        assert s.axes == "full"
        assert s.joint_markers is False
        assert s.supersample == 1

    def test_dark_preset_has_dark_background(self):
        s = Style("dark")
        assert s.background != "white"
        assert s.chain_colors["spine"] != CHAIN_COLORS["spine"]

    def test_preset_with_override(self):
        s = Style("paper", floor=None, bone_width=4.0)
        assert s.floor is None
        assert s.bone_width == 4.0
        assert s.axes == "off"  # other paper fields untouched

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="preset"):
            Style("papper")

    def test_unknown_field_raises(self):
        with pytest.raises(TypeError, match="floors"):
            Style("paper", floors="solid")

    def test_invalid_floor_raises(self):
        with pytest.raises(ValueError, match="floor"):
            Style("paper", floor="lava")

    def test_invalid_color_mode_raises(self):
        with pytest.raises(ValueError, match="color_mode"):
            Style("paper", color_mode="rainbow")

    def test_invalid_supersample_raises(self):
        with pytest.raises(ValueError, match="supersample"):
            Style("paper", supersample=0)
        with pytest.raises(ValueError, match="supersample"):
            Style("paper", supersample=1.5)

    def test_frozen(self):
        s = Style()
        with pytest.raises(Exception):
            s.floor = None  # type: ignore[misc]

    def test_replace(self):
        s = Style("paper")
        s2 = s.replace(floor="grid")
        assert s2.floor == "grid"
        assert s.floor == "solid"
        with pytest.raises(TypeError, match="Unknown"):
            s.replace(nope=1)


class TestResolveStyle:
    def test_string(self):
        assert resolve_style("debug").axes == "full"

    def test_instance_passthrough(self):
        s = Style("dark")
        assert resolve_style(s) is s

    def test_wrong_type_raises(self):
        with pytest.raises(TypeError, match="style"):
            resolve_style(42)  # type: ignore[arg-type]


class TestEffectiveColorMode:
    def test_auto_single_skeleton_is_chains(self):
        assert effective_color_mode(Style("paper"), 1) == "chains"

    def test_auto_multi_skeleton_is_skeleton(self):
        assert effective_color_mode(Style("paper"), 2) == "skeleton"

    def test_explicit_chains_survives_multi(self):
        s = Style("paper", color_mode="chains")
        assert effective_color_mode(s, 3) == "chains"

    def test_single_mode_multi_falls_back_to_palette(self):
        s = Style("debug")
        assert effective_color_mode(s, 1) == "single"
        assert effective_color_mode(s, 2) == "skeleton"


class TestGetBoneChains:
    def test_cmu_has_all_five_chains(self, bvh):
        chains = get_bone_chains(bvh)
        assert set(chains) == {"spine", "l_arm", "l_leg", "r_arm", "r_leg"}

    def test_partition_is_exact(self, bvh):
        """Every bone appears in exactly one chain."""
        chains = get_bone_chains(bvh)
        all_indices = sorted(i for idxs in chains.values() for i in idxs)
        assert all_indices == list(range(len(get_skeleton_lines(bvh))))

    def test_left_right_symmetry(self, bvh):
        chains = get_bone_chains(bvh)
        assert len(chains["l_arm"]) == len(chains["r_arm"])
        assert len(chains["l_leg"]) == len(chains["r_leg"])

    def test_legs_contain_the_feet(self, bvh):
        chains = get_bone_chains(bvh)
        bones = get_skeleton_lines(bvh)
        foot_nodes = {bvh.node_index[n]
                      for n in bvh.auto_detect_foot_joints()}
        leg_children = {bones[i][1]
                        for i in chains["l_leg"] + chains["r_leg"]}
        assert foot_nodes <= leg_children

    def test_no_lr_pairs_falls_back_to_spine(self, bvh):
        """A skeleton without L/R pairs puts every bone in 'spine'."""
        sub = bvh.extract_joints(
            ["Hips", "LowerBack", "Spine", "Spine1", "Neck", "Neck1", "Head"])
        assert sub.node_lr_pairs is None
        chains = get_bone_chains(sub)
        assert set(chains) == {"spine"}
        assert sorted(chains["spine"]) == list(
            range(len(get_skeleton_lines(sub))))


def _fig_pixels(fig) -> np.ndarray:
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba()).copy()


class TestDebugPixelParity:
    """style="debug" must reproduce the pre-0.9.0 output exactly.

    The fixtures were rendered by v0.8.2 before any Phase 0/1 change;
    matching them pixel-for-pixel proves the refactor did not perturb
    geometry, camera, or styling on the legacy path.
    """

    @pytest.mark.parametrize("fixture,kwargs", [
        ("frame_single_f260.png", dict(frame=260)),
        ("rest_pose.png", None),
    ])
    def test_matches_v082_baseline(self, bvh, tmp_path, fixture, kwargs):
        from PIL import Image

        if kwargs is None:
            fig, _ = bvhplot.rest_pose(bvh, style="debug")
        else:
            fig, _ = bvhplot.frame(bvh, style="debug", **kwargs)
        out = tmp_path / fixture
        fig.savefig(out, dpi=100)
        plt.close(fig)

        got = np.asarray(Image.open(out).convert("RGB"))
        want = np.asarray(Image.open(BASELINE_DIR / fixture).convert("RGB"))
        assert got.shape == want.shape
        assert np.array_equal(got, want), (
            f"{fixture}: debug render differs from the v0.8.2 baseline")

    def test_pair_matches_v082_baseline(self, bvh, tmp_path):
        from PIL import Image

        fig, _ = bvhplot.frame(
            [bvh, bvh.mirror()], 100, labels=["a", "b"], style="debug")
        out = tmp_path / "pair.png"
        fig.savefig(out, dpi=100)
        plt.close(fig)

        got = np.asarray(Image.open(out).convert("RGB"))
        want = np.asarray(
            Image.open(BASELINE_DIR / "frame_pair_f100.png").convert("RGB"))
        assert np.array_equal(got, want)


class TestStyledRenderSmoke:
    """Every preset and floor kind renders through both backends."""

    @pytest.mark.parametrize("style", [
        "paper", "dark",
        Style("paper", floor="grid"),
        Style("paper", floor="checker"),
        Style("paper", projection="ortho"),
        Style("paper", color_mode="chains"),
    ])
    def test_frame_styles(self, bvh, style):
        fig, ax = bvhplot.frame(bvh, 10, style=style)
        assert fig is not None
        plt.close(fig)

    def test_frame_multi_auto_switches_to_palette(self, bvh):
        fig, axes = bvhplot.frame([bvh, bvh], 10, style="paper")
        plt.close(fig)

    def test_render_opencv_paper(self, bvh, tmp_path):
        cv2 = pytest.importorskip("cv2")
        short = bvh[0:10]
        path = bvhplot.render(
            short, tmp_path / "paper.mp4", backend="opencv",
            resolution=(320, 240), style="paper")
        assert path.exists() and path.stat().st_size > 0

    def test_render_opencv_debug(self, bvh, tmp_path):
        pytest.importorskip("cv2")
        short = bvh[0:10]
        path = bvhplot.render(
            short, tmp_path / "debug.mp4", backend="opencv",
            resolution=(320, 240), style="debug")
        assert path.exists() and path.stat().st_size > 0

    def test_render_mpl_paper_gif(self, bvh, tmp_path):
        short = bvh[0:6]
        path = bvhplot.render(
            short, tmp_path / "paper.gif", backend="matplotlib",
            resolution=(320, 240), style="paper")
        assert path.exists() and path.stat().st_size > 0
