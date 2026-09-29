"""Tests for the Style system and bone-chain classification (Phase 1)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from pybvh import read_bvh_file, bvhplot
from pybvh.bvhplot._from_bvh import get_bone_chains, get_skeleton_lines
from pybvh.bvhplot._style import (
    CHAIN_COLORS,
    Style,
    effective_color_mode,
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

    def test_junction_bones_stay_in_spine(self, bvh):
        """Torso->limb connector bones (unpaired parent, paired child)
        belong to the spine chain: limbs start at the shoulder ball and
        hip socket. Regression for the arm-colored "vertebra" on rigs
        whose shoulder joints sit on the spine axis."""
        chains = get_bone_chains(bvh)
        bones = get_skeleton_lines(bvh)
        pairs = bvh.node_lr_pairs
        paired = {n for pair in pairs for n in pair}
        junction = {i for i, (p, c) in enumerate(bones)
                    if c in paired and p not in paired}
        assert junction, "walk skeleton must have torso->limb junctions"
        assert junction <= set(chains["spine"])
        limb_indices = {i for name, idxs in chains.items()
                        if name != "spine" for i in idxs}
        assert all(bones[i][0] in paired and bones[i][1] in paired
                   for i in limb_indices)

    def test_no_lr_pairs_falls_back_to_spine(self, bvh):
        """A skeleton without L/R pairs puts every bone in 'spine'."""
        sub = bvh.extract_joints(
            ["Hips", "LowerBack", "Spine", "Spine1", "Neck", "Neck1", "Head"])
        assert sub.node_lr_pairs is None
        chains = get_bone_chains(sub)
        assert set(chains) == {"spine"}
        assert sorted(chains["spine"]) == list(
            range(len(get_skeleton_lines(sub))))


def _mid_frame(path: Path) -> np.ndarray:
    """The middle frame of a rendered clip, as RGB."""
    from PIL import Image

    if path.suffix == ".gif":
        clip = Image.open(path)
        clip.seek(clip.n_frames // 2)
        return np.asarray(clip.convert("RGB"))

    import cv2
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_FRAMES,
                int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) // 2)
    ok, frame = capture.read()
    capture.release()
    assert ok, f"could not read a frame back from {path}"
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _fig_pixels(fig) -> np.ndarray:
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba()).copy()


def _text_mask(fig, shape, dpi, margin=2):
    """Pixels covered by the figure's visible text, at the given save dpi."""
    from matplotlib.text import Text

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    scale = dpi / fig.dpi
    height, width = shape[:2]
    mask = np.zeros((height, width), dtype=bool)
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text().strip():
            continue
        box = text.get_window_extent(renderer)
        x0 = max(int(np.floor(box.x0 * scale)) - margin, 0)
        x1 = min(int(np.ceil(box.x1 * scale)) + margin, width)
        # display y grows upward, image rows grow downward
        r0 = max(height - int(np.ceil(box.y1 * scale)) - margin, 0)
        r1 = min(height - int(np.floor(box.y0 * scale)) + margin, height)
        mask[r0:r1, x0:x1] = True
    return mask


def _assert_matches_v082(fig, fixture, tmp_path):
    from PIL import Image

    out = tmp_path / fixture
    fig.savefig(out, dpi=100)
    got = np.asarray(Image.open(out).convert("RGB"))
    want = np.asarray(Image.open(BASELINE_DIR / fixture).convert("RGB"))
    assert got.shape == want.shape
    text = _text_mask(fig, got.shape, dpi=100)
    plt.close(fig)

    assert text.mean() < 0.10, (
        "the text mask covers too much of the figure for the comparison "
        "to mean anything")
    differs = (got != want).any(axis=2) & ~text
    assert not differs.any(), (
        f"{fixture}: {differs.sum()} pixels outside text differ from the "
        f"v0.8.2 baseline")


class TestDebugPixelParity:
    """style="debug" must reproduce the pre-0.9.0 output exactly.

    The fixtures were rendered by v0.8.2 before any Phase 0/1 change;
    matching them pixel-for-pixel proves the refactor did not perturb
    geometry, camera, or styling on the legacy path.

    Every pixel is compared exactly except those inside the rectangles
    where matplotlib draws text. Glyph rasterization belongs to matplotlib
    and changes between its versions — 3.11 draws the same tick labels as
    3.9 with different pixels, while every line, pane and grid pixel stays
    identical — so a byte-exact comparison could only pass on the one
    matplotlib that rendered the fixtures. Text *placement* is still
    checked: a label that moves leaves its baseline pixels outside the mask.
    """

    @pytest.mark.parametrize("fixture,kwargs", [
        ("frame_single_f260.png", dict(frame=260)),
        ("rest_pose.png", None),
    ])
    def test_matches_v082_baseline(self, bvh, tmp_path, fixture, kwargs):
        if kwargs is None:
            fig, _ = bvhplot.rest_pose(bvh, style="debug")
        else:
            fig, _ = bvhplot.frame(bvh, style="debug", **kwargs)
        _assert_matches_v082(fig, fixture, tmp_path)

    def test_pair_matches_v082_baseline(self, bvh, tmp_path):
        fig, _ = bvhplot.frame(
            [bvh, bvh.mirror()], 100, labels=["a", "b"], style="debug")
        _assert_matches_v082(fig, "frame_pair_f100.png", tmp_path)


class TestVectorExport:
    """Figures must survive the trip to a vector file.

    Saving a figure for print means ``savefig(..., bbox_inches="tight")``
    to PDF or SVG. Those canvases carry no ``get_renderer``, and the 3D
    label patch used to call it unconditionally, so every such save
    raised ``AttributeError`` — on the one path a paper figure takes.
    """

    @pytest.mark.parametrize("suffix", [".pdf", ".svg"])
    @pytest.mark.parametrize("style", ["paper", "debug"])
    def test_stills_save_tight_to_vector(self, bvh, tmp_path, suffix, style):
        fig, _ = bvhplot.frame(bvh, frame=100, style=style)
        out = tmp_path / f"figure{suffix}"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        assert out.stat().st_size > 1000

    def test_sequence_saves_tight_to_vector(self, bvh, tmp_path):
        fig, _ = bvhplot.sequence(bvh, n_poses=4)
        out = tmp_path / "sequence.pdf"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        assert out.stat().st_size > 1000

    def test_axis_labels_still_fit_in_the_crop(self, bvh):
        """The patch itself must keep working where it applies.

        On an Agg canvas — Jupyter's inline path, which crops to the
        tight bbox — every visible axis label must lie inside the crop.
        mplot3d places them outside the axes rectangle at some camera
        angles, and its own tight bbox leaves them out.
        """
        from matplotlib.transforms import Bbox

        fig, ax = bvhplot.frame(bvh, frame=100,
                                style=Style("paper", axes="full"))
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        tight = fig.get_tightbbox(renderer)
        to_inches = fig.dpi_scale_trans.inverted()
        labels = [getattr(ax, name).label for name in
                  ("xaxis", "yaxis", "zaxis")]
        extents = [label.get_window_extent(renderer).transformed(to_inches)
                   for label in labels if label.get_visible()]
        plt.close(fig)

        assert extents, "full axes should carry visible axis labels"
        for extent in extents:
            assert Bbox.union([tight, extent]).extents == pytest.approx(
                tight.extents), "an axis label falls outside the tight crop"


class TestFraming:
    """Animated output frames the motion, not a cube around it."""

    def _view(self, bvh, camera="side"):
        from pybvh.bvhplot import _prepare
        return _prepare(bvh, None, "world", camera).views[0]

    def test_box_follows_each_axis_of_the_motion(self, bvh):
        """A travelling clip must not spend its vertical extent on travel."""
        from pybvh.bvhplot._viewport import framing_bounds

        lo, hi = framing_bounds(self._view(bvh))
        spans = hi - lo
        travel = float(spans.max())
        assert travel > 2 * float(np.median(spans)), (
            "fixture no longer travels far enough to exercise framing")
        # the cube the old code framed to would have made every axis this long
        assert float(spans.min()) < 0.6 * travel

    def test_floor_is_inside_the_box(self, bvh):
        from pybvh.bvhplot._viewport import framing_bounds

        view = self._view(bvh)
        lo, hi = framing_bounds(view)
        assert lo[view.up_index] <= view.floor_height <= hi[view.up_index]

    def test_rotating_squares_the_ground_off(self, bvh):
        """An orbiting camera needs an azimuth-invariant footprint."""
        from pybvh.bvhplot._viewport import framing_bounds

        view = self._view(bvh)
        lo, hi = framing_bounds(view, rotating=True)
        ground = [i for i in range(3) if i != view.up_index]
        assert (hi - lo)[ground[0]] == pytest.approx((hi - lo)[ground[1]])

        # and it must still contain the motion it squared off around
        fixed_lo, fixed_hi = framing_bounds(view)
        assert np.all(lo[ground] <= fixed_lo[ground] + 1e-9)
        assert np.all(hi[ground] >= fixed_hi[ground] - 1e-9)

    def test_orbit_holds_one_scale(self, bvh):
        """The same pose must project to the same size at every azimuth.

        Without a fixed scale the drawing is refitted every frame and
        the character pulses in and out as the camera comes round.
        """
        from pybvh.bvhplot._viewport import make_viewport

        view = self._view(bvh, camera="front")
        viewport = make_viewport([view], framing="clip", motion="turntable")
        assert viewport.rotating

        # A vertical probe: spinning the camera about the up axis cannot
        # change its projected length, so any change is the scale moving.
        # (The skeleton itself is a poor probe — its own silhouette
        # genuinely changes height as it turns.)
        middle = (viewport.lo + viewport.hi) / 2.0
        reach = viewport.up_vector * viewport.half_span
        probe = np.stack([middle - reach, middle + reach])

        lengths = []
        for frame in range(0, view.coords.shape[0], 20):
            pixels = viewport.project(probe, (1920, 1080), frame)
            lengths.append(abs(int(pixels[1, 1]) - int(pixels[0, 1])))
        assert len(lengths) > 10
        assert max(lengths) == pytest.approx(min(lengths), rel=0.01)

    @pytest.mark.parametrize("clip", ["bvh_data/bvh_test1.bvh", BVH_PATH])
    def test_the_projection_stays_isotropic(self, clip):
        """One world unit must cover the same screen distance in every
        direction, whichever axis the rig calls up.

        mplot3d stores a box aspect rolled to the *current* vertical
        axis, so setting it before the camera pairs it with the wrong
        limits and stretches the drawing along one screen direction —
        by a factor of 5 on a +y-up rig, while a +z-up rig, where the
        roll is the identity, looks perfectly fine.
        """
        from mpl_toolkits.mplot3d import proj3d
        from pybvh.bvhplot import _prepare
        from pybvh.bvhplot._style import resolve_style
        from pybvh.bvhplot._matplotlib import _setup_animated_panel
        from pybvh.bvhplot._viewport import make_viewport

        view = _prepare(read_bvh_file(clip), None, "world", "side").views[0]
        fig = plt.figure(figsize=(19.2, 10.8))
        ax = fig.add_subplot(111, projection="3d")
        _setup_animated_panel(ax, view, make_viewport([view], framing="clip"),
                              resolve_style("paper"),
                              np.asarray(view.bones, dtype=int), 0, 1)

        # The world -> screen map, differenced about the box centre so
        # the perspective projection is linearized where the motion is.
        projection = ax.get_proj()
        centre = view.coords.reshape(-1, 3).mean(axis=0)
        step = 0.01 * float(np.abs(view.coords).max())
        columns = []
        for axis in np.eye(3):
            ahead = np.array(proj3d.proj_transform(
                *(centre + step * axis), projection)[:2])
            behind = np.array(proj3d.proj_transform(
                *(centre - step * axis), projection)[:2])
            columns.append((ahead - behind) / (2 * step))
        plt.close(fig)

        screen = np.array(columns).T                    # (2, 3)
        norms = np.linalg.norm(screen, axis=1)
        assert norms[0] == pytest.approx(norms[1], rel=0.01), (
            "screen axes are scaled differently — the drawing is stretched")
        assert abs(screen[0] @ screen[1]) < 0.01 * norms.prod(), (
            "screen axes are not perpendicular — the drawing is sheared")

    @pytest.mark.parametrize("backend,suffix", [
        ("matplotlib", ".gif"), ("opencv", ".mp4")])
    def test_a_walking_clip_fills_the_frame(self, bvh, tmp_path,
                                            backend, suffix):
        """The end-to-end promise, in both video backends.

        A cubic box put the walker at 20% of frame height in matplotlib
        and 34% in OpenCV; fitting the clip's own box takes them to 41%
        and 46%. Anything back near the old numbers means the framing
        regressed to a cube.
        """
        if backend == "opencv":
            pytest.importorskip("cv2")
        clip = bvh[:240].resample(20)
        out = bvhplot.render(clip, tmp_path / f"walk{suffix}", fps=20,
                             backend=backend, camera="side")
        frame = _mid_frame(out)
        coloured = (frame.max(axis=2) - frame.min(axis=2)) > 40   # bones
        rows = np.where(coloured.any(axis=1))[0]
        occupied = (rows[-1] - rows[0] + 1) / frame.shape[0]
        assert occupied > 0.35, (
            f"{backend}: subject fills only {occupied:.0%} of the frame")


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


class TestReviewFixes:
    """Regression tests for the v0.9.0 code-review findings."""

    def test_style_does_not_alias_preset_dicts(self):
        s = Style("paper")
        s.chain_colors["l_arm"] = "#FF0000"
        assert Style("paper").chain_colors["l_arm"] == CHAIN_COLORS["l_arm"]
        assert CHAIN_COLORS["l_arm"] != "#FF0000"

    def test_dark_background_by_luminance_not_string(self):
        from pybvh.bvhplot._colors import is_dark_background
        assert not is_dark_background("#F5F5F7")   # light gray is light
        assert not is_dark_background("snow")
        assert not is_dark_background((1.0, 1.0, 1.0))
        assert is_dark_background("#16181D")
        assert is_dark_background("black")

    def test_floor_gray_unified_across_backends(self):
        """The vedo offscreen floor had drifted to #EDEDF1; all solid
        floors now read one palette."""
        from pybvh.bvhplot import _colors, _vedo_offscreen, _matplotlib
        import inspect
        assert "#EDEDF1" not in inspect.getsource(_vedo_offscreen)
        assert _colors.FLOOR_LIGHT["face"] == "#E8E8EC"

    def test_negative_up_floor_at_ground_not_head(self, bvh):
        from tests.synthetic_bvh import make_neg_y_up_bvh
        from pybvh.bvhplot._from_bvh import make_scene
        neg = make_neg_y_up_bvh()
        coords = neg.node_positions()
        scene = make_scene([neg], [coords], "front", None,
                           canonical_floor=False)
        view = scene.views[0]
        assert view.up_sign == -1.0
        # ground = coordinate MAXIMUM for -y up
        assert view.floor_height == pytest.approx(
            float(coords[..., view.up_index].max()))
        # "below the floor" moves toward larger coordinates
        assert view.below_floor(1.0) > view.floor_height

    def test_paper_frame_zorders(self, bvh):
        """Floor under bones under joints, by explicit zorder."""
        fig, ax = bvhplot.frame(bvh, 10, style="paper")
        zorders = sorted(c.get_zorder() for c in ax.collections)
        assert zorders[0] == 0.5          # floor
        assert 3 in zorders               # joint markers on top
        plt.close(fig)

    def test_frame_filepath_written_on_matplotlib(self, bvh, tmp_path):
        out = tmp_path / "pose.png"
        fig, ax = bvhplot.frame(bvh, 10, filepath=out)
        assert out.exists() and out.stat().st_size > 0
        plt.close(fig)


class TestBoneDepthSorting:
    """Bones must occlude correctly: far-side limbs may never paint
    over near-side ones (v0.9.0 follow-up review finding)."""

    @staticmethod
    def _render_crossing(collection_cls, seg_order):
        """Render two crossing colored segments and return the RGB buffer."""
        near = [(-1.0, -1.0, -1.0), (1.0, -1.0, 1.0)]
        far = [(-1.0, 1.0, 1.0), (1.0, 1.0, -1.0)]
        segs = {"near": near, "far": far}
        colors = {"near": (1.0, 0.0, 0.0, 1.0), "far": (0.0, 0.0, 1.0, 1.0)}
        fig = plt.figure(figsize=(2, 2), dpi=60)
        ax = fig.add_subplot(111, projection="3d")
        ax.computed_zorder = False
        ax.set_axis_off()
        ax.add_collection3d(collection_cls(
            [segs[k] for k in seg_order],
            colors=[colors[k] for k in seg_order], linewidths=6))
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
        ax.view_init(elev=10, azim=-90, vertical_axis="z")
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        return buf

    def test_depth_sorted_collection_is_order_invariant(self):
        """The depth-sorted collection must produce the same image no
        matter what order its segments arrive in — proof that draw
        order comes from depth, not from insertion."""
        from pybvh.bvhplot._matplotlib import _DepthSortedLine3DCollection
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
        a = self._render_crossing(_DepthSortedLine3DCollection,
                                  ["near", "far"])
        b = self._render_crossing(_DepthSortedLine3DCollection,
                                  ["far", "near"])
        assert np.array_equal(a, b)
        # Sanity that the test has power: the plain collection IS
        # order-dependent on the same input.
        c = self._render_crossing(Line3DCollection, ["near", "far"])
        d = self._render_crossing(Line3DCollection, ["far", "near"])
        assert not np.array_equal(c, d)

    def test_paper_uses_depth_sorted_debug_uses_plain(self, bvh):
        """The gate: paper (manual z-order) depth-sorts; debug keeps the
        fixed-order collection for pre-0.9.0 pixel parity."""
        from pybvh.bvhplot._matplotlib import _DepthSortedLine3DCollection
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        def bone_collections(style):
            fig, ax = bvhplot.frame(bvh, 100, style=style)
            found = [c for c in ax.collections
                     if isinstance(c, Line3DCollection)]
            plt.close(fig)
            return found

        paper = bone_collections("paper")
        assert any(isinstance(c, _DepthSortedLine3DCollection)
                   for c in paper)
        debug = bone_collections("debug")
        assert debug and all(
            not isinstance(c, _DepthSortedLine3DCollection) for c in debug)
