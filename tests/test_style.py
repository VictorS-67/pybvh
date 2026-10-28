"""Tests for the Style system and bone-chain classification (Phase 1)."""

from __future__ import annotations

import dataclasses
import inspect
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from pybvh import bvhplot, read_bvh_file
from pybvh.bvhplot import _backends
from pybvh.bvhplot._from_bvh import get_bone_chains, get_skeleton_lines, make_scene
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
        with pytest.raises(dataclasses.FrozenInstanceError, match="floor"):
            s.floor = None  # type: ignore[misc]

    def test_replace(self):
        s = Style("paper")
        s2 = s.replace(floor="grid")
        assert s2.floor == "grid"
        assert s.floor == "solid"
        with pytest.raises(TypeError, match="Unknown"):
            s.replace(nope=1)

    @pytest.mark.parametrize(
        "set_by_caller",
        [
            Style("paper", floor="solid"),
            Style("paper", floor="solid", projection="persp"),
            Style("paper", floor=None).replace(floor="solid"),
            Style("paper").replace(projection="persp"),
        ],
        ids=["constructor", "two fields", "replace back", "replace"],
    )
    def test_equality_ignores_which_fields_the_caller_set(self, set_by_caller):
        """Equality is the same look: a field set to the preset's own
        value draws what the preset draws."""
        assert set_by_caller == Style("paper")
        assert repr(set_by_caller) == repr(Style("paper"))

    @pytest.mark.parametrize("style", [Style("paper"), Style("paper", floor=None)])
    def test_unhashable_whatever_the_caller_set(self, style):
        """A Style holds a dict (``chain_colors``), so it has no hash."""
        with pytest.raises(TypeError):
            hash(style)


_A_COLOR = "a matplotlib color, such as 'white', '#16181D' or (0.1, 0.2, 0.8)"


class TestStyleValidation:
    """Every field is checked where the Style is made, by the constructor
    and by ``replace`` alike, so that a bad value fails at the caller's
    line rather than inside a backend."""

    @staticmethod
    def _made_both_ways(field, value):
        """Build a paper Style with *field* set to *value*, once through
        the constructor and once through ``replace``."""
        return [
            lambda: Style("paper", **{field: value}),
            lambda: Style("paper").replace(**{field: value}),
        ]

    @pytest.mark.parametrize(
        "field,value,message",
        [
            ("floor_alpha", 1.5, "floor_alpha must be a number in [0, 1], got 1.5"),
            ("floor_alpha", -0.1, "floor_alpha must be a number in [0, 1], got -0.1"),
            ("floor_alpha", "0.5", "floor_alpha must be a number in [0, 1], got '0.5'"),
            # A bool is no number here, as 1 is no bool for shadow.
            ("floor_alpha", True, "floor_alpha must be a number in [0, 1], got True"),
            ("joint_size", True, "joint_size must be a positive number, got True"),
            ("dpi", True, "dpi must be a positive number or None, got True"),
            ("joint_size", 0, "joint_size must be a positive number, got 0"),
            ("joint_size", -9.0, "joint_size must be a positive number, got -9.0"),
            ("joint_size", "9", "joint_size must be a positive number, got '9'"),
            ("dpi", 0, "dpi must be a positive number or None, got 0"),
            ("dpi", -300, "dpi must be a positive number or None, got -300"),
            ("dpi", "300", "dpi must be a positive number or None, got '300'"),
            ("shadow", 1, "shadow must be True or False, got 1"),
            ("shadow", "yes", "shadow must be True or False, got 'yes'"),
            ("shadow", None, "shadow must be True or False, got None"),
            ("bone_color", "bleu", f"bone_color must be {_A_COLOR}, got 'bleu'"),
            ("bone_color", (1.5, 0.0, 0.0), f"bone_color must be {_A_COLOR}, got (1.5, 0.0, 0.0)"),
            ("bone_color", None, f"bone_color must be {_A_COLOR}, got None"),
            ("background", "whte", f"background must be {_A_COLOR}, got 'whte'"),
            ("background", "#16181", f"background must be {_A_COLOR}, got '#16181'"),
            ("joint_color", "whte", f"joint_color must be {_A_COLOR}, got 'whte'"),
            ("joint_markers", "yes", "joint_markers must be True or False, got 'yes'"),
            (
                "chain_colors",
                "red",
                "chain_colors must be a dict from chain name to color, got 'red'",
            ),
            (
                "chain_colors",
                {"spine": "whte"},
                f"chain_colors['spine'] must be {_A_COLOR}, got 'whte'",
            ),
        ],
    )
    def test_an_invalid_value_is_refused(self, field, value, message):
        for make in self._made_both_ways(field, value):
            with pytest.raises(ValueError) as raised:
                make()
            assert str(raised.value) == message

    @pytest.mark.parametrize(
        "field,value",
        [
            ("floor_alpha", 0),
            ("floor_alpha", 1.0),
            ("joint_size", 0.5),
            ("dpi", 1),
            ("dpi", None),
            ("shadow", False),
            ("bone_color", (0.0, 0.0, 1.0)),
            ("bone_color", "#E69F00"),
            ("background", "snow"),
            ("background", (1, 1, 1)),
            ("joint_color", "red"),
            ("joint_markers", False),
            # A chain left out is drawn in the spine color.
            ("chain_colors", {"spine": "black"}),
        ],
    )
    def test_a_valid_value_is_kept(self, field, value):
        for make in self._made_both_ways(field, value):
            assert getattr(make(), field) == value

    @pytest.mark.parametrize("preset", ["paper", "debug", "dark"])
    def test_every_preset_passes_its_own_checks(self, preset):
        assert Style(preset) == Style(preset)


# The drawing path each column of the Style docstring's table names: under
# the call (or the group) of its first header row, the label of its second.
_TABLE_COLUMNS = {
    ("frame", "mpl"): ("frame", "matplotlib"),
    ("frame", "vedo"): ("frame", "vedo"),
    ("render", "mpl"): ("render", "matplotlib"),
    ("render", "cv"): ("render", "opencv"),
    ("render", "vedo"): ("render", "vedo"),
    ("play", "mpl"): ("play", "matplotlib"),
    ("play", "cv"): ("play", "opencv_notebook"),
    ("play", "k3d"): ("play", "k3d"),
    ("play", "vedo"): ("play", "vedo"),
    ("matplotlib alone", "rest"): ("rest_pose", "matplotlib"),
    ("matplotlib alone", "offs"): ("sequence", "offset"),
    ("matplotlib alone", "over"): ("sequence", "overlay"),
    ("matplotlib alone", "traj"): ("trajectory", "matplotlib"),
}


def _docstring_table() -> tuple[list[tuple[str, str]], list[tuple[str, list[str]]]]:
    """The Style docstring's table: the drawing path of each column, and
    each row's field and cells."""
    lines = inspect.cleandoc(Style.__doc__ or "").splitlines()
    headers = [i for i, line in enumerate(lines) if line.split()[:1] == ["field"]]
    assert len(headers) == 1, "the Style docstring holds no table, or several"
    at = headers[0]
    groups = [(m.start(), m.group()) for m in re.finditer(r"\S+(?: \S+)*", lines[at - 1])]
    columns = []
    for label in list(re.finditer(r"\S+", lines[at]))[1:]:
        group = [name for start, name in groups if start <= label.start()][-1]
        columns.append(_TABLE_COLUMNS[group, label.group()])
    rows = []
    for line in lines[at + 1 :]:
        if not line.strip():
            break
        field, *cells = line.split()
        rows.append((field, cells))
    return columns, rows


def _cell(call: str, path: str, field: str) -> str:
    """What the table should say of *field* on the drawing path: ``yes``
    drawn at every value, the values drawn, ``n/a`` accepted, ``no``
    refused."""
    drawn = _backends._draws(call, path)
    if field in drawn.fields:
        return "yes"
    if field in _backends._STYLE_ACCEPTED[call]:
        return "n/a"
    values = drawn.values.get(field, frozenset())
    return ",".join(sorted(str(value) for value in values)) or "no"


class TestTheDocstringTable:
    """The table in the Style docstring says what each drawing path does
    with each field its caller sets, as the declarations the backend
    choice reads say it."""

    def test_a_column_per_drawing_path(self):
        columns, _ = _docstring_table()
        paths = [
            (call, path) for call, ways in _backends._DECLARING_MODULES.items() for path in ways
        ]
        assert sorted(columns) == sorted(paths)

    def test_a_row_per_field(self):
        _, rows = _docstring_table()
        fields = [f.name for f in dataclasses.fields(Style) if f.compare]
        assert sorted(field for field, _ in rows) == sorted(fields)

    def test_every_cell_says_what_the_declarations_say(self):
        columns, rows = _docstring_table()
        wrong = [
            f"{field} on {call} {path}: {cell!r}, declared {_cell(call, path, field)!r}"
            for field, cells in rows
            for (call, path), cell in zip(columns, cells, strict=True)
            if cell != _cell(call, path, field)
        ]
        assert not wrong, "\n".join(wrong)


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
        foot_nodes = {bvh.node_index[n] for n in bvh.auto_detect_foot_joints()}
        leg_children = {bones[i][1] for i in chains["l_leg"] + chains["r_leg"]}
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
        junction = {i for i, (p, c) in enumerate(bones) if c in paired and p not in paired}
        assert junction, "walk skeleton must have torso->limb junctions"
        assert junction <= set(chains["spine"])
        limb_indices = {i for name, idxs in chains.items() if name != "spine" for i in idxs}
        assert all(bones[i][0] in paired and bones[i][1] in paired for i in limb_indices)

    def test_no_lr_pairs_falls_back_to_spine(self, bvh):
        """A skeleton without L/R pairs puts every bone in 'spine'."""
        sub = bvh.extract_joints(["Hips", "LowerBack", "Spine", "Spine1", "Neck", "Neck1", "Head"])
        assert sub.node_lr_pairs is None
        chains = get_bone_chains(sub)
        assert set(chains) == {"spine"}
        assert sorted(chains["spine"]) == list(range(len(get_skeleton_lines(sub))))


def _mid_frame(path: Path) -> np.ndarray:
    """The middle frame of a rendered clip, as RGB."""
    from PIL import Image

    if path.suffix == ".gif":
        clip = Image.open(path)
        clip.seek(clip.n_frames // 2)
        return np.asarray(clip.convert("RGB"))

    import cv2

    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) // 2)
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
        "the text mask covers too much of the figure for the comparison to mean anything"
    )
    differs = (got != want).any(axis=2) & ~text
    assert not differs.any(), (
        f"{fixture}: {differs.sum()} pixels outside text differ from the v0.8.2 baseline"
    )


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

    @pytest.mark.parametrize(
        "fixture,kwargs",
        [
            ("frame_single_f260.png", dict(frame=260)),
            ("rest_pose.png", None),
        ],
    )
    def test_matches_v082_baseline(self, bvh, tmp_path, fixture, kwargs):
        if kwargs is None:
            fig, _ = bvhplot.rest_pose(bvh, style="debug")
        else:
            fig, _ = bvhplot.frame(bvh, style="debug", **kwargs)
        _assert_matches_v082(fig, fixture, tmp_path)

    def test_pair_matches_v082_baseline(self, bvh, tmp_path):
        fig, _ = bvhplot.frame([bvh, bvh.mirror()], 100, labels=["a", "b"], style="debug")
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

        fig, ax = bvhplot.frame(bvh, frame=100, style=Style("paper", axes="full"))
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        tight = fig.get_tightbbox(renderer)
        to_inches = fig.dpi_scale_trans.inverted()
        labels = [getattr(ax, name).label for name in ("xaxis", "yaxis", "zaxis")]
        extents = [
            label.get_window_extent(renderer).transformed(to_inches)
            for label in labels
            if label.get_visible()
        ]
        plt.close(fig)

        assert extents, "full axes should carry visible axis labels"
        for extent in extents:
            assert Bbox.union([tight, extent]).extents == pytest.approx(tight.extents), (
                "an axis label falls outside the tight crop"
            )


def _whole_clip_view(bvh, camera):
    """The view of every frame of *bvh* under ``centered="world"``, as an
    animated output draws it."""
    return make_scene(
        [bvh], [bvh.node_positions(centered="world")], camera, None, clip_frames=slice(None)
    ).views[0]


class TestFraming:
    """Animated output frames the motion, not a cube around it."""

    def test_box_follows_each_axis_of_the_motion(self, bvh):
        """A travelling clip must not spend its vertical extent on travel."""
        from pybvh.bvhplot._viewport import framing_bounds

        lo, hi = framing_bounds(_whole_clip_view(bvh, "side"))
        spans = hi - lo
        travel = float(spans.max())
        assert travel > 2 * float(np.median(spans)), (
            "fixture no longer travels far enough to exercise framing"
        )
        # the cube the old code framed to would have made every axis this long
        assert float(spans.min()) < 0.6 * travel

    def test_floor_is_inside_the_box(self, bvh):
        from pybvh.bvhplot._viewport import framing_bounds

        view = _whole_clip_view(bvh, "side")
        lo, hi = framing_bounds(view)
        assert lo[view.up_index] <= view.floor_height <= hi[view.up_index]

    def test_rotating_squares_the_ground_off(self, bvh):
        """An orbiting camera needs an azimuth-invariant footprint."""
        from pybvh.bvhplot._viewport import framing_bounds

        view = _whole_clip_view(bvh, "side")
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

        view = _whole_clip_view(bvh, "front")
        viewport = make_viewport([view], framing="clip", schedule="turntable")
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

        from pybvh.bvhplot._matplotlib import _setup_animated_panel
        from pybvh.bvhplot._style import resolve_style
        from pybvh.bvhplot._viewport import make_viewport

        view = _whole_clip_view(read_bvh_file(clip), "side")
        fig = plt.figure(figsize=(19.2, 10.8))
        ax = fig.add_subplot(111, projection="3d")
        _setup_animated_panel(
            ax,
            view,
            make_viewport([view], framing="clip"),
            resolve_style("paper"),
            np.asarray(view.bones, dtype=int),
            0,
            1,
        )

        # The world -> screen map, differenced about the box centre so
        # the perspective projection is linearized where the motion is.
        projection = ax.get_proj()
        centre = view.coords.reshape(-1, 3).mean(axis=0)
        step = 0.01 * float(np.abs(view.coords).max())
        columns = []
        for axis in np.eye(3):
            ahead = np.array(proj3d.proj_transform(*(centre + step * axis), projection)[:2])
            behind = np.array(proj3d.proj_transform(*(centre - step * axis), projection)[:2])
            columns.append((ahead - behind) / (2 * step))
        plt.close(fig)

        screen = np.array(columns).T  # (2, 3)
        norms = np.linalg.norm(screen, axis=1)
        assert norms[0] == pytest.approx(norms[1], rel=0.01), (
            "screen axes are scaled differently — the drawing is stretched"
        )
        assert abs(screen[0] @ screen[1]) < 0.01 * norms.prod(), (
            "screen axes are not perpendicular — the drawing is sheared"
        )

    @pytest.mark.parametrize("backend,suffix", [("matplotlib", ".gif"), ("opencv", ".mp4")])
    def test_a_walking_clip_fills_the_frame(self, bvh, tmp_path, backend, suffix):
        """The end-to-end promise, in both video backends.

        A cubic box put the walker at 20% of frame height in matplotlib
        and 34% in OpenCV; fitting the clip's own box takes them to 41%
        and 46%. Anything back near the old numbers means the framing
        regressed to a cube.
        """
        if backend == "opencv":
            pytest.importorskip("cv2")
        clip = bvh[:240].resample(20)
        out = bvhplot.render(
            clip, tmp_path / f"walk{suffix}", fps=20, backend=backend, camera="side"
        )
        frame = _mid_frame(out)
        coloured = (frame.max(axis=2) - frame.min(axis=2)) > 40  # bones
        rows = np.where(coloured.any(axis=1))[0]
        occupied = (rows[-1] - rows[0] + 1) / frame.shape[0]
        assert occupied > 0.35, f"{backend}: subject fills only {occupied:.0%} of the frame"


class TestStyledRenderSmoke:
    """Every preset and floor kind renders through both backends."""

    @pytest.mark.parametrize(
        "style",
        [
            "paper",
            "dark",
            Style("paper", floor="grid"),
            Style("paper", floor="checker"),
            Style("paper", projection="ortho"),
            Style("paper", color_mode="chains"),
        ],
    )
    def test_frame_styles(self, bvh, style):
        fig, ax = bvhplot.frame(bvh, 10, style=style)
        assert fig is not None
        plt.close(fig)

    def test_frame_multi_auto_switches_to_palette(self, bvh):
        fig, axes = bvhplot.frame([bvh, bvh], 10, style="paper")
        plt.close(fig)

    def test_render_opencv_paper(self, bvh, tmp_path):
        pytest.importorskip("cv2")
        short = bvh[0:10]
        path = bvhplot.render(
            short, tmp_path / "paper.mp4", backend="opencv", resolution=(320, 240), style="paper"
        )
        assert path.exists() and path.stat().st_size > 0

    def test_render_opencv_debug(self, bvh, tmp_path):
        pytest.importorskip("cv2")
        short = bvh[0:10]
        path = bvhplot.render(
            short, tmp_path / "debug.mp4", backend="opencv", resolution=(320, 240), style="debug"
        )
        assert path.exists() and path.stat().st_size > 0

    def test_render_mpl_paper_gif(self, bvh, tmp_path):
        short = bvh[0:6]
        path = bvhplot.render(
            short,
            tmp_path / "paper.gif",
            backend="matplotlib",
            resolution=(320, 240),
            style="paper",
        )
        assert path.exists() and path.stat().st_size > 0


# How far, in levels of 255 per channel, a joint pixel may stray from
# the joint color. Where two joints overlap, a pixel inside their union
# that neither covers fully keeps a trace of what lies under it: 1 level
# at most inside the joints of the walk test pictures with the joints on
# top, against 255 where a bone covers them.
JOINT_COLOR_LEVELS = 8


def _joints_kept(thin: np.ndarray, wide: np.ndarray) -> bool:
    """Whether every joint pixel of the picture over thin bones is a
    joint pixel of the picture over wide ones.

    A joint pixel is the red the joint tests draw joints in, to within
    ``JOINT_COLOR_LEVELS``. The thin picture's joint rims are left out:
    a pixel a joint covers all but a sliver of keeps a trace of what
    lies under it, which differs between the two.
    """

    def red(picture: np.ndarray) -> np.ndarray:
        distance = np.abs(picture.astype(int) - (255, 0, 0))
        return np.all(distance <= JOINT_COLOR_LEVELS, axis=-1)

    joints, kept = red(thin), red(wide)
    inside = joints.copy()
    inside[1:] &= joints[:-1]
    inside[:-1] &= joints[1:]
    inside[:, 1:] &= joints[:, :-1]
    inside[:, :-1] &= joints[:, 1:]
    return bool(inside.any()) and not (inside & ~kept).any()


# Large red joints and no floor, so the joints and bones alone fill the
# picture; the bones are drawn in chain colors, none of them red.
RED_JOINTS = Style("paper", joint_size=120, joint_color="red", floor=None)


class TestMatplotlibVideosDrawTheJointsOverTheBones:
    """The joints sit over the bones in every matplotlib picture, as in
    ``frame``: every pixel a joint holds over thin bones it still holds
    over wide ones, which it would lose to a bone drawn over it."""

    def test_render(self, walk, tmp_path, apng_frames):
        def first_frame(bone_width):
            path = tmp_path / f"{bone_width}.apng"
            bvhplot.render(
                walk,
                path,
                backend="matplotlib",
                resolution=(320, 240),
                fps=10,
                style=RED_JOINTS.replace(bone_width=bone_width),
            )
            return apng_frames(path, 10)[0]

        assert _joints_kept(first_frame(0.5), first_frame(8.0))

    def test_play(self, walk, environment, monkeypatch):
        """In a notebook, play's matplotlib player is a page holding
        every frame as a PNG; its first is read back."""
        import base64
        import io
        import re

        from PIL import Image

        ipython_display = pytest.importorskip("IPython.display")
        shown = []
        monkeypatch.setattr(ipython_display, "display", shown.append)
        environment(notebook=True)

        def first_frame(bone_width):
            bvhplot.play(
                walk, backend="matplotlib", fps=10, style=RED_JOINTS.replace(bone_width=bone_width)
            )
            page = shown[-1].data
            png = re.search(r'frames\[0\] = "data:image/png;base64,([^"]+)"', page)
            assert png is not None
            image = Image.open(io.BytesIO(base64.b64decode(png.group(1))))
            return np.asarray(image.convert("RGB"))

        assert _joints_kept(first_frame(0.5), first_frame(8.0))


def _contrast(color, other) -> float:
    """The WCAG contrast ratio of two colors, 1 (the same) to 21
    (black on white); 4.5 is its minimum for body text."""
    from matplotlib.colors import to_rgb

    def luminance(c):
        rgb = np.array(to_rgb(c))
        linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        return float(linear @ (0.2126, 0.7152, 0.0722))

    hi, lo = sorted((luminance(color), luminance(other)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


class TestPanelTitlesAreLegible:
    """Panel titles stand out from the style's background, dark or
    light."""

    def test_frame(self, bvh):
        fig, axs = bvhplot.frame([bvh, bvh], 10, style="dark", labels=["A", "B"])
        background = resolve_style("dark").background
        assert all(_contrast(ax.title.get_color(), background) >= 4.5 for ax in axs)
        plt.close(fig)

    def test_frame_keeps_black_titles_on_a_light_background(self, bvh):
        """matplotlib's own title color, as before."""
        fig, axs = bvhplot.frame([bvh, bvh], 10, style="paper", labels=["A", "B"])
        assert all(matplotlib.colors.to_hex(ax.title.get_color()) == "#000000" for ax in axs)
        plt.close(fig)

    def test_render(self, walk, tmp_path, apng_frames):
        """The title is the pixels a label adds to the picture; its
        brightest stands out from the background."""

        def first_frame(labels):
            path = tmp_path / f"{labels}.apng"
            bvhplot.render(
                walk,
                path,
                backend="matplotlib",
                resolution=(320, 240),
                fps=10,
                style="dark",
                labels=labels,
            )
            return apng_frames(path, 10)[0]

        titled, untitled = first_frame(["WALK"]), first_frame(None)
        title = np.any(titled != untitled, axis=-1)
        assert title.any()
        brightest = titled[title][np.argmax(titled[title].sum(axis=-1))] / 255
        assert _contrast(brightest, resolve_style("dark").background) >= 4.5


class TestGridFloorReadsFloorAlpha:
    """A grid floor is drawn at the style's ``floor_alpha``, as the solid
    and checker floors are: the more opaque, the farther the picture
    from the same one without a floor. matplotlib and OpenCV drew every
    grid at 0.8."""

    @pytest.fixture
    def drawn(self, bvh, tmp_path, apng_frames):
        """The picture *call* draws of the walk's first frames with
        *style*: ``frame``'s figure, or the first frame ``render`` writes
        to an ``.apng`` on *call*'s backend, read back."""

        def draw(call, style):
            if call == "frame":
                fig, _ = bvhplot.frame(bvh, 10, style=style)
                pixels = _fig_pixels(fig)[..., :3]
                plt.close(fig)
                return pixels
            if call == "render opencv":
                pytest.importorskip("cv2")
            path = tmp_path / "grid.apng"
            bvhplot.render(
                bvh[0:2], path, backend=call.split()[1], resolution=(160, 120), fps=10, style=style
            )
            return apng_frames(path, 10)[0]

        return draw

    @pytest.mark.parametrize("call", ["frame", "render matplotlib", "render opencv"])
    def test_a_more_opaque_grid_is_farther_from_no_floor(self, drawn, call):
        def away_from_floorless(floor_alpha):
            grid = drawn(call, Style("paper", floor="grid", floor_alpha=floor_alpha))
            return np.abs(grid.astype(int) - floorless.astype(int)).sum()

        floorless = drawn(call, Style("paper", floor=None))
        faint, opaque = away_from_floorless(0.3), away_from_floorless(0.85)
        assert 0 < faint < opaque


class TestReviewFixes:
    """Regression tests for the v0.9.0 code-review findings."""

    def test_style_does_not_alias_preset_dicts(self):
        s = Style("paper")
        s.chain_colors["l_arm"] = "#FF0000"
        assert Style("paper").chain_colors["l_arm"] == CHAIN_COLORS["l_arm"]
        assert CHAIN_COLORS["l_arm"] != "#FF0000"

    def test_dark_background_by_luminance_not_string(self):
        from pybvh.bvhplot._colors import is_dark_background

        assert not is_dark_background("#F5F5F7")  # light gray is light
        assert not is_dark_background("snow")
        assert not is_dark_background((1.0, 1.0, 1.0))
        assert is_dark_background("#16181D")
        assert is_dark_background("black")

    def test_floor_gray_unified_across_backends(self):
        """The vedo offscreen floor had drifted to #EDEDF1; all solid
        floors now read one palette."""
        import inspect

        from pybvh.bvhplot import _colors, _vedo_offscreen

        assert "#EDEDF1" not in inspect.getsource(_vedo_offscreen)
        assert _colors.FLOOR_LIGHT["face"] == "#E8E8EC"

    def test_negative_up_floor_at_ground_not_head(self, bvh):
        from pybvh.bvhplot._from_bvh import make_scene
        from tests.synthetic_bvh import make_neg_y_up_bvh

        neg = make_neg_y_up_bvh()
        coords = neg.node_positions()
        scene = make_scene([neg], [coords], "front", None, canonical_floor=False)
        view = scene.views[0]
        assert view.up_sign == -1.0
        # ground = coordinate MAXIMUM for -y up
        assert view.floor_height == pytest.approx(float(coords[..., view.up_index].max()))
        # "below the floor" moves toward larger coordinates
        assert view.below_floor(1.0) > view.floor_height

    def test_paper_frame_zorders(self, bvh):
        """Floor under bones under joints, by explicit zorder."""
        fig, ax = bvhplot.frame(bvh, 10, style="paper")
        zorders = sorted(c.get_zorder() for c in ax.collections)
        assert zorders[0] == 0.5  # floor
        assert 3 in zorders  # joint markers on top
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
        ax.add_collection3d(
            collection_cls(
                [segs[k] for k in seg_order], colors=[colors[k] for k in seg_order], linewidths=6
            )
        )
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.view_init(elev=10, azim=-90, vertical_axis="z")
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        return buf

    def test_depth_sorted_collection_is_order_invariant(self):
        """The depth-sorted collection must produce the same image no
        matter what order its segments arrive in — proof that draw
        order comes from depth, not from insertion."""
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        from pybvh.bvhplot._matplotlib import _DepthSortedLine3DCollection

        a = self._render_crossing(_DepthSortedLine3DCollection, ["near", "far"])
        b = self._render_crossing(_DepthSortedLine3DCollection, ["far", "near"])
        assert np.array_equal(a, b)
        # Sanity that the test has power: the plain collection IS
        # order-dependent on the same input.
        c = self._render_crossing(Line3DCollection, ["near", "far"])
        d = self._render_crossing(Line3DCollection, ["far", "near"])
        assert not np.array_equal(c, d)

    def test_paper_uses_depth_sorted_debug_uses_plain(self, bvh):
        """The gate: paper (manual z-order) depth-sorts; debug keeps the
        fixed-order collection for pre-0.9.0 pixel parity."""
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        from pybvh.bvhplot._matplotlib import _DepthSortedLine3DCollection

        def bone_collections(style):
            fig, ax = bvhplot.frame(bvh, 100, style=style)
            found = [c for c in ax.collections if isinstance(c, Line3DCollection)]
            plt.close(fig)
            return found

        paper = bone_collections("paper")
        assert any(isinstance(c, _DepthSortedLine3DCollection) for c in paper)
        debug = bone_collections("debug")
        assert debug and all(not isinstance(c, _DepthSortedLine3DCollection) for c in debug)
