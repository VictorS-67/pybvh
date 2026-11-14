#!/usr/bin/env python3
"""Generate golden reference fixtures for pybvh's differential tests.

Run ONCE, offline, in the fixture env `tests/fixtures/.venv`, pinned by `tests/fixtures/requirements.txt` (it has the reference libraries; `tests/fixtures/README.md` makes it). The committed `.npz` outputs are then loaded by numpy-only tests in the development `.venv` at the root of the checkout — so CI validates pybvh against scipy/pytransform3d WITHOUT those libraries being a runtime (or even CI) dependency.

    tests/fixtures/.venv/bin/python tests/fixtures/generate_fixtures.py

Each fixture stores its INPUTS, the reference OUTPUTS, and a `meta` JSON string documenting the exact convention mapping (the antidote to "definition drift": a future mismatch should be debugged against this meta, not assumed to be a bug).

References used:
- scipy.spatial.transform.Rotation  (rotation conversions)
- pytransform3d                      (SE(3) exp/log/interp — added when those land)

One fixture is different in kind: `foot_contacts_pinned.npz` is a BEHAVIOR PIN of pybvh's own contact detection, `foot_contacts` and `ground_contacts` (no external reference). It is excluded from the default run above and regenerates only via an explicit flag, in the development `.venv` — see `gen_foot_contacts` for the re-baselining warning:

    .venv/bin/python tests/fixtures/generate_fixtures.py --foot-contacts-pin

`follow_azimuths_pinned.npz` is a second behavior pin of the same kind, of the follow camera's azimuth schedule (`gen_follow_azimuths`):

    .venv/bin/python tests/fixtures/generate_fixtures.py --follow-azimuths-pin

`lr_pairs_pinned.npz` is a third, of every value the L/R pairs feed on every bundled clip (`gen_lr_pairs`):

    .venv/bin/python tests/fixtures/generate_fixtures.py --lr-pairs-pin
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import warnings
from typing import TYPE_CHECKING

import numpy as np

# The reference libraries (scipy, pytransform3d) are imported inside the
# gen_* functions that need them, not here: the behavior-pin generator and
# the pin tests (tests/test_analysis.py) import this module in the
# numpy-only development `.venv`, where those libraries do not exist. The
# import below is for annotations only and never runs.
if TYPE_CHECKING:
    from scipy.spatial.transform import Rotation as SR

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 0xB7  # fixed: fixtures must be reproducible


def pybvh_quat_from_scipy(rot: SR) -> np.ndarray:
    """scipy quat (x,y,z,w) -> pybvh convention (w,x,y,z), canonical w>=0."""
    q = np.atleast_2d(rot.as_quat())  # (...,4) scalar-LAST
    q = np.stack([q[..., 3], q[..., 0], q[..., 1], q[..., 2]], axis=-1)
    q = np.where(q[..., 0:1] < 0.0, -q, q)  # canonical w>=0 (scalar-first)
    return q


def _save(name: str, meta: dict, **arrays) -> None:
    np.savez(os.path.join(HERE, name + ".npz"), meta=json.dumps(meta), **arrays)
    print(f"  wrote {name}.npz  ({', '.join(arrays)})")


def gen_rotations() -> None:
    from scipy.spatial.transform import Rotation as SR

    rng = np.random.default_rng(SEED)

    # --- inputs: rotation matrices from bounded-angle rotvecs (clean,
    #     elementwise-comparable) + identity. Angles kept in [0.05, 0.85*pi]
    #     to avoid the axis/sign double-cover ambiguity at 0 and pi. ---
    axes = rng.normal(size=(96, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    angles = rng.uniform(0.05, 0.85 * np.pi, size=(96, 1))
    rotvec = np.concatenate([np.zeros((1, 3)), axes * angles], axis=0)
    rots = SR.from_rotvec(rotvec)
    mats = rots.as_matrix()

    _save(
        "rotmat_to_quat",
        {
            "ref": "scipy.spatial.transform.Rotation",
            "quat_convention": "(w,x,y,z) scalar-first, canonical w>=0",
            "angle_range_rad": [0.0, 0.85 * np.pi],
            "seed": hex(SEED),
        },
        rotmat=mats,
        quat_wxyz=pybvh_quat_from_scipy(rots),
    )

    _save(
        "rotmat_to_axisangle",
        {
            "ref": "scipy.spatial.transform.Rotation.as_rotvec",
            "axisangle_convention": "rotvec = axis * angle, angle in [0, pi]",
            "seed": hex(SEED),
        },
        rotmat=mats,
        rotvec=rots.as_rotvec(),
    )

    # --- euler -> rotmat: forward direction is unique for ANY angle, so use
    #     full range. pybvh euler = intrinsic, pre-mult R=R_first@..., order
    #     string read left-to-right; scipy uppercase seq = intrinsic, same. ---
    eul = np.concatenate([np.zeros((1, 3)), rng.uniform(-np.pi, np.pi, size=(96, 3))], axis=0)
    order = "ZYX"
    rotmat = SR.from_euler(order, eul, degrees=False).as_matrix()
    _save(
        "euler_zyx_to_rotmat",
        {
            "ref": "scipy.spatial.transform.Rotation.from_euler",
            "euler_convention": "intrinsic, R=R_Z@R_Y@R_X, radians, "
            "first angle <-> first axis letter",
            "order": order,
            "seed": hex(SEED),
        },
        euler=eul,
        rotmat=rotmat,
    )


def _se3_twists() -> np.ndarray:
    """Curated se(3) twists [omega(3), v(3)] stressing the cases that break
    implementations: theta->0 (V left-Jacobian Taylor series), theta->pi (log
    branch), and large translation with small rotation (V coupling)."""
    rng = np.random.default_rng(SEED ^ 0x5E3)
    thetas = [0.0, 1e-9, 1e-6, 1e-3, 1e-2, 0.5, np.pi / 2, 2.3, np.pi - 1e-6, np.pi - 1e-9, np.pi]
    axes = [np.array([1.0, 0, 0]), np.array([0, 0, 1.0]), np.array([1, 1, 1.0]) / np.sqrt(3)]
    vs = [
        np.zeros(3),
        np.array([1.0, 0, 0]),
        np.array([0.0, -3.0, 2.0]),
        np.array([100.0, 0.0, 0.0]),  # large translation -> V coupling
        np.array([5.0, -5.0, 5.0]),
    ]
    rows = [np.concatenate([a * th, v]) for th in thetas for a in axes for v in vs]
    for _ in range(24):  # general random twists
        a = rng.normal(size=3)
        a /= np.linalg.norm(a)
        th = rng.uniform(0.05, np.pi - 0.05)
        rows.append(np.concatenate([a * th, rng.normal(size=3) * rng.uniform(0.1, 10)]))
    return np.asarray(rows, dtype=float)


def gen_se3() -> None:
    import pytransform3d.transformations as pt

    twist = _se3_twists()  # (N,6) [omega, v], rotation-first
    transform = np.array([pt.transform_from_exponential_coordinates(x) for x in twist])
    theta = np.linalg.norm(twist[:, :3], axis=1)
    _save(
        "se3_exp_log",
        {
            "ref": "pytransform3d.transformations",
            "twist_order": "[omega(3), v(3)] rotation-first (Modern Robotics / Vemulapalli)",
            "coupling": "v is V-left-Jacobian-coupled, NOT raw translation",
            "edge_thetas": "0, 1e-9..1e-2, pi/2, pi-eps, pi",
            "seed": hex(SEED),
        },
        twist=twist,
        transform=transform,
        theta=theta,
    )

    rng = np.random.default_rng(SEED ^ 0x5C2)
    T0, T1, tval, interp = [], [], [], []
    for _ in range(6):
        A = pt.transform_from_exponential_coordinates(rng.normal(size=6))
        B = pt.transform_from_exponential_coordinates(rng.normal(size=6))
        rel = pt.exponential_coordinates_from_transform(np.linalg.inv(A) @ B)
        for t in (0.0, 0.25, 0.5, 0.75, 1.0):
            T0.append(A)
            T1.append(B)
            tval.append(t)
            interp.append(A @ pt.transform_from_exponential_coordinates(t * rel))
    _save(
        "se3_screw_interp",
        {
            "ref": "pytransform3d",
            "definition": "A @ exp(t*log(inv(A)@B)) screw geodesic",
            "endpoints": "t=0 -> A, t=1 -> B",
            "seed": hex(SEED),
        },
        T0=np.array(T0),
        T1=np.array(T1),
        t=np.array(tval),
        interp=np.array(interp),
    )


def gen_geodesic() -> None:
    from scipy.spatial.transform import Rotation as SR

    rng = np.random.default_rng(SEED ^ 0x6E0)

    def rotvecs(n):
        a = rng.normal(size=(n, 3))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        return a * rng.uniform(0.0, np.pi, size=(n, 1))

    rv1 = np.concatenate([rotvecs(48), np.array([[0.0, 0, 0], [0.0, 0, 0]])])
    rv2 = np.concatenate([rotvecs(48), np.array([[0.0, 0, 0], [np.pi - 1e-6, 0, 0]])])
    R1, R2 = SR.from_rotvec(rv1), SR.from_rotvec(rv2)
    _save(
        "rotation_geodesic",
        {
            "ref": "scipy: (R1.inv()*R2).magnitude()",
            "unit": "radians",
            "edges": "identical (0), near-pi",
            "seed": hex(SEED),
        },
        rotmat_a=R1.as_matrix(),
        rotmat_b=R2.as_matrix(),
        angle=(R1.inv() * R2).magnitude(),
    )


def _load_smoothness_reference():
    """Fetch the Balasubramanian SPARC reference (ISC license, siva82kb/SPARC),
    pinned to a commit, and import it. Its code is NOT committed — only the
    numbers it produces are. Needs network at regeneration time (rare)."""
    import importlib.util
    import tempfile
    import urllib.request

    sha = "7deff21add7e3b6403869c8932dff31bceacb472"
    url = f"https://raw.githubusercontent.com/siva82kb/SPARC/{sha}/scripts/smoothness.py"
    code = urllib.request.urlopen(url, timeout=30).read()
    path = os.path.join(tempfile.gettempdir(), f"_sparc_ref_{sha[:8]}.py")
    with open(path, "wb") as f:
        f.write(code)
    spec = importlib.util.spec_from_file_location("sparc_ref", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, url


def _smoothness_signals():
    """Deterministic 1-D speed profiles spanning smooth -> jerky (L=200, fs=100)."""
    L, fs = 200, 100.0
    t = np.linspace(0.0, 1.0, L)
    minjerk = (t**2) * (1.0 - t) ** 2  # ~minimum-jerk speed bell
    minjerk /= minjerk.max()
    s0 = minjerk
    s1 = minjerk + np.roll(minjerk, 40)  # two overlapping segments
    s2 = minjerk + 0.25 * np.sin(30 * np.pi * t) ** 2
    s3 = minjerk + 0.10 * np.sin(8 * np.pi * t) + 0.05 * np.sin(22 * np.pi * t)
    return np.vstack([s0, s1, s2, s3]).astype(float), fs


# SPARC parameter sets to pin. The first is the default pybvh ships; the
# rest exist because `padlevel` / `fc` / `amp_th` are the whole point of
# that function and had no reference coverage at all — only the default
# path was ever checked against the reference, which is the same blind
# spot that hid the dimensionless-jerk normalizer divergence.
SPARC_PARAM_SETS = [
    {"padlevel": 4, "fc": 10.0, "amp_th": 0.05},  # pybvh's defaults
    {"padlevel": 1, "fc": 10.0, "amp_th": 0.05},  # minimal zero-padding
    {"padlevel": 4, "fc": 2.0, "amp_th": 0.05},  # band cut below the default's
    {"padlevel": 4, "fc": 20.0, "amp_th": 0.01},  # wider band, lower threshold
]


def gen_smoothness() -> None:
    ref, url = _load_smoothness_reference()
    sig, fs = _smoothness_signals()
    dlj = np.array([ref.dimensionless_jerk(s, fs) for s in sig])
    ldlj = np.array([ref.log_dimensionless_jerk(s, fs) for s in sig])

    arrays = {"signals": sig, "fs": np.array(fs), "dlj": dlj, "ldlj": ldlj}
    for i, params in enumerate(SPARC_PARAM_SETS):
        # [0] = scalar SAL
        arrays[f"sparc{i}"] = np.array([ref.sparc(s, fs, **params)[0] for s in sig])
    # `sparc` (no index) stays the default-parameter row under its original
    # name so the committed numbers keep their identity across this change
    arrays["sparc"] = arrays["sparc0"]

    _save(
        "smoothness",
        {
            "ref": url,
            "license": "ISC (siva82kb/SPARC)",
            "sparc_params": SPARC_PARAM_SETS[0],
            "sparc_param_sets": SPARC_PARAM_SETS,
            "input": "1-D speed profile, fs Hz",
            "signals": "deterministic",
            "note": "signal 3 goes negative on purpose — these measures are "
            "defined on a signed scalar velocity, not only a magnitude",
        },
        **arrays,
    )


# ----------------------------------------------------------------------------
# contact detection behavior pin (self-referential golden — no external
# reference)
# ----------------------------------------------------------------------------

# The pinned runs. Each builder takes a FRESH Bvh of the CMU walk clip and
# returns the detector it calls (a bound foot_contacts or ground_contacts, of
# that clip or of a clip cut from it) and the kwargs for the call;
# call_pinned_run adds return_info=True. tests/test_analysis.py imports this
# list and replays the exact same calls against the fixture, so the pinned
# calls and the replayed calls cannot drift apart. New runs are appended:
# a run's index is its key prefix in the fixture.


def _run_default(bvh):
    return bvh.foot_contacts, {}


def _run_explicit_feet_reversed(bvh):
    # Reversed auto-detection: pins column order = feet order, and
    # that explicit feet bypasses the floor cache.
    return bvh.foot_contacts, {"feet": list(reversed(bvh.auto_detect_feet()))}


def _run_method_velocity(bvh):
    return bvh.foot_contacts, {"method": "velocity"}


def _run_method_height_floor0(bvh):
    return bvh.foot_contacts, {"method": "height", "floor": 0.0}


def _run_height_reference_floor(bvh):
    return bvh.foot_contacts, {"height_reference": "floor"}


def _run_adaptive(bvh):
    return bvh.foot_contacts, {"adaptive": True}


def _run_clip_centered_first(bvh):
    # The clip moved as node_positions(centered="first_ground") moves its positions:
    # its first frame's root to the origin in the ground plane. The pin was
    # taken by passing those positions as coords=, which the detectors no
    # longer take; the clip moved the same way gives the same labels.
    up = bvh.world_up_axis.index
    ground = [axis for axis in range(3) if axis != up]
    root_pos = bvh.root_pos.copy()
    root_pos[:, ground] -= root_pos[0, ground]
    bvh.root_pos = root_pos
    return bvh.foot_contacts, {}


def _run_no_morphology(bvh):
    return bvh.foot_contacts, {
        "hysteresis": 0.0,
        "min_contact_duration": 0.0,
        "min_gap_duration": 0.0,
    }


def _run_explicit_thresholds(bvh):
    # Explicit floats near the clip's auto-calibrated defaults (so the labels
    # stay meaningful); pins that the report's skeleton_scale is reported
    # though no threshold was derived from it, since the foot skate uses it.
    return bvh.foot_contacts, {"vel_threshold": 2.0, "height_threshold": 0.5}


def _run_ground_contacts_by_name(bvh):
    # ground_contacts under its own defaults (height_reference="floor", no
    # rest-height check), on the ankles: nodes named, not auto-detected.
    return bvh.ground_contacts, {"nodes": ["LeftFoot", "RightFoot"]}


def _run_ground_contacts_by_node(bvh):
    # The toe tips as nodes: end sites, found by their place in the tree (an
    # end site's name is generated), and named by their own names in the
    # report's joints. The pin was taken with their node indices, which the
    # detectors no longer take; the nodes select the same rows.
    toes = bvh.auto_detect_feet()
    tips = [next(node for node in bvh.nodes if node.parent is toe) for toe in toes]
    return bvh.ground_contacts, {"nodes": tips}


def _run_method_height_floor_min(bvh):
    # Under the default method the stance calibration absorbs the floor's
    # level and the contacts equal the default run's; under "height" they
    # depend on it.
    return bvh.foot_contacts, {"method": "height", "floor": "min"}


def _run_method_height_adaptive(bvh):
    return bvh.foot_contacts, {"method": "height", "adaptive": True}


def _run_method_height_adaptive_first_second(bvh):
    # Over the whole walk the height split falls back to the margin on both
    # feet (the run above); over its first second it is accepted on the right
    # foot and falls back on the left, so one call pins both outcomes.
    first_second = bvh[: round(1.0 / bvh.frame_time)]
    return first_second.foot_contacts, {"method": "height", "adaptive": True}


def _run_one_frame(bvh):
    # A one-frame clip, as a pose file is: no speed can be measured.
    return bvh[0].foot_contacts, {}


FOOT_CONTACT_RUNS = [
    ("default", _run_default),
    ("explicit_feet_reversed", _run_explicit_feet_reversed),
    ("method_velocity", _run_method_velocity),
    ("method_height_floor0", _run_method_height_floor0),
    ("height_reference_floor", _run_height_reference_floor),
    ("adaptive", _run_adaptive),
    ("clip_centered_first", _run_clip_centered_first),
    ("no_morphology", _run_no_morphology),
    ("explicit_thresholds", _run_explicit_thresholds),
    ("ground_contacts_by_name", _run_ground_contacts_by_name),
    ("ground_contacts_by_node", _run_ground_contacts_by_node),
    ("method_height_floor_min", _run_method_height_floor_min),
    ("method_height_adaptive", _run_method_height_adaptive),
    ("one_frame", _run_one_frame),
    ("method_height_adaptive_first_second", _run_method_height_adaptive_first_second),
]


def call_pinned_run(build, bvh) -> tuple:
    """Make one pinned call: the detector ``build`` names, with its kwargs and ``return_info=True``; returns ``(contacts, report)``."""
    detector, kwargs = build(bvh)
    return detector(return_info=True, **kwargs)


def report_fields(report) -> dict:
    """A ``ContactReport`` as ``{field: value}``, every field included, ``None`` ones too: the view the pin fixture stores and the pin tests compare.

    The tested nodes are given by their names, which an npz can hold: the pin compares what was tested, and the tests of the report check that its ``nodes`` are the clip's own.
    """
    fields = {field.name: getattr(report, field.name) for field in dataclasses.fields(report)}
    fields["nodes"] = tuple(node.name for node in report.nodes)
    return fields


def gen_foot_contacts() -> None:
    # ------------------------------------------------------------------
    # !! BEHAVIOR PIN — regenerating RE-BASELINES it !!
    #
    # Unlike every other fixture in this file, this one has NO external
    # reference: it freezes pybvh's OWN contact detection output
    # (foot_contacts and ground_contacts: contacts + the full report) so
    # a refactor can be proven bit-identical.
    # The committed .npz must come from the PRE-refactor tree. If the
    # pin test fails after a code change, the change altered behavior —
    # rerunning this generator does not "fix" that, it silently erases
    # the evidence. Regenerate only to deliberately re-baseline.
    # ------------------------------------------------------------------
    repo_root = os.path.dirname(os.path.dirname(HERE))
    # Pin the CURRENT tree's pybvh, never an installed copy.
    sys.path.insert(0, repo_root)
    from pybvh import read_bvh_file

    bvh_path = os.path.join(repo_root, "bvh_data", "cmu_12_01_walk.bvh")
    arrays: dict[str, np.ndarray] = {}
    for i, (_name, build) in enumerate(FOOT_CONTACT_RUNS, start=1):
        # Fresh Bvh per run: no floor-cache state carries over between runs.
        bvh = read_bvh_file(bvh_path)
        contacts, report = call_pinned_run(build, bvh)
        fields = report_fields(report)
        arrays[f"run{i}/contacts"] = contacts
        arrays[f"run{i}/__keys__"] = np.array(sorted(fields))
        # An npz holds no None: the None fields are listed, the others stored.
        arrays[f"run{i}/__none__"] = np.array(
            sorted(key for key, value in fields.items() if value is None), dtype=str
        )
        for key, value in fields.items():
            if value is not None:
                arrays[f"run{i}/{key}"] = np.asarray(value)
    _save(
        "foot_contacts_pinned",
        {
            "ref": "pybvh itself — BEHAVIOR PIN, no external reference",
            "source": "bvh_data/cmu_12_01_walk.bvh",
            "runs": [name for name, _ in FOOT_CONTACT_RUNS],
            "env": "the development .venv (numpy-only), NOT the fixture env",
            "warning": "regenerating re-baselines the pin; the committed "
            "fixture must come from the pre-refactor tree",
        },
        **arrays,
    )


FOLLOW_AZIMUTH_BASE = -20.0


def gen_follow_azimuths() -> None:
    # ------------------------------------------------------------------
    # !! BEHAVIOR PIN — regenerating RE-BASELINES it !!
    #
    # Same kind as gen_foot_contacts: no external reference, it freezes
    # the follow camera's per-frame azimuths on the CMU walk so that a
    # refactor of the facing geometry or of the viewport can be proven
    # not to move the camera. Seven of its values are also written out
    # in tests/test_plot.py; a regenerated fixture that disagrees with
    # them fails there. Regenerate only to deliberately re-baseline, as
    # when the schedule began smoothing out the stride's sway (#22).
    # ------------------------------------------------------------------
    repo_root = os.path.dirname(os.path.dirname(HERE))
    sys.path.insert(0, repo_root)
    from pybvh import read_bvh_file
    from pybvh.bvhplot._from_bvh import make_scene
    from pybvh.bvhplot._viewport import compute_follow_azimuths

    bvh = read_bvh_file(os.path.join(repo_root, "bvh_data", "cmu_12_01_walk.bvh"))
    view = make_scene([bvh], [bvh.node_positions()], "front", None).views[0]
    _save(
        "follow_azimuths_pinned",
        {
            "ref": "pybvh itself — BEHAVIOR PIN, no external reference",
            "source": "bvh_data/cmu_12_01_walk.bvh",
            "quantity": "compute_follow_azimuths(view, base_azim), degrees, one per frame",
            "base_azim": FOLLOW_AZIMUTH_BASE,
            "env": "the development .venv (numpy-only), NOT the fixture env",
            "warning": "regenerating re-baselines the pin",
        },
        azimuths=compute_follow_azimuths(view, FOLLOW_AZIMUTH_BASE),
    )


def lr_pin_clip_paths(repo_root: str) -> list[str]:
    """The bundled clips the L/R pin covers: every ``.bvh`` of ``bvh_data/``
    and ``bvh_data/synthetic/``, as paths relative to *repo_root*, sorted."""
    paths = []
    for folder in ("bvh_data", os.path.join("bvh_data", "synthetic")):
        for name in sorted(os.listdir(os.path.join(repo_root, folder))):
            if name.endswith(".bvh"):
                paths.append(os.path.join(folder, name))
    return paths


# The fields of OrientationReport that hold a string or None and that the
# L/R pairs feed, pinned as they are; body_anchor and body_anchor_candidates,
# which hold nodes, are pinned as node indices. world_up_inherited is left
# out: the pairs do not feed it.
REPORT_FIELDS = (
    "rest_up",
    "rest_up_source",
    "rest_left",
    "rest_forward",
    "rest_forward_source",
    "world_up",
    "world_up_source",
    "lr_mapping_source",
    "body_anchor_source",
)


def lr_pin_values(bvh) -> dict[str, np.ndarray]:
    """Every value of *bvh* that the L/R pairs feed, as arrays.

    ``lr_pairs``, ``node_lr_pairs``, the body anchor, the fields of the
    orientation report that the pairs feed (``REPORT_FIELDS``, the body
    anchor and its candidates), the facing and the result of ``mirror``
    (with the clip's own mapping, and with it passed explicitly, the strict
    path). A value
    that raises is stored as the string ``"<ExceptionType>: <message>"``,
    and ``None`` as the string ``"None"``. ``tests/test_lr_pairs_pin.py``
    replays it against the fixture.
    """
    from pybvh import transforms

    def guarded(read):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return read()
        except Exception as error:  # noqa: BLE001 - the error is the pinned value
            return f"{type(error).__name__}: {error}"

    def as_array(value):
        if value is None:
            return np.array("None")
        if isinstance(value, str):
            return np.array(value)
        return np.asarray(value)

    values: dict[str, object] = {
        "lr_pairs": guarded(lambda: bvh.lr_pairs),
        "node_lr_pairs": guarded(lambda: bvh.node_lr_pairs),
        "body_anchor": guarded(
            lambda: None if bvh.body_anchor is None else bvh.nodes.index(bvh.body_anchor)
        ),
    }
    # The documented fields of the orientation report that the pairs feed,
    # each its own key, a node as its index in ``bvh.nodes``.
    report = guarded(lambda: bvh.orientation_report)
    if isinstance(report, str):
        values["orientation_report"] = report
    else:
        for field in REPORT_FIELDS:
            values[f"orientation_report/{field}"] = getattr(report, field)
        values["orientation_report/body_anchor"] = (
            None if report.body_anchor is None else bvh.nodes.index(report.body_anchor)
        )
        values["orientation_report/body_anchor_candidates"] = np.array(
            [bvh.nodes.index(node) for node in report.body_anchor_candidates], dtype=np.int64
        )
    # Every frame would make the fixture megabytes; the pairs act on each
    # frame alike, so about twenty evenly spaced frames pin them as well.
    frames = slice(None, None, max(1, bvh.frame_count // 20))
    facing = guarded(bvh.facing_frame)
    if isinstance(facing, str):
        values["facing"] = facing
    else:
        for field in ("forward", "left", "up", "valid"):
            values[f"facing/{field}"] = getattr(facing, field)[frames]
    for label, kwargs in (("mirror", {}), ("mirror_explicit", {"lr_mapping": bvh.lr_mapping})):
        mirrored = guarded(lambda kwargs=kwargs: transforms.mirror(bvh, **kwargs))
        if isinstance(mirrored, str):
            values[label] = mirrored
        else:
            values[f"{label}/rest_pose_positions"] = mirrored.rest_pose_positions()
            values[f"{label}/root_pos"] = mirrored.root_pos[frames]
            values[f"{label}/joint_angles"] = mirrored.joint_angles[frames]
            values[f"{label}/joint_translations"] = mirrored.joint_translations[frames]
    return {key: as_array(value) for key, value in values.items()}


def gen_lr_pairs() -> None:
    # ------------------------------------------------------------------
    # !! BEHAVIOR PIN — regenerating RE-BASELINES it !!
    #
    # Same kind as gen_foot_contacts: no external reference, it freezes
    # every value the L/R pairs feed, on every bundled clip, so that the
    # resolver of the name pairs (#216) can be proven bit-identical. The
    # committed .npz must come from the tree before that change.
    # ------------------------------------------------------------------
    repo_root = os.path.dirname(os.path.dirname(HERE))
    sys.path.insert(0, repo_root)
    from pybvh import read_bvh_file

    arrays: dict[str, np.ndarray] = {}
    clips = lr_pin_clip_paths(repo_root)
    for path in clips:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bvh = read_bvh_file(os.path.join(repo_root, path))
        for key, value in lr_pin_values(bvh).items():
            arrays[f"{path}/{key}"] = value
    _save(
        "lr_pairs_pinned",
        {
            "ref": "pybvh itself — BEHAVIOR PIN, no external reference",
            "source": clips,
            "quantity": "lr_pin_values(bvh) per bundled clip",
            "env": "the development .venv (numpy-only), NOT the fixture env",
            "warning": "regenerating re-baselines the pin; the committed "
            "fixture must come from the tree before the change it guards",
        },
        **arrays,
    )


if __name__ == "__main__":
    if "--foot-contacts-pin" in sys.argv:
        print("RE-BASELINING the foot_contacts behavior pin in", HERE)
        gen_foot_contacts()
    elif "--follow-azimuths-pin" in sys.argv:
        print("RE-BASELINING the follow azimuths behavior pin in", HERE)
        gen_follow_azimuths()
    elif "--lr-pairs-pin" in sys.argv:
        print("RE-BASELINING the L/R pairs behavior pin in", HERE)
        gen_lr_pairs()
    else:
        print("Generating golden fixtures into", HERE)
        gen_rotations()
        gen_se3()
        gen_geodesic()
        gen_smoothness()  # fetches the SPARC reference offline; commits numbers only
    print("done.")
