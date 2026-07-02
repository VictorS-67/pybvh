# Golden reference fixtures

Frozen `.npz` arrays for pybvh's **differential tests** — pybvh outputs compared
against an independent reference implementation on the same input.

The directory also holds two hand-written **parser-edge `.bvh` fixtures** used by `tests/test_bvh.py::TestReadWriteReadEquality` (no regeneration involved): `rotation_first_root.bvh` (a root that declares its rotation channels before its position channels) and `full_precision_frame_time.bvh` (a many-digit non-integer-rate `Frame Time` that must not be snapped or truncated).

## Running the tests (no reference libraries needed)

The `.npz` fixtures here are **committed**, and the tests (`tests/test_*_golden.py`)
only `np.load` them. So anyone who clones the repo can run the full suite with
just the normal dev deps — **scipy / pytransform3d are *not* required to run tests**:

```bash
pip install -e ".[dev]"      # or: conda run -n pybvh ...
pytest tests/ -v
```

This keeps pybvh numpy-only at runtime *and* in CI (the charter), while still
validating against scipy-derived ground truth.

## Regenerating the fixtures (only when adding/changing one)

The references are run **once, offline**, in a dedicated env, and the outputs are
committed. Recreate that env reproducibly — conda (pinned) **or** pip:

```bash
# conda (pinned versions → reproducible reference values)
conda env create -f tests/fixtures/environment.yml
conda run -n pybvh_test python tests/fixtures/generate_fixtures.py

# …or pip, into any env
pip install -e ".[fixtures]"          # scipy + pytransform3d
python tests/fixtures/generate_fixtures.py
```

Then re-run the golden tests in the normal env:

```bash
conda run -n pybvh pytest tests/ -k golden -v
```

Pinned reference versions (see `environment.yml`): scipy 1.17, pytransform3d 3.15,
numpy 2.4 — pin so regeneration is deterministic (no version drift in the golden
values).

## Convention discipline (important)

Each fixture embeds a `meta` JSON string documenting the exact convention mapping
(quaternion order, Euler intrinsic/extrinsic, angle range, seed). Differential
testing's one real failure mode is **definition drift** — a quantity defined
slightly differently in the reference than in pybvh. When a golden test fails,
**read the `meta` first**: confirm it's a real bug, not a convention gap, before
"fixing" working code.

## Current fixtures

| File | Input → reference | Reference | Tested by |
|---|---|---|---|
| `euler_zyx_to_rotmat.npz` | Euler (ZYX, rad) → rotmat | scipy | `test_rotations_golden.py` (active) |
| `rotmat_to_quat.npz` | rotmat → quat (w,x,y,z) | scipy | active |
| `rotmat_to_axisangle.npz` | rotmat → rotvec | scipy | active |
| `se3_exp_log.npz` | twist `[ω,v]` ↔ 4×4 transform | pytransform3d | `test_se3_golden.py` (skips until `rotations.se3_exp/log`) |
| `se3_screw_interp.npz` | (T0, T1, t) → screw geodesic | pytransform3d | skips until `rotations.screw_interpolate` |
| `rotation_geodesic.npz` | (R1, R2) → angle | scipy | skips until `rotations.rotation_geodesic_distance` |
| `smoothness.npz` | speed profile → SPARC / DLJ / LDLJ | siva82kb/SPARC (ISC) | `test_smoothness_golden.py` (skips until `analysis.sparc` etc.) |

The SE(3)/smoothness tests are committed now (pre-built oracles) and **skip until
the corresponding functions exist**, then auto-validate. SE(3) fixtures deliberately
over-cover the failure-prone regimes: θ→0 (V left-Jacobian Taylor), θ→π (log branch),
pure translation, and large-translation V-coupling.

**Convention locks (pinned by these fixtures):** se(3) twist = `[ω(3), v(3)]`
rotation-first, V-Jacobian-coupled (= pytransform3d / Vemulapalli 2014). SPARC
defaults `padlevel=4, fc=10 Hz, amp_th=0.05`.

> **Note on the smoothness reference:** it's not on PyPI, so `gen_smoothness()`
> fetches `scripts/smoothness.py` from siva82kb/SPARC **at a pinned commit**
> (`7deff21…`) over the network at regeneration time, runs it offline, and commits
> only the numbers. Its code never enters this repo. So regenerating the smoothness
> fixture needs network (the others need only `environment.yml`).
