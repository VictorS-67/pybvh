# Golden reference fixtures

Frozen `.npz` arrays for pybvh's **differential tests** — pybvh outputs compared against an independent reference implementation on the same input.

The directory also holds two hand-written **parser-edge `.bvh` fixtures** used by `tests/test_bvh.py::TestReadWriteReadEquality` (no regeneration involved): `rotation_first_root.bvh` (a root that declares its rotation channels before its position channels) and `full_precision_frame_time.bvh` (a many-digit non-integer-rate `Frame Time` that must not be snapped or truncated).

It also holds `pr_hygiene_event.json`, a recorded `pull_request` event payload (PR #54, trimmed) that `tests/test_check_pr_hygiene.py` feeds to the pr-hygiene check.

And it holds `public_namespace.txt`, the names a user can import from the package and from each of its public modules, which `tests/test_public_namespace.py` compares with the live package; the test's docstring says which names count. After a deliberate change to the public surface, `python tests/test_public_namespace.py` rewrites the list, and the edit to it shows the change in the diff.

## Running the tests (no reference libraries needed)

The `.npz` fixtures here are **committed**, and the tests (`tests/test_*_golden.py`) only `np.load` them. So anyone who clones the repo can run the full suite with just the normal dev deps — **scipy / pytransform3d are *not* required to run tests**:

```bash
pip install -e ".[dev]"      # or: conda run -n pybvh ...
python -m pytest tests/ -v
```

This keeps pybvh numpy-only at runtime *and* in CI (the charter), while still validating against scipy-derived ground truth.

## Regenerating the fixtures (only when adding/changing one)

The references are run **once, offline**, in a dedicated env, and the outputs are committed. Recreate that env reproducibly — conda (pinned) **or** pip:

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
conda run -n pybvh python -m pytest tests/ -k golden -v
```

Pinned reference versions (see `environment.yml`): scipy 1.17, pytransform3d 3.15, numpy 2.4 — pin so regeneration is deterministic (no version drift in the golden values).

## Convention discipline (important)

Each fixture embeds a `meta` JSON string documenting the exact convention mapping (quaternion order, Euler intrinsic/extrinsic, angle range, seed). Differential testing's one real failure mode is **definition drift** — a quantity defined slightly differently in the reference than in pybvh. When a golden test fails, **read the `meta` first**: confirm it's a real bug, not a convention gap, before "fixing" working code.

## Current fixtures

| File | Input → reference | Reference | Tested by |
|---|---|---|---|
| `euler_zyx_to_rotmat.npz` | Euler (ZYX, rad) → rotmat | scipy | `test_rotations_golden.py` (active) |
| `rotmat_to_quat.npz` | rotmat → quat (w,x,y,z) | scipy | active |
| `rotmat_to_axisangle.npz` | rotmat → rotvec | scipy | active |
| `se3_exp_log.npz` | twist `[ω,v]` ↔ 4×4 transform | pytransform3d | `test_se3_golden.py` (active) |
| `se3_screw_interp.npz` | (T0, T1, t) → screw geodesic | pytransform3d | active |
| `rotation_geodesic.npz` | (R1, R2) → angle | scipy | active |
| `smoothness.npz` | speed profile → SPARC / DLJ / LDLJ | siva82kb/SPARC (ISC) | `test_smoothness_golden.py` (active) |
| `foot_contacts_pinned.npz` | CMU walk clip → contacts + full `info` dict for 9 `foot_contacts` parameterizations | **pybvh itself (behavior pin)** | `test_analysis.py::TestFootContactsPinnedGolden` (active) |
| `follow_azimuths_pinned.npz` | CMU walk clip → the follow camera's azimuth per frame (degrees, base azimuth −20°) | **pybvh itself (behavior pin)** | `test_plot.py::TestComputeFollowAzimuths` (active) |

**`foot_contacts_pinned.npz` is a behavior pin, not a reference fixture:** it freezes pybvh's *own* `foot_contacts` output bit-exactly so refactors of the contacts machinery can be proven behavior-neutral. It is excluded from the default generator run and regenerates only via `conda run -n pybvh python tests/fixtures/generate_fixtures.py --foot-contacts-pin` — and doing so **re-baselines the pin**, so the committed file must come from the pre-refactor tree; never regenerate it to make a failing pin test pass.

`follow_azimuths_pinned.npz` is a second behavior pin of the same kind, of the follow camera's azimuth schedule: it proves that a refactor of the facing geometry or of the viewport does not move the camera. `test_plot.py::TestComputeFollowAzimuths` compares both `compute_follow_azimuths` and the viewport's `follow` schedule with it, and writes seven of its values out, so a re-baselined fixture fails there. It regenerates only via `conda run -n pybvh python tests/fixtures/generate_fixtures.py --follow-azimuths-pin`, which re-baselines the pin, under the same rule.

The SE(3) and smoothness fixtures were committed as pre-built oracles before the functions they test existed; those functions shipped in 0.8.0 (`rotations.se3_exp`, `se3_log`, `screw_interpolate`, `rotation_geodesic_distance`; `analysis.sparc`, `dimensionless_jerk`, `log_dimensionless_jerk`), and the tests now validate them. SE(3) fixtures deliberately over-cover the failure-prone regimes: θ→0 (V left-Jacobian Taylor), θ→π (log branch), pure translation, and large-translation V-coupling.

**Convention locks (pinned by these fixtures):** se(3) twist = `[ω(3), v(3)]` rotation-first, V-Jacobian-coupled (= pytransform3d / Vemulapalli 2014). SPARC defaults `padlevel=4, fc=10 Hz, amp_th=0.05`.

> **Note on the smoothness reference:** it's not on PyPI, so `gen_smoothness()` fetches `scripts/smoothness.py` from siva82kb/SPARC **at a pinned commit** (`7deff21…`) over the network at regeneration time, runs it offline, and commits only the numbers. Its code never enters this repo. So regenerating the smoothness fixture needs network (the others need only `environment.yml`).
