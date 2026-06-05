#!/usr/bin/env python3
"""Generate golden reference fixtures for pybvh's differential tests.

Run ONCE, offline, in the `pybvh_test` conda env (which has the reference
libraries). The committed `.npz` outputs are then loaded by numpy-only tests
in the `pybvh` env — so CI validates pybvh against scipy/pytransform3d WITHOUT
those libraries being a runtime (or even CI) dependency.

    conda run -n pybvh_test python tests/fixtures/generate_fixtures.py

Each fixture stores its INPUTS, the reference OUTPUTS, and a `meta` JSON string
documenting the exact convention mapping (the antidote to "definition drift":
a future mismatch should be debugged against this meta, not assumed to be a bug).

References used:
- scipy.spatial.transform.Rotation  (rotation conversions)
- pytransform3d                      (SE(3) exp/log/interp — added when those land)
"""
from __future__ import annotations
import json
import os

import numpy as np
from scipy.spatial.transform import Rotation as SR
import pytransform3d.transformations as pt

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 0xB7  # fixed: fixtures must be reproducible


def pybvh_quat_from_scipy(rot: SR) -> np.ndarray:
    """scipy quat (x,y,z,w) -> pybvh convention (w,x,y,z), canonical w>=0."""
    q = np.atleast_2d(rot.as_quat())          # (...,4) scalar-LAST
    q = np.stack([q[..., 3], q[..., 0], q[..., 1], q[..., 2]], axis=-1)
    q = np.where(q[..., 0:1] < 0.0, -q, q)    # canonical w>=0 (scalar-first)
    return q


def _save(name: str, meta: dict, **arrays) -> None:
    np.savez(os.path.join(HERE, name + ".npz"),
             meta=json.dumps(meta), **arrays)
    print(f"  wrote {name}.npz  ({', '.join(arrays)})")


def gen_rotations() -> None:
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

    _save("rotmat_to_quat",
          {"ref": "scipy.spatial.transform.Rotation",
           "quat_convention": "(w,x,y,z) scalar-first, canonical w>=0",
           "angle_range_rad": [0.0, 0.85 * np.pi], "seed": hex(SEED)},
          rotmat=mats, quat_wxyz=pybvh_quat_from_scipy(rots))

    _save("rotmat_to_axisangle",
          {"ref": "scipy.spatial.transform.Rotation.as_rotvec",
           "axisangle_convention": "rotvec = axis * angle, angle in [0, pi]",
           "seed": hex(SEED)},
          rotmat=mats, rotvec=rots.as_rotvec())

    # --- euler -> rotmat: forward direction is unique for ANY angle, so use
    #     full range. pybvh euler = intrinsic, pre-mult R=R_first@..., order
    #     string read left-to-right; scipy uppercase seq = intrinsic, same. ---
    eul = np.concatenate([np.zeros((1, 3)),
                          rng.uniform(-np.pi, np.pi, size=(96, 3))], axis=0)
    order = "ZYX"
    rotmat = SR.from_euler(order, eul, degrees=False).as_matrix()
    _save("euler_zyx_to_rotmat",
          {"ref": "scipy.spatial.transform.Rotation.from_euler",
           "euler_convention": "intrinsic, R=R_Z@R_Y@R_X, radians, "
                               "first angle <-> first axis letter",
           "order": order, "seed": hex(SEED)},
          euler=eul, rotmat=rotmat)


def _se3_twists() -> np.ndarray:
    """Curated se(3) twists [omega(3), v(3)] stressing the cases that break
    implementations: theta->0 (V left-Jacobian Taylor series), theta->pi (log
    branch), and large translation with small rotation (V coupling)."""
    rng = np.random.default_rng(SEED ^ 0x5E3)
    thetas = [0.0, 1e-9, 1e-6, 1e-3, 1e-2, 0.5, np.pi / 2, 2.3,
              np.pi - 1e-6, np.pi - 1e-9, np.pi]
    axes = [np.array([1., 0, 0]), np.array([0, 0, 1.]),
            np.array([1, 1, 1.]) / np.sqrt(3)]
    vs = [np.zeros(3), np.array([1., 0, 0]), np.array([0., -3., 2.]),
          np.array([100., 0., 0.]),          # large translation -> V coupling
          np.array([5., -5., 5.])]
    rows = [np.concatenate([a * th, v]) for th in thetas for a in axes for v in vs]
    for _ in range(24):                        # general random twists
        a = rng.normal(size=3); a /= np.linalg.norm(a)
        th = rng.uniform(0.05, np.pi - 0.05)
        rows.append(np.concatenate([a * th, rng.normal(size=3) * rng.uniform(0.1, 10)]))
    return np.asarray(rows, dtype=float)


def gen_se3() -> None:
    twist = _se3_twists()                       # (N,6) [omega, v], rotation-first
    transform = np.array([pt.transform_from_exponential_coordinates(x) for x in twist])
    theta = np.linalg.norm(twist[:, :3], axis=1)
    _save("se3_exp_log",
          {"ref": "pytransform3d.transformations",
           "twist_order": "[omega(3), v(3)] rotation-first (Modern Robotics / Vemulapalli)",
           "coupling": "v is V-left-Jacobian-coupled, NOT raw translation",
           "edge_thetas": "0, 1e-9..1e-2, pi/2, pi-eps, pi", "seed": hex(SEED)},
          twist=twist, transform=transform, theta=theta)

    rng = np.random.default_rng(SEED ^ 0x5C2)
    T0, T1, tval, interp = [], [], [], []
    for _ in range(6):
        A = pt.transform_from_exponential_coordinates(rng.normal(size=6))
        B = pt.transform_from_exponential_coordinates(rng.normal(size=6))
        rel = pt.exponential_coordinates_from_transform(np.linalg.inv(A) @ B)
        for t in (0.0, 0.25, 0.5, 0.75, 1.0):
            T0.append(A); T1.append(B); tval.append(t)
            interp.append(A @ pt.transform_from_exponential_coordinates(t * rel))
    _save("se3_screw_interp",
          {"ref": "pytransform3d", "definition": "A @ exp(t*log(inv(A)@B)) screw geodesic",
           "endpoints": "t=0 -> A, t=1 -> B", "seed": hex(SEED)},
          T0=np.array(T0), T1=np.array(T1), t=np.array(tval), interp=np.array(interp))


def gen_geodesic() -> None:
    rng = np.random.default_rng(SEED ^ 0x6E0)

    def rotvecs(n):
        a = rng.normal(size=(n, 3)); a /= np.linalg.norm(a, axis=1, keepdims=True)
        return a * rng.uniform(0.0, np.pi, size=(n, 1))

    rv1 = np.concatenate([rotvecs(48), np.array([[0., 0, 0], [0., 0, 0]])])
    rv2 = np.concatenate([rotvecs(48), np.array([[0., 0, 0], [np.pi - 1e-6, 0, 0]])])
    R1, R2 = SR.from_rotvec(rv1), SR.from_rotvec(rv2)
    _save("rotation_geodesic",
          {"ref": "scipy: (R1.inv()*R2).magnitude()", "unit": "radians",
           "edges": "identical (0), near-pi", "seed": hex(SEED)},
          rotmat_a=R1.as_matrix(), rotmat_b=R2.as_matrix(),
          angle=(R1.inv() * R2).magnitude())


def _load_smoothness_reference():
    """Fetch the Balasubramanian SPARC reference (ISC license, siva82kb/SPARC),
    pinned to a commit, and import it. Its code is NOT committed — only the
    numbers it produces are. Needs network at regeneration time (rare)."""
    import importlib.util, tempfile, urllib.request
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
    minjerk = (t ** 2) * (1.0 - t) ** 2            # ~minimum-jerk speed bell
    minjerk /= minjerk.max()
    s0 = minjerk
    s1 = minjerk + np.roll(minjerk, 40)            # two overlapping segments
    s2 = minjerk + 0.25 * np.sin(30 * np.pi * t) ** 2
    s3 = minjerk + 0.10 * np.sin(8 * np.pi * t) + 0.05 * np.sin(22 * np.pi * t)
    return np.vstack([s0, s1, s2, s3]).astype(float), fs


def gen_smoothness() -> None:
    ref, url = _load_smoothness_reference()
    sig, fs = _smoothness_signals()
    sparc = np.array([ref.sparc(s, fs)[0] for s in sig])      # [0] = scalar SAL
    dlj = np.array([ref.dimensionless_jerk(s, fs) for s in sig])
    ldlj = np.array([ref.log_dimensionless_jerk(s, fs) for s in sig])
    _save("smoothness",
          {"ref": url, "license": "ISC (siva82kb/SPARC)",
           "sparc_params": {"padlevel": 4, "fc": 10.0, "amp_th": 0.05},
           "input": "1-D speed profile, fs Hz", "signals": "deterministic"},
          signals=sig, fs=np.array(fs), sparc=sparc, dlj=dlj, ldlj=ldlj)


if __name__ == "__main__":
    print("Generating golden fixtures into", HERE)
    gen_rotations()
    gen_se3()
    gen_geodesic()
    gen_smoothness()   # fetches the SPARC reference offline; commits numbers only
    print("done.")
