# Data Augmentation

All augmentation transforms support seeded randomization for reproducibility. Available as both `Bvh` methods and standalone functions in `pybvh.transforms`.

## Transforms

### Left-right mirroring

```python
from pybvh import transforms

# Auto-detects joint pairs and lateral axis
bvh_mirrored = transforms.mirror(bvh)

# Or explicit
mapping = bvh.lr_mapping                      # name pairs (or None)
pairs = transforms.auto_detect_lr_pairs(bvh)  # index pairs
```

![Original and mirrored skeleton side by side: the pose is reflected and every Left/Right joint pair has swapped data](../gallery/img/mirror.png)

*Mirroring is not just an axis flip: the lateral axis is negated **and** every `Left*`/`Right*` pair swaps data, so the result stays anatomically possible. ([Gallery](../gallery/index.md), section 3, for every transform drawn.)*

#### L/R pair detection

pybvh auto-detects left/right joint pairs at load time by scanning joint names. The heuristic recognizes:

- `Left*` / `Right*` (any case), substring anywhere in the name — `LeftArm`, `leftArm`, `LeftEye`
- `L*` / `R*` prefix followed by an uppercase letter — `LArm`, `RArm`
- `*.L` / `*.R`, `*_L` / `*_R`, and their lowercase variants — `Arm.L`, `leg_r`
- `*.Left` / `*_Left` and lowercase variants — `Arm.Left`, `leg_right`
- Mixamo namespace prefix stripped automatically — `mixamorig:LeftArm`
- Blender numbered duplicates stripped automatically — `Arm.L.001`

The detected mapping lives on `bvh.lr_mapping` (a `dict[str, str]` or `None`). You can inspect, override, or provide it explicitly:

```python
# Inspect
print(bvh.lr_mapping)  # {"LeftArm": "RightArm", ...} or None

# Override post-load (B1 setter)
bvh.lr_mapping = {"arm.L": "arm.R", "leg.L": "leg.R"}

# Or provide at load time (B3 kwarg)
bvh = pybvh.read_bvh_file("weird.bvh", lr_mapping={"arm.L": "arm.R", ...})
bvh_list = pybvh.read_bvh_directory("dataset/", lr_mapping={...})  # applied to every file
```

If the heuristic can't parse the skeleton's naming, `bvh.lr_mapping` is `None` and `mirror()` raises a `ValueError` pointing at the remediation above. User-set mappings are lost on `bvh.write()` — re-apply after reading.

### Vertical rotation

Angles are in radians (like `Bvh.joint_angles`); pass `degrees=True` to work in degrees.

```python
bvh_rotated = transforms.rotate_vertical(bvh, np.pi / 2)
bvh_rotated = transforms.rotate_vertical(bvh, 90, degrees=True)   # same thing
bvh_random = transforms.random_rotate_vertical(bvh, rng=np.random.default_rng(42))  # uniform over (-pi, pi)
```

### Speed perturbation

```python
bvh_fast = transforms.perturb_speed(bvh, factor=1.5)  # 1.5x faster
bvh_slow = transforms.perturb_speed(bvh, factor=0.7)  # slower
```

### Noise injection

`sigma` is the rotation-noise standard deviation in radians. By default noised angles are left unwrapped (BVH channels can legitimately hold values outside `[-π, π]`); pass `wrap=True` to wrap them into `[-π, π]`.

```python
bvh_noisy = transforms.add_noise(bvh, sigma=0.02, sigma_pos=0.5, rng=rng)
```

### Root translation

```python
bvh_shifted = transforms.translate_root(bvh, offset=[100, 0, 0])
```

### Frame dropout

Dropped frames are re-synthesized by SLERP between the nearest kept neighbours; kept frames are preserved exactly.

```python
bvh_dropped = transforms.drop_frames(bvh, drop_rate=0.1, rng=rng)
```

## Composing transforms

All Bvh-level transforms return new Bvh objects (when `inplace=False`, the default), so they can be chained:

```python
rng = np.random.default_rng(42)

augmented = (bvh
    .mirror()
    .rotate_vertical(np.pi / 2)
    .add_noise(sigma=0.02, rng=rng)
    .perturb_speed(1.2))
```

For reproducible augmentation pipelines, pass a seeded `rng` to each stochastic transform. Deterministic transforms (`mirror`, `rotate_vertical`, `scale`) don't need one.

## Array-level functions

For ML pipelines that work with pre-extracted arrays (not Bvh objects):

```python
from pybvh.transforms import rotate_angles_vertical, mirror_angles

# Operate directly on (F, J, 3) Euler arrays
new_angles, new_pos = rotate_angles_vertical(
    bvh.joint_angles, bvh.root_pos, angle=np.pi / 4,
    up_idx=1, root_order="ZYX")

# Mirror with index pairs
pairs = transforms.auto_detect_lr_pairs(bvh)
rot_ch = [list(n.rot_channels) for n in bvh.nodes if not n.is_end_site()]
m_angles, m_pos = mirror_angles(
    bvh.joint_angles, bvh.root_pos, pairs, lateral_idx=0, rot_channels=rot_ch)
```

!!! note "For tensor-level augmentation pipelines"
    pybvh provides per-Bvh transforms in Euler space. For batched augmentation on packed tensors in quaternion or 6D space (e.g., during training), see the companion library [pybvh-ml](https://github.com/VictorS-67/pybvh-ml).

!!! info "See also"
    [Gallery](../gallery/index.md) — every transform drawn as before/after (section 3) · [Transforms API](../api/transforms.md) — full signatures · [World Up & Orientation](world-up.md) — the up-axis and L/R conventions `mirror` and `rotate_vertical` rely on
