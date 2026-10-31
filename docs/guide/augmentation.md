# Data Augmentation

All augmentation transforms support seeded randomization for reproducibility. They are available as both `Bvh` methods and standalone functions in `pybvh.transforms`.

## Transforms

### Left-right mirroring

```python
from pybvh import transforms

# Auto-detects joint pairs and lateral axis
bvh_mirrored = transforms.mirror(bvh)

# Or explicit
mapping = bvh.lr_mapping  # name pairs (or None)
pairs = bvh.lr_pairs      # index pairs (or None)
```

![Original and mirrored skeleton side by side: the pose is reflected and every Left/Right joint pair has swapped data](../gallery/img/mirror.png)

*Mirroring is not just an axis flip. It negates the lateral axis, **and** every `Left*`/`Right*` pair swaps data. So the result stays anatomically possible. ([Gallery](../gallery/index.md), section 3, for every transform drawn.)*

#### L/R pair detection

pybvh auto-detects left/right joint pairs at load time by scanning joint names. The heuristic recognizes:

- `Left*` / `Right*` (any case), substring anywhere in the name — `LeftArm`, `leftArm`, `LeftEye`
- `L*` / `R*` prefix followed by an uppercase letter — `LArm`, `RArm`
- `*.L` / `*.R`, `*_L` / `*_R`, and their lowercase variants — `Arm.L`, `leg_r`
- `*.Left` / `*_Left` and lowercase variants — `Arm.Left`, `leg_right`
- Mixamo namespace prefix stripped automatically — `mixamorig:LeftArm`
- Blender numbered duplicates stripped automatically — `Arm.L.001`

The detected mapping is on `bvh.lr_mapping` (a `dict[str, str]` or `None`). You can inspect, override, or provide it explicitly:

```python
# Inspect
print(bvh.lr_mapping)  # {"LeftArm": "RightArm", ...} or None

# Override post-load (B1 setter)
bvh.lr_mapping = {"arm.L": "arm.R", "leg.L": "leg.R"}

# Or provide at load time (B3 kwarg)
bvh = pybvh.read_bvh_file("weird.bvh", lr_mapping={"arm.L": "arm.R", ...})
bvh_list = pybvh.read_bvh_directory("dataset/", lr_mapping={...})  # applied to every file
```

If the heuristic can't parse the skeleton's naming, `bvh.lr_mapping` is `None`. Then `mirror()` raises a `ValueError` that points at the remediation above. `bvh.write()` loses a mapping that you set. Set it again after you read the file.

### Vertical rotation

Angles are in radians, like `Bvh.joint_angles`. To work in degrees, pass `degrees=True`.

```python
bvh_rotated = transforms.rotate_vertical(bvh, np.pi / 2)
bvh_rotated = transforms.rotate_vertical(bvh, 90, degrees=True)   # same thing
bvh_random = transforms.random_rotate_vertical(bvh, rng=np.random.default_rng(42))  # uniform over (-pi, pi)
```

By default, the rotation turns about the **world origin**. So a character captured away from the origin sweeps along an arc, and it does not turn where it stands. To choose another point, use `pivot=`:

```python
bvh_turned = transforms.rotate_vertical(bvh, np.pi / 2, pivot="root")        # turn in place
bvh_turned = transforms.rotate_vertical(bvh, np.pi / 2, pivot=[100, 0, -50])  # about a fixed landmark
```

`pivot="root"` is the first frame's root position, projected to the ground plane. pybvh reads only the two horizontal components of an explicit point, because every point on a vertical line spans the same rotation axis. So the choice never alters heights. It never alters joint angles either: the pivot moves the root trajectory and nothing else.

Equivalently, `pivot="root"` is *center on the first-frame root → rotate about the origin → un-center*. So a pipeline that already centers its clips gets turn-in-place from the default, and it needs no `pivot=`.

### Speed perturbation

```python
bvh_fast = transforms.perturb_speed(bvh, factor=1.5)  # 1.5x faster
bvh_slow = transforms.perturb_speed(bvh, factor=0.7)  # slower
```

### Noise injection

Rotation noise and translation noise are separate calls, because their `sigma`s are in different units: radians for joint angles, and the skeleton's length unit for the root. By default, the noised angles stay unwrapped. BVH channels can legitimately hold values outside `[-π, π]`. To wrap the angles into `[-π, π]`, pass `wrap=True`.

```python
bvh_noisy = transforms.add_rotation_noise(bvh, sigma=0.02, rng=rng)      # radians
bvh_noisy = transforms.add_position_noise(bvh_noisy, sigma=0.5, rng=rng)  # length
```

Only `add_rotation_noise` takes `degrees=True`. A translation has no angular unit.

### Root translation

```python
bvh_shifted = transforms.translate_root(bvh, offset=[100, 0, 0])
```

### Frame dropout

Dropped frames are re-synthesized by SLERP between the nearest kept neighbours. The kept frames stay exactly as they were.

```python
bvh_dropped = transforms.drop_frames(bvh, drop_rate=0.1, rng=rng)
```

## Composing transforms

All Bvh-level transforms return new Bvh objects when `inplace=False`, the default. So you can chain them:

```python
rng = np.random.default_rng(42)

augmented = (bvh
    .mirror()
    .rotate_vertical(np.pi / 2)
    .add_rotation_noise(sigma=0.02, rng=rng)
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
    up_idx=1, root_order="ZYX")   # same pivot= as the Bvh-level twin

# Mirror with index pairs
pairs = bvh.lr_pairs or []
rot_ch = [n.rot_channels for n in bvh.nodes if not n.is_end_site()]
m_angles, m_pos = mirror_angles(
    bvh.joint_angles, bvh.root_pos, pairs, lateral_idx=0, rot_channels=rot_ch)
```

!!! note "For tensor-level augmentation pipelines"
    pybvh provides per-Bvh transforms in Euler space. For batched augmentation on packed tensors in quaternion or 6D space (e.g., during training), see the companion library [pybvh-ml](https://github.com/VictorS-67/pybvh-ml).

!!! info "See also"
    [Gallery](../gallery/index.md) — every transform drawn as before/after (section 3) · [Transforms API](../api/transforms.md) — full signatures · [World Up & Orientation](world-up.md) — the up-axis and L/R conventions `mirror` and `rotate_vertical` rely on
