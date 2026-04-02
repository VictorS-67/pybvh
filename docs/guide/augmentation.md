# Data Augmentation

All augmentation transforms support seeded randomization for reproducibility. Available as both `Bvh` methods and standalone functions in `pybvh.transforms`.

## Transforms

### Left-right mirroring

```python
from pybvh import transforms

# Auto-detects joint pairs and lateral axis
bvh_mirrored = transforms.mirror(bvh)

# Or explicit
mapping = transforms.auto_detect_lr_mapping(bvh)  # name pairs
pairs = transforms.auto_detect_lr_pairs(bvh)       # index pairs
```

### Vertical rotation

```python
bvh_rotated = transforms.rotate_vertical(bvh, angle_deg=90)
bvh_random = transforms.random_rotate_vertical(bvh, rng=np.random.default_rng(42))
```

### Speed perturbation

```python
bvh_fast = transforms.speed_perturbation(bvh, factor=1.5)  # 1.5x faster
bvh_slow = transforms.speed_perturbation(bvh, factor=0.7)  # slower
```

### Noise injection

```python
bvh_noisy = transforms.add_joint_noise(bvh, sigma_deg=1.0, sigma_pos=0.5, rng=rng)
```

### Root translation

```python
bvh_shifted = transforms.translate_root(bvh, offset=[100, 0, 0])
```

### Frame dropout

```python
bvh_dropped = transforms.dropout_frames(bvh, drop_rate=0.1, rng=rng)
```

## Array-level functions

For ML pipelines that work with pre-extracted arrays (not Bvh objects):

```python
from pybvh.transforms import rotate_angles_vertical, mirror_angles

# Operate directly on (F, J, 3) Euler arrays
new_angles, new_pos = rotate_angles_vertical(
    bvh.joint_angles, bvh.root_pos, angle_deg=45,
    up_idx=1, root_order="ZYX")

# Mirror with index pairs
pairs = transforms.auto_detect_lr_pairs(bvh)
rot_ch = [list(n.rot_channels) for n in bvh.nodes if not n.is_end_site()]
m_angles, m_pos = mirror_angles(
    bvh.joint_angles, bvh.root_pos, pairs, lateral_idx=0, rot_channels=rot_ch)
```

!!! note "For quaternion/6D array augmentation"
    Array-level augmentation in quaternion and 6D space is provided by the companion library [pybvh-ml](https://github.com/VictorS-67/pybvh-ml).
