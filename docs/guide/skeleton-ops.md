# Skeleton Operations

## Euler order conversion

```python
bvh_xyz = bvh.change_all_euler_orders("XYZ")
bvh_single = bvh.single_joint_euler_angle("Hips", "XYZ")
```

## Skeleton scaling

```python
bvh_scaled = bvh.scale_skeleton(0.01)         # uniform
bvh_scaled = bvh.scale_skeleton([1, 2, 1])    # per-axis
```

## Retargeting

```python
reference = pybvh.read_bvh_file("reference_skeleton.bvh")
bvh_retarget = bvh.change_skeleton(reference)

# With name mapping (when joint names differ)
bvh_retarget = bvh.change_skeleton(reference, name_mapping={
    "Hips": "pelvis", "Spine": "spine_01"
})
```

## Joint extraction

```python
upper = bvh.extract_joints(["Hips", "Spine", "Neck", "Head"])
```

## Frame operations

```python
clip = bvh.slice_frames(10, 50)
combined = bvh.concat(other_bvh)
bvh_30fps = bvh.resample(30)
```

## Pandas integration

```python
import pandas as pd

df = pd.DataFrame(bvh.get_df_constructor(mode="euler"))

from pybvh import df_to_bvh
bvh_from_df = df_to_bvh(bvh.nodes, df)
```

## Inplace convention

All mutation methods default to `inplace=False` (return a new Bvh):

```python
bvh2 = bvh.scale_skeleton(0.01)                     # new object
bvh.scale_skeleton(0.01, inplace=True)               # modifies self, returns None
```
