# Skeleton Operations

## Euler order conversion

```python
# Change all joints at once
bvh_xyz = bvh.change_euler_order("XYZ")

# Change a single joint only
bvh_single = bvh.change_euler_order("XYZ", joint="Hips")
```

## Skeleton scaling

```python
bvh_scaled = bvh.scale(0.01)   # uniform — scales offsets AND root translation
```

## Retargeting

```python
reference = pybvh.read_bvh_file("reference_skeleton.bvh")
bvh_retarget = bvh.retarget(reference)

# With name mapping (when joint names differ)
bvh_retarget = bvh.retarget(reference, name_mapping={
    "Hips": "pelvis", "Spine": "spine_01"
})
```

## Joint extraction

```python
upper = bvh.extract_joints(["Hips", "Spine", "Neck", "Head"])
```

## Frame operations

```python
clip = bvh[10:50]              # frame slicing (steps work too: bvh[::2])
combined = bvh + other_bvh     # concatenation (same skeleton required)
bvh_30fps = bvh.resample(30)
```

## Pandas integration

```python
import pandas as pd

df = pd.DataFrame(bvh.to_df_dict(mode="euler"))

from pybvh import df_to_bvh
bvh_from_df = df_to_bvh(bvh.nodes, df)
```

## Inplace convention

All mutation methods default to `inplace=False` (return a new Bvh):

```python
bvh2 = bvh.scale(0.01)                     # new object
bvh.scale(0.01, inplace=True)              # modifies self, returns None
```
