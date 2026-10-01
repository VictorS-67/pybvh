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

The three skeleton operations on one shared scale — scaled to half, retargeted back to the original proportions, and reduced to 11 joints:

![Four skeleton variants drawn at one shared scale: original, half-size scale, a tall clip retargeted to the original proportions, and an 11-joint extraction](../gallery/img/skeleton-ops.png)

## Frame operations

```python
clip = bvh[10:50]              # frame slicing (steps work too: bvh[::2])
combined = bvh + other_bvh     # concatenation (same skeleton required)
bvh_30fps = bvh.resample(30)
```

## Pandas integration

`to_df_dict()` exports the motion as labeled columns; `to_node_table()` exports the skeleton as a node table — one plain dict per node in `nodes` order, with the parent given as an index. Together they carry everything needed to rebuild the Bvh with `Bvh.from_df(table, df)`, which also takes the `bvh.nodes` list in place of the table (the module-level `pybvh.df_to_bvh` is the same function):

```python
import pandas as pd
from pybvh import Bvh

df = pd.DataFrame(bvh.to_df_dict(mode="euler"))
table = bvh.to_node_table()   # [{name, parent, offset, rot_channels, ...}, ...]

bvh_from_table = Bvh.from_df(table, df)
bvh_from_nodes = Bvh.from_df(bvh.nodes, df)
```

The skeleton comes back exactly: the table identifies nodes by position, not by name, so a joint with two end sites and joints sharing a name come back as they were, with their names, offsets and channel orders. The motion comes back within float precision, because the `_rot` columns carry degrees and `from_df` converts them back to radians, and the frame time is derived from the `time` column (the Notes of `df_to_bvh`'s docstring give both rules). In the columns, a repeated node name is labelled the way pandas labels repeated CSV headers — the first keeps its name, later ones get `.1`, `.2` (`Arm_Z_rot`, `Arm.1_Z_rot`) — and the suffix is a column label only: the nodes keep their names. `from_df` binds the columns by label, so they may come in any order and columns outside the expected set are ignored. `to_df_dict(mode="coordinates")` is one-way: positions do not determine joint angles, and `from_df` reads euler-mode frames only. A skeleton built by hand takes the same route — [`nodes_from_table`](../api/index.md#pybvh.nodes_from_table) builds the node list from a table and [`nodes_to_table`](../api/index.md#pybvh.nodes_to_table) writes one back.

## Inplace convention

All mutation methods default to `inplace=False` (return a new Bvh):

```python
bvh2 = bvh.scale(0.01)                     # new object
bvh.scale(0.01, inplace=True)              # modifies self, returns None
```

!!! info "See also"
    [Gallery](../gallery/index.md) — `scale`/`retarget`/`extract_joints` and `resample` drawn (section 4) · [Bvh Class API](../api/bvh.md) — full signatures · [Core Concepts](core-concepts.md) — the index spaces these operations preserve
