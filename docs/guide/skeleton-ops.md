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

`to_df_dict()` exports the motion as labeled columns; `to_node_table()` exports the skeleton as a node table — one plain dict per node in `nodes` order, with the parent given as an index. Together they carry everything needed to rebuild the Bvh with `Bvh.from_df(table, df)`, which also takes `bvh.nodes` in place of the table (the module-level `pybvh.df_to_bvh` is the same function):

```python
import pandas as pd
from pybvh import Bvh

df = pd.DataFrame(bvh.to_df_dict(mode="euler"))
table = bvh.to_node_table()   # [{name, parent, offset, rot_channels, ...}, ...]

bvh_from_table = Bvh.from_df(table, df)
bvh_from_nodes = Bvh.from_df(bvh.nodes, df)
```

The skeleton comes back exactly: the table identifies nodes by position, not by name, so a joint with two end sites and joints sharing a name come back as they were, with their names, offsets and channel orders. The motion comes back within float precision, because the `_rot` columns carry degrees and `from_df` converts them back to radians, and the frame time is derived from the `time` column (the Notes of `df_to_bvh`'s docstring give both rules). In the columns, a repeated node name is labelled the way pandas labels repeated CSV headers — the first keeps its name, later ones get `.1`, `.2` (`Arm_Z_rot`, `Arm.1_Z_rot`) — and the suffix is a column label only: the nodes keep their names. `from_df` binds the columns by label, so they may come in any order and columns outside the expected set are ignored. `to_df_dict(mode="coordinates")` is one-way: positions do not determine joint angles, and `from_df` reads euler-mode frames only. A skeleton built by hand takes the same route — [`nodes_from_table`](../api/index.md#pybvh.nodes_from_table) builds the node list from a table and [`nodes_to_table`](../api/index.md#pybvh.nodes_to_table) writes one back.

## Editing a skeleton

The nodes a `Bvh` holds are frozen: assigning a node's `name`, `offset`, `parent`, `children` or channels raises `AttributeError`, and `node.offset` is a read-only array. The clip computes its forward kinematics, its rest pose and its name indexes from the skeleton, so a node edited in place would leave them describing the old one. `bvh.nodes` is a tuple, `node.children` a tuple once the clip owns the node, and channel orders are tuples (`('Z', 'Y', 'X')`); take `list(...)` where you need a list, and `node.offset.copy()` for a writable offset.

To change a skeleton, edit it as a node table and build a new clip from it:

```python
from pybvh import Bvh, nodes_to_table, nodes_from_table

table = nodes_to_table(bvh.nodes)    # plain dicts, one per node, in `nodes` order
i = bvh.node_index["Spine"]
table[i]["offset"] = table[i]["offset"] * 1.1
table[i]["name"] = "Spine0"

edited = Bvh(nodes=nodes_from_table(table), root_pos=bvh.root_pos,
             joint_angles=bvh.joint_angles, frame_time=bvh.frame_time)
```

The table holds copies, so editing it leaves `bvh` as it was. Keep one row per joint of `joint_angles`, in the same order: the motion is matched to the joints by position. To keep a subset of the joints, use `extract_joints`, which drops their columns too. The new clip is built like a clip read from a file: it guesses its own world up and detects its L/R pairs from the joint names; pass `world_up=` or `lr_mapping=` to `Bvh(...)` to set them.

Nodes not yet handed to a `Bvh` stay editable, so a skeleton built by hand (nodes created, `parent`, `children` and `offset` set) works as before, as long as it is finished before it reaches `Bvh(...)`.

## Inplace convention

All mutation methods default to `inplace=False` (return a new Bvh):

```python
bvh2 = bvh.scale(0.01)                     # new object
bvh.scale(0.01, inplace=True)              # modifies self, returns None
```

A skeleton edit made in place (`scale`, `mirror`, `retarget`, `change_euler_order`, the reorientations) swaps a new skeleton into the clip: a node, or `bvh.nodes`, taken before the call keeps the old values, so read `bvh.nodes` again after it.

!!! info "See also"
    [Gallery](../gallery/index.md) — `scale`/`retarget`/`extract_joints` and `resample` drawn (section 4) · [Bvh Class API](../api/bvh.md) — full signatures · [Core Concepts](core-concepts.md) — the index spaces these operations preserve
