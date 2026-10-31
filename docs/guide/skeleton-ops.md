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

`to_df_dict()` exports the motion as labeled columns. `to_node_table()` exports the skeleton as a node table: one plain dict per node, in `nodes` order, with the parent given as an index. Together they carry everything needed to rebuild the Bvh with `Bvh.from_df(table, df)`. `from_df` also takes `bvh.nodes` in place of the table. The module-level `pybvh.df_to_bvh` is the same function:

```python
import pandas as pd
from pybvh import Bvh

df = pd.DataFrame(bvh.to_df_dict(mode="euler"))
table = bvh.to_node_table()   # [{name, parent, offset, rot_channels, ...}, ...]

bvh_from_table = Bvh.from_df(table, df)
bvh_from_nodes = Bvh.from_df(bvh.nodes, df)
```

The skeleton comes back exactly. The table identifies nodes by position, not by name. So a joint with two end sites, and joints that share a name, come back as they were, with their names, offsets and channel orders.

The motion comes back within float precision. The reason is that the `_rot` columns carry degrees, and `from_df` converts them back to radians. `from_df` derives the frame time from the `time` column. The Notes of `df_to_bvh`'s docstring give both rules.

In the columns, a repeated node name gets the label that pandas gives repeated CSV headers. The first keeps its name, and later ones get `.1`, `.2` (`Arm_Z_rot`, `Arm.1_Z_rot`). The suffix is a column label only: the nodes keep their names. `from_df` binds the columns by label. So the columns may come in any order, and `from_df` ignores columns outside the expected set.

`to_df_dict(mode="coordinates")` is one-way. Positions do not determine joint angles. `from_df` reads euler-mode frames only.

A skeleton built by hand takes the same route. [`nodes_from_table`](../api/index.md#pybvh.nodes_from_table) builds the node list from a table. [`nodes_to_table`](../api/index.md#pybvh.nodes_to_table) writes one back.

The two forms follow the rule of `Bvh(...)` and `copy(nodes=...)`. A table is data, so `from_df` builds the clip's own nodes from it. The clip takes over a node list: it holds those node objects and freezes them. So `bvh_from_nodes` above shares `bvh`'s nodes. And a node that you built and passed in raises `AttributeError` when you edit it afterwards. If you mean to keep editing the nodes, pass `nodes_to_table(nodes)` instead.

## Editing a skeleton

The nodes that a `Bvh` holds are frozen. An assignment to a node's `name`, `offset`, `parent`, `children` or channels raises `AttributeError`. `node.offset` is a read-only array. The clip computes its forward kinematics, its rest pose and its name indexes from the skeleton. So a node edited in place would leave them describing the old one.

`bvh.nodes` is a tuple. `node.children` is a tuple once the clip owns the node. Channel orders are tuples (`('Z', 'Y', 'X')`). Where you need a list, take `list(...)`. For a writable offset, take `node.offset.copy()`.

To change a skeleton, edit it as a node table and copy the clip onto it:

```python
table = bvh.to_node_table()    # plain dicts, one per node, in `nodes` order
i = bvh.node_index["Spine"]
table[i]["offset"] = table[i]["offset"] * 1.1
table[i]["name"] = "Spine0"

edited = bvh.copy(nodes=table)
```

The table holds copies, so editing it leaves `bvh` as it was. `copy(nodes=...)` also takes a node list, such as another clip's `nodes` or one built with `nodes_from_table`. It takes the list over as `Bvh(...)` does. The copy owns those nodes and freezes them. So a later edit to one of them raises, instead of missing the clip. Nodes that another clip owns are shared, because they are frozen already. A copy that raises leaves the list editable. A table is data, so it gives the copy nodes of its own. The copy carries the motion, the frame time, the source path and your settings, as `copy()` does.

The motion is matched to the joints by position. So keep both of these:

- One row per joint of `joint_angles`, in the same order.
- Each joint's Euler order.

A different joint count raises `ValueError`. So does a changed order, because the angles are only meaningful in the order they were recorded in. `change_euler_order` converts them with the order. To keep a subset of the joints, use `extract_joints`, which drops their columns too. An L/R mapping that you set follows its joints by position. So a renamed joint keeps its pair under the new name. Pairs detected from the names are detected again from the new names.

World up is inherited, set or guessed, as by every clip pybvh derives, because new offsets do not change which way is up. `edited.world_up = edited.world_up_inferred` guesses it from the new skeleton.

To build a clip from a skeleton and motion that do not come from one clip, pass them to the constructor: `Bvh(nodes=nodes_from_table(table), root_pos=..., joint_angles=..., frame_time=...)`. Such a clip is built like a clip read from a file. It guesses its own world up. It detects its L/R pairs from the joint names. To set them, pass `world_up=` or `lr_mapping=`.

Nodes not yet handed to a `Bvh` stay editable. So a skeleton built by hand (nodes created, `parent`, `children` and `offset` set) works as before, as long as it is finished before it reaches `Bvh(...)`.

## Inplace convention

All mutation methods default to `inplace=False` (return a new Bvh):

```python
bvh2 = bvh.scale(0.01)                     # new object
bvh.scale(0.01, inplace=True)              # modifies self, returns None
```

A skeleton edit made in place (`scale`, `mirror`, `retarget`, `change_euler_order`, the reorientations) swaps a new skeleton into the clip. A node, or `bvh.nodes`, taken before the call keeps the old values. So read `bvh.nodes` again after the call.

!!! info "See also"
    [Gallery](../gallery/index.md) — `scale`/`retarget`/`extract_joints` and `resample` drawn (section 4) · [Bvh Class API](../api/bvh.md) — full signatures · [Core Concepts](core-concepts.md) — the index spaces these operations preserve
