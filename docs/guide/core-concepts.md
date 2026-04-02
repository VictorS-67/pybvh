# Core Concepts

## The BVH format

A BVH file has two sections:

- **HIERARCHY** — defines a skeleton as a tree of joints. Each joint has an offset (bone vector from parent) and rotation channels (e.g., `Zrotation Yrotation Xrotation`). Leaf nodes called "End Sites" have only an offset.
- **MOTION** — per-frame data. Each frame is a row of floats: root position (3 values) followed by Euler angles for every joint.

## The Bvh object

`Bvh` is the central container. It holds:

| Attribute | Shape | Description |
|---|---|---|
| `root_pos` | `(F, 3)` | Root translation per frame |
| `joint_angles` | `(F, J, 3)` | Euler angles (degrees) per joint per frame |
| `nodes` | list | All skeleton nodes (joints + end sites) in depth-first order |
| `joint_names` | list of str | Non-end-site joint names |
| `euler_orders` | list of str | Per-joint Euler orders (e.g. `['ZYX', ...]`) |
| `edges` | list of tuples | Skeleton edge list as `(child_idx, parent_idx)` |
| `frame_frequency` | float | Seconds per frame |

## Index spaces

pybvh has two index spaces:

- **`node_index`** — includes end sites. Maps to `bvh.nodes` list. Used by `get_spatial_coord()`.
- **`joint_angles` index** — excludes end sites. Maps to `bvh.joint_names` and `bvh.joint_angles` axis 1. Used by `edges`, `euler_orders`, augmentation functions.

Most functions use the `joint_angles` index space. The docstring of each function specifies which space it expects.

## Centered modes

Several functions accept a `centered` parameter:

- `"world"` — positions as stored in the BVH file
- `"skeleton"` — root forced to `(0, 0, 0)` every frame
- `"first"` — first frame's root at origin, subsequent frames relative
