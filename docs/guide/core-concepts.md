# Core Concepts

## The BVH format

A BVH file has two sections:

- **HIERARCHY** defines a skeleton as a tree of joints. Each joint has an offset (bone vector from parent) and rotation channels (e.g., `Zrotation Yrotation Xrotation`). Leaf nodes called "End Sites" have only an offset.
- **MOTION** holds the per-frame data: the number of frames (`Frames:`), the seconds per frame (`Frame Time:`), then one row of floats per frame. Its columns are the channels, in the depth-first order in which the HIERARCHY declares them. The root's six values (position and rotation) come first. Then come three Euler angles for every joint other than the root.

## The Bvh object

`Bvh` is the central container. It holds:

| Attribute | Shape | Description |
|---|---|---|
| `root_pos` | `(F, 3)` | Root translation per frame |
| `joint_angles` | `(F, J, 3)` | Euler angles (**radians**) per joint per frame |
| `nodes` | tuple | All skeleton nodes (joints + end sites) in depth-first order; frozen, see [Editing a skeleton](skeleton-ops.md#editing-a-skeleton) |
| `joint_names` | list of str | Non-end-site joint names |
| `euler_orders` | list of str | Per-joint Euler orders (e.g. `['ZYX', ...]`) |
| `edges` | list of tuples | Skeleton edge list as `(child_idx, parent_idx)` |
| `frame_time` | float | Seconds per frame |
| `world_up` | str | Gravity axis (e.g. `'+y'`, `'+z'`). Auto-detected, settable. |

## Index spaces

pybvh has two index spaces:

- **`node_index`** includes end sites. It maps to `bvh.nodes`. `node_positions()` uses it.
- The **`joint_angles` index** excludes end sites. It maps to `bvh.joint_names` and to axis 1 of `bvh.joint_angles`. `edges`, `euler_orders`, and augmentation functions use it.

Most functions use the `joint_angles` index space. The docstring of each function specifies which space it expects.

## Centered modes

Several functions accept a `centered` parameter:

- `"world"` — positions as stored in the BVH file
- `"skeleton"` — root forced to `(0, 0, 0)` every frame
- `"first"` — ground-plane centering. pybvh subtracts the first frame's root position in the two horizontal axes only. The `world_up` coordinate stays as it is. So the motion starts above the origin at its original height.

One look makes the difference obvious. The figure shows the same walking clip and the same frame in three coordinate frames. The blue trail is the root's path over the whole clip:

![The three centered modes on a walking clip: world keeps file coordinates, first starts the path at the origin, skeleton pins the root so the trail collapses to a point](../gallery/img/centered-modes.png)

*More one-figure-per-feature visuals: the [Gallery](../gallery/index.md).*

!!! info "See also"
    [World Up & Orientation](world-up.md) — the up-axis and facing conventions behind `centered` · [Bvh Class API](../api/bvh.md) — every attribute and method · [Quick Start](../getting-started/quickstart.md)
