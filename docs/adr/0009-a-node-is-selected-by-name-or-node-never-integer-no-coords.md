# A node is selected by its name or by the node itself, never by an integer, and the clip's methods read the clip's own positions

Through v0.9.0 a caller selected nodes in three incompatible ways. Names (`foot_joints`, the descriptor methods' `joint`, `extract_joints(joint_names)`) could not reach every node: names repeat in real files, the two end sites of one joint share a generated name, and `node_index` keeps the last node of a repeated name, so a repeated name resolved to one of its nodes without an error. Integers (`ground_contacts(joints)`) did not say which index space they counted in. Node objects were accepted only by `change_euler_order`, which then looked the node up by its name. Separately, the public functions and methods of `bvh` and `analysis` that compute from positions took `coords=`, positions the caller had computed, and only five of them checked that the array was node-shaped: the others read a joint-shaped array with node indices, without an error or with a bare `IndexError`. Issue #127 implements the decisions below.

## Decision

**1. A selecting parameter takes a name or a `BvhNode`, never an integer.** `pybvh.NodeSelector` is the alias `str | BvhNode`. A parameter that selects one node is typed `NodeSelector`; one that selects a set of nodes is typed `Sequence[NodeSelector]`. An integer or a `bool` raises `TypeError`, and the message says that node `i` is `bvh.nodes[i]`. A bare `str` or `BvhNode` where a set is expected raises `TypeError` too.

**2. A name matches exactly one node of the index space the function uses.** A node is found by identity among the clip's `nodes`: a node of another clip, or one taken before an in-place edit replaced the clip's skeleton, raises `ValueError`. A name is counted among the nodes of the function's space only. A missing name raises `ValueError`; a repeated name raises `ValueError` and lists the nodes as `bvh.nodes[i]`, so that the caller can pass the one meant. A set that selects one node twice, by the same entry or by a name and its node, raises `ValueError`. A function that works in joint space refuses an end site. So a name that one joint and one end site share resolves in joint space and raises in node space.

**3. The parameter's name says the space.** A parameter that accepts any node is `node` / `nodes` (`node_a`, `node_b` on `segment_axis_angle`); `joint` / `joints` stay only where end sites are refused (`range_of_motion`, `change_euler_order`, `extract_joints`). The feet are `feet`, and `auto_detect_feet` returns nodes, so that detected feet are never resolved by name again.

**4. One resolver, on `Bvh`.** `Bvh.index(node, space)` resolves one node and is the public form; the private `Bvh._indices(nodes, space)` resolves a set and adds the refusals of decision 1 and of a node selected twice. Every selecting parameter of `Bvh`, `analysis` and `features` goes through them. `node_index` and `joint_index` are unchanged: they stay name-keyed dicts that keep the last node of a repeated name.

**5. Integers stay where the array fixes the space.** The array-level functions (`transforms.mirror_angles`' joint-space pairs, `geometry`) take indices into the array they are given, which has one index space. The name maps (`lr_mapping`, `retarget`'s `name_mapping`, the `masses` of `kinetic_energy` given as a mapping) key values by names and are not selections in the sense of this record.

**6. `coords=` is removed from every public function and method of `bvh` and `analysis`.** Each reads the clip's stored forward-kinematics positions, without a copy where the result does not escape. `centered=` stays. Positions the caller holds (smoothed, predicted or edited) take the array path: `bvh.index(node, space="node")`, then `pybvh.geometry` or `pybvh.signal`. bvhplot's `coords=` stays: drawing given positions is its purpose.

`tests/test_node_selection.py` holds one table with one row per selecting parameter, which checks each refusal on each row, and a guard over the public signatures of `Bvh`, `analysis` and `features`: a parameter named `coords`, a selecting parameter not typed with `NodeSelector`, a selecting parameter without its row, and a parameter typed with `int` that the guard does not list as a frame, an axis or a count each fail it. Each row is called with an end site: a parameter named `joint` or `joints` must refuse it, and every other must accept it.

## Why

**An integer fails without an error between the two index spaces.** On `bvh_data/cmu_12_01_walk.bvh`, `joint_index["RightFoot"]` is 9 and node 9 is `RightLeg`. On v0.9.0, `bvh.ground_contacts([bvh.joint_index["RightFoot"]])` tested `RightLeg` and returned 524 frames of contacts, all zero, with no error; `ground_contacts(["RightFoot"])` returns 64 contact frames (both checked on e5ad243). The descriptor methods already refused integers for this reason. A `BvhNode` belongs to no index space, so each function maps it into the space it uses.

**A name alone cannot reach every node, and the last node of a repeated name is a silent choice.** CODING_STANDARDS says that nodes are resolved by identity and position, never by name, and `Bvh.joint_tips` already returned node indices because end-site names may collide with a real joint's name. A repeated name that raises and names the nodes points the caller to the node, which reaches each of them.

**Four reviews agreed on names or nodes and no integers.** On 2026-10-25, four independent reviews, by four different models, judged the question on API practice, not on the work or the breakage:

| Review | Verdict |
| --- | --- |
| 1 | Names or `BvhNode`, never integers, for sets and single nodes; a name must match exactly one node or raise; `bvh.nodes[i]` for an index; auto-detection returns nodes; rename `ground_contacts(joints=)` to `nodes` |
| 2 | The same, except that a repeated name resolves to its last node, documented, and the node object reaches the others; integers stay only in array-level functions, where the array fixes the space |
| 3 | Names or identity-checked `BvhNode`, no integers; an ambiguous name raises; auto-detection returns nodes; a node selected twice is refused; the same rule for single nodes |
| 4 | As review 3; `extract_joints` takes the same selectors |

They split on a repeated name (raise, or the last node), on auto-detection returning nodes, on the renames and on refusing a node selected twice. The maintainer settled each as decisions 2 and 3 state: a repeated name raises, the parameters are renamed by the glossary's line between a joint and a node, and a node selected twice is refused.

**`coords=` read the wrong space without an error, and saved about a millisecond.** On `bvh_test1.bvh`, `bvh.path_length("RightShoulder")` is 68.43, and with `coords=bvh.joint_positions()` it is 79.67, another node's path, with no error (checked on e5ad243). Only five entry points of `analysis` checked the node axis, and none checked the frame axis. `coords=` was added to skip forward kinematics, which the clip stores since v0.8.0. On the walk concatenated to 33,536 frames × 38 nodes, a fresh forward-kinematics pass costs 406 ms and a stored `node_positions()` costs 0.9 ms, a copy of the stored array (measured on 2026-11-02). Passing the clip's own positions back in therefore saved about a millisecond per call. Of the two lab libraries that use pybvh, pybvh-qualities passed the clip's own `node_positions()` back in, only to share one forward-kinematics pass, and pybvh-ml did not pass `coords=`.

## Considered options

- **Names or node-space integers everywhere**: rejected. An integer does not say its space, and the walk above shows the wrong node returned without an error.
- **Names only**: rejected. A name cannot reach one of two nodes that share it.
- **A repeated name resolves to its last node, documented**: rejected. It is the silent choice that `node_index` makes, and a caller who does not read the note gets one node of several.
- **Keep `joints=` on the parameters that accept end sites**: rejected. GLOSSARY defines a joint as a node with rotation channels, end sites excluded, and the API's other names (`joint_index` / `node_index`) keep that line.
- **Keep `coords=` and check its shape**: rejected. The clip stores its positions, so the parameter saved about a millisecond per call, and every function that took it would need the check.
- **A public array-level contact or facing function for positions the caller holds**: not added. No caller has asked for one; `geometry` and `signal` serve the other descriptors.

## Consequences

Each of these is a breaking change in v0.10.0, without a deprecation cycle under the 0.x policy, each with its migration in the CHANGELOG. A caller with an integer `i` passes `bvh.nodes[i]`. A caller with a repeated name passes the node. A caller who passed the clip's own positions as `coords=` drops the argument; one who holds other positions uses `bvh.index` and then `geometry` or `signal`. `node_index` and `joint_index` still keep the last node of a repeated name; #120 takes up the name-keyed dicts and `retarget`'s name binding.
