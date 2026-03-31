# pybvh — Project Charter

## What pybvh is

pybvh is a Python library for reading, writing, and manipulating BVH motion capture files. It is the foundational layer: it understands motion capture data and exposes it as clean, structured NumPy arrays. It does not assume what the consumer will do with that data.

## Core mission

**Parse, transform, and analyze BVH motion data faithfully and efficiently.**

pybvh owns the journey from `.bvh` file to structured NumPy arrays and back. Everything it does is grounded in what a BVH file contains: skeleton hierarchy, joint rotations, root translation, frame timing.

## Design principles

1. **Framework-agnostic.** Output is always NumPy. No PyTorch, no TensorFlow, no JAX. Users convert to their framework of choice.
2. **Lightweight.** Minimal dependencies (NumPy, Matplotlib). No scipy, no h5py, no ML frameworks. The library should install in seconds and never conflict with anything.
3. **Self-contained math.** All rotation conversions, forward kinematics, and interpolation are implemented in pure NumPy. No external math libraries.
4. **Vectorized.** All numerical operations are batch-vectorized with NumPy. No Python loops over frames.
5. **Faithful to the format.** pybvh preserves BVH semantics exactly. Read-write round-trips are lossless (within float precision). Skeleton topology, Euler orders, frame timing — nothing is silently altered.

## What pybvh owns

- **BVH I/O**: Reading and writing `.bvh` files with full hierarchy and motion data preservation
- **The Bvh object**: The central container holding skeleton + motion data, with validated properties
- **Rotation math**: Conversions between all standard representations (Euler, quaternion, 6D, rotation matrix, axis-angle), SLERP interpolation
- **Forward kinematics**: Computing 3D joint positions from angles
- **Skeleton operations**: Retargeting, scaling, joint extraction, Euler order changes
- **Frame operations**: Slicing, concatenation, resampling
- **Spatial transforms**: Rotation, mirroring, translation, noise, speed perturbation, frame dropout — at both Bvh-object and raw array level
- **Motion analysis**: Velocities, accelerations, angular velocities, foot contacts, root trajectory, feature export — these are properties of motion, not ML-specific concepts
- **Batch loading**: Directory-level I/O with optional parallelism
- **Visualization**: Static and animated 3D skeleton plots

## What pybvh does NOT own

- **Tensor layouts** for specific model architectures (CTV, TVC, etc.)
- **Dataset classes** or data loaders for any ML framework
- **Training pipeline concerns**: HDF5 export, augmentation schedulers, collate functions
- **Skeleton-graph construction**: Adjacency matrices, attention masks, body-part partitions
- **Model-specific preprocessing**: Normalization schemes tied to specific papers, label handling

## The boundary

pybvh understands *motion capture data*. It does not understand *what you are doing with it*. A biomechanics researcher, a game developer, and an ML researcher all use the same pybvh — the library never favors one consumer over another.

## Ecosystem position

pybvh is the foundation that other libraries build on. Its sister library **pybvh-ml** (separate repo) provides the ML-specific layer (tensor packing, PyTorch Datasets, augmentation pipelines). pybvh never depends on or knows about pybvh-ml. The dependency flows one way: `pybvh-ml -> pybvh`.

## Development guidelines

- Run tests with: `conda run -n pybvh pytest tests/ -v`
- pybvh-ml tests use a separate env: `conda run -n pybvh_ml pytest tests/ -v`
- README is the PyPI page — must look professional, not like a personal project
- Never add PyTorch/TensorFlow as dependencies — numpy-only output
