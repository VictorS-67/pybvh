# Installation

## From PyPI

```bash
pip install pybvh
```

## Optional extras

```bash
pip install "pybvh[pandas]"        # DataFrame import/export
pip install "pybvh[opencv]"        # Fast video rendering (~1000 fps)
pip install "pybvh[interactive]"   # k3d for Jupyter notebook playback
pip install "pybvh[viewer]"        # vedo for desktop interactive viewer
pip install "pybvh[all-viz]"       # All visualization backends
```

Pandas is only used for DataFrame import/export — it is not required for core functionality.

## Requirements

- Python >= 3.9
- NumPy >= 1.21
- Matplotlib >= 3.7

Optional visualization backends have their own requirements:

- OpenCV: `opencv-python >= 4.5`
- k3d: `k3d >= 2.14`
- vedo: `vedo >= 2024.5`
