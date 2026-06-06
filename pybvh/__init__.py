from __future__ import annotations

from pathlib import Path as _Path

__version__ = "0.7.0"

from .bvh import Bvh
from .io import read_bvh_file, write_bvh_file
from .df_to_bvh import df_to_bvh
from .spatial_coord import frames_to_node_positions

from .batch import (
    read_bvh_directory, batch_to_numpy, harmonize, HarmonizeReport,
    compute_normalization_stats, normalize_array, denormalize_array,
)

from . import bvhplot
from . import rotations
from . import transforms
from . import geometry
from . import analysis
from . import packing


def api_rename_path() -> _Path:
    """Return the on-disk path to the bundled `API_RENAME.md` reference.

    The file documents every renamed / removed symbol from earlier
    pybvh versions, with the new name to use instead. Useful when
    migrating downstream code:

        >>> import pybvh
        >>> print(pybvh.api_rename_path().read_text())
    """
    return _Path(__file__).parent / "API_RENAME.md"
