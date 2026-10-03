"""Read, write and analyze BVH motion capture data as NumPy arrays.

A ``Bvh`` holds one clip: its skeleton, the root translation, shape
``(F, 3)`` in the file's length unit, and the joint rotations, shape
``(F, J, 3)``, as Euler angles in radians (files store degrees; the
conversion happens on reading and writing). ``read_bvh_file`` loads one
file and ``read_bvh_directory`` a folder of them. The package
namespace re-exports the main entry points; everything else lives in its
submodule, for example ``rotations``, ``transforms``, ``analysis``,
``features`` or ``bvhplot``.
"""

from __future__ import annotations

__version__ = "0.9.1"

# "X as X" declares a re-export (the PEP 484 convention), so linters do not
# report these imports as unused.
from . import analysis as analysis
from . import batch as batch
from . import bvhplot as bvhplot
from . import features as features
from . import geometry as geometry
from . import io as io
from . import rotations as rotations
from . import signal as signal
from . import transforms as transforms
from .analysis import relative_scale_factor as relative_scale_factor
from .batch import HarmonizeReport as HarmonizeReport
from .batch import batch_to_numpy as batch_to_numpy
from .batch import harmonize as harmonize
from .batch import read_bvh_directory as read_bvh_directory
from .bvh import Bvh as Bvh
from .dataframe import df_to_bvh as df_to_bvh
from .io import read_bvh_file as read_bvh_file
from .io import write_bvh_file as write_bvh_file
from .node_tree import nodes_from_table as nodes_from_table
from .node_tree import nodes_to_table as nodes_to_table
from .spatial_coord import FkTopology as FkTopology
from .spatial_coord import frames_to_node_positions as frames_to_node_positions
from .tools import Axis as Axis
from .tools import parse_axis as parse_axis
