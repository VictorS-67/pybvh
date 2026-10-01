from __future__ import annotations

__version__ = "0.9.0"

# "X as X" declares a re-export (the PEP 484 convention), so linters do not
# report these imports as unused.
from .bvh import Bvh as Bvh
from .node_tree import nodes_from_table as nodes_from_table, nodes_to_table as nodes_to_table
from .tools import Axis as Axis, parse_axis as parse_axis
from .io import read_bvh_file as read_bvh_file, write_bvh_file as write_bvh_file
from .dataframe import df_to_bvh as df_to_bvh
from .spatial_coord import FkTopology as FkTopology, frames_to_node_positions as frames_to_node_positions

from .batch import (
    read_bvh_directory as read_bvh_directory, batch_to_numpy as batch_to_numpy,
    harmonize as harmonize, HarmonizeReport as HarmonizeReport,
)
from .analysis import relative_scale_factor as relative_scale_factor

from . import io as io
from . import batch as batch
from . import bvhplot as bvhplot
from . import rotations as rotations
from . import transforms as transforms
from . import geometry as geometry
from . import analysis as analysis
from . import signal as signal
from . import features as features
