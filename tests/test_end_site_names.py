"""Tests for the names pybvh generates for end sites.

A BVH file gives an end site no name, so pybvh builds one from its
parent: ``Hand_EndSite``, then ``Hand_EndSite2``, ``Hand_EndSite3`` for
the next end sites of the same joint, counted by their position under
it. One rule names them, whether the reader, ``nodes_from_table`` or
``extract_joints`` builds the node. A name given in a node table is kept
as it is.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pybvh import df_to_bvh, nodes_from_table, read_bvh_file, write_bvh_file

BVH_DATA = Path(__file__).parent.parent / "bvh_data"
FIXTURES = Path(__file__).parent / "fixtures"
BUNDLED_FILES = sorted(BVH_DATA.glob("*.bvh")) + [
    FIXTURES / "rotation_first_root.bvh",
    FIXTURES / "full_precision_frame_time.bvh",
]

# A joint with three end sites between two of its child joints: the
# number counts the end sites of the joint, not all of its children.
_THREE_TIPS = """\
HIERARCHY
ROOT Hips
{
  OFFSET 0 0 0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
  JOINT Hand
  {
    OFFSET 0 10 0
    CHANNELS 3 Zrotation Xrotation Yrotation
    End Site
    {
      OFFSET 1 0 0
    }
    JOINT Finger
    {
      OFFSET 0 1 0
      CHANNELS 3 Zrotation Xrotation Yrotation
      End Site
      {
        OFFSET 0 1 0
      }
    }
    End Site
    {
      OFFSET 0 0 1
    }
    End Site
    {
      OFFSET -1 0 0
    }
  }
}
MOTION
Frames: 2
Frame Time: 0.033333
0 0 0 0 0 0 0 0 0 0 0 0
1 2 3 10 20 30 5 6 7 -5 -6 -7
"""


@pytest.fixture
def three_tips(tmp_path):
    path = tmp_path / "three_tips.bvh"
    path.write_text(_THREE_TIPS)
    return read_bvh_file(path)


class TestReader:
    def test_three_end_sites_of_one_joint_are_numbered(self, three_tips):
        assert [n.name for n in three_tips.nodes] == [
            "Hips",
            "Hand",
            "Hand_EndSite",
            "Finger",
            "Finger_EndSite",
            "Hand_EndSite2",
            "Hand_EndSite3",
        ]

    def test_each_numbered_end_site_is_selected_by_its_name(self, three_tips):
        assert three_tips.index("Hand_EndSite", space="node") == 2
        assert three_tips.index("Hand_EndSite2", space="node") == 5
        assert three_tips.index("Hand_EndSite3", space="node") == 6

    @pytest.mark.parametrize("path", BUNDLED_FILES, ids=lambda p: p.name)
    def test_every_end_site_of_a_bundled_clip_is_named_after_its_parent(self, path):
        bvh = read_bvh_file(path)
        end_sites = [n for n in bvh.nodes if n.is_end_site()]
        assert end_sites
        for node in end_sites:
            assert node.parent is not None
            assert node.name == node.parent.name + "_EndSite"

    def test_a_clip_written_and_read_again_has_the_same_names(self, three_tips, tmp_path):
        path = tmp_path / "again.bvh"
        write_bvh_file(three_tips, path)
        again = read_bvh_file(path)
        assert [n.name for n in again.nodes] == [n.name for n in three_tips.nodes]


class TestNodeTable:
    def test_entries_without_names_are_numbered_like_the_reader(self, three_tips):
        table = three_tips.to_node_table()
        for entry in table:
            if "rot_channels" not in entry:
                del entry["name"]
        nodes = nodes_from_table(table)
        assert [n.name for n in nodes] == [n.name for n in three_tips.nodes]

    def test_an_explicit_end_site_name_is_kept(self):
        nodes = nodes_from_table(
            [
                {"name": "Hips", "parent": None, "offset": [0, 0, 0], "rot_channels": "ZYX"},
                {"name": "Hand", "parent": 0, "offset": [0, 1, 0], "rot_channels": "ZYX"},
                {"name": "EndSiteHand", "parent": 1, "offset": [1, 0, 0]},
                {"parent": 1, "offset": [0, 0, 2]},
                {"name": "Tip", "parent": 1, "offset": [0, 0, 3]},
            ]
        )
        assert [n.name for n in nodes] == [
            "Hips",
            "Hand",
            "EndSiteHand",
            "Hand_EndSite2",
            "Tip",
        ]


class TestExtractJoints:
    def test_the_end_site_it_adds_is_named_after_its_joint(self, three_tips):
        reduced = three_tips.extract_joints(["Hips"])
        assert [n.name for n in reduced.nodes] == ["Hips", "Hips_EndSite"]

    def test_the_end_sites_it_keeps_keep_their_names(self, three_tips):
        reduced = three_tips.extract_joints(["Hips", "Hand"])
        assert [n.name for n in reduced.nodes] == [
            "Hips",
            "Hand",
            "Hand_EndSite",
            "Hand_EndSite2",
            "Hand_EndSite3",
        ]


class TestDataFrame:
    def test_the_labels_follow_the_names(self, three_tips):
        columns = three_tips.to_df_dict(mode="coordinates")
        end_site_labels = [label for label in columns if "EndSite" in label]
        assert end_site_labels == [
            "Hand_EndSite_X",
            "Hand_EndSite_Y",
            "Hand_EndSite_Z",
            "Finger_EndSite_X",
            "Finger_EndSite_Y",
            "Finger_EndSite_Z",
            "Hand_EndSite2_X",
            "Hand_EndSite2_Y",
            "Hand_EndSite2_Z",
            "Hand_EndSite3_X",
            "Hand_EndSite3_Y",
            "Hand_EndSite3_Z",
        ]

    def test_a_frame_exported_and_read_back_gives_the_same_clip(self, three_tips):
        df = pd.DataFrame(three_tips.to_df_dict(mode="euler"))
        back = df_to_bvh(three_tips.to_node_table(), df)
        # The names are compared exactly; the angles come back with float
        # noise, which this test is not about.
        assert back.matches_hierarchy(three_tips, atol=0)
        assert [n.name for n in back.nodes] == [n.name for n in three_tips.nodes]
        np.testing.assert_allclose(back.joint_angles, three_tips.joint_angles, atol=1e-12)
        np.testing.assert_array_equal(back.root_pos, three_tips.root_pos)
