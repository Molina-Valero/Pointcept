"""
Forest point cloud dataset for Pointcept.
"""

from pathlib import Path
import numpy as np
from .builder import DATASETS
from .defaults import DefaultDataset


@DATASETS.register_module()
class SegmentedForestsDataset(DefaultDataset):
    """
    Forest segmentation dataset.

    data_root/
        train/
            plot_01/
                coord.npy     (N, 3)  float32
                normal.npy    (N, 3)  float32
                segment.npy   (N,)    int16
        val/  ...
        test/ ...
    """

    CLASSES = ("shrub", "ground", "crown", "stem", "dead_downwood")

    def get_data_list(self):
        split_dir = Path(self.data_root) / self.split
        scenes = sorted(p for p in split_dir.iterdir() if p.is_dir())
        return [str(s) for s in scenes]

    def get_data(self, idx):
        scene_dir = Path(self.data_list[idx % len(self.data_list)])
        coord   = np.load(scene_dir / "coord.npy")
        normal  = np.load(scene_dir / "normal.npy")
        segment = np.load(scene_dir / "segment.npy").astype(np.int32)
        return dict(
            coord   = coord.astype(np.float32),
            normal  = normal.astype(np.float32),
            segment = segment,
            name    = scene_dir.name,
        )

    def get_data_name(self, idx):
        return Path(self.data_list[idx % len(self.data_list)]).name
