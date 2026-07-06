"""
Forest point cloud dataset for Pointcept.
"""

import os
from collections.abc import Sequence
import numpy as np
from .builder import DATASETS
from .defaults import DefaultDataset


@DATASETS.register_module()
class SegmentedForestsDataset(DefaultDataset):
    """
    Forest segmentation dataset.

    Flat layout: each plot IS a scene (no room level beneath it).

    data_root/
        plot_01/
            coord.npy     (N, 3)  float32
            normal.npy    (N, 3)  float32
            segment.npy   (N,)    int16
        plot_02/ ...
        ...

    The split is a plot name or a list of plot names, set in the config
    (like S3DIS uses Area names). Each split entry maps directly to one
    plot / scene directory under data_root. Example config:

        train = dict(split=["plot_01", ..., "plot_11"], ...)
        val   = dict(split=["plot_12", "plot_13"],      ...)
        test  = dict(split=["plot_14", "plot_15"],      ...)
    """

    CLASSES = ("shrub", "ground", "crown", "stem", "dead_downwood")

    def get_data_list(self):
        # split may be a single plot name ("plot_01") or a list of them.
        if isinstance(self.split, str):
            splits = [self.split]
        elif isinstance(self.split, Sequence):
            splits = list(self.split)
        else:
            raise NotImplementedError(
                f"Unsupported split type {type(self.split)}: {self.split}"
            )

        data_list = []
        for plot_name in splits:
            plot_dir = os.path.join(self.data_root, plot_name)
            if not os.path.isdir(plot_dir):
                raise FileNotFoundError(f"Plot directory not found: {plot_dir}")
            data_list.append(plot_dir)

        return data_list

    def get_data(self, idx):
        scene_dir = self.data_list[idx % len(self.data_list)]
        coord   = np.load(os.path.join(scene_dir, "coord.npy"))
        normal  = np.load(os.path.join(scene_dir, "normal.npy"))
        segment = np.load(os.path.join(scene_dir, "segment.npy")).astype(np.int32)
        return dict(
            coord   = coord.astype(np.float32),
            normal  = normal.astype(np.float32),
            segment = segment,
            name    = os.path.basename(scene_dir),
        )

    def get_data_name(self, idx):
        return os.path.basename(self.data_list[idx % len(self.data_list)])
