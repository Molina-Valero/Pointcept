"""
Forest point cloud dataset for Pointcept.
"""

from pathlib import Path
from collections.abc import Sequence
import numpy as np
from .builder import DATASETS
from .defaults import DefaultDataset


@DATASETS.register_module()
class SegmentedForestsDataset(DefaultDataset):
    """
    Forest segmentation dataset.

    Flat layout (all plots directly under data_root):

    data_root/
        plot_01/
            coord.npy     (N, 3)  float32
            normal.npy    (N, 3)  float32
            segment.npy   (N,)    int16
        plot_02/ ...
        ...

    There are no train/val/test subdirectories on disk. Instead, each
    logical split maps to an explicit list of plot names below. Edit
    SPLIT_PLOTS to change which plots are used for train / val / test.
    """

    CLASSES = ("shrub", "ground", "crown", "stem", "dead_downwood")

    # --- Which plots belong to each split -------------------------------
    # Edit these lists to control the split. Names must match the folder
    # names under data_root exactly. Keep the three sets disjoint to avoid
    # train/test leakage.
    SPLIT_PLOTS = {
        "train": [
            "plot_01", "plot_02", "plot_03", "plot_04", "plot_05",
            "plot_06", "plot_07", "plot_08", "plot_09", "plot_10",
            "plot_11",
        ],
        "val": ["plot_12", "plot_13"],
        "test": ["plot_14", "plot_15"],
    }
    # --------------------------------------------------------------------

    def get_data_list(self):
        # split may be a single string ("train") or a sequence
        # (("train", "val")); normalize to a list of split names.
        if isinstance(self.split, str):
            splits = [self.split]
        elif isinstance(self.split, Sequence):
            splits = list(self.split)
        else:
            raise TypeError(
                f"Unsupported split type {type(self.split)}: {self.split}"
            )

        data_root = Path(self.data_root)
        data_list = []
        for split in splits:
            if split not in self.SPLIT_PLOTS:
                raise KeyError(
                    f"Unknown split '{split}'. "
                    f"Known splits: {list(self.SPLIT_PLOTS)}"
                )
            for plot_name in self.SPLIT_PLOTS[split]:
                plot_dir = data_root / plot_name
                if not plot_dir.is_dir():
                    raise FileNotFoundError(
                        f"Plot directory not found: {plot_dir}"
                    )
                data_list.append(str(plot_dir))

        return sorted(data_list)

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
