"""
Preprocessing script for forest point cloud dataset.
Input:  per-plot .txt files with columns [X, Y, Z, label]
Output: per-plot .npy files with keys: coord, normal, segment

Directory structure expected:
    dataset_root/
        raw/
            train/
                plot_01.txt
                plot_02.txt
                ...
            val/
                plot_xx.txt
                ...
            test/
                plot_xx.txt
                ...

Output structure:
    output_root/
        train/
            plot_01/
                coord.npy       # (N, 3) float32  - XYZ
                normal.npy      # (N, 3) float32  - estimated normals
                segment.npy     # (N,)   int16    - class labels 0..4
        val/
            ...
        test/
            ...

Usage:
    python preprocess_SegmentedForests.py \
        --dataset_root /path/to/raw \
        --output_root  /path/to/processed \
        --num_workers  8
"""

import argparse
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import open3d as o3d


# ── label mapping ────────────────────────────────────────────────────────────
# Adjust if your integer codes differ from this
LABEL_NAMES = {
    0: "shrub",
    1: "ground",
    2: "crown",
    3: "stem",
    4: "dead_downwood",
}
IGNORE_LABEL = -1   # label value that will be masked during training


def estimate_normals(xyz: np.ndarray,
                     radius: float = 0.3,
                     max_nn: int = 30) -> np.ndarray:
    """Estimate per-point normals using Open3D KD-tree search."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius, max_nn=max_nn
        )
    )
    # Orient normals consistently upward (helps for ground/stem distinction)
    pcd.orient_normals_to_align_with_direction(orientation_reference=[0, 0, 1])
    return np.asarray(pcd.normals, dtype=np.float32)


def load_txt(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a .txt point cloud file.
    Expected columns: X Y Z label
    Returns xyz (N,3) float32 and labels (N,) int16.
    Extend here if you have extra columns (intensity, RGB, etc.).
    """
    data = np.loadtxt(path, dtype=np.float32)

    if data.ndim == 1:
        data = data[None]   # single-point edge case

    xyz    = data[:, :3]
    labels = data[:,  3].astype(np.int16)
    return xyz, labels


def process_scene(txt_path: Path,
                  output_root: Path,
                  split: str,
                  normal_radius: float,
                  normal_max_nn: int) -> str:
    scene_name = txt_path.stem          # e.g. "plot_01"
    out_dir    = output_root / split / scene_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── skip if already done ─────────────────────────────────────────────────
    if (out_dir / "coord.npy").exists() and \
       (out_dir / "normal.npy").exists() and \
       (out_dir / "segment.npy").exists():
        return f"[skip] {scene_name}"

    # ── load ─────────────────────────────────────────────────────────────────
    xyz, labels = load_txt(txt_path)

    # ── centre XY (keep Z absolute for normal orientation) ──────────────────
    centroid    = xyz.mean(axis=0)
    centroid[2] = 0.0           # do not shift Z
    xyz         = xyz - centroid

    # ── normals ──────────────────────────────────────────────────────────────
    normals = estimate_normals(xyz, radius=normal_radius, max_nn=normal_max_nn)

    # ── save ─────────────────────────────────────────────────────────────────
    np.save(out_dir / "coord.npy",   xyz)
    np.save(out_dir / "normal.npy",  normals)
    np.save(out_dir / "segment.npy", labels)

    return f"[done] {scene_name}  pts={len(xyz):,}  labels={np.unique(labels).tolist()}"


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess forest point clouds for Pointcept / PTv3"
    )
    parser.add_argument("--dataset_root", required=True,
                        help="Root folder containing train/ val/ test/ subdirs of .txt files")
    parser.add_argument("--output_root",  required=True,
                        help="Where to write processed data")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                        help="Which splits to process")
    parser.add_argument("--normal_radius", type=float, default=0.3,
                        help="KD-tree search radius for normal estimation (metres)")
    parser.add_argument("--normal_max_nn", type=int,   default=30,
                        help="Max neighbours for normal estimation")
    parser.add_argument("--num_workers",   type=int,   default=4,
                        help="Parallel workers (set 1 to debug)")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_root  = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        split_dir = dataset_root / split
        if not split_dir.exists():
            print(f"[warn] split dir not found: {split_dir}, skipping")
            continue

        txt_files = sorted(split_dir.glob("*.txt"))
        if not txt_files:
            print(f"[warn] no .txt files found in {split_dir}")
            continue

        print(f"\n── {split}: {len(txt_files)} scenes ──")

        worker = partial(
            process_scene,
            output_root   = output_root,
            split         = split,
            normal_radius = args.normal_radius,
            normal_max_nn = args.normal_max_nn,
        )

        if args.num_workers == 1:
            for f in txt_files:
                print(worker(f))
        else:
            with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
                for msg in pool.map(worker, txt_files):
                    print(msg)

    print("\nDone.")


if __name__ == "__main__":
    main()
