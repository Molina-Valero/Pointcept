"""
Add per-point surface normals to preprocessed S3DIS (or similarly
structured) point cloud data.

For every room folder under --data_root that has coord.npy but no
normal.npy, this estimates per-point normals via PCA over the
k-nearest-neighbor local patch -- the same underlying method Open3D's
estimate_normals() uses -- and writes normal.npy alongside the
existing files.

If Open3D is installed, it's used directly (matches Pointcept's
official preprocess_s3dis.py --parse_normals path exactly). Otherwise
this falls back to a pure NumPy/SciPy implementation, no extra
dependencies, no GPU required.

Usage:
    python compute_normals.py --data_root data/s3dis
    python compute_normals.py --data_root data/s3dis --knn 30 --overwrite
"""
import argparse
import glob
import os

import numpy as np
from scipy.spatial import cKDTree

try:
    import open3d as o3d

    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False


def estimate_normals_open3d(coord, knn):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord.astype(np.float64))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    return np.asarray(pcd.normals, dtype=np.float32)


def estimate_normals_numpy(coord, knn):
    tree = cKDTree(coord)
    _, idx = tree.query(coord, k=knn + 1, workers=-1)
    nbr = coord[idx[:, 1:]]  # exclude the point itself
    centered = nbr - nbr.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", centered, centered) / centered.shape[1]
    _, eigvec = np.linalg.eigh(cov.astype(np.float64))  # ascending eigenvalues
    normal = eigvec[:, :, 0].astype(np.float32)  # eigenvector of smallest eigenvalue
    norm = np.linalg.norm(normal, axis=1, keepdims=True)
    return normal / np.clip(norm, 1e-8, None)


def process_room(room_dir, knn, overwrite):
    coord_path = os.path.join(room_dir, "coord.npy")
    normal_path = os.path.join(room_dir, "normal.npy")
    if not os.path.exists(coord_path):
        return
    if os.path.exists(normal_path) and not overwrite:
        return

    coord = np.load(coord_path).astype(np.float32)
    if HAS_OPEN3D:
        normal = estimate_normals_open3d(coord, knn)
    else:
        normal = estimate_normals_numpy(coord, knn)

    np.save(normal_path, normal)
    backend = "open3d" if HAS_OPEN3D else "numpy"
    print(f"[{backend}] {room_dir}: {coord.shape[0]} pts -> normal.npy")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        required=True,
        help="Root of preprocessed S3DIS data (contains Area_1, Area_2, ...)",
    )
    parser.add_argument(
        "--knn", type=int, default=30, help="Neighborhood size for normal estimation"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute normal.npy even if it already exists",
    )
    args = parser.parse_args()

    room_dirs = sorted(
        d
        for d in glob.glob(os.path.join(args.data_root, "*", "*"))
        if os.path.isdir(d)
    )

    if not HAS_OPEN3D:
        print(
            "open3d not found - falling back to a NumPy/SciPy PCA implementation "
            "(slightly slower, same underlying method, no orientation propagation)."
        )

    for room_dir in room_dirs:
        process_room(room_dir, args.knn, args.overwrite)


if __name__ == "__main__":
    main()
