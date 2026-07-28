#!/usr/bin/env python3
"""
Local, memory-safe PTv3 inference on large forest point clouds (Pointcept).

Large plots (millions of points) can exhaust GPU VRAM or system RAM when pushed
through the model whole. This script tiles each cloud in the XY plane, runs the
official Pointcept tester on one tile at a time (bounded memory), then stitches
the per-tile predictions back into the full cloud. Each point is predicted
exactly once — by the tile whose CORE cell contains it — while an overlap margin
gives the model neighbouring context at tile borders (standard sliding-window
inference).

Preprocessing faithfully mirrors your training pipeline
(preprocess_SegmentedForests.py): per-plot XY centering with Z absolute, and
Open3D normals (same radius / max_nn / upward orientation). Normals are computed
once on the FULL centered cloud, then carried into each tile, so tile borders get
the same normals they would have had in training.

Stages:
  1. PREPROCESS + TILE  raw clouds -> data_root/<split>/<plot>__tIII_JJJ/
                        {coord.npy, normal.npy, segment.npy, tile_meta.npz}
  2. INFER              tools/test.py (SemSegTester) over all tile "scenes",
                        via a generated inference config. Predictions land in
                        save_path/result/<tile>_pred.npy
  3. MERGE + EXPORT     stitch tiles per plot -> <plot>_pred.ply (colored by
                        class + label field) and <plot>_pred.txt (x y z label),
                        in the ORIGINAL coordinate frame.

Classes (from your training script):
    0 shrub | 1 ground | 2 crown | 3 stem | 4 dead_downwood

------------------------------------------------------------------------------
USAGE
------------------------------------------------------------------------------
  cd /home/juan/Pointcept
  python run_inference.py --raw_dir /home/juan/data/pruebas

  # if CUDA still OOMs, lower the VOXEL budget (that is what fills VRAM):
  python run_inference.py --raw_dir /home/juan/data/pruebas \
      --max_voxels_per_tile 400000 --no_tta

  # if the machine runs out of *system* RAM instead:
  python run_inference.py --raw_dir /home/juan/data/pruebas \
      --max_points_per_tile 1500000 --normal_block_points 1500000

MEMORY NOTES
  * Tiles are sized by an actual budget, not by average density: any tile
    whose context set exceeds --max_points_per_tile raw points or
    --max_voxels_per_tile occupied voxels is halved recursively. A uniform
    grid derived from the mean density does not work on TLS forest plots,
    where the neighbourhood of each scanner position is orders of magnitude
    denser than the plot average.
  * --presample_grid thins the cloud to one point per 1 cm cell before
    inference and expands the predictions back to every raw point at the end.
    The network voxelizes at 2 cm anyway, so this is nearly free accuracy-wise
    and typically removes most of the points of a multi-scan cloud.
  * Predictions are cached in <save_path>/result. Re-running skips tiles that
    are already done; predictions that no longer match their tile are deleted
    automatically (a leftover one is what triggers the
    "assert output.shape == target.shape" crash in Pointcept's tester).

Run `python run_inference.py --help` for all options.
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

import numpy as np

# =============================================================================
# CONFIG  -- defaults tuned to your setup; override any of these on the CLI.
# =============================================================================
DEFAULTS = dict(
    project_dir="/home/juan/Pointcept",
    exp_dir="/home/juan/Pointcept/exp/SegmentedForests/semseg-pt-v3m1-0-base-paper_weight",
    data_root="/home/juan/Pointcept/data/SegmentedForests",
    split="inference_input",
    save_path="/home/juan/Pointcept/data/SegmentedForests/inference",
    raw_dir="/home/juan/Pointcept/data/SegmentedForests/inference_raw",
    num_gpus=1,
    normal_radius=0.3,       # must match training preprocessing
    normal_max_nn=30,        # must match training preprocessing
    max_points_per_tile=3_000_000,  # HARD cap on raw points per tile (RAM)
    max_voxels_per_tile=800_000,    # HARD cap on 2cm voxels per tile (VRAM)
    voxel_size=0.02,         # must match GridSample grid_size in the config
    presample_grid=0.01,     # thin raw cloud to 1 pt / cell before inference
    normal_block_points=3_000_000,  # block size for blockwise normal estimation
    tile_size=0.0,           # 0 = auto from max_points_per_tile; else metres
    tile_overlap=3.0,        # context margin around each tile core, metres
)

CLASS_NAMES = ["shrub", "ground", "crown", "stem", "dead_downwood"]

BASE_PALETTE = np.array(
    [
        [89, 161, 79],    # 0 shrub  - green
        [156, 117, 95],   # 1 ground - brown
        [118, 183, 178],  # 2 crown  - teal
        [225, 87, 89],    # 3 stem   - red
        [237, 201, 72],   # 4 dead   - yellow
        [78, 121, 167], [242, 142, 43], [176, 122, 161],
        [255, 157, 167], [186, 176, 172],
    ],
    dtype=np.uint8,
)

TILE_SEP = "__t"   # scene name = "<plot>__tIII_JJJ"


# =============================================================================
# raw readers
# =============================================================================
def _read_text_fast(path, delimiter):
    """pandas C parser: ~10x faster and far less RAM than np.loadtxt.
    Returns None if pandas is unavailable or the file does not parse."""
    try:
        import pandas as pd
    except ImportError:
        return None
    try:
        df = pd.read_csv(
            path,
            sep=delimiter if delimiter else r"\s+",
            engine="c" if delimiter else "python",
            header=None,
            comment="#",
            dtype=np.float64,
            on_bad_lines="skip",
        )
    except Exception:
        try:  # header line present?
            df = pd.read_csv(path, sep=delimiter if delimiter else r"\s+",
                             engine="python", dtype=np.float64,
                             on_bad_lines="skip")
        except Exception:
            return None
    arr = df.to_numpy(dtype=np.float64, copy=False)
    del df
    if arr.ndim == 1:
        arr = arr[None]
    ok = ~np.isnan(arr[:, :3]).any(axis=1)
    return arr[ok] if not ok.all() else arr


def read_raw(path, label_col=None):
    ext = os.path.splitext(path)[1].lower()

    if ext in (".txt", ".xyz", ".csv", ".pts"):
        delimiter = "," if ext == ".csv" else None
        arr = _read_text_fast(path, delimiter)
        if arr is None:
            # slow fallback (np.loadtxt needs ~10x the file size in RAM)
            try:
                arr = np.loadtxt(path, delimiter=delimiter, dtype=np.float64)
            except ValueError:
                arr = np.genfromtxt(path, delimiter=delimiter, invalid_raise=False)
                arr = arr[~np.isnan(arr).any(axis=1)]
        if arr.ndim == 1:
            arr = arr[None]
        xyz = arr[:, :3].astype(np.float64)
        labels = None
        if label_col is not None and arr.shape[1] > label_col:
            labels = arr[:, label_col].astype(np.int64)
        return xyz, labels

    if ext == ".ply":
        return _read_ply_xyz(path).astype(np.float64), None

    if ext in (".las", ".laz"):
        try:
            import laspy
        except ImportError:
            sys.exit("ERROR: reading .las/.laz needs laspy.  pip install laspy[lazrs]")
        las = laspy.read(path)
        return np.vstack([las.x, las.y, las.z]).T.astype(np.float64), None

    raise ValueError(f"Unsupported file type: {ext} ({path})")


_PLY_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


def _read_ply_xyz(path):
    try:
        from plyfile import PlyData

        v = PlyData.read(path)["vertex"].data
        return np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float32)
    except ImportError:
        pass
    with open(path, "rb") as f:
        assert f.readline().strip() == b"ply", "not a PLY file"
        fmt, n_vert, props, in_vertex = None, None, [], False
        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("format"):
                fmt = line.split()[1]
            elif line.startswith("element"):
                p = line.split()
                in_vertex = p[1] == "vertex"
                if in_vertex:
                    n_vert = int(p[2])
            elif line.startswith("property") and in_vertex:
                _, t, name = line.split()[:3]
                props.append((name, _PLY_TYPES[t]))
            elif line == "end_header":
                break
        names = [p[0] for p in props]
        if fmt == "ascii":
            data = np.loadtxt(f, max_rows=n_vert)
            if data.ndim == 1:
                data = data[None]
            xi, yi, zi = names.index("x"), names.index("y"), names.index("z")
            return data[:, [xi, yi, zi]].astype(np.float32)
        if fmt == "binary_little_endian":
            dt = np.dtype([(n, t) for n, t in props])
            data = np.fromfile(f, dtype=dt, count=n_vert)
            return np.vstack([data["x"], data["y"], data["z"]]).T.astype(np.float32)
        sys.exit(f"PLY format '{fmt}' unsupported; install plyfile.")


# =============================================================================
# normals + tiling
# =============================================================================
def estimate_normals(xyz, radius, max_nn, block_points=0):
    """Open3D hybrid normals, oriented +Z -- identical to the training
    preprocessing.  With block_points > 0 the cloud is processed in spatial
    blocks padded by `radius`, which gives bit-identical normals at a small
    fraction of the peak RAM (a single KD-tree over 80M points is what kills
    most machines here)."""
    import open3d as o3d

    def _run(sub):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(sub.astype(np.float64))
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius, max_nn=max_nn)
        )
        pcd.orient_normals_to_align_with_direction(orientation_reference=[0, 0, 1])
        return np.asarray(pcd.normals, dtype=np.float32)

    n = xyz.shape[0]
    if not block_points or n <= block_points:
        return _run(xyz)

    # blocks along X, padded by `radius` so every kept point sees its full
    # neighbourhood -> same result as computing on the whole cloud at once
    out = np.zeros((n, 3), dtype=np.float32)
    order = np.argsort(xyz[:, 0], kind="stable")
    xs = xyz[order, 0]
    nblocks = int(np.ceil(n / block_points))
    print(f"    normals: {nblocks} block(s) of ~{block_points:,} pts "
          f"(padded by {radius} m)")
    for b in range(nblocks):
        lo, hi = b * block_points, min((b + 1) * block_points, n)
        pad_lo = int(np.searchsorted(xs, xs[lo] - radius, side="left"))
        pad_hi = int(np.searchsorted(xs, xs[hi - 1] + radius, side="right"))
        sel = order[pad_lo:pad_hi]
        nrm = _run(xyz[sel])
        keep = slice(lo - pad_lo, hi - pad_lo)
        out[order[lo:hi]] = nrm[keep]
        del nrm
    return out


def voxel_keys(xyz, voxel):
    """int64 hash of the voxel each point falls in (for counting only)."""
    ijk = np.floor(xyz / voxel).astype(np.int64)
    ijk -= ijk.min(axis=0)
    d = ijk.max(axis=0) + 1
    return (ijk[:, 0] * d[1] + ijk[:, 1]) * d[2] + ijk[:, 2]


def presample(xyz, grid):
    """Keep one point per `grid`-sized voxel.  Returns (keep_idx, inverse) so
    that keep_idx[inverse] maps every original point to its representative.

    The network never sees finer detail than the 2 cm GridSample in the config,
    so thinning at 1 cm costs essentially nothing and typically removes 80-95%
    of the points of a multi-scan TLS cloud."""
    keys = voxel_keys(xyz, grid)
    uniq, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    del keys, uniq
    return first.astype(np.int64), inverse.astype(np.int32)


class _XYIndex:
    """Uniform XY bucket grid for fast axis-aligned box queries."""

    def __init__(self, xy, cell):
        self.xy = xy
        self.cell = float(cell)
        self.origin = xy.min(axis=0)
        ij = np.floor((xy - self.origin) / self.cell).astype(np.int64)
        self.nx = int(ij[:, 0].max()) + 1
        self.ny = int(ij[:, 1].max()) + 1
        keys = ij[:, 0] * self.ny + ij[:, 1]
        del ij
        self.order = np.argsort(keys, kind="stable")
        ks = keys[self.order]
        del keys
        self.uniq, starts = np.unique(ks, return_index=True)
        self.starts = starts
        self.ends = np.append(starts[1:], ks.size)

    def query(self, x0, x1, y0, y1):
        """Indices of points with x0 <= x < x1 and y0 <= y < y1."""
        i0 = max(int(np.floor((x0 - self.origin[0]) / self.cell)), 0)
        i1 = min(int(np.floor((x1 - self.origin[0]) / self.cell)), self.nx - 1)
        j0 = max(int(np.floor((y0 - self.origin[1]) / self.cell)), 0)
        j1 = min(int(np.floor((y1 - self.origin[1]) / self.cell)), self.ny - 1)
        if i1 < i0 or j1 < j0:
            return np.empty(0, dtype=np.int64)
        ii, jj = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1),
                             indexing="ij")
        want = (ii.ravel() * self.ny + jj.ravel())
        pos = np.searchsorted(self.uniq, want)
        ok = pos < self.uniq.size
        pos, want = pos[ok], want[ok]
        pos = pos[self.uniq[pos] == want]
        if pos.size == 0:
            return np.empty(0, dtype=np.int64)
        cand = np.concatenate([self.order[self.starts[p]:self.ends[p]]
                               for p in pos])
        p = self.xy[cand]
        m = ((p[:, 0] >= x0) & (p[:, 0] < x1) &
             (p[:, 1] >= y0) & (p[:, 1] < y1))
        return cand[m]


def start_tile_size(xy, max_points_per_tile, explicit):
    """Initial guess only -- the adaptive splitter fixes it per tile."""
    if explicit and explicit > 0:
        return float(explicit)
    n = xy.shape[0]
    ext = xy.max(0) - xy.min(0)
    if n <= max_points_per_tile:
        return float(max(ext.max(), 1.0) + 1.0)
    area = max(ext[0] * ext[1], 1e-6)
    return float(max(np.sqrt(max_points_per_tile / max(n / area, 1e-9)), 1.0))


def tile_indices(xyz, tile_size, overlap, max_points, max_voxels, voxel_size,
                 min_tile=1.0, min_overlap=0.5):
    """Split the XY plane into tiles whose CONTEXT set respects both budgets.

    Returns [(ctx_idx, is_core_over_ctx), ...].  Cores partition every point
    exactly once; ctx adds the overlap margin used only as model context.

    Unlike a fixed grid derived from the average density, this recursively
    halves any tile that is still too big -- which is what actually happens in
    forest scans, where the area around each scanner position can be 50x denser
    than the plot average (that is why you asked for 250k-point tiles and got a
    10.1M-point one)."""
    xy = np.ascontiguousarray(xyz[:, :2])
    index = _XYIndex(xy, tile_size)
    x0, y0 = xy.min(axis=0)
    x1, y1 = xy.max(axis=0)
    nx = max(1, int(np.ceil((x1 - x0) / tile_size)))
    ny = max(1, int(np.ceil((y1 - y0) / tile_size)))

    stack = []
    for i in range(nx):
        for j in range(ny):
            bx0, bx1 = x0 + i * tile_size, x0 + (i + 1) * tile_size
            by0, by1 = y0 + j * tile_size, y0 + (j + 1) * tile_size
            if i == nx - 1:
                bx1 = np.nextafter(x1, np.inf)
            if j == ny - 1:
                by1 = np.nextafter(y1, np.inf)
            stack.append((bx0, bx1, by0, by1, overlap))

    tiles, forced, shrunk = [], 0, 0
    scratch = np.zeros(xy.shape[0], dtype=bool)
    while stack:
        bx0, bx1, by0, by1, ov = stack.pop()
        core_idx = index.query(bx0, bx1, by0, by1)
        if core_idx.size == 0:
            continue
        ctx_idx = index.query(bx0 - ov, bx1 + ov, by0 - ov, by1 + ov)
        too_many_pts = ctx_idx.size > max_points
        nvox = (np.unique(voxel_keys(xyz[ctx_idx], voxel_size)).size
                if (max_voxels and not too_many_pts) else 0)
        if too_many_pts or (max_voxels and nvox > max_voxels):
            wide = (bx1 - bx0) >= (by1 - by0)
            splittable = ((bx1 - bx0) if wide else (by1 - by0)) > 2 * min_tile
            # If the CONTEXT ring, not the core, is what busts the budget,
            # splitting the core does not help -- shrink the margin instead.
            margin_bound = ctx_idx.size > 2 * core_idx.size
            if ov > min_overlap and (margin_bound or not splittable):
                stack.append((bx0, bx1, by0, by1, max(0.5 * ov, min_overlap)))
                shrunk += 1
                continue
            if splittable:
                if wide:
                    mid = 0.5 * (bx0 + bx1)
                    stack.append((bx0, mid, by0, by1, ov))
                    stack.append((mid, bx1, by0, by1, ov))
                else:
                    mid = 0.5 * (by0 + by1)
                    stack.append((bx0, bx1, by0, mid, ov))
                    stack.append((bx0, bx1, mid, by1, ov))
                continue
            forced += 1  # dense spot smaller than min_tile with minimal margin
        scratch[core_idx] = True
        is_core = scratch[ctx_idx]
        scratch[core_idx] = False
        tiles.append((ctx_idx, is_core))
    if shrunk:
        print(f"    note: reduced the context margin on {shrunk} dense tile(s) "
              f"(down to {min_overlap} m) to stay inside the budget")
    if forced:
        print(f"    WARNING: {forced} tile(s) still exceed the budget at the "
              f"minimum tile size ({min_tile} m) -- lower --presample_grid or "
              f"the budgets if these OOM")
    return tiles


# =============================================================================
# STAGE 1 : preprocess + tile
# =============================================================================
def preprocess(raw_dir, data_root, split, normal_radius, normal_max_nn,
               max_points_per_tile, tile_size_opt, overlap,
               label_col=None, center=True, max_voxels_per_tile=0,
               voxel_size=0.02, presample_grid=0.0, normal_block_points=0):
    split_dir = os.path.join(data_root, split)
    os.makedirs(split_dir, exist_ok=True)

    patterns = ("*.txt", "*.xyz", "*.csv", "*.pts", "*.ply", "*.las", "*.laz")
    raw_files = sorted(f for p in patterns
                       for f in glob.glob(os.path.join(raw_dir, p)))
    if not raw_files:
        sys.exit(f"ERROR: no raw clouds in {raw_dir}")

    # Old tile folders would otherwise linger and be merged with new ones.
    for d in sorted(glob.glob(os.path.join(split_dir, f"*{TILE_SEP}*"))):
        if os.path.isdir(d):
            shutil.rmtree(d)

    print(f"[preprocess] {len(raw_files)} raw cloud(s) in {raw_dir}")
    print(f"[preprocess] budgets: <= {max_points_per_tile:,} pts and "
          f"<= {max_voxels_per_tile:,} voxels ({voxel_size} m) per tile")
    tile_scenes = []
    for path in raw_files:
        plot = os.path.splitext(os.path.basename(path))[0]
        xyz, labels = read_raw(path, label_col=label_col)
        has_gt = labels is not None
        n_raw = xyz.shape[0]

        centroid = xyz.mean(axis=0)
        centroid[2] = 0.0
        coord = (xyz - centroid).astype(np.float32) if center \
            else xyz.astype(np.float32)
        del xyz

        # ---- normals at FULL resolution (same as training), blockwise RAM ----
        normals = estimate_normals(coord, normal_radius, normal_max_nn,
                                   block_points=normal_block_points)
        if labels is None:
            labels = np.zeros((n_raw,), dtype=np.int16)
        else:
            labels = labels.astype(np.int16)

        if has_gt:
            # Stash full-resolution ground truth (original point order, pre-
            # tiling) so export() can score the merged predictions against it
            # later if --eval is requested. One file per plot, not per tile.
            np.save(os.path.join(split_dir, f"{plot}__gt.npy"), labels)

        # ---- thin the cloud (predictions are expanded back at export) ----
        if presample_grid and presample_grid > 0:
            keep, inverse = presample(coord, presample_grid)
            np.save(os.path.join(split_dir, f"{plot}__inv.npy"), inverse)
            del inverse
            coord, normals, labels = coord[keep], normals[keep], labels[keep]
            del keep
            print(f"  {plot:28s} presample {presample_grid} m: "
                  f"{n_raw:,} -> {coord.shape[0]:,} pts "
                  f"({100.0 * coord.shape[0] / n_raw:.1f}%)")
        else:
            inv_path = os.path.join(split_dir, f"{plot}__inv.npy")
            if os.path.isfile(inv_path):
                os.remove(inv_path)

        n_work = coord.shape[0]
        tsize = start_tile_size(coord[:, :2], max_points_per_tile, tile_size_opt)
        tiles = tile_indices(coord, tsize, overlap,
                             max_points=max_points_per_tile,
                             max_voxels=max_voxels_per_tile,
                             voxel_size=voxel_size)
        biggest = max(len(t[0]) for t in tiles)
        print(f"  {plot:28s} pts={n_work:>10,}  start_tile={tsize:6.1f}m  "
              f"tiles={len(tiles):4d}  max_tile_pts={biggest:,}")

        for k, (ctx_idx, is_core) in enumerate(tiles):
            scene = f"{plot}{TILE_SEP}{k:05d}"
            out_dir = os.path.join(split_dir, scene)
            os.makedirs(out_dir, exist_ok=True)
            np.save(os.path.join(out_dir, "coord.npy"), coord[ctx_idx])
            np.save(os.path.join(out_dir, "normal.npy"), normals[ctx_idx])
            np.save(os.path.join(out_dir, "segment.npy"),
                    labels[ctx_idx].astype(np.int16))
            np.savez(os.path.join(out_dir, "tile_meta.npz"),
                     ctx_idx=ctx_idx.astype(np.int64),
                     is_core=is_core,
                     centroid=centroid.astype(np.float64),
                     full_n=np.int64(n_work),
                     n_raw=np.int64(n_raw),
                     raw_path=np.array(path),
                     centered=np.bool_(center),
                     plot=np.array(plot))
            tile_scenes.append(scene)
        del coord, normals, labels

    print(f"[preprocess] wrote {len(tile_scenes)} tile scene(s) to {split_dir}")
    return tile_scenes


# =============================================================================
# STAGE 2 : run the official tester over all tiles
# =============================================================================
def build_inference_config(exp_dir, out_path, tester_data_root, scenes,
                           disable_tta=False):
    src = os.path.join(exp_dir, "config.py")
    if not os.path.isfile(src):
        sys.exit(f"ERROR: config not found: {src}")
    with open(src, "r") as f:
        text = f.read()

    lines = [
        "",
        "",
        "# ==== appended by run_inference.py (inference overrides) ====",
        f"data['test']['data_root'] = {tester_data_root!r}",
        f"data['test']['split'] = {tuple(scenes)!r}",
        # NOTE: Pointcept's default_setup() RECOMPUTES the *_per_gpu values
        # from these top-level keys (num_worker_per_gpu = num_worker //
        # world_size), so overriding num_worker_per_gpu here has no effect.
        # Set the top-level keys instead.
        "batch_size_test = 1",
        "num_worker = 0",   # 0 -> load in the main process, no /dev/shm traffic
        "num_worker_per_gpu = 0",
        "batch_size_test_per_gpu = 1",
        # lower GPU memory: mixed precision + periodic cache clearing
        "enable_amp = True",
        "empty_cache = True",
        "empty_cache_per_epoch = True",
    ]
    if disable_tta:
        lines += [
            "try:",
            "    data['test']['test_cfg']['aug_transform'] = ["
            "[dict(type='RandomRotateTargetAngle', angle=[0], axis='z', "
            "center=[0, 0, 0], p=1)]]",
            "except Exception:",
            "    pass",
        ]
    with open(out_path, "w") as f:
        f.write(text + "\n".join(lines) + "\n")
    return out_path


def run_tester(project_dir, exp_dir, tester_data_root, scenes, save_path,
               num_gpus, disable_tta=False, tiles_per_run=1):
    weight = os.path.join(exp_dir, "model", "model_best.pth")
    if not os.path.isfile(weight):
        sys.exit(f"ERROR: weight not found: {weight}")

    os.makedirs(save_path, exist_ok=True)

    # -------------------------------------------------------------------
    # Pointcept's SemSegTester caches <scene>_pred.npy in save_path/result and
    # REUSES it ("loaded pred and label") instead of re-running the model. If
    # the tiling changed since that file was written, the cached prediction has
    # a different length than the tile, and the tester dies on
    #   assert output.shape == target.shape   (utils/misc.py)
    # So: delete predictions whose length no longer matches, and skip tiles
    # that are already correctly done (that makes the whole run resumable).
    # -------------------------------------------------------------------
    result_dir = os.path.join(save_path, "result")
    os.makedirs(result_dir, exist_ok=True)
    todo, done, stale = [], 0, 0
    for s in scenes:
        pred_path = os.path.join(result_dir, f"{s}_pred.npy")
        coord_path = os.path.join(tester_data_root, s, "coord.npy")
        if os.path.isfile(pred_path):
            try:
                n_pred = np.load(pred_path, mmap_mode="r").shape[0]
                n_pts = np.load(coord_path, mmap_mode="r").shape[0]
            except Exception:
                n_pred, n_pts = -1, -2
            if n_pred == n_pts:
                done += 1
                continue
            os.remove(pred_path)
            stale += 1
        todo.append(s)
    if stale:
        print(f"[infer] removed {stale} stale prediction(s) left over from an "
              f"earlier tiling")
    if done:
        print(f"[infer] {done} tile(s) already predicted -- skipping "
              f"(delete {result_dir} to force a full re-run)")
    scenes = todo
    if not scenes:
        print("[infer] nothing to do, all tiles already predicted.")
        return

    env = os.environ.copy()
    env["PYTHONPATH"] = project_dir + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("WANDB_MODE", "disabled")
    # Reduce allocator fragmentation across runs (the exact issue that killed
    # tile 3 in a single-process run).
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # Split tiles into groups. tiles_per_run=1 => a fresh process per tile, so
    # GPU memory is fully released between tiles and cannot accumulate. Set to 0
    # to run everything in one process (faster, but risks the OOM you just hit).
    if tiles_per_run and tiles_per_run > 0:
        groups = [scenes[i:i + tiles_per_run]
                  for i in range(0, len(scenes), tiles_per_run)]
    else:
        groups = [scenes]

    cfg_path = os.path.join(save_path, "inference_config.py")
    print(f"[infer] {len(scenes)} tile(s) in {len(groups)} run(s)  "
          f"(tiles_per_run={tiles_per_run}, tta={'off' if disable_tta else 'on'}, "
          f"amp=on)")

    for gi, group in enumerate(groups, 1):
        build_inference_config(exp_dir, cfg_path, tester_data_root, group,
                               disable_tta=disable_tta)
        cmd = [
            sys.executable, "tools/test.py",
            "--config-file", cfg_path,
            "--num-gpus", str(num_gpus),
            "--options", f"save_path={save_path}", f"weight={weight}",
        ]
        label = group[0] if len(group) == 1 else f"{group[0]} (+{len(group)-1})"
        print(f"\n[infer] run {gi}/{len(groups)}: {label}")
        rc = subprocess.run(cmd, cwd=project_dir, env=env).returncode
        if rc != 0:
            sys.exit(
                f"ERROR: tools/test.py failed on run {gi}/{len(groups)}.\n"
                "If this is CUDA OOM, lower the VOXEL budget (that is what\n"
                "drives VRAM), e.g.\n"
                "  --max_voxels_per_tile 400000   (or lower)\n"
                "and/or add  --no_tta .\n"
                "If it is a host-RAM MemoryError, lower\n"
                "  --max_points_per_tile / --normal_block_points ."
            )
    print("\n[infer] all runs finished.")


# =============================================================================
# STAGE 3 : merge tiles -> per-plot PLY / TXT (original coordinate frame)
# =============================================================================
def palette(n):
    if n <= len(BASE_PALETTE):
        return BASE_PALETTE[:n]
    rng = np.random.RandomState(42)
    extra = rng.randint(0, 255, size=(n - len(BASE_PALETTE), 3), dtype=np.uint8)
    return np.vstack([BASE_PALETTE, extra])


def write_ply(path, coord, rgb, label):
    n = coord.shape[0]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property int label\nend_header\n"
    )
    dtype = np.dtype(
        [("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
         ("red", "u1"), ("green", "u1"), ("blue", "u1"), ("label", "<i4")]
    )
    d = np.empty(n, dtype=dtype)
    d["x"], d["y"], d["z"] = coord[:, 0], coord[:, 1], coord[:, 2]
    d["red"], d["green"], d["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    d["label"] = label
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        d.tofile(f)


def plot_of(scene):
    return scene.split(TILE_SEP)[0]


# =============================================================================
# metrics (optional; only available for plots where --label_col supplied real
# ground-truth labels at preprocessing time)
# =============================================================================
def confusion_matrix(gt, pred, num_classes):
    idx = gt.astype(np.int64) * num_classes + pred.astype(np.int64)
    cm = np.bincount(idx, minlength=num_classes * num_classes)
    return cm.reshape(num_classes, num_classes)


def metrics_from_confusion(cm, class_names):
    n = cm.shape[0]
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)      # gt count per class
    pred_count = cm.sum(axis=0).astype(np.float64)   # predicted count per class
    union = support + pred_count - tp

    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, tp / union, np.nan)
        precision = np.where(pred_count > 0, tp / pred_count, np.nan)
        recall = np.where(support > 0, tp / support, np.nan)  # = per-class acc.
        f1 = np.where((precision + recall) > 0,
                      2 * precision * recall / (precision + recall), np.nan)

    overall_acc = tp.sum() / max(cm.sum(), 1)
    valid = support > 0
    mean_iou = float(np.nanmean(iou[valid])) if valid.any() else float("nan")
    mean_acc = float(np.nanmean(recall[valid])) if valid.any() else float("nan")

    per_class = []
    for c in range(n):
        name = class_names[c] if c < len(class_names) else str(c)
        per_class.append(dict(
            class_id=c, class_name=name, support=int(support[c]),
            iou=None if np.isnan(iou[c]) else float(iou[c]),
            precision=None if np.isnan(precision[c]) else float(precision[c]),
            recall=None if np.isnan(recall[c]) else float(recall[c]),
            f1=None if np.isnan(f1[c]) else float(f1[c]),
        ))
    return dict(overall_accuracy=float(overall_acc), mean_accuracy=mean_acc,
                mean_iou=mean_iou, per_class=per_class)


def print_metrics(title, m):
    fmt = lambda v: f"{v:6.3f}" if v is not None else "   n/a"
    print(f"\n[eval] {title}")
    print(f"  {'class':16s} {'support':>10s} {'IoU':>7s} {'Prec':>7s} "
          f"{'Recall':>7s} {'F1':>7s}")
    for c in m["per_class"]:
        print(f"  {c['class_name']:16s} {c['support']:>10,d} "
              f"{fmt(c['iou'])} {fmt(c['precision'])} "
              f"{fmt(c['recall'])} {fmt(c['f1'])}")
    print(f"  {'-' * 16} overall_acc={m['overall_accuracy']:.4f}  "
          f"mAcc={m['mean_accuracy']:.4f}  mIoU={m['mean_iou']:.4f}")


def export(data_root, split, save_path, scenes, write_txt=True,
          eval_metrics=False, ignore_index=-1, class_names=None,
          metrics_path=None):
    class_names = class_names or CLASS_NAMES
    result_dir = os.path.join(save_path, "result")
    split_dir = os.path.join(data_root, split)
    if not os.path.isdir(result_dir):
        sys.exit(f"ERROR: no result dir at {result_dir}. Did the tester run?")

    # group tile scenes by original plot
    plots = {}
    for s in scenes:
        plots.setdefault(plot_of(s), []).append(s)

    all_cm, all_metrics = None, {}

    print(f"\n[export] merging tiles for {len(plots)} plot(s)")
    for plot, tile_scenes in sorted(plots.items()):
        full_n = None
        centroid = None
        raw_path = None
        # first pass: determine full_n / centroid
        for s in tile_scenes:
            meta = np.load(os.path.join(split_dir, s, "tile_meta.npz"),
                           allow_pickle=True)
            full_n = int(meta["full_n"])
            centroid = meta["centroid"].astype(np.float64)
            if "raw_path" in meta.files:
                raw_path = str(meta["raw_path"])
            if "centered" in meta.files and not bool(meta["centered"]):
                centroid = np.zeros(3, dtype=np.float64)
            break

        full_pred = np.full(full_n, -1, dtype=np.int64)
        full_coord = np.zeros((full_n, 3), dtype=np.float64)
        filled = 0

        for s in tile_scenes:
            sdir = os.path.join(split_dir, s)
            meta = np.load(os.path.join(sdir, "tile_meta.npz"), allow_pickle=True)
            ctx_idx = meta["ctx_idx"]
            is_core = meta["is_core"]
            coord = np.load(os.path.join(sdir, "coord.npy")).astype(np.float64)

            cands = (glob.glob(os.path.join(result_dir, f"{s}_pred.npy"))
                     + glob.glob(os.path.join(result_dir, f"{s}.npy"))
                     + glob.glob(os.path.join(result_dir, f"*{s}*pred*.npy")))
            if not cands:
                print(f"  WARNING: no prediction for tile '{s}'")
                continue
            pred = np.load(cands[0]).astype(np.int64).reshape(-1)
            if pred.shape[0] != coord.shape[0]:
                # Truncating here would silently mislabel the whole tile; this
                # only happens with a prediction left over from a different
                # tiling, so refuse it and let the tile be re-run.
                print(f"  WARNING: '{s}': prediction has {pred.shape[0]:,} "
                      f"points but the tile has {coord.shape[0]:,} -- stale "
                      f"file, skipping (delete {cands[0]} and re-run)")
                continue
            ctx_idx2, is_core2 = ctx_idx, is_core

            core_orig = ctx_idx2[is_core2]
            full_pred[core_orig] = pred[is_core2]
            full_coord[core_orig] = coord[is_core2] + centroid
            filled += core_orig.shape[0]

        # ---- expand back to full raw resolution if the cloud was presampled --
        inv_path = os.path.join(split_dir, f"{plot}__inv.npy")
        if os.path.isfile(inv_path):
            inv = np.load(inv_path)
            full_pred = full_pred[inv]
            raw_xyz = None
            if raw_path and os.path.isfile(raw_path):
                try:
                    raw_xyz, _ = read_raw(raw_path)
                except Exception as e:  # noqa: BLE001
                    print(f"  WARNING: {plot}: could not re-read {raw_path} "
                          f"({e})")
            if raw_xyz is not None and raw_xyz.shape[0] == inv.shape[0]:
                full_coord = raw_xyz
            else:
                print(f"  WARNING: {plot}: raw cloud unavailable, expanding "
                      f"coordinates from the presampled points instead")
                full_coord = full_coord[inv]
            del inv, raw_xyz
            print(f"  {plot}: expanded {filled:,} predicted point(s) to "
                  f"{full_pred.shape[0]:,} raw point(s)")

        missing = int((full_pred < 0).sum())
        keep = full_pred >= 0

        gt_full = None
        if eval_metrics:
            gt_path = os.path.join(split_dir, f"{plot}__gt.npy")
            if os.path.isfile(gt_path):
                gt_full = np.load(gt_path).astype(np.int64)
            else:
                print(f"  WARNING: {plot}: --eval requested but no ground "
                      f"truth found (use --label_col when preprocessing)")

        if missing:
            print(f"  WARNING: {plot}: {missing:,} points unpredicted "
                  f"(missing tiles?)")
            full_coord, full_pred = full_coord[keep], full_pred[keep]
            if gt_full is not None:
                gt_full = gt_full[keep]

        rgb = palette(int(full_pred.max()) + 1)[full_pred]
        ply = os.path.join(save_path, f"{plot}_pred.ply")
        write_ply(ply, full_coord.astype(np.float32), rgb, full_pred)
        present = [f"{c}:{CLASS_NAMES[c] if c < len(CLASS_NAMES) else '?'}"
                   for c in sorted(np.unique(full_pred))]
        print(f"  {plot}: {full_pred.shape[0]:,} pts from "
              f"{len(tile_scenes)} tile(s)  classes[{', '.join(present)}]")

        if write_txt:
            txt = os.path.join(save_path, f"{plot}_pred.txt")
            np.savetxt(txt,
                       np.column_stack([full_coord, full_pred.astype(np.float64)]),
                       fmt=["%.4f", "%.4f", "%.4f", "%d"])

        if gt_full is not None:
            valid = gt_full != ignore_index
            n_ignored = int((~valid).sum())
            if n_ignored:
                print(f"  [eval] {plot}: ignoring {n_ignored:,} point(s) "
                      f"with label == {ignore_index}")
            gt_valid, pred_valid = gt_full[valid], full_pred[valid]
            if gt_valid.size:
                nc = int(max(gt_valid.max(), pred_valid.max(),
                             len(class_names) - 1) + 1)
                cm = confusion_matrix(gt_valid, pred_valid, nc)
                all_cm = cm if all_cm is None else all_cm + cm
                plot_metrics = metrics_from_confusion(cm, class_names)
                print_metrics(plot, plot_metrics)
                all_metrics[plot] = plot_metrics
            else:
                print(f"  [eval] {plot}: no valid labeled points to score")

    if eval_metrics and all_cm is not None:
        overall = metrics_from_confusion(all_cm, class_names)
        print_metrics("OVERALL (all plots combined)", overall)
        all_metrics["__overall__"] = overall
        out = metrics_path or os.path.join(save_path, "metrics.json")
        with open(out, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\n[eval] metrics written to {out}")
    elif eval_metrics:
        print("\n[eval] --eval was set but no plot had ground-truth labels "
              "(use --label_col at preprocessing time).")

    print("[export] done.")


# =============================================================================
# main
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Memory-safe (tiled) PTv3 inference on forest point clouds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--project_dir", default=DEFAULTS["project_dir"])
    p.add_argument("--exp_dir", default=DEFAULTS["exp_dir"])
    p.add_argument("--data_root", default=DEFAULTS["data_root"])
    p.add_argument("--split", default=DEFAULTS["split"])
    p.add_argument("--save_path", default=DEFAULTS["save_path"])
    p.add_argument("--raw_dir", default=DEFAULTS["raw_dir"])
    p.add_argument("--num_gpus", type=int, default=DEFAULTS["num_gpus"])
    p.add_argument("--normal_radius", type=float, default=DEFAULTS["normal_radius"])
    p.add_argument("--normal_max_nn", type=int, default=DEFAULTS["normal_max_nn"])
    p.add_argument("--max_points_per_tile", type=int,
                   default=DEFAULTS["max_points_per_tile"],
                   help="HARD cap on raw points per tile (system RAM). Tiles "
                        "that exceed it are split recursively.")
    p.add_argument("--max_voxels_per_tile", type=int,
                   default=DEFAULTS["max_voxels_per_tile"],
                   help="HARD cap on occupied voxels per tile. This is what "
                        "actually drives VRAM: the tester voxelizes at "
                        "--voxel_size before the forward pass. Lower it if you "
                        "still get CUDA OOM, raise it if the GPU is idle.")
    p.add_argument("--voxel_size", type=float, default=DEFAULTS["voxel_size"],
                   help="Must match GridSample grid_size in the model config.")
    p.add_argument("--presample_grid", type=float,
                   default=DEFAULTS["presample_grid"],
                   help="Keep one point per cell of this size before "
                        "inference (0 = off). Predictions are expanded back to "
                        "every raw point at export. Must stay below "
                        "--voxel_size to be lossless.")
    p.add_argument("--normal_block_points", type=int,
                   default=DEFAULTS["normal_block_points"],
                   help="Estimate normals in blocks of this many points "
                        "(0 = whole cloud at once, needs a lot of RAM).")
    p.add_argument("--tiles_per_run", type=int, default=1,
                   help="Tiles per tester process. 1 (default) = fresh GPU "
                        "memory per tile (safest). 0 = all in one process.")
    p.add_argument("--tile_size", type=float, default=DEFAULTS["tile_size"],
                   help="Metric tile size (m). 0 = auto from max_points_per_tile.")
    p.add_argument("--tile_overlap", type=float, default=DEFAULTS["tile_overlap"],
                   help="Context margin around each tile (m).")
    p.add_argument("--no_tta", action="store_true",
                   help="Disable test-time augmentation (less memory & faster; "
                        "slightly lower accuracy).")
    p.add_argument("--label_col", type=int, default=None,
                   help="Label column index in text files (for evaluation).")
    p.add_argument("--no_center", action="store_true")
    p.add_argument("--skip_preprocess", action="store_true")
    p.add_argument("--skip_infer", action="store_true")
    p.add_argument("--no_txt", action="store_true")
    p.add_argument("--eval", action="store_true",
                   help="Compute accuracy/precision/recall/F1/IoU metrics by "
                        "comparing merged predictions against ground-truth "
                        "labels. Requires --label_col to have been set when "
                        "the clouds were preprocessed (labelled raw files).")
    p.add_argument("--ignore_index", type=int, default=-1,
                   help="Ground-truth label value to exclude from metrics "
                        "(e.g. unlabeled/unknown points).")
    p.add_argument("--metrics_path", default=None,
                   help="Where to write metrics.json (default: "
                        "<save_path>/metrics.json).")
    return p.parse_args()


def main():
    a = parse_args()

    if a.skip_preprocess:
        split_dir = os.path.join(a.data_root, a.split)
        scenes = sorted(d for d in os.listdir(split_dir)
                        if os.path.isdir(os.path.join(split_dir, d)))
        print(f"[preprocess] skipped; using {len(scenes)} existing tile(s).")
    else:
        if a.presample_grid and a.presample_grid >= a.voxel_size:
            sys.exit(f"ERROR: --presample_grid ({a.presample_grid}) must be "
                     f"smaller than --voxel_size ({a.voxel_size}), otherwise "
                     f"you are throwing away detail the model would have used.")
        scenes = preprocess(
            a.raw_dir, a.data_root, a.split,
            a.normal_radius, a.normal_max_nn,
            a.max_points_per_tile, a.tile_size, a.tile_overlap,
            label_col=a.label_col, center=not a.no_center,
            max_voxels_per_tile=a.max_voxels_per_tile,
            voxel_size=a.voxel_size,
            presample_grid=a.presample_grid,
            normal_block_points=a.normal_block_points,
        )

    if not a.skip_infer:
        tester_data_root = os.path.join(a.data_root, a.split)
        run_tester(a.project_dir, a.exp_dir, tester_data_root, scenes,
                   a.save_path, a.num_gpus, disable_tta=a.no_tta,
                   tiles_per_run=a.tiles_per_run)
    else:
        print("[infer] skipped by request.")

    if a.eval and a.label_col is None and not a.skip_preprocess:
        print("[warn] --eval was set but --label_col was not; no ground "
              "truth will be available to score against unless the tile "
              "split already contains '<plot>__gt.npy' files from a prior "
              "labelled run.")

    export(a.data_root, a.split, a.save_path, scenes, write_txt=not a.no_txt,
          eval_metrics=a.eval, ignore_index=a.ignore_index,
          class_names=CLASS_NAMES, metrics_path=a.metrics_path)
    print(f"\nAll done. Outputs in: {a.save_path}")


if __name__ == "__main__":
    main()
