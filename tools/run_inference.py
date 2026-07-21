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

  # if it still runs out of memory, make tiles smaller:
  python run_inference.py --raw_dir /home/juan/data/pruebas \
      --max_points_per_tile 1000000 --no_tta

Run `python run_inference.py --help` for all options.
"""

import argparse
import glob
import os
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
    raw_dir="/home/juan/Pointcept/data/SegmentedForests/new_clouds",
    num_gpus=1,
    normal_radius=0.3,       # must match training preprocessing
    normal_max_nn=30,        # must match training preprocessing
    max_points_per_tile=1_000_000,  # target tile size (auto-derives tile_size)
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
def read_raw(path, label_col=None):
    ext = os.path.splitext(path)[1].lower()

    if ext in (".txt", ".xyz", ".csv", ".pts"):
        delimiter = "," if ext == ".csv" else None
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
def estimate_normals(xyz, radius, max_nn):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    pcd.orient_normals_to_align_with_direction(orientation_reference=[0, 0, 1])
    return np.asarray(pcd.normals, dtype=np.float32)


def choose_tile_size(xyz, max_points_per_tile, explicit):
    """Pick a metric tile size so each tile holds ~max_points_per_tile points."""
    if explicit and explicit > 0:
        return float(explicit)
    n = xyz.shape[0]
    if n <= max_points_per_tile:
        # one tile covers everything
        ext = xyz[:, :2].max(0) - xyz[:, :2].min(0)
        return float(max(ext.max(), 1.0) + 1.0)
    ext = xyz[:, :2].max(0) - xyz[:, :2].min(0)
    area = max(ext[0] * ext[1], 1e-6)
    density = n / area                       # points per m^2
    size = np.sqrt(max_points_per_tile / max(density, 1e-9))
    return float(max(size, 1.0))


def tile_indices(xyz, tile_size, overlap):
    """Return list of (ctx_idx, is_core_over_ctx, (i, j)). Cores partition all
    points exactly once; ctx adds an overlap margin for context."""
    x, y = xyz[:, 0], xyz[:, 1]
    x0, y0 = x.min(), y.min()
    nx = max(1, int(np.ceil((x.max() - x0) / tile_size)))
    ny = max(1, int(np.ceil((y.max() - y0) / tile_size)))
    ix = np.minimum(((x - x0) / tile_size).astype(int), nx - 1)
    iy = np.minimum(((y - y0) / tile_size).astype(int), ny - 1)
    tiles = []
    for i in range(nx):
        for j in range(ny):
            core = (ix == i) & (iy == j)
            if not core.any():
                continue
            cx0, cx1 = x0 + i * tile_size, x0 + (i + 1) * tile_size
            cy0, cy1 = y0 + j * tile_size, y0 + (j + 1) * tile_size
            ctx = ((x >= cx0 - overlap) & (x < cx1 + overlap) &
                   (y >= cy0 - overlap) & (y < cy1 + overlap))
            ctx_idx = np.where(ctx)[0]
            tiles.append((ctx_idx, core[ctx_idx], (i, j)))
    return tiles


# =============================================================================
# STAGE 1 : preprocess + tile
# =============================================================================
def preprocess(raw_dir, data_root, split, normal_radius, normal_max_nn,
               max_points_per_tile, tile_size_opt, overlap,
               label_col=None, center=True):
    split_dir = os.path.join(data_root, split)
    os.makedirs(split_dir, exist_ok=True)

    patterns = ("*.txt", "*.xyz", "*.csv", "*.pts", "*.ply", "*.las", "*.laz")
    raw_files = sorted(f for p in patterns
                       for f in glob.glob(os.path.join(raw_dir, p)))
    if not raw_files:
        sys.exit(f"ERROR: no raw clouds in {raw_dir}")

    print(f"[preprocess] {len(raw_files)} raw cloud(s) in {raw_dir}")
    tile_scenes = []
    for path in raw_files:
        plot = os.path.splitext(os.path.basename(path))[0]
        xyz, labels = read_raw(path, label_col=label_col)

        centroid = xyz.mean(axis=0)
        centroid[2] = 0.0
        coord = (xyz - centroid).astype(np.float32) if center \
            else xyz.astype(np.float32)

        normals = estimate_normals(coord, normal_radius, normal_max_nn)
        if labels is None:
            labels = np.zeros((coord.shape[0],), dtype=np.int16)
        else:
            labels = labels.astype(np.int16)

        tsize = choose_tile_size(coord, max_points_per_tile, tile_size_opt)
        tiles = tile_indices(coord, tsize, overlap)
        biggest = max(len(t[0]) for t in tiles)
        print(f"  {plot:28s} pts={coord.shape[0]:>10,}  "
              f"tile_size={tsize:6.1f}m  tiles={len(tiles):3d}  "
              f"max_tile_pts={biggest:,}")

        for ctx_idx, is_core, (i, j) in tiles:
            scene = f"{plot}{TILE_SEP}{i:03d}_{j:03d}"
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
                     full_n=np.int64(coord.shape[0]),
                     plot=np.array(plot))
            tile_scenes.append(scene)

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
        # correct Pointcept option names
        "batch_size_test_per_gpu = 1",
        "num_worker_per_gpu = 2",
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
                "If this is still CUDA OOM, make tiles smaller, e.g.\n"
                "  --max_points_per_tile 700000   (or lower)\n"
                "and/or add  --no_tta ."
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


def export(data_root, split, save_path, scenes, write_txt=True):
    result_dir = os.path.join(save_path, "result")
    split_dir = os.path.join(data_root, split)
    if not os.path.isdir(result_dir):
        sys.exit(f"ERROR: no result dir at {result_dir}. Did the tester run?")

    # group tile scenes by original plot
    plots = {}
    for s in scenes:
        plots.setdefault(plot_of(s), []).append(s)

    print(f"\n[export] merging tiles for {len(plots)} plot(s)")
    for plot, tile_scenes in sorted(plots.items()):
        full_n = None
        centroid = None
        # first pass: determine full_n / centroid
        for s in tile_scenes:
            meta = np.load(os.path.join(split_dir, s, "tile_meta.npz"),
                           allow_pickle=True)
            full_n = int(meta["full_n"])
            centroid = meta["centroid"].astype(np.float64)
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
                m = min(pred.shape[0], coord.shape[0])
                pred, coord, ctx_idx2, is_core2 = (
                    pred[:m], coord[:m], ctx_idx[:m], is_core[:m])
            else:
                ctx_idx2, is_core2 = ctx_idx, is_core

            core_orig = ctx_idx2[is_core2]
            full_pred[core_orig] = pred[is_core2]
            full_coord[core_orig] = coord[is_core2] + centroid
            filled += core_orig.shape[0]

        missing = int((full_pred < 0).sum())
        if missing:
            print(f"  WARNING: {plot}: {missing:,} points unpredicted "
                  f"(missing tiles?)")
            keep = full_pred >= 0
            full_coord, full_pred = full_coord[keep], full_pred[keep]

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
                   help="Target points per tile; smaller = less memory.")
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
    return p.parse_args()


def main():
    a = parse_args()

    if a.skip_preprocess:
        split_dir = os.path.join(a.data_root, a.split)
        scenes = sorted(d for d in os.listdir(split_dir)
                        if os.path.isdir(os.path.join(split_dir, d)))
        print(f"[preprocess] skipped; using {len(scenes)} existing tile(s).")
    else:
        scenes = preprocess(
            a.raw_dir, a.data_root, a.split,
            a.normal_radius, a.normal_max_nn,
            a.max_points_per_tile, a.tile_size, a.tile_overlap,
            label_col=a.label_col, center=not a.no_center,
        )

    if not a.skip_infer:
        tester_data_root = os.path.join(a.data_root, a.split)
        run_tester(a.project_dir, a.exp_dir, tester_data_root, scenes,
                   a.save_path, a.num_gpus, disable_tta=a.no_tta,
                   tiles_per_run=a.tiles_per_run)
    else:
        print("[infer] skipped by request.")

    export(a.data_root, a.split, a.save_path, scenes, write_txt=not a.no_txt)
    print(f"\nAll done. Outputs in: {a.save_path}")


if __name__ == "__main__":
    main()
