# PTv3 Inference — `run_inference.py`

Memory-safe (tiled) inference for large forest point clouds using a trained
Pointcept PTv3 semantic segmentation model.

## What it does

Large plots (millions of points) can exceed GPU/RAM if pushed through the
model whole. This script:

1. **Preprocesses + tiles** raw point clouds — centers each plot in XY,
   estimates normals with Open3D (matching the training preprocessing), and
   splits each plot into overlapping tiles (a "core" region predicted once,
   plus a context margin for boundary-aware predictions).
2. **Runs inference** by calling Pointcept's official `tools/test.py` on each
   tile via a generated config, using your trained checkpoint.
3. **Merges + exports** per-tile predictions back into full, per-plot point
   clouds in the original coordinate frame — writing colored `.ply` and
   `.txt` outputs, and optional accuracy metrics if ground truth is supplied.

Classes: `0 shrub | 1 ground | 2 crown | 3 stem | 4 dead_downwood`

## Requirements

- A trained Pointcept experiment folder containing:
  - `config.py` (the training config)
  - `model/model_best.pth` (the checkpoint)
- Python packages: `numpy`, `open3d` (required for normals).
  Optional: `laspy` for `.las`/`.laz` input, `plyfile` for faster `.ply` reads
  (falls back to a built-in parser if not installed).

## Basic usage

```bash
cd /home/juan/Pointcept
python run_inference.py --raw_dir /home/juan/data/pruebas
```

Supported raw input formats: `.txt .xyz .csv .pts .ply .las .laz`

If it runs out of memory, shrink the tiles and disable TTA:

```bash
python run_inference.py --raw_dir /home/juan/data/pruebas \
    --max_points_per_tile 1000000 --no_tta
```

Run `python run_inference.py --help` for the full option list.

## Key options

| Flag | Default | Purpose |
|---|---|---|
| `--exp_dir` | `exp/SegmentedForests/semseg-pt-v3m1-0-base-paper_weight` | Trained experiment folder (`config.py` + `model/model_best.pth`) |
| `--raw_dir` | `data/SegmentedForests/new_clouds` | Folder of raw point clouds to run inference on |
| `--data_root` / `--split` | `data/SegmentedForests` / `inference_input` | Scratch space for generated tiles |
| `--save_path` | `data/SegmentedForests/inference` | Where predictions and merged outputs go |
| `--max_points_per_tile` | `1,000,000` | Target points per tile; lower = less memory |
| `--tile_size` | auto | Metric tile size (m); overrides auto-sizing if set |
| `--tile_overlap` | `3.0` | Context margin around each tile core (m) |
| `--tiles_per_run` | `1` | Tiles per tester subprocess. `1` = fresh GPU memory per tile (safest). `0` = all tiles in one process (faster, riskier) |
| `--no_tta` | off | Disable test-time augmentation (less memory/faster, slightly lower accuracy) |
| `--normal_radius` / `--normal_max_nn` | `0.3` / `30` | Must match training preprocessing |
| `--label_col` | `None` | Label column index in text files, for evaluation |
| `--eval` | off | Compute IoU/Precision/Recall/F1 against ground truth (requires `--label_col`) |
| `--ignore_index` | `-1` | Ground-truth value excluded from metrics |
| `--no_center` | off | Skip XY centering |
| `--no_txt` | off | Skip writing `<plot>_pred.txt` |
| `--skip_preprocess` | off | Reuse existing tiles in `data_root/split/` |
| `--skip_infer` | off | Reuse existing predictions in `save_path/result/` |

## Evaluating against ground truth

If your raw files include a label column:

```bash
python run_inference.py --raw_dir /home/juan/data/labeled_test \
    --label_col 3 --eval
```

Ground truth is stashed once per plot at preprocessing time and compared to
merged predictions after export. Results print to console and are written to
`metrics.json` (or `--metrics_path`).

## Outputs

In `--save_path`:

- `<plot>_pred.ply` — merged, colored point cloud with an integer `label`
  field, in the original coordinate frame
- `<plot>_pred.txt` — `x y z label` (unless `--no_txt`)
- `metrics.json` — per-class + overall IoU/Precision/Recall/F1 (only with
  `--eval`)
- `result/` — raw per-tile predictions from `tools/test.py`
- `inference_config.py` — the generated Pointcept test config (for reference/debugging)

## Troubleshooting

- **CUDA OOM**: lower `--max_points_per_tile` (e.g. `700000` or less), add
  `--no_tta`, keep `--tiles_per_run 1`.
- **Missing tile predictions / warnings on merge**: check `save_path/result/`
  for `<tile>_pred.npy` files; a failed tester run for one tile leaves that
  tile's core points unpredicted (reported and dropped, not silently kept).
- **`--eval` requested but no ground truth found**: re-run preprocessing with
  `--label_col <index>` set so `<plot>__gt.npy` files are generated.
- **Normals differ from training**: `--normal_radius`/`--normal_max_nn` must
  match `preprocess_SegmentedForests.py` exactly, or tile borders and
  predictions can be inconsistent with what the model was trained on.
