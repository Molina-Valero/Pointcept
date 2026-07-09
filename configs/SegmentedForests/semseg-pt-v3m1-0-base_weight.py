_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 12        # total bs across all GPUs
num_worker = 24
mix_prob   = 0.8
empty_cache = False
enable_amp  = True

# ---------------------------------------------------------------------------
# shared constants (single source of truth for the rest of the file)
# ---------------------------------------------------------------------------
dataset_type = "SegmentedForestsDataset"
data_root    = "data/SegmentedForests"
num_classes  = 5
ignore_index = -1
names = ["shrub", "ground", "crown", "stem", "dead_downwood"]

train_split = (
    "plot_02", "plot_04", "plot_05", "plot_06", "plot_08", "plot_09",
    "plot_10", "plot_11", "plot_12", "plot_13", "plot_14", "plot_15",
)
# NOTE: matches base.py — the val plots carry a "_val" suffix on disk
val_split = ("plot_01_val", "plot_03_val", "plot_07_val")

# ---------------------------------------------------------------------------
# class weights from the TRAIN split.
# Computed once and cached to disk; recomputed only if the cache is missing.
# Falls back to uniform weights when the data isn't present (e.g. inference
# on a machine without the dataset mounted) so the config never crashes.
#   scheme: "sqrt_inv" (gentle, recommended) | "inv" | "enet"
# ---------------------------------------------------------------------------
def _compute_class_weights(scheme="sqrt_inv", cache=True):
    # imports are local on purpose: Pointcept's Config deepcopies every
    # module-level name in this file, and module objects can't be copied
    import os
    import glob
    import numpy as np

    cache_path = os.path.join(data_root, f"class_weights_{scheme}.npy")
    if cache and os.path.isfile(cache_path):
        return np.load(cache_path).astype(np.float32).tolist()

    counts = np.zeros(num_classes, dtype=np.int64)
    for plot in train_split:
        files = []
        # adjust these globs if your on-disk layout differs
        for pat in (
            os.path.join(data_root, plot, "segment.npy"),
            os.path.join(data_root, plot, "**", "segment.npy"),
            os.path.join(data_root, "**", plot, "**", "segment.npy"),
        ):
            files.extend(glob.glob(pat, recursive=True))
        for f in sorted(set(files)):
            seg = np.load(f).reshape(-1).astype(np.int64)
            seg = seg[(seg >= 0) & (seg < num_classes)]  # drop ignore/stray labels
            counts += np.bincount(seg, minlength=num_classes)[:num_classes]

    if counts.sum() == 0:
        # data not found -> don't crash test/inference launches
        return [1.0] * num_classes

    freq = counts / counts.sum()
    if scheme == "inv":
        w = 1.0 / (freq + 1e-8)
    elif scheme == "sqrt_inv":
        w = 1.0 / np.sqrt(freq + 1e-8)
    elif scheme == "enet":
        w = 1.0 / np.log(1.02 + freq)
    else:
        raise ValueError(f"unknown scheme: {scheme}")
    w = (w / w.mean()).astype(np.float32)  # normalise to mean 1

    if cache:
        np.save(cache_path, w)
    return [float(x) for x in w]  # plain Python floats, safe to deepcopy/dump


class_weight = _compute_class_weights(scheme="sqrt_inv")
del _compute_class_weights  # keep the config namespace clean of callables

# model settings
model = dict(
    type="DefaultSegmentorV2",
    num_classes=num_classes,
    backbone_out_channels=64,
    backbone=dict(
        type="PT-v3m1",
        in_channels=6,              # XYZ coord (3) + normal (3)
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        enc_mode=False,
        # PDNorm off — training on a single dataset from scratch
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("Forest",),
    ),
    criteria=[
        dict(
            type="CrossEntropyLoss",
            weight=class_weight,      # per-class weights (order matches `names`)
            loss_weight=1.0,
            ignore_index=ignore_index,
        ),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=ignore_index),
    ],
)

# scheduler settings
epoch      = 1000
eval_epoch = 1000            # matches base.py

optimizer = dict(type="AdamW", lr=0.006, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.006, 0.0006],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.0006)]

# dataset settings
data = dict(
    num_classes=num_classes,
    ignore_index=ignore_index,
    names=names,

    # training
    train=dict(
        type=dataset_type,
        split=train_split,
        data_root=data_root,
        transform=[
            # Centre each scene vertically
            dict(type="CenterShift", apply_z=True),

            # Random point dropout (helps with varying scan densities)
            dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),

            # Rotations: full 360° around Z (LiDAR heading), tiny tilt on X/Y
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1/64, 1/64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1/64, 1/64], axis="y", p=0.5),

            # Scale: simulate distance variation / scan resolution differences
            dict(type="RandomScale", scale=[0.9, 1.1]),

            # Flip around vertical axis
            dict(type="RandomFlip", p=0.5),

            # Small noise on point positions
            dict(type="RandomJitter", sigma=0.005, clip=0.02),

            # Voxelise at 2 cm — adjust if your data is sparser (e.g. 0.05 for TLS)
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),

            # Cap point count — tune to your GPU memory
            dict(type="SphereCrop", point_max=120000, mode="random"),

            # Shift centroid to XY origin (after crop)
            dict(type="CenterShift", apply_z=False),

            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("coord", "normal"),  # 3+3 = 6 input channels
            ),
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),

    # validation
    val=dict(
        type=dataset_type,
        split=val_split,
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "origin_segment", "inverse"),
                feat_keys=("coord", "normal"),
            ),
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),

    # val PRECISELY
    test=dict(
        type=dataset_type,
        split=val_split,
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index"),
                    feat_keys=("coord", "normal"),
                ),
            ],
            # TTA: 4 rotations × 3 scales (similar to ScanNet but suited to
            # outdoor scenes where scale and heading matter)
            aug_transform=[
                [dict(type="RandomRotateTargetAngle", angle=[0],   axis="z", center=[0,0,0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[1/2], axis="z", center=[0,0,0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[1],   axis="z", center=[0,0,0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[3/2], axis="z", center=[0,0,0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[0],   axis="z", center=[0,0,0], p=1),
                 dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomRotateTargetAngle", angle=[1/2], axis="z", center=[0,0,0], p=1),
                 dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomRotateTargetAngle", angle=[1],   axis="z", center=[0,0,0], p=1),
                 dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomRotateTargetAngle", angle=[3/2], axis="z", center=[0,0,0], p=1),
                 dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomRotateTargetAngle", angle=[0],   axis="z", center=[0,0,0], p=1),
                 dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomRotateTargetAngle", angle=[1/2], axis="z", center=[0,0,0], p=1),
                 dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomRotateTargetAngle", angle=[1],   axis="z", center=[0,0,0], p=1),
                 dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomRotateTargetAngle", angle=[3/2], axis="z", center=[0,0,0], p=1),
                 dict(type="RandomScale", scale=[1.05, 1.05])],
            ],
        ),
        ignore_index=ignore_index,
    ),
)
