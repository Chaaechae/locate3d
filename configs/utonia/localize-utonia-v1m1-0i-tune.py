"""
Variant 0i: stage-1 hyperparameter tune of 0h, encoder STILL frozen.

Why this config exists
----------------------
0h (ARKit + ScanNet, no ScanNet++) hit val_Acc@0.25=0.54 / AccAll@0.25=
0.50 / AccAll@0.5=0.42 at epoch 15. The hyperparameters in 0h were
inherited from 0f, which was tuned for ARKit-only with derived
"point inside GT box" proxy masks. Now that the dominant supervision
comes from real per-point instance masks (ScanNet), several knobs
become wrong-by-default and capping ceiling:

1. ``bce_pos_weight=100`` was set when positive class was ~0.3% of
   points (ARKit inside-box rule for ~0.45 m boxes in 40k-point
   scenes). Real ScanNet instance masks are 5-15% positive on
   average -> the imbalance ratio is more like 6:1 to 20:1, not
   330:1. pos_weight=100 over-weights positives now, inflating BCE
   gradient on already-large positive sets. Drop to 30.
2. ``loss_weight_dice=5`` was the workaround for BCE collapsing to
   trivial-zero under heavy imbalance. With real masks and lower
   pos_weight, BCE no longer collapses, so Dice doesn't need to
   dominate. Drop to 2.
3. ``max_points_train=40000`` was a memory-cap heuristic. With H100s
   and bs_per_gpu=4, we can comfortably afford 60k points per scene
   for richer supervision (more positive points per entity ->
   sharper masks).
4. ``infer_threshold=0.5`` is the symmetric default. We see
   AccAll@0.5/AccAll@0.25 = 0.42/0.50 = 0.84 (good ratio), so the
   issue isn't box-too-loose -- but very slightly tightening to
   0.55 trims oversized masks at object boundaries and should add
   a bit to Acc@0.5 with negligible cost to Acc@0.25.

Everything else (data loaders, augmentation, scheduler, encoder
freeze) is identical to 0h so this is a clean A/B for "did the
hyperparameter retune help on top of better supervision?".

How to run
----------
    python tools/train.py \\
        --config-file configs/utonia/localize-utonia-v1m1-0i-tune.py \\
        -w /group-volume/utonia.pth \\
        --num-gpus 4
"""

_base_ = ["../_base_/default_runtime.py"]

batch_size = 16
num_worker = 32
mix_prob = 0.0
clip_grad = 10.0
empty_cache = False
empty_cache_per_epoch = True
enable_amp = True
amp_dtype = "bfloat16"
find_unused_parameters = True
evaluate = True
enable_wandb = False

train = dict(type="Locate3DTrainer")

# Same data layout as 0h. ScanNet++ is auto-skipped if its JSON / data
# root are absent.
arkit_root = "/group-volume/3Ddataset/arkitscenes-compressed"
scannet_root = "/group-volume/3Ddataset/scannet-compressed"
scannetpp_root = "/group-volume/3Ddataset/scannetpp-compressed"

# --- Dataset inclusion env-var guards ---
# Set the following before launching to opt sub-corpora out for ablations:
#   LOCATE3D_USE_ARKIT=0      -> drop ARKitScenes from train + val
#   LOCATE3D_USE_SCANNETPP=0  -> drop ScanNet++ from train + val
# Default = "1" = include. The empty annotation paths below cause _maybe()
# to skip the dataset cleanly without rebuilding any Compose pipelines.
import os as _os
_USE_ARKIT = _os.environ.get("LOCATE3D_USE_ARKIT", "1") == "1"
_USE_SCANNETPP = _os.environ.get("LOCATE3D_USE_SCANNETPP", "1") == "1"

arkit_train_ann = "locate-3d/locate3d_data/train_arkitscenes.json" if _USE_ARKIT else None
arkit_val_ann = "locate-3d/locate3d_data/val_arkitscenes.json" if _USE_ARKIT else None
scannet_train_ann = "locate-3d/locate3d_data/train_scannet.json"
scannet_val_ann = "locate-3d/locate3d_data/val_scannet.json"
scannetpp_train_ann = "locate-3d/locate3d_data/train_scannetpp.json" if _USE_SCANNETPP else None
scannetpp_val_ann = "locate-3d/locate3d_data/val_scannetpp.json" if _USE_SCANNETPP else None

utonia_pretrained_path = None
weight = utonia_pretrained_path

model = dict(
    type="Locate3DSegDetector",
    backbone_out_channels=54,
    d_model=512,
    freeze_backbone=False,
    freeze_text_encoder=True,
    text_encoder="clip",
    # 0i v2: previous attempt at "tune for real masks" (pos_weight 100->30,
    # dice 5->2, infer_threshold 0.5->0.55, max_points 40k->60k) was
    # MEASURED WORSE -- ARKit+ScanNet val_Acc@0.25 fell from 0.54 (0h
    # epoch 15) to 0.20 (0i previous, epoch 100). Likely culprits:
    # (1) infer_threshold=0.55 systematically shrinks predicted masks
    #     and the AABB derived from them -- IoU collapses, especially
    #     against larger GT boxes.
    # (2) Lower pos_weight + lower dice weight removed the recall-bias
    #     that 0h was implicitly tuned for. In grounding, slightly
    #     OVER-sized predicted boxes still pass IoU > 0.25 against
    #     decent GT coverage; under-sized boxes don't. The original
    #     0h hyperparameters were not arbitrary -- they were
    #     task-aligned recall-favouring choices.
    #
    # Reverting to 0h hyperparameters so 0i now serves a clean role:
    # "0h, but a longer schedule + the env-var dataset toggles, ready
    # to produce the stage-1 checkpoint for 0j to fine-tune from".
    loss_weight_bce=1.0,
    loss_weight_dice=5.0,
    bce_pos_weight=100.0,
    max_points_train=40000,
    max_points_eval=40000,
    infer_threshold=0.5,
    backbone=dict(
        type="PT-v3m3",
        in_channels=9,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(54, 108, 216, 432, 576),
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(54, 108, 216, 432),
        dec_num_head=(3, 6, 12, 24),
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
        traceable=False,
        mask_token=False,
        enc_mode=False,
        freeze_encoder=True,       # encoder STILL frozen in stage 1
        rope_base=10,
        shift_coords=None,
        jitter_coords=1.1,
        rescale_coords=1.2,
    ),
)

epoch = 100
eval_epoch = 100
base_lr = 2e-4
optimizer = dict(type="AdamW", lr=base_lr, weight_decay=0.01)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[base_lr, base_lr * 2],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="backbone.dec", lr=base_lr * 2)]

_common_keys = (
    "coord",
    "grid_coord",
    "caption",
    "positive_map",
    "primary_object_id",
    "scene_id",
    "name",
)

_arkit_aug = [
    dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
    dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
    dict(type="ChromaticJitter", p=0.95, std=0.05),
    dict(type="RandomFlipBoxAware", p=0.5),
    dict(type="RandomScaleBoxAware", scale=(0.9, 1.1)),
    dict(
        type="GridSample",
        grid_size=0.02,
        hash_type="fnv",
        mode="train",
        return_grid_coord=True,
    ),
    dict(type="NormalizeColor"),
    dict(type="ToTensor"),
]

_scannet_aug = [
    dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
    dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
    dict(type="ChromaticJitter", p=0.95, std=0.05),
    dict(type="RandomFlip", p=0.5),
    dict(type="RandomScale", scale=[0.9, 1.1]),
    dict(
        type="GridSample",
        grid_size=0.02,
        hash_type="fnv",
        mode="train",
        return_grid_coord=True,
    ),
    dict(type="NormalizeColor"),
    dict(type="ToTensor"),
]

train_transform_arkit = _arkit_aug + [
    dict(
        type="Collect",
        keys=_common_keys + ("boxes_xyzxyz",),
        feat_keys=("coord", "color", "normal"),
    ),
]
train_transform_scannet = _scannet_aug + [
    dict(
        type="Collect",
        keys=_common_keys + ("instance",),
        feat_keys=("coord", "color", "normal"),
    ),
]

val_transform_arkit = [
    dict(
        type="GridSample",
        grid_size=0.02,
        hash_type="fnv",
        mode="train",
        return_grid_coord=True,
    ),
    dict(type="NormalizeColor"),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=_common_keys + ("boxes_xyzxyz",),
        feat_keys=("coord", "color", "normal"),
    ),
]
val_transform_scannet = [
    dict(
        type="GridSample",
        grid_size=0.02,
        hash_type="fnv",
        mode="train",
        return_grid_coord=True,
    ),
    dict(type="NormalizeColor"),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=_common_keys + ("instance",),
        feat_keys=("coord", "color", "normal"),
    ),
]


def _maybe(dataset_type, ann_file, data_root, transform, loop):
    import os as _os
    if ann_file is None or not _os.path.isfile(ann_file):
        return None
    if data_root is None or not _os.path.isdir(data_root):
        return None
    return dict(
        type=dataset_type,
        annotation_file=ann_file,
        data_root=data_root,
        transform=transform,
        test_mode=False,
        loop=loop,
    )


_train_datasets = [d for d in (
    _maybe("ARKitScenesLocate3DDataset", arkit_train_ann, arkit_root, train_transform_arkit, loop=10),
    _maybe("ScanNetLocate3DDataset", scannet_train_ann, scannet_root, train_transform_scannet, loop=1),
    _maybe("ScanNetPPLocate3DDataset", scannetpp_train_ann, scannetpp_root, train_transform_scannet, loop=1),
) if d is not None]

assert len(_train_datasets) > 0, (
    "0i-tune found no usable train dataset. Set at least one of "
    "arkit_* / scannet_* / scannetpp_* to existing paths."
)

_val_datasets = [d for d in (
    _maybe("ARKitScenesLocate3DDataset", arkit_val_ann, arkit_root, val_transform_arkit, loop=1),
    _maybe("ScanNetLocate3DDataset", scannet_val_ann, scannet_root, val_transform_scannet, loop=1),
    _maybe("ScanNetPPLocate3DDataset", scannetpp_val_ann, scannetpp_root, val_transform_scannet, loop=1),
) if d is not None]


if len(_train_datasets) == 1:
    data = dict(
        train=_train_datasets[0],
        val=_val_datasets[0] if _val_datasets else _train_datasets[0],
    )
else:
    data = dict(
        train=dict(
            type="ConcatDataset",
            datasets=_train_datasets,
            loop=1,
        ),
        val=(
            _val_datasets[0]
            if len(_val_datasets) == 1
            else dict(
                type="ConcatDataset",
                datasets=_val_datasets,
                loop=1,
            )
        ),
    )

hooks = [
    dict(
        type="CheckpointLoader",
        keywords="module.student.backbone",
        replacement="module.backbone",
    ),
    dict(type="Locate3DStartupSanity"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(
        type="Locate3DSegDetectorEvaluator",
        iou_thresholds=(0.25, 0.5),
        eval_every_n_epochs=5,
    ),
    dict(type="Locate3DMetricsLogger", log_train_every=1),
    dict(type="CheckpointSaver", save_freq=None),
]

del _maybe, _train_datasets, _val_datasets
del _os, _USE_ARKIT, _USE_SCANNETPP
del _common_keys, _arkit_aug, _scannet_aug
del train_transform_arkit, train_transform_scannet
del val_transform_arkit, val_transform_scannet
