"""
Variant 0h: 0f (SegDetector) trained on the JOINT corpus of ARKitScenes +
ScanNet + ScanNet++ Locate-3D annotations.

Motivation
----------
Our 0f run confirmed:
- The SegDetector per-point BCE+Dice loss DOES learn on Utonia features
  (Acc@0.25 0 -> 0.03 within 12 epochs on ARKitScenes alone).
- The DETR-style decoder paths (0c/0e/0g) all collapse (class loss to
  trivial minimum, match_iou stuck near 0) because the sparse matching
  needs CLIP-aligned features from the encoder, which our 9-dim raw
  (coord, color, normal) Utonia input doesn't provide.

The biggest remaining gap vs. the original Locate-3D paper is data
scale: Meta trained on the combined corpus (~100k annotations), we had
only ARKitScenes (~991). Adding ScanNet + ScanNet++ annotations lifts
the training budget by ~100x.

ScanNet / ScanNet++ annotation JSONs reference per-point instance IDs
instead of shipping bboxes directly: a new ``ScanNetLocate3DDataset``
(and its ScanNet++ sibling) reads the scene's preprocessed
``instance.npy`` and derives per-entity mask + AABB post-GridSample.
These real per-point masks are STRICTLY BETTER supervision than
ARKit's "point inside GT box" proxy, so ``Locate3DSegDetector`` was
updated to prefer ``input_dict["point_masks"]`` when provided.

REQUIRES: the user to set the three data-root paths below to match
their Pointcept-preprocessed ScanNet and ScanNet++ scene trees, plus
the ScanNet / ScanNet++ Locate-3D annotation JSONs. Leave any of
them as ``None`` to train on a subset only.

How to run
----------
    python tools/train.py \\
        --config-file configs/utonia/localize-utonia-v1m1-0h-combined.py \\
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

# --------- dataset paths (EDIT THESE TO YOUR ENVIRONMENT) ---------
arkit_root = "/group-volume/3Ddataset/arkitscenes-compressed"
scannet_root = "/group-volume/3Ddataset/scannet"
scannetpp_root = "/group-volume/3Ddataset/scannetpp"

# Annotation JSONs. ARKit ones are in this repo; for ScanNet / ScanNet++
# you'll need the corresponding Locate-3D JSONs (published alongside
# train_arkitscenes.json / val_arkitscenes.json). Set to None for any
# dataset you want to skip.
arkit_train_ann = "locate-3d/locate3d_data/train_arkitscenes.json"
arkit_val_ann = "locate-3d/locate3d_data/val_arkitscenes.json"
scannet_train_ann = "locate-3d/locate3d_data/train_scannet.json"
scannet_val_ann = "locate-3d/locate3d_data/val_scannet.json"
scannetpp_train_ann = "locate-3d/locate3d_data/train_scannetpp.json"
scannetpp_val_ann = "locate-3d/locate3d_data/val_scannetpp.json"
# -------------------------------------------------------------------

utonia_pretrained_path = None
weight = utonia_pretrained_path

model = dict(
    type="Locate3DSegDetector",
    backbone_out_channels=54,
    d_model=512,
    freeze_backbone=False,
    freeze_text_encoder=True,
    text_encoder="clip",
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
        freeze_encoder=True,
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

# Per-dataset transform pipelines because ARKit ships ``boxes_xyzxyz``
# pre-transform (set in ARKitScenesLocate3DDataset.get_data) while
# ScanNet / ScanNet++ adapters instead carry ``instance`` through the
# pipeline and build the boxes POST-transform. Each needs its own
# ``Collect.keys`` list.
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
    # Flip/scale here update coord + normal only. The ScanNet adapter
    # derives boxes POST-transform from the sampled instance, so those
    # boxes track whatever the final coord ends up being -- no box-aware
    # variants needed on this path.
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
    """Wrap ``dict(type=..., ...)`` if the annotation file exists, else
    ``None`` so the dataset isn't instantiated."""
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
    "0h-combined found no usable train dataset. Set at least one of "
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
    ),
    dict(type="Locate3DMetricsLogger", log_train_every=1),
    dict(type="CheckpointSaver", save_freq=None),
]
