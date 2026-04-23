"""
Variant 0f: alternative recipe — caption-conditioned per-point SEGMENTATION
that derives a bbox from the predicted mask, instead of a DETR-style
multi-query set-prediction decoder.

Why this config exists
----------------------
Utonia was pretrained as a dense per-point representation learner (DINO-
style masking + 2D-3D DINOv2 alignment). Its downstream strength -- proven
in ``semseg-utonia-v1m1-*`` -- is per-point classification under dense
supervision. The Locate-3D decoder was designed for 3D-JEPA features which
already carry visually-text-aligned semantics; forcing a sparse set-
prediction loss through Utonia features may be fighting the pretraining
objective.

This variant plays to Utonia's strength:

1. Run the FULL-resolution backbone (enc_mode=False). Encoder is frozen;
   the U-Net decoder trains from scratch under dense point-level
   supervision, same as the successful semseg-dec recipe.
2. Per-point features are projected to a CLIP-text space.
3. For each entity in the caption, we mean-pool its positive-token text
   features into a (d_model,) "entity text" vector and dot it with every
   point's projected feature to obtain a per-point score.
4. Training: binary cross-entropy + Dice between sigmoid(score) and the
   "point is inside GT box g" indicator. Every point is supervised.
5. Inference: threshold sigmoid(score) > 0.5; AABB of the surviving
   points is the predicted box. If too few points survive, fall back to
   top-5% by score.

Multi-entity coverage is structural -- one prediction channel per entity,
so query collapse cannot happen. If a caption has 2 GTs, the model emits 2
boxes; if 5, it emits 5.

Limitations
-----------
- Boxes are only as tight as the mask. A small sub-voxel object gets a
  box no smaller than the voxel size. At grid_size=0.02 this is fine
  for Acc@0.25.
- The evaluator ``Locate3DGroundingEvaluator`` currently expects Locate-3D
  outputs (pred_logits per query). For val of 0f we rely on the train-time
  dbg_mask_iou / dbg_mask_covered25 metrics (logged every step); the
  Pointcept evaluator entry in the hook list is kept for now but the
  displayed Acc@0.25 values will be 0 -- use dbg_mask_* instead. A small
  evaluator adapter is a follow-up if needed.

How to run
----------
    python tools/train.py \\
        --config-file configs/utonia/localize-utonia-v1m1-0f-arkitscenes.py \\
        -w /group-volume/utonia.pth \\
        --num-gpus <N>
"""

_base_ = ["../_base_/default_runtime.py"]

batch_size = 4
num_worker = 16
mix_prob = 0.0
clip_grad = 10.0
empty_cache = False
empty_cache_per_epoch = True
enable_amp = True
amp_dtype = "bfloat16"
find_unused_parameters = True
evaluate = True
enable_wandb = False

# Reuse the same Locate-3D trainer: it already does per-sample caption / box
# targets via the locate3d collate, which is what we need here too.
train = dict(type="Locate3DTrainer")

arkit_root = "/group-volume/3Ddataset/arkitscenes-compressed"
train_annotation = "locate-3d/locate3d_data/train_arkitscenes.json"
val_annotation = "locate-3d/locate3d_data/val_arkitscenes.json"

utonia_pretrained_path = None
weight = utonia_pretrained_path

model = dict(
    type="Locate3DSegDetector",
    backbone_out_channels=54,                   # U-Net decoder output
    d_model=512,                                # shared point-text space
    freeze_backbone=False,
    freeze_text_encoder=True,
    text_encoder="clip",
    loss_weight_bce=1.0,
    loss_weight_dice=1.0,
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
        enc_mode=False,                         # full-resolution output
        freeze_encoder=True,                    # freeze pretrained encoder only
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
    max_lr=[base_lr, base_lr * 2],              # U-Net dec at 2x
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="backbone.dec", lr=base_lr * 2)]

train_transform = [
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
    dict(
        type="Collect",
        keys=(
            "coord",
            "grid_coord",
            "caption",
            "boxes_xyzxyz",
            "positive_map",
            "primary_object_id",
            "scene_id",
            "name",
        ),
        feat_keys=("coord", "color", "normal"),
    ),
]

val_transform = [
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
        keys=(
            "coord",
            "grid_coord",
            "caption",
            "boxes_xyzxyz",
            "positive_map",
            "primary_object_id",
            "scene_id",
            "name",
        ),
        feat_keys=("coord", "color", "normal"),
    ),
]

data = dict(
    train=dict(
        type="ARKitScenesLocate3DDataset",
        annotation_file=train_annotation,
        data_root=arkit_root,
        split=("Training", "Validation"),
        transform=train_transform,
        test_mode=False,
        loop=10,
    ),
    val=dict(
        type="ARKitScenesLocate3DDataset",
        annotation_file=val_annotation,
        data_root=arkit_root,
        split=("Training", "Validation"),
        transform=val_transform,
        test_mode=False,
        loop=1,
    ),
)

hooks = [
    dict(
        type="CheckpointLoader",
        keywords="module.",
        replacement="module.backbone.",
    ),
    dict(type="Locate3DStartupSanity"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="Locate3DMetricsLogger", log_train_every=1),
    dict(type="CheckpointSaver", save_freq=None),
]
