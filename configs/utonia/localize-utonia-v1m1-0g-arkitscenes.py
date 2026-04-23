"""
Variant 0g: use the STAGE-3 encoder feature (1/8 res, 432-d) instead of the
bottleneck. This is the level Utonia's 2D-3D DINOv2 alignment actually
trained ("enc2d_upcast_level=3").

Why
---
- 0c uses ``enc_mode=True`` returning the bottleneck (stride 16, 576 ch).
  But Utonia's pretext connects the DINOv2 image-patch alignment head at
  stage 3 of the encoder (432 ch, stride 8), not at the bottleneck. The
  bottleneck is only supervised by masked-patch reconstruction, which is
  a local-geometry objective; the stage-3 features are what carry the
  semantic, DINOv2-adjacent structure that language-grounding benefits
  from.
- Stage-3 resolution is stride 8 vs 16 → 2x finer voxel spacing (0.16 m
  vs 0.32 m). For median ARKitScenes GT size (0.45 m) that's 2-3 voxels
  across instead of 1-2 — enough for non-trivial IoU.
- Channel count 432 vs 576 is negligibly different after the
  ``feat_proj: Linear(432, 256)`` bottleneck.

We access stage-3 via the GridPooling ``pooling_parent`` chain: the
bottleneck point's parent IS the stage-3 point (encoded at 432 ch after
that stage's Block stack but before the final pool). The new
``Locate3DLocalizer.backbone_feature_level=1`` knob steps up one level
from whatever the backbone returns.

How to run
----------
    python tools/train.py \\
        --config-file configs/utonia/localize-utonia-v1m1-0g-arkitscenes.py \\
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

train = dict(type="Locate3DTrainer")

arkit_root = "/group-volume/3Ddataset/arkitscenes-compressed"
train_annotation = "locate-3d/locate3d_data/train_arkitscenes.json"
val_annotation = "locate-3d/locate3d_data/val_arkitscenes.json"

utonia_pretrained_path = None
weight = utonia_pretrained_path

decoder_cfg = dict(
    d_model=768,
    num_queries=64,
    num_decoder_layers=8,
    transformer_n_heads=12,
    transformer_dim_feedforward=3072,
    transformer_dropout=0.1,
    transformer_max_drop_path=0.0,
    transformer_use_checkpointing=True,
    freeze_text_encoder=True,
    text_encoder="clip",
    text_conditioned_queries=True,
)

model = dict(
    type="Locate3DLocalizer",
    # Stage-3 encoder channels. PT-v3m3 enc_channels=(54,108,216,432,576),
    # so stage 3 (zero-indexed) = 432.
    backbone_out_channels=432,
    decoder_input_feat_dim=256,
    freeze_backbone=True,
    # KEY: step 1 level up from the backbone's returned point. Combined
    # with enc_mode=True (returns bottleneck = stage 4), this lands us on
    # stage 3 (432 ch at stride 8).
    backbone_feature_level=1,
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
        drop_path=0.0,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        traceable=False,
        mask_token=False,
        enc_mode=True,
        freeze_encoder=False,
        rope_base=10,
        shift_coords=None,
        jitter_coords=None,
        rescale_coords=None,
    ),
    decoder=decoder_cfg,
    matcher_cost_class=1.0,
    matcher_cost_bbox=5.0,
    matcher_cost_giou=4.0,
    loss_weight_class=1.0,
    loss_weight_bbox=5.0,
    loss_weight_giou=3.0,
    loss_weight_mask_bce=0.0,
    loss_weight_mask_dice=0.0,
    focal_alpha=0.25,
    focal_gamma=2.0,
    aux_loss=True,
    aux_layer_weights=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
    max_points_train=None,
    max_points_eval=None,
)

epoch = 50
eval_epoch = 50
base_lr = 1e-4
optimizer = dict(type="AdamW", lr=base_lr, weight_decay=0.01)
scheduler = dict(
    type="OneCycleLR",
    max_lr=base_lr,
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

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
        keywords="module.student.backbone",
        replacement="module.backbone",
    ),
    dict(type="Locate3DStartupSanity"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="Locate3DDebugPrinter", print_every=50),
    dict(type="Locate3DGroundingEvaluator", iou_thresholds=(0.25, 0.5)),
    dict(type="Locate3DMetricsLogger", log_train_every=1),
    dict(type="Locate3DVizHook", num_scenes=3, top_k=5, viz_freq=1),
    dict(type="CheckpointSaver", save_freq=None),
]
