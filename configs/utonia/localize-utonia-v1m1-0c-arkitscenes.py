"""
Variant 0c: use the PRETRAINED encoder bottleneck (enc_mode=True) instead of
the random-init U-Net decoder output.

Why this config exists
----------------------
Utonia pretraining (``pretrain-utonia-v1m1-0-base_stagev1.py``) runs with
``enc_mode=True`` — the pretext updates only the ``embedding`` + ``enc``
stacks. The released checkpoint therefore contains NO ``dec.*`` weights.
Our earlier 0a / 0b configs ran the backbone with ``enc_mode=False``, which
instantiates a full U-Net decoder stack whose weights are RANDOM at step 0
(``strict=False`` CheckpointLoader silently leaves missing keys alone).
0b then sets ``freeze_backbone=True`` on top of that — freezing the random
decoder weights forever. Net result: we fed the Locate-3D decoder 54-dim
features from a random U-Net decoder. Unsurprisingly match_iou stayed at
~0.015 and boxes never approached GT.

This config sidesteps the issue by running the backbone in ``enc_mode=True``:
the backbone returns the 576-dim encoder bottleneck (1/16 voxel resolution
≈ 0.32 m voxels at grid_size=0.02) straight from fully-pretrained Utonia
weights. No random decoder involved.

Tradeoffs
---------
- (+) All features used downstream are 100% pretrained.
- (+) Fewer points for the decoder (~1-3k per scene vs ~40k), so faster
  iteration and more headroom to bump num_queries if needed.
- (-) Coarser spatial resolution (0.32 m voxels). For the median ARKitScenes
  GT box (size ~0.45 m) that's 1-2 voxels across, so ~IoU-0.25 precision is
  the realistic ceiling at this resolution. If higher is needed, see 0d.

How to run
----------
Identical CLI to 0a/0b::

    python tools/train.py \\
        --config-file configs/utonia/localize-utonia-v1m1-0c-arkitscenes.py \\
        -w /group-volume/utonia.pth \\
        --num-gpus <N>
"""

_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
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
)

model = dict(
    type="Locate3DLocalizer",
    # KEY CHANGE: bottleneck channel count (enc_channels[-1]=576), not 54.
    backbone_out_channels=576,
    decoder_input_feat_dim=256,
    # Backbone is fully pretrained now (only ``embedding`` + ``enc`` are
    # loaded and used). freeze=True is safe because we're freezing actually-
    # pretrained weights, not random ones.
    freeze_backbone=True,
    backbone=dict(
        type="PT-v3m3",
        in_channels=9,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(54, 108, 216, 432, 576),
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        # dec_* kwargs are kept for structural compat but unused when enc_mode=True.
        dec_depths=(2, 2, 2, 2),
        dec_channels=(54, 108, 216, 432),
        dec_num_head=(3, 6, 12, 24),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        # Drop-path and coord perturbation add noise through a FROZEN backbone,
        # never getting trained away -- turn them off.
        drop_path=0.0,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        traceable=False,
        mask_token=False,
        # THE key flag. Returns the bottleneck point (576-d at 1/16 res).
        enc_mode=True,
        freeze_encoder=False,  # superseded by freeze_backbone=True
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
    # Bottleneck resolution gives at most a few thousand points per scene, so
    # the subsample cap is a no-op. Set to None so the code path is clean.
    max_points_train=None,
    max_points_eval=None,
)

# scheduler
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
        loop=1,
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
