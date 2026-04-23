"""
Variant 0e: 0c (pretrained encoder bottleneck) + every architectural
enhancement we added. Try this if 0c plateaus.

Deltas vs 0c
------------
- ``text_conditioned_queries=True`` in decoder_cfg: seed every query's
  content embedding with a pooled CLIP text summary at layer 0 (zero-init
  so step-0 behavior is bit-exact, starts contributing as weights move).
  Addresses the "all queries start content-identical" symmetry that makes
  early layers uninformative for the shared BBoxHead.
- ``aux_layer_weights=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)``: linear ramp
  across the 7 aux layers; the final (non-aux) layer stays at 1.0. Keeps
  deep-supervision benefits for query specialization while preventing
  noisy early-layer gradients from owning the shared BBoxHead.
- ``loop=10``: ~500k gradient updates over 50 epochs, matching the Locate-3D
  paper's update budget class rather than the 50k we had.
- Box-aware geometric augmentation (``RandomFlipBoxAware``,
  ``RandomScaleBoxAware``) — safe with axis-aligned GT boxes.

How to run
----------
    python tools/train.py \\
        --config-file configs/utonia/localize-utonia-v1m1-0e-arkitscenes.py \\
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
    # NEW: enable text-pooled init so queries break content symmetry at step 0.
    text_conditioned_queries=True,
)

model = dict(
    type="Locate3DLocalizer",
    backbone_out_channels=576,
    decoder_input_feat_dim=256,
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
    # NEW: linear ramp 0.2 -> 0.8 across 7 aux layers; final stays at 1.0.
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
        keywords="module.",
        replacement="module.backbone.",
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
