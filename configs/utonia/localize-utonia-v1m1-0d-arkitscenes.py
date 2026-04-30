"""
Variant 0d: keep the full-resolution U-Net decoder but TRAIN IT from scratch,
freezing only the pretrained encoder.

Why this config exists
----------------------
0c switches to ``enc_mode=True`` to dodge the random-U-Net-decoder problem
but pays a ~16x resolution cost (0.32 m voxels). If that coarse resolution
caps Acc@0.25 below what we need, the alternative is to keep
``enc_mode=False`` (54-dim at original resolution) BUT actually train the
U-Net decoder stack from scratch while the pretrained encoder stays frozen.

This is exactly the recipe used by ``semseg-utonia-v1m1-0b-scannet-dec.py``,
which successfully trains the same U-Net decoder (ScanNet dense labels, 800
epochs). We replicate the knobs here: ``freeze_encoder=True`` inside the
backbone config + ``freeze_backbone=False`` at the model level so that only
the encoder's ``embedding`` + ``enc`` stacks freeze, and the ``dec`` stack
gets real gradients.

Tradeoffs
---------
- (+) Full voxel-grid-size resolution for the decoder's cross-attention.
- (+) Final feature dim is 54 (cheap to project to d_model=768).
- (-) U-Net decoder is ~3.7M params starting from random init. On 991
  annotations × 50 epochs = ~50k updates that WILL NOT converge. We bump
  ``loop=10`` → 500k updates. Still an order of magnitude less than the
  semseg-dec recipe, but sparse referring-expression loss alone is a weaker
  training signal than dense per-point classification, so this is the
  practical ceiling without extra data.
- (-) Slower per-step (decoder cross-attention over ~20-40k points).

How to run
----------
    python tools/train.py \\
        --config-file configs/utonia/localize-utonia-v1m1-0d-arkitscenes.py \\
        -w /group-volume/utonia.pth \\
        --num-gpus <N>
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
    backbone_out_channels=54,             # U-Net decoder output
    decoder_input_feat_dim=256,
    freeze_backbone=False,                # let the U-Net dec stack train
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
        enc_mode=False,                   # full U-Net forward
        freeze_encoder=True,              # freeze ONLY pretrained encoder part
        rope_base=10,
        shift_coords=None,
        jitter_coords=1.1,
        rescale_coords=1.2,
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
    max_points_train=40000,
    max_points_eval=40000,
)

# Per-param-group LR: give the U-Net decoder a higher LR than the Locate-3D
# decoder so it can actually learn from random init in 50 epochs. Keyword
# "backbone.dec" matches only the U-Net decoder submodule inside the
# backbone; the frozen encoder has requires_grad=False and is skipped.
epoch = 100                               # 2x the 0a/0b budget
eval_epoch = 100
base_lr = 2e-4
optimizer = dict(type="AdamW", lr=base_lr, weight_decay=0.01)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[base_lr, base_lr * 2],        # Locate-3D decoder at base_lr, U-Net dec at 2x
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
        loop=10,                           # 10x per-epoch iterations
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
