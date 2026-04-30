"""
A/B variant of localize-utonia-v1m1-0a-arkitscenes.py with the encoder
fully frozen.

Motivation
----------
The 0a run (encoder trained at 0.1x LR alongside the decoder) produced
stable, spatially-diversified queries (thanks to the fixed anchor grid)
but the bbox head's per-query offset regression failed to converge:
loss_bbox / loss_giou plateaued and dbg_match_iou stayed ~0.02 across 5
epochs. Plausible cause: a random-init decoder backpropagating into the
pretrained Utonia encoder destabilizes the very features the decoder is
trying to cross-attend to, so the BBoxHead never gets a stable spatial
signal and its offset weights do not learn meaningfully.

Since the Utonia encoder is reported to carry more geometric knowledge
than the 3D-JEPA encoder used in the Locate-3D paper, a common and safer
recipe is to freeze it and let only the decoder stack (queries,
cross-attention, text alignment head, bbox head) learn on top. This
config implements exactly that so the two runs can be compared directly.

Only difference vs. 0a: ``model.freeze_backbone=True``. Everything else
-- dataset, anchors, matcher / loss weights, transforms, hooks -- is
unchanged so this is a clean single-variable ablation.

Use the same CLI as 0a, e.g.::

    python tools/train.py \\
        --config-file configs/utonia/localize-utonia-v1m1-0b-arkitscenes.py \\
        -w /group-volume/utonia.pth \\
        --num-gpus <N>
"""

_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 16  # total bs across all gpus (per-GPU = batch_size // world_size)
num_worker = 32
mix_prob = 0.0
# clip_grad is intentionally loose. Locate-3D's decoder shares one BBoxHead
# across all 9 supervised layers (final + 8 aux), so the gradient summed at
# the head's parameters accumulates 9x for every batch -- combined with
# loss_weight_bbox=5 + loss_weight_giou=3 in raw meters, the global grad
# norm sits in the hundreds at init. clip_grad=1.0 was scaling the
# bbox-head update down ~100-1000x and the head's offset weights essentially
# could not move (loss_bbox stalled at ~3, dbg_match_iou ~0.015). 10.0
# leaves the head room to learn while still catching genuine spikes.
clip_grad = 10.0
empty_cache = False
empty_cache_per_epoch = True
enable_amp = True
amp_dtype = "bfloat16"
find_unused_parameters = True
evaluate = True
enable_wandb = False

# use a dedicated trainer that understands per-sample caption / box targets
train = dict(type="Locate3DTrainer")

# dataset paths
arkit_root = "/group-volume/3Ddataset/arkitscenes-compressed"
train_annotation = "locate-3d/locate3d_data/train_arkitscenes.json"
val_annotation = "locate-3d/locate3d_data/val_arkitscenes.json"

# path to a Utonia pretrain checkpoint. Pointcept's CheckpointLoader
# reads from ``cfg.weight``; wire it through so setting this variable
# automatically loads the ckpt. The CLI ``-w`` flag overrides this.
utonia_pretrained_path = None
weight = utonia_pretrained_path

# decoder configuration
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

# model settings -- note freeze_backbone=True below is the ONLY
# difference vs. the 0a config.
model = dict(
    type="Locate3DLocalizer",
    backbone_out_channels=54,
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
        freeze_encoder=False,
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

# scheduler settings
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
# No backbone param group: the encoder is frozen so there is nothing to
# schedule a separate LR for. All remaining trainable params (decoder,
# queries, text-alignment / bbox heads) follow the single base_lr.

# dataset settings
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

# hooks
hooks = [
    dict(
        type="CheckpointLoader",
        keywords="module.student.backbone",
        replacement="module.backbone",
    ),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="Locate3DDebugPrinter", print_every=50),
    dict(type="Locate3DGroundingEvaluator", iou_thresholds=(0.25, 0.5)),
    dict(type="Locate3DMetricsLogger", log_train_every=1),
    dict(type="Locate3DVizHook", num_scenes=3, top_k=5, viz_freq=1),
    dict(type="CheckpointSaver", save_freq=None),
]
