"""
Utonia encoder + Locate-3D decoder on ARKitScenes localization downstream.

Loads a Utonia-pretrained PT-v3m3 encoder (from
`pretrain-utonia-v1m1-0-base_stagev1.py`) and trains the Locate-3D
language-conditioned transformer decoder (Meta's facebookresearch/locate-3d
`models/locate_3d_decoder.py`) to predict 3D bounding boxes / masks for the
referring expressions in `locate-3d/locate3d_data/{train,val}_arkitscenes.json`.

The training pipeline uses the MDETR / BUTD-DETR-style set-prediction loss
(Hungarian matcher + sigmoid-focal text alignment + L1 + GIoU) as described in
the Locate-3D paper (https://arxiv.org/abs/2504.14151).
"""

_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 16  # total bs across all gpus
num_worker = 16
mix_prob = 0.0
clip_grad = 1.0
empty_cache = False
enable_amp = True
amp_dtype = "bfloat16"
find_unused_parameters = True
evaluate = True

# use a dedicated trainer that understands per-sample caption / box targets
train = dict(type="Locate3DTrainer")

# dataset paths
arkit_root = "/group-volume/arkitscenes-compressed"
train_annotation = "locate-3d/locate3d_data/train_arkitscenes.json"
val_annotation = "locate-3d/locate3d_data/val_arkitscenes.json"

# path to a Utonia pretrain checkpoint (fill in when available)
utonia_pretrained_path = None

# decoder configuration
decoder_cfg = dict(
    d_model=768,
    num_queries=256,
    num_decoder_layers=8,
    transformer_n_heads=12,
    transformer_dim_feedforward=3072,
    transformer_dropout=0.1,
    transformer_max_drop_path=0.0,
    transformer_use_checkpointing=True,
    freeze_text_encoder=True,
    text_encoder="clip",
)

# model settings
model = dict(
    type="Locate3DLocalizer",
    backbone_out_channels=54,  # output channel of PT-v3m3 at level 0
    decoder_input_feat_dim=256,
    freeze_backbone=False,
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
    matcher_cost_giou=2.0,
    loss_weight_class=1.0,
    loss_weight_bbox=5.0,
    loss_weight_giou=2.0,
    loss_weight_mask_bce=0.0,
    loss_weight_mask_dice=0.0,
    focal_alpha=0.25,
    focal_gamma=2.0,
    aux_loss=True,
    max_points_train=40000,
)

# scheduler settings
epoch = 50
eval_epoch = 50
base_lr = 1e-4
optimizer = dict(type="AdamW", lr=base_lr, weight_decay=0.01)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[base_lr, base_lr * 0.1],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
# lower LR for the pretrained encoder backbone
param_dicts = [dict(keyword="backbone", lr=base_lr * 0.1)]

# dataset settings
# NOTE: geometric augmentations (rotate/flip/scale/center-shift) are avoided
# so that the point cloud stays in the same frame as the absolute gt_boxes
# coming from the Locate-3D annotations. Only color/photometric augmentation
# and voxel-grid subsampling are applied.
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
            # localization target fields are passed through as-is
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
    # load only the student.backbone weights from the Utonia pretrain ckpt
    dict(
        type="CheckpointLoader",
        keywords="module.student.backbone",
        replacement="module.backbone",
    ),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    # log query-collapse / matching diagnostics to stdout every 50 iters
    dict(type="Locate3DDebugPrinter", print_every=50),
    dict(type="Locate3DGroundingEvaluator", iou_thresholds=(0.25, 0.5)),
    # plotly HTML with RGB pointcloud + gt/pred boxes for 3 fixed val scenes
    dict(type="Locate3DVizHook", num_scenes=3, top_k=5, viz_freq=1),
    dict(type="CheckpointSaver", save_freq=None),
]
