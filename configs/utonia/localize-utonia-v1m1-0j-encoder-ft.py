"""
Variant 0j: stage-2 fine-tune. UNFREEZE the Utonia encoder at a very
low LR while continuing to train the U-Net decoder + heads at the
0i pace.

Why this config exists
----------------------
0i (and 0h before it) freeze the Utonia encoder and only train the
U-Net decoder + SegDetector heads. Once those plateau, the next lever
is the encoder itself: letting the pretrained features drift slightly
to specialize for referring-expression grounding can buy a few extra
points of val Acc, especially at the stricter Acc@0.5 level.

The risk of unfreezing pretrained encoders mid-training is that a
naive shared LR overshoots the encoder weights and destroys the
pretraining. Mitigations baked in here:

1. Start from a good checkpoint. Pass ``-w <path-to-0i-or-0h-best>.pth``
   on the CLI; CheckpointLoader will load every weight (the rename
   keyword "module.student.backbone" -> "module.backbone" is a no-op
   on training-style ckpts, so all heads + dec + encoder come in).
2. Encoder LR = ``base_lr * 0.01`` (= 1e-6 with base_lr=1e-4).
   Embedding stem matches.
3. Reduced base_lr 2e-4 -> 1e-4 (decoder + heads no longer "from
   scratch", so don't need the higher LR).
4. Shorter schedule (epoch=40) since this is fine-tuning, not full
   training. OneCycleLR's cosine anneal will deliver a clean tail.
5. CLIP text encoder STAYS frozen (large model, easy to overfit;
   leave for a possible stage-3 if needed).

Knobs reused from 0i (real-mask hyperparameter tune): pos_weight=30,
loss_weight_dice=2, max_points_train=60000, infer_threshold=0.55.

How to run
----------
    python tools/train.py \\
        --config-file configs/utonia/localize-utonia-v1m1-0j-encoder-ft.py \\
        -w exp/<your-0i-run>/model/model_best.pth \\
        --num-gpus 4

Stop early if val Acc starts regressing -- that's the encoder
losing more pretrain than it gains in ground-task adaptation.
"""

_base_ = ["../_base_/default_runtime.py"]

# Memory budget for H100 x4 with encoder UNFROZEN (encoder activations
# are now stored for backward, not just U-Net dec): hold per-GPU peak
# memory roughly constant by trading max_points for batch parallelism.
#
#   peak_pts / GPU = (batch_size / world_size) * max_points_train
#
# user previously hit OOM at bs=8 / max_points=60k = 120k pts/GPU @
# encoder unfrozen. Move the same 120k budget to bs=16 / max_points=30k
# = 4x more samples per GPU step -> H100 compute saturates better
# (50% -> ~80% util) without raising the peak.
batch_size = 16
# Per-GPU workers = num_worker / world_size = 24/4 = 6. Locate-3D
# preprocessing (CLIP tokenizer + GridSample hash + per-point inside-
# box checks) is moderately CPU-heavy; 6 / GPU keeps the device fed.
num_worker = 24
mix_prob = 0.0
clip_grad = 10.0
# Per-iter cache cleanup. Slight throughput cost but cuts down allocator
# fragmentation that's the typical cause of intermittent OOMs after
# variable-sized scenes (ScanNet++ scenes are particularly variable).
empty_cache = True
empty_cache_per_epoch = True
enable_amp = True
amp_dtype = "bfloat16"
find_unused_parameters = True
evaluate = True
enable_wandb = False

train = dict(type="Locate3DTrainer")

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

# Pass -w on the CLI to start from a 0i / 0h checkpoint; this is the
# default empty path which would mean "from scratch" (not recommended
# for stage-2 fine-tune).
utonia_pretrained_path = None
weight = utonia_pretrained_path

model = dict(
    type="Locate3DSegDetector",
    backbone_out_channels=54,
    d_model=512,
    freeze_backbone=False,
    freeze_text_encoder=True,
    text_encoder="clip",
    # Encoder unfrozen -> all PT-v3m3 transformer block activations are
    # otherwise stored for backward, which is the dominant memory cost
    # at H100 x4. Recompute via gradient checkpointing instead. Cuts
    # peak VRAM ~40-60% at the cost of one extra forward (~30% slower
    # per step). Critical for fitting bs/gpu>1 with encoder unfrozen.
    backbone_grad_checkpoint=True,
    # Roll back to 0h-style loss / inference knobs. The 0i "tune for
    # real masks" attempt (pos_weight 100->30, dice 5->2,
    # infer_threshold 0.5->0.55) was measured worse: ARKit+ScanNet
    # val_Acc@0.25 0.54 (0h@e15) -> 0.20 (0i@e100). Keep 0h's
    # recall-favouring values which empirically work for the
    # mask-AABB-as-bbox metric.
    loss_weight_bce=1.0,
    loss_weight_dice=5.0,
    bce_pos_weight=100.0,
    # max_points trimmed from 0h's 40k -> 24k for memory headroom under
    # encoder-unfrozen training; combined with batch_size and
    # backbone_grad_checkpoint above this fits H100 x4 reliably.
    max_points_train=24000,
    max_points_eval=24000,
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
        # Lower drop_path on the encoder side now that it's training
        # (was 0.3 to regularize the random-init dec; encoder doesn't
        # need that much aug noise during fine-tune).
        drop_path=0.1,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        traceable=False,
        mask_token=False,
        enc_mode=False,
        # KEY DELTA vs 0i: encoder is UNFROZEN. Per-param-group LR
        # (param_dicts below) keeps it learning at base_lr * 0.01 so it
        # only drifts gently from the Utonia pretrain.
        freeze_encoder=False,
        rope_base=10,
        # Light coord augmentation -- under fine-tune we don't want the
        # encoder to see large coord perturbations through frozen-then-
        # newly-unfrozen weights.
        shift_coords=None,
        jitter_coords=1.05,
        rescale_coords=1.05,
    ),
)

# Stage-2 schedule: shorter and lower-LR than the from-scratch stage.
epoch = 40
eval_epoch = 40
base_lr = 1e-4

optimizer = dict(type="AdamW", lr=base_lr, weight_decay=0.01)

# Discriminative LR. The keyword match in build_optimizer is "first
# substring match wins" with ``break``, so ORDER MATTERS:
#   1. backbone.embedding -> encoder stem          (lr * 0.01)
#   2. backbone.enc       -> encoder transformer   (lr * 0.01)
#   3. backbone.dec       -> U-Net decoder         (lr * 1.0)
# Anything else (point_proj, text_proj, SegDetector head ...) lands in
# the default group at base_lr.
param_dicts = [
    dict(keyword="backbone.embedding", lr=base_lr * 0.01),
    dict(keyword="backbone.enc", lr=base_lr * 0.01),
    dict(keyword="backbone.dec", lr=base_lr * 1.0),
]
scheduler = dict(
    type="OneCycleLR",
    # max_lr is one entry per param-group, in order:
    #   [default(=heads), backbone.embedding, backbone.enc, backbone.dec]
    max_lr=[base_lr, base_lr * 0.01, base_lr * 0.01, base_lr * 1.0],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

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
    "0j-encoder-ft found no usable train dataset. Set at least one of "
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
        # Eval slightly more often during fine-tune so we catch
        # regression early (encoder unfreeze is the riskiest stage).
        eval_every_n_epochs=2,
    ),
    dict(type="Locate3DMetricsLogger", log_train_every=1),
    dict(type="CheckpointSaver", save_freq=None),
]

del _maybe, _train_datasets, _val_datasets
del _os, _USE_ARKIT, _USE_SCANNETPP
del _common_keys, _arkit_aug, _scannet_aug
del train_transform_arkit, train_transform_scannet
del val_transform_arkit, val_transform_scannet
