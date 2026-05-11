"""Utonia × 3D-JEPA — Stage 1: Additive JEPA finetune.

Utonia v1m1 stagev1 pretrained weight 위에 JEPA-style 연속 latent prediction 손실을
추가해서 짧게 finetune. 자세한 설계: docs/utonia_jepa_staged_plan.md

차이 (v1m1 stagev1 대비):
- model.type: Utonia-v1m1 → Utonia-v1m3
- predictor (JEPAPredictor) 및 jepa_loss 추가
- 4-loss 가중치 재배분 + jepa 2/10 추가 (합 = 1 유지)
- finetune: epoch 15, lr=base_lr/5, momentum 0.994 → 0.999
- CheckpointLoader strict=False (새 predictor 모듈 random init)
- ★ 학습 전: cfg.weight를 v1m1 stagev1 final checkpoint 경로로 설정해야 함

검증: semseg-utonia-v1m1-0a-scannet-lin.py / -0c-scannet-ft.py를 새 weight로 재실행 후
v1m1 baseline과 mIoU 비교.
"""

_base_ = ["../_base_/default_runtime.py"]

# ★ 학습 전 이 경로를 v1m1 stagev1 final checkpoint로 채워야 함
weight = "exp/utonia/pretrain-utonia-v1m1-0-base_stagev1/model/model_last.pth"
resume = False

# misc custom setting
crop_h = 518
crop_w = 518
patch_size = 14
batch_size = 256
num_worker = 1024
mix_prob = 0.0
clip_grad = 1.0

empty_cache = True
enable_amp = True
amp_dtype = "bfloat16"
evaluate = False
find_unused_parameters = True


# model settings
model = dict(
    type="Utonia-v1m3",
    patch_h=crop_h // patch_size,
    patch_w=crop_w // patch_size,
    image_weight_name="dinov2_vitg14_reg",
    image_weight_path="facebook/dinov2-with-registers-giant",
    backbone_out_channels=1332,
    embedding_channels=64,
    student_pretrained=False,
    enc2d_upcast_level=3,
    # backbone - student & teacher
    backbone=dict(
        type="PT-v3m3",
        in_channels=9,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(54, 108, 216, 432, 576),
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
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
        enc_mode=True,
        traceable=True,
        mask_token=True,
        rope_base=10,
        shift_coords=None,
        jitter_coords=1.1,
        rescale_coords=1.2,
    ),
    teacher_custom=dict(
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
    ),
    head_in_channels=576,
    head_hidden_channels=4096,
    head_embed_channels=256,
    head_num_prototypes=4096,
    enc2d_head_in_channels=1536,
    enc2d_head_hidden_channels=4096,
    enc2d_head_embed_channels=256,
    enc2d_head_num_prototypes=4096,
    num_global_view=2,
    num_local_view=4,
    mask_size_start=10,
    mask_size_base=40,
    mask_size_warmup_ratio=0.05,
    mask_ratio_start=0.3,
    mask_ratio_base=0.7,
    mask_ratio_warmup_ratio=0.05,
    mask_jitter=0.5,
    teacher_temp_start=0.04,
    teacher_temp_base=0.07,
    teacher_temp_warmup_ratio=0.05,
    student_temp=0.1,
    # === Stage 1: rebalanced 4-loss + new jepa loss (합 = 1) ===
    mask_loss_weight=1 / 10,
    roll_mask_loss_weight=1 / 10,
    unmask_loss_weight=2 / 10,
    enc2d_loss_weight=4 / 10,
    jepa_loss_weight=2 / 10,
    # === JEPA predictor settings ===
    jepa_predictor_hidden_channels=2048,
    jepa_predictor_num_layers=3,
    jepa_predictor_dropout=0.0,
    jepa_use_cosine=False,   # smooth_l1 on normalized features (좀 더 안정적)
    jepa_normalize=True,
    # finetune이라 EMA 더 느리게
    momentum_base=0.999,
    momentum_final=1,
    match_max_k=8,
    match_max_r=0.32,
    up_cast_level=0,
    enc2d_cos_shift=True,
    sonata_model_type="online",
)

# scheduler settings — finetune: epoch 15, lr 1/5
epoch = 15
eval_epoch = 15       # cfg.epoch % cfg.eval_epoch == 0 만족 (default 100 그대로면 assert 실패)
base_lr = 0.0008      # v1m1 stagev1: 0.004 → 1/5
lr_decay = 0.9
base_wd = 0.04
final_wd = 0.2

dec_depths = model["backbone"]["enc_depths"]
param_dicts = [
    dict(
        keyword=f"enc{e}.block{b}.",
        lr=base_lr * lr_decay ** (sum(dec_depths) - sum(dec_depths[:e]) - b - 1),
    )
    for e in range(len(dec_depths))
    for b in range(dec_depths[e])
]
# Predictor는 새 모듈 — 별도 lr_scale 없이 base_lr로 학습되도록 param_dicts에 추가하지 않음
# (param_dicts에 매칭되지 않는 파라미터는 자동으로 base_lr 그룹 0번에 들어감)
del dec_depths

optimizer = dict(type="AdamW", lr=base_lr, weight_decay=base_wd)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[base_lr] + [g["lr"] for g in param_dicts],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

# dataset settings — v1m1 stagev1과 동일
outdoor_transform = [
    dict(
        type="Update",
        keys_dict={
            "index_valid_keys": (
                "coord",
                "origin_coord",
                "color",
                "normal",
                "superpoint",
                "strength",
                "instance",
                "correspondence",
                "global_correspondence",
            )
        },
    ),
    dict(
        type="ImgAugmentation",
        crop_h=crop_h,
        crop_w=crop_w,
        patch_h=crop_h // patch_size,
        patch_w=crop_w // patch_size,
        patch_size=patch_size,
        imgtransforms=[
            dict(type="ImgChromaticJitter", p=0.95, std=0.05),
            dict(type="ImgGaussianBlur", p=0.5),
            dict(
                type="Imgnormalize",
                mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD,
            ),
        ],
    ),
    dict(type="Copy", keys_dict={"coord": "origin_coord"}),
    dict(type="RandomScale", scale=[0.18, 0.22]),
    dict(type="GridSample", grid_size=0.01, hash_type="fnv", mode="train"),
    dict(type="RandomDropColor", drop_ratio=1.0, drop_application_ratio=0.2),
    dict(type="RandomDropColor", drop_ratio=0.1, drop_application_ratio=0.5),
    dict(type="RandomDropNormal", drop_ratio=1.0, drop_application_ratio=0.2),
    dict(type="RandomDropNormal", drop_ratio=0.1, drop_application_ratio=0.5),
    dict(
        type="MultiViewGenerator",
        view_keys=("coord", "origin_coord", "color", "correspondence", "normal"),
        if_frame_selected=True,
        global_view_num=2,
        global_view_scale=(0.4, 1.0),
        local_view_num=4,
        local_view_scale=(0.1, 0.4),
        global_shared_transform=[
            dict(type="NormalizeColor"),
        ],
        global_transform=[
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.8),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.8),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.8),
            dict(type="RandomFlip", p=0.5),
            dict(
                type="PointClip",
                point_cloud_range=(
                    -75.2 * 0.2,
                    -75.2 * 0.2,
                    -4 * 0.2,
                    75.2 * 0.2,
                    75.2 * 0.2,
                    2 * 0.2,
                ),
            ),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomJitter", sigma=0.0025, clip=0.01),
        ],
        local_transform=[
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.8),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.8),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.8),
            dict(type="RandomFlip", p=0.5),
            dict(
                type="PointClip",
                point_cloud_range=(
                    -75.2 * 0.2,
                    -75.2 * 0.2,
                    -4 * 0.2,
                    75.2 * 0.2,
                    75.2 * 0.2,
                    2 * 0.2,
                ),
            ),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomJitter", sigma=0.0025, clip=0.01),
            dict(type="NormalizeColor"),
        ],
        max_size=32768,
        enc2d_max_size=32768,
        enc2d_scale=(0.8, 1),
    ),
    dict(type="ToTensor"),
    dict(type="Update", keys_dict={"grid_size": 0.01}),
    dict(
        type="Collect",
        keys=(
            "global_origin_coord",
            "global_coord",
            "global_offset",
            "local_origin_coord",
            "local_coord",
            "local_offset",
            "grid_size",
            "name",
            "images",
            "global_correspondence",
            "img_num",
        ),
        offset_keys_dict=dict(),
        global_feat_keys=("global_coord", "global_color", "global_normal"),
        local_feat_keys=("local_coord", "local_color", "local_normal"),
    ),
]

obj_transform = [
    dict(
        type="Update",
        keys_dict={
            "index_valid_keys": (
                "coord",
                "origin_coord",
                "color",
                "normal",
                "superpoint",
                "strength",
                "segment",
                "instance",
                "correspondence",
                "global_correspondence",
            )
        },
    ),
    dict(
        type="ImgAugmentation",
        crop_h=crop_h,
        crop_w=crop_w,
        patch_h=crop_h // patch_size,
        patch_w=crop_w // patch_size,
        patch_size=patch_size,
        imgtransforms=[
            dict(type="ImgChromaticJitter", p=0.95, std=0.05),
            dict(type="ImgGaussianBlur", p=0.5),
            dict(
                type="Imgnormalize",
                mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD,
            ),
        ],
    ),
    dict(type="NormalizeCoord"),
    dict(type="RandomScale", scale=[0.5, 1.5]),
    dict(type="Copy", keys_dict={"coord": "origin_coord"}),
    dict(type="GridSample", grid_size=0.01, hash_type="fnv", mode="train"),
    dict(type="RandomDropColor", drop_ratio=1.0, drop_application_ratio=0.2),
    dict(type="RandomDropColor", drop_ratio=0.1, drop_application_ratio=0.5),
    dict(type="RandomDropNormal", drop_ratio=1.0, drop_application_ratio=0.2),
    dict(type="RandomDropNormal", drop_ratio=0.1, drop_application_ratio=0.5),
    dict(
        type="MultiViewGenerator",
        global_view_num=2,
        global_view_scale=(0.8, 1.0),
        local_view_num=4,
        local_view_scale=(0.6, 0.8),
        global_shared_transform=[
            dict(
                type="RandomColorJitter",
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.02,
                p=0.8,
            ),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="NormalizeColor"),
        ],
        global_transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomShift", shift=((-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2))),
            dict(type="RandomScale", scale=[0.5, 1.5]),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.8),
            dict(type="RandomRotate", angle=[-1, 1], axis="x", p=0.8),
            dict(type="RandomRotate", angle=[-1, 1], axis="y", p=0.8),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
        ],
        local_transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomShift", shift=((-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2))),
            dict(type="RandomScale", scale=[0.5, 1.5]),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.8),
            dict(type="RandomRotate", angle=[-1, 1], axis="x", p=0.8),
            dict(type="RandomRotate", angle=[-1, 1], axis="y", p=0.8),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(
                type="RandomColorJitter",
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.02,
                p=0.8,
            ),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="NormalizeColor"),
        ],
        max_size=65536,
        enc2d_max_size=65536,
        enc2d_scale=(0.8, 1),
    ),
    dict(type="ToTensor"),
    dict(type="Update", keys_dict={"grid_size": 0.01}),
    dict(
        type="Collect",
        keys=(
            "global_origin_coord",
            "global_coord",
            "global_offset",
            "local_origin_coord",
            "local_coord",
            "local_offset",
            "grid_size",
            "name",
            "images",
            "global_correspondence",
            "img_num",
        ),
        offset_keys_dict=dict(),
        global_feat_keys=("global_coord", "global_color", "global_normal"),
        local_feat_keys=("local_coord", "local_color", "local_normal"),
    ),
]

indoor_transform = [
    dict(
        type="Update",
        keys_dict={
            "index_valid_keys": (
                "coord",
                "origin_coord",
                "color",
                "normal",
                "superpoint",
                "strength",
                "segment",
                "instance",
                "correspondence",
                "global_correspondence",
            )
        },
    ),
    dict(
        type="ImgAugmentation",
        crop_h=crop_h,
        crop_w=crop_w,
        patch_h=crop_h // patch_size,
        patch_w=crop_w // patch_size,
        patch_size=patch_size,
        imgtransforms=[
            dict(type="ImgChromaticJitter", p=0.95, std=0.05),
            dict(type="ImgGaussianBlur", p=0.5),
            dict(
                type="Imgnormalize",
                mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD,
            ),
        ],
    ),
    dict(type="Copy", keys_dict={"coord": "origin_coord"}),
    dict(type="RandomScale", scale=[0.45, 0.55]),
    dict(type="GridSample", grid_size=0.01, hash_type="fnv", mode="train"),
    dict(type="RandomDropColor", drop_ratio=1.0, drop_application_ratio=0.2),
    dict(type="RandomDropColor", drop_ratio=0.1, drop_application_ratio=0.5),
    dict(type="RandomDropNormal", drop_ratio=1.0, drop_application_ratio=0.2),
    dict(type="RandomDropNormal", drop_ratio=0.1, drop_application_ratio=0.5),
    dict(
        type="MultiViewGenerator",
        global_view_num=2,
        global_view_scale=(0.4, 1.0),
        local_view_num=4,
        local_view_scale=(0.1, 0.4),
        global_shared_transform=[
            dict(
                type="RandomColorJitter",
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.02,
                p=0.8,
            ),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="NormalizeColor"),
        ],
        global_transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.8),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.8),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.8),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.0025, clip=0.01),
            dict(type="ElasticDistortion", distortion_params=[[0.1, 0.2], [0.4, 0.8]]),
        ],
        local_transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.8),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.8),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.8),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.0025, clip=0.01),
            dict(type="ElasticDistortion", distortion_params=[[0.1, 0.2], [0.4, 0.8]]),
            dict(
                type="RandomColorJitter",
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.02,
                p=0.8,
            ),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="NormalizeColor"),
        ],
        max_size=65536,
        enc2d_max_size=65536,
        enc2d_scale=(0.8, 1),
    ),
    dict(type="ToTensor"),
    dict(type="Update", keys_dict={"grid_size": 0.01}),
    dict(
        type="Collect",
        keys=(
            "global_origin_coord",
            "global_coord",
            "global_offset",
            "local_origin_coord",
            "local_coord",
            "local_offset",
            "grid_size",
            "name",
            "images",
            "global_correspondence",
            "img_num",
        ),
        offset_keys_dict=dict(),
        global_feat_keys=("global_coord", "global_color", "global_normal"),
        local_feat_keys=("local_coord", "local_color", "local_normal"),
    ),
]

data_weight = None
data_length = None
data = dict(
    train=dict(
        type="ConcatDataset",
        datasets=[
            # Waymo / PartNet은 Stage 1 단순 검증을 위해 비활성화 (실내 두 셋만 사용)
            # dict(
            #     type="WaymoImagePointDataset",
            #     if_sweep=True,
            #     if_img=True,
            #     sweeps=3,
            #     sweep_gap=1,
            #     crop_h=crop_h,
            #     crop_w=crop_w,
            #     patch_size=patch_size,
            #     split=["training", "validation"],
            #     data_root="data/waymo",
            #     transform=outdoor_transform,
            #     test_mode=False,
            #     loop=1,
            # ),
            # dict(
            #     type="PartNetDataDataset",
            #     crop_h=crop_h,
            #     crop_w=crop_w,
            #     patch_size=patch_size,
            #     split=["train"],
            #     data_root="data/partnet_data_v0",
            #     transform=obj_transform,
            #     test_mode=False,
            #     loop=1,
            # ),
            dict(
                type="DefaultImagePointDataset",
                crop_h=crop_h,
                crop_w=crop_w,
                patch_size=patch_size,
                split=["train", "val", "test"],
                data_root="/group-volume/3Ddataset/scannet-compressed",
                transform=indoor_transform,
                test_mode=False,
                loop=1,
            ),
            dict(
                type="DefaultImagePointDataset",
                crop_h=crop_h,
                crop_w=crop_w,
                patch_size=patch_size,
                split=["train", "val", "test"],
                data_root="/group-volume/3Ddataset/structure3d-compressed",
                transform=indoor_transform,
                test_mode=False,
                loop=1,
            ),
        ],
    )
)

hooks = [
    # strict=False → 새로 추가된 self.predictor가 missing key로 보고되지만 정상 동작
    dict(type="CheckpointLoader", strict=False),
    dict(type="ModelHook"),
    dict(type="WeightDecaySchedular", base_value=base_wd, final_value=final_wd),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="CheckpointSaver", save_freq=5),
]
