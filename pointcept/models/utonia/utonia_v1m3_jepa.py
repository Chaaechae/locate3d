"""
Utonia V1M3 — Stage 1: Additive JEPA

Utonia v1m1 위에 *연속 latent prediction (JEPA-style)* 신호를 5번째 손실로 추가한 변형.

- 입력 포맷은 v1m1 그대로 (xyz + color + normal).
- 기존 mask / roll_mask / unmask / enc2d 4-loss 모두 유지.
- 새로 추가:
    * JEPAPredictor: student의 masked 위치 raw feature를 받아 target feature를 예측.
    * jepa_loss: smooth-L1 (또는 cosine) on L2-normalized features,
      target = EMA teacher backbone의 raw feature (stop-grad).
- Collapse 방지: 기존 Sinkhorn + EMA + (새) predictor 비대칭이 함께 작동.

자세한 단계 계획: docs/utonia_jepa_staged_plan.md
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch_scatter

from pointcept.models.utils.structure import Point
from pointcept.models.builder import MODELS
from pointcept.utils.comm import get_world_size

from .utonia_v1m1_base import Utonia


class JEPAPredictor(nn.Module):
    """Lightweight pointwise predictor for Stage 1.

    Stage 1에서는 backbone이 이미 masked 입력을 통과시켜 *masked 위치의 feature*를 만들어
    두기 때문에, 별도의 cross-attention 없이 *그 feature를 target feature 공간으로 사상*하는
    얕은 MLP만으로도 의미 있는 jepa-style 신호를 만들 수 있다.

    Context-only forward + cross-attention predictor는 Stage 3에서 도입.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=3,
        dropout=0.0,
    ):
        super().__init__()
        assert num_layers >= 2
        layers = []
        c = in_channels
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(c, hidden_channels))
            layers.append(nn.GELU())
            layers.append(nn.LayerNorm(hidden_channels))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            c = hidden_channels
        layers.append(nn.Linear(c, out_channels))
        self.net = nn.Sequential(*layers)

    def forward(self, feat):
        return self.net(feat)


@MODELS.register_module("Utonia-v1m3")
class UtoniaJEPA(Utonia):
    """Utonia + Stage 1 JEPA.

    v1m1과 동일한 학습 흐름을 유지하되, mask 손실 블록 직후에 jepa 손실을 추가한다.
    """

    def __init__(
        self,
        *args,
        # JEPA-specific
        jepa_loss_weight=2 / 10,
        jepa_predictor_hidden_channels=2048,
        jepa_predictor_num_layers=3,
        jepa_predictor_dropout=0.0,
        jepa_use_cosine=False,
        jepa_normalize=True,
        # Re-balanced 4-loss weights (sum + jepa = 1)
        mask_loss_weight=1 / 10,
        roll_mask_loss_weight=1 / 10,
        unmask_loss_weight=2 / 10,
        enc2d_loss_weight=4 / 10,
        **kwargs,
    ):
        super().__init__(
            *args,
            mask_loss_weight=mask_loss_weight,
            roll_mask_loss_weight=roll_mask_loss_weight,
            unmask_loss_weight=unmask_loss_weight,
            enc2d_loss_weight=enc2d_loss_weight,
            **kwargs,
        )

        self.jepa_loss_weight = jepa_loss_weight
        self.jepa_use_cosine = jepa_use_cosine
        self.jepa_normalize = jepa_normalize

        # up_cast 후 concat된 backbone 출력 차원 = OnlineCluster.mlp 첫 Linear의 in_features
        for head_name in ("mask_head", "unmask_head"):
            if head_name in self.student:
                head_in = self.student[head_name].mlp[0].in_features
                break
        else:
            raise RuntimeError("Utonia-v1m3 requires at least one of {mask_head, unmask_head} on student.")

        self.predictor = JEPAPredictor(
            in_channels=head_in,
            hidden_channels=jepa_predictor_hidden_channels,
            out_channels=head_in,
            num_layers=jepa_predictor_num_layers,
            dropout=jepa_predictor_dropout,
        )

    def forward(self, data_dict, return_point=False):
        if return_point:
            return super().forward(data_dict, return_point=True)

        # ===== Prepare points & masks (v1m1과 동일) =====
        with torch.no_grad():
            global_point = Point(
                feat=data_dict["global_feat"],
                coord=data_dict["global_coord"],
                origin_coord=data_dict["global_origin_coord"],
                offset=data_dict["global_offset"],
                grid_size=data_dict["grid_size"][0],
            )

            global_mask, global_cluster = self.generate_mask(
                global_point.coord, global_point.offset, global_point.grid_size
            )
            mask_global_coord = global_point.coord.clone().detach()
            if self.mask_jitter is not None:
                mask_global_coord[global_mask] += torch.clip(
                    torch.randn_like(mask_global_coord[global_mask]).mul(
                        self.mask_jitter * data_dict["grid_size"][0]
                    ),
                    max=(self.mask_jitter * data_dict["grid_size"][0]) * 2,
                )

            mask_global_point = Point(
                feat=data_dict["global_feat"],
                coord=mask_global_coord,
                origin_coord=data_dict["global_origin_coord"],
                mask=global_mask,
                offset=data_dict["global_offset"],
                grid_size=data_dict["grid_size"][0],
            )
            major_view_correspondence = data_dict["global_correspondence"]

            local_point = Point(
                feat=data_dict["local_feat"],
                coord=data_dict["local_coord"],
                origin_coord=data_dict["local_origin_coord"],
                offset=data_dict["local_offset"],
                grid_size=data_dict["grid_size"][0],
            )

            result_dict = dict(loss=[])

            # teacher forward
            global_point_ = self.teacher.backbone(global_point)
            global_point_ = self.up_cast(global_point_)

            # ★ JEPA target은 head 적용 *전* raw feature.
            # v1m1은 아래에서 global_point_.feat을 head 출력으로 덮어쓰므로 여기서 보관.
            if self.jepa_loss_weight > 0:
                teacher_raw_feat = global_point_.feat.detach().clone()

            # teacher head forward (v1m1과 동일)
            if self.mask_loss_weight > 0 or self.roll_mask_loss_weight > 0:
                global_point_.feat = self.teacher.mask_head(global_point_.feat)
            else:
                global_point_.feat = self.teacher.unmask_head(global_point_.feat)

        # ===== mask / roll_mask loss (+ jepa loss) =====
        # student backbone forward는 mask/roll/jepa 어느 하나라도 켜져 있으면 필요
        need_student_masked_forward = (
            self.mask_loss_weight > 0
            or self.roll_mask_loss_weight > 0
            or self.jepa_loss_weight > 0
        )

        if need_student_masked_forward:
            mask_global_point_ = self.student.backbone(mask_global_point)
            mask_global_point_ = self.up_cast(mask_global_point_)

            # student raw feat (jepa용) — head 적용 전에 보관
            if self.jepa_loss_weight > 0:
                student_raw_feat = mask_global_point_.feat

        if self.mask_loss_weight > 0 or self.roll_mask_loss_weight > 0:
            mask_pred_sim = self.student.mask_head(mask_global_point_.feat)

            if self.mask_loss_weight > 0:
                with torch.no_grad():
                    match_index = self.match_neighbour(
                        mask_global_point_.origin_coord,
                        mask_global_point_.offset,
                        global_point_.origin_coord,
                        global_point_.offset,
                    )
                    mask_target_sim = self.sinkhorn_knopp(
                        global_point_.feat[match_index[:, 1]],
                        self.teacher_temp,
                    )

                mask_loss = -torch.sum(
                    mask_target_sim
                    * F.log_softmax(
                        mask_pred_sim[match_index[:, 0]] / self.student_temp, dim=-1
                    ),
                    dim=-1,
                )
                mask_loss = torch_scatter.segment_coo(
                    mask_loss,
                    index=mask_global_point_.batch[match_index[:, 0]],
                    reduce="mean",
                ).mean()
                result_dict["mask_loss"] = mask_loss
                result_dict["loss"].append(mask_loss * self.mask_loss_weight)

            if self.roll_mask_loss_weight > 0:
                roll_global_point_ = self.roll_point(global_point_)
                with torch.no_grad():
                    match_index = self.match_neighbour(
                        mask_global_point_.origin_coord,
                        mask_global_point_.offset,
                        roll_global_point_.origin_coord,
                        roll_global_point_.offset,
                    )
                    roll_mask_target_sim = self.sinkhorn_knopp(
                        roll_global_point_.feat[match_index[:, 1]],
                        self.teacher_temp,
                    )
                roll_mask_loss = -torch.sum(
                    roll_mask_target_sim
                    * F.log_softmax(
                        mask_pred_sim[match_index[:, 0]] / self.student_temp, dim=-1
                    ),
                    dim=-1,
                )
                roll_mask_loss = torch_scatter.segment_coo(
                    roll_mask_loss,
                    index=mask_global_point_.batch[match_index[:, 0]],
                    reduce="mean",
                ).mean()
                result_dict["roll_mask_loss"] = roll_mask_loss
                result_dict["loss"].append(roll_mask_loss * self.roll_mask_loss_weight)

        # ===== JEPA loss (★ 신규) =====
        # Stage 1: student(masked input) ↔ teacher(unmasked input) 매칭된 모든 페어에
        # 연속 latent regression. Stage 3에서 mask 위치 한정 + context-only forward로 강화.
        if self.jepa_loss_weight > 0:
            with torch.no_grad():
                jepa_match = self.match_neighbour(
                    mask_global_point_.origin_coord,
                    mask_global_point_.offset,
                    global_point_.origin_coord,
                    global_point_.offset,
                )
                student_idx = jepa_match[:, 0]
                teacher_idx = jepa_match[:, 1]

            z_student = self.predictor(student_raw_feat[student_idx])
            z_target = teacher_raw_feat[teacher_idx]

            if self.jepa_normalize:
                z_student = F.normalize(z_student, dim=-1, eps=1e-6)
                z_target = F.normalize(z_target, dim=-1, eps=1e-6)

            if self.jepa_use_cosine:
                jepa_loss = (1 - (z_student * z_target).sum(dim=-1))
            else:
                jepa_loss = F.smooth_l1_loss(z_student, z_target, reduction="none").sum(dim=-1)

            jepa_loss = torch_scatter.segment_coo(
                jepa_loss,
                index=mask_global_point_.batch[student_idx],
                reduce="mean",
            ).mean()
            result_dict["jepa_loss"] = jepa_loss
            result_dict["loss"].append(jepa_loss * self.jepa_loss_weight)

        # ===== unmask loss (v1m1과 동일) =====
        if self.unmask_loss_weight > 0:
            local_point_ = self.student.backbone(local_point)
            local_point_ = self.up_cast(local_point_)
            unmask_pred_sim = self.student.unmask_head(local_point_.feat)
            with torch.no_grad():
                principal_view_mask = global_point_.batch % self.num_global_view == 0
                principal_view_batch = (
                    global_point_.batch[principal_view_mask] // self.num_global_view
                )
                match_index = self.match_neighbour(
                    local_point_.origin_coord,
                    local_point_.offset[self.num_local_view - 1 :: self.num_local_view],
                    global_point_.origin_coord[principal_view_mask],
                    self._batch2offset(principal_view_batch),
                )
                unmask_target_sim = self.sinkhorn_knopp(
                    global_point_.feat[principal_view_mask][match_index[:, 1]],
                    self.teacher_temp,
                )
            unmask_loss = -torch.sum(
                unmask_target_sim
                * F.log_softmax(
                    unmask_pred_sim[match_index[:, 0]] / self.student_temp, dim=-1
                ),
                dim=-1,
            )
            unmask_loss = torch_scatter.segment_coo(
                unmask_loss,
                index=local_point_.batch[match_index[:, 0]],
                reduce="mean",
            ).mean()
            result_dict["unmask_loss"] = unmask_loss
            result_dict["loss"].append(unmask_loss * self.unmask_loss_weight)

        # ===== enc2d loss (v1m1과 동일) =====
        if self.enc2d_loss_weight > 0:
            self._compute_enc2d_loss(
                data_dict,
                mask_global_point,
                mask_global_point_ if need_student_masked_forward else None,
                global_mask,
                major_view_correspondence,
                result_dict,
            )

        result_dict["loss"] = sum(result_dict["loss"])

        if get_world_size() > 1:
            for loss_id, loss in result_dict.items():
                dist.all_reduce(loss, op=dist.ReduceOp.AVG)

        return result_dict

    # --- helpers ---
    @staticmethod
    def _batch2offset(batch):
        from pointcept.models.utils import batch2offset
        return batch2offset(batch)

    def _compute_enc2d_loss(
        self,
        data_dict,
        mask_global_point,
        mask_global_point_,
        global_mask,
        major_view_correspondence,
        result_dict,
    ):
        """v1m1 forward의 enc2d 블록을 추출한 헬퍼.

        student backbone forward를 이미 했으면 mask_global_point_를 그대로 사용,
        없으면 새로 forward (v1m1과 동일한 케이스 분기 유지).
        """
        from pointcept.models.utils import offset2batch, bincount2offset

        if mask_global_point_ is None:
            mask_global_point_ = self.student.backbone(mask_global_point)
            mask_global_point_ = self.up_cast(mask_global_point_)

        mask_global_point_enc2d = self.up_cast(
            mask_global_point_,
            upcast_level=self.enc2d_upcast_level - self.up_cast_level,
        )
        to_feature = self.pool_corr(mask_global_point_enc2d, major_view_correspondence)
        data_dict_global_offset = torch.cat(
            [torch.tensor([0]).cuda(), to_feature["offset"]], dim=0
        )
        enc2d_count = (
            data_dict_global_offset[
                1 : len(data_dict_global_offset) : self.num_global_view
            ]
            - data_dict_global_offset[
                0 : len(data_dict_global_offset) - 1 : self.num_global_view
            ]
        )
        enc2d_offset = torch.cat(
            [torch.tensor([0]).cuda(), torch.cumsum(enc2d_count, dim=0)]
        )
        enc2d_mask = torch.cat(
            [
                torch.arange(0, c, device=enc2d_count.device)
                + data_dict_global_offset[i * self.num_global_view]
                for i, c in enumerate(enc2d_count)
            ],
            dim=0,
        )

        offset_points_3d = enc2d_offset[1:]
        batch_points_3d = offset2batch(offset_points_3d)
        imgs = data_dict["images"]
        feature3d = to_feature["feat"][enc2d_mask]
        correspondence = to_feature["correspondence"][enc2d_mask]
        v0 = correspondence.shape[1]
        mask = torch.any(correspondence != torch.tensor([-1, -1]).cuda(), dim=2)
        valid_index = torch.where(mask)

        bincount_img_num = data_dict["img_num"]
        offset_img_num = bincount2offset(bincount_img_num)
        total_img_num = offset_img_num[-1]

        if total_img_num > 0:
            with torch.no_grad():
                feature2d = self.ENC2D_forward(imgs)
                feature2d = feature2d.contiguous().view(-1, feature2d.shape[-1])
                feature2d_mask = feature2d

            offset_img_num = torch.cat([torch.tensor([0]).cuda(), offset_img_num])[:-1]
            batch_index = batch_points_3d[valid_index[0]]
            batch_img_num = offset_img_num[batch_index]

            feature3d_mask = feature3d[valid_index[0]]

            feature_index = torch.cat(
                [
                    batch_img_num.unsqueeze(-1),
                    valid_index[1].unsqueeze(-1),
                    correspondence[valid_index],
                ],
                dim=-1,
            )
            feature_index = feature_index.long()
            feature_index = (
                feature_index[:, 0] * self.patch_h * self.patch_w
                + feature_index[:, 1] * self.patch_h * self.patch_w
                + feature_index[:, 2] * self.patch_w
                + feature_index[:, 3]
            )
            feature_index = feature_index.long()
            feature3d_mask = torch_scatter.scatter_mean(
                feature3d_mask, feature_index, dim=0, dim_size=feature2d.shape[0]
            )
            feature3d_mask = self.patch_proj(feature3d_mask)
            feature_index = torch.unique(feature_index)
            feature2d_mask = feature2d_mask[feature_index]
            feature3d_mask = feature3d_mask[feature_index]

            if self.enc2d_cos_shift:
                feature2d_mask = feature2d_mask - feature2d_mask.mean(dim=-1, keepdim=True)
                feature3d_mask = feature3d_mask - feature3d_mask.mean(dim=-1, keepdim=True)
            cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            loss = (1 - cos(feature2d_mask, feature3d_mask)).mean() * 10

            result_dict["enc2d_loss"] = loss
            result_dict["loss"].append(loss * self.enc2d_loss_weight)
        elif (
            self.mask_loss_weight
            + self.unmask_loss_weight
            + self.roll_mask_loss_weight
            > 0
        ):
            result_ssl_loss = sum(result_dict["loss"]) / (
                self.mask_loss_weight
                + self.unmask_loss_weight
                + self.roll_mask_loss_weight
            )
            result_dict["enc2d_loss"] = result_ssl_loss
            result_dict["loss"].append(result_ssl_loss * self.enc2d_loss_weight)
