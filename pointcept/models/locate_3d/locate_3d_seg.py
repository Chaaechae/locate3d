"""
Utonia-encoder + Locate3D-decoder downstream model for referring expression
localization (box + optional segmentation mask).

The encoder produces per-point features via a PT-v3 backbone pretrained under
the Utonia framework. Those features are padded across the batch and fed into
the Locate3D language-conditioned transformer decoder, which outputs
text-token alignment logits, per-point masks and 3D bounding boxes for each
object query. The set-prediction loss follows the MDETR / BUTD-DETR recipe
adapted to 3D as described in the Locate-3D paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from pointcept.models.builder import MODELS, build_model
from pointcept.models.utils.structure import Point
from pointcept.models.utils import offset2batch, offset2bincount

from .locate_3d_decoder import Locate3DDecoder
from .matcher import HungarianMatcher
from .criterion import Locate3DCriterion
from .bbox_utils import box_volume, box_iou_3d


def _pad_points(feats_list, coords_list):
    lengths = [f.shape[0] for f in feats_list]
    max_len = max(lengths)
    B = len(feats_list)
    F_dim = feats_list[0].shape[1]
    device = feats_list[0].device
    dtype = feats_list[0].dtype

    feats = torch.zeros(B, max_len, F_dim, device=device, dtype=dtype)
    coords = torch.zeros(B, max_len, 3, device=device, dtype=coords_list[0].dtype)
    mask = torch.ones(B, max_len, dtype=torch.bool, device=device)  # True = PAD
    for i, (f, c) in enumerate(zip(feats_list, coords_list)):
        n = f.shape[0]
        feats[i, :n] = f
        coords[i, :n] = c
        mask[i, :n] = False
    return feats, coords, mask, lengths


def _split_by_offset(tensor, offset):
    bins = offset2bincount(offset).tolist()
    return list(torch.split(tensor, bins, dim=0))


@MODELS.register_module("Locate3DLocalizer")
class Locate3DLocalizer(nn.Module):
    """Backbone (PT-v3 / Utonia encoder) + Locate3D decoder + losses."""

    def __init__(
        self,
        backbone,
        decoder,
        backbone_out_channels: int,
        decoder_input_feat_dim: int = 256,
        freeze_backbone: bool = False,
        matcher_cost_class: float = 1.0,
        matcher_cost_bbox: float = 5.0,
        matcher_cost_giou: float = 2.0,
        loss_weight_class: float = 1.0,
        loss_weight_bbox: float = 5.0,
        loss_weight_giou: float = 2.0,
        loss_weight_mask_bce: float = 0.0,
        loss_weight_mask_dice: float = 0.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        aux_loss: bool = True,
        max_points_train: int = 40000,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # project encoder feats to the decoder input dimension
        self.feat_proj = nn.Linear(backbone_out_channels, decoder_input_feat_dim)

        self.decoder = Locate3DDecoder(input_feat_dim=decoder_input_feat_dim, **decoder)

        self.matcher = HungarianMatcher(
            cost_class=matcher_cost_class,
            cost_bbox=matcher_cost_bbox,
            cost_giou=matcher_cost_giou,
        )
        self.criterion = Locate3DCriterion(
            matcher=self.matcher,
            weight_class=loss_weight_class,
            weight_bbox=loss_weight_bbox,
            weight_giou=loss_weight_giou,
            weight_mask_bce=loss_weight_mask_bce,
            weight_mask_dice=loss_weight_mask_dice,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            aux_loss=aux_loss,
        )
        self.max_points_train = max_points_train

    def _run_backbone(self, input_dict):
        point = Point(input_dict)
        if self.freeze_backbone:
            with torch.no_grad():
                point = self.backbone(point)
        else:
            point = self.backbone(point)
        # U-Net decoder recovery: collapse any residual pooling stack
        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
        return point

    def _encode(self, input_dict):
        point = self._run_backbone(input_dict)
        feats = self.feat_proj(point.feat)
        coords = point.origin_coord if "origin_coord" in point.keys() else point.coord
        offset = point.offset

        feats_list = _split_by_offset(feats, offset)
        coords_list = _split_by_offset(coords, offset)

        # For training, optionally subsample to bound memory / compute
        if self.training and self.max_points_train is not None:
            sub_feats, sub_coords, sub_idx = [], [], []
            for f, c in zip(feats_list, coords_list):
                if f.shape[0] > self.max_points_train:
                    idx = torch.randperm(f.shape[0], device=f.device)[: self.max_points_train]
                    sub_feats.append(f[idx])
                    sub_coords.append(c[idx])
                    sub_idx.append(idx)
                else:
                    sub_feats.append(f)
                    sub_coords.append(c)
                    sub_idx.append(torch.arange(f.shape[0], device=f.device))
            feats_list, coords_list = sub_feats, sub_coords
        else:
            sub_idx = [torch.arange(f.shape[0], device=f.device) for f in feats_list]

        feats, coords, pad_mask, lengths = _pad_points(feats_list, coords_list)
        return feats, coords, pad_mask, sub_idx, coords_list

    def _build_targets(self, input_dict, sub_idx, coords_list):
        """Construct per-sample dict {positive_map, boxes_xyzxyz[, masks]}."""
        targets = []
        positive_maps = input_dict["positive_map"]  # list[B] of (G, T) tensors
        gt_boxes = input_dict["boxes_xyzxyz"]        # list[B] of (G, 6) tensors
        gt_masks = input_dict.get("point_masks", None)  # list[B] of (G, N_full) or None

        for b in range(len(positive_maps)):
            t = {
                "positive_map": positive_maps[b],
                "boxes_xyzxyz": gt_boxes[b],
            }
            if gt_masks is not None and gt_masks[b] is not None:
                # subsample masks to the points we kept in the encoder view
                idx = sub_idx[b]
                t["masks"] = gt_masks[b][:, idx]
            targets.append(t)
        return targets

    def forward(self, input_dict):
        feats, coords, pad_mask, sub_idx, coords_list = self._encode(input_dict)
        captions = input_dict["caption"]  # list[str]

        out = self.decoder(feats, coords, pad_mask, captions)

        result = {"pred_logits": out["pred_logits"], "pred_boxes": out["pred_boxes"]}

        if "positive_map" in input_dict:
            targets = self._build_targets(input_dict, sub_idx, coords_list)
            losses = self.criterion(out, targets)
            match_indices = losses.pop("_match_indices", None)

            # Flatten: only scalar-tensor values go back to the trainer (the
            # InformationWriter calls `.item()` on every key).
            for k, v in losses.items():
                if isinstance(v, torch.Tensor) and v.ndim == 0:
                    result[k] = v

            # ------- debug diagnostics (query-collapse / matching quality) -------
            with torch.no_grad():
                diag = self._debug_metrics(out, targets, match_indices, input_dict)
            result.update(diag)

            # stash matching + raw preds for hooks (not consumed by trainer logs)
            self._last_match_indices = match_indices
            self._last_outputs = out
            self._last_targets = targets
        else:
            # inference only
            result["pred_masks"] = out["pred_masks"]
            result["pred_logits"] = out["pred_logits"]

        return result

    @staticmethod
    def _pairwise_cosine(x):
        x = F.normalize(x, dim=-1)
        return x @ x.t()

    def _debug_metrics(self, outputs, targets, indices, input_dict):
        """Scalar diagnostics per batch. All returned as 0-dim tensors so the
        Pointcept InformationWriter auto-logs them as train_batch/dbg_*."""
        device = outputs["pred_logits"].device
        pred_boxes = outputs["pred_boxes"]   # (B, Q, 6)
        pred_logits = outputs["pred_logits"] # (B, Q, T)
        B, Q, _ = pred_boxes.shape

        # Predicted-box diversity (per-scene, then averaged).
        centers = 0.5 * (pred_boxes[..., :3] + pred_boxes[..., 3:])  # (B, Q, 3)
        sizes = (pred_boxes[..., 3:] - pred_boxes[..., :3]).clamp_min(0)
        center_std = centers.std(dim=1).mean()     # avg across xyz & batch
        size_std = sizes.std(dim=1).mean()

        # Matched query usage + matching-quality.
        all_matched_q = []
        match_ious = []
        match_iou_primary = []
        gt_covered = 0
        num_gt_total = 0
        for b, (src, tgt) in enumerate(indices):
            num_gt_total += int(targets[b]["boxes_xyzxyz"].shape[0])
            if len(src) == 0:
                continue
            all_matched_q.append(src.to(device))
            pbox = pred_boxes[b, src]                               # (G, 6)
            gbox = targets[b]["boxes_xyzxyz"].to(device)[tgt]       # (G, 6)
            iou = torch.diagonal(box_iou_3d(pbox, gbox)[0])          # (G,)
            match_ious.append(iou)
            gt_covered += int((iou > 0.25).sum().item())
            primary = int(input_dict.get("primary_object_id", [0]*B)[b]
                          if not isinstance(input_dict.get("primary_object_id", [0]*B)[b], torch.Tensor)
                          else input_dict["primary_object_id"][b].flatten()[0].item())
            # If the primary gt was matched, record its IoU.
            tgt_cpu = tgt.cpu().tolist() if isinstance(tgt, torch.Tensor) else list(tgt)
            if primary in tgt_cpu:
                pos_in_matched = tgt_cpu.index(primary)
                match_iou_primary.append(iou[pos_in_matched])

        if len(all_matched_q) == 0:
            match_iou = torch.zeros((), device=device)
            query_entropy = torch.zeros((), device=device)
            primary_iou = torch.zeros((), device=device)
            unique_ratio = torch.zeros((), device=device)
        else:
            all_q = torch.cat(all_matched_q)            # (M,)
            hist = torch.bincount(all_q, minlength=Q).float()
            p = hist / hist.sum().clamp_min(1.0)
            # entropy in nats; max = log(M) if all different queries
            query_entropy = -(p[p > 0] * p[p > 0].log()).sum()
            unique_ratio = (hist > 0).sum().float() / max(all_q.numel(), 1)
            match_iou = torch.cat(match_ious).mean()
            primary_iou = (
                torch.stack(match_iou_primary).mean()
                if len(match_iou_primary) > 0
                else torch.zeros((), device=device)
            )

        coverage = torch.tensor(
            gt_covered / max(num_gt_total, 1), device=device, dtype=torch.float32
        )

        return {
            "dbg_match_iou": match_iou.detach().float(),
            "dbg_match_iou_primary": primary_iou.detach().float(),
            "dbg_gt_covered25": coverage.detach(),
            "dbg_center_std": center_std.detach().float(),
            "dbg_size_std": size_std.detach().float(),
            "dbg_query_entropy": query_entropy.detach().float(),
            "dbg_query_unique_ratio": unique_ratio.detach().float(),
        }
