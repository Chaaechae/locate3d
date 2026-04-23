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
        aux_layer_weights=None,
        max_points_train: int = 40000,
        max_points_eval: int = 60000,
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
            aux_layer_weights=aux_layer_weights,
        )
        self.max_points_train = max_points_train
        self.max_points_eval = max_points_eval

    def _run_backbone(self, input_dict):
        point = Point(input_dict)
        if self.freeze_backbone:
            with torch.no_grad():
                point = self.backbone(point)
        else:
            point = self.backbone(point)
        # Discard the pooling chain recorded by GridPooling(traceable=True).
        # Two cases:
        #   - enc_mode=False: the U-Net ``dec`` stack already walked and consumed the
        #     chain via GridUnpooling, so this loop is a no-op.
        #   - enc_mode=True: ``dec`` was skipped, so the pooling_parent chain is still
        #     attached to the bottleneck point. We must NOT try to reconstruct a
        #     full-resolution feature by channel-concat -- that would produce
        #     ~1386-d features (sum of enc_channels) from random-init paths since
        #     no decoder weights were used, and break the
        #     ``feat_proj=Linear(backbone_out_channels, 256)`` head downstream.
        # In both cases we just drop references so Python can free the up-stream
        # per-level Point objects instead of keeping the full pyramid in memory.
        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                point.pop("pooling_parent")
                point.pop("pooling_inverse")
                if "idx_ptr" in point.keys():
                    point.pop("idx_ptr")
        return point

    def _encode(self, input_dict):
        point = self._run_backbone(input_dict)
        feats = self.feat_proj(point.feat)
        coords = point.origin_coord if "origin_coord" in point.keys() else point.coord
        offset = point.offset

        feats_list = _split_by_offset(feats, offset)
        coords_list = _split_by_offset(coords, offset)

        # Cap per-scene point count in both training and eval.  The decoder's
        # cross-attention is the memory bottleneck (8 layers with 256 queries
        # attending to every point), so passing the raw ~200k-point ARKitScenes
        # clouds straight through the decoder will OOM even on 80GB GPUs at
        # eval time. ``max_points_{train,eval}`` = None disables the cap.
        max_pts = self.max_points_train if self.training else self.max_points_eval
        if max_pts is not None:
            sub_feats, sub_coords, sub_idx = [], [], []
            for f, c in zip(feats_list, coords_list):
                if f.shape[0] > max_pts:
                    idx = torch.randperm(f.shape[0], device=f.device)[:max_pts]
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

        # During training, the Pointcept InformationWriter calls `.item()` on
        # every value in the returned dict, so only scalar tensors may be
        # returned. Non-scalar prediction tensors (pred_logits / pred_boxes /
        # pred_masks) are only needed by eval / viz hooks, so we include them
        # only when not training.
        result = {}
        if not self.training:
            result["pred_logits"] = out["pred_logits"]
            result["pred_boxes"] = out["pred_boxes"]
            result["pred_masks"] = out["pred_masks"]

        if "positive_map" in input_dict:
            targets = self._build_targets(input_dict, sub_idx, coords_list)
            losses = self.criterion(out, targets)
            match_indices = losses.pop("_match_indices", None)

            for k, v in losses.items():
                if isinstance(v, torch.Tensor) and v.ndim == 0:
                    result[k] = v

            # ------- debug diagnostics (query-collapse / matching quality) -------
            with torch.no_grad():
                diag = self._debug_metrics(out, targets, match_indices, input_dict)
            result.update(diag)

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

        # Text-alignment score per query (max over valid tokens of sigmoid logits).
        # Used for the top-K recall metric below.
        text_attn_mask = outputs.get("text_attn_mask", None)
        if text_attn_mask is not None:
            valid_tok = (~text_attn_mask).to(pred_logits.dtype)               # (B, T)
            neg_inf = pred_logits.new_tensor(float("-inf"))
            logits_masked = torch.where(
                valid_tok.unsqueeze(1).bool(), pred_logits, neg_inf
            )
            query_score = logits_masked.sigmoid().amax(dim=-1)                # (B, Q)
        else:
            query_score = pred_logits.sigmoid().amax(dim=-1)

        # Matched query usage + matching-quality.
        all_matched_q = []
        match_ious = []
        match_iou_primary = []
        gt_covered = 0
        num_gt_total = 0
        samples_all_covered = 0           # per-sample: every GT in matched pair > 0.25
        topk_recall_hits = 0              # per-sample: top-K by text score covers every GT
        num_nonempty_samples = 0
        for b, (src, tgt) in enumerate(indices):
            n_gt_b = int(targets[b]["boxes_xyzxyz"].shape[0])
            num_gt_total += n_gt_b
            if n_gt_b == 0:
                continue
            num_nonempty_samples += 1
            gbox_b = targets[b]["boxes_xyzxyz"].to(device)                    # (G, 6)

            # Match-based metrics (only meaningful when everyone was matched).
            if len(src) > 0:
                all_matched_q.append(src.to(device))
                pbox = pred_boxes[b, src]                                     # (G, 6)
                gbox_matched = gbox_b[tgt]
                iou = torch.diagonal(box_iou_3d(pbox, gbox_matched)[0])       # (G,)
                match_ious.append(iou)
                gt_covered += int((iou > 0.25).sum().item())
                if len(src) == n_gt_b and bool((iou > 0.25).all()):
                    samples_all_covered += 1

                primary = int(
                    input_dict.get("primary_object_id", [0] * B)[b]
                    if not isinstance(
                        input_dict.get("primary_object_id", [0] * B)[b], torch.Tensor
                    )
                    else input_dict["primary_object_id"][b].flatten()[0].item()
                )
                tgt_cpu = tgt.cpu().tolist() if isinstance(tgt, torch.Tensor) else list(tgt)
                if primary in tgt_cpu:
                    match_iou_primary.append(iou[tgt_cpu.index(primary)])

            # Top-K-by-text-score recall: does the top-K set of queries cover
            # every GT at IoU > 0.25? Measures "if the matcher were perfect,
            # would the scored queries still localize all entities?"
            topk = min(n_gt_b, pred_boxes.shape[1])
            if topk > 0:
                _, top_idx = torch.topk(query_score[b], k=topk)
                top_boxes = pred_boxes[b, top_idx]                            # (K, 6)
                pairwise = box_iou_3d(top_boxes, gbox_b)[0]                   # (K, G)
                if bool((pairwise.max(dim=0).values > 0.25).all()):
                    topk_recall_hits += 1

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
        all_cov = torch.tensor(
            samples_all_covered / max(num_nonempty_samples, 1),
            device=device, dtype=torch.float32,
        )
        topk_rec = torch.tensor(
            topk_recall_hits / max(num_nonempty_samples, 1),
            device=device, dtype=torch.float32,
        )

        return {
            "dbg_match_iou": match_iou.detach().float(),
            "dbg_match_iou_primary": primary_iou.detach().float(),
            "dbg_gt_covered25": coverage.detach(),
            # Sample-level strict coverage: every GT entity matched at IoU>0.25.
            "dbg_all_gt_covered25": all_cov.detach(),
            # Upper-bound recall: top-K-by-text-score queries cover every GT.
            # If this is high but dbg_match_iou is low, the matcher, not the
            # head, is the problem.
            "dbg_topk_recall25": topk_rec.detach(),
            "dbg_center_std": center_std.detach().float(),
            "dbg_size_std": size_std.detach().float(),
            "dbg_query_entropy": query_entropy.detach().float(),
            "dbg_query_unique_ratio": unique_ratio.detach().float(),
        }
