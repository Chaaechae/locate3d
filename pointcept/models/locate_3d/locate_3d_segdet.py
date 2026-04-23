"""
Locate3DSegDetector: an alternative to the Locate-3D set-prediction decoder
that reframes referring-expression localization as per-entity segmentation.

Motivation
----------
Utonia's pretext is DINO-style masked contrastive learning on dense per-point
features -- the encoder is strongest at distinguishing points by their local
geometric/semantic content, not at answering "which query owns which box?"
This module plays to that strength:

- For each GT entity ``g`` mentioned in the caption we pool the CLIP text
  embeddings over the entity's positive-map tokens into an "entity text"
  vector.
- The entity text vector is dotted against projected per-point features to
  get a per-point logit "does this point belong to entity ``g``?"
- Training loss is per-point binary cross-entropy (+ dice) against the
  inside-axis-aligned-box indicator derived from GT boxes. This is dense,
  per-point supervision -- the same signal Utonia was pretrained on --
  rather than DETR's sparse set-prediction signal.
- At inference, per-entity masks are thresholded and the AABB of the mask
  points is returned as the predicted box.

The matcher/Hungarian/aux-loss machinery is unnecessary here: there is no
one-to-many set prediction -- every caption entity has exactly one
prediction channel (the entity text vector is the "query"). Multi-entity
coverage is structural. Query collapse is structurally impossible.

This is NOT architecturally identical to Locate-3D. It is an alternate
recipe that trades box-regression expressiveness for simpler supervision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from pointcept.models.builder import MODELS, build_model
from pointcept.models.utils.structure import Point
from pointcept.models.utils import offset2bincount

from .bbox_utils import box_iou_3d


def _split_by_offset(tensor, offset):
    bins = offset2bincount(offset).tolist()
    return list(torch.split(tensor, bins, dim=0))


def _points_in_boxes(coord, boxes):
    """(N, 3) × (G, 6 xyzxyz) → (G, N) bool inside-box mask."""
    if boxes.numel() == 0:
        return torch.zeros((0, coord.shape[0]), dtype=torch.bool, device=coord.device)
    mins = boxes[:, :3].unsqueeze(1)       # (G, 1, 3)
    maxs = boxes[:, 3:].unsqueeze(1)       # (G, 1, 3)
    p = coord.unsqueeze(0)                  # (1, N, 3)
    return ((p >= mins) & (p <= maxs)).all(dim=-1)


def _dice_loss(pred, target, eps=1e-6):
    # pred, target: (E, N) sigmoid outputs
    pred = pred.sigmoid()
    num = 2 * (pred * target).sum(-1)
    den = pred.sum(-1) + target.sum(-1) + eps
    return (1 - num / den).mean()


@MODELS.register_module("Locate3DSegDetector")
class Locate3DSegDetector(nn.Module):
    """Caption-conditioned per-point segmenter.

    Output per forward:
      - ``point_feats`` (projected)  (used by training loss)
      - per-entity ``pred_logits``   (B-list of (G_b, N_b) tensors)
      - per-entity ``pred_boxes``    (B-list of (G_b, 6) tensors, inference)
      - scalar train losses           (BCE + dice + optional class)
    """

    def __init__(
        self,
        backbone,
        backbone_out_channels: int,
        d_model: int = 512,
        freeze_backbone: bool = False,
        freeze_text_encoder: bool = True,
        text_encoder: str = "clip",
        loss_weight_bce: float = 1.0,
        loss_weight_dice: float = 1.0,
        # pos_weight on the BCE positive class. A typical ARKitScenes scene has
        # ~40k points but only ~40-100 points inside any given GT box (~0.1%).
        # With pos_weight=None, the BCE is dominated by the 99.9% negatives and
        # converges to the trivial all-negative minimum (observed: loss_bce
        # collapses to ~0.04 while loss_dice barely moves). Setting
        # pos_weight >> 1 rebalances the gradient toward positives. Default
        # 100 roughly mirrors the neg/pos ratio.
        bce_pos_weight: float = 100.0,
        max_points_train: int = 40000,
        max_points_eval: int = 40000,
        infer_threshold: float = 0.5,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.d_model = d_model
        # Point feature projector: pretrained encoder dim → shared d_model
        self.point_proj = nn.Sequential(
            nn.Linear(backbone_out_channels, d_model),
            nn.LayerNorm(d_model, eps=1e-12),
        )

        # CLIP text encoder (frozen by default; same convention as Locate-3D)
        from transformers import AutoTokenizer, CLIPTextModelWithProjection

        assert text_encoder in ["clip", "clip-large"], "Only CLIP models supported"
        clip_name = "openai/clip-vit-large-patch14"
        self.tokenizer = AutoTokenizer.from_pretrained(clip_name)
        self.text_encoder = CLIPTextModelWithProjection.from_pretrained(clip_name)
        self.text_hidden = self.text_encoder.config.hidden_size
        self.max_tokens = 77
        self.freeze_text_encoder = freeze_text_encoder
        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        self.text_proj = nn.Sequential(
            nn.Linear(self.text_hidden, d_model),
            nn.LayerNorm(d_model, eps=1e-12),
        )

        self.loss_weight_bce = loss_weight_bce
        self.loss_weight_dice = loss_weight_dice
        self.bce_pos_weight = float(bce_pos_weight) if bce_pos_weight else None
        self.max_points_train = max_points_train
        self.max_points_eval = max_points_eval
        self.infer_threshold = infer_threshold

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_text_encoder:
            self.text_encoder.eval()
        return self

    # ---- backbone ----
    def _run_backbone(self, input_dict):
        point = Point(input_dict)
        if self.freeze_backbone:
            with torch.no_grad():
                point = self.backbone(point)
        else:
            point = self.backbone(point)
        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                point.pop("pooling_parent")
                point.pop("pooling_inverse")
                if "idx_ptr" in point.keys():
                    point.pop("idx_ptr")
        return point

    # ---- text ----
    def _encode_text(self, captions, device):
        tok = self.tokenizer.batch_encode_plus(
            list(captions), padding="max_length", return_tensors="pt",
            max_length=self.max_tokens, truncation=True,
        ).to(device)
        if self.freeze_text_encoder:
            with torch.no_grad():
                out = self.text_encoder(**tok)
        else:
            out = self.text_encoder(**tok)
        text_feats = self.text_proj(out.last_hidden_state)         # (B, T, d_model)
        return text_feats, tok

    # ---- entity-text pooling ----
    @staticmethod
    def _pool_entity_text(text_feats, positive_map):
        """text_feats: (T, d_model). positive_map: (G, T) float {0,1}.
        Returns (G, d_model) entity text vectors (mean-pool over positive tokens).
        """
        denom = positive_map.sum(-1, keepdim=True).clamp_min(1.0)   # (G, 1)
        return (positive_map @ text_feats) / denom                   # (G, d_model)

    # ---- forward ----
    def forward(self, input_dict):
        point = self._run_backbone(input_dict)
        feats_full = self.point_proj(point.feat)                     # (ΣN, d_model)
        coords_full = (
            point.origin_coord
            if "origin_coord" in point.keys()
            else point.coord
        )
        offset = point.offset
        feats_list = _split_by_offset(feats_full, offset)
        coords_list = _split_by_offset(coords_full, offset)

        # Subsample for tractable BCE at train time.
        max_pts = self.max_points_train if self.training else self.max_points_eval
        sub_feats, sub_coords, sub_idx = [], [], []
        for f, c in zip(feats_list, coords_list):
            if max_pts is not None and f.shape[0] > max_pts:
                idx = torch.randperm(f.shape[0], device=f.device)[:max_pts]
                sub_feats.append(f[idx])
                sub_coords.append(c[idx])
                sub_idx.append(idx)
            else:
                sub_feats.append(f)
                sub_coords.append(c)
                sub_idx.append(torch.arange(f.shape[0], device=f.device))

        B = len(sub_feats)
        captions = input_dict["caption"]
        device = feats_full.device
        text_feats_b, _ = self._encode_text(captions, device)        # (B, T, d_model)

        positive_maps = input_dict.get("positive_map", None)         # list of (G, T)
        gt_boxes = input_dict.get("boxes_xyzxyz", None)              # list of (G, 6)
        # Real per-point masks from ScanNet / ScanNet++ (derived from the
        # scene's ``instance`` labels aligned to the sub-sampled point
        # cloud). For ARKit, ``gt_masks_list`` will be None and we fall
        # back to the inside-GT-box indicator.
        gt_masks_list = input_dict.get("point_masks", None)          # list of (G, N_full)

        pred_logits_list = []
        pred_boxes_list = []
        losses = {"loss_bce": 0.0, "loss_dice": 0.0}
        loss_n = 0

        for b in range(B):
            f_b = sub_feats[b]                                       # (N_b, d_model)
            c_b = sub_coords[b]                                      # (N_b, 3)
            if positive_maps is None or positive_maps[b] is None:
                pred_logits_list.append(None)
                pred_boxes_list.append(None)
                continue
            pos_map = positive_maps[b].to(device=device, dtype=f_b.dtype)  # (G, T)
            G = int(pos_map.shape[0])
            if G == 0:
                pred_logits_list.append(torch.zeros(0, f_b.shape[0], device=device))
                pred_boxes_list.append(torch.zeros(0, 6, device=device))
                continue
            entity_text = self._pool_entity_text(text_feats_b[b], pos_map)   # (G, d)
            # Cosine-scaled dot product. Scale by 1/sqrt(d) so logit magnitude
            # is stable across d_model values.
            scale = 1.0 / (f_b.shape[-1] ** 0.5)
            logits = (entity_text @ f_b.t()) * scale                 # (G, N_b)
            pred_logits_list.append(logits)

            # Training targets: (G, N_b) mask.
            if self.training and (gt_masks_list is not None or gt_boxes is not None):
                if gt_masks_list is not None and b < len(gt_masks_list) and gt_masks_list[b] is not None:
                    # Real instance mask from the dataset. sub_idx[b] maps
                    # the dataset's mask to the backbone-level subsample.
                    full_mask = gt_masks_list[b].to(device=device).bool()  # (G, N_full)
                    if full_mask.shape[1] != c_b.shape[0]:
                        # mask was built at dataset resolution; subset to
                        # current sampled indices.
                        idx_b = sub_idx[b]
                        if idx_b.numel() == full_mask.shape[1]:
                            target_mask = full_mask.to(logits.dtype)
                        else:
                            # Clamp index against the mask's point axis in
                            # case of off-by-one mismatches.
                            idx_b = idx_b[idx_b < full_mask.shape[1]]
                            target_mask = full_mask[:, idx_b].to(logits.dtype)
                            # If we had to drop points, keep logits in sync
                            if target_mask.shape[1] != logits.shape[1]:
                                logits = logits[:, : target_mask.shape[1]]
                    else:
                        target_mask = full_mask.to(logits.dtype)
                else:
                    # Fallback: inside-box proxy mask (ARKit: no real mask).
                    gbox = gt_boxes[b].to(device=device, dtype=c_b.dtype)
                    target_mask = _points_in_boxes(c_b, gbox).to(logits.dtype)
                # per-entity BCE. pos_weight rebalances the ~99.9% negative
                # / ~0.1% positive ratio; without it BCE converges to the
                # trivial all-negative minimum (predict 0 everywhere) and
                # stops learning the positive class.
                if self.bce_pos_weight is not None:
                    pos_weight = logits.new_tensor(self.bce_pos_weight)
                else:
                    pos_weight = None
                bce = F.binary_cross_entropy_with_logits(
                    logits, target_mask, reduction="mean", pos_weight=pos_weight,
                )
                dice = _dice_loss(logits, target_mask)
                losses["loss_bce"] = losses["loss_bce"] + bce
                losses["loss_dice"] = losses["loss_dice"] + dice
                loss_n += 1

            # Inference / viz: derive AABB from mask-probability threshold
            if not self.training:
                probs = logits.sigmoid()                             # (G, N_b)
                boxes_pred = torch.zeros(G, 6, device=device, dtype=c_b.dtype)
                for g in range(G):
                    m = probs[g] > self.infer_threshold
                    if m.sum() < 3:
                        # fallback: top-5% points to avoid degenerate empty box
                        k = max(3, int(0.05 * c_b.shape[0]))
                        _, idx_top = torch.topk(probs[g], k=min(k, c_b.shape[0]))
                        pts = c_b[idx_top]
                    else:
                        pts = c_b[m]
                    boxes_pred[g, :3] = pts.min(dim=0).values
                    boxes_pred[g, 3:] = pts.max(dim=0).values
                pred_boxes_list.append(boxes_pred)
            else:
                pred_boxes_list.append(None)

        result = {}
        if self.training and loss_n > 0:
            bce_scalar = losses["loss_bce"] / loss_n
            dice_scalar = losses["loss_dice"] / loss_n
            result["loss_bce"] = bce_scalar
            result["loss_dice"] = dice_scalar
            result["loss"] = (
                self.loss_weight_bce * bce_scalar
                + self.loss_weight_dice * dice_scalar
            )
            # minimal debug metric: mean IoU between predicted mask-derived box
            # and GT box, per-entity, micro-averaged.
            with torch.no_grad():
                ious = []
                for b in range(B):
                    pl = pred_logits_list[b]
                    if pl is None or pl.shape[0] == 0:
                        continue
                    gbox = gt_boxes[b].to(device=device, dtype=sub_coords[b].dtype)
                    probs = pl.sigmoid()
                    G = pl.shape[0]
                    pboxes = torch.zeros(G, 6, device=device, dtype=gbox.dtype)
                    for g in range(G):
                        m = probs[g] > self.infer_threshold
                        if m.sum() < 3:
                            k = max(3, int(0.05 * sub_coords[b].shape[0]))
                            _, idx_top = torch.topk(
                                probs[g], k=min(k, sub_coords[b].shape[0])
                            )
                            pts = sub_coords[b][idx_top]
                        else:
                            pts = sub_coords[b][m]
                        pboxes[g, :3] = pts.min(dim=0).values
                        pboxes[g, 3:] = pts.max(dim=0).values
                    iou = torch.diagonal(box_iou_3d(pboxes, gbox)[0])
                    ious.append(iou)
                if len(ious) > 0:
                    result["dbg_mask_iou"] = torch.cat(ious).mean().detach().float()
                    result["dbg_mask_covered25"] = (
                        (torch.cat(ious) > 0.25).float().mean().detach()
                    )
                else:
                    result["dbg_mask_iou"] = torch.zeros((), device=device)
                    result["dbg_mask_covered25"] = torch.zeros((), device=device)
        else:
            # eval mode: populate boxes/logits for downstream evaluator
            # (not yet integrated; grounding evaluator expects Locate3D schema)
            result["pred_boxes_per_entity"] = pred_boxes_list
            result["pred_logits_per_entity"] = pred_logits_list

        return result
