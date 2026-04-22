"""
Loss for the Locate3D decoder.

Implements the set-prediction training loss used by MDETR / BUTD-DETR style
referring expression grounding models, extended to 3D boxes (L1 + GIoU) and,
optionally, point-level mask supervision (BCE + Dice) when gt masks are
available.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bbox_utils import generalized_box_iou_3d


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_boxes: float,
    alpha: float = 0.25,
    gamma: float = 2.0,
):
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean(1).sum() / max(num_boxes, 1)


def dice_loss(pred, target, num_boxes, eps: float = 1e-6):
    pred = pred.sigmoid()
    num = 2 * (pred * target).sum(-1)
    den = pred.sum(-1) + target.sum(-1) + eps
    return (1 - num / den).sum() / max(num_boxes, 1)


class Locate3DCriterion(nn.Module):
    def __init__(
        self,
        matcher,
        weight_class: float = 1.0,
        weight_bbox: float = 5.0,
        weight_giou: float = 2.0,
        weight_mask_bce: float = 1.0,
        weight_mask_dice: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        aux_loss: bool = True,
    ):
        super().__init__()
        self.matcher = matcher
        self.weight_class = weight_class
        self.weight_bbox = weight_bbox
        self.weight_giou = weight_giou
        self.weight_mask_bce = weight_mask_bce
        self.weight_mask_dice = weight_mask_dice
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.aux_loss = aux_loss

    def _single_layer_loss(self, outputs, targets, indices, num_boxes):
        losses = {}
        device = outputs["pred_logits"].device

        # Compute losses in float32 for numerical stability under AMP/bf16.
        pred_logits = outputs["pred_logits"].float()       # (B, Q, T)
        pred_boxes = outputs["pred_boxes"].float()         # (B, Q, 6)
        text_attn_mask = outputs["text_attn_mask"]         # (B, T) True = pad

        # --- text alignment ---
        tgt_map = torch.zeros_like(pred_logits)            # float32
        for b, (src, tgt) in enumerate(indices):
            if len(src) == 0:
                continue
            pos_map = targets[b]["positive_map"].to(
                device=device, dtype=tgt_map.dtype
            )
            tgt_map[b, src] = pos_map[tgt]

        valid = (~text_attn_mask).unsqueeze(1).expand_as(pred_logits)
        flat_logits = pred_logits.masked_select(valid)
        flat_tgt = tgt_map.masked_select(valid)
        losses["loss_class"] = sigmoid_focal_loss(
            flat_logits.unsqueeze(0),
            flat_tgt.unsqueeze(0),
            num_boxes,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
        )

        # --- bbox ---
        src_boxes = []
        tgt_boxes = []
        for b, (src, tgt) in enumerate(indices):
            if len(src) == 0:
                continue
            src_boxes.append(pred_boxes[b, src])
            tgt_boxes.append(
                targets[b]["boxes_xyzxyz"].to(device=device, dtype=pred_boxes.dtype)[tgt]
            )
        if len(src_boxes) > 0:
            src_boxes = torch.cat(src_boxes, dim=0)
            tgt_boxes = torch.cat(tgt_boxes, dim=0)
            loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction="none").sum() / max(num_boxes, 1)
            giou = torch.diag(generalized_box_iou_3d(src_boxes, tgt_boxes))
            loss_giou = (1 - giou).sum() / max(num_boxes, 1)
        else:
            loss_bbox = pred_boxes.sum() * 0.0
            loss_giou = pred_boxes.sum() * 0.0
        losses["loss_bbox"] = loss_bbox
        losses["loss_giou"] = loss_giou

        # --- masks (optional) ---
        if "pred_masks" in outputs and any("masks" in t for t in targets):
            pred_masks = outputs["pred_masks"].float()  # (B, Q, N)
            src_masks = []
            tgt_masks = []
            for b, (src, tgt) in enumerate(indices):
                if len(src) == 0 or "masks" not in targets[b]:
                    continue
                src_masks.append(pred_masks[b, src])
                tgt_masks.append(
                    targets[b]["masks"].to(device=device, dtype=pred_masks.dtype)[tgt]
                )
            if len(src_masks) > 0:
                src_masks = torch.cat(src_masks, dim=0)
                tgt_masks = torch.cat(tgt_masks, dim=0)
                losses["loss_mask_bce"] = F.binary_cross_entropy_with_logits(
                    src_masks, tgt_masks, reduction="none"
                ).mean(1).sum() / max(num_boxes, 1)
                losses["loss_mask_dice"] = dice_loss(src_masks, tgt_masks, num_boxes)

        return losses

    def forward(self, outputs, targets):
        device = outputs["pred_logits"].device
        num_boxes = sum(len(t["boxes_xyzxyz"]) for t in targets)
        num_boxes = max(num_boxes, 1)

        indices = self.matcher(outputs, targets)
        losses = self._single_layer_loss(outputs, targets, indices, num_boxes)
        # Expose the final-layer matching result for debug / visualization.
        # Stored as a non-scalar key that the model forward pops before returning.
        losses["_match_indices"] = indices

        if self.aux_loss and "aux_outputs" in outputs:
            for i, aux in enumerate(outputs["aux_outputs"]):
                aux["text_attn_mask"] = outputs["text_attn_mask"]
                aux_indices = self.matcher(aux, targets)
                aux_losses = self._single_layer_loss(aux, targets, aux_indices, num_boxes)
                for k, v in aux_losses.items():
                    losses[f"{k}_aux_{i}"] = v

        # total weighted loss
        total = torch.zeros((), device=device)
        for k, v in losses.items():
            if k.startswith("_"):
                continue
            base = k.split("_aux_")[0]
            if base == "loss_class":
                total = total + self.weight_class * v
            elif base == "loss_bbox":
                total = total + self.weight_bbox * v
            elif base == "loss_giou":
                total = total + self.weight_giou * v
            elif base == "loss_mask_bce":
                total = total + self.weight_mask_bce * v
            elif base == "loss_mask_dice":
                total = total + self.weight_mask_dice * v
        losses["loss"] = total
        return losses
