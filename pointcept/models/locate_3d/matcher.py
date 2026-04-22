"""
Hungarian matcher for the Locate3D decoder.

Follows the BUTD-DETR / MDETR style matcher: matches each predicted query to at
most one ground-truth target using the weighted sum of text-alignment cost,
bounding-box L1 cost and GIoU cost.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .bbox_utils import generalized_box_iou_3d


class HungarianMatcher(nn.Module):
    def __init__(
        self,
        cost_class: float = 1.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        outputs:
            pred_logits: (B, Q, T)  text-token alignment logits
            pred_boxes: (B, Q, 6)   xyzxyz
        targets: list of dict with keys
            positive_map: (G, T) float, 1 on positive tokens
            boxes_xyzxyz: (G, 6)
        Returns list of (pred_idx, gt_idx) per sample.
        """
        B, Q, T = outputs["pred_logits"].shape
        # cast to float32 so cost computation (cdist / giou) is numerically
        # stable and compatible with float32 ground-truth tensors under AMP.
        out_prob = outputs["pred_logits"].float().sigmoid()  # (B, Q, T)
        out_bbox = outputs["pred_boxes"].float()              # (B, Q, 6)

        indices = []
        for b in range(B):
            tgt = targets[b]
            pos_map = tgt["positive_map"].to(device=out_prob.device, dtype=out_prob.dtype)
            tgt_bbox = tgt["boxes_xyzxyz"].to(device=out_bbox.device, dtype=out_bbox.dtype)
            G = pos_map.shape[0]
            if G == 0:
                indices.append(
                    (
                        torch.as_tensor([], dtype=torch.long),
                        torch.as_tensor([], dtype=torch.long),
                    )
                )
                continue

            # Classification cost (soft token).
            #   pos_tokens contribute  -log(p) cost
            #   neg_tokens contribute  -log(1-p) cost
            prob = out_prob[b]  # (Q, T)
            token_cnt = pos_map.sum(-1).clamp_min(1.0)  # (G,)
            pos_cost = -(prob.clamp_min(1e-8).log())  # (Q, T)
            neg_cost = -((1 - prob).clamp_min(1e-8).log())  # (Q, T)
            # cost[q,g] = mean over positive tokens of pos_cost[q,t]
            #          + mean over (all) tokens of neg_cost[q,t] w.r.t. 1-pos_map[g,t]
            cost_class = (pos_cost @ pos_map.t()) / token_cnt.unsqueeze(0)
            # Simple class cost: focus on positive tokens only (MDETR-style soft token).

            # Bbox L1 cost on center+size representation for scale-invariance.
            pred = out_bbox[b]
            cost_bbox = torch.cdist(pred, tgt_bbox, p=1)

            # GIoU cost
            giou = generalized_box_iou_3d(pred, tgt_bbox)
            cost_giou = -giou

            C = (
                self.cost_class * cost_class
                + self.cost_bbox * cost_bbox
                + self.cost_giou * cost_giou
            )
            C = C.detach().cpu().numpy()

            row, col = linear_sum_assignment(C)
            indices.append(
                (torch.as_tensor(row, dtype=torch.long), torch.as_tensor(col, dtype=torch.long))
            )
        return indices
