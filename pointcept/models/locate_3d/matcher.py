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

from .bbox_utils import generalized_box_iou_3d, box_xyzxyz_to_cxcyczwhd


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

            # Classification cost (MDETR soft-token).
            # ``-prob @ pos_map.T`` gives each (query, gt) pair the negative
            # probability mass the query puts on the gt's positive tokens.
            # Unlike ``-log(prob)`` this is bounded and varies meaningfully
            # across queries even at initialization (when sigmoid(logits) is
            # close to 0.5 for all queries), so the Hungarian solver is not
            # forced to rely on the bbox / GIoU costs alone.
            prob = out_prob[b]  # (Q, T)
            cost_class = -(prob @ pos_map.t())  # (Q, G)

            # Bbox L1 cost on center+size (cxcyczwhd) so translation and
            # extent errors are on the same scale regardless of absolute
            # world-coordinate magnitude.
            pred = out_bbox[b]
            pred_csz = box_xyzxyz_to_cxcyczwhd(pred)
            tgt_csz = box_xyzxyz_to_cxcyczwhd(tgt_bbox)
            cost_bbox = torch.cdist(pred_csz, tgt_csz, p=1)

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
