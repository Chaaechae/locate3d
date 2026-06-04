"""
Bounding box utilities for the Locate3D localization head.

Port of facebookresearch/locate-3d/models/model_utils/bbox_utils.py with the
additional helpers (center/size <-> min/max) that are required for training.
"""

import torch


@torch.jit.script
def box_cxcyczwhd_to_xyzxyz_jit(x):
    centers = x[..., :3]
    dims = torch.clamp(x[..., 3:], min=1e-6)
    return torch.cat([centers - 0.5 * dims, centers + 0.5 * dims], dim=-1)


def box_xyzxyz_to_cxcyczwhd(x):
    mins = x[..., :3]
    maxs = x[..., 3:]
    centers = 0.5 * (mins + maxs)
    dims = (maxs - mins).clamp_min(1e-6)
    return torch.cat([centers, dims], dim=-1)


def box_volume(box_xyzxyz):
    d = (box_xyzxyz[..., 3:] - box_xyzxyz[..., :3]).clamp_min(0.0)
    return d[..., 0] * d[..., 1] * d[..., 2]


def box_iou_3d(a, b):
    """3D IoU between N×6 and M×6 xyzxyz boxes. Returns (N, M) iou and (N, M) union."""
    vol_a = box_volume(a)
    vol_b = box_volume(b)

    lt = torch.max(a[:, None, :3], b[None, :, :3])
    rb = torch.min(a[:, None, 3:], b[None, :, 3:])
    wh = (rb - lt).clamp_min(0.0)
    inter = wh[..., 0] * wh[..., 1] * wh[..., 2]

    union = vol_a[:, None] + vol_b[None, :] - inter
    iou = inter / union.clamp_min(1e-6)
    return iou, union


def generalized_box_iou_3d(a, b):
    """Generalized 3D IoU."""
    iou, union = box_iou_3d(a, b)

    lt = torch.min(a[:, None, :3], b[None, :, :3])
    rb = torch.max(a[:, None, 3:], b[None, :, 3:])
    wh = (rb - lt).clamp_min(0.0)
    enclosing = wh[..., 0] * wh[..., 1] * wh[..., 2]

    return iou - (enclosing - union) / enclosing.clamp_min(1e-6)
