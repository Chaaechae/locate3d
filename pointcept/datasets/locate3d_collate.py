"""
Collate function for Locate-3D style referring expression datasets.

Handles the mix of:
  - per-point pointcept tensors (coord, color, normal, feat, grid_coord, ...)
    which are concatenated along the point axis with an accompanying ``offset``;
  - per-sample scalar / variable-length fields (caption, per-scene boxes and
    per-scene positive maps) which are gathered into per-sample lists so the
    Locate-3D decoder loss can keep separate targets per scene.
"""

from collections.abc import Mapping, Sequence
import numpy as np
import torch
from torch.utils.data.dataloader import default_collate


# Keys that must be kept per-sample rather than concatenated.
_PER_SAMPLE_LIST_KEYS = {
    "caption",
    "scene_id",
    "boxes_xyzxyz",
    "positive_map",
    "point_masks",
    "primary_object_id",
}

# Keys whose values are per-point tensors that must receive an offset.
_POINT_TENSOR_KEYS = {
    "coord",
    "origin_coord",
    "grid_coord",
    "color",
    "normal",
    "feat",
    "strength",
    "segment",
    "instance",
    "superpoint",
}


def _as_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    return torch.as_tensor(x)


def locate3d_collate_fn(batch):
    assert isinstance(batch[0], Mapping), "ARKitScenesLocate3DDataset returns dicts"

    out = {}
    keys = set()
    for b in batch:
        keys.update(b.keys())

    point_offsets = []  # cumulative number of points per sample
    running = 0

    # Determine per-sample point count from whichever point-key we find first.
    point_count_key = None
    for k in _POINT_TENSOR_KEYS:
        if k in batch[0]:
            point_count_key = k
            break
    if point_count_key is None:
        point_count_key = "coord"

    for b in batch:
        n = int(_as_tensor(b[point_count_key]).shape[0]) if point_count_key in b else 0
        running += n
        point_offsets.append(running)
    out["offset"] = torch.tensor(point_offsets, dtype=torch.int64)

    for k in keys:
        if k == "offset":
            # already computed above
            continue
        if k in _PER_SAMPLE_LIST_KEYS:
            vals = []
            for b in batch:
                v = b.get(k, None)
                if v is None:
                    vals.append(None)
                elif isinstance(v, (int, float, str)):
                    vals.append(v)
                else:
                    vals.append(_as_tensor(v))
            out[k] = vals
        elif k in _POINT_TENSOR_KEYS:
            vals = [_as_tensor(b[k]) for b in batch if k in b]
            if len(vals) == 0:
                continue
            out[k] = torch.cat(vals, dim=0)
        elif k == "name":
            out[k] = [b.get(k, "") for b in batch]
        elif k == "grid_size":
            # keep the scalar (they're all equal)
            v = batch[0][k]
            out[k] = v if isinstance(v, torch.Tensor) else torch.as_tensor(v)
        else:
            # fall back to default collate for any remaining keys
            try:
                vals = [b[k] for b in batch if k in b]
                if len(vals) == 0:
                    continue
                if isinstance(vals[0], torch.Tensor):
                    out[k] = torch.cat(vals, dim=0)
                else:
                    out[k] = default_collate(vals)
            except Exception:
                out[k] = [b.get(k, None) for b in batch]
    return out
