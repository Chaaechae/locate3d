"""
Standalone plotly visualization for Locate3DSegDetector (0f / 0h) models.

Loads a trained checkpoint and runs inference on a small set of validation
scenes from ARKitScenes / ScanNet / ScanNetPP, writing one interactive
plotly HTML per scene that shows:

- The scene's RGB point cloud.
- Each GT entity as a dashed box (primary entity highlighted).
- Each predicted entity as a solid box (derived by the SegDetector as
  AABB of ``sigmoid(score) > threshold`` points).
- Optional per-entity mask overlay (points above threshold colored with
  the entity's color).
- The caption as the figure title.

Usage
-----

    # 0f checkpoint viz on ARKitScenes val (default)
    python tools/visualize_locate3d.py \\
        --config-file configs/utonia/localize-utonia-v1m1-0f-arkitscenes.py \\
        --weight exp/<your-run>/model/model_last.pth \\
        --num-scenes 5

    # 0h checkpoint viz on ScanNet val
    python tools/visualize_locate3d.py \\
        --config-file configs/utonia/localize-utonia-v1m1-0h-combined.py \\
        --weight exp/<your-run>/model/model_last.pth \\
        --dataset scannet \\
        --num-scenes 5 \\
        --output-dir viz_scannet/

    # Specific scene IDs
    python tools/visualize_locate3d.py ... --scene-ids scene0011_00,scene0025_00

Outputs
-------

HTML files under ``--output-dir`` (default ``viz_output/``). Open each
HTML in a browser; plotly gives you rotate / zoom / toggle-per-entity.
"""

import argparse
import copy
import os
import sys

# Make the repo-local ``pointcept`` package importable regardless of the
# user's current working directory. ``tools/visualize_locate3d.py`` lives
# one level below the repo root, so add the parent directory to sys.path
# before the first pointcept import. Equivalent to what a user-managed
# ``pip install -e .`` would achieve but without requiring that setup.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch

from pointcept.utils.config import Config
from pointcept.models.builder import build_model
from pointcept.datasets.locate3d_collate import locate3d_collate_fn

# Trigger dataset registration
import pointcept.datasets  # noqa: F401
from pointcept.datasets import (
    ARKitScenesLocate3DDataset,
    ScanNetLocate3DDataset,
    ScanNetPPLocate3DDataset,
)
from pointcept.datasets.transform import Compose


# Plotly qualitative palette (D3 "Category10"-ish)
_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def _box_edges(b):
    """16-point polyline tracing all 12 edges of an axis-aligned xyzxyz box."""
    x0, y0, z0, x1, y1, z1 = [float(v) for v in b]
    corners = [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
    ]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    xs, ys, zs = [], [], []
    for a, b in edges:
        xs.extend([corners[a][0], corners[b][0], None])
        ys.extend([corners[a][1], corners[b][1], None])
        zs.extend([corners[a][2], corners[b][2], None])
    return xs, ys, zs


def _make_color_str(hex_color):
    """Plotly line colors accept hex strings directly."""
    return hex_color


def _load_checkpoint(model, weight_path):
    """Load ``weight_path`` into ``model``, stripping any 'module.' prefix
    (training runs save with DDP wrapping)."""
    ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    cleaned = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module."):]
        cleaned[k] = v
    info = model.load_state_dict(cleaned, strict=False)
    print(
        f"[load] missing={len(info.missing_keys)} unexpected={len(info.unexpected_keys)}"
    )
    return model


def _build_val_transform_from_cfg(cfg):
    """Pick a reasonable val-transform spec out of the config. Configs
    vary in how they name the val transform list; try several keys."""
    # 0c-0g / 0f / 0h (scannet) variants
    for name in ("val_transform_scannet", "val_transform_arkit", "val_transform"):
        t = getattr(cfg, name, None)
        if t is not None:
            return list(t)
    # Fallback: minimal grid-sample + normalize + ToTensor + Collect
    return [
        dict(
            type="GridSample",
            grid_size=0.02,
            hash_type="fnv",
            mode="train",
            return_grid_coord=True,
        ),
        dict(type="NormalizeColor"),
        dict(type="ToTensor"),
        dict(
            type="Collect",
            keys=(
                "coord", "grid_coord", "caption", "positive_map",
                "primary_object_id", "scene_id", "name", "boxes_xyzxyz",
            ),
            feat_keys=("coord", "color", "normal"),
        ),
    ]


def _collect_for_dataset(cfg, kind):
    """Build a Collect-aware val transform list appropriate for the
    chosen dataset (ScanNet family needs 'instance' to survive, ARKit
    keeps 'boxes_xyzxyz' pre-transform)."""
    common = (
        "coord", "grid_coord", "caption", "positive_map",
        "primary_object_id", "scene_id", "name",
    )
    pre = [
        dict(
            type="GridSample",
            grid_size=0.02,
            hash_type="fnv",
            mode="train",
            return_grid_coord=True,
        ),
        dict(type="NormalizeColor"),
        dict(type="ToTensor"),
    ]
    if kind == "arkitscenes":
        return pre + [
            dict(
                type="Collect",
                keys=common + ("boxes_xyzxyz",),
                feat_keys=("coord", "color", "normal"),
            )
        ]
    else:  # scannet / scannetpp
        return pre + [
            dict(
                type="Collect",
                keys=common + ("instance",),
                feat_keys=("coord", "color", "normal"),
            )
        ]


def _build_dataset(args, cfg):
    kind = args.dataset
    # Annotation + data roots: prefer CLI args, else fall back to cfg.
    if kind == "arkitscenes":
        ann = args.annotation_file or getattr(
            cfg, "arkit_val_ann",
            "locate-3d/locate3d_data/val_arkitscenes.json",
        )
        root = args.data_root or getattr(
            cfg, "arkit_root",
            "/group-volume/3Ddataset/arkitscenes-compressed",
        )
        transform = _collect_for_dataset(cfg, "arkitscenes")
        return ARKitScenesLocate3DDataset(
            annotation_file=ann,
            data_root=root,
            transform=transform,
            test_mode=True,
            loop=1,
        )
    elif kind == "scannet":
        ann = args.annotation_file or getattr(
            cfg, "scannet_val_ann",
            "locate-3d/locate3d_data/val_scannet.json",
        )
        root = args.data_root or getattr(
            cfg, "scannet_root",
            "/group-volume/3Ddataset/scannet-compressed",
        )
        transform = _collect_for_dataset(cfg, "scannet")
        return ScanNetLocate3DDataset(
            annotation_file=ann,
            data_root=root,
            transform=transform,
            test_mode=True,
            loop=1,
        )
    elif kind == "scannetpp":
        ann = args.annotation_file or getattr(
            cfg, "scannetpp_val_ann",
            "locate-3d/locate3d_data/val_scannetpp.json",
        )
        root = args.data_root or getattr(
            cfg, "scannetpp_root",
            "/group-volume/3Ddataset/scannetpp-compressed",
        )
        transform = _collect_for_dataset(cfg, "scannetpp")
        return ScanNetPPLocate3DDataset(
            annotation_file=ann,
            data_root=root,
            transform=transform,
            test_mode=True,
            loop=1,
        )
    else:
        raise ValueError(f"unknown --dataset {kind!r}")


def _pick_indices(dataset, num_scenes, scene_ids_filter=None):
    """Select at most ``num_scenes`` sample indices, preferring distinct
    scene_ids and respecting an optional scene_id filter."""
    anns = dataset.anns
    if scene_ids_filter:
        wanted = set(scene_ids_filter)
        chosen = [i for i, a in enumerate(anns) if a["scene_id"] in wanted]
        return chosen[:num_scenes]
    chosen = []
    seen = set()
    for i, a in enumerate(anns):
        if a["scene_id"] in seen:
            continue
        seen.add(a["scene_id"])
        chosen.append(i)
        if len(chosen) >= num_scenes:
            break
    if len(chosen) < num_scenes:
        # fill with duplicates if we ran out of scenes
        rest = [i for i in range(len(anns)) if i not in chosen]
        chosen.extend(rest[: num_scenes - len(chosen)])
    return chosen


def _get_color_from_feat(feat_tensor):
    """``feat_keys=(coord, color, normal)`` → feat is (N, 9). Color is
    the middle 3 channels."""
    arr = feat_tensor.float().cpu().numpy()
    if arr.shape[1] >= 6:
        return arr[:, 3:6]
    return np.full((arr.shape[0], 3), 0.5, dtype=np.float32)


def _render_scene(
    out_path, coord, color, gt_boxes, pred_boxes,
    pred_logits=None, infer_threshold=0.5,
    caption="", scene_id="", primary_idx=0,
    entity_names=None, max_points=60000, draw_masks=True,
):
    try:
        import plotly.graph_objects as go
    except ImportError as e:
        print(f"[error] plotly not installed: {e}")
        return

    color = np.asarray(color, dtype=np.float32)
    if color.size > 0 and color.max() > 1.5:
        color = color / 255.0
    color = np.clip(color, 0.0, 1.0)

    N = coord.shape[0]
    if N > max_points:
        idx = np.random.choice(N, max_points, replace=False)
    else:
        idx = np.arange(N)
    c_sub = coord[idx]
    col_sub = color[idx]
    # pred_logits is (G, N) at the same N the model saw (which may be
    # subsampled inside the model for max_points_{train,eval}; we only
    # get back top-K point logit so best-effort align here). If shape
    # doesn't match, skip mask overlay.
    if pred_logits is not None and pred_logits.shape[1] == N:
        logit_sub = pred_logits[:, idx]
    else:
        logit_sub = None

    rgb_str = [
        f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})"
        for r, g, b in col_sub
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=c_sub[:, 0], y=c_sub[:, 1], z=c_sub[:, 2],
            mode="markers",
            marker=dict(size=1.2, color=rgb_str, opacity=0.6),
            name="scene",
            showlegend=False,
            hoverinfo="skip",
        )
    )

    G = max(len(gt_boxes), len(pred_boxes))
    for g in range(G):
        ent_color = _PALETTE[g % len(_PALETTE)]
        name = (
            entity_names[g]
            if entity_names is not None and g < len(entity_names)
            else f"entity_{g}"
        )
        is_primary = g == primary_idx
        suffix = " (primary)" if is_primary else ""

        # GT box: dashed
        if g < len(gt_boxes):
            xs, ys, zs = _box_edges(gt_boxes[g])
            fig.add_trace(
                go.Scatter3d(
                    x=xs, y=ys, z=zs, mode="lines",
                    line=dict(
                        color=ent_color,
                        width=5 if is_primary else 3,
                        dash="dash",
                    ),
                    name=f"GT {name}{suffix}",
                )
            )
        # Predicted box: solid
        if g < len(pred_boxes):
            xs, ys, zs = _box_edges(pred_boxes[g])
            fig.add_trace(
                go.Scatter3d(
                    x=xs, y=ys, z=zs, mode="lines",
                    line=dict(
                        color=ent_color,
                        width=7 if is_primary else 5,
                    ),
                    name=f"Pred {name}{suffix}",
                )
            )

        # Mask overlay: points exceeding threshold for this entity
        if draw_masks and logit_sub is not None and g < logit_sub.shape[0]:
            prob_g = 1.0 / (1.0 + np.exp(-logit_sub[g]))  # sigmoid
            mask_g = prob_g > infer_threshold
            if mask_g.any():
                mask_pts = c_sub[mask_g]
                fig.add_trace(
                    go.Scatter3d(
                        x=mask_pts[:, 0], y=mask_pts[:, 1], z=mask_pts[:, 2],
                        mode="markers",
                        marker=dict(
                            size=3 if is_primary else 2,
                            color=ent_color,
                            opacity=0.9,
                        ),
                        name=f"Mask {name}{suffix}",
                        hoverinfo="skip",
                    )
                )

    title = f"<b>{scene_id}</b><br><sub>{caption}</sub>"
    fig.update_layout(
        title=title,
        scene=dict(
            aspectmode="data",
            xaxis_title="x", yaxis_title="y", zaxis_title="z",
        ),
        height=850,
        margin=dict(l=0, r=0, t=60, b=0),
        legend=dict(itemsizing="constant"),
    )
    fig.write_html(out_path)
    print(f"[wrote] {out_path}")


def _sample_to_gpu(batch):
    gpu = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            gpu[k] = v.cuda(non_blocking=True)
        else:
            gpu[k] = v
    return gpu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-file", required=True, help="0f / 0h config used for training")
    ap.add_argument("--weight", required=True, help="trained checkpoint path")
    ap.add_argument(
        "--dataset",
        choices=["arkitscenes", "scannet", "scannetpp"],
        default="arkitscenes",
    )
    ap.add_argument("--num-scenes", type=int, default=5)
    ap.add_argument("--output-dir", default="viz_output")
    ap.add_argument("--scene-ids", default=None, help="comma-separated scene_id filter")
    ap.add_argument(
        "--annotation-file", default=None,
        help="override the dataset's val JSON path",
    )
    ap.add_argument(
        "--data-root", default=None,
        help="override the dataset's preprocessed scene root",
    )
    ap.add_argument(
        "--infer-threshold", type=float, default=0.5,
        help="sigmoid(score) threshold for mask visualization",
    )
    ap.add_argument("--no-mask", action="store_true", help="skip mask overlay")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = Config.fromfile(args.config_file)
    # cfg.weight would otherwise still be None; we pass weight explicitly.
    cfg.weight = args.weight

    # -- model --
    model = build_model(cfg.model)
    model_type = model.__class__.__name__
    if model_type != "Locate3DSegDetector":
        print(
            f"[warn] model is {model_type!r}, not Locate3DSegDetector. "
            f"This script expects the 0f / 0h seg-based outputs "
            f"(pred_boxes_per_entity / pred_logits_per_entity). Trying "
            f"anyway; you may need to adjust for DETR-style configs."
        )
    _load_checkpoint(model, args.weight)
    model = model.cuda().eval()

    # -- dataset --
    dataset = _build_dataset(args, cfg)
    print(f"[dataset] {args.dataset}: {len(dataset)} annotations "
          f"across {len(set(a['scene_id'] for a in dataset.anns))} scenes")

    scene_ids_filter = None
    if args.scene_ids:
        scene_ids_filter = [s.strip() for s in args.scene_ids.split(",") if s.strip()]
    indices = _pick_indices(dataset, args.num_scenes, scene_ids_filter)
    print(f"[pick] {len(indices)} scenes selected: "
          f"{[dataset.anns[i]['scene_id'] for i in indices]}")

    os.makedirs(args.output_dir, exist_ok=True)

    for rank, ds_idx in enumerate(indices):
        try:
            sample = dataset[ds_idx]
        except Exception as e:
            print(f"[skip] sample {ds_idx}: {e}")
            continue

        batch = locate3d_collate_fn([copy.deepcopy(sample)])
        batch_gpu = _sample_to_gpu(batch)

        with torch.no_grad():
            out = model(batch_gpu)

        pred_boxes_per = out.get("pred_boxes_per_entity", None)
        pred_logits_per = out.get("pred_logits_per_entity", None)
        if pred_boxes_per is None or pred_boxes_per[0] is None:
            print(f"[skip] model returned no pred_boxes_per_entity for {ds_idx}")
            continue

        pred_boxes = pred_boxes_per[0].float().cpu().numpy()
        pred_logits = (
            pred_logits_per[0].float().cpu().numpy()
            if (pred_logits_per is not None and pred_logits_per[0] is not None)
            else None
        )

        # GT + meta
        gt_list = batch.get("boxes_xyzxyz", None)
        gt_boxes = (
            gt_list[0].float().cpu().numpy()
            if gt_list is not None and gt_list[0] is not None
            else np.zeros((0, 6), dtype=np.float32)
        )
        caption = batch["caption"][0] if isinstance(batch["caption"], list) else str(batch["caption"])
        scene_id = batch["scene_id"][0] if isinstance(batch["scene_id"], list) else str(batch["scene_id"])
        pid_raw = batch.get("primary_object_id", [0])[0]
        if isinstance(pid_raw, torch.Tensor):
            primary = int(pid_raw.flatten()[0].item())
        else:
            primary = int(pid_raw) if pid_raw is not None else 0

        coord = batch["coord"].float().cpu().numpy()
        color = _get_color_from_feat(batch["feat"])

        ann = dataset.anns[ds_idx]
        entity_name_field = ann.get("object_name", "")
        # Give at least the primary entity a human-readable name.
        G = max(gt_boxes.shape[0], pred_boxes.shape[0])
        entity_names = [f"entity_{i}" for i in range(G)]
        if entity_name_field and primary < len(entity_names):
            entity_names[primary] = str(entity_name_field)

        out_path = os.path.join(
            args.output_dir,
            f"scene{rank:02d}_{args.dataset}_{scene_id}_{ann.get('ann_id', ds_idx)}.html",
        )
        _render_scene(
            out_path=out_path,
            coord=coord,
            color=color,
            gt_boxes=gt_boxes,
            pred_boxes=pred_boxes,
            pred_logits=pred_logits,
            infer_threshold=args.infer_threshold,
            caption=caption,
            scene_id=scene_id,
            primary_idx=primary,
            entity_names=entity_names,
            draw_masks=not args.no_mask,
        )

        del out, batch, batch_gpu
        torch.cuda.empty_cache()

    print(f"\nAll outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()
