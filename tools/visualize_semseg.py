"""
Plotly visualization for SemSeg (semseg-utonia / scannet) checkpoints.

Renders per-scene 3D point clouds colored by:
- ground-truth class (left scene), and
- model's predicted class (right scene),
side-by-side in one HTML file. Points where pred != gt are highlighted
in a separate trace so disagreements stand out.

Usage::

    python tools/visualize_semseg.py \\
        --config-file configs/utonia/semseg-utonia-v1m1-0b-scannet-dec.py \\
        --weight exp/<run>/model/model_last.pth \\
        --num-scenes 5 \\
        --output-dir viz_semseg/

Optional flags:
    --scene-ids scene0011_00,scene0025_00   # filter to specific scenes
    --max-points 120000                     # render cap per panel
    --error-only                            # only render the disagreement panel
"""

import argparse
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np
import torch

from pointcept.utils.config import Config
from pointcept.models.builder import build_model
from pointcept.datasets.builder import build_dataset
from pointcept.datasets.preprocessing.scannet.meta_data.scannet200_constants import (
    CLASS_LABELS_20,
    SCANNET_COLOR_MAP_20,
)


def _build_class_palette():
    """Reorder SCANNET_COLOR_MAP_20 by VALID_CLASS_IDS_20 -> 0..19 indexing.
    The label tensors at training/eval time are already remapped to 0..19,
    not the raw scannet class ids, so we want palette[label_idx]."""
    from pointcept.datasets.preprocessing.scannet.meta_data.scannet200_constants import (
        VALID_CLASS_IDS_20,
    )
    palette = np.zeros((len(VALID_CLASS_IDS_20), 3), dtype=np.float32)
    for i, cid in enumerate(VALID_CLASS_IDS_20):
        c = SCANNET_COLOR_MAP_20.get(cid, (200.0, 200.0, 200.0))
        palette[i] = np.array(c, dtype=np.float32) / 255.0
    return palette


def _load_checkpoint(model, weight_path):
    ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    cleaned = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module."):]
        cleaned[k] = v
    info = model.load_state_dict(cleaned, strict=False)
    print(
        f"[load] missing={len(info.missing_keys)} "
        f"unexpected={len(info.unexpected_keys)}"
    )
    return model


def _to_gpu(d):
    return {
        k: (v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v)
        for k, v in d.items()
    }


def _palette_to_rgb_str(palette, labels):
    """labels: (N,) int in [0..K) or -1 for ignore. Returns list of
    plotly rgb strings."""
    out = []
    for L in labels:
        if L < 0 or L >= palette.shape[0]:
            out.append("rgb(120,120,120)")  # ignored / out-of-range
            continue
        r, g, b = palette[L]
        out.append(f"rgb({int(r*255)},{int(g*255)},{int(b*255)})")
    return out


def _render(out_path, coord, gt_labels, pred_labels, palette,
            scene_id="", max_points=120000, error_only=False):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as e:
        print(f"[error] plotly not installed: {e}")
        return

    N = coord.shape[0]
    if N > max_points:
        sel = np.random.choice(N, max_points, replace=False)
    else:
        sel = np.arange(N)
    coord = coord[sel]
    gt_labels = gt_labels[sel]
    pred_labels = pred_labels[sel]

    err_mask = (gt_labels >= 0) & (pred_labels != gt_labels)

    if error_only:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter3d(
                x=coord[:, 0], y=coord[:, 1], z=coord[:, 2],
                mode="markers",
                marker=dict(
                    size=2.0,
                    color=["rgb(220,220,220)" if not e else "rgb(255,0,0)"
                           for e in err_mask],
                    opacity=0.85,
                ),
                name="errors",
                hoverinfo="skip",
            )
        )
        fig.update_layout(
            title=f"<b>{scene_id}</b><br>"
                  f"<sub>{int(err_mask.sum())} / {N} points wrong "
                  f"({err_mask.mean()*100:.1f}%)</sub>",
            scene=dict(aspectmode="data"),
            height=850,
            margin=dict(l=0, r=0, t=80, b=0),
        )
        fig.write_html(out_path)
        print(f"[wrote] {out_path}")
        return

    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "scatter3d"}] * 3],
        subplot_titles=("GT class", "Pred class", "Errors (red)"),
        horizontal_spacing=0.01,
    )

    fig.add_trace(
        go.Scatter3d(
            x=coord[:, 0], y=coord[:, 1], z=coord[:, 2],
            mode="markers",
            marker=dict(
                size=1.6,
                color=_palette_to_rgb_str(palette, gt_labels),
                opacity=0.85,
            ),
            name="gt",
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter3d(
            x=coord[:, 0], y=coord[:, 1], z=coord[:, 2],
            mode="markers",
            marker=dict(
                size=1.6,
                color=_palette_to_rgb_str(palette, pred_labels),
                opacity=0.85,
            ),
            name="pred",
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scatter3d(
            x=coord[:, 0], y=coord[:, 1], z=coord[:, 2],
            mode="markers",
            marker=dict(
                size=1.6,
                color=["rgb(220,220,220)" if not e else "rgb(255,0,0)"
                       for e in err_mask],
                opacity=0.85,
            ),
            name="errors",
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1, col=3,
    )

    correct = (gt_labels >= 0).sum()
    err_pct = (err_mask.sum() / max(correct, 1)) * 100
    fig.update_layout(
        title=(f"<b>{scene_id}</b> &nbsp; "
               f"<sub>error: {err_mask.sum()} / {correct} pts "
               f"({err_pct:.1f}%)</sub>"),
        height=720,
        margin=dict(l=0, r=0, t=70, b=0),
    )
    # share aspect across the three sub-scenes
    for k in ("scene", "scene2", "scene3"):
        fig.layout[k].aspectmode = "data"

    fig.write_html(out_path)
    print(f"[wrote] {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-file", required=True)
    ap.add_argument("--weight", required=True)
    ap.add_argument("--num-scenes", type=int, default=5)
    ap.add_argument("--scene-ids", default=None)
    ap.add_argument("--output-dir", default="viz_semseg")
    ap.add_argument("--max-points", type=int, default=120000)
    ap.add_argument("--error-only", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    cfg = Config.fromfile(args.config_file)

    model = build_model(cfg.model).cuda().eval()
    _load_checkpoint(model, args.weight)

    val_dataset = build_dataset(cfg.data.val)

    # Resolve indices to render.
    anns = list(val_dataset.data_list)  # ScanNetDataset.data_list = list of scene dirs
    scene_ids = []
    for a in anns:
        sid = os.path.basename(a) if isinstance(a, str) else str(a)
        scene_ids.append(sid)
    if args.scene_ids:
        wanted = set(s.strip() for s in args.scene_ids.split(",") if s.strip())
        picks = [i for i, s in enumerate(scene_ids) if s in wanted]
    else:
        picks = list(range(min(args.num_scenes, len(anns))))
    print(f"[picks] {[scene_ids[i] for i in picks]}")

    palette = _build_class_palette()

    for idx in picks:
        try:
            sample = val_dataset[idx]
        except Exception as e:
            print(f"[skip] {scene_ids[idx]}: {type(e).__name__}: {e}")
            continue
        # SemSegEvaluator-style forward:
        #   batch = collate; coord/feat/segment all (N, ...)
        # Forward expects a dict with the standard pointcept keys.
        with torch.no_grad():
            input_dict = _to_gpu({
                k: v for k, v in sample.items()
                if isinstance(v, torch.Tensor)
            })
            # Pointcept Default model returns dict with 'seg_logits' (N, C).
            out = model(input_dict)
        seg_logits = out.get("seg_logits", None)
        if seg_logits is None:
            print(f"[skip] {scene_ids[idx]}: model has no seg_logits in output")
            continue
        pred = seg_logits.argmax(dim=-1).cpu().numpy()

        coord = sample["coord"].cpu().numpy() if isinstance(sample["coord"], torch.Tensor) \
            else np.asarray(sample["coord"])
        gt = sample.get("segment", sample.get("segment20", None))
        if gt is None:
            print(f"[skip] {scene_ids[idx]}: no segment label key")
            continue
        gt_np = gt.cpu().numpy() if isinstance(gt, torch.Tensor) else np.asarray(gt)
        # Pointcept's ignore label is -1 in remapped space.
        gt_np = gt_np.astype(np.int64)

        # Sanity: align lengths
        n = min(coord.shape[0], pred.shape[0], gt_np.shape[0])
        coord, pred, gt_np = coord[:n], pred[:n], gt_np[:n]

        out_html = os.path.join(args.output_dir, f"{scene_ids[idx]}.html")
        _render(
            out_path=out_html,
            coord=coord,
            gt_labels=gt_np,
            pred_labels=pred,
            palette=palette,
            scene_id=scene_ids[idx],
            max_points=args.max_points,
            error_only=args.error_only,
        )

    # Print legend (class -> color) once for reference.
    print("\n[legend] class colors:")
    for i, name in enumerate(CLASS_LABELS_20):
        r, g, b = palette[i]
        print(f"  [{i:2d}] {name:18s}  rgb({int(r*255)},{int(g*255)},{int(b*255)})")


if __name__ == "__main__":
    main()
