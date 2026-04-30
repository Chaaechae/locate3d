"""
Analyze whether Meta Locate-3D's predictions on the user's
preprocessed cache have a SYSTEMATIC offset from the ground truth.

If they do (e.g. consistent +0.3 m bias on the y axis), most of the
4-5x gap between paper Acc@0.25 (~0.62) and our measured Acc@0.25
(~0.14) is explained by a preprocessing-pipeline shift, and a single
calibration vector recovers a large chunk of the gap. If they don't,
the predictions are just per-sample noisy and there's nothing to
"align" -- the gap is real model / data drift.

What the script computes (over the first --num-samples utterances):

  per-axis center delta   = pred_center - gt_center
  per-axis size ratio     = pred_size  / gt_size       (log)
  euclidean center dist   = ||pred_center - gt_center||
  raw IoU distribution    = histogram of per-sample IoU
  aligned IoU             = IoU after subtracting MEAN delta from preds
  rescaled IoU            = IoU after also rescaling preds by MEAN ratio

Outputs three things:
  1. printed summary stats
  2. an optional matplotlib histogram (--plot-out)
  3. a pickled dict for later analysis (--records-out)

Usage:

    LOCATE3D_USE_ARKIT=0 LOCATE3D_USE_SCANNETPP=0 \\
    LOCATE3D_CLIP_PATH=/group-volume/CLIP/clip-vit-large-patch14 \\
    python tools/analyze_baseline_alignment.py \\
        --annotations locate-3d/locate3d_data/val_scannet.json \\
        --scannet-data-dir /group-volume/Scannet/scans \\
        --cache-path /group-volume/ARKitscenes_cache_copy \\
        --weight /group-volume/locate-3d/model.safetensors \\
        --num-samples 100 \\
        --matcher raw_logits \\
        --box-source mask_aabb
"""

import argparse
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np
import torch

_LOCATE3D = os.path.join(_REPO, "locate-3d")
if _LOCATE3D not in sys.path:
    sys.path.insert(0, _LOCATE3D)
_TOOLS = os.path.dirname(os.path.abspath(__file__))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)
from _pytorch3d_shim import install_shim as _install_pytorch3d_shim  # noqa: E402
_install_pytorch3d_shim()

from eval_locate3d_baseline import (  # noqa: E402
    _load_model,
    _resolve_oid_token_set,
    _word_text_to_token_indices,
    _xyzxyz_from_anything,
    _iou_3d_xyzxyz,
)


def _box_center_size(box_xyzxyz):
    a = np.asarray(box_xyzxyz, dtype=np.float64).reshape(-1)
    center = (a[:3] + a[3:]) / 2.0
    size = np.maximum(a[3:] - a[:3], 1e-6)
    return center, size


def _shift_box(box_xyzxyz, delta):
    """Translate a box by delta (3,)."""
    a = np.asarray(box_xyzxyz, dtype=np.float64).reshape(-1).copy()
    a[:3] += delta
    a[3:] += delta
    return a


def _rescale_box(box_xyzxyz, scale):
    """Rescale box around its OWN center by per-axis scale factor (3,)."""
    a = np.asarray(box_xyzxyz, dtype=np.float64).reshape(-1)
    center = (a[:3] + a[3:]) / 2.0
    half = (a[3:] - a[:3]) / 2.0
    new_half = half * np.asarray(scale)
    return np.concatenate([center - new_half, center + new_half])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--scannet-data-dir", default=None)
    ap.add_argument("--scannetpp-data-dir", default=None)
    ap.add_argument("--arkitscenes-data-dir", default=None)
    ap.add_argument("--cache-path", required=True)
    ap.add_argument("--weight",
                    default="/group-volume/locate-3d/model.safetensors")
    ap.add_argument("--config", default="locate-3d/config/locate_3d.yaml")
    ap.add_argument("--num-samples", type=int, default=100)
    ap.add_argument("--downsample-pts", type=int, default=30000)
    ap.add_argument("--matcher", default="raw_logits",
                    choices=("post_process", "raw_logits"))
    ap.add_argument("--box-source", default="mask_aabb",
                    choices=("pred_box", "mask_aabb"))
    ap.add_argument("--mask-threshold", type=float, default=0.5)
    ap.add_argument("--records-out", default=None,
                    help="optional .pt path to dump per-sample records")
    ap.add_argument("--plot-out", default=None,
                    help="optional .png histogram of IoU before / after "
                         "alignment (requires matplotlib)")
    args = ap.parse_args()

    # Load Meta dataset, model.
    from locate3d_data.locate3d_dataset import Locate3DDataset
    dataset = Locate3DDataset(
        annotations_fpath=args.annotations,
        return_featurized_pointcloud=True,
        scannet_data_dir=args.scannet_data_dir,
        scannetpp_data_dir=args.scannetpp_data_dir,
        arkitscenes_data_dir=args.arkitscenes_data_dir,
        cache_path=args.cache_path,
    )
    annos = list(dataset.annos)
    n_total = min(args.num_samples, len(annos))
    print(f"[setup] {n_total}/{len(annos)} samples; matcher={args.matcher} "
          f"box_source={args.box_source}")

    model = _load_model(args.weight, args.config)
    tokenizer = model.decoder.tokenizer

    # Per-sample records.
    deltas = []          # pred_center - gt_center, list of (3,)
    log_ratios = []      # log(pred_size / gt_size), list of (3,)
    raw_ious = []
    pred_centers = []
    gt_centers = []
    pred_boxes = []
    gt_boxes = []
    sample_meta = []

    for idx in range(n_total):
        try:
            data = dataset[idx]
        except Exception as e:
            print(f"[skip] sample {idx}: {type(e).__name__}: {e}")
            continue
        if "featurized_sensor_pointcloud" not in data:
            continue

        ann = annos[idx]

        # Same length-aware downsample as eval_locate3d_baseline.py.
        psp = {
            k: (v.cpu() if torch.is_tensor(v) else v)
            for k, v in data["featurized_sensor_pointcloud"].items()
        }
        point_keys = [k for k in psp if k in
                      ("points", "rgb", "features_clip", "features_dino")]
        N = min(psp[k].shape[0] for k in point_keys)
        for k in point_keys:
            if psp[k].shape[0] != N:
                psp[k] = psp[k][:N]
        if N > args.downsample_pts:
            indices = torch.randperm(N)[:args.downsample_pts]
            for k in point_keys:
                psp[k] = psp[k][indices]
        data["featurized_sensor_pointcloud"] = psp

        utterance = data["lang_data"]["text_caption"]
        oids, tokens_per_oid = _resolve_oid_token_set(ann)
        word_to_clip = _word_text_to_token_indices(
            tokenizer, utterance, ann.get("token", [])
        )
        clip_tokens_per_oid = {
            oid: set().union(
                *(word_to_clip[wi] for wi in tokens_per_oid[oid]
                  if wi in word_to_clip)
            ) if tokens_per_oid[oid] else set()
            for oid in oids
        }

        # Resolve GT box for primary entity.
        ann_oids = ann.get("object_ids", []) or []
        gt_arr = data["lang_data"].get("gt_boxes", None)
        primary_oid = oids[0]
        gt_box = None
        gt_boxes_raw = ann.get("gt_boxes", None)
        if gt_boxes_raw is not None and primary_oid < len(gt_boxes_raw):
            raw = gt_boxes_raw[primary_oid]
            if raw is not None:
                gt_box = _xyzxyz_from_anything(raw)
        elif gt_arr is not None and primary_oid in ann_oids:
            gi = ann_oids.index(primary_oid)
            if gi < len(gt_arr):
                cand = gt_arr[gi]
                if torch.is_tensor(cand):
                    cand_np = cand.cpu().numpy()
                else:
                    cand_np = np.asarray(cand)
                if np.isfinite(cand_np).all():
                    gt_box = _xyzxyz_from_anything(cand_np)
        if gt_box is None:
            continue

        # Run inference + extract pred for primary entity (matcher path).
        psp_cuda = {
            k: (v.cuda(non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in data["featurized_sensor_pointcloud"].items()
        }
        try:
            with torch.no_grad():
                if args.matcher == "raw_logits":
                    raw = model(psp_cuda, utterance)
                    logits = raw["pred_logits"][0]      # (Q, T)
                    masks_all = raw["pred_masks"][0]    # (Q, N)
                    boxes_all = raw["pred_boxes"][0]    # (Q, 6)
                    target = clip_tokens_per_oid.get(primary_oid, set())
                    target_idx = torch.tensor(
                        sorted(t for t in target if t < logits.shape[1]),
                        device=logits.device, dtype=torch.long,
                    )
                    if target_idx.numel() == 0:
                        continue
                    sig = torch.sigmoid(logits)
                    scores = sig[:, target_idx].mean(dim=1)
                    qi = int(scores.argmax().item())
                    if args.box_source == "mask_aabb":
                        mask_t = torch.sigmoid(masks_all[qi])
                        m = mask_t.detach().cpu().numpy().reshape(-1)
                        pts = data["featurized_sensor_pointcloud"]["points"]
                        pts_np = (pts.cpu().numpy()
                                  if torch.is_tensor(pts)
                                  else np.asarray(pts))
                        n_align = min(m.shape[0], pts_np.shape[0])
                        sel = m[:n_align] > args.mask_threshold
                        if sel.sum() == 0:
                            continue
                        mp = pts_np[:n_align][sel]
                        pred_box = np.concatenate(
                            [mp.min(0), mp.max(0)]
                        ).astype(np.float32)
                    else:
                        pb = boxes_all[qi].detach().cpu().numpy()
                        pred_box = _xyzxyz_from_anything(pb)
                else:
                    instances = model.inference(psp_cuda, utterance)
                    target = clip_tokens_per_oid.get(primary_oid, set())
                    best = (None, -1, -1.0)
                    for ii, inst in enumerate(instances):
                        pclip = set(int(t) for t in inst["tokens_assigned"])
                        ov = len(target & pclip)
                        cf = float(inst.get("confidence", 0.0))
                        if ov > best[1] or (ov == best[1] and cf > best[2]):
                            best = (ii, ov, cf)
                    if best[0] is None or best[1] <= 0:
                        continue
                    inst = instances[best[0]]
                    if args.box_source == "mask_aabb":
                        mask_t = inst.get("mask", None)
                        if mask_t is None:
                            continue
                        m = mask_t.detach().cpu().numpy().reshape(-1)
                        pts = data["featurized_sensor_pointcloud"]["points"]
                        pts_np = (pts.cpu().numpy()
                                  if torch.is_tensor(pts)
                                  else np.asarray(pts))
                        n_align = min(m.shape[0], pts_np.shape[0])
                        sel = m[:n_align] > args.mask_threshold
                        if sel.sum() == 0:
                            continue
                        mp = pts_np[:n_align][sel]
                        pred_box = np.concatenate(
                            [mp.min(0), mp.max(0)]
                        ).astype(np.float32)
                    else:
                        pb = inst["bbox"].detach().cpu().numpy()
                        pred_box = _xyzxyz_from_anything(pb)
        except Exception as e:
            print(f"[skip] inference {idx}: {type(e).__name__}: {e}")
            continue

        gtc, gts = _box_center_size(gt_box)
        prc, prs = _box_center_size(pred_box)
        delta = prc - gtc
        log_ratio = np.log(prs / gts)
        raw_iou = _iou_3d_xyzxyz(pred_box, gt_box)

        deltas.append(delta)
        log_ratios.append(log_ratio)
        raw_ious.append(raw_iou)
        pred_centers.append(prc)
        gt_centers.append(gtc)
        pred_boxes.append(pred_box)
        gt_boxes.append(gt_box)
        sample_meta.append({
            "idx": idx,
            "scene_id": ann.get("scene_id"),
            "ann_id": ann.get("ann_id"),
            "primary_oid": primary_oid,
            "caption": utterance,
        })

        if (len(raw_ious)) % 25 == 0:
            print(f"[progress] processed {len(raw_ious)}/{n_total}")

    if not raw_ious:
        print("[empty] no usable samples; aborting analysis.")
        return

    deltas = np.stack(deltas)         # (N, 3)
    log_ratios = np.stack(log_ratios) # (N, 3)
    raw_ious = np.array(raw_ious)
    print(f"\n[collected] N={len(raw_ious)} usable samples")

    print("\n=== raw IoU distribution ===")
    print(f"  mean={raw_ious.mean():.4f}  median={np.median(raw_ious):.4f}")
    for t in (0.25, 0.5):
        print(f"  Acc@{t} = {(raw_ious >= t).mean():.4f}")
    print(f"  IoU=0 fraction = {(raw_ious == 0).mean():.4f}")

    print("\n=== center delta (pred - gt) ===")
    for axis, name in zip(range(3), "xyz"):
        d = deltas[:, axis]
        print(f"  axis {name}: mean={d.mean():+.4f} m  median={np.median(d):+.4f} m  "
              f"std={d.std():.4f}  |p25={np.percentile(d, 25):+.4f}  "
              f"p75={np.percentile(d, 75):+.4f}|")
    eu = np.linalg.norm(deltas, axis=1)
    print(f"  euclidean center dist: mean={eu.mean():.4f}  median={np.median(eu):.4f}  "
          f"p90={np.percentile(eu, 90):.4f}")

    print("\n=== size ratio (pred / gt, log) ===")
    for axis, name in zip(range(3), "xyz"):
        r = np.exp(log_ratios[:, axis])
        print(f"  axis {name}: ratio mean={r.mean():.3f}  median={np.median(r):.3f} "
              f"(values <1 = pred smaller; values >1 = pred bigger)")

    # Translation alignment: subtract MEAN delta from all preds, recompute IoU.
    mean_delta = deltas.mean(axis=0)
    aligned_ious = []
    for i in range(len(raw_ious)):
        shifted_pred = _shift_box(pred_boxes[i], -mean_delta)
        aligned_ious.append(_iou_3d_xyzxyz(shifted_pred, gt_boxes[i]))
    aligned_ious = np.array(aligned_ious)
    print("\n=== aligned IoU (subtract mean delta) ===")
    print(f"  mean delta applied = ({mean_delta[0]:+.4f}, {mean_delta[1]:+.4f}, "
          f"{mean_delta[2]:+.4f}) m")
    print(f"  mean={aligned_ious.mean():.4f}  median={np.median(aligned_ious):.4f}")
    for t in (0.25, 0.5):
        delta_acc = (aligned_ious >= t).mean() - (raw_ious >= t).mean()
        print(f"  Acc@{t} = {(aligned_ious >= t).mean():.4f}  "
              f"(raw was {(raw_ious >= t).mean():.4f}, change "
              f"{'+' if delta_acc >= 0 else ''}{delta_acc:.4f})")

    # Translation + scale alignment.
    mean_log_ratio = log_ratios.mean(axis=0)
    inv_scale = np.exp(-mean_log_ratio)  # divide pred by mean ratio
    rescaled_ious = []
    for i in range(len(raw_ious)):
        shifted = _shift_box(pred_boxes[i], -mean_delta)
        rescaled = _rescale_box(shifted, inv_scale)
        rescaled_ious.append(_iou_3d_xyzxyz(rescaled, gt_boxes[i]))
    rescaled_ious = np.array(rescaled_ious)
    print("\n=== aligned + rescaled IoU (subtract mean delta + divide by mean size ratio) ===")
    print(f"  inv_scale applied = ({inv_scale[0]:.3f}, {inv_scale[1]:.3f}, "
          f"{inv_scale[2]:.3f})")
    print(f"  mean={rescaled_ious.mean():.4f}  median={np.median(rescaled_ious):.4f}")
    for t in (0.25, 0.5):
        delta_acc = (rescaled_ious >= t).mean() - (raw_ious >= t).mean()
        print(f"  Acc@{t} = {(rescaled_ious >= t).mean():.4f}  "
              f"(raw was {(raw_ious >= t).mean():.4f}, change "
              f"{'+' if delta_acc >= 0 else ''}{delta_acc:.4f})")

    # Interpretation hint.
    print("\n=== interpretation hint ===")
    if (aligned_ious >= 0.25).mean() - (raw_ious >= 0.25).mean() > 0.10:
        print("  Aligned Acc@0.25 jumps >=10pp -> SYSTEMATIC translation bias")
        print("  in the model's predictions on this cache. Likely cause: a")
        print("  preprocessing pipeline difference (centering / axis-align /")
        print("  voxelization) between paper-train and user-cache.")
    elif (aligned_ious >= 0.25).mean() - (raw_ious >= 0.25).mean() > 0.03:
        print("  Aligned Acc@0.25 improves modestly (3-10pp). Mixed: small")
        print("  systematic bias + per-sample noise.")
    else:
        print("  Aligned Acc@0.25 barely changes (<3pp). Predictions are")
        print("  per-sample NOISY, not systematically shifted. Calibration")
        print("  won't help; root cause is upstream (cache features differ")
        print("  from train distribution, or model truly underperforms here).")

    if args.records_out:
        torch.save({
            "deltas": deltas,
            "log_ratios": log_ratios,
            "raw_ious": raw_ious,
            "aligned_ious": aligned_ious,
            "rescaled_ious": rescaled_ious,
            "mean_delta": mean_delta,
            "mean_log_ratio": mean_log_ratio,
            "sample_meta": sample_meta,
        }, args.records_out)
        print(f"[save] records -> {args.records_out}")

    if args.plot_out:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 5))
            bins = np.linspace(0, 1, 21)
            ax.hist(raw_ious, bins=bins, alpha=0.5, label="raw")
            ax.hist(aligned_ious, bins=bins, alpha=0.5, label="aligned (delta)")
            ax.hist(rescaled_ious, bins=bins, alpha=0.5,
                    label="aligned+rescaled")
            ax.axvline(0.25, color="r", linestyle="--", label="Acc@0.25 threshold")
            ax.set_xlabel("IoU")
            ax.set_ylabel("samples")
            ax.set_title(f"Pred vs GT IoU (N={len(raw_ious)})")
            ax.legend()
            fig.tight_layout()
            fig.savefig(args.plot_out, dpi=120)
            print(f"[save] plot -> {args.plot_out}")
        except ImportError:
            print("[warn] matplotlib not available; skipping plot")


if __name__ == "__main__":
    main()
