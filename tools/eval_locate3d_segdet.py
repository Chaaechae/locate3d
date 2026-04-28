"""
Standalone evaluation for Locate3DSegDetector (0f / 0h / 0i / 0j)
checkpoints. Avoids ``tools/test.py`` because that path is wired for
SemSegTester (which expects ``cfg.data.test`` and a ScanNetDataset
schema we don't have here).

Mirrors ``Locate3DSegDetectorEvaluator.eval`` so the numbers it prints
are identical to what the training-time evaluator hook would report on
the same val set.

Usage::

    # Full combined-corpus val (ARKit + ScanNet + ScanNetPP):
    python tools/eval_locate3d_segdet.py \\
        --config-file configs/utonia/localize-utonia-v1m1-0j-encoder-ft.py \\
        --weight exp/<run>/model/model_best.pth

    # ScanNet-only val (uses the same env-var toggles as training):
    LOCATE3D_USE_ARKIT=0 LOCATE3D_USE_SCANNETPP=0 \\
    python tools/eval_locate3d_segdet.py \\
        --config-file configs/utonia/localize-utonia-v1m1-0j-encoder-ft.py \\
        --weight exp/<run>/model/model_best.pth

Prints Acc@0.25 / Acc@0.5 / AccAll@0.25 / AccAll@0.5 over the val
loader and a per-batch progress line every 25 batches.
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
from pointcept.datasets.locate3d_collate import locate3d_collate_fn
import pointcept.datasets  # noqa: F401  (registers ScanNet/ARKit/ScanNetPP)


def _iou_3d(a, b):
    lt = torch.maximum(a[:3], b[:3])
    rb = torch.minimum(a[3:], b[3:])
    wh = (rb - lt).clamp_min(0.0)
    inter = wh[0] * wh[1] * wh[2]
    vol_a = (a[3:] - a[:3]).clamp_min(0.0).prod()
    vol_b = (b[3:] - b[:3]).clamp_min(0.0).prod()
    union = vol_a + vol_b - inter
    return float((inter / union.clamp_min(1e-6)).item())


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
    if info.missing_keys:
        print(f"[load] sample missing: {info.missing_keys[:5]}")
    if info.unexpected_keys:
        print(f"[load] sample unexpected: {info.unexpected_keys[:5]}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-file", required=True)
    ap.add_argument("--weight", required=True)
    ap.add_argument("--iou-thresholds", default="0.25,0.5")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="val batch size; 1 keeps the same per-sample "
                         "metric semantics as training-time eval.")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--max-batches", type=int, default=None,
                    help="stop after N batches (smoke test). None=full.")
    args = ap.parse_args()

    iou_thresholds = [float(t) for t in args.iou_thresholds.split(",")]

    cfg = Config.fromfile(args.config_file)
    print(f"[cfg] loaded {args.config_file}")
    if not hasattr(cfg.data, "val"):
        raise RuntimeError(
            "config has no data.val; check the localize-* config "
            "templates that build data dict from env vars."
        )

    # Build val dataset (respects LOCATE3D_USE_ARKIT / LOCATE3D_USE_SCANNETPP
    # env vars baked into the config).
    val_dataset = build_dataset(cfg.data.val)
    print(f"[data] val dataset: {type(val_dataset).__name__} "
          f"len={len(val_dataset)}")
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=locate3d_collate_fn,
    )

    # Build model + load checkpoint.
    model = build_model(cfg.model).cuda().eval()
    _load_checkpoint(model, args.weight)

    total = 0
    hits = {t: 0 for t in iou_thresholds}
    total_all = 0
    hits_all = {t: 0 for t in iou_thresholds}
    skipped = 0

    n_batches = len(val_loader)
    if args.max_batches is not None:
        n_batches = min(n_batches, args.max_batches)
    print(f"[eval] iterating {n_batches} batches")

    for i, input_dict in enumerate(val_loader):
        if args.max_batches is not None and i >= args.max_batches:
            break
        try:
            for key in list(input_dict.keys()):
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output_dict = model(input_dict)

            pred_boxes_per = output_dict.get("pred_boxes_per_entity", None)
            if pred_boxes_per is None:
                skipped += 1
                continue

            boxes_list = input_dict["boxes_xyzxyz"]
            primary_ids = input_dict.get(
                "primary_object_id", [0] * len(boxes_list)
            )

            for b in range(len(boxes_list)):
                pred_b = pred_boxes_per[b]
                if pred_b is None or pred_b.shape[0] == 0:
                    continue
                gt_boxes = boxes_list[b].to(pred_b.device)
                pid = primary_ids[b] if b < len(primary_ids) else 0
                if isinstance(pid, torch.Tensor):
                    primary = int(pid.flatten()[0].item())
                else:
                    primary = int(pid)
                G = int(gt_boxes.shape[0])

                K = min(pred_b.shape[0], G)
                for g in range(K):
                    iou = _iou_3d(pred_b[g], gt_boxes[g])
                    total_all += 1
                    for t in iou_thresholds:
                        if iou >= t:
                            hits_all[t] += 1

                if primary >= G or primary >= pred_b.shape[0]:
                    continue
                iou_p = _iou_3d(pred_b[primary], gt_boxes[primary])
                total += 1
                for t in iou_thresholds:
                    if iou_p >= t:
                        hits[t] += 1
        except Exception as e:
            print(f"[skip] batch {i}: {type(e).__name__}: {e}")
            skipped += 1
            continue

        if (i + 1) % 25 == 0 or i + 1 == n_batches:
            running = {
                f"Acc@{t:g}": hits[t] / max(total, 1)
                for t in iou_thresholds
            }
            running.update({
                f"AccAll@{t:g}": hits_all[t] / max(total_all, 1)
                for t in iou_thresholds
            })
            print(
                f"[{i+1}/{n_batches}] " +
                " / ".join(f"{k}: {v:.4f}" for k, v in running.items())
            )

    print()
    print("=" * 60)
    print(f"FINAL (primary_N={total}, all_N={total_all}, "
          f"skipped={skipped})")
    print("=" * 60)
    for t in iou_thresholds:
        print(f"  Acc@{t:g}     = {hits[t] / max(total, 1):.4f}")
    for t in iou_thresholds:
        print(f"  AccAll@{t:g}  = {hits_all[t] / max(total_all, 1):.4f}")


if __name__ == "__main__":
    main()
