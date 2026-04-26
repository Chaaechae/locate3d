"""
Evaluate Meta's official Locate-3D model on val_arkitscenes /
val_scannet / val_scannetpp using the SAME Acc@0.25 / Acc@0.5 +
AccAll@0.25 / AccAll@0.5 metrics our in-house evaluators report, so
the numbers are directly comparable to our 0h / 0i / 0j runs.

Pre-requisites
--------------
1. Meta's locate-3d repo is checked out at ``locate-3d/`` in this repo.
2. The safetensors checkpoint at ``--weight`` (default
   ``/group-volume/locate-3d/model.safetensors``).
3. Meta's preprocessed featurized point-cloud cache for whichever
   datasets you want to evaluate. Per Meta's README:

       python -m preprocessing.run_preprocessing \\
           --l3dd_annotations_fpath locate3d_data/dataset/val_<dataset>.json \\
           --scannet_data_dir   <SCANNET_DIR> \\
           --scannetpp_data_dir <SCANNETPP_DIR> \\
           --arkitscenes_data_dir <ARKIT_DIR>

   The cache is written under ``locate-3d/cache/`` by default.
4. Original raw data dirs for ScanNet / ScanNet++ / ARKitScenes (the
   featurized cache lookup needs to resolve scene_id -> path).

Usage
-----
    python tools/eval_locate3d_baseline.py \\
        --annotations locate-3d/locate3d_data/val_scannet.json \\
        --scannet-data-dir   /path/to/scannet/raw \\
        --scannetpp-data-dir /path/to/scannetpp/raw \\
        --arkitscenes-data-dir /path/to/arkitscenes/raw \\
        --weight /group-volume/locate-3d/model.safetensors \\
        --config locate-3d/config/locate_3d.yaml \\
        --cache-path locate-3d/cache \\
        --output eval_locate3d_baseline_scannet.json

Repeat with ``--annotations val_scannetpp.json`` etc. for the other
splits. Compare the printed metrics against the corresponding
val_extras logged by Locate3DSegDetectorEvaluator in our runs.

Reads
-----
- locate-3d/models/locate_3d.py     (Locate3D model class)
- locate-3d/locate3d_data/locate3d_dataset.py (Locate3DDataset loader)
- locate-3d/config/locate_3d.yaml   (model config)
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import yaml

# Make Meta's locate-3d package importable.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOCATE3D = os.path.join(_REPO, "locate-3d")
if _LOCATE3D not in sys.path:
    sys.path.insert(0, _LOCATE3D)

# Make ``tools/_pytorch3d_shim.py`` importable + install the shim BEFORE
# any Meta module loads. ``models.encoder_3djepa`` imports
# ``pytorch3d.renderer.implicit.harmonic_embedding.HarmonicEmbedding`` at
# top level; if real pytorch3d is not installed (it requires a CUDA build
# toolchain that's painful to set up on the cluster), the shim provides a
# bit-exact replacement registered under the same import path.
_TOOLS = os.path.dirname(os.path.abspath(__file__))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)
from _pytorch3d_shim import install_shim as _install_pytorch3d_shim  # noqa: E402
_shim_installed = _install_pytorch3d_shim()
if _shim_installed:
    print("[shim] real pytorch3d not found; using local HarmonicEmbedding shim")


def _iou_3d_xyzxyz(a, b):
    """3D IoU between two xyzxyz boxes (numpy arrays of length 6)."""
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    lt = np.maximum(a[:3], b[:3])
    rb = np.minimum(a[3:], b[3:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = float(wh.prod())
    vol_a = float(np.clip(a[3:] - a[:3], 0.0, None).prod())
    vol_b = float(np.clip(b[3:] - b[:3], 0.0, None).prod())
    union = vol_a + vol_b - inter + 1e-9
    return inter / union


def _xyzxyz_from_anything(box):
    """Accept (6,) xyzxyz or (3, 2) min/max box; return (6,) xyzxyz."""
    arr = np.asarray(box).astype(np.float32)
    if arr.shape == (6,):
        return arr
    if arr.shape == (3, 2):
        mins = np.minimum(arr[:, 0], arr[:, 1])
        maxs = np.maximum(arr[:, 0], arr[:, 1])
        return np.concatenate([mins, maxs])
    if arr.shape == (2, 3):
        mins = np.minimum(arr[0], arr[1])
        maxs = np.maximum(arr[0], arr[1])
        return np.concatenate([mins, maxs])
    raise ValueError(f"unrecognized box shape {arr.shape}")


def _resolve_oid_token_set(ann):
    """Return (ordered_oids, tokens_per_oid_dict).

    Mirrors the dataset adapters' convention: primary object_id first,
    others in first-seen order. tokens_per_oid_dict is OID -> set(int)
    of word indices into ann["token"].
    """
    token_words = ann.get("token", [])
    oids = []
    tokens_per_oid = {}
    for token_idx_list, labels in ann.get("entities", []):
        for lab in labels:
            try:
                oid = int(str(lab).split("_")[0])
            except ValueError:
                continue
            if oid not in tokens_per_oid:
                tokens_per_oid[oid] = set()
                oids.append(oid)
            for ti in token_idx_list:
                if 0 <= int(ti) < len(token_words):
                    tokens_per_oid[oid].add(int(ti))

    primary_oid = int(ann.get("object_id", oids[0] if oids else 0))
    if primary_oid in oids:
        oids.remove(primary_oid)
    oids = [primary_oid] + oids
    if not oids:
        oids = [0]
        tokens_per_oid[0] = set()
    return oids, tokens_per_oid


def _word_text_to_token_indices(tokenizer, utterance, token_words):
    """For each WORD index ``wi`` (referencing ``token_words``), figure
    out which CLIP TOKEN positions cover that word in the tokenizer's
    encoding of ``utterance``. Returns a dict {wi: set(int)}.

    Meta's prediction's ``tokens_assigned`` is a list of CLIP token
    positions, but the annotation's ``entities`` is a list of WORD
    positions. To compare them, we project both into the same space
    (CLIP token positions). This function does the projection from word
    -> CLIP tokens via the tokenizer's offset_mapping.
    """
    # Reconstruct (start, end) char span of each word in utterance.
    word_spans = []
    cursor = 0
    for w in token_words:
        word_spans.append((cursor, cursor + len(w)))
        cursor += len(w) + 1  # +1 for the joining space

    enc = tokenizer(
        utterance,
        return_offsets_mapping=True,
        padding="max_length",
        truncation=True,
        max_length=77,
        return_tensors="np",
    )
    offset_mapping = enc["offset_mapping"][0]  # (T, 2)

    word_to_clip = {}
    for wi, (ws, we) in enumerate(word_spans):
        toks = set()
        for ti, (ts, te) in enumerate(offset_mapping):
            if ts == 0 and te == 0:
                continue
            if ts < we and te > ws:
                toks.add(int(ti))
        word_to_clip[wi] = toks
    return word_to_clip


def _load_model(weight_path, config_path):
    """Build Meta's Locate3D model from a yaml config and load weights
    (either a .safetensors or a .pt with state_dict)."""
    from models.locate_3d import Locate3D

    with open(config_path, "r") as f:
        raw_cfg = yaml.safe_load(f)
    model = Locate3D(raw_cfg)

    if weight_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state = load_file(weight_path)
    else:
        state = torch.load(weight_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]

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
    # Meta's Locate3D.train() override does not return self, so .eval()
    # propagates None back to the caller. Mutate in-place instead of
    # reassigning so we keep the loaded module.
    model.cuda()
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", required=True,
                    help="path to val_<dataset>.json")
    ap.add_argument("--scannet-data-dir", default=None)
    ap.add_argument("--scannetpp-data-dir", default=None)
    ap.add_argument("--arkitscenes-data-dir", default=None)
    ap.add_argument("--weight", default="/group-volume/locate-3d/model.safetensors")
    ap.add_argument("--config", default="locate-3d/config/locate_3d.yaml")
    ap.add_argument("--cache-path", default="locate-3d/cache")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="evaluate only the first N samples (sanity check)")
    ap.add_argument("--downsample-pts", type=int, default=30000)
    ap.add_argument("--iou-thresholds", default="0.25,0.5")
    ap.add_argument("--output", default=None,
                    help="optional path to write per-sample + summary JSON")
    args = ap.parse_args()

    iou_thresholds = [float(t) for t in args.iou_thresholds.split(",")]

    # Build dataset
    from locate3d_data.locate3d_dataset import Locate3DDataset
    from models.locate_3d import downsample

    dataset = Locate3DDataset(
        annotations_fpath=args.annotations,
        return_featurized_pointcloud=True,
        scannet_data_dir=args.scannet_data_dir,
        scannetpp_data_dir=args.scannetpp_data_dir,
        arkitscenes_data_dir=args.arkitscenes_data_dir,
        cache_path=args.cache_path,
    )
    # Meta's annotation list lives at dataset.annos.
    annos = list(dataset.annos)
    n_total = len(annos)
    if args.max_samples is not None:
        n_total = min(n_total, args.max_samples)
    print(f"[dataset] {args.annotations}: {len(annos)} annotations; "
          f"evaluating first {n_total}")

    model = _load_model(args.weight, args.config)
    tokenizer = model.decoder.tokenizer

    # Counters
    total_primary = 0
    hits_primary = {t: 0 for t in iou_thresholds}
    total_all = 0
    hits_all = {t: 0 for t in iou_thresholds}

    per_sample_records = []

    t0 = time.time()
    for idx in range(n_total):
        try:
            data = dataset[idx]
        except Exception as e:
            print(f"[skip] sample {idx} ({annos[idx].get('scene_id')}): "
                  f"{type(e).__name__}: {e}")
            continue

        if "featurized_sensor_pointcloud" not in data:
            print(f"[skip] sample {idx}: no featurized pointcloud cache; "
                  "run Meta's preprocessing.run_preprocessing first")
            continue

        # Downsample for speed (Meta's example does the same).
        data["featurized_sensor_pointcloud"] = downsample(
            data["featurized_sensor_pointcloud"], args.downsample_pts
        )

        ann = annos[idx]
        utterance = data["lang_data"]["text_caption"]
        gt_boxes_raw = ann.get("gt_boxes", None)

        # Resolve entity ordering + WORD-index -> CLIP-token-index map.
        oids, tokens_per_oid = _resolve_oid_token_set(ann)
        word_to_clip = _word_text_to_token_indices(
            tokenizer, utterance, ann.get("token", [])
        )
        clip_tokens_per_oid = {
            oid: set().union(*(word_to_clip[wi] for wi in tokens_per_oid[oid]
                              if wi in word_to_clip))
            if tokens_per_oid[oid] else set()
            for oid in oids
        }

        # Resolve GT box per oid. Two cases:
        #   - ARKit: ann["gt_boxes"][oid] is (3, 2) min/max.
        #   - ScanNet / ScanNet++: no gt_boxes in JSON -- Locate3DDataset
        #     returns AABB derived from instance mask in
        #     ``data["lang_data"]["gt_boxes"]`` aligned to oids order.
        gt_boxes_xyzxyz = []
        for oid in oids:
            box = None
            if gt_boxes_raw is not None and oid < len(gt_boxes_raw):
                box = _xyzxyz_from_anything(gt_boxes_raw[oid])
            else:
                gt_arr = data["lang_data"].get("gt_boxes", None)
                if gt_arr is not None:
                    # Meta returns a torch tensor of shape (G, 3, 2).
                    gt_idx = list(data["lang_data"].get("object_ids", oids)).index(oid) \
                        if oid in list(data["lang_data"].get("object_ids", [])) \
                        else None
                    if gt_idx is not None and gt_idx < len(gt_arr):
                        box = _xyzxyz_from_anything(gt_arr[gt_idx].cpu().numpy())
            gt_boxes_xyzxyz.append(box)

        # Run inference
        try:
            with torch.no_grad():
                instances = model.inference(
                    data["featurized_sensor_pointcloud"], utterance
                )
        except Exception as e:
            print(f"[skip] inference failed for sample {idx}: "
                  f"{type(e).__name__}: {e}")
            continue

        # For each entity, find the predicted instance with maximal CLIP-
        # token overlap (strictly > 0). Tie-break by confidence.
        per_entity_best = {}  # oid -> (instance_idx, overlap, confidence)
        for oid in oids:
            target_clip_tokens = clip_tokens_per_oid[oid]
            best = (None, -1, -1.0)
            if not target_clip_tokens:
                per_entity_best[oid] = best
                continue
            for ii, inst in enumerate(instances):
                pred_clip_tokens = set(int(t) for t in inst["tokens_assigned"])
                overlap = len(target_clip_tokens & pred_clip_tokens)
                conf = float(inst.get("confidence", 0.0))
                if overlap > best[1] or (overlap == best[1] and conf > best[2]):
                    best = (ii, overlap, conf)
            per_entity_best[oid] = best

        # Compute IoU for primary + all entities
        per_entity_record = []
        primary_oid = oids[0]
        for oid in oids:
            gt_idx = oids.index(oid)
            gt_box = gt_boxes_xyzxyz[gt_idx]
            best_ii, best_overlap, best_conf = per_entity_best[oid]
            iou = 0.0
            if gt_box is not None and best_ii is not None and best_overlap > 0:
                pred_bbox = instances[best_ii]["bbox"].detach().cpu().numpy()
                pred_xyzxyz = _xyzxyz_from_anything(pred_bbox)
                iou = _iou_3d_xyzxyz(pred_xyzxyz, gt_box)

            per_entity_record.append(dict(
                oid=int(oid), iou=float(iou), match_overlap=int(best_overlap),
                match_conf=float(best_conf),
            ))
            total_all += 1
            for t in iou_thresholds:
                if iou >= t:
                    hits_all[t] += 1
            if oid == primary_oid:
                total_primary += 1
                for t in iou_thresholds:
                    if iou >= t:
                        hits_primary[t] += 1

        per_sample_records.append(dict(
            scene_id=ann.get("scene_id"),
            ann_id=ann.get("ann_id"),
            primary_oid=int(primary_oid),
            entities=per_entity_record,
            n_pred_instances=len(instances),
        ))

        if (idx + 1) % 25 == 0 or idx + 1 == n_total:
            dt = time.time() - t0
            metrics_so_far = {
                f"Acc@{t:g}": hits_primary[t] / max(total_primary, 1)
                for t in iou_thresholds
            }
            metrics_so_far.update({
                f"AccAll@{t:g}": hits_all[t] / max(total_all, 1)
                for t in iou_thresholds
            })
            ms_per = dt / (idx + 1) * 1000
            print(f"[{idx+1}/{n_total}] {ms_per:.1f} ms/sample : " +
                  " / ".join(f"{k}: {v:.4f}" for k, v in metrics_so_far.items()))

    summary = {
        f"Acc@{t:g}": hits_primary[t] / max(total_primary, 1)
        for t in iou_thresholds
    }
    summary.update({
        f"AccAll@{t:g}": hits_all[t] / max(total_all, 1)
        for t in iou_thresholds
    })
    summary["primary_N"] = total_primary
    summary["all_N"] = total_all

    print()
    print("=" * 60)
    print("FINAL")
    print("=" * 60)
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(
                {"summary": summary, "per_sample": per_sample_records},
                f,
                indent=2,
                default=float,
            )
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
