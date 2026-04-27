"""
Plotly visualization for Meta's open-weight Locate-3D model.

Mirror of ``tools/visualize_locate3d.py`` (which targets our 0f / 0h
Locate3DSegDetector checkpoints), but uses Meta's ``Locate3D`` model and
``Locate3DDataset`` instead of the pointcept pipeline. Useful side-by-
side comparison: same scenes/captions, same HTML layout, different
underlying model.

Usage::

    LOCATE3D_CLIP_PATH=/group-volume/CLIP/clip-vit-large-patch14 \\
    python tools/visualize_locate3d_baseline.py \\
        --annotations locate-3d/locate3d_data/val_scannet.json \\
        --scannet-data-dir /group-volume/Scannet/scans \\
        --cache-path /group-volume/ARKitscenes_cache_copy \\
        --weight /group-volume/locate-3d/model.safetensors \\
        --num-scenes 5 \\
        --output-dir viz_baseline_scannet/

One HTML per sample is written under ``--output-dir``. GT uses the
vivid neon palette + corner markers (matches ``visualize_locate3d.py``);
predicted boxes use the muted D3 palette.
"""

import argparse
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np
import torch

# Make Meta's locate-3d/ python sources importable.
_LOCATE3D = os.path.join(_REPO, "locate-3d")
if _LOCATE3D not in sys.path:
    sys.path.insert(0, _LOCATE3D)

# Bypass real pytorch3d (cluster has no CUDA build toolchain) before any
# Meta module touches ``models.encoder_3djepa``.
_TOOLS = os.path.dirname(os.path.abspath(__file__))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)
from _pytorch3d_shim import install_shim as _install_pytorch3d_shim  # noqa: E402
_install_pytorch3d_shim()

# Reuse the renderer + helpers (no pointcept deps) so both viz scripts
# produce identically-styled HTMLs.
from _locate3d_viz_common import _render_scene, _MUTED_PALETTE  # noqa: E402

# Reuse model load + token mapping from the baseline eval.
from eval_locate3d_baseline import (  # noqa: E402
    _load_model,
    _resolve_oid_token_set,
    _word_text_to_token_indices,
    _xyzxyz_from_anything,
)


def _build_psp(data, downsample_pts):
    """Same length-aware downsample we use in eval_locate3d_baseline.

    Returns the (cpu) featurized dict ready for inference and the
    (downsampled) coord / rgb numpy arrays for rendering.
    """
    psp = {
        k: (v.cpu() if torch.is_tensor(v) else v)
        for k, v in data["featurized_sensor_pointcloud"].items()
    }
    point_keys = [
        k for k in psp
        if k in ("points", "rgb", "features_clip", "features_dino")
    ]
    lengths = {k: psp[k].shape[0] for k in point_keys}
    N = min(lengths.values())
    for k in point_keys:
        if psp[k].shape[0] != N:
            psp[k] = psp[k][:N]
    if N > downsample_pts:
        indices = torch.randperm(N)[:downsample_pts]
        for k in point_keys:
            psp[k] = psp[k][indices]
    return psp


def _resolve_gt_boxes(ann, lang_data, oids):
    """Same resolution as eval_locate3d_baseline.

    Returns list[Optional[np.ndarray (6,)]] aligned to ``oids``.
    """
    gt_boxes_raw = ann.get("gt_boxes", None)
    ann_oids = ann.get("object_ids", []) or []
    gt_arr = lang_data.get("gt_boxes", None)
    out = []
    for oid in oids:
        box = None
        if gt_boxes_raw is not None and oid < len(gt_boxes_raw):
            raw = gt_boxes_raw[oid]
            if raw is not None:
                box = _xyzxyz_from_anything(raw)
        elif gt_arr is not None and oid in ann_oids:
            gi = ann_oids.index(oid)
            if gi < len(gt_arr):
                cand = gt_arr[gi]
                if torch.is_tensor(cand):
                    cand = cand.cpu().numpy()
                else:
                    cand = np.asarray(cand)
                if np.isfinite(cand).all():
                    box = _xyzxyz_from_anything(cand)
        out.append(box)
    return out


def _instances_to_per_entity_box(
    instances, oids, clip_tokens_per_oid
):
    """For each oid, pick the instance with max CLIP-token overlap
    (tie-break on confidence). Returns dict oid -> (bbox or None,
    instance_idx or None)."""
    result = {}
    for oid in oids:
        tgt = clip_tokens_per_oid.get(oid, set())
        best = (None, -1, -1.0)  # (idx, overlap, conf)
        if tgt:
            for ii, inst in enumerate(instances):
                pred_clip = set(int(t) for t in inst["tokens_assigned"])
                overlap = len(tgt & pred_clip)
                conf = float(inst.get("confidence", 0.0))
                if (overlap > best[1]
                        or (overlap == best[1] and conf > best[2])):
                    best = (ii, overlap, conf)
        if best[0] is None or best[1] <= 0:
            result[oid] = (None, None)
        else:
            inst = instances[best[0]]
            bbox = inst["bbox"].detach().cpu().numpy()
            result[oid] = (_xyzxyz_from_anything(bbox), best[0])
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--scannet-data-dir", default=None)
    ap.add_argument("--scannetpp-data-dir", default=None)
    ap.add_argument("--arkitscenes-data-dir", default=None)
    ap.add_argument("--cache-path", required=True)
    ap.add_argument("--weight", default="/group-volume/locate-3d/model.safetensors")
    ap.add_argument("--config", default="locate-3d/config/locate_3d.yaml")
    ap.add_argument("--num-scenes", type=int, default=5)
    ap.add_argument("--scene-ids", type=str, default=None,
                    help="comma-separated scene_ids to filter to "
                         "(takes priority over --num-scenes)")
    ap.add_argument("--max-samples-search", type=int, default=200,
                    help="how many annotations to scan when picking "
                         "scenes if --scene-ids is not given")
    ap.add_argument("--downsample-pts", type=int, default=30000)
    ap.add_argument("--output-dir", default="viz_baseline/")
    ap.add_argument("--pred-mode", default="both",
                    choices=("box", "point", "both"),
                    help="box: only AABB. point: only mask points "
                         "(sigmoid > --mask-threshold). both: AABB + "
                         "mask points overlaid.")
    ap.add_argument("--mask-threshold", type=float, default=0.5)
    ap.add_argument("--scene-point-size", type=float, default=2.2,
                    help="plotly marker size for the scene RGB cloud "
                         "(higher = scene more visible)")
    ap.add_argument("--scene-opacity", type=float, default=0.9,
                    help="opacity for scene RGB cloud (0..1)")
    ap.add_argument("--scene-max-points", type=int, default=120000,
                    help="cap rendered scene point count")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

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
    print(f"[dataset] {len(annos)} annotations")

    # Pick which sample indices to render.
    if args.scene_ids:
        wanted = set(s.strip() for s in args.scene_ids.split(",") if s.strip())
        picks = [
            i for i, a in enumerate(annos) if a.get("scene_id") in wanted
        ]
    else:
        seen = set()
        picks = []
        for i in range(min(args.max_samples_search, len(annos))):
            sid = annos[i].get("scene_id")
            if sid in seen:
                continue
            seen.add(sid)
            picks.append(i)
            if len(picks) >= args.num_scenes:
                break
    print(f"[picks] {len(picks)} samples: "
          f"{[annos[i].get('scene_id') for i in picks]}")

    model = _load_model(args.weight, args.config)
    tokenizer = model.decoder.tokenizer

    for idx in picks:
        try:
            data = dataset[idx]
        except Exception as e:
            print(f"[skip] sample {idx}: {type(e).__name__}: {e}")
            continue
        if "featurized_sensor_pointcloud" not in data:
            print(f"[skip] sample {idx}: no featurized cache")
            continue

        ann = annos[idx]
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
        gt_boxes = _resolve_gt_boxes(ann, data["lang_data"], oids)

        psp_cpu = _build_psp(data, args.downsample_pts)
        psp_cuda = {
            k: (v.cuda(non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in psp_cpu.items()
        }
        try:
            with torch.no_grad():
                instances = model.inference(psp_cuda, utterance)
        except Exception as e:
            print(f"[skip] inference {idx}: {type(e).__name__}: {e}")
            continue

        per_oid = _instances_to_per_entity_box(
            instances, oids, clip_tokens_per_oid
        )

        # Build per-entity arrays in oids order. Also build per-entity
        # mask logits aligned to the input points so the renderer can
        # draw the points where ``sigmoid(logit) > infer_threshold``.
        gt_box_list = []
        pred_box_list = []
        entity_names = []
        entity_tokens = []
        per_entity_mask_logits = []  # list of (N,) numpy or None per oid
        N_pts = psp_cpu["points"].shape[0]
        for oid in oids:
            gb = gt_boxes[oids.index(oid)]
            pb, ii = per_oid[oid]
            if gb is not None:
                gt_box_list.append(gb)
            if pb is not None:
                pred_box_list.append(pb)
            words = [
                ann["token"][wi] for wi in sorted(tokens_per_oid.get(oid, set()))
                if wi < len(ann.get("token", []))
            ]
            entity_names.append(f"oid_{oid}")
            entity_tokens.append(words)

            # Build the per-entity logit row.
            if ii is None:
                per_entity_mask_logits.append(np.full(N_pts, -10.0,
                                                      dtype=np.float32))
                continue
            mask = instances[ii].get("mask")
            if mask is None:
                per_entity_mask_logits.append(np.full(N_pts, -10.0,
                                                      dtype=np.float32))
                continue
            # Meta returns mask already passed through sigmoid. The
            # renderer expects logits and re-applies sigmoid; convert
            # back via logit transform with epsilon clip.
            m_np = mask.detach().cpu().numpy().reshape(-1).astype(np.float32)
            n_align = min(m_np.shape[0], N_pts)
            row = np.full(N_pts, -10.0, dtype=np.float32)
            p = np.clip(m_np[:n_align], 1e-6, 1.0 - 1e-6)
            row[:n_align] = np.log(p / (1.0 - p))
            per_entity_mask_logits.append(row)
        pred_logits = np.stack(per_entity_mask_logits, axis=0) \
            if per_entity_mask_logits else None

        # Caption coloring: word index → vivid pred palette color.
        caption_words = list(ann.get("token", []))
        word_color_map = [None] * len(caption_words)
        for gi, oid in enumerate(oids):
            color = _MUTED_PALETTE[gi % len(_MUTED_PALETTE)]
            for wi in tokens_per_oid.get(oid, set()):
                if 0 <= wi < len(caption_words):
                    word_color_map[wi] = color

        # Use the points the model actually saw.
        coord = psp_cpu["points"].numpy()
        if "rgb" in psp_cpu and torch.is_tensor(psp_cpu["rgb"]):
            color = psp_cpu["rgb"].numpy()
        else:
            color = np.full_like(coord, 0.5)

        out_html = os.path.join(
            args.output_dir,
            f"baseline_{ann.get('scene_id')}_ann{ann.get('ann_id')}.html",
        )
        draw_boxes = args.pred_mode in ("box", "both")
        draw_masks = args.pred_mode in ("point", "both")
        _render_scene(
            out_path=out_html,
            coord=coord,
            color=color,
            gt_boxes=gt_box_list,
            pred_boxes=pred_box_list,
            pred_logits=pred_logits if draw_masks else None,
            infer_threshold=args.mask_threshold,
            caption=utterance,
            scene_id=str(ann.get("scene_id")),
            primary_idx=0,
            entity_names=entity_names,
            entity_tokens=entity_tokens,
            caption_token_colormap=word_color_map,
            caption_word_list=caption_words,
            draw_masks=draw_masks,
            draw_boxes=draw_boxes,
            scene_point_size=args.scene_point_size,
            scene_opacity=args.scene_opacity,
            max_points=args.scene_max_points,
        )

    print(f"[done] HTMLs under {args.output_dir}")


if __name__ == "__main__":
    main()
