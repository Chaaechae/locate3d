"""
Visualize 0h SegDetector + EntityHead chain inference on a SINGLE
raw scene + free-form caption -- no annotation JSON, no entity field,
just a folder of .npy files and a caption string.

Use case: deploy the trained system on a brand-new dataset (or a user
query against an existing scene with arbitrary text), and confirm
visually that the chained pipeline picks the right object.

Expected scene layout::

    <scene-path>/
        coord.npy        (N, 3) float32 -- 3D point positions
        color.npy        (N, 3) uint8 or float32 -- RGB
        normal.npy       (N, 3) float32 -- per-point surface normal (optional)

Same format that the Concerto / Pointcept scannet preprocessing script
produces; if you have only ``coord.npy`` and ``color.npy``, normal is
filled with zeros (model still runs but normal-aware augmentations
become no-ops).

Usage::

    LOCATE3D_CLIP_PATH=/group-volume/CLIP/clip-vit-large-patch14 \\
    python tools/visualize_locate3d_raw.py \\
        --config-file configs/utonia/localize-utonia-v1m1-0h-combined.py \\
        --weight exp/<0h-run>/model/model_h.pth \\
        --entity-head exp/entity_head/run0/model_best.pth \\
        --scene-path /path/to/scene_folder/ \\
        --caption "the chair next to the table" \\
        --output viz_raw/scene0_query0.html

For multiple captions on the same scene, just rerun with different
``--caption`` / ``--output``.
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
from pointcept.datasets.transform import Compose

# Plotly rendering helpers (no pointcept deps inside).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _locate3d_viz_common import (
    _MUTED_PALETTE,
    _render_scene,
)


def _load_segdet_checkpoint(model, weight_path):
    ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    cleaned = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module."):]
        cleaned[k] = v
    info = model.load_state_dict(cleaned, strict=False)
    print(f"[load] SegDet missing={len(info.missing_keys)} "
          f"unexpected={len(info.unexpected_keys)}")
    return model


def _load_entity_head(weight_path):
    """Returns (entity_head, clip_text, clip_tok)."""
    from pointcept.models.locate_3d.entity_head import EntityHead
    from transformers import AutoTokenizer, CLIPTextModel

    eh_ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
    K = eh_ckpt.get("max_entities", 4)
    clip_path = eh_ckpt.get(
        "clip_path",
        os.environ.get("LOCATE3D_CLIP_PATH",
                       "openai/clip-vit-large-patch14"),
    )
    is_local = os.path.isdir(clip_path)
    print(f"[entity-head] loading from {weight_path}, "
          f"K={K} clip_path={clip_path}")
    clip_tok = AutoTokenizer.from_pretrained(
        clip_path, local_files_only=is_local
    )
    clip_text = CLIPTextModel.from_pretrained(
        clip_path, local_files_only=is_local
    ).cuda().eval()
    for p in clip_text.parameters():
        p.requires_grad = False
    entity_head = EntityHead(
        d_model=clip_text.config.hidden_size,
        max_entities=K,
        n_layers=eh_ckpt.get("args", {}).get("n_layers", 2),
    ).cuda().eval()
    info = entity_head.load_state_dict(
        eh_ckpt["state_dict"], strict=False
    )
    print(f"[entity-head] missing={len(info.missing_keys)} "
          f"unexpected={len(info.unexpected_keys)}")
    return entity_head, clip_text, clip_tok


def _load_raw_scene(scene_path):
    """Read coord.npy / color.npy / normal.npy from a folder."""
    coord = np.load(os.path.join(scene_path, "coord.npy")).astype(np.float32)
    color_path = os.path.join(scene_path, "color.npy")
    if os.path.isfile(color_path):
        color = np.load(color_path).astype(np.float32)
    else:
        color = np.full_like(coord, 128.0)  # grey if missing
    normal_path = os.path.join(scene_path, "normal.npy")
    if os.path.isfile(normal_path):
        normal = np.load(normal_path).astype(np.float32)
    else:
        normal = np.zeros_like(coord)
    return coord, color, normal


def _build_transform():
    """Same transform spec ScanNetLocate3DDataset uses at test time."""
    return Compose([
        dict(type="GridSample", grid_size=0.02, hash_type="fnv",
             mode="train", return_grid_coord=True),
        dict(type="NormalizeColor"),
        dict(type="ToTensor"),
        dict(
            type="Collect",
            keys=("coord", "grid_coord", "caption",
                  "primary_object_id", "scene_id", "name"),
            feat_keys=("coord", "color", "normal"),
        ),
    ])


def _predict_positive_map(entity_head, clip_text, clip_tok, caption):
    """Caption -> (G_actual, 77) positive_map via EntityHead."""
    enc = clip_tok(
        [caption], return_tensors="pt",
        padding="max_length", truncation=True, max_length=77,
    )
    input_ids = enc.input_ids.cuda()
    attention_mask = enc.attention_mask.cuda()
    with torch.no_grad():
        text_feats = clip_text(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state                                  # (1, 77, 768)
        pred = entity_head.predict_positive_map(
            text_feats, attention_mask=attention_mask
        )                                                    # (1, K, 77)
    row_active = pred.sum(dim=-1) > 0
    G = int(row_active[0].sum().item())
    if G == 0:
        return None, 0
    pred_filtered = pred[0][row_active[0]]                   # (G, 77)
    return pred_filtered.float(), G


def _entity_word_list(entity_head, clip_text, clip_tok, caption, positive_map):
    """For each predicted entity, list the WORDS it owns. Used to
    label the legend + color-tag caption."""
    # Tokenize again to get offset_mapping (word spans).
    enc = clip_tok(
        caption, return_tensors="pt",
        padding="max_length", truncation=True, max_length=77,
        return_offsets_mapping=True,
    )
    offsets = enc["offset_mapping"][0].tolist()
    # Each row of positive_map is (T,) — token indices belonging to that entity.
    G = positive_map.shape[0]
    entity_words = [[] for _ in range(G)]
    # Reconstruct caption char ranges -> word string lookup.
    # Simple approach: pick characters covered by each entity's tokens.
    for g in range(G):
        sel = positive_map[g].bool().cpu().numpy()
        spans = []
        for ti, (s, e) in enumerate(offsets):
            if sel[ti] and not (s == 0 and e == 0):
                spans.append((s, e))
        # Merge contiguous spans into substrings.
        if not spans:
            continue
        spans.sort()
        merged = [spans[0]]
        for s, e in spans[1:]:
            ps, pe = merged[-1]
            if s <= pe + 1:
                merged[-1] = (ps, max(pe, e))
            else:
                merged.append((s, e))
        words = [caption[s:e] for s, e in merged]
        entity_words[g] = words
    return entity_words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-file", required=True,
                    help="0h / 0j config used for the SegDetector")
    ap.add_argument("--weight", required=True,
                    help="0h SegDetector .pth checkpoint")
    ap.add_argument("--entity-head", required=True,
                    help="EntityHead .pth (required -- no annotation "
                         "JSON to fall back on)")
    ap.add_argument("--scene-path", required=True,
                    help="folder with coord.npy / color.npy / normal.npy")
    ap.add_argument("--caption", required=True,
                    help="free-form text query, e.g. "
                         "'the chair next to the desk'")
    ap.add_argument("--output", default="viz_raw.html",
                    help="output HTML path")
    ap.add_argument("--pred-mode", default="paint",
                    choices=("overlay", "paint"),
                    help="paint: replace mesh point color with entity "
                         "color where mask > threshold. overlay: extra "
                         "scatter trace on top.")
    ap.add_argument("--no-box", action="store_true",
                    help="skip pred bbox rendering")
    ap.add_argument("--infer-threshold", type=float, default=0.5)
    ap.add_argument("--scene-point-size", type=float, default=2.2)
    ap.add_argument("--scene-opacity", type=float, default=0.9)
    ap.add_argument("--scene-max-points", type=int, default=120000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # -- load scene --
    print(f"[scene] reading {args.scene_path}")
    coord_np, color_np, normal_np = _load_raw_scene(args.scene_path)
    print(f"[scene] N_raw={coord_np.shape[0]}")

    sample = {
        "coord": coord_np,
        "color": color_np,
        "normal": normal_np,
        "caption": args.caption,
        "scene_id": os.path.basename(os.path.normpath(args.scene_path)),
        "name": os.path.basename(os.path.normpath(args.scene_path)),
        "primary_object_id": 0,
    }
    transform = _build_transform()
    data_dict = transform(sample)

    # Collate to batch size 1 (mirror locate3d_collate behavior for
    # the keys SegDetector actually reads).
    coord = data_dict["coord"]
    feat = data_dict["feat"]
    grid_coord = data_dict["grid_coord"]
    offset = torch.tensor([coord.shape[0]], dtype=torch.long)
    batch_gpu = {
        "coord": coord.cuda(),
        "feat": feat.cuda(),
        "grid_coord": grid_coord.cuda(),
        "offset": offset.cuda(),
        "caption": [args.caption],
        "scene_id": [sample["scene_id"]],
        "name": [sample["name"]],
        "primary_object_id": [0],
    }
    print(f"[scene] N_after_grid_sample={coord.shape[0]}")

    # -- load entity head + segdet --
    entity_head, clip_text, clip_tok = _load_entity_head(args.entity_head)
    cfg = Config.fromfile(args.config_file)
    cfg.weight = args.weight
    model = build_model(cfg.model)
    if model.__class__.__name__ != "Locate3DSegDetector":
        raise RuntimeError(
            f"expected Locate3DSegDetector, got "
            f"{model.__class__.__name__}"
        )
    _load_segdet_checkpoint(model, args.weight)
    model = model.cuda().eval()
    # Disable eval-time point subsampling for clean paint.
    for attr in ("max_points_eval", "max_points_train"):
        if hasattr(model, attr):
            setattr(model, attr, None)

    # -- predict positive_map from caption --
    pos_map, G = _predict_positive_map(
        entity_head, clip_text, clip_tok, args.caption
    )
    if G == 0:
        print(f"[entity-head] predicted 0 entities for caption "
              f"{args.caption!r}; aborting.")
        return
    print(f"[entity-head] predicted G={G} entities")
    pos_map = pos_map.cuda()
    batch_gpu["positive_map"] = [pos_map]

    # -- run SegDetector --
    with torch.no_grad():
        out = model(batch_gpu)
    pred_boxes_per = out.get("pred_boxes_per_entity", None)
    if pred_boxes_per is None or pred_boxes_per[0].shape[0] == 0:
        print("[skip] SegDetector returned no boxes; aborting.")
        return
    pred_boxes = pred_boxes_per[0].float().cpu().numpy()
    pred_logits_per = out.get(
        "pred_logits_full_per_entity",
        out.get("pred_logits_per_entity", None),
    )
    pred_logits = (
        pred_logits_per[0].float().cpu().numpy()
        if (pred_logits_per is not None and pred_logits_per[0] is not None)
        else None
    )

    coord_np = coord.cpu().numpy()
    # color was normalized to [0, 1] inside NormalizeColor; renderer
    # checks max() > 1.5 to autoscale, but feat already contains the
    # /255 color in slots [3:6]. Easier: re-read raw color and align
    # to grid-sampled coord by KD-tree NN. But grid_sample preserves
    # ``coord`` rows that are unique voxel centers -- which are the
    # exact ones in feat. So feat[:, 3:6] is already the color we want.
    color_np_post = feat[:, 3:6].cpu().numpy()

    # Per-entity paint mask from full-coord logits + threshold.
    pred_paint_masks = None
    if pred_logits is not None and args.pred_mode == "paint":
        pred_paint_masks = []
        for g in range(pred_logits.shape[0]):
            prob = 1.0 / (1.0 + np.exp(-pred_logits[g]))
            pred_paint_masks.append(prob > args.infer_threshold)
        # Quick diagnostic so the user knows the model fired on the caption.
        for g in range(pred_logits.shape[0]):
            prob_g = 1.0 / (1.0 + np.exp(-pred_logits[g]))
            n_above = int((prob_g > args.infer_threshold).sum())
            print(f"[entity {g}] prob_max={float(prob_g.max()):.3f} "
                  f"n_above_{args.infer_threshold}={n_above}")

    # Caption coloring: highlight words per entity. Each entity gets
    # the muted palette color matching its render color.
    entity_words = _entity_word_list(
        entity_head, clip_text, clip_tok, args.caption,
        pos_map.float(),
    )
    caption_words = args.caption.split()
    word_color_map = [None] * len(caption_words)
    cum_char = 0
    word_spans = []
    for w in caption_words:
        word_spans.append((cum_char, cum_char + len(w)))
        cum_char += len(w) + 1
    for g, sub_words in enumerate(entity_words):
        color_hex = _MUTED_PALETTE[g % len(_MUTED_PALETTE)]
        for sub in sub_words:
            for wi, w in enumerate(caption_words):
                if sub.strip() and sub.strip() in w:
                    word_color_map[wi] = color_hex

    entity_names = [f"entity_{g}" for g in range(pred_logits.shape[0])] \
        if pred_logits is not None else []
    entity_tokens = entity_words if entity_words else None

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".",
                exist_ok=True)
    _render_scene(
        out_path=args.output,
        coord=coord_np,
        color=color_np_post,
        gt_boxes=[],
        pred_boxes=([] if args.no_box else pred_boxes),
        pred_logits=pred_logits if args.pred_mode == "overlay" else None,
        infer_threshold=args.infer_threshold,
        caption=args.caption,
        scene_id=sample["scene_id"],
        primary_idx=0,
        entity_names=entity_names,
        entity_tokens=entity_tokens,
        caption_token_colormap=word_color_map,
        caption_word_list=caption_words,
        draw_masks=(args.pred_mode == "overlay"),
        draw_boxes=not args.no_box,
        scene_point_size=args.scene_point_size,
        scene_opacity=args.scene_opacity,
        max_points=args.scene_max_points,
        pred_paint_masks=pred_paint_masks,
    )


if __name__ == "__main__":
    main()
