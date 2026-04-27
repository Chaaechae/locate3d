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

# Plotly rendering helpers live in a separate module so the baseline
# (Meta open-weight) viz can reuse them without pulling pointcept.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _locate3d_viz_common import (  # noqa: E402
    _VIVID_PALETTE,
    _MUTED_PALETTE,
    _box_edges,
    _box_corners,
    _render_scene,
)




def _sample_to_gpu(batch):
    gpu = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            gpu[k] = v.cuda(non_blocking=True)
        else:
            gpu[k] = v
    return gpu


def _resolve_entity_meta(ann, num_boxes, primary_idx):
    """Figure out which caption tokens (words) correspond to each entity
    slot in the model output, and return (names, tokens, colormap).

    The Locate-3D annotation schema stores entities as
    ``[[token_indices, ["{oid}_label"]], ...]`` where ``token_indices``
    are whitespace-word indices into ``ann["token"]``.

    Both ARKitScenes and ScanNet / ScanNet++ adapters re-order entities
    so the PRIMARY object_id (``ann["object_id"]``) occupies slot 0;
    any remaining oids follow in first-seen order. We replicate that
    ordering here so the per-entity outputs from the model line up 1-1
    with ``entity_names[g]`` / ``entity_tokens[g]``.

    Parameters
    ----------
    ann : dict
        The raw annotation row from the dataset's ``anns`` list.
    num_boxes : int
        max(len(gt_boxes), len(pred_boxes)) -- output arrays are padded
        to this length with placeholder labels for any missing entities.
    primary_idx : int
        Index within the ordered oid list that holds the primary. Matches
        what the dataset's ``primary_object_id`` field means.

    Returns
    -------
    entity_names : list[str]
        Human-readable name per entity slot (uses ``ann["object_name"]``
        for the primary when available, otherwise the joined word tokens).
    entity_tokens : list[list[str]]
        Ordered list of caption words each entity was derived from.
    caption_token_colormap : list[str | None]
        One entry per word in ``ann["description"].split(" ")``: the hex
        vivid color of whichever entity claims that word, or None if no
        entity claims it. Used by ``_render_scene`` to color-highlight
        the caption HTML title so the caption word color matches the
        vivid GT box color in 3D.
    """
    token_words = ann.get("token", [])
    entities = ann.get("entities", [])
    description = ann.get("description", "")

    # 1) union of oids in the same order used by the adapters.
    oids = []
    for _, labels in entities:
        for lab in labels:
            try:
                oid = int(str(lab).split("_")[0])
            except ValueError:
                continue
            if oid not in oids:
                oids.append(oid)
    primary_oid = int(ann.get("object_id", oids[0] if oids else 0))
    if primary_oid in oids:
        oids.remove(primary_oid)
    oids = [primary_oid] + oids
    if len(oids) == 0:
        oids = [0]

    # 2) word-index sets per oid.
    tokens_per_oid = {oid: set() for oid in oids}
    for token_idx_list, labels in entities:
        for lab in labels:
            try:
                oid = int(str(lab).split("_")[0])
            except ValueError:
                continue
            if oid not in tokens_per_oid:
                continue
            for ti in token_idx_list:
                if 0 <= int(ti) < len(token_words):
                    tokens_per_oid[oid].add(int(ti))

    entity_names = []
    entity_tokens = []
    primary_name = ann.get("object_name", "") or ""
    for slot in range(num_boxes):
        if slot < len(oids):
            oid = oids[slot]
            words = [token_words[i] for i in sorted(tokens_per_oid.get(oid, set()))]
            entity_tokens.append(words)
            if slot == primary_idx and primary_name:
                entity_names.append(str(primary_name))
            elif len(words) > 0:
                entity_names.append(" ".join(words))
            else:
                entity_names.append(f"entity_{slot}")
        else:
            entity_names.append(f"entity_{slot}")
            entity_tokens.append([])

    # 3) per-word color assignment. We index by the annotation's
    # ``token`` word list (which the entities' indices refer to) --
    # NOT by ``description.split(" ")`` which can differ in
    # whitespace/punctuation. The render side rebuilds the caption
    # from this word list so index alignment is exact.
    colormap = [None] * len(token_words)
    for slot, oid in enumerate(oids):
        color = _VIVID_PALETTE[slot % len(_VIVID_PALETTE)]
        for ti in tokens_per_oid.get(oid, set()):
            if 0 <= ti < len(colormap):
                colormap[ti] = color
    return entity_names, entity_tokens, colormap, token_words


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
        G = max(gt_boxes.shape[0], pred_boxes.shape[0])

        # Resolve:
        #   entity_names[g]  = human-readable name per entity
        #   entity_tokens[g] = list of caption words that drove entity g
        #   caption_token_colormap[wi] = matched entity color for word wi
        # so the caption HTML can color-highlight "which word fed which box".
        entity_names, entity_tokens, cap_colormap, cap_words = _resolve_entity_meta(
            ann=ann,
            num_boxes=G,
            primary_idx=primary,
        )

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
            entity_tokens=entity_tokens,
            caption_token_colormap=cap_colormap,
            caption_word_list=cap_words,
            draw_masks=not args.no_mask,
        )

        del out, batch, batch_gpu
        torch.cuda.empty_cache()

    print(f"\nAll outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()
