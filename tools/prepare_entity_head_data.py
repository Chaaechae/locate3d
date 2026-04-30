"""
Pre-tokenize Locate-3D referring annotations into token-level entity
labels for EntityHead training.

For each annotation we:
1. Reconstruct the caption (joining ``ann["token"]`` with spaces).
2. Resolve entity ordering: primary first (``ann["object_id"]``), then
   secondary entities in first-seen order. Same convention the
   downstream 0h decoder uses, so the EntityHead's predicted ordering
   matches what the decoder expects.
3. Build per-CLIP-token labels:
     -1 = padding / "no entity" (articles, prepositions, special tokens)
      0 = primary entity tokens
      1, 2, ... = secondary entity tokens
4. Cache to disk as a single .pt the trainer can consume directly.

Usage::

    LOCATE3D_CLIP_PATH=/group-volume/CLIP/clip-vit-large-patch14 \\
    python tools/prepare_entity_head_data.py \\
        --annotations \\
            locate-3d/locate3d_data/train_arkitscenes.json \\
            locate-3d/locate3d_data/train_scannet.json \\
            locate-3d/locate3d_data/train_scannetpp.json \\
        --output exp/entity_head/train.pt \\
        --max-entities 4

    python tools/prepare_entity_head_data.py \\
        --annotations \\
            locate-3d/locate3d_data/val_arkitscenes.json \\
            locate-3d/locate3d_data/val_scannet.json \\
            locate-3d/locate3d_data/val_scannetpp.json \\
        --output exp/entity_head/val.pt \\
        --max-entities 4
"""

import argparse
import json
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np
import torch


def _resolve_oid_token_set(ann):
    """Mirror tools/eval_locate3d_baseline.py's helper. Returns
    (ordered_oids, tokens_per_oid_dict).

    Primary first, others in first-seen order. tokens_per_oid_dict is
    OID -> set(int) of WORD indices into ann["token"].
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


def _word_to_clip_token_indices(tokenizer, utterance, token_words, max_length):
    """Map each WORD index (into ``token_words``) to the set of CLIP
    token positions that cover it, using the tokenizer's
    offset_mapping. Returns dict {word_idx: set(clip_token_idx)}.

    Matches tools/eval_locate3d_baseline.py's
    ``_word_text_to_token_indices`` exactly so the labels we generate
    here line up with the inference-time tokenization.
    """
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
        max_length=max_length,
        return_tensors="np",
    )
    offset_mapping = enc["offset_mapping"][0]  # (T, 2)
    word_to_clip = {}
    for wi, (ws, we) in enumerate(word_spans):
        toks = set()
        for ti, (ts, te) in enumerate(offset_mapping):
            if ts == 0 and te == 0:
                continue  # special tokens / padding
            if ts < we and te > ws:
                toks.add(int(ti))
        word_to_clip[wi] = toks
    return enc, word_to_clip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", nargs="+", required=True,
                    help="one or more annotation JSON files")
    ap.add_argument("--output", required=True,
                    help="output .pt path")
    ap.add_argument("--max-entities", type=int, default=4,
                    help="cap on G per caption. Annotations with more "
                         "are truncated to the first --max-entities "
                         "(primary kept).")
    ap.add_argument("--max-length", type=int, default=77,
                    help="CLIP tokenizer max_length (77 matches "
                         "openai/clip-vit-large-patch14)")
    ap.add_argument("--clip-path",
                    default=os.environ.get(
                        "LOCATE3D_CLIP_PATH",
                        "openai/clip-vit-large-patch14",
                    ),
                    help="CLIP tokenizer path. Defaults to env var "
                         "LOCATE3D_CLIP_PATH so the same tokenizer the "
                         "0h decoder uses is reproduced exactly.")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    print(f"[clip] loading tokenizer from {args.clip_path}")
    is_local = os.path.isdir(args.clip_path)
    tokenizer = AutoTokenizer.from_pretrained(
        args.clip_path, local_files_only=is_local
    )

    all_input_ids = []
    all_attention_mask = []
    all_token_labels = []
    all_n_entities = []
    all_meta = []

    n_kept = 0
    n_skipped = 0
    n_truncated = 0
    for ann_path in args.annotations:
        with open(ann_path) as f:
            anns = json.load(f)
        print(f"[load] {ann_path}: {len(anns)} annotations")
        for ann_idx, ann in enumerate(anns):
            token_words = ann.get("token", [])
            if not token_words:
                n_skipped += 1
                continue
            utterance = " ".join(token_words)

            # Resolve entity ordering.
            oids, tokens_per_oid = _resolve_oid_token_set(ann)
            G = len(oids)
            if G > args.max_entities:
                # Keep primary + first (max_entities-1) secondaries.
                oids = oids[: args.max_entities]
                G = args.max_entities
                n_truncated += 1

            # CLIP-tokenize + word -> CLIP-token map.
            enc, word_to_clip = _word_to_clip_token_indices(
                tokenizer, utterance, token_words, args.max_length
            )
            input_ids = enc["input_ids"][0]            # (T,)
            attention_mask = enc["attention_mask"][0]  # (T,)

            # Build per-CLIP-token entity label.
            labels = np.full(args.max_length, -1, dtype=np.int64)
            for entity_idx, oid in enumerate(oids):
                wi_set = tokens_per_oid.get(oid, set())
                for wi in wi_set:
                    for ti in word_to_clip.get(wi, set()):
                        if 0 <= ti < args.max_length:
                            labels[ti] = entity_idx

            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_token_labels.append(labels)
            all_n_entities.append(G)
            all_meta.append({
                "scene_id": ann.get("scene_id"),
                "ann_id": ann.get("ann_id"),
                "primary_oid": int(oids[0]),
                "oids": oids,
                "utterance": utterance,
            })
            n_kept += 1

    print(f"[stats] kept={n_kept} skipped={n_skipped} "
          f"truncated_to_max_entities={n_truncated}")

    payload = {
        "input_ids": torch.from_numpy(np.stack(all_input_ids)),
        "attention_mask": torch.from_numpy(np.stack(all_attention_mask)),
        "token_labels": torch.from_numpy(np.stack(all_token_labels)),
        "n_entities": torch.tensor(all_n_entities, dtype=torch.long),
        "meta": all_meta,
        "max_entities": args.max_entities,
        "max_length": args.max_length,
        "clip_path": args.clip_path,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".",
                exist_ok=True)
    torch.save(payload, args.output)
    print(f"[save] {args.output}  "
          f"input_ids.shape={tuple(payload['input_ids'].shape)}")

    # Quick sanity print: distribution of entity counts + label coverage.
    n_ent_tensor = payload["n_entities"]
    print(f"[sanity] n_entities histogram: "
          f"{torch.bincount(n_ent_tensor).tolist()}")
    label_tensor = payload["token_labels"]
    n_neg = int((label_tensor == -1).sum())
    n_pos = int((label_tensor >= 0).sum())
    print(f"[sanity] tokens: {n_pos} entity-tagged, {n_neg} no-entity "
          f"(no-entity ratio = {n_neg / (n_neg + n_pos):.3f})")


if __name__ == "__main__":
    main()
