"""
ARKitScenes dataset adapter for the Locate-3D referring expression task.

Each sample is a (scene, caption, gt_boxes) triple taken from the Locate-3D
annotation JSONs (train_arkitscenes.json / val_arkitscenes.json). The scene's
raw 3D point cloud is read from the pre-processed Pointcept storage
(`data_root/{Training,Validation}/{scene_id}/{coord,color,normal}.npy`) and
augmented by the usual pointcept transforms.

The caption is also tokenized offline with the CLIP tokenizer so that a
``positive_map`` can be computed per ground-truth box - each row of the map
indicates which CLIP token positions refer to the corresponding object
according to the ``entities`` field of the annotation.
"""

import os
import json
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from collections.abc import Sequence

from pointcept.utils.logger import get_root_logger
from .builder import DATASETS
from .transform import Compose


def _load_clip_tokenizer():
    """Lazy-load the CLIP tokenizer shared across workers."""
    from transformers import AutoTokenizer

    if not hasattr(_load_clip_tokenizer, "_tok"):
        _load_clip_tokenizer._tok = AutoTokenizer.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
    return _load_clip_tokenizer._tok


def _char_spans_for_tokens(description, token_words, token_indices):
    """Return (char_start, char_end) for the substring corresponding to
    whitespace-tokenized token_indices in description."""
    spans = []
    # token_words was produced by description.split(" "); reconstruct offsets.
    offsets = []
    cur = 0
    for w in token_words:
        offsets.append((cur, cur + len(w)))
        cur += len(w) + 1  # +1 for the separating space

    # For each contiguous run of indices, emit a single span.
    if len(token_indices) == 0:
        return spans
    token_indices = sorted(set(int(i) for i in token_indices))
    run_start = token_indices[0]
    prev = token_indices[0]
    for i in token_indices[1:]:
        if i == prev + 1:
            prev = i
            continue
        spans.append((offsets[run_start][0], offsets[prev][1]))
        run_start = i
        prev = i
    spans.append((offsets[run_start][0], offsets[prev][1]))
    return spans


def _box_corners_to_xyzxyz(gt_boxes_entry):
    """Convert annotation-style box [[xmin,xmax],[ymin,ymax],[zmin,zmax]] to
    (xmin, ymin, zmin, xmax, ymax, zmax)."""
    arr = np.asarray(gt_boxes_entry, dtype=np.float32)  # (3, 2)
    mins = arr[:, 0]
    maxs = arr[:, 1]
    lo = np.minimum(mins, maxs)
    hi = np.maximum(mins, maxs)
    return np.concatenate([lo, hi], axis=0).astype(np.float32)


@DATASETS.register_module()
class ARKitScenesLocate3DDataset(Dataset):
    """Referring-expression dataset over ARKitScenes using Locate-3D labels."""

    PC_VALID_ASSETS = ("coord", "color", "normal")

    def __init__(
        self,
        annotation_file,
        data_root,
        transform=None,
        split=("Training", "Validation"),
        test_mode=False,
        loop=1,
        max_tokens=77,
        ignore_index=-1,
    ):
        super().__init__()
        self.annotation_file = annotation_file
        self.data_root = data_root
        self.transform = Compose(transform)
        self.split = split if isinstance(split, Sequence) and not isinstance(split, str) else [split]
        self.test_mode = test_mode
        self.loop = loop if not test_mode else 1
        self.max_tokens = max_tokens
        self.ignore_index = ignore_index

        with open(annotation_file, "r") as f:
            anns = json.load(f)

        self.scene_dirs = self._index_scenes()

        # Drop annotations for scenes we don't have on disk.
        self.anns = [a for a in anns if a["scene_id"] in self.scene_dirs]

        logger = get_root_logger()
        logger.info(
            "Locate3D ARKitScenes: {} annotations on {} scenes (dropped {}).".format(
                len(self.anns), len(self.scene_dirs), len(anns) - len(self.anns)
            )
        )

    def _index_scenes(self):
        scene_dirs = {}
        for split in self.split:
            for p in glob.glob(os.path.join(self.data_root, split, "*")):
                if os.path.isdir(p):
                    scene_dirs[os.path.basename(p)] = p
        return scene_dirs

    def __len__(self):
        return len(self.anns) * self.loop

    def get_data_name(self, idx):
        ann = self.anns[idx % len(self.anns)]
        return f"{ann['scene_id']}_{ann.get('primary_key', ann.get('ann_id', idx))}"

    def _load_scene(self, scene_id):
        scene_dir = self.scene_dirs[scene_id]
        out = {}
        for asset in self.PC_VALID_ASSETS:
            p = os.path.join(scene_dir, f"{asset}.npy")
            if os.path.exists(p):
                out[asset] = np.load(p)
        return out

    def _build_positive_map(self, description, token_words, entities, object_ids):
        """Return positive_map (G, max_tokens) float tensor."""
        tokenizer = _load_clip_tokenizer()
        enc = tokenizer(
            description,
            return_offsets_mapping=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_tokens,
            return_tensors="np",
        )
        offset_mapping = enc["offset_mapping"][0]  # (max_tokens, 2)

        char_spans_per_object = {oid: [] for oid in object_ids}
        for token_idx_list, labels in entities:
            for label in labels:
                # label like "0_object"
                try:
                    oid = int(str(label).split("_")[0])
                except ValueError:
                    continue
                if oid not in char_spans_per_object:
                    continue
                for s, e in _char_spans_for_tokens(description, token_words, token_idx_list):
                    char_spans_per_object[oid].append((s, e))

        G = len(object_ids)
        pos_map = np.zeros((G, self.max_tokens), dtype=np.float32)
        for g, oid in enumerate(object_ids):
            spans = char_spans_per_object.get(oid, [])
            if len(spans) == 0:
                continue
            for t, (tok_start, tok_end) in enumerate(offset_mapping):
                if tok_start == 0 and tok_end == 0:
                    continue
                for s, e in spans:
                    if tok_start < e and tok_end > s:
                        pos_map[g, t] = 1.0
                        break
        return pos_map

    def get_data(self, idx):
        ann = self.anns[idx % len(self.anns)]
        scene = self._load_scene(ann["scene_id"])
        coord = scene["coord"].astype(np.float32)
        color = scene.get("color", np.zeros_like(coord)).astype(np.float32)
        normal = scene.get("normal", np.zeros_like(coord)).astype(np.float32)

        description = ann["description"]
        token_words = ann["token"]
        entities = ann["entities"]
        gt_boxes_raw = ann["gt_boxes"]

        object_ids = list(range(len(gt_boxes_raw)))
        boxes_xyzxyz = np.stack(
            [_box_corners_to_xyzxyz(b) for b in gt_boxes_raw], axis=0
        )  # (G, 6)
        positive_map = self._build_positive_map(
            description, token_words, entities, object_ids
        )  # (G, T)

        return dict(
            coord=coord,
            color=color,
            normal=normal,
            # carried through the transform pipeline unchanged
            caption=description,
            primary_object_id=int(ann.get("object_id", 0)),
            boxes_xyzxyz=boxes_xyzxyz.astype(np.float32),
            positive_map=positive_map.astype(np.float32),
            scene_id=ann["scene_id"],
            name=self.get_data_name(idx),
        )

    def __getitem__(self, idx):
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        return data_dict
