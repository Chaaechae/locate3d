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

        # Flexible scene-id resolution: fall back to prefix / suffix /
        # stripped-prefix matching (e.g. "arkit_42445211" on disk vs.
        # "42445211" in the annotation JSON).
        resolved = []
        dropped = []
        missing_coord = []
        # Only accept scene dirs that actually contain a coord.npy file.
        # Cache per-dir validity to avoid hitting the FS repeatedly.
        valid_dir_cache = {}

        def _dir_has_coord(path):
            if path in valid_dir_cache:
                return valid_dir_cache[path]
            ok = os.path.isfile(os.path.join(path, "coord.npy"))
            valid_dir_cache[path] = ok
            return ok

        for a in anns:
            sid = a["scene_id"]
            match = self.scene_dirs.get(sid) or self._fuzzy_resolve(sid)
            if match is None:
                dropped.append(sid)
                continue
            if not _dir_has_coord(match):
                missing_coord.append(sid)
                continue
            a = dict(a)
            a["_resolved_scene_dir"] = match
            resolved.append(a)
        self.anns = resolved

        logger = get_root_logger()
        logger.info(
            "Locate3D ARKitScenes: data_root={} splits={} -> indexed {} scene dirs.".format(
                self.data_root, list(self.split), len(self.scene_dirs)
            )
        )
        if len(self.scene_dirs) == 0:
            sample_globs = sorted(
                os.listdir(self.data_root)
                if os.path.isdir(self.data_root)
                else []
            )[:5]
            logger.warning(
                "  No scene directories found under {} / {{{}}}. "
                "data_root contents (first 5): {}".format(
                    self.data_root, ",".join(self.split), sample_globs
                )
            )
        if len(anns) > 0:
            sample_scene_ids = [a["scene_id"] for a in anns[:5]]
            sample_on_disk = list(self.scene_dirs.keys())[:5]
            logger.info(
                "  annotation scene_id samples: {} ; on-disk samples: {}".format(
                    sample_scene_ids, sample_on_disk
                )
            )
        logger.info(
            "  -> kept {} / {} annotations "
            "({} unresolved scene_id, {} missing coord.npy).".format(
                len(self.anns), len(anns), len(dropped), len(missing_coord)
            )
        )
        if len(missing_coord) > 0:
            logger.warning(
                "  missing coord.npy samples: {}".format(missing_coord[:5])
            )
        if len(self.anns) == 0:
            raise RuntimeError(
                "ARKitScenesLocate3DDataset: 0 annotations resolved. "
                "Check `data_root` (={!r}) and `split` (={!r}). "
                "Each scene must be located at "
                "data_root/<split>/<scene_id>/{{coord,color,normal}}.npy.".format(
                    self.data_root, self.split
                )
            )

    def _index_scenes(self):
        scene_dirs = {}
        for split in self.split:
            split_root = os.path.join(self.data_root, split)
            if not os.path.isdir(split_root):
                continue
            for name in os.listdir(split_root):
                full = os.path.join(split_root, name)
                if os.path.isdir(full):
                    scene_dirs[name] = full
        return scene_dirs

    def _fuzzy_resolve(self, scene_id):
        """Try to find a matching dir by trailing-id match or prefix strip.
        Returns the resolved dir path or None."""
        if not hasattr(self, "_fuzzy_cache"):
            # precompute {digits-only-part: dir} and {name.endswith(scene_id): dir}
            by_suffix = {}
            by_digits = {}
            for name, p in self.scene_dirs.items():
                digits = "".join(ch for ch in name if ch.isdigit())
                by_digits.setdefault(digits, p)
                by_suffix.setdefault(name, p)
            self._fuzzy_cache = (by_suffix, by_digits)

        by_suffix, by_digits = self._fuzzy_cache
        digits = "".join(ch for ch in scene_id if ch.isdigit())
        # exact digits match
        if digits in by_digits:
            return by_digits[digits]
        # endswith match
        for name, p in by_suffix.items():
            if name.endswith(scene_id) or scene_id.endswith(name):
                return p
        return None

    def __len__(self):
        return len(self.anns) * self.loop

    def get_data_name(self, idx):
        ann = self.anns[idx % len(self.anns)]
        return f"{ann['scene_id']}_{ann.get('primary_key', ann.get('ann_id', idx))}"

    def _load_scene(self, ann):
        scene_dir = ann.get("_resolved_scene_dir") or self.scene_dirs[ann["scene_id"]]
        out = {}
        for asset in self.PC_VALID_ASSETS:
            p = os.path.join(scene_dir, f"{asset}.npy")
            if os.path.exists(p):
                out[asset] = np.load(p)
        if "coord" not in out:
            raise FileNotFoundError(
                f"ARKitScenesLocate3DDataset: scene {ann.get('scene_id')} at "
                f"{scene_dir} is missing coord.npy (contents: "
                f"{sorted(os.listdir(scene_dir))[:10]})"
            )
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
        scene = self._load_scene(ann)
        coord = scene["coord"].astype(np.float32)
        color = scene.get("color")
        color = (
            color.astype(np.float32)
            if color is not None
            else np.zeros_like(coord, dtype=np.float32)
        )
        normal = scene.get("normal")
        normal = (
            normal.astype(np.float32)
            if normal is not None
            else np.zeros_like(coord, dtype=np.float32)
        )

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

        # Center points and GT boxes jointly around the scene centroid so the
        # decoder regresses boxes in a near-zero frame. ARKitScenes raw coords
        # live in world meters with arbitrary per-scene offsets, which makes
        # the bbox-head regression hard to fit from random init. Subtracting
        # the same centroid from coord and boxes_xyzxyz keeps the two in the
        # same frame so IoU (evaluator/viz) stays correct.
        centroid = coord.mean(axis=0).astype(np.float32)  # (3,)
        coord = coord - centroid
        shift6 = np.concatenate([centroid, centroid], axis=0)
        boxes_xyzxyz = boxes_xyzxyz.astype(np.float32) - shift6

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
