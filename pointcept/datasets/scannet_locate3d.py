"""
ScanNet / ScanNet++ dataset adapters for the Locate-3D referring-expression
task.

Both datasets follow the SAME Locate-3D annotation schema as ARKitScenes
(``object_id``, ``entities``, ``token``, ``description``, ``scene_id``,
``scene_dataset``) EXCEPT that they do not ship ``gt_boxes`` directly.
Ground-truth masks are derived from the scene's per-point ``instance``
array (Pointcept's preprocessed layout writes this as ``instance.npy``),
with ``mask_g = (instance == object_id_g)``. The corresponding bounding
box is the AABB of the masked points.

Because GridSample (and any other index-valid transform) subsamples the
point cloud, we can't compute the mask in ``get_data()`` -- the mask
indices would stop matching coord after augmentation. Instead, we:

1. Leave raw ``instance`` in the data_dict (it's already in Pointcept's
   default ``index_valid_keys`` so GridSample updates it in lockstep
   with coord).
2. Store the caption's ordered ``object_ids`` on the side.
3. In ``__getitem__``, after the full transform pipeline runs, we
   derive ``point_masks`` by comparing the NOW-subsampled ``instance``
   tensor against ``object_ids``. The resulting mask is guaranteed to
   line up with ``coord`` and ``feat`` seen by the model.

Path layout expected (Pointcept convention):

    data_root/<split>/<scene_id>/{coord,color,normal,instance}.npy

where ``<split>`` is ``train`` / ``val`` for ScanNet, or whichever split
directories the user has on disk for ScanNet++.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from collections.abc import Sequence

from pointcept.utils.logger import get_root_logger
from .builder import DATASETS
from .transform import Compose
from .arkitscenes_locate3d import (
    _load_clip_tokenizer,
    _char_spans_for_tokens,
)


# --- shared helpers --------------------------------------------------------

def _build_positive_map_from_entities(
    description, token_words, entities, object_ids, max_tokens
):
    """(G, T) float mask: 1 on tokens that refer to each object_id.

    Mirrors ARKitScenesLocate3DDataset._build_positive_map but exposed as a
    module-level function so the ScanNet/ScanNet++ adapters can reuse it
    without instantiating the ARKit class.
    """
    tokenizer = _load_clip_tokenizer()
    enc = tokenizer(
        description,
        return_offsets_mapping=True,
        padding="max_length",
        truncation=True,
        max_length=max_tokens,
        return_tensors="np",
    )
    offset_mapping = enc["offset_mapping"][0]  # (max_tokens, 2)

    char_spans_per_object = {oid: [] for oid in object_ids}
    for token_idx_list, labels in entities:
        for label in labels:
            try:
                oid = int(str(label).split("_")[0])
            except ValueError:
                continue
            if oid not in char_spans_per_object:
                continue
            for s, e in _char_spans_for_tokens(description, token_words, token_idx_list):
                char_spans_per_object[oid].append((s, e))

    G = len(object_ids)
    pos_map = np.zeros((G, max_tokens), dtype=np.float32)
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


def _load_scene_generic(scene_dir, required=("coord", "instance"), optional=("color", "normal")):
    out = {}
    missing = []
    for name in required:
        p = os.path.join(scene_dir, f"{name}.npy")
        if not os.path.isfile(p):
            missing.append(name)
            continue
        out[name] = np.load(p)
    if missing:
        raise FileNotFoundError(
            f"{scene_dir}: missing required assets {missing}; "
            f"dir contents: {sorted(os.listdir(scene_dir))[:10]}"
        )
    for name in optional:
        p = os.path.join(scene_dir, f"{name}.npy")
        if os.path.isfile(p):
            out[name] = np.load(p)
    return out


# --- base adapter shared by ScanNet / ScanNetPP ---------------------------

class _BaseScanNetFamilyLocate3DDataset(Dataset):
    """Common logic for both ScanNet and ScanNet++ Locate-3D adapters.

    Subclasses set ``DATASET_TAG`` (used for filtering the annotation JSON
    by ``scene_dataset`` field) and ``DEFAULT_SPLIT_DIRS`` (the sub-
    directories under ``data_root`` that contain per-scene folders).
    """

    DATASET_TAG = ""          # e.g. "ScanNet", "ScanNetPP"
    DEFAULT_SPLIT_DIRS = ("train", "val")

    def __init__(
        self,
        annotation_file,
        data_root,
        transform=None,
        split=None,
        test_mode=False,
        loop=1,
        max_tokens=77,
        ignore_index=-1,
    ):
        super().__init__()
        self.annotation_file = annotation_file
        self.data_root = data_root
        self.transform = Compose(transform)
        if split is None:
            split = self.DEFAULT_SPLIT_DIRS
        self.split = (
            split if isinstance(split, Sequence) and not isinstance(split, str)
            else [split]
        )
        self.test_mode = test_mode
        self.loop = loop if not test_mode else 1
        self.max_tokens = max_tokens
        self.ignore_index = ignore_index

        with open(annotation_file, "r") as f:
            anns = json.load(f)

        # Filter annotations by this adapter's dataset tag (the same JSON
        # may contain rows from ScanNet + ScanNet++ + ARKitScenes).
        if self.DATASET_TAG:
            anns = [
                a for a in anns
                if str(a.get("scene_dataset", "")).lower() == self.DATASET_TAG.lower()
            ]

        self.scene_dirs = self._index_scenes()

        resolved = []
        dropped = []
        for a in anns:
            sid = a["scene_id"]
            p = self.scene_dirs.get(sid)
            if p is None:
                dropped.append(sid)
                continue
            if not os.path.isfile(os.path.join(p, "instance.npy")):
                dropped.append(sid)
                continue
            a = dict(a)
            a["_resolved_scene_dir"] = p
            resolved.append(a)
        self.anns = resolved

        logger = get_root_logger()
        logger.info(
            "{}: data_root={} splits={} -> indexed {} scene dirs, "
            "kept {} / {} annotations ({} dropped).".format(
                self.__class__.__name__, self.data_root, list(self.split),
                len(self.scene_dirs), len(self.anns), len(anns), len(dropped),
            )
        )
        if len(self.anns) == 0:
            raise RuntimeError(
                f"{self.__class__.__name__}: 0 annotations resolved. "
                f"Check data_root={self.data_root!r} and split={self.split!r}; "
                f"each scene must live at data_root/<split>/<scene_id>/ and "
                f"contain coord.npy + instance.npy."
            )

    def _index_scenes(self):
        scene_dirs = {}
        for split in self.split:
            root = os.path.join(self.data_root, split)
            if not os.path.isdir(root):
                continue
            for name in os.listdir(root):
                full = os.path.join(root, name)
                if os.path.isdir(full):
                    scene_dirs[name] = full
        return scene_dirs

    def __len__(self):
        return len(self.anns) * self.loop

    def get_data_name(self, idx):
        ann = self.anns[idx % len(self.anns)]
        return f"{ann['scene_id']}_{ann.get('primary_key', ann.get('ann_id', idx))}"

    def _ordered_object_ids(self, ann):
        """Union of object ids across the caption's entities, with the
        primary ``object_id`` placed first (so downstream code that uses
        ``primary_object_id=0`` is correct)."""
        oids = []
        for _, labels in ann.get("entities", []):
            for lab in labels:
                try:
                    oid = int(str(lab).split("_")[0])
                    if oid not in oids:
                        oids.append(oid)
                except ValueError:
                    continue
        primary = int(ann.get("object_id", oids[0] if oids else 0))
        if primary in oids:
            oids.remove(primary)
        oids = [primary] + oids
        if len(oids) == 0:
            oids = [0]
        return oids

    def get_data(self, idx):
        """Pre-transform sample. Returns coord, color, normal, instance
        (all per-point), caption, plus the bookkeeping needed to build
        point_masks / boxes_xyzxyz AFTER the transform pipeline has run.
        """
        ann = self.anns[idx % len(self.anns)]
        scene_dir = ann["_resolved_scene_dir"]
        scene = _load_scene_generic(scene_dir)

        coord = scene["coord"].astype(np.float32)
        color = scene.get(
            "color", np.zeros_like(coord, dtype=np.float32)
        ).astype(np.float32)
        normal = scene.get(
            "normal", np.zeros_like(coord, dtype=np.float32)
        ).astype(np.float32)
        instance = scene["instance"]
        if instance.ndim == 2:
            # ScanNet++ stores (N, K); take the first channel.
            instance = instance[:, 0]
        instance = instance.reshape(-1).astype(np.int64)

        oids = self._ordered_object_ids(ann)
        primary_idx = 0  # by construction: primary is first in oids

        # Center the coord cloud (same convention as ARKit adapter); no
        # box shift is needed here because boxes are derived POST-
        # transform from the mask.
        centroid = coord.mean(axis=0).astype(np.float32)
        coord = coord - centroid

        # positive_map is independent of the point cloud → compute now.
        pos_map = _build_positive_map_from_entities(
            ann["description"], ann["token"], ann["entities"], oids, self.max_tokens
        )

        return dict(
            coord=coord,
            color=color,
            normal=normal,
            instance=instance,                    # per-point; GridSample will subsample
            caption=ann["description"],
            primary_object_id=int(primary_idx),
            positive_map=pos_map.astype(np.float32),
            scene_id=ann["scene_id"],
            name=self.get_data_name(idx),
            # Side-channel: not a tensor, not a point attr. Used POST-
            # transform to derive point_masks + boxes_xyzxyz aligned with
            # the final subsampled point set.
            _locate3d_oids=oids,
        )

    def __getitem__(self, idx):
        data_dict = self.get_data(idx)
        oids = data_dict.pop("_locate3d_oids")
        data_dict = self.transform(data_dict)

        # After transforms: instance is now aligned with coord. Build per-
        # entity mask + AABB for downstream loss / eval.
        instance_t = data_dict.get("instance", None)
        coord_t = data_dict.get("coord", None)
        if instance_t is None or coord_t is None:
            # Collect may have dropped instance if the config forgot to
            # include it. Fall back: return empty mask/box; dataset
            # effectively skipped.
            G = len(oids)
            N = 0 if coord_t is None else int(coord_t.shape[0])
            data_dict["point_masks"] = torch.zeros((G, N), dtype=torch.bool)
            data_dict["boxes_xyzxyz"] = torch.zeros((G, 6), dtype=torch.float32)
            return data_dict

        inst = instance_t if isinstance(instance_t, torch.Tensor) else torch.as_tensor(instance_t)
        coord = coord_t if isinstance(coord_t, torch.Tensor) else torch.as_tensor(coord_t)
        oids_t = torch.as_tensor(oids, dtype=inst.dtype)  # (G,)

        masks = inst.unsqueeze(0) == oids_t.unsqueeze(1)  # (G, N) bool
        boxes = torch.zeros((masks.shape[0], 6), dtype=torch.float32)
        for g in range(masks.shape[0]):
            m = masks[g]
            if bool(m.any()):
                pts = coord[m]
                boxes[g, :3] = pts.min(dim=0).values
                boxes[g, 3:] = pts.max(dim=0).values
            else:
                # Degenerate; keep small finite box so IoU math doesn't NaN.
                boxes[g] = torch.tensor([0, 0, 0, 1e-3, 1e-3, 1e-3])

        data_dict["point_masks"] = masks
        data_dict["boxes_xyzxyz"] = boxes.float()
        return data_dict


# --- concrete adapters ----------------------------------------------------

@DATASETS.register_module()
class ScanNetLocate3DDataset(_BaseScanNetFamilyLocate3DDataset):
    DATASET_TAG = "ScanNet"
    DEFAULT_SPLIT_DIRS = ("train", "val")


@DATASETS.register_module()
class ScanNetPPLocate3DDataset(_BaseScanNetFamilyLocate3DDataset):
    DATASET_TAG = "ScanNetPP"
    DEFAULT_SPLIT_DIRS = ("train", "val")
