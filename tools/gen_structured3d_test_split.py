"""
Generate splits/test.json for the Structured3D image-point dataset.

Structured3D ships only ``splits/train.json`` and ``splits/val.json``; the Utonia
pretrain configs also reference a ``test`` split. This tool scans the on-disk
``images/test/<scene>/<room>/`` tree and writes a ``test.json`` whose entries follow
the *exact* same schema and path convention as the existing ``val.json``.

Why template-driven (instead of hardcoding paths):
  The split json stores paths that are consumed verbatim by the dataset loader
  (``Image.open`` / ``np.load`` / ``os.listdir`` — they are NOT joined with data_root).
  So the only safe source of truth for the path prefix, the per-room sub-structure
  (image subdir, correspondence subdir, extensions) and the entry-key format is the
  existing ``val.json``. This script learns all of that from ``val.json`` and replays
  it over the ``test`` folder.

Schema of each entry (see pointcept/datasets/defaults.py:get_data):
    "<name>": {
        "pointclouds":     "<room dir>",              # dir holding coord/color/normal .npy
        "images":          ["<.../prsp/0.png>", ...], # ordered; images[i] <-> correspondences[i]
        "correspondences": ["<.../0.npy>", ...],      # ordered to match images
    }

Usage:
    # 1) First PROVE the conventions are right by reproducing val.json from disk:
    python tools/gen_structured3d_test_split.py \
        --root /group-volume/3Ddataset/data/structured3d --verify

    # 2) Then generate test.json:
    python tools/gen_structured3d_test_split.py \
        --root /group-volume/3Ddataset/data/structured3d

Run --verify first: if the scanned-val output does not match val.json byte-for-byte,
the conventions differ and test.json would be wrong — fix before trusting the output.
"""

import argparse
import glob
import json
import os
import re
import sys


def numeric_key(path):
    """Sort key that orders '0.png', '1.png', ..., '10.png' numerically (by stem int)."""
    stem = os.path.splitext(os.path.basename(path))[0]
    nums = re.findall(r"\d+", stem)
    return (int(nums[0]) if nums else -1, stem)


def learn_convention(val_json_path):
    """Infer path prefix, per-room sub-structure and key format from val.json."""
    with open(val_json_path, "r", encoding="utf-8") as f:
        val = json.load(f)
    if not val:
        raise SystemExit(f"{val_json_path} is empty; cannot learn the convention.")

    first_name = next(iter(val))
    entry = val[first_name]

    required = ("pointclouds", "images", "correspondences")
    for k in required:
        if k not in entry:
            raise SystemExit(
                f"val.json entry '{first_name}' is missing key '{k}'. "
                f"Found keys: {list(entry.keys())}"
            )
    extra = [k for k in entry.keys() if k not in required]
    if extra:
        print(
            f"WARNING: val.json entries carry extra keys {extra} that this tool does "
            f"not know how to regenerate; they will be omitted from test.json.",
            file=sys.stderr,
        )

    room_json = entry["pointclouds"].rstrip("/")
    marker = "/images/"
    if marker not in room_json:
        raise SystemExit(
            f"Unexpected pointclouds path (no '{marker}'): {room_json}"
        )
    idx = room_json.index(marker)
    json_prefix = room_json[:idx]  # e.g. "data/structure3d" or absolute root
    tail = room_json[idx + len(marker):]  # "<split>/<scene>/<room>"
    parts = tail.split("/")
    if len(parts) < 3:
        raise SystemExit(f"Cannot parse <split>/<scene>/<room> from: {room_json}")
    val_split, scene, room = parts[0], parts[1], parts[2]

    img0 = entry["images"][0]
    corr0 = entry["correspondences"][0]
    img_subdir = os.path.relpath(os.path.dirname(img0), room_json)
    corr_subdir = os.path.relpath(os.path.dirname(corr0), room_json)
    img_ext = os.path.splitext(img0)[1]
    corr_ext = os.path.splitext(corr0)[1]

    # Key format: turn the concrete name into a template by substituting scene/room.
    key_template = first_name.replace(scene, "{scene}").replace(room, "{room}")
    if key_template.format(scene=scene, room=room) != first_name:
        raise SystemExit(
            f"Could not derive a stable key template from name '{first_name}' "
            f"(scene='{scene}', room='{room}')."
        )

    conv = dict(
        json_prefix=json_prefix,
        val_split=val_split,
        img_subdir=img_subdir,
        corr_subdir=corr_subdir,
        img_ext=img_ext,
        corr_ext=corr_ext,
        key_template=key_template,
        key_order=[k for k in entry.keys() if k in required],
    )
    print("Learned convention from val.json:")
    for k, v in conv.items():
        print(f"  {k:14s}: {v}")
    return conv, val


def build_split(root, split_name, conv):
    """Scan <root>/images/<split_name> on disk and build the split dict."""
    images_dir = os.path.join(root, "images", split_name)
    if not os.path.isdir(images_dir):
        raise SystemExit(f"Directory not found: {images_dir}")

    out = {}
    n_rooms = n_skipped = 0
    for scene in sorted(os.listdir(images_dir)):
        scene_disk = os.path.join(images_dir, scene)
        if not os.path.isdir(scene_disk):
            continue
        for room in sorted(os.listdir(scene_disk)):
            room_disk = os.path.join(scene_disk, room)
            if not os.path.isdir(room_disk):
                continue

            img_glob = os.path.join(room_disk, conv["img_subdir"], "*" + conv["img_ext"])
            corr_glob = os.path.join(
                room_disk, conv["corr_subdir"], "*" + conv["corr_ext"]
            )
            imgs = sorted(glob.glob(img_glob), key=numeric_key)
            corrs = sorted(glob.glob(corr_glob), key=numeric_key)

            if not imgs:
                n_skipped += 1
                continue
            if len(imgs) != len(corrs):
                print(
                    f"WARNING: {scene}/{room} has {len(imgs)} images but "
                    f"{len(corrs)} correspondences; image[i]<->corr[i] alignment "
                    f"may be off.",
                    file=sys.stderr,
                )

            room_json = "{prefix}/images/{split}/{scene}/{room}".format(
                prefix=conv["json_prefix"], split=split_name, scene=scene, room=room
            )
            images_json = [
                "{room}/{sub}/{name}".format(
                    room=room_json, sub=conv["img_subdir"], name=os.path.basename(p)
                )
                for p in imgs
            ]
            corr_json = [
                "{room}/{sub}/{name}".format(
                    room=room_json, sub=conv["corr_subdir"], name=os.path.basename(p)
                )
                for p in corrs
            ]
            entry = {
                "pointclouds": room_json,
                "images": images_json,
                "correspondences": corr_json,
            }
            # Preserve key order exactly as val.json had it.
            ordered = {k: entry[k] for k in conv["key_order"]}
            key = conv["key_template"].format(scene=scene, room=room)
            out[key] = ordered
            n_rooms += 1

    print(
        f"Scanned split '{split_name}': {n_rooms} rooms, "
        f"{n_skipped} skipped (no images)."
    )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="/group-volume/3Ddataset/data/structured3d",
        help="Structured3D data root (contains images/ and splits/).",
    )
    parser.add_argument(
        "--val-json",
        default=None,
        help="Template split json (default: <root>/splits/val.json).",
    )
    parser.add_argument(
        "--split-name", default="test", help="Split to scan/generate (default: test)."
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output path (default: <root>/splits/<split-name>.json).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Regenerate the 'val' split from disk and diff against val.json to "
        "validate the inferred conventions. Writes nothing.",
    )
    args = parser.parse_args()

    val_json = args.val_json or os.path.join(args.root, "splits", "val.json")
    if not os.path.isfile(val_json):
        raise SystemExit(f"Template not found: {val_json}")

    conv, val_data = learn_convention(val_json)

    if args.verify:
        print("\n=== VERIFY: rebuilding 'val' from disk and comparing to val.json ===")
        rebuilt = build_split(args.root, conv["val_split"], conv)
        if rebuilt == val_data:
            print("VERIFY PASS: scanned val split matches val.json exactly. "
                  "Conventions are correct -> test.json will be trustworthy.")
            return
        # Detailed diff to help fixing.
        rk, vk = set(rebuilt), set(val_data)
        print(f"VERIFY FAIL: rebuilt {len(rebuilt)} entries vs val.json {len(val_data)}.")
        if rk - vk:
            print(f"  only in rebuilt (sample): {sorted(rk - vk)[:5]}")
        if vk - rk:
            print(f"  only in val.json (sample): {sorted(vk - rk)[:5]}")
        for name in sorted(rk & vk):
            if rebuilt[name] != val_data[name]:
                print(f"  first differing entry: {name}")
                for k in val_data[name]:
                    if rebuilt[name].get(k) != val_data[name].get(k):
                        print(f"    key '{k}':")
                        print(f"      val.json : {val_data[name].get(k)}")
                        print(f"      rebuilt  : {rebuilt[name].get(k)}")
                break
        raise SystemExit(1)

    out_path = args.out or os.path.join(args.root, "splits", f"{args.split_name}.json")
    if os.path.exists(out_path):
        raise SystemExit(
            f"Refusing to overwrite existing {out_path}. Remove it or pass --out."
        )
    result = build_split(args.root, args.split_name, conv)
    if not result:
        raise SystemExit(f"No rooms found under {args.root}/images/{args.split_name}.")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f)
    print(f"\nWrote {len(result)} entries -> {out_path}")
    print("TIP: run with --verify first if you have not already, to confirm the "
          "path conventions reproduce val.json exactly.")


if __name__ == "__main__":
    main()
