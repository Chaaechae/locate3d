"""
Diagnose why a split's samples are being dropped: inspect the *actual* paths stored in a
split json and report, for a few entries, whether each referenced path resolves on disk.

This is the fast way to tell apart the common failure modes:
  * the json's path prefix folder name differs from disk (e.g. 'structure3d' vs
    'structured3d'), so everything misses;
  * the paths are relative and the current working directory has no matching 'data/'
    (base_path / symlink problem);
  * coord and correspondence files are genuinely mismatched (out-of-range indices).

It prints each path verbatim, whether it is absolute, and whether it exists both as-is
(relative to the current cwd) and with an optional --base-path prepended.

Usage:
    python tools/diagnose_split_paths.py \
        --data-root /group-volume/3Ddataset/data/structured3d --split train --num 5

    # also test resolving relative paths under a base dir (e.g. your training cwd):
    python tools/diagnose_split_paths.py \
        --data-root /group-volume/3Ddataset/data/structured3d --split train \
        --base-path /group-volume/3Ddataset
"""

import argparse
import json
import os

try:
    import numpy as np
except Exception:  # numpy is optional; only needed for shape / index checks
    np = None


def resolve(path, base_path):
    """Return the first of [as-is, base_path/path] that exists, else None."""
    candidates = [path]
    if base_path and not os.path.isabs(path):
        candidates.append(os.path.join(base_path, path))
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True, help="dataset root (holds splits/).")
    ap.add_argument("--split", default="train")
    ap.add_argument("--num", type=int, default=5, help="entries to inspect in detail.")
    ap.add_argument(
        "--base-path",
        default=None,
        help="optional base dir to prepend to RELATIVE json paths when checking.",
    )
    args = ap.parse_args()

    print(f"cwd               : {os.getcwd()}")
    print(f"data-root         : {args.data_root}")
    print(f"base-path         : {args.base_path}")

    split_json = os.path.join(args.data_root, "splits", f"{args.split}.json")
    print(f"split json        : {split_json}  exists={os.path.isfile(split_json)}")
    if not os.path.isfile(split_json):
        raise SystemExit("Split json not found.")

    with open(split_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    names = list(data.keys())
    print(f"entries in split  : {len(names)}")
    if not names:
        raise SystemExit("Empty split.")

    # --- detailed look at the first --num entries -------------------------------------
    print("\n================ sample entries ================")
    for name in names[: args.num]:
        e = data[name]
        print(f"\n[{name}]  keys={list(e.keys())}")

        pc = e.get("pointclouds")
        print(f"  pointclouds (raw): {pc}")
        print(f"    is_abs={os.path.isabs(pc) if pc else '-'}  "
              f"exists_asis={os.path.exists(pc) if pc else '-'}")
        pc_real = resolve(pc, args.base_path) if pc else None
        print(f"    resolved        : {pc_real}")
        coord_n = None
        if pc_real and os.path.isdir(pc_real):
            assets = sorted(os.listdir(pc_real))
            print(f"    dir contents    : {assets[:12]}{' ...' if len(assets) > 12 else ''}")
            coord_path = os.path.join(pc_real, "coord.npy")
            if np is not None and os.path.isfile(coord_path):
                try:
                    coord_n = int(np.load(coord_path, mmap_mode="r").shape[0])
                    print(f"    coord.npy points: {coord_n}")
                except Exception as ex:
                    print(f"    coord.npy load failed: {ex}")

        imgs = e.get("images", []) or []
        img_ok = sum(1 for p in imgs if resolve(p, args.base_path))
        print(f"  images: {img_ok}/{len(imgs)} resolve")
        if imgs:
            print(f"    first (raw)     : {imgs[0]}")
            print(f"    first resolved  : {resolve(imgs[0], args.base_path)}")

        corrs = e.get("correspondences", []) or []
        corr_ok = sum(1 for p in corrs if resolve(p, args.base_path))
        print(f"  correspondences: {corr_ok}/{len(corrs)} resolve")
        if corrs:
            print(f"    first (raw)     : {corrs[0]}")
            cr = resolve(corrs[0], args.base_path)
            print(f"    first resolved  : {cr}")
            if np is not None and cr and os.path.isfile(cr):
                try:
                    ci = np.load(cr)
                    last = ci[:, -1]
                    print(f"    corr shape={ci.shape}  point-index "
                          f"min={last.min():.0f} max={last.max():.0f}"
                          + (f"  coord_n={coord_n}  "
                             f"OUT_OF_RANGE={'YES' if (coord_n is not None and last.max() >= coord_n) else 'no'}"
                             if coord_n is not None else ""))
                except Exception as ex:
                    print(f"    corr load failed: {ex}")

    # --- aggregate over the whole split ----------------------------------------------
    print("\n================ aggregate over split ================")
    n_pc_miss = n_img_any = n_img_all_miss = n_corr_all_miss = 0
    checked = 0
    for name in names:
        e = data[name]
        pc = e.get("pointclouds")
        if not pc or not resolve(pc, args.base_path):
            n_pc_miss += 1
        imgs = e.get("images", []) or []
        corrs = e.get("correspondences", []) or []
        if imgs:
            n_img_any += 1
            if not any(resolve(p, args.base_path) for p in imgs):
                n_img_all_miss += 1
        if corrs and not any(resolve(p, args.base_path) for p in corrs):
            n_corr_all_miss += 1
        checked += 1
    print(f"entries checked            : {checked}")
    print(f"pointclouds dir missing    : {n_pc_miss}")
    print(f"entries w/ images          : {n_img_any}")
    print(f"  ... all images missing   : {n_img_all_miss}")
    print(f"correspondences all missing: {n_corr_all_miss}")

    if n_pc_miss == checked:
        print("\n=> ALL pointclouds missing: the json path prefix does not match disk. "
              "Compare the 'pointclouds (raw)' string above to the real folder name "
              "(e.g. 'structure3d' vs 'structured3d'), or set --base-path / fix the "
              "symlink so the relative prefix resolves.")
    elif n_img_all_miss == n_img_any and n_img_any > 0:
        print("\n=> pointclouds resolve but ALL images miss: the images/ subtree prefix "
              "differs from disk. Compare 'images first (raw)' to the real layout.")


if __name__ == "__main__":
    main()
