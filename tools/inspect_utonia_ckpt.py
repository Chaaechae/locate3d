"""
Quick standalone inspection of a Utonia pretrain checkpoint, without spinning
up the whole training harness. Answers the question "did the CheckpointLoader
actually load the encoder weights, or silently leave them at random?"

Usage::

    python tools/inspect_utonia_ckpt.py \\
        --ckpt /group-volume/utonia.pth \\
        --config configs/utonia/localize-utonia-v1m1-0c-arkitscenes.py

Prints:
- Top-level keys of the checkpoint (state_dict? epoch? optimizer?).
- First 10 state_dict keys as-is (raw).
- What those keys become after the CheckpointLoader rewrite
  (module.student.backbone -> module.backbone, then strip module. for ws=1).
- For the model described by the config: total backbone.* keys and how many
  would be covered by the rewritten ckpt keys (exact string match).

If the "backbone keys covered" count is 0, the Utonia encoder is not loading
and every downstream run has a random-init encoder.
"""

import argparse
import importlib.util
import os
import sys

import torch


def _load_cfg_module(path):
    spec = importlib.util.spec_from_file_location("cfg_module", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cfg_module"] = mod
    spec.loader.exec_module(mod)
    return mod


def _apply_checkpoint_loader_rewrite(raw_keys, world_size=1,
                                     keyword="module.student.backbone",
                                     replacement="module.backbone"):
    out = []
    for k in raw_keys:
        k2 = k if k.startswith("module.") else "module." + k
        if keyword in k2:
            k2 = k2.replace(keyword, replacement, 1)
        if world_size == 1:
            k2 = k2[7:]  # drop "module."
        out.append(k2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True,
                    help="localize-utonia config path; we only need it to "
                         "instantiate the model so we can list backbone keys")
    ap.add_argument("--world-size", type=int, default=1)
    ap.add_argument("--build-model", action="store_true",
                    help="Actually instantiate the model to enumerate "
                         "backbone keys. Without this, we'll just show the "
                         "ckpt side.")
    args = ap.parse_args()

    if not os.path.isfile(args.ckpt):
        print(f"ERROR: ckpt file not found: {args.ckpt}")
        sys.exit(1)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    print(f"=== ckpt file: {args.ckpt} ===")
    if isinstance(ckpt, dict):
        print(f"top-level keys: {list(ckpt.keys())}")
        sd = ckpt.get("state_dict", ckpt)
    else:
        sd = ckpt
    if not hasattr(sd, "keys"):
        print(f"ckpt is not a dict of tensors. type={type(sd)}")
        sys.exit(1)
    raw_keys = list(sd.keys())
    print(f"\ntotal state_dict keys: {len(raw_keys)}")
    print("first 10 raw keys:")
    for k in raw_keys[:10]:
        print(f"  {k}")
    if len(raw_keys) > 10:
        print("last 5 raw keys:")
        for k in raw_keys[-5:]:
            print(f"  {k}")

    rewritten = _apply_checkpoint_loader_rewrite(raw_keys, args.world_size)
    print("\nfirst 10 keys after CheckpointLoader rewrite:")
    for k in rewritten[:10]:
        print(f"  {k}")

    has_backbone_prefix = sum(1 for k in rewritten if k.startswith("backbone."))
    print(f"\n{has_backbone_prefix} / {len(rewritten)} rewritten keys "
          f"start with 'backbone.' (these would attempt to load into the model)")

    if not args.build_model:
        print("\n(skipping model build; pass --build-model to check EXACT overlap)")
        return

    # Build the model described by the config so we can list backbone.* keys.
    # This path requires pointcept + transformers + torch_scatter etc.
    try:
        cfg = _load_cfg_module(args.config)
    except Exception as e:
        print(f"Could not load config {args.config}: {e}")
        sys.exit(1)

    from pointcept.models.builder import build_model
    model = build_model(cfg.model)
    model_keys = list(model.state_dict().keys())
    backbone_keys = [k for k in model_keys if k.startswith("backbone.")]

    ckpt_set = set(rewritten)
    loaded = [k for k in backbone_keys if k in ckpt_set]
    missing = [k for k in backbone_keys if k not in ckpt_set]
    print(f"\n=== model: {args.config} ===")
    print(f"model state_dict total keys: {len(model_keys)}")
    print(f"model.backbone.* keys:        {len(backbone_keys)}")
    print(f"EXACT overlap (loaded):       {len(loaded)}")
    print(f"missing after rewrite:        {len(missing)}")
    if len(loaded) == 0:
        print("\nFATAL: no backbone keys would be loaded. "
              "Check that the ckpt's raw keys have the 'module.student.backbone' "
              "prefix that the CheckpointLoader keyword expects.")
    elif len(missing) > 0:
        print("\nexample missing keys (first 10):")
        for k in missing[:10]:
            print(f"  {k}")


if __name__ == "__main__":
    main()
