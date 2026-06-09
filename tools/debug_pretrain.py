"""
Debug / sanity-check runner for Utonia pretrain (e.g. the DINOv3 distillation configs).

Goal: decide *early* — within a handful of iterations of the first epoch(s) — whether
training is wired up correctly and heading in the intended direction, without paying for
a full run. It reuses the real Trainer (model / dataloader / optimizer / scheduler /
hooks), so the schedulers (teacher temp, momentum, mask ratio) and the EMA teacher update
behave exactly as in a real run; only the number of iterations is capped.

What it checks:
  1. Config consistency  : crop divisible by patch_size, patch_h/w == crop // patch_size.
  2. 2D encoder wiring    : runs the frozen DINO(v2/v3) encoder on a real image batch and
                            verifies the token layout ([CLS, registers, patches]), that the
                            sliced patch-token count == patch_h * patch_w, and that the
                            feature dim == enc2d_head_in_channels (the patch_proj target).
  3. Frozen teacher       : the 2D encoder has no trainable parameters.
  4. Loss health          : every loss term is finite (no NaN/Inf) on every step.
  5. Gradient health      : the student receives finite, non-zero gradients.
  6. Direction            : reports the trend of the total loss and the 2D-distillation
                            loss (enc2d_loss) across the captured window (first vs. last
                            half mean). A clear downward enc2d_loss is the signal that the
                            3D backbone is starting to match the DINOv3 patch features.

Usage (single GPU, quickest):
    python tools/debug_pretrain.py \
        --config-file configs/utonia/pretrain-utonia-v1m1-0-base_stagev1_dinov3.py \
        --num-gpus 1 --max-iters 50 --epochs 1

Usage (exercise the real distributed / gloo + multi-H100 setup):
    python tools/debug_pretrain.py \
        --config-file configs/utonia/pretrain-utonia-v1m1-0-base_stagev2_dinov3.py \
        --num-gpus 4 --dist-backend gloo --max-iters 30 --epochs 1
"""

import os
import sys

import torch

from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pointcept.engines.launch import launch
from pointcept.engines.train import TRAINERS
from pointcept.utils.events import EventStorage
import pointcept.utils.comm as comm


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------
def _unwrap(model):
    """Return the underlying module whether or not it is DDP-wrapped."""
    return model.module if hasattr(model, "module") else model


def _log(msg):
    """Only the main process prints, to keep the debug output readable."""
    if comm.is_main_process():
        print(msg, flush=True)


def _verdict(ok):
    return "PASS" if ok else "FAIL"


# --------------------------------------------------------------------------------------
# individual checks
# --------------------------------------------------------------------------------------
def check_config_consistency(cfg):
    """Pure config-level checks for the patch grid (matters when moving 14 -> 16)."""
    crop_h, crop_w = cfg.crop_h, cfg.crop_w
    patch_size = cfg.patch_size
    patch_h = cfg.model.patch_h
    patch_w = cfg.model.patch_w

    ok = True
    _log("\n[1] Config consistency")
    _log(f"    crop=({crop_h}x{crop_w})  patch_size={patch_size}  "
         f"patch grid=({patch_h}x{patch_w})  tokens={patch_h * patch_w}")

    for name, crop in (("crop_h", crop_h), ("crop_w", crop_w)):
        if crop % patch_size != 0:
            _log(f"    {name}={crop} is NOT divisible by patch_size={patch_size}")
            ok = False
    if patch_h != crop_h // patch_size or patch_w != crop_w // patch_size:
        _log("    model.patch_h/patch_w do not equal crop // patch_size")
        ok = False
    # crop == patch_h * patch_size keeps ImgAugmentation crop_start == [0, 0],
    # i.e. the precomputed correspondence grid stays aligned.
    if patch_h * patch_size != crop_h or patch_w * patch_size != crop_w:
        _log("    WARNING: crop != patch_grid * patch_size; ImgAugmentation may shift "
             "the crop and misalign correspondences.")
        ok = False
    _log(f"    -> {_verdict(ok)}")
    return ok


@torch.no_grad()
def check_2d_encoder(net, images, cfg):
    """Run the frozen 2D encoder on a real image batch and verify shapes / layout."""
    _log("\n[2] 2D encoder (DINO) wiring")
    ok = True

    if images is None or images.shape[0] == 0:
        _log("    No images in the first batch; skipping (try another batch/dataset).")
        return True

    n_probe = min(2, images.shape[0])
    x = images[:n_probe].cuda().float()

    enc = net.enc2d_model
    # transformers configs expose these for DINOv2-with-registers and DINOv3 alike.
    hidden = getattr(enc.config, "hidden_size", None)
    n_reg = getattr(enc.config, "num_register_tokens", None)
    hf_patch = getattr(enc.config, "patch_size", None)
    _log(f"    encoder.config: hidden_size={hidden}  num_register_tokens={n_reg}  "
         f"patch_size={hf_patch}")

    amp_dtype = dict(float16=torch.float16, bfloat16=torch.bfloat16)[cfg.amp_dtype]
    with torch.amp.autocast(device_type="cuda", enabled=cfg.enable_amp, dtype=amp_dtype):
        feats = net.ENC2D_forward(x)  # the exact path used in the loss

    n_tokens = net.patch_h * net.patch_w
    _log(f"    ENC2D_forward output: {tuple(feats.shape)}  "
         f"(expected [*, {n_tokens}, {cfg.model.enc2d_head_in_channels}])")

    if feats.shape[-2] != n_tokens:
        _log(f"    patch-token count {feats.shape[-2]} != patch_h*patch_w {n_tokens}")
        ok = False
    if feats.shape[-1] != cfg.model.enc2d_head_in_channels:
        _log(f"    feature dim {feats.shape[-1]} != enc2d_head_in_channels "
             f"{cfg.model.enc2d_head_in_channels} (patch_proj target mismatch)")
        ok = False
    if hidden is not None and feats.shape[-1] != hidden:
        _log(f"    feature dim {feats.shape[-1]} != encoder hidden_size {hidden}")
        ok = False

    # Cross-check the register layout that ENC2D_forward relies on (slices trailing patches).
    if hasattr(enc, "vision_model") or "radio" in net.image_weight_name:
        _log("    (SigLIP/RADIO branch — register-layout cross-check skipped.)")
    else:
        raw = enc(x).last_hidden_state
        expected = 1 + (n_reg or 0) + n_tokens
        _log(f"    last_hidden_state seq len = {raw.shape[1]} "
             f"(expected 1 CLS + {n_reg} reg + {n_tokens} patches = {expected})")
        if raw.shape[1] != expected:
            _log("    sequence length does not match [CLS, registers, patches] layout; "
                 "the trailing-slice in ENC2D_forward may grab wrong tokens.")
            ok = False

    _log(f"    -> {_verdict(ok)}")
    return ok


def check_frozen_teacher(net):
    _log("\n[3] Frozen 2D teacher")
    n_trainable = sum(
        p.numel() for p in net.enc2d_model.parameters() if p.requires_grad
    )
    ok = n_trainable == 0
    _log(f"    trainable params in enc2d_model = {n_trainable}  -> {_verdict(ok)}")
    return ok


def grad_global_norm(net):
    """L2 norm over student parameters that currently hold a gradient."""
    sq = 0.0
    n = 0
    for p in net.student.parameters():
        if p.grad is not None:
            g = p.grad.detach()
            sq += float(g.float().pow(2).sum().item())
            n += 1
    return (sq ** 0.5), n


# --------------------------------------------------------------------------------------
# main debug routine (runs inside launch, on every rank)
# --------------------------------------------------------------------------------------
def debug_worker(cfg, max_iters, epochs):
    # Make the debug run self-contained and side-effect free.
    cfg.enable_wandb = False
    cfg.eval_epoch = max(cfg.eval_epoch, 1)

    # Same setup the real entrypoint performs: derive num_worker_per_gpu,
    # batch_size_per_gpu, seeds, etc. (must run before building the trainer).
    cfg = default_setup(cfg)

    trainer = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))
    net = _unwrap(trainer.model)

    # The real Trainer.train() runs everything inside an EventStorage context; both
    # before_train() and the per-step hooks read trainer.storage, so we must open one
    # here (otherwise: AttributeError: Trainer has no attribute 'storage').
    storage = EventStorage()
    storage.__enter__()
    trainer.storage = storage

    # Hook lifecycle: sets up teacher-temp / momentum / mask schedulers and EMA, exactly
    # like a real run, so the captured trend is meaningful.
    trainer.before_train()

    results = {}
    results["config"] = check_config_consistency(cfg)
    results["frozen_teacher"] = check_frozen_teacher(net)

    loss_keys = ["loss", "enc2d_loss", "mask_loss", "unmask_loss", "roll_mask_loss"]
    history = {k: [] for k in loss_keys}
    all_finite = True
    grad_ok = True
    encoder_checked = False
    results["encoder"] = None

    _log("\n[4/5] Running capped training loop "
         f"(epochs={epochs}, max_iters/epoch={max_iters}) ...")

    global_step = 0
    for epoch in range(epochs):
        if comm.get_world_size() > 1:
            trainer.train_loader.sampler.set_epoch(epoch)
        trainer.model.train()
        trainer.epoch = epoch
        trainer.data_iterator = enumerate(trainer.train_loader)
        trainer.before_epoch()

        for i, input_dict in trainer.data_iterator:
            if i >= max_iters:
                break
            trainer.comm_info["iter"] = i
            trainer.comm_info["input_dict"] = input_dict

            # One-time encoder sanity check on the first real image batch.
            if not encoder_checked:
                imgs = input_dict.get("images", None)
                if imgs is not None and imgs.shape[0] > 0:
                    results["encoder"] = check_2d_encoder(net, imgs, cfg)
                    encoder_checked = True
                    _log("\n[4/5] (continuing capped training loop) ...")

            trainer.before_step()
            trainer.run_step()
            trainer.after_step()

            out = trainer.comm_info["model_output_dict"]
            line = [f"epoch {epoch} iter {i:>4d}"]
            for k in loss_keys:
                if k in out:
                    v = float(out[k].item())
                    history[k].append(v)
                    line.append(f"{k}={v:.4f}")
                    if not torch.isfinite(torch.tensor(v)):
                        all_finite = False
            gnorm, gcount = grad_global_norm(net)
            if gcount == 0 or not torch.isfinite(torch.tensor(gnorm)):
                grad_ok = False
            line.append(f"grad_norm={gnorm:.3e}")
            _log("    " + "  ".join(line))
            global_step += 1

    results["losses_finite"] = all_finite
    results["grad_health"] = grad_ok

    # Close the EventStorage context opened above (the trend/summary below do not use it).
    storage.__exit__(None, None, None)

    # ---- direction / trend -----------------------------------------------------------
    _log("\n[6] Direction (first-half vs last-half mean)")
    trend_ok = True
    for k in ("loss", "enc2d_loss"):
        vals = history.get(k, [])
        if len(vals) < 4:
            _log(f"    {k}: not enough steps to assess trend ({len(vals)})")
            continue
        half = len(vals) // 2
        first = sum(vals[:half]) / half
        last = sum(vals[half:]) / (len(vals) - half)
        delta = last - first
        arrow = "down (good)" if delta < 0 else "up/flat"
        _log(f"    {k}: first={first:.4f}  last={last:.4f}  delta={delta:+.4f}  -> {arrow}")
        if k == "enc2d_loss" and delta >= 0:
            trend_ok = False
    results["direction"] = trend_ok

    # ---- summary ---------------------------------------------------------------------
    _log("\n==================== DEBUG SUMMARY ====================")
    order = [
        ("Config consistency", "config"),
        ("2D encoder wiring", "encoder"),
        ("Frozen 2D teacher", "frozen_teacher"),
        ("Losses finite", "losses_finite"),
        ("Gradient health", "grad_health"),
        ("enc2d_loss decreasing", "direction"),
    ]
    hard_fail = False
    for label, key in order:
        val = results.get(key, None)
        if val is None:
            status = "SKIP"
        else:
            status = _verdict(val)
            # 'direction' over a tiny window is advisory, not a hard failure.
            if not val and key != "direction":
                hard_fail = True
        _log(f"    {label:<28s}: {status}")
    _log("======================================================")
    if hard_fail:
        _log("RESULT: problems detected — fix the FAIL items before launching a full run.")
    else:
        _log("RESULT: wiring looks correct. If enc2d_loss is also trending down, "
             "the DINOv3 distillation is heading in the intended direction.")

    comm.synchronize()
    # Non-zero exit on hard failure makes this usable in CI / scripted checks.
    if hard_fail and comm.is_main_process():
        sys.exit(1)


def main():
    parser = default_argument_parser()
    parser.add_argument(
        "--max-iters", type=int, default=50,
        help="number of iterations per epoch to run before stopping (default: 50)",
    )
    parser.add_argument(
        "--epochs", type=int, default=1,
        help="number of (capped) epochs to run (default: 1)",
    )
    args = parser.parse_args()

    cfg = default_config_parser(args.config_file, args.options)
    # Keep all debug outputs inside the repo (do not depend on /tmp).
    if cfg.save_path in (None, "", "exp/default"):
        stem = os.path.splitext(os.path.basename(args.config_file))[0]
        cfg.save_path = os.path.join("exp", "debug_pretrain", stem)
        os.makedirs(os.path.join(cfg.save_path, "model"), exist_ok=True)
    cfg.resume = False

    launch(
        debug_worker,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        cfg=(cfg, args.max_iters, args.epochs),
        backend=args.dist_backend,
    )


if __name__ == "__main__":
    main()
