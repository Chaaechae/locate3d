"""
Standalone trainer for ``pointcept.models.locate_3d.entity_head.EntityHead``.

Reads pre-tokenized data from ``tools/prepare_entity_head_data.py`` (a
``.pt`` containing ``input_ids / attention_mask / token_labels / meta``),
encodes the captions through CLIP-L (frozen), and trains the
EntityHead's per-token classifier on token-level entity labels.

Single-GPU only -- the head is small (a few MB) and the bottleneck is
CLIP text encoder forward, which is fixed cost per batch. Multi-GPU
adds little. Train wall-clock on a single H100: ~20-30 min for 10
epochs over ~100k captions.

Usage::

    LOCATE3D_CLIP_PATH=/group-volume/CLIP/clip-vit-large-patch14 \\
    python tools/train_entity_head.py \\
        --train-data exp/entity_head/train.pt \\
        --val-data   exp/entity_head/val.pt \\
        --output-dir exp/entity_head/run0 \\
        --epochs 10 \\
        --batch-size 64

Outputs:
    <output-dir>/model_best.pth  — best val token-acc checkpoint
    <output-dir>/model_last.pth  — final epoch checkpoint
    <output-dir>/train.log       — per-epoch metrics
"""

import argparse
import json
import os
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from pointcept.models.locate_3d.entity_head import EntityHead


class EntityHeadDataset(Dataset):
    """Wrap the .pt produced by prepare_entity_head_data.py."""

    def __init__(self, pt_path):
        d = torch.load(pt_path, map_location="cpu", weights_only=False)
        self.input_ids = d["input_ids"]
        self.attention_mask = d["attention_mask"]
        self.token_labels = d["token_labels"]
        self.n_entities = d["n_entities"]
        self.meta = d["meta"]
        self.max_entities = d["max_entities"]
        self.max_length = d["max_length"]
        self.clip_path = d["clip_path"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "token_labels": self.token_labels[idx],
            "n_entities": self.n_entities[idx],
        }


def _collate(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "token_labels": torch.stack([b["token_labels"] for b in batch]),
        "n_entities": torch.stack([b["n_entities"] for b in batch]),
    }


def _build_clip_text_encoder(clip_path):
    from transformers import CLIPTextModel
    is_local = os.path.isdir(clip_path)
    print(f"[clip] loading text encoder from {clip_path}")
    enc = CLIPTextModel.from_pretrained(
        clip_path, local_files_only=is_local
    )
    for p in enc.parameters():
        p.requires_grad = False
    return enc.eval()


@torch.no_grad()
def _encode_text(clip, input_ids, attention_mask):
    """CLIPTextModel returns last_hidden_state + pooler_output. We
    want last_hidden_state (per-token features)."""
    out = clip(input_ids=input_ids, attention_mask=attention_mask)
    return out.last_hidden_state  # (B, T, 768)


def _evaluate(head, clip, loader, device, K):
    """Compute three metrics on the val loader:
    - token_acc: per-token accuracy over entity tokens (-1 ignored)
    - entity_recall: per-entity, fraction recovered (any predicted token
      for that entity)
    - primary_acc: per-utterance, primary entity (E0) recovered
    """
    head.eval()
    n_token_correct = 0
    n_token_total = 0
    n_entity_recovered = 0
    n_entity_total = 0
    n_primary_recovered = 0
    n_utt_total = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_labels = batch["token_labels"].to(device)  # (B, T) in {-1, 0..K-1}
        n_entities = batch["n_entities"].to(device)      # (B,)

        text_feats = _encode_text(clip, input_ids, attention_mask)
        logits = head(text_feats, attention_mask=attention_mask)  # (B, T, K+1)
        pred = logits.argmax(dim=-1)                              # (B, T) in [0, K]

        valid = token_labels >= 0
        n_token_total += int(valid.sum())
        n_token_correct += int(((pred == token_labels) & valid).sum())

        # Per-entity recovery
        B = token_labels.shape[0]
        for b in range(B):
            G = int(n_entities[b].item())
            n_entity_total += G
            n_utt_total += 1
            primary_recovered = False
            for g in range(G):
                # Did the model predict ANY token as entity g?
                hit = bool(((pred[b] == g) & attention_mask[b].bool()).any())
                if hit:
                    n_entity_recovered += 1
                    if g == 0:
                        primary_recovered = True
            if primary_recovered:
                n_primary_recovered += 1

    return {
        "token_acc": n_token_correct / max(n_token_total, 1),
        "entity_recall": n_entity_recovered / max(n_entity_total, 1),
        "primary_acc": n_primary_recovered / max(n_utt_total, 1),
        "n_token_total": n_token_total,
        "n_entity_total": n_entity_total,
        "n_utt_total": n_utt_total,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-data", required=True)
    ap.add_argument("--val-data", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "train.log")
    log_f = open(log_path, "a")
    def log(msg):
        print(msg)
        log_f.write(msg + "\n")
        log_f.flush()
    log(f"[args] {vars(args)}")

    # -- data --
    train_ds = EntityHeadDataset(args.train_data)
    val_ds = EntityHeadDataset(args.val_data)
    log(f"[data] train={len(train_ds)} val={len(val_ds)} "
        f"max_entities={train_ds.max_entities} "
        f"clip_path={train_ds.clip_path}")
    if train_ds.max_entities != val_ds.max_entities:
        raise RuntimeError(
            f"max_entities mismatch: train={train_ds.max_entities} "
            f"val={val_ds.max_entities}"
        )
    K = train_ds.max_entities
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=_collate, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=_collate,
    )

    # -- model --
    device = "cuda"
    clip = _build_clip_text_encoder(train_ds.clip_path).to(device)
    head = EntityHead(
        d_model=clip.config.hidden_size,
        max_entities=K,
        n_layers=args.n_layers,
    ).to(device)
    n_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
    log(f"[model] EntityHead trainable params: {n_params:,}")

    # -- optim --
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(args.warmup_ratio * total_steps)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        # cosine to 0
        import math
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # -- train loop --
    best_metric = -1.0
    global_step = 0
    for epoch in range(args.epochs):
        head.train()
        t_epoch = time.time()
        running_loss = 0.0
        running_n = 0
        for it, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            token_labels = batch["token_labels"].to(device, non_blocking=True)

            text_feats = _encode_text(clip, input_ids, attention_mask)
            logits = head(text_feats, attention_mask=attention_mask)
            loss = head.loss(logits, token_labels,
                             label_smoothing=args.label_smoothing)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * input_ids.size(0)
            running_n += input_ids.size(0)
            global_step += 1

            if (it + 1) % args.log_every == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                log(f"[epoch {epoch}][iter {it+1}/{len(train_loader)}] "
                    f"loss={running_loss/running_n:.4f} lr={lr_now:.2e}")

        epoch_dt = time.time() - t_epoch
        log(f"[epoch {epoch}] train_loss={running_loss/max(running_n,1):.4f} "
            f"time={epoch_dt:.1f}s")

        # -- validate --
        metrics = _evaluate(head, clip, val_loader, device, K)
        log(f"[epoch {epoch}] val: "
            f"token_acc={metrics['token_acc']:.4f} "
            f"entity_recall={metrics['entity_recall']:.4f} "
            f"primary_acc={metrics['primary_acc']:.4f} "
            f"(n_token={metrics['n_token_total']} "
            f"n_entity={metrics['n_entity_total']} "
            f"n_utt={metrics['n_utt_total']})")

        # Save model_last every epoch.
        torch.save({
            "epoch": epoch + 1,
            "state_dict": head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "args": vars(args),
            "max_entities": K,
            "clip_path": train_ds.clip_path,
            "val_metrics": metrics,
        }, os.path.join(args.output_dir, "model_last.pth.tmp"))
        os.replace(
            os.path.join(args.output_dir, "model_last.pth.tmp"),
            os.path.join(args.output_dir, "model_last.pth"),
        )
        # Track best by primary_acc (most directly tied to grounding
        # downstream usage; token_acc / entity_recall are
        # diagnostically useful but primary is what we ultimately
        # need to be right).
        if metrics["primary_acc"] > best_metric:
            best_metric = metrics["primary_acc"]
            import shutil
            shutil.copyfile(
                os.path.join(args.output_dir, "model_last.pth"),
                os.path.join(args.output_dir, "model_best.pth"),
            )
            log(f"[epoch {epoch}] new best primary_acc={best_metric:.4f}")

    log(f"[done] best primary_acc={best_metric:.4f}")
    log_f.close()


if __name__ == "__main__":
    main()
