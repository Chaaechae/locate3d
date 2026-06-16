"""
Standalone EntityHead inspection: feed it captions and see which
tokens it assigns to which entity. No scene, no SegDetector -- just
text -> CLIP -> EntityHead -> per-token entity labels.

Use for sanity-checking the trained head before / instead of running
the full chained inference. If EntityHead is mis-segmenting captions
here, the chained pipeline will fail too.

Usage::

    LOCATE3D_CLIP_PATH=/group-volume/CLIP/clip-vit-large-patch14 \\
    python tools/inspect_entity_head.py \\
        --weight exp/entity_head/run0/model_best.pth \\
        --queries \\
            "the chair next to the desk" \\
            "a bag under the table" \\
            "the lamp on the small white table" \\
            "the leftmost cup on the kitchen counter"

Or read queries from a file (one caption per line)::

    python tools/inspect_entity_head.py \\
        --weight exp/entity_head/run0/model_best.pth \\
        --query-file queries.txt
"""

import argparse
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import torch
import numpy as np


# ANSI colors for entity highlighting in the terminal. Cycled by entity
# index. The "no entity" / padding tokens stay default.
_ANSI = [
    "\033[1;95m",  # magenta
    "\033[1;92m",  # green
    "\033[1;93m",  # yellow
    "\033[1;94m",  # blue
    "\033[1;96m",  # cyan
    "\033[1;91m",  # red
]
_RESET = "\033[0m"


def _load_entity_head(weight_path):
    from pointcept.models.locate_3d.entity_head import EntityHead
    from transformers import AutoTokenizer, CLIPTextModel

    eh_ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
    K = eh_ckpt.get("max_entities", 4)
    clip_path = eh_ckpt.get(
        "clip_path",
        os.environ.get("LOCATE3D_CLIP_PATH",
                       "openai/clip-vit-large-patch14"),
    )
    is_local = os.path.isdir(clip_path)
    print(f"[entity-head] from {weight_path}, K={K}, clip={clip_path}")
    tok = AutoTokenizer.from_pretrained(
        clip_path, local_files_only=is_local
    )
    clip_text = CLIPTextModel.from_pretrained(
        clip_path, local_files_only=is_local
    ).cuda().eval()
    for p in clip_text.parameters():
        p.requires_grad = False
    head = EntityHead(
        d_model=clip_text.config.hidden_size,
        max_entities=K,
        n_layers=eh_ckpt.get("args", {}).get("n_layers", 2),
    ).cuda().eval()
    info = head.load_state_dict(eh_ckpt["state_dict"], strict=False)
    if info.missing_keys or info.unexpected_keys:
        print(f"[entity-head] missing={len(info.missing_keys)} "
              f"unexpected={len(info.unexpected_keys)}")
    return head, clip_text, tok, K


def _inspect(head, clip_text, tok, caption, K, no_color=False):
    """Run EntityHead on a single caption and pretty-print the
    per-token / per-entity assignment."""
    enc = tok(
        caption, return_tensors="pt",
        padding="max_length", truncation=True, max_length=77,
        return_offsets_mapping=True,
    )
    input_ids = enc["input_ids"].cuda()
    attention_mask = enc["attention_mask"].cuda()
    offsets = enc["offset_mapping"][0].tolist()
    token_ids = enc["input_ids"][0].tolist()

    with torch.no_grad():
        text_feats = clip_text(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        logits = head(text_feats, attention_mask=attention_mask)   # (1, T, K+1)
    probs = torch.softmax(logits[0], dim=-1).cpu().numpy()         # (T, K+1)
    pred = probs.argmax(axis=-1)                                   # (T,)

    # Reconstruct token strings.
    token_strs = tok.convert_ids_to_tokens(token_ids)
    valid_t = int(attention_mask[0].sum().item())

    # Per-entity word groups (using offset_mapping to slice caption).
    entity_words = [[] for _ in range(K)]
    for ti, (s, e) in enumerate(offsets):
        if ti >= valid_t:
            break
        if s == 0 and e == 0:
            continue
        cls = int(pred[ti])
        if cls < K:
            entity_words[cls].append((s, e, caption[s:e]))

    # Print results.
    print()
    print("=" * 78)
    print(f"caption: {caption!r}")

    # Colored caption: highlight character ranges by entity.
    if not no_color:
        char_class = [-1] * len(caption)
        for ti, (s, e) in enumerate(offsets):
            if ti >= valid_t or (s == 0 and e == 0):
                continue
            cls = int(pred[ti])
            if cls < K:
                for c in range(s, min(e, len(caption))):
                    char_class[c] = cls
        out = []
        prev = -1
        for i, ch in enumerate(caption):
            cls = char_class[i]
            if cls != prev:
                if prev != -1:
                    out.append(_RESET)
                if cls != -1:
                    out.append(_ANSI[cls % len(_ANSI)])
                prev = cls
            out.append(ch)
        if prev != -1:
            out.append(_RESET)
        print("colored:", "".join(out))

    # Per-entity summary.
    nonempty = [(g, ws) for g, ws in enumerate(entity_words) if ws]
    if not nonempty:
        print("[no entities predicted]")
    else:
        for g, ws in nonempty:
            words = " ".join(w[2] for w in ws)
            color = _ANSI[g % len(_ANSI)] if not no_color else ""
            reset = _RESET if not no_color else ""
            label = "primary" if g == 0 else f"secondary_{g}"
            print(f"  {color}E{g} ({label}){reset}: {words!r}  "
                  f"({len(ws)} CLIP tokens)")

    # Per-token detail (useful for catching off-by-one / sub-token
    # split errors).
    print("  per-token:")
    for ti in range(valid_t):
        s, e = offsets[ti]
        if s == 0 and e == 0:
            # special token (BOS/EOS)
            tok_str = token_strs[ti]
            cls = int(pred[ti])
            cls_str = f"E{cls}" if cls < K else "_"
            print(f"    [{ti:2d}] {tok_str:>15s}  span=(special)         "
                  f"-> {cls_str}  (prob={probs[ti, cls]:.3f})")
            continue
        tok_str = token_strs[ti]
        word = caption[s:e]
        cls = int(pred[ti])
        cls_str = f"E{cls}" if cls < K else "_"
        prob_top = probs[ti, cls]
        # Also show runner-up for ambiguous tokens.
        top2 = np.argsort(probs[ti])[::-1][:2]
        if top2[0] != cls:
            top2 = top2[::-1]
        runner = int(top2[1])
        runner_str = f"E{runner}" if runner < K else "_"
        runner_prob = probs[ti, runner]
        print(f"    [{ti:2d}] {tok_str:>15s}  "
              f"span=({s:>3d},{e:>3d}) {word!r:>12s}  "
              f"-> {cls_str}({prob_top:.2f}) "
              f"runner={runner_str}({runner_prob:.2f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", required=True,
                    help="EntityHead .pth (from train_entity_head.py)")
    ap.add_argument("--queries", nargs="*", default=None,
                    help="space-separated captions to inspect")
    ap.add_argument("--query-file", default=None,
                    help="path to a text file, one caption per line")
    ap.add_argument("--no-color", action="store_true",
                    help="disable ANSI colors in output (for log files)")
    args = ap.parse_args()

    captions = list(args.queries or [])
    if args.query_file:
        with open(args.query_file) as f:
            captions += [
                line.rstrip("\n") for line in f if line.strip()
            ]
    if not captions:
        raise SystemExit(
            "no captions given. Pass --queries '...' or --query-file PATH."
        )

    head, clip_text, tok, K = _load_entity_head(args.weight)
    for cap in captions:
        _inspect(head, clip_text, tok, cap, K, no_color=args.no_color)


if __name__ == "__main__":
    main()
