# 3D Referring-Expression Localization on Utonia features

This directory documents a downstream task built on top of this Pointcept fork:
**grounding a natural-language referring expression to 3D boxes** in an indoor
scene, using a **Utonia-pretrained PT-v3 encoder**
([arXiv:2603.03283](https://arxiv.org/abs/2603.03283)) and the **Locate-3D
dataset** ([arXiv:2504.14151](https://arxiv.org/pdf/2504.14151)) over
**ScanNet + ARKitScenes + ScanNet++**.

> New here? Read [`LEARNINGS.md`](./LEARNINGS.md) first — it explains *why* the
> architecture looks the way it does (short version: a DETR-style decoder never
> converged on Utonia features, so grounding is done as per-point segmentation).

---

## 1. What the system does

Given a scene point cloud and a query like **"a chair near the table"**, the
full framework returns one 3D box per entity (`chair`, `table`). It is built
from **two independently-trained modules** that are chained at inference:

```
  free-form query                     scene point cloud (coord, color, normal)
"a chair near the table"                          │
        │                                         │
        ▼                                         ▼
  ┌───────────────┐   entities          ┌──────────────────────────┐
  │  EntityHead   │ ───────────────────▶│   Locate3DSegDetector     │
  │ (text → spans)│  chair, table       │  Utonia encoder (frozen)  │
  └───────────────┘  + positive_map     │  + U-Net decoder + CLIP   │
                                         │  per-point mask per entity│
                                         └──────────────────────────┘
                                                  │
                                                  ▼
                                   per-entity point mask → axis-aligned box
```

- **`Locate3DSegDetector`** (`pointcept/models/locate_3d/locate_3d_segdet.py`)
  — the grounding model. For each entity it pools that entity's CLIP text
  vector, dots it against every point's projected feature to get a per-point
  score, thresholds the score into a mask, and takes the mask's AABB as the
  box. Trained by configs `0h → 0i → 0j`.
- **`EntityHead`** (`pointcept/models/locate_3d/entity_head.py`) — a small
  transformer over CLIP token embeddings that classifies which tokens belong to
  which entity (and a "no entity" class). It converts a raw query into the
  `positive_map` the grounding model consumes. Trained standalone via
  `tools/train_entity_head.py`.

---

## 2. Repository map (localization-relevant files)

```
configs/utonia/
  localize-utonia-v1m1-0h-combined.py    # stage-1: train on joint corpus (encoder frozen)
  localize-utonia-v1m1-0i-tune.py        # stage-1: 0h + longer schedule + dataset toggles
  localize-utonia-v1m1-0j-encoder-ft.py  # stage-2: unfreeze encoder, fine-tune

pointcept/models/locate_3d/
  locate_3d_segdet.py   # Locate3DSegDetector (the grounding model)
  entity_head.py        # EntityHead (text → entity spans)
  bbox_utils.py         # 3D IoU helpers used by debug metrics

pointcept/datasets/
  arkitscenes_locate3d.py   # ARKitScenesLocate3DDataset (boxes ship pre-transform)
  scannet_locate3d.py       # ScanNet & ScanNetPP datasets (masks derived post-transform)
  locate3d_collate.py       # mixed-corpus collate (locate3d_collate_fn)

pointcept/engines/
  train.py                  # Locate3DTrainer + DDP-safe OOM-skip
  hooks/locate3d.py         # Locate3DStartupSanity, Locate3DMetricsLogger, viz hooks
  hooks/evaluator.py        # Locate3DSegDetectorEvaluator (Acc@IoU)

tools/
  train.py                      # main training entry point (Pointcept harness)
  eval_locate3d_segdet.py       # standalone eval for a trained SegDetector checkpoint
  prepare_entity_head_data.py   # build EntityHead training data from annotation JSONs
  train_entity_head.py          # train the EntityHead
  visualize_locate3d.py         # viz on annotated val scenes (Plotly HTML)
  visualize_locate3d_raw.py     # full framework: EntityHead → SegDetector on a raw scene
  _locate3d_viz_common.py       # shared Plotly helpers

locate-3d/locate3d_data/        # Locate-3D annotation JSONs + upstream reference code
```

---

## 3. Prerequisites

### 3.1 Pretrained weights

| What | How it's used |
|---|---|
| **Utonia PT-v3 checkpoint** | Passed via `-w` to `tools/train.py`. `CheckpointLoader` remaps `module.student.backbone` → `module.backbone`. |
| **CLIP (ViT-L/14)** | Text encoder. Point `LOCATE3D_CLIP_PATH` at a local HF directory (default `/group-volume/CLIP/clip-vit-large-patch14`). Loaded with `local_files_only=True`. |

```bash
export LOCATE3D_CLIP_PATH=/path/to/clip-vit-large-patch14
```

### 3.2 Datasets (Pointcept-preprocessed per-scene `.npy`)

Each config expects three roots; set them at the top of the config or via the
defaults. Layout:

```
<root>/{train,val}/<scene_id>/{coord,color,normal,instance,...}.npy
# ARKitScenes uses {Training,Validation}; the adapter handles that automatically.
```

- **ARKitScenes** boxes ship inside the annotation JSON.
- **ScanNet / ScanNet++** boxes are derived from each scene's `instance.npy`
  *after* `GridSample`, so those scenes must include `instance.npy`.

### 3.3 Annotation JSONs

The **ARKitScenes** JSONs ship in this repo
(`locate-3d/locate3d_data/{train,val}_arkitscenes.json`). Only **sample**
ScanNet / ScanNet++ JSONs are included
(`train_scannet_sample.json`, `train_scannetpp_sample.json`). Download the full
ScanNet / ScanNet++ Locate-3D annotation JSONs from Meta's release and drop
them next to the ARKit ones (same directory):

> https://github.com/facebookresearch/locate-3d/tree/main/locate3d_data/dataset

Name them `train_scannet.json`, `val_scannet.json`, `train_scannetpp.json`,
`val_scannetpp.json`. Any corpus whose JSON or data root is missing is
**silently skipped** (see the `_maybe()` guard in each config), so you can start
with whatever you have.

### 3.4 Dataset on/off toggles

For ablations you can drop sub-corpora without editing the config:

```bash
LOCATE3D_USE_ARKIT=0      # exclude ARKitScenes (train + val)
LOCATE3D_USE_SCANNETPP=0  # exclude ScanNet++ (train + val)
# default "1" = include
```

---

## 4. Stage A — train the grounding model (`0h → 0i → 0j`)

All three use the Pointcept harness. `-w` is the **starting checkpoint**.

**Step 1 — `0h`: train on the joint corpus, encoder frozen.**

```bash
python tools/train.py \
    --config-file configs/utonia/localize-utonia-v1m1-0h-combined.py \
    -w /path/to/utonia.pth \
    --num-gpus 4
```

**Step 2 — `0i`: same recipe, longer schedule + dataset toggles** (produces the
stage-1 checkpoint that `0j` fine-tunes from). `0i` deliberately keeps the `0h`
loss/inference knobs — see the §5 trap in `LEARNINGS.md`.

```bash
python tools/train.py \
    --config-file configs/utonia/localize-utonia-v1m1-0i-tune.py \
    -w /path/to/utonia.pth \
    --num-gpus 4
```

**Step 3 — `0j`: unfreeze the encoder and fine-tune** from the best `0i`/`0h`
checkpoint. This is the riskiest stage; it evals every 2 epochs and you should
**stop early if val Acc regresses**.

```bash
python tools/train.py \
    --config-file configs/utonia/localize-utonia-v1m1-0j-encoder-ft.py \
    -w exp/<your-0i-run>/model/model_best.pth \
    --num-gpus 4
```

### What you'll see while training

Outputs land under `exp/<config-name>/`:

- `model/model_best.pth`, `model/model_last.pth` (+ mid-epoch snapshots if
  `iter_save_freq` is set)
- `metrics_train_iter.{jsonl,csv}` — per-iteration loss/lr/debug scalars
- `metrics_train_epoch.{jsonl,csv}` — per-epoch training averages
- `metrics_val.{jsonl,csv}` — per-eval-epoch `Acc@0.25 / Acc@0.5` etc.

The **`Locate3DStartupSanity`** hook prints, at startup, whether the Utonia
weights actually loaded and the seg/box head weight norms — **check this first**
if results look like training-from-scratch (this was our most expensive bug;
see `LEARNINGS.md` §6).

---

## 5. Stage B — train the EntityHead (text → entities)

This is independent of Stage A and only needs the annotation JSONs + CLIP.

**Step 1 — build tokenized training data** from the same annotation JSONs:

```bash
python tools/prepare_entity_head_data.py \
    --annotations locate-3d/locate3d_data/train_arkitscenes.json \
                  locate-3d/locate3d_data/train_scannet.json \
                  locate-3d/locate3d_data/train_scannetpp.json \
    --output exp/entity_head/train.pt \
    --max-entities 4 --clip-path "$LOCATE3D_CLIP_PATH"

python tools/prepare_entity_head_data.py \
    --annotations locate-3d/locate3d_data/val_arkitscenes.json \
                  locate-3d/locate3d_data/val_scannet.json \
                  locate-3d/locate3d_data/val_scannetpp.json \
    --output exp/entity_head/val.pt \
    --max-entities 4 --clip-path "$LOCATE3D_CLIP_PATH"
```

**Step 2 — train the head** (single GPU is enough; it's a tiny model):

```bash
python tools/train_entity_head.py \
    --train-data exp/entity_head/train.pt \
    --val-data   exp/entity_head/val.pt \
    --output-dir exp/entity_head/run0 \
    --epochs 10
```

Produces `exp/entity_head/run0/model_best.pth`, which stores the trained
weights plus `max_entities` and the CLIP path so inference can reconstruct the
tokenizer.

---

## 6. Evaluation

Score a trained SegDetector checkpoint with the **same metric** the training
evaluator uses (Acc@0.25 / Acc@0.5 on the primary entity, plus AccAll across
all entities):

```bash
python tools/eval_locate3d_segdet.py \
    --config-file configs/utonia/localize-utonia-v1m1-0h-combined.py \
    --weight exp/<run>/model/model_best.pth \
    --iou-thresholds 0.25,0.5
```

### Expected results (measured during development)

| Setup | val Acc@0.25 | AccAll@0.25 | AccAll@0.5 |
|---|---|---|---|
| `0f` ARKit-only, ~991 anns | ~0.03 @ ep12 | — | — |
| `0h` ARKit + ScanNet | **0.54** @ ep15 | 0.50 | 0.42 |
| `0i` (bad retune, reverted) | 0.20 @ ep100 | — | — |

`0h` is the headline working result; `0i`/`0j` are the longer-schedule and
encoder-fine-tune stages on top of it. The `0i (bad retune)` row is kept as a
cautionary data point — see `LEARNINGS.md` §5.

---

## 7. Run the whole framework (inference + visualization)

### 7.1 On annotated validation scenes

Renders the scene, the GT boxes, the predicted boxes/masks, and the caption to
an interactive **Plotly HTML** file:

```bash
python tools/visualize_locate3d.py \
    --config-file configs/utonia/localize-utonia-v1m1-0h-combined.py \
    --weight exp/<run>/model/model_best.pth \
    --num-scenes 5 \
    --output-dir viz_output
# open viz_output/*.html in a browser
```

### 7.2 On a raw scene with a free-form query (full EntityHead → SegDetector chain)

This is the end-to-end framework: it runs `EntityHead` on the query to extract
entities, builds the `positive_map`, then runs the SegDetector and renders the
result. No annotation JSON needed.

```bash
python tools/visualize_locate3d_raw.py \
    --config-file configs/utonia/localize-utonia-v1m1-0h-combined.py \
    --weight exp/<run>/model/model_best.pth \
    --entity-head exp/entity_head/run0/model_best.pth \
    --scene-path /path/to/scene_dir \
    --caption "a chair near the table" \
    --output viz_raw.html
# open viz_raw.html — each detected entity is painted + boxed
```

`--scene-path` points at a preprocessed scene directory (the `{coord,color,
normal}.npy` layout from §3.2). `--pred-mode paint` (default) colors the points
selected for each entity; `--infer-threshold` controls mask tightness.

---

## 8. Quick reference

| I want to… | Command |
|---|---|
| Train stage-1 grounding | `tools/train.py --config-file …/0h-combined.py -w utonia.pth` |
| Continue / longer schedule | `tools/train.py --config-file …/0i-tune.py -w utonia.pth` |
| Fine-tune the encoder | `tools/train.py --config-file …/0j-encoder-ft.py -w <0i best>` |
| Build EntityHead data | `tools/prepare_entity_head_data.py --annotations … --output …` |
| Train EntityHead | `tools/train_entity_head.py --train-data … --val-data … --output-dir …` |
| Score a checkpoint | `tools/eval_locate3d_segdet.py --config-file … --weight …` |
| Visualize (val) | `tools/visualize_locate3d.py --config-file … --weight …` |
| Visualize (raw + query) | `tools/visualize_locate3d_raw.py … --entity-head … --caption "…"` |

Environment variables: `LOCATE3D_CLIP_PATH` (required),
`LOCATE3D_USE_ARKIT` / `LOCATE3D_USE_SCANNETPP` (optional toggles),
`DIST_BACKEND=gloo` (if NCCL is unavailable).
