# Locate-3D-on-Utonia: what we learned getting it to train

This document is the post-mortem of a long debugging effort. Most of the
configs that produced these lessons (`localize-utonia-v1m1-0a` … `0g`) were
**deleted** during cleanup because only `0h` / `0i` / `0j` are needed to
reproduce the working pipeline. Their *reasoning* lives here so nobody has to
re-learn it the hard way.

If you only read one thing: **a DETR-style set-prediction decoder never
converged on Utonia features. Reframing grounding as per-point segmentation
(mask → axis-aligned box) is what worked.** Everything below is the story of
how we got there.

---

## 0. The goal

Take the Utonia-pretrained PT-v3 point encoder
([arXiv:2603.03283](https://arxiv.org/abs/2603.03283)) and train a *downstream*
3D referring-expression localization model on the Locate-3D dataset
([arXiv:2504.14151](https://arxiv.org/pdf/2504.14151)), using **ScanNet +
ARKitScenes + ScanNet++** jointly.

We deliberately diverge from the Locate-3D paper in two ways:

1. **Grounding head.** The paper uses a language-conditioned DETR-style
   decoder over *3D-JEPA* features (which are already vision-language
   aligned). We ground with a **per-point segmentation head** instead
   (see §2 for why).
2. **Text → entity decomposition is learned separately.** The paper feeds the
   whole referring expression into the decoder. We split the problem: a small
   **EntityHead** is trained to pull the *entities* out of a free-form query
   ("a chair near the table" → `chair`, `table`), and the grounding model is
   trained to localize each entity. The two modules are trained independently
   and **chained at inference time**.

---

## 1. Phase 0 — wiring the DETR decoder (branch `claude/utonia-localization-task`)

The first attempt followed the paper literally: Utonia encoder → Locate-3D
language-conditioned transformer decoder → Hungarian-matched set-prediction
loss (sigmoid-focal text alignment + L1 + GIoU). The commit log on that branch
is essentially a list of footguns we hit, in order:

| Symptom | Fix | Lesson |
|---|---|---|
| dtype mismatch, wrong `data_root`, `InformationWriter` scalar crash | `fc55a54` | The Pointcept harness expects scalar-only outputs in the log dict; everything else must be popped before `InformationWriter`. |
| Can't train without wandb / on a single box | `57e1abe` | Added `gloo` backend support + JSONL/CSV metric logging. **You do not need wandb to run this.** |
| OOM after epoch 1, `KeyError: 'coord'` | `7a5d658` | Variable-size scenes fragment the allocator; per-epoch cache clearing + careful key handling in the dataset. |
| Class loss dominates, matcher degenerate, targets in wrong frame | `47dbf03`, `e6e7607` | The classification head was drowning the box-matching cost; targets were in camera frame, not world frame. Rebalanced matcher/loss weights. |
| **Pretrained weights were silently NOT loading** | `250d220` | `utonia_pretrained_path` was never wired to `cfg.weight`, so `CheckpointLoader` had nothing to load and we were training from scratch *without noticing*. This is the single most expensive bug in the project — see §6. |
| Queries collapse onto each other | `152dfb2`, `dcff1c6` | Added query-collapse debug metrics + per-epoch Plotly viz, then anchored decoder queries to a fixed spatial grid to break symmetry. |

After all of that, the bbox head **still** would not converge:
`dbg_match_iou` sat at ~0.02 for the whole run. The boxes never approached the
ground truth. That moved us to the diagnosis phase.

---

## 2. Phase 1 — why the DETR decoder collapses (deleted configs 0a–0e, 0g)

Each of these configs was a single-variable probe. Summarized:

- **0a** — encoder trained at 0.1× LR alongside the decoder. Anchored queries
  stayed spatially diverse, but `loss_bbox`/`loss_giou` plateaued and
  `match_iou` ~0.02. Hypothesis: a random-init decoder back-propagating into
  the pretrained encoder destabilizes the very features it's trying to
  cross-attend to.
- **0b** — freeze the encoder entirely. This is where we found the **real**
  root cause: the released Utonia checkpoint was pretrained with
  `enc_mode=True`, so it contains **only `embedding` + `enc` weights, no
  `dec.*`**. Running the backbone with `enc_mode=False` instantiates a *random*
  U-Net decoder, and `CheckpointLoader(strict=False)` silently leaves it
  random. `freeze_backbone=True` then froze that random decoder *forever*. We
  were feeding the language decoder 54-dim features from a **random** network.
- **0c** — `enc_mode=True`: use the pretrained 576-d encoder *bottleneck*
  (stride 16 ≈ 0.32 m voxels). 100% pretrained features, but so coarse that
  IoU@0.25 is the realistic ceiling for ~0.45 m ARKit objects (1–2 voxels
  across).
- **0d** — `enc_mode=False` but actually **train the U-Net decoder from
  scratch** with the encoder frozen (the recipe that works for semantic
  segmentation). `loop=10` to get the update budget up.
- **0e** — 0c + every decoder trick: text-conditioned queries (seed each query
  with a pooled CLIP summary, zero-init so step-0 is unchanged), aux-layer loss
  weight ramp, box-aware augmentation. Still plateaus.
- **0g** — use the stage-3 encoder feature (432-d, stride 8) instead of the
  bottleneck, because that's the level where Utonia's 2D–3D DINOv2 alignment
  head actually attaches. Finer (0.16 m) *and* more semantic than the
  bottleneck, which is only supervised by masked-patch reconstruction.

**Conclusion of Phase 1:** every sparse / set-prediction variant collapsed.
Hungarian matching needs per-query features that are **CLIP-aligned** so the
text-alignment cost is meaningful at init. 3D-JEPA gives that for free; Utonia,
fed raw 9-dim `(coord, color, normal)`, does not. We **abandoned the DETR
decoder** (and deleted `locate_3d_decoder.py`, `matcher.py`, `criterion.py`,
`locate_3d_seg.py`/`Locate3DLocalizer` during cleanup).

---

## 3. Phase 2 — the segmentation reframe that worked (deleted config 0f)

`0f` plays to Utonia's actual strength — it was pretrained as a *dense
per-point* representation learner, and its proven downstream is per-point
classification under dense supervision.

The recipe (now `Locate3DSegDetector`):

1. Run the full-resolution backbone, encoder frozen, **U-Net decoder trained
   from scratch** (the semseg recipe).
2. Project each point's feature into the CLIP text space.
3. For each entity in the caption, mean-pool its positive-token CLIP text
   vector and **dot it with every point** → a per-point score.
4. Train with **BCE + Dice** against the "point belongs to entity *g*"
   indicator. Every point is supervised.
5. Inference: `sigmoid(score) > threshold`; the **AABB of the surviving
   points** is the predicted box (fall back to top-5% by score if too few
   survive).

Why this is the right shape for the problem:

- **Multi-entity is structural.** One prediction channel per entity ⇒ query
  collapse is *impossible*. A 2-GT caption emits 2 boxes; a 5-GT caption emits
  5.
- **It matches the pretraining objective** (dense per-point), so the frozen
  features are immediately useful.

Result: Acc@0.25 went from 0 → **0.03 within 12 epochs on ARKit-only**
(~991 annotations). The method learns; **data scale** is the bottleneck.

Knobs we learned here (and which turn out to matter a lot in §4):

- ARKit's "point inside GT box" proxy mask makes the positive class ~0.3% of
  points. Under that imbalance, plain BCE collapses to the trivial all-zero
  minimum, so we need **`bce_pos_weight=100`** and **`loss_weight_dice=5`** to
  keep a gradient on positives.

---

## 4. Phase 3 — scale + real masks (`0h`, KEPT)

`0h` is `0f` on the **joint corpus** (ARKitScenes + ScanNet + ScanNet++),
~100× more annotations.

The interesting engineering wrinkle: ScanNet / ScanNet++ Locate-3D annotations
reference **per-point instance IDs**, not boxes. So a new
`ScanNetLocate3DDataset` (and its ScanNet++ sibling) reads the scene's
preprocessed `instance.npy` and derives the per-entity mask + AABB **after**
`GridSample`. These **real** per-point masks are strictly better supervision
than ARKit's inside-box proxy, so the model prefers `point_masks` when present.

Because the two corpora carry boxes at different stages, they need **separate
transform pipelines**:

- **ARKit** ships `boxes_xyzxyz` *pre*-transform ⇒ needs **box-aware** flip/
  scale so the boxes track the augmented coords.
- **ScanNet/++** carry `instance` through the pipeline and build boxes
  *post*-transform ⇒ plain flip/scale is fine; boxes are derived from whatever
  the final coords are.

Result: ARKit+ScanNet **val Acc@0.25 = 0.54**, AccAll@0.25 = 0.50,
AccAll@0.5 = 0.42 by **epoch 15**.

---

## 5. Phase 4 — the hyperparameter-tuning trap (`0i`, KEPT)

This one is worth internalizing. After `0h` it seemed *obvious* that the `0f`
knobs (tuned for ARKit's 0.3%-positive proxy masks) were now wrong, because
real ScanNet masks are 5–15% positive. So `0i` originally tried:
`pos_weight 100→30`, `dice 5→2`, `max_points 40k→60k`,
`infer_threshold 0.5→0.55`.

**It was measured *worse*: val Acc@0.25 fell 0.54 → 0.20.**

Why:

- `infer_threshold=0.55` systematically **shrinks** the predicted mask and the
  AABB derived from it ⇒ IoU collapses against larger GT boxes.
- Lower `pos_weight` + lower `dice` removed the **recall bias** that `0h` was
  implicitly tuned for. For the mask-AABB-as-box metric, a slightly
  **oversized** box still passes IoU > 0.25; an **undersized** one doesn't. The
  `0h` values were not arbitrary — they were *task-aligned, recall-favouring*
  choices.

So `0i` was **reverted to the `0h` knobs**. Its surviving role is "`0h` + a
longer schedule + env-var dataset toggles" — i.e. the config that produces the
stage-1 checkpoint that `0j` fine-tunes from. **Lesson: validate every
'obvious' retune against the metric; intuition about loss balance does not
transfer across a mask→box decision rule.**

---

## 6. Phase 5 — fine-tuning the encoder (`0j`, KEPT)

Once the decoder + heads plateau with the encoder frozen, the next lever is the
encoder itself. `0j` **unfreezes** Utonia and lets the pretrained features
drift slightly to specialize for grounding.

The risk is a naive shared LR overshooting and destroying the pretraining.
Mitigations baked into `0j`:

- **Discriminative LR** via `param_dicts`: encoder stem + transformer at
  `base_lr * 0.01`, U-Net decoder at `base_lr`, heads at `base_lr`. Note
  `build_optimizer` does **first-substring-match-wins with `break`**, so the
  **order of `param_dicts` matters** (`backbone.embedding` → `backbone.enc` →
  `backbone.dec`).
- **Gradient checkpointing** (`backbone_grad_checkpoint=True`): with the
  encoder unfrozen, all transformer-block activations would be kept for
  backward — the dominant memory cost. Recompute instead (~30% slower step,
  ~40–60% less VRAM).
- Hold per-GPU peak ≈ constant: trade `max_points` for batch size
  (peak ≈ `batch/world × max_points`).
- Lower `drop_path` 0.3 → 0.1, lighter coord aug, shorter schedule (40 ep),
  **eval every 2 epochs** to catch regression early — encoder unfreeze is the
  riskiest stage. Stop early if val Acc regresses.

---

## 7. Cross-cutting engineering lessons

These bit us repeatedly and are baked into the kept code/hooks:

- **Silent checkpoint non-loading is the #1 time-sink.** Always confirm the
  Utonia weights actually loaded. The `Locate3DStartupSanity` hook does this:
  it reports key overlap and the bbox/seg head weight norms at startup, and
  auto-suggests the right `keyword`/`replacement`. The mapping for the released
  Utonia checkpoint is `module.student.backbone` → `module.backbone`
  (`CheckpointLoader` in every kept config).
- **`enc_mode` subtlety.** The released checkpoint has only `embedding` + `enc`
  weights. `enc_mode=False` creates a **random** U-Net decoder. Either *train*
  it (what `0h`/`0i`/`0j` do, with `freeze_encoder=True` at the backbone level
  and `freeze_backbone=False` at the model level) or set `enc_mode=True`.
  **Never freeze a random decoder.**
- **OOM resilience for long mixed-corpus runs.** ScanNet++ scenes are highly
  variable in size and fragment the allocator late in an epoch. The trainer
  does a **DDP-safe OOM-skip** (all ranks skip the batch together so gradient
  state stays in sync) and logs host + GPU memory. `CheckpointSaver` supports
  `iter_save_freq` for **mid-epoch** snapshots so a late-epoch OOM/eviction
  doesn't cost the whole epoch.
- **No wandb required.** `Locate3DMetricsLogger` writes
  `metrics_train_iter.{jsonl,csv}`, `metrics_train_epoch.*`, and
  `metrics_val.*`. `gloo` backend is supported via `DIST_BACKEND=gloo`.
- **Mixed-corpus collation.** `locate3d_collate_fn` keeps per-sample lists
  (`caption`, `boxes_xyzxyz`, `positive_map`, `point_masks`, …) and
  concatenates per-point tensors with an `offset` array. ARKit samples never
  carry `instance`; ScanNet drops `instance` after building masks — so batches
  from different corpora stay shape-compatible.

---

## 8. The config lineage at a glance

```
DETR-style (ABANDONED)                 Segmentation (KEPT)
─────────────────────                  ───────────────────
0a  enc@0.1xLR + decoder               0f  per-point mask → AABB (ARKit only)
0b  frozen encoder  ─┐                     │  proved the approach learns
0c  enc bottleneck   │ all collapse        ▼
0d  train U-Net dec  │ (match_iou ~0.02) 0h  + ScanNet + ScanNet++ (real masks)
0e  0c + query tricks│                     │  val Acc@0.25 = 0.54 @ ep15
0g  stage-3 feature ─┘                     ▼
                                        0i  hyperparam retune (reverted to 0h
                                            knobs) → stage-1 checkpoint
                                            ▼
                                        0j  unfreeze encoder, discriminative
                                            LR + grad checkpoint (stage-2 FT)
```

See [`README.md`](./README.md) for how to actually run `0h → 0i → 0j` and the
EntityHead pipeline.
