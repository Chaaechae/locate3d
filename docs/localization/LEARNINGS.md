# Locate-3D-on-Utonia: 학습을 성공시키며 배운 것들

이 문서는 긴 디버깅 과정의 post-mortem입니다. 이 교훈들을 만들어낸 대부분의
config(`localize-utonia-v1m1-0a` … `0g`)는 정리 과정에서 **삭제**되었습니다 —
동작하는 파이프라인을 재현하는 데는 `0h` / `0i` / `0j`만 필요하기 때문입니다.
그 *이유*들은 여기에 남겨, 누구도 같은 고생을 반복하지 않도록 합니다.

한 가지만 읽어야 한다면: **DETR 방식의 set-prediction decoder는 Utonia
feature에서 끝내 수렴하지 못했다. grounding을 per-point segmentation(mask →
axis-aligned 박스)으로 재구성한 것이 성공의 열쇠였다.** 아래는 거기에 도달한
이야기입니다.

---

## 0. 목표

Utonia로 사전학습된 PT-v3 point encoder
([arXiv:2603.03283](https://arxiv.org/abs/2603.03283))를 받아서, Locate-3D
데이터셋([arXiv:2504.14151](https://arxiv.org/pdf/2504.14151)) 위에서 **ScanNet +
ARKitScenes + ScanNet++**를 함께 사용해 *다운스트림* 3D 참조표현 localization
모델을 학습합니다.

Locate-3D 논문과 의도적으로 두 가지가 다릅니다:

1. **Grounding head.** 논문은 *3D-JEPA* feature(이미 vision-language로 정렬됨)
   위에서 언어 조건부 DETR 방식 decoder를 씁니다. 우리는 대신 **per-point
   segmentation head**로 grounding합니다(이유는 §2).
2. **텍스트 → entity 분해를 따로 학습.** 논문은 참조표현 전체를 decoder에
   넣습니다. 우리는 문제를 분리했습니다: 작은 **EntityHead**가 자유 형식 질의에서
   *entity*를 뽑아내고("a chair near the table" → `chair`, `table`), grounding
   모델은 각 entity를 localize하도록 학습됩니다. 두 모듈은 독립적으로 학습되고
   **inference 시점에 연결**됩니다.

---

## 1. Phase 0 — DETR decoder 연결 (브랜치 `claude/utonia-localization-task`)

첫 시도는 논문을 그대로 따랐습니다: Utonia encoder → Locate-3D 언어 조건부
transformer decoder → Hungarian-matched set-prediction loss(sigmoid-focal text
alignment + L1 + GIoU). 그 브랜치의 커밋 로그는 사실상 우리가 차례로 밟은
지뢰들의 목록입니다:

| 증상 | 수정 | 교훈 |
|---|---|---|
| dtype mismatch, 잘못된 `data_root`, `InformationWriter` scalar crash | `fc55a54` | Pointcept harness는 로그 dict에 scalar만 기대함; 나머지는 `InformationWriter` 전에 pop해야 함. |
| wandb 없이 / 단일 머신에서 학습 불가 | `57e1abe` | `gloo` 백엔드 + JSONL/CSV metric 로깅 추가. **이 작업에 wandb는 필요 없습니다.** |
| epoch 1 후 OOM, `KeyError: 'coord'` | `7a5d658` | 크기가 제각각인 장면이 allocator를 단편화; epoch별 캐시 정리 + 데이터셋의 키 처리 주의. |
| 분류 loss가 지배, matcher degenerate, 타겟이 잘못된 프레임 | `47dbf03`, `e6e7607` | 분류 head가 box-matching cost를 압도; 타겟이 world frame이 아니라 camera frame이었음. matcher/loss weight 재조정. |
| **사전학습 가중치가 조용히 로드되지 않음** | `250d220` | `utonia_pretrained_path`가 `cfg.weight`에 연결되지 않아 `CheckpointLoader`가 로드할 게 없었고, *눈치채지 못한 채* from-scratch로 학습 중이었음. 이 프로젝트에서 가장 비싼 버그 — §6 참고. |
| query들이 서로 collapse | `152dfb2`, `dcff1c6` | query-collapse debug metric + epoch별 Plotly viz 추가, 이후 decoder query를 고정 공간 grid에 anchor해 대칭성을 깸. |

이 모든 걸 거쳐도 bbox head는 **여전히** 수렴하지 않았습니다:
`dbg_match_iou`가 런 내내 ~0.02에 머물렀습니다. 박스가 ground truth 근처에도
가지 못했습니다. 그래서 진단 단계로 넘어갔습니다.

---

## 2. Phase 1 — 왜 DETR decoder가 collapse하는가 (삭제된 config 0a–0e, 0g)

각 config는 단일 변수 probe였습니다. 요약:

- **0a** — encoder를 0.1× LR로 decoder와 함께 학습. anchor된 query는 공간적으로
  다양하게 유지됐지만 `loss_bbox`/`loss_giou`가 plateau하고 `match_iou` ~0.02.
  가설: random-init decoder가 사전학습 encoder로 backprop하면서, 정작 cross-
  attend하려는 feature 자체를 불안정하게 만든다.
- **0b** — encoder를 완전히 freeze. 여기서 **진짜** 근본 원인을 발견: 공개된
  Utonia checkpoint는 `enc_mode=True`로 사전학습되어 **`embedding` + `enc`
  가중치만 있고 `dec.*`가 없음**. backbone을 `enc_mode=False`로 돌리면 *random*
  U-Net decoder가 생기고, `CheckpointLoader(strict=False)`는 그것을 조용히
  random으로 둠. 그 위에 `freeze_backbone=True`는 그 random decoder를 *영원히*
  freeze함. 우리는 언어 decoder에 **random** 네트워크의 54-dim feature를 먹이고
  있었음.
- **0c** — `enc_mode=True`: 사전학습된 576-d encoder *bottleneck*(stride 16 ≈
  0.32 m voxel) 사용. 100% 사전학습 feature지만, ~0.45 m ARKit 물체(1–2 voxel
  너비)에는 IoU@0.25가 현실적인 상한.
- **0d** — `enc_mode=False`로 두되 encoder를 freeze한 채 **U-Net decoder를
  scratch부터 실제로 학습**(semantic segmentation에서 통하는 레시피). update
  budget을 위해 `loop=10`.
- **0e** — 0c + 모든 decoder 트릭: text-conditioned query(각 query를 pooling된
  CLIP 요약으로 seed, zero-init이라 step-0은 동일), aux-layer loss weight ramp,
  box-aware augmentation. 그래도 plateau.
- **0g** — bottleneck 대신 stage-3 encoder feature(432-d, stride 8) 사용 —
  Utonia의 2D–3D DINOv2 alignment head가 실제로 붙는 레벨. bottleneck보다 더
  세밀(0.16 m)하고 *더 의미적*. bottleneck은 masked-patch reconstruction으로만
  감독되기 때문.

**Phase 1의 결론:** 모든 sparse / set-prediction 변형이 collapse했습니다.
Hungarian matching은 text-alignment cost가 init 시점부터 의미 있으려면 per-query
feature가 **CLIP-정렬**되어 있어야 합니다. 3D-JEPA는 이를 공짜로 주지만, raw
9-dim `(coord, color, normal)`을 먹는 Utonia는 그렇지 않습니다. 그래서 **DETR
decoder를 폐기**했습니다(정리 중 `locate_3d_decoder.py`, `matcher.py`,
`criterion.py`, `locate_3d_seg.py`/`Locate3DLocalizer` 삭제).

---

## 3. Phase 2 — 성공한 segmentation 재구성 (삭제된 config 0f)

`0f`는 Utonia의 실제 강점을 활용합니다 — Utonia는 *dense per-point* 표현
학습자로 사전학습됐고, 검증된 다운스트림은 dense 감독 하의 per-point
classification입니다.

레시피(현재의 `Locate3DSegDetector`):

1. full-resolution backbone을 돌림, encoder는 frozen, **U-Net decoder는 scratch
   부터 학습**(semseg 레시피).
2. 각 point의 feature를 CLIP 텍스트 공간으로 project.
3. 캡션의 각 entity에 대해, 그 entity의 positive-token CLIP 텍스트 벡터를
   mean-pool하고 **모든 point와 내적** → per-point score.
4. "point가 entity *g*에 속하는가" 지표에 대해 **BCE + Dice**로 학습. 모든
   point가 감독됨.
5. inference: `sigmoid(score) > threshold`; **살아남은 point들의 AABB**가 예측
   박스(너무 적게 남으면 score 상위 5%로 fallback).

이 형태가 문제에 맞는 이유:

- **multi-entity가 구조적.** entity마다 prediction 채널 하나 ⇒ query
  collapse가 *불가능*. GT 2개짜리 캡션은 박스 2개, 5개짜리는 5개를 냄.
- **사전학습 목표와 일치**(dense per-point)하므로 frozen feature가 즉시 유용.

결과: ARKit-only(~991 annotation)에서 Acc@0.25가 12 epoch 안에 0 → **0.03**.
방법은 학습이 되고, 병목은 **데이터 규모**.

여기서 배운 노브(§4에서 크게 중요해짐):

- ARKit의 "point가 GT 박스 안" proxy mask는 positive 클래스가 point의 ~0.3%.
  이 불균형에서 plain BCE는 trivial한 all-zero 최소점으로 collapse하므로,
  positive에 gradient를 유지하려면 **`bce_pos_weight=100`**과
  **`loss_weight_dice=5`**가 필요.

---

## 4. Phase 3 — 규모 + 진짜 mask (`0h`, 유지됨)

`0h`는 `0f`를 **통합 코퍼스**(ARKitScenes + ScanNet + ScanNet++)에 적용 —
annotation이 ~100배.

흥미로운 엔지니어링 디테일: ScanNet / ScanNet++ Locate-3D annotation은 박스가
아니라 **per-point instance ID**를 참조합니다. 그래서 새 `ScanNetLocate3DDataset`
(그리고 ScanNet++ 형제)이 장면의 전처리된 `instance.npy`를 읽어 `GridSample`
*이후*에 entity별 mask + AABB를 유도합니다. 이 **진짜** per-point mask는 ARKit의
inside-box proxy보다 엄밀히 더 나은 감독이라, 모델은 `point_masks`가 있으면 그걸
선호합니다.

두 코퍼스가 박스를 서로 다른 단계에 갖고 있어 **별도의 transform 파이프라인**이
필요합니다:

- **ARKit**: `boxes_xyzxyz`를 transform *이전*에 ship ⇒ 박스가 augment된 coord를
  따라가도록 **box-aware** flip/scale 필요.
- **ScanNet/++**: `instance`를 파이프라인을 통해 운반하고 박스를 transform
  *이후*에 생성 ⇒ plain flip/scale로 충분; 박스는 최종 coord에서 유도됨.

결과: ARKit+ScanNet에서 **val Acc@0.25 = 0.54**, AccAll@0.25 = 0.50,
AccAll@0.5 = 0.42 (**epoch 15**).

---

## 5. Phase 4 — 하이퍼파라미터 튜닝의 함정 (`0i`, 유지됨)

이건 꼭 내재화할 가치가 있습니다. `0h` 이후 `0f`의 노브(ARKit의 0.3%-positive
proxy mask에 맞춰 튜닝됨)가 이제 틀렸다는 게 *명백해* 보였습니다 — 진짜 ScanNet
mask는 5–15% positive니까요. 그래서 `0i`는 처음에 이렇게 시도했습니다:
`pos_weight 100→30`, `dice 5→2`, `max_points 40k→60k`,
`infer_threshold 0.5→0.55`.

**측정 결과 *더 나빠졌습니다*: val Acc@0.25가 0.54 → 0.20.**

이유:

- `infer_threshold=0.55`는 예측 mask와 거기서 유도된 AABB를 체계적으로
  **축소** ⇒ 더 큰 GT 박스에 대해 IoU가 붕괴.
- 낮은 `pos_weight` + 낮은 `dice`는 `0h`가 암묵적으로 튜닝됐던 **recall
  bias**를 제거. mask-AABB-as-box metric에서는 약간 **큰** 박스도 IoU > 0.25를
  통과하지만, **작은** 박스는 통과 못 함. `0h`의 값들은 임의가 아니라
  *태스크에 정렬된, recall을 선호하는* 선택이었음.

그래서 `0i`는 **`0h` 노브로 되돌려졌습니다**. 살아남은 역할은 "`0h` + 더 긴
스케줄 + env-var dataset 토글" — 즉 `0j`가 fine-tune할 stage-1 checkpoint를
만드는 config. **교훈: 모든 '명백한' retune을 metric으로 검증하라; loss 균형에
대한 직관은 mask→box 결정 규칙을 넘어 그대로 전이되지 않는다.**

---

## 6. Phase 5 — encoder fine-tune (`0j`, 유지됨)

encoder를 frozen한 채 decoder + head가 plateau하면, 다음 레버는 encoder
자체입니다. `0j`는 Utonia를 **unfreeze**해서 사전학습 feature가 grounding에
특화되도록 살짝 drift하게 둡니다.

위험은 naive한 공유 LR이 overshoot해서 사전학습을 망가뜨리는 것. `0j`에 내장된
완화책:

- `param_dicts`를 통한 **discriminative LR**: encoder stem + transformer는
  `base_lr * 0.01`, U-Net decoder는 `base_lr`, head는 `base_lr`. `build_optimizer`
  는 **first-substring-match-wins + `break`**이므로 `param_dicts`의 **순서가
  중요**(`backbone.embedding` → `backbone.enc` → `backbone.dec`).
- **Gradient checkpointing**(`backbone_grad_checkpoint=True`): encoder를
  unfreeze하면 모든 transformer-block activation이 backward를 위해 저장됨 —
  지배적 메모리 비용. 대신 recompute(스텝 ~30% 느림, VRAM ~40–60% 절감).
- per-GPU peak를 ≈ 일정하게 유지: `max_points`를 batch size와 trade
  (peak ≈ `batch/world × max_points`).
- `drop_path` 0.3 → 0.1로 낮춤, coord aug 더 가볍게, 스케줄 짧게(40 ep),
  **2 epoch마다 eval**해서 regression을 조기 포착 — encoder unfreeze가 가장
  위험한 단계. val Acc가 나빠지면 조기 종료.

---

## 7. 전반에 걸친 엔지니어링 교훈

반복적으로 우리를 괴롭혔고, 유지된 코드/훅에 녹아 있습니다:

- **checkpoint가 조용히 로드되지 않는 것이 #1 시간 낭비.** 항상 Utonia 가중치가
  실제로 로드됐는지 확인하세요. `Locate3DStartupSanity` 훅이 이를 해줍니다:
  시작 시점에 키 overlap과 bbox/seg head 가중치 norm을 출력하고, 올바른
  `keyword`/`replacement`를 자동 제안합니다. 공개 Utonia checkpoint의 매핑은
  `module.student.backbone` → `module.backbone`(모든 유지된 config의
  `CheckpointLoader`).
- **`enc_mode` 미묘함.** 공개 checkpoint는 `embedding` + `enc` 가중치만 있음.
  `enc_mode=False`는 **random** U-Net decoder를 만듦. 그걸 *학습*하거나
  (`0h`/`0i`/`0j`가 하는 것: backbone 레벨 `freeze_encoder=True` + model 레벨
  `freeze_backbone=False`) `enc_mode=True`로 두세요. **random decoder를 절대
  freeze하지 마세요.**
- **긴 혼합 코퍼스 런을 위한 OOM 내성.** ScanNet++ 장면은 크기가 매우 제각각이라
  epoch 후반에 allocator를 단편화합니다. trainer는 **DDP-safe OOM-skip**(모든
  rank가 함께 배치를 skip해 gradient 상태 동기 유지)을 하고 host + GPU 메모리를
  로깅합니다. `CheckpointSaver`는 `iter_save_freq`로 **epoch 중간** 스냅샷을
  지원해, 후반 OOM/eviction이 epoch 전체를 날리지 않게 합니다.
- **wandb 불필요.** `Locate3DMetricsLogger`가
  `metrics_train_iter.{jsonl,csv}`, `metrics_train_epoch.*`, `metrics_val.*`를
  씁니다. `gloo` 백엔드는 `DIST_BACKEND=gloo`로 지원됩니다.
- **혼합 코퍼스 collation.** `locate3d_collate_fn`이 per-sample 리스트(`caption`,
  `boxes_xyzxyz`, `positive_map`, `point_masks`, …)를 유지하고 per-point tensor를
  `offset` 배열과 함께 concat합니다. ARKit 샘플은 `instance`를 절대 갖지 않고,
  ScanNet은 mask를 만든 뒤 `instance`를 drop — 그래서 서로 다른 코퍼스의 배치가
  shape-호환됩니다.

---

## 8. config 계보 한눈에 보기

```
DETR 방식 (폐기됨)                       Segmentation (유지됨)
─────────────────────                  ───────────────────
0a  enc@0.1xLR + decoder               0f  per-point mask → AABB (ARKit only)
0b  frozen encoder  ─┐                     │  방법이 학습됨을 증명
0c  enc bottleneck   │ 모두 collapse       ▼
0d  train U-Net dec  │ (match_iou ~0.02) 0h  + ScanNet + ScanNet++ (진짜 mask)
0e  0c + query 트릭  │                     │  val Acc@0.25 = 0.54 @ ep15
0g  stage-3 feature ─┘                     ▼
                                        0i  하이퍼파라미터 retune (0h 노브로
                                            되돌림) → stage-1 checkpoint
                                            ▼
                                        0j  encoder unfreeze, discriminative
                                            LR + grad checkpoint (stage-2 FT)
```

`0h → 0i → 0j`와 EntityHead 파이프라인을 실제로 돌리는 방법은
[`README.md`](./README.md)를 보세요.
