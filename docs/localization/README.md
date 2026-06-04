# Utonia feature 기반 3D 참조표현 Localization

이 디렉터리는 이 Pointcept fork 위에 올린 다운스트림 태스크를 설명합니다:
**자연어 참조표현(referring expression)을 3D 박스로 grounding** 하는 작업으로,
**Utonia로 사전학습된 PT-v3 encoder**
([arXiv:2603.03283](https://arxiv.org/abs/2603.03283))와 **Locate-3D 데이터셋**
([arXiv:2504.14151](https://arxiv.org/pdf/2504.14151))을 사용해
**ScanNet + ARKitScenes + ScanNet++** 위에서 학습합니다.

> 처음 보시나요? [`LEARNINGS.md`](./LEARNINGS.md)를 먼저 읽으세요 — 왜 지금의
> 구조가 되었는지를 설명합니다(요약: DETR 방식 decoder가 Utonia feature에서
> 수렴하지 못해서, grounding을 per-point segmentation으로 재구성했습니다).

---

## 1. 시스템이 하는 일

장면 point cloud와 **"a chair near the table"** 같은 질의가 주어지면, 전체
framework는 각 entity(`chair`, `table`)마다 3D 박스를 하나씩 반환합니다.
이는 **독립적으로 학습된 두 모듈**을 inference 시점에 연결한 것입니다:

```
  자유 형식 질의                         장면 point cloud (coord, color, normal)
"a chair near the table"                          │
        │                                         │
        ▼                                         ▼
  ┌───────────────┐   entities          ┌──────────────────────────┐
  │  EntityHead   │ ───────────────────▶│   Locate3DSegDetector     │
  │ (텍스트→span)  │  chair, table       │  Utonia encoder (frozen)  │
  └───────────────┘  + positive_map     │  + U-Net decoder + CLIP   │
                                         │  entity별 per-point mask  │
                                         └──────────────────────────┘
                                                  │
                                                  ▼
                                   entity별 point mask → axis-aligned 박스
```

- **`Locate3DSegDetector`** (`pointcept/models/locate_3d/locate_3d_segdet.py`)
  — grounding 모델. 각 entity에 대해 그 entity의 CLIP 텍스트 벡터를 pooling하고,
  모든 point의 projected feature와 내적해 per-point score를 얻고, 이를
  threshold해서 mask로 만든 뒤, mask의 AABB를 박스로 취합니다. config
  `0h → 0i → 0j`로 학습합니다.
- **`EntityHead`** (`pointcept/models/locate_3d/entity_head.py`) — CLIP 토큰
  임베딩 위의 작은 transformer로, 어떤 토큰이 어느 entity에 속하는지(+ "no
  entity" 클래스) 분류합니다. 원시 질의를 grounding 모델이 소비하는
  `positive_map`으로 변환합니다. `tools/train_entity_head.py`로 단독 학습합니다.

---

## 2. 저장소 구성 (localization 관련 파일)

```
configs/utonia/
  localize-utonia-v1m1-0h-combined.py    # stage-1: 통합 코퍼스 학습 (encoder frozen)
  localize-utonia-v1m1-0i-tune.py        # stage-1: 0h + 더 긴 스케줄 + dataset 토글
  localize-utonia-v1m1-0j-encoder-ft.py  # stage-2: encoder unfreeze, fine-tune

pointcept/models/locate_3d/
  locate_3d_segdet.py   # Locate3DSegDetector (grounding 모델)
  entity_head.py        # EntityHead (텍스트 → entity span)
  bbox_utils.py         # debug metric에 쓰는 3D IoU 유틸

pointcept/datasets/
  arkitscenes_locate3d.py   # ARKitScenesLocate3DDataset (박스가 transform 이전에 존재)
  scannet_locate3d.py       # ScanNet & ScanNetPP 데이터셋 (mask를 transform 이후에 유도)
  locate3d_collate.py       # 혼합 코퍼스 collate (locate3d_collate_fn)

pointcept/engines/
  train.py                  # Locate3DTrainer + DDP-safe OOM-skip
  hooks/locate3d.py         # Locate3DStartupSanity, Locate3DMetricsLogger, viz 훅
  hooks/evaluator.py        # Locate3DSegDetectorEvaluator (Acc@IoU)

tools/
  train.py                      # 메인 학습 엔트리포인트 (Pointcept harness)
  eval_locate3d_segdet.py       # 학습된 SegDetector checkpoint 단독 평가
  prepare_entity_head_data.py   # annotation JSON에서 EntityHead 학습 데이터 생성
  train_entity_head.py          # EntityHead 학습
  visualize_locate3d.py         # annotated val 장면 시각화 (Plotly HTML)
  visualize_locate3d_raw.py     # 전체 framework: 원시 장면에 EntityHead → SegDetector
  _locate3d_viz_common.py       # 공유 Plotly 헬퍼

locate-3d/locate3d_data/        # Locate-3D annotation JSON + upstream 참조 코드
```

---

## 3. 사전 준비물

### 3.1 사전학습 가중치

| 무엇 | 어떻게 쓰이나 |
|---|---|
| **Utonia PT-v3 checkpoint** | `tools/train.py`에 `-w`로 전달. `CheckpointLoader`가 `module.student.backbone` → `module.backbone`으로 키를 remap. |
| **CLIP (ViT-L/14)** | 텍스트 encoder. `LOCATE3D_CLIP_PATH`를 로컬 HF 디렉터리로 지정(기본값 `/group-volume/CLIP/clip-vit-large-patch14`). `local_files_only=True`로 로드됨. |

```bash
export LOCATE3D_CLIP_PATH=/path/to/clip-vit-large-patch14
```

### 3.2 데이터셋 (Pointcept 전처리된 per-scene `.npy`)

각 config는 세 개의 root를 요구합니다. config 상단에서 설정하거나 기본값을
사용하세요. 레이아웃:

```
<root>/{train,val}/<scene_id>/{coord,color,normal,instance,...}.npy
# ARKitScenes는 {Training,Validation}을 사용 — adapter가 자동 처리.
```

- **ARKitScenes**: 박스가 annotation JSON 안에 들어 있음.
- **ScanNet / ScanNet++**: 박스를 각 장면의 `instance.npy`에서 `GridSample`
  *이후*에 유도하므로, 해당 장면들은 반드시 `instance.npy`를 포함해야 함.

### 3.3 Annotation JSON

**ARKitScenes** JSON은 이 저장소에 포함되어 있습니다
(`locate-3d/locate3d_data/{train,val}_arkitscenes.json`). ScanNet / ScanNet++는
**샘플** JSON만 포함되어 있습니다
(`train_scannet_sample.json`, `train_scannetpp_sample.json`). 전체 ScanNet /
ScanNet++ Locate-3D annotation JSON은 Meta 릴리스에서 받아서 ARKit JSON과 같은
디렉터리에 두세요:

> https://github.com/facebookresearch/locate-3d/tree/main/locate3d_data/dataset

`train_scannet.json`, `val_scannet.json`, `train_scannetpp.json`,
`val_scannetpp.json`으로 이름 지으면 됩니다. JSON이나 data root가 없는 코퍼스는
**자동으로 건너뜁니다**(각 config의 `_maybe()` 가드 참고). 따라서 가진 것만으로도
바로 시작할 수 있습니다.

### 3.4 데이터셋 on/off 토글

ablation을 위해 config를 수정하지 않고 서브 코퍼스를 제외할 수 있습니다:

```bash
LOCATE3D_USE_ARKIT=0      # ARKitScenes 제외 (train + val)
LOCATE3D_USE_SCANNETPP=0  # ScanNet++ 제외 (train + val)
# 기본값 "1" = 포함
```

---

## 4. Stage A — grounding 모델 학습 (`0h → 0i → 0j`)

셋 다 Pointcept harness를 사용합니다. `-w`는 **시작 checkpoint**입니다.

**Step 1 — `0h`: 통합 코퍼스 학습, encoder frozen.**

```bash
python tools/train.py \
    --config-file configs/utonia/localize-utonia-v1m1-0h-combined.py \
    -w /path/to/utonia.pth \
    --num-gpus 4
```

**Step 2 — `0i`: 동일 레시피, 더 긴 스케줄 + dataset 토글**(`0j`가 fine-tune할
stage-1 checkpoint를 생성). `0i`는 의도적으로 `0h`의 loss/inference 노브를
유지합니다 — `LEARNINGS.md` §5의 함정 참고.

```bash
python tools/train.py \
    --config-file configs/utonia/localize-utonia-v1m1-0i-tune.py \
    -w /path/to/utonia.pth \
    --num-gpus 4
```

**Step 3 — `0j`: encoder를 unfreeze하고 fine-tune**, 가장 좋은 `0i`/`0h`
checkpoint에서 시작. 가장 위험한 단계라 2 epoch마다 eval하며, **val Acc가
나빠지면 조기 종료**해야 합니다.

```bash
python tools/train.py \
    --config-file configs/utonia/localize-utonia-v1m1-0j-encoder-ft.py \
    -w exp/<your-0i-run>/model/model_best.pth \
    --num-gpus 4
```

### 학습 중 보게 되는 것

출력물은 `exp/<config-name>/` 아래에 생성됩니다:

- `model/model_best.pth`, `model/model_last.pth` (+ `iter_save_freq` 설정 시
  epoch 중간 스냅샷)
- `metrics_train_iter.{jsonl,csv}` — iteration별 loss/lr/debug 스칼라
- `metrics_train_epoch.{jsonl,csv}` — epoch별 학습 평균
- `metrics_val.{jsonl,csv}` — eval-epoch별 `Acc@0.25 / Acc@0.5` 등

**`Locate3DStartupSanity`** 훅이 시작 시점에 Utonia 가중치가 실제로 로드됐는지와
seg/box head 가중치 norm을 출력합니다 — 결과가 from-scratch처럼 보이면 **이걸
먼저 확인**하세요(우리가 가장 비싸게 치른 버그입니다; `LEARNINGS.md` §6 참고).

---

## 5. Stage B — EntityHead 학습 (텍스트 → entities)

Stage A와 독립적이며 annotation JSON + CLIP만 필요합니다.

**Step 1 — 동일한 annotation JSON에서 토큰화된 학습 데이터 생성:**

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

**Step 2 — head 학습** (작은 모델이라 1 GPU로 충분):

```bash
python tools/train_entity_head.py \
    --train-data exp/entity_head/train.pt \
    --val-data   exp/entity_head/val.pt \
    --output-dir exp/entity_head/run0 \
    --epochs 10
```

`exp/entity_head/run0/model_best.pth`가 생성됩니다. 학습된 가중치와 함께
`max_entities`, CLIP 경로를 저장해서 inference 시 tokenizer를 복원할 수 있게
합니다.

---

## 6. 평가

학습된 SegDetector checkpoint를 학습 시 evaluator와 **동일한 metric**(primary
entity의 Acc@0.25 / Acc@0.5, 그리고 모든 entity에 대한 AccAll)으로 채점:

```bash
python tools/eval_locate3d_segdet.py \
    --config-file configs/utonia/localize-utonia-v1m1-0h-combined.py \
    --weight exp/<run>/model/model_best.pth \
    --iou-thresholds 0.25,0.5
```

### 기대 결과 (개발 중 측정값)

| 설정 | val Acc@0.25 | AccAll@0.25 | AccAll@0.5 |
|---|---|---|---|
| 단일 코퍼스 (ARKit only), 초기 검증 | ~0.03 @ ep12 | — | — |
| `0h` ARKit + ScanNet | **0.54** @ ep15 | 0.50 | 0.42 |

`0h`가 핵심 성공 결과이고, `0i`/`0j`는 그 위에 각각 "더 긴 스케줄"과 "encoder
fine-tune"을 더한 단계입니다. 단일 코퍼스 행은 "방식 자체는 학습되지만 데이터
규모가 병목"임을 보여주는 초기 검증 결과입니다 — 자세한 배경은
[`LEARNINGS.md`](./LEARNINGS.md) 참고.

---

## 7. 전체 framework 실행 (inference + 시각화)

### 7.1 Annotated 검증 장면에서

장면, GT 박스, 예측 박스/mask, 캡션을 인터랙티브 **Plotly HTML**로 렌더링:

```bash
python tools/visualize_locate3d.py \
    --config-file configs/utonia/localize-utonia-v1m1-0h-combined.py \
    --weight exp/<run>/model/model_best.pth \
    --num-scenes 5 \
    --output-dir viz_output
# viz_output/*.html 를 브라우저에서 열기
```

### 7.2 자유 형식 질의로 원시 장면에서 (EntityHead → SegDetector 전체 체인)

이것이 end-to-end framework입니다: 질의에 `EntityHead`를 돌려 entity를 추출하고,
`positive_map`을 만든 뒤, SegDetector를 실행해 결과를 렌더링합니다. annotation
JSON이 필요 없습니다.

```bash
python tools/visualize_locate3d_raw.py \
    --config-file configs/utonia/localize-utonia-v1m1-0h-combined.py \
    --weight exp/<run>/model/model_best.pth \
    --entity-head exp/entity_head/run0/model_best.pth \
    --scene-path /path/to/scene_dir \
    --caption "a chair near the table" \
    --output viz_raw.html
# viz_raw.html 열기 — 검출된 각 entity가 색칠 + 박스로 표시됨
```

`--scene-path`는 전처리된 장면 디렉터리(§3.2의 `{coord,color,normal}.npy`
레이아웃)를 가리킵니다. `--pred-mode paint`(기본)는 각 entity로 선택된 point를
색칠하고, `--infer-threshold`는 mask의 타이트함을 조절합니다.

---

## 8. 빠른 참조

| 하고 싶은 것 | 명령 |
|---|---|
| stage-1 grounding 학습 | `tools/train.py --config-file …/0h-combined.py -w utonia.pth` |
| 이어서 / 더 긴 스케줄 | `tools/train.py --config-file …/0i-tune.py -w utonia.pth` |
| encoder fine-tune | `tools/train.py --config-file …/0j-encoder-ft.py -w <0i best>` |
| EntityHead 데이터 생성 | `tools/prepare_entity_head_data.py --annotations … --output …` |
| EntityHead 학습 | `tools/train_entity_head.py --train-data … --val-data … --output-dir …` |
| checkpoint 채점 | `tools/eval_locate3d_segdet.py --config-file … --weight …` |
| 시각화 (val) | `tools/visualize_locate3d.py --config-file … --weight …` |
| 시각화 (원시 + 질의) | `tools/visualize_locate3d_raw.py … --entity-head … --caption "…"` |

환경 변수: `LOCATE3D_CLIP_PATH` (필수),
`LOCATE3D_USE_ARKIT` / `LOCATE3D_USE_SCANNETPP` (선택 토글),
`DIST_BACKEND=gloo` (NCCL을 못 쓸 때).
