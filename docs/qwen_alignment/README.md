# Qwen3.5-VL ↔ Utonia(PTv3) 정렬(distillation) 학습

이 디렉터리는 Utonia로 사전학습된 **PT-v3 3D point encoder를 Qwen3.5-VL 비전
타워에 정렬(align)** 시키는 다운스트림 학습을 다룹니다. 목표는 정렬된 3D
feature를 Qwen3.5 / VG-LLM 기반 공간추론(VSI / re-VSI)에 넣었을 때 성능을 올리는
것입니다 — 즉, Utonia의 3D 표현을 VLM이 이미 소비하는 2D vision manifold에
맞춥니다.

이 코드는 별도 저장소(`qwen-uton`)의 실험에서 가져와 이 Pointcept fork에 직접
통합했습니다.

> **자세한 실험·진단·분석 기록은 한국어로 이미 정리되어 있습니다:**
> - [`QWEN3_5_ALIGNMENT_SUMMARY.md`](./QWEN3_5_ALIGNMENT_SUMMARY.md) — 동기 →
>   학습 방법(loss·실행법) → 평가 → 결과 → 시도한 것/알게 된 사실 → 분석 순.
>   처음 보는 사람용 종합 정리.
> - [`QWEN_ALIGN.md`](./QWEN_ALIGN.md) — feature 차원·loss·진단 지표(`pos_bc`,
>   `discrim_gap_bc` 등)까지 들어간 더 깊은 기술 기록.
>
> 이 README는 **이 저장소(locate3d)에서 어떻게 돌리는가**에 집중합니다.

---

## 1. 무엇이 포함됐나

원본 `qwen-uton` 저장소는 Pointcept를 `third_party/Pointcept` 서브모듈로 두고
`install_into_pointcept.sh`로 심링크해서 썼습니다. **이 저장소는 Pointcept fork
자체이므로 서브모듈/심링크/install 스크립트가 필요 없습니다** — 모든 파일이 이미
제자리에 직접 통합되어 있습니다.

| 파일 | 역할 |
|---|---|
| `configs/utonia/distill-utonia-v1m3-indoor-SSL-qwen3_5-4b.py` | 정렬 **+ 가벼운 SSL**(정규화). **평가에 쓴 레시피.** |
| `configs/utonia/distill-utonia-v1m3-indoor-noSSL-qwen3_5-4b.py` | **정렬만**(SSL 끔). 깨끗한 A/B 비교용. |
| `configs/utonia/eval-utonia-v1m1-dinov2-scannet.py` | DINOv2 baseline(원본 Utonia) eval 레시피. |
| `pointcept/models/utonia/utonia_v1m3b_qwen3_5_distill_ema.py` | 두 distill config가 쓰는 모델 (`Utonia-v1m3b_qwen3_5_distill_ema`). EMA teacher + 2D-3D 정렬. |
| `pointcept/models/utonia/utonia_v1m3a_qwen3_5_align_only.py` | 정렬 전용 변형 (`Utonia-v1m3a_qwen3_5_align_only`). |
| `pointcept/models/utonia/utonia_v1m2_qwen3_5_distill.py` | 초기 distill 변형 (`Utonia-v1m2_qwen3_5_distill`). |
| `pointcept/datasets/skip_on_error_dataset.py` | `SkipOnErrorImagePointDataset` — 불량 장면을 crash 대신 로깅 후 skip. `pointcept/datasets/__init__.py`에 등록됨. |
| `tools/test_qwen3_5_vit_path.py` | Qwen3.5 ViT forward 경로 smoke test (학습 전 사전 점검). |
| `tools/eval_alignment.py` / `eval_alignment_full.py` | 정렬 품질 평가 (pos/neg patch-cosine, R@K, MRR, CKA). |
| `tools/eval_2d_to_3d_retrieval.py` | 2D → 3D retrieval (배포 환경에 가까운 방향). |
| `tools/feature_rank_diagnostic.py` | effective-rank 진단 (collapse vs smooth manifold 구분). |

> 통합 시 한 가지 부수효과: `pointcept/models/utonia/__init__.py`에 `Config.dump`를
> yapf 실패에 대해 non-fatal로 만드는 방어용 monkey-patch가 추가됩니다(이 distill
> 레시피들이 yapf crash를 유발했기 때문). idempotent하고 dump를 더 견고하게만
> 만들므로 다른 레시피(localization 포함)에도 안전합니다.

---

## 2. 사전 준비물 (외부 경로는 모두 환경 변수)

config는 외부 경로를 **환경 변수**로 해석합니다 — config 파일을 수정할 필요가
없습니다.

```bash
# Qwen3.5-4B (로컬 경로 또는 HF repo id). 비전 타워만 사용, LM은 해제.
export QWEN3_5_4B_PATH=/data/hf/Qwen3.5-4B

# Utonia warm-start ckpt — 강력 권장. student와 teacher backbone 양쪽에 로드.
export UTONIA_PRETRAINED_CKPT=/data/ckpt/utonia.pth
# 양쪽에 다른 ckpt를 쓰려면 개별 지정:
# export UTONIA_STUDENT_CKPT=/data/ckpt/utonia.pth
# export UTONIA_TEACHER_CKPT=/data/ckpt/utonia.pth

# 전처리된 3D 데이터셋 루트. config는 ${DATASET_ROOT}/data/<name> 로 접근.
export DATASET_ROOT=/path/to/3Ddataset
```

ckpt 로더는 두 포맷을 자동 감지합니다: 공개 Utonia HF ckpt(`dict(config=...,
state_dict=...)`, raw PTv3 키)와 Pointcept 학습 포맷 ckpt(`module.student.
backbone.*` 키). backbone 가중치만 로드하고, 새 정렬 모듈(`patch_proj`,
`enc2d_head_*`, mask/unmask head)은 random으로 두어 full LR로 학습합니다.

### 데이터 레이아웃 (2D-3D 정렬은 이미지가 필요)

이 레시피는 `DefaultImagePointDataset` 기반이라 point cloud와 함께 per-frame
이미지/대응관계가 필요합니다. Concerto/Utonia 전처리 레이아웃:

```
${DATASET_ROOT}/data/scannet/
├── train/<scene_id>.pth                 # point cloud
├── val/<scene_id>.pth
├── splits/{train,val,test}.json
└── images/{train,val}/
    ├── color/<scene_id>/<frame>.jpg
    ├── correspondence/<scene_id>/<frame>.npy   # 픽셀 ↔ point index
    ├── intrinsic/<scene_id>/<frame>.txt
    └── pose/<scene_id>/<frame>.txt
```

> **경로 주의:** 전처리된 데이터의 `splits.json`이 `data/scannet/...` 처럼
> **cwd 상대 경로**를 담고 있을 수 있습니다. 그런 경우 저장소 루트에서 학습을
> 실행하고 `data/`를 데이터 위치로 심링크하세요:
> `ln -s ${DATASET_ROOT}/data data` (이 저장소의 `.gitignore`는 `data/`를 이미
> 무시합니다). config의 `data_root`는 `${DATASET_ROOT}` 절대 경로를 쓰므로
> `DATASET_ROOT`만 맞으면 됩니다.

---

## 3. 학습 전 사전 점검 (권장)

Qwen3.5 ViT forward 경로가 이 transformers 버전에서 동작하는지 먼저 확인:

```bash
python tools/test_qwen3_5_vit_path.py --model "$QWEN3_5_4B_PATH"
# 기대: [4/4 PASS], strategy=[manual: position_embeddings=(cos,sin) ...]
```

---

## 4. 학습 실행 (저장소 루트에서)

원본 저장소의 `run_train.sh`(서브모듈 init + 심링크 + cluster 경로)는 여기선
필요 없습니다. 직접 실행하세요:

```bash
# 정렬 + 가벼운 SSL (평가에 쓴 레시피) — 권장 시작점
python tools/train.py \
    --config-file configs/utonia/distill-utonia-v1m3-indoor-SSL-qwen3_5-4b.py \
    --num-gpus 1 \
    --options save_path=exp/utonia_q35_ssl

# 정렬만 (SSL 끔) — A/B 비교용
python tools/train.py \
    --config-file configs/utonia/distill-utonia-v1m3-indoor-noSSL-qwen3_5-4b.py \
    --num-gpus 1 \
    --options save_path=exp/utonia_q35_noSSL
```

NCCL을 못 쓰는 환경이면 `DIST_BACKEND=gloo`를 앞에 붙이세요(이 저장소의
`launch.py`가 이미 지원). 멀티 GPU는 `--num-gpus N`.

### 핵심 노브: `backbone_lr_scale` (config 상단)

backbone과 새 정렬 head에 서로 다른 LR을 줍니다:

| 그룹 | LR | 이유 |
|---|---|---|
| `enc{e}.block{b}.*` (backbone block) | `base_lr * scale * 0.9^k` | Utonia geometry 보존; layer-wise decay. |
| `student/teacher.backbone.*` (catch-all) | `base_lr * scale` | embedding, GridPooling down 등. |
| `patch_proj.*`, `enc2d_head_*.*`, mask/unmask head | `base_lr` | 새로 init → full LR. |

- **기본 `0.05`** — `UTONIA_PRETRAINED_CKPT`가 설정됐다고 가정(warm-start).
- backbone을 scratch부터 학습하면 **`1.0`**으로.

### 첫 ~100 스텝에서 확인할 것

- `loss`가 유한하고 감소.
- `enc2d_loss`가 0이 아니고 감소 (이게 2D-3D 정렬).
- `mask_loss` / `unmask_loss`가 유한 (NaN 없음). teacher warm-start가 꺼져 있으면
  초기엔 거의 uniform이다가 수백 스텝 후 떨어지기 시작.

---

## 5. 정렬 품질 평가

학습된 checkpoint로 정렬이 실제로 됐는지 측정합니다(모두 `--config-file`,
`--weight`, `--out-dir` 인자):

```bash
# pos/neg patch-cosine 히스토그램 (가장 단순)
python tools/eval_alignment.py \
    --config-file configs/utonia/distill-utonia-v1m3-indoor-SSL-qwen3_5-4b.py \
    --weight exp/utonia_q35_ssl/model/model_best.pth \
    --num-scenes 50 --out-dir exp/utonia_q35_ssl/align_eval

# within-scene retrieval(R@K, MRR) + linear CKA 추가
python tools/eval_alignment_full.py \
    --config-file configs/utonia/distill-utonia-v1m3-indoor-SSL-qwen3_5-4b.py \
    --weight exp/utonia_q35_ssl/model/model_best.pth \
    --num-scenes 50 --out-dir exp/utonia_q35_ssl/align_eval_full

# 2D → 3D retrieval (배포에 가까운 방향)
python tools/eval_2d_to_3d_retrieval.py \
    --config-file configs/utonia/distill-utonia-v1m3-indoor-SSL-qwen3_5-4b.py \
    --weight exp/utonia_q35_ssl/model/model_best.pth \
    --num-scenes 50 --out-dir exp/utonia_q35_ssl/retr_2d3d

# effective-rank 진단 (feature collapse 여부)
python tools/feature_rank_diagnostic.py \
    --config-file configs/utonia/distill-utonia-v1m3-indoor-SSL-qwen3_5-4b.py \
    --weight exp/utonia_q35_ssl/model/model_best.pth \
    --out-dir exp/utonia_q35_ssl/rank
```

DINOv2 baseline(원본 Utonia)과 비교하려면 `eval-utonia-v1m1-dinov2-scannet.py`로
같은 평가를 돌려 대조하세요.

---

## 6. 다운스트림 precompute 주의

공개 Utonia inference 경로는 5개 encoder stage를 입력 grid로 모두 concat해
1386-d를 만듭니다. 이 레시피는 `enc2d_upcast_level=3`(1332-d) 표현을 학습하므로,
distill된 checkpoint를 소비하는 precompute는 **정확히 3번의 upcast iteration**을
해야 합니다 — 예: `precompute_utonia_features.py`에서 open-ended
`while "pooling_parent" in point.keys(): ...`를 `for _ in range(3): ...`로 바꾸면
2D-3D 정렬 head가 학습 시 본 것과 일치하는 1332-d per-grid feature가 나옵니다.

---

## 7. 경로/통합 점검 체크리스트

이 저장소로 가져오면서 확인한 것들 (학습 시 경로 문제 방지):

- ✅ `utonia_v1m1_base.py`는 원본 저장소와 **동일** — distill 모델이 같은 base
  위에 올라감.
- ✅ distill 모델이 쓰는 pointcept 심볼(`PointModel`, `CosineScheduler`,
  `offset2batch`, `bincount2offset`, `Point`, `MODELS`/`build_model`)이 모두 이
  저장소에 존재.
- ✅ config가 참조하는 모든 registered type(transform `ImgAugmentation` /
  `MultiViewGenerator` / `RandomDropColor` …, hook `ModelHook` /
  `WeightDecaySchedular`, `DefaultTrainer`, `PT-v3m3`)이 모두 존재.
- ✅ `SkipOnErrorImagePointDataset`은 `pointcept/datasets/__init__.py`에 등록,
  base인 `DefaultImagePointDataset`도 존재.
- ✅ `launch.py`는 이 저장소가 이미 `DIST_BACKEND`(nccl|gloo)를 지원 — 덮어쓰지
  않음.
- ✅ config에 하드코딩된 절대 경로 없음 — 전부 `QWEN3_5_4B_PATH` /
  `UTONIA_PRETRAINED_CKPT` / `DATASET_ROOT` 환경 변수로 해석.
- ✅ `_base_ = ["../_base_/default_runtime.py"]`가 `configs/_base_/`로 정상 해석.
- ✅ 새 파일 전부 `py_compile` 통과.
