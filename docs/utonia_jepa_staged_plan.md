# Utonia × 3D-JEPA: Staged Pretraining Plan

Utonia의 입력 포맷(xyz + color + normal)을 유지하면서 Locate3D의 3D-JEPA 학습 방식을
Utonia에 점진적으로 도입하는 계획서.

## 배경 비교

| | Utonia | Locate3D 3D-JEPA |
|---|---|---|
| 입력 | xyz / color / normal | xyz + CLIP(768) + DINO(768) lifted features |
| Pretext | mask / roll_mask / unmask / enc2d (4-loss) | masked latent prediction (단일 JEPA) |
| Head | OnlineCluster (MLP + 4096 prototypes) | 없음, 연속 feature regression |
| Collapse 방지 | Sinkhorn-Knopp + EMA + prototype 균등화 | EMA + stop-grad + predictor 비대칭 |
| Augmentation | global 2 + local 4 multi-view | 단일 view + masking |

Utonia 입력 형태를 유지하면 LiDAR/실내/오브젝트 모든 도메인 호환성이 보존되므로,
*입력 포맷에 의존하지 않는 JEPA 학습 메커니즘만* 이식한다.

## 검증 프로토콜

매 단계 종료 시:

1. **출발점**: 직전 단계의 최종 weight (Stage 1은 Utonia v1m1 weight)
2. **Pretrain finetune**: 10~20 epoch (full pretrain의 1/5~1/10)
3. **Downstream**: ScanNet semseg
   - `semseg-utonia-v1m1-0a-scannet-lin.py` (linear probe, 표현 품질)
   - `semseg-utonia-v1m1-0c-scannet-ft.py` (full finetune, 실용 성능)
4. **결정 기준**: Linear probe mIoU 우선. 동등 이상이면 다음 단계 진행.

비교 표:

| Stage | ScanNet Lin mIoU | ScanNet FT mIoU | 진행 결정 |
|---|---|---|---|
| Baseline (Utonia v1m1) | - | - | 기준 |
| Stage 1 (additive JEPA) | | | ≥ baseline → Stage 2 |
| Stage 2 (block mask) | | | ≥ Stage 1 → Stage 3 |
| Stage 3 (context-only) | | | ≥ Stage 2 → Stage 4 |
| Stage 4 (prototype↓) | | | 실험적 ablation |

---

## Stage 1 — Additive JEPA (가장 안전한 출발점)

### 목적
Utonia 구조에 *연속 latent prediction 신호 하나만* 추가해서 기존 4-loss와 공존시킨다.
표현이 무너지지 않으면서 fine-grained semantic을 추가 학습.

### 추가 컴포넌트
1. **Predictor 모듈**
   - 입력: student backbone의 context feature + masked 위치의 좌표 query
   - 출력: masked 위치에서의 예측 feature ∈ ℝ^d
   - 구조: 가벼운 transformer 2~4 layer (small PTv3 또는 self-attention block)
2. **JEPA loss (5번째 loss)**
   ```
   z_target = teacher.backbone(global_point).feat[masked_idx]   # stop-grad
   z_pred   = predictor(student_context_feat, masked_coord_query)
   jepa_loss = smooth_l1(F.normalize(z_pred), F.normalize(z_target))
   ```

### 유지하는 것
- 입력 채널 (xyz + color + normal, in_channels=9)
- `generate_mask` 그대로 (random grid patch)
- mask / roll_mask / unmask / enc2d 4 loss 모두 유지
- multi-view (global 2 + local 4)
- EMA teacher 그대로 → JEPA의 target encoder로 재활용

### 손실 가중치 (합 = 1 유지)
| Loss | v1m1 | v1m3 (Stage 1) |
|---|---|---|
| mask | 1/8 | 1/10 |
| roll_mask | 1/8 | 1/10 |
| unmask | 2/8 | 2/10 |
| enc2d | 4/8 | 4/10 |
| **jepa** | 0 | **2/10** |

### 학습 설정 (finetune)
- Init weight: Utonia stage v1 pretrained checkpoint
- LR: pretrain의 1/5 (base_lr=0.0008)
- Layer-wise lr decay 유지
- Predictor만 base_lr (decay 없음, 새 모듈이므로)
- Epoch: 10~20
- EMA momentum: 0.999 (느린 갱신, finetune 안정성)

### 왜 안전한가
- 기존 모든 신호 그대로 + 추가 신호 1개 → 기존 representation 손상 risk 낮음
- Predictor는 JEPA loss에만 영향, downstream에서 버려짐
- 실패해도 (linear probe ↓) JEPA weight만 줄여서 재시도 가능

### 검증 기대
- Linear probe mIoU: baseline 대비 +0.3 ~ +1.0
- 떨어지면 → jepa weight 1/20로 줄여 재시도
- 동등이면 → predictor 용량 부족 가능성, layer 수 늘려 재시도

---

## Stage 2 — JEPA-native Masking

Stage 1이 양(+) 신호일 때 진행.

### 변경
- `generate_mask`에 **multi-block masking** 모드 추가
  - 기존: random patch들을 mask
  - 추가: 공간적으로 연속된 큰 block을 4~6개 sample, block 단위로 통째 mask
- jepa_loss는 block mask 사용, mask_loss는 기존 random patch 유지 (분리)

### 왜 강해지나
- Block masking은 *국소 텍스처 보간*으로 풀 수 없음 → long-range context reasoning 강제
- I-JEPA가 image SSL에서 보여준 큰 ablation 효과

### 검증 기대
- Linear probe mIoU: Stage 1 대비 +0.3 ~ +0.8
- 큰 객체 경계(floor/wall) 일관성 향상

---

## Stage 3 — Context-only Forward (진짜 JEPA 구조)

Stage 2 안정 시 진행.

### 변경
- 기존: 마스크 위치에 `mask_token` 넣고 student backbone 통과
- 변경: **마스크 위치 점을 입력에서 제거**, context만 backbone에 통과
- Predictor가 query coord로 마스크 위치를 채움
- mask_loss는 predictor 출력으로 계산 (student backbone의 mask_token 출력 사용 안 함)
- jepa_loss와 mask_loss가 같은 predictor 공유

### 주의
- PTv3는 sparse여서 token 제거가 자연스러움
- Downstream과 입력 분포 차이 발생 가능 (downstream은 전체 입력)
  → finetune 단계에서 backbone이 학습으로 메움
- `mask_token` 파라미터는 unused 또는 제거

### 검증 기대
- Linear probe mIoU: Stage 2 대비 +0.2 ~ +0.5
- 효과가 작거나 동등할 수도 있음 (mask_token이 이미 잘 동작)
- 동등이면 Stage 2에서 멈춰도 무방

---

## Stage 4 — Prototype 의존도 축소 (실험적 ablation)

Stage 3까지 안정 시 진행. 핵심 질문: *"Utonia가 clustering 없이 JEPA만으로 좋은 표현을 얻는가?"*

### 변경 (cosine 스케줄로 점진적)
| Loss | Stage 3 | Stage 4 final |
|---|---|---|
| mask | 1/10 | 0 |
| roll_mask | 1/10 | 0 |
| unmask | 2/10 | 유지 |
| enc2d | 4/10 | 유지 |
| jepa | 2/10 | **6/10** |

- Prototype head freeze 또는 제거
- Sinkhorn 호출 제거

### 결과 해석
- 유지 → clustering 불필요, 파이프라인 단순화 가능
- 하락 → Utonia 입력에 clustering이 본질적으로 필요 (Stage 3에서 멈추는 게 옳음)

둘 다 학술적으로 의미 있는 결과.

---

## 구현 위치

| 파일 | 변경 |
|---|---|
| `pointcept/models/utonia/utonia_v1m3_jepa.py` | 새 module: v1m1 상속 + predictor + jepa_loss |
| `pointcept/models/utonia/__init__.py` | v1m3 import 추가 |
| `configs/utonia/pretrain-utonia-v1m3-0-jepa-stage1.py` | Stage 1 config |
| `configs/utonia/pretrain-utonia-v1m3-1-jepa-stage2.py` | (Stage 2 이후 추가) |
| `configs/utonia/pretrain-utonia-v1m3-2-jepa-stage3.py` | (Stage 3 이후 추가) |
| `configs/utonia/pretrain-utonia-v1m3-3-jepa-stage4.py` | (Stage 4 이후 추가) |

## 진행 메모

- Stage 1 → Stage 2로 넘어가기 전에 반드시 semseg lin/ft mIoU 표 채우기
- 각 단계에서 학습 중 5 loss의 스케일을 첫 epoch에 로그로 확인 → 한 항이 다른 항을 압도하면 가중치 재조정
- EMA collapse 모니터: `teacher.feat.std(dim=0).mean()` 가 시간에 따라 0에 수렴하면 collapse 의심
