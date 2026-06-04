# 설계 결정과 배운 것들

이 문서는 이 localization 파이프라인이 **왜 지금의 모습이 되었는지**를
설명합니다. 코드만 봐서는 알기 어려운 "왜 이 구조이고, 왜 이 값인가"를 처음
보는 사람도 따라올 수 있게 정리했습니다.

한 가지만 기억한다면: **참조표현을 박스로 직접 회귀(DETR 방식)하는 대신,
entity별 per-point mask를 예측하고 그 mask의 경계상자(AABB)를 박스로 쓰는
segmentation 방식이 핵심이다.** 아래는 그 결정에 이른 이야기입니다.

---

## 0. 목표

Utonia로 사전학습된 PT-v3 point encoder
([arXiv:2603.03283](https://arxiv.org/abs/2603.03283))를 가져와, Locate-3D
데이터셋([arXiv:2504.14151](https://arxiv.org/pdf/2504.14151)) 위에서 **ScanNet +
ARKitScenes + ScanNet++**를 함께 써서 3D 참조표현 localization 모델을
학습하는 것.

Locate-3D 논문과 의도적으로 두 가지가 다릅니다:

1. **Grounding 방식이 다름.** 논문은 박스를 직접 예측하는 언어 조건부 decoder를
   씁니다. 우리는 **per-point segmentation** 후 mask에서 박스를 유도합니다(§1).
2. **텍스트 → entity 분해를 따로 학습.** 논문은 참조표현 전체를 한 번에
   넣습니다. 우리는 작은 **EntityHead**가 질의에서 entity를 먼저 뽑고
   ("a chair near the table" → `chair`, `table`), grounding 모델이 각 entity를
   localize합니다. 두 모듈은 따로 학습되어 inference 때 연결됩니다.

---

## 1. 왜 segmentation 방식인가 (박스 직접 예측을 버린 이유)

처음에는 논문을 그대로 따라, **언어 조건부 DETR 방식 decoder**(고정 개수의
query를 두고, Hungarian matching으로 query↔정답을 매칭한 뒤 박스를 회귀)로
시작했습니다. 결과는 **수렴 실패**였습니다 — 박스가 정답 근처에도 가지 못하고,
매칭 품질 지표가 바닥에 머물렀습니다.

근본 원인은 이렇습니다. Hungarian matching이 의미 있으려면, 학습 초기부터 각
query feature가 **텍스트와 정렬(CLIP-aligned)** 되어 있어야 합니다. 그래야
"이 query가 이 단어를 가리킨다"는 신호가 처음부터 존재합니다. 논문이 쓰는
3D-JEPA feature는 이 정렬을 공짜로 제공하지만, **Utonia는 raw 9차원 입력(좌표 +
색 + 법선)으로부터 학습된 인코더라 텍스트 정렬 feature를 제공하지 않습니다.**
정렬되지 않은 feature 위에서 sparse한 매칭은 init에서부터 무의미한 신호를
주고, decoder는 끝내 학습 궤도에 오르지 못합니다.

이 결론에 이르기까지 DETR 틀 안에서 여러 변형을 시도했고, 각각에서 별도의
교훈을 얻었습니다(모두 결국 같은 collapse로 끝남):

- **인코더를 같이 학습하면** random 초기화된 decoder가 사전학습 인코더로
  역전파하면서, 정작 참조하려는 feature 자체를 망가뜨림.
- **인코더를 freeze하면** 한 가지 함정이 드러남: 공개 Utonia checkpoint에는
  인코더 가중치만 있고 **U-Net decoder 가중치가 없음**. 기본 설정으로 돌리면
  decoder가 random으로 생성되는데, 그걸 그대로 freeze하면 decoder에 영원히
  random feature를 먹이게 됨(자세히는 §6의 `enc_mode`).
- **사전학습된 인코더 bottleneck을 쓰면** 100% 사전학습 feature지만 해상도가
  너무 거칠어(voxel ~0.32 m) 작은 가구에는 IoU 상한이 낮음.
- **U-Net decoder를 scratch부터 제대로 학습**하거나, **query에 텍스트 요약을
  주입**하거나, **더 세밀한 중간 단계 feature**를 쓰는 등 여러 보강을 더해도
  결과는 동일하게 plateau.

그래서 DETR 방식을 폐기하고 **per-point segmentation으로 재구성**했습니다.

---

## 2. Segmentation 방식 (현재 구조)

이 방식은 Utonia의 실제 강점을 활용합니다 — Utonia는 *dense per-point* 표현
학습자로 사전학습됐고, 검증된 강점은 dense 감독 하의 per-point 분류입니다.

`Locate3DSegDetector`가 하는 일:

1. 인코더는 frozen, full-resolution **U-Net decoder는 scratch부터 학습**.
2. 각 point feature를 CLIP 텍스트 공간으로 project.
3. 캡션의 각 entity에 대해, 그 entity 토큰들의 CLIP 텍스트 벡터를 평균내고
   **모든 point와 내적** → point마다 "이 entity에 속하는 정도" score.
4. "point가 entity *g*의 영역 안에 있는가" 정답에 대해 **BCE + Dice**로 학습.
   모든 point가 감독을 받음.
5. inference: `sigmoid(score)`가 threshold를 넘는 point들을 모으고, **그
   point들의 경계상자(AABB)**가 예측 박스(너무 적게 남으면 score 상위 일부로
   fallback).

이 형태가 문제에 맞는 이유:

- **multi-entity가 구조적.** entity마다 출력 채널이 하나라, DETR에서 보던
  "query들이 한 점으로 뭉쳐버리는" 문제가 *원천적으로 불가능*. 정답이 2개면
  박스 2개, 5개면 5개를 냄.
- **사전학습 목표(dense per-point)와 일치**하므로 frozen feature가 즉시 유용.

여기서 얻은 핵심 노브:

- 정답 영역에 속하는 point는 전체의 극히 일부(수 %)라 클래스 불균형이 큽니다.
  이때 평범한 BCE는 "전부 음성"이라는 자명한 최소점으로 무너집니다. 이를 막으려고
  **양성 가중치(`bce_pos_weight`)를 크게** 주고 **Dice loss 비중(`loss_weight_dice`)
  을 높여** 양성에 gradient가 살아있게 합니다. 이 값들은 임의가 아니라 불균형을
  버티기 위한 선택입니다(§4에서 이게 왜 중요한지 다시 나옵니다).

> 단일 코퍼스(ARKit only)만으로도 이 방식이 **학습된다는 것**을 먼저 확인했고,
> 그때 병목은 모델이 아니라 **데이터 규모**였습니다. 그래서 다음 단계는
> 데이터를 키우는 것이었습니다.

---

## 3. 규모 + 진짜 mask (`0h`)

`0h`는 위 방식을 **통합 코퍼스**(ARKitScenes + ScanNet + ScanNet++)로 확장합니다.
annotation 수가 크게 늘어납니다.

엔지니어링 디테일 하나: 데이터셋마다 정답의 형태가 다릅니다.

- **ARKitScenes**: 정답이 **박스**로 주어짐. "point가 박스 안인가"를 근사 정답
  (proxy mask)으로 씀.
- **ScanNet / ScanNet++**: 정답이 박스가 아니라 **per-point instance ID**로
  주어짐. 그래서 데이터셋 adapter가 장면의 `instance.npy`를 읽어, `GridSample`
  *이후*에 entity별 mask와 박스를 직접 유도함. 이 **진짜** per-point mask는
  ARKit의 근사 정답보다 엄밀히 더 나은 감독이라, 모델은 진짜 mask가 있으면 그걸
  우선합니다.

두 코퍼스가 박스를 서로 다른 시점에 갖고 있어 **transform 파이프라인도
분리**됩니다:

- ARKit은 박스가 augmentation *이전*에 존재 ⇒ flip/scale이 박스도 같이 변환해야
  함(box-aware).
- ScanNet/++은 `instance`를 파이프라인 끝까지 운반해 *이후*에 박스를 만듦 ⇒
  평범한 flip/scale로 충분(최종 좌표에서 박스를 유도하므로).

**결과:** ARKit+ScanNet에서 val Acc@0.25 = 0.54, AccAll@0.25 = 0.50,
AccAll@0.5 = 0.42 (epoch 15 기준).

---

## 4. 하이퍼파라미터 튜닝의 함정 (`0i`)

이건 꼭 기억할 가치가 있습니다. `0h` 이후, §2의 노브들이 이제 틀렸다는 게
*명백해* 보였습니다. 근사 정답(point의 ~0.3%만 양성)에 맞춰 튜닝한 값인데, 진짜
mask는 양성 비율이 훨씬 높으니까요. 그래서 양성 가중치와 Dice 비중을 낮추고,
inference threshold를 살짝 올려 mask를 더 타이트하게 만들어 봤습니다.

**측정 결과 오히려 나빠졌습니다: val Acc@0.25가 0.54 → 0.20.**

이유:

- threshold를 올리면 예측 mask가, 그리고 거기서 나온 박스가 **체계적으로
  작아져서** 큰 정답 박스에 대해 IoU가 무너집니다.
- 양성 가중치/Dice 비중을 낮추면 `0h`가 암묵적으로 갖고 있던 **recall 편향**이
  사라집니다. mask에서 박스를 뽑는 이 metric에서는, 약간 **큰** 박스는
  IoU > 0.25를 통과하지만 **작은** 박스는 통과하지 못합니다. 즉 `0h`의 값들은
  임의가 아니라 *과제에 맞춰 recall을 선호하도록* 잡힌 값이었습니다.

그래서 `0i`는 `0h`의 값으로 되돌렸고, 역할은 "`0h` + 더 긴 스케줄 + 데이터셋
on/off 토글" — 다음 단계(`0j`)가 fine-tune할 **stage-1 checkpoint**를 만드는
config가 되었습니다. **교훈: '명백해 보이는' 튜닝도 반드시 metric으로 검증하라.
loss 균형에 대한 직관은 'mask에서 박스를 뽑는' 결정 규칙을 넘어 그대로 전이되지
않는다.**

---

## 5. 인코더 fine-tune (`0j`)

인코더를 frozen한 채 decoder와 head가 더 이상 좋아지지 않으면, 다음 레버는
인코더 자체입니다. `0j`는 Utonia 인코더를 **unfreeze**해서, 사전학습 feature가
grounding에 맞게 살짝 적응(drift)하도록 둡니다.

위험은, 단순히 같은 학습률을 주면 인코더가 과하게 움직여 사전학습을 망가뜨리는
것입니다. `0j`에 들어간 안전장치:

- **그룹별 차등 학습률**: 인코더는 아주 낮은 학습률(`base_lr`의 1%), decoder와
  head는 정상 학습률. 인코더는 살짝만 움직입니다.
- **gradient checkpointing**: 인코더를 학습시키면 backward를 위해 모든 중간
  activation을 저장해야 해 메모리가 급증합니다. 대신 forward를 다시 계산해
  메모리를 크게 아낍니다(스텝은 약간 느려짐).
- 메모리 예산을 일정하게: point 수를 줄이는 대신 batch를 키우는 식으로 GPU당
  peak을 맞춤.
- augmentation을 더 가볍게, 스케줄을 더 짧게, **eval을 더 자주**(2 epoch마다)
  해서 regression을 조기에 잡음. 가장 위험한 단계이므로 **val Acc가 나빠지기
  시작하면 조기 종료**.

---

## 6. 전반에 걸친 엔지니어링 교훈

반복적으로 시간을 잡아먹었고, 지금의 코드/훅에 녹아 있는 것들:

- **checkpoint가 조용히 로드되지 않는 것이 가장 큰 함정.** 가중치 경로가
  실제로 연결되지 않으면, 아무 경고 없이 **from-scratch로 학습**하면서도 눈치채지
  못할 수 있습니다. `Locate3DStartupSanity` 훅이 시작 시점에 "가중치가 실제로
  로드됐는지 + head 가중치 norm"을 출력합니다. 결과가 이상하면 **이걸 먼저
  확인**하세요. (공개 Utonia checkpoint는 키 이름이 달라서 `CheckpointLoader`가
  `module.student.backbone` → `module.backbone`으로 remap합니다.)
- **`enc_mode` 주의.** 공개 checkpoint에는 인코더 가중치만 있습니다. 기본
  설정으로 돌리면 U-Net decoder가 *random*으로 생성됩니다. 그러니 그 decoder를
  *학습*시키거나(현재 config들이 하는 방식: 인코더만 freeze, decoder는 학습),
  아니면 인코더 bottleneck 모드를 쓰세요. **random decoder를 freeze하지
  마세요.**
- **긴 혼합 코퍼스 학습의 OOM 내성.** ScanNet++ 장면은 크기가 매우 제각각이라
  epoch 후반에 메모리 단편화로 OOM이 납니다. trainer는 **모든 rank가 함께 배치를
  건너뛰는 OOM-skip**(분산 학습의 gradient 동기 유지)을 하고, checkpoint를
  **epoch 중간에도 저장**할 수 있어 후반 OOM이 epoch 전체를 날리지 않습니다.
- **wandb 불필요.** metric은 JSONL/CSV로 기록됩니다. NCCL을 못 쓰면
  `DIST_BACKEND=gloo`로 학습할 수 있습니다.
- **혼합 코퍼스 collation.** collate 함수가 캡션·박스·mask 같은 per-sample
  리스트는 그대로 두고, per-point tensor만 이어 붙입니다. ARKit 샘플은
  `instance`가 없고 ScanNet은 mask를 만든 뒤 `instance`를 버리므로, 서로 다른
  데이터셋의 배치가 모양 충돌 없이 섞입니다.

---

## 7. 3단계 학습 요약

```
0h  통합 코퍼스에서 인코더 frozen으로 학습 (진짜 per-point mask 활용)
     │   val Acc@0.25 = 0.54
     ▼
0i  같은 방식 + 더 긴 스케줄 + 데이터셋 토글  →  stage-1 checkpoint 생성
     │   (튜닝을 시도했다가 더 나빠져서 0h 값으로 되돌림 — §4)
     ▼
0j  인코더 unfreeze, 차등 학습률 + gradient checkpoint로 fine-tune (stage-2)
```

실제로 돌리는 방법은 [`README.md`](./README.md)를 보세요.
