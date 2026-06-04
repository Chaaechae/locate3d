# -*- coding: utf-8 -*-
"""Generate the Utonia downstream work report as a Word (.docx) document."""
import os
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

OUT = "reports/Utonia_다운스트림_업무보고서.docx"
os.makedirs("reports", exist_ok=True)

doc = Document()

# ---- base styles: Korean-friendly font ----
NORMAL = doc.styles["Normal"]
NORMAL.font.name = "맑은 고딕"
NORMAL.font.size = Pt(10.5)
rpr = NORMAL.element.get_or_add_rPr()
rfonts = rpr.get_or_add_rFonts()
rfonts.set(qn("w:ascii"), "맑은 고딕")
rfonts.set(qn("w:hAnsi"), "맑은 고딕")
rfonts.set(qn("w:eastAsia"), "맑은 고딕")

ACCENT = RGBColor(0x1F, 0x3A, 0x5F)
for hname, sz in (("Heading 1", 15), ("Heading 2", 12.5), ("Heading 3", 11)):
    st = doc.styles[hname]
    st.font.name = "맑은 고딕"
    st.font.size = Pt(sz)
    st.font.color.rgb = ACCENT
    r = st.element.get_or_add_rPr().get_or_add_rFonts()
    r.set(qn("w:eastAsia"), "맑은 고딕")


def set_cell_bg(cell, hexcolor):
    tcpr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), hexcolor)
    tcpr.append(shd)


def para(text="", bold=False, italic=False, size=None, space_after=6, align=None):
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    if text:
        run = p.add_run(text)
        run.bold = bold
        run.italic = italic
        if size:
            run.font.size = Pt(size)
    return p


def rich(parts, space_after=6):
    """parts: list of (text, bold, italic)."""
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    for tp in parts:
        text = tp[0]
        bold = tp[1] if len(tp) > 1 else False
        italic = tp[2] if len(tp) > 2 else False
        run = p.add_run(text)
        run.bold = bold
        run.italic = italic
    return p


def bullet(text, level=0):
    p = doc.add_paragraph(style="List Bullet")
    if level:
        p.paragraph_format.left_indent = Inches(0.25 * (level + 1))
    p.paragraph_format.space_after = Pt(2)
    p.add_run(text)
    return p


def heading(text, level=1):
    h = doc.add_heading(text, level=level)
    return h


def table(headers, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = t.rows[0].cells
    for i, htext in enumerate(headers):
        set_cell_bg(hdr[i], "1F3A5F")
        para_cell = hdr[i].paragraphs[0]
        run = para_cell.add_run(htext)
        run.bold = True
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        run.font.size = Pt(9.5)
    for r in rows:
        cells = t.add_row().cells
        for i, val in enumerate(r):
            pc = cells[i].paragraphs[0]
            run = pc.add_run(str(val))
            run.font.size = Pt(9.5)
            if i == 0 and len(r) > 1:
                run.bold = True
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Inches(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


# ========================= TITLE =========================
title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = title.add_run("Utonia 인코더 기반 다운스트림 작업 보고서")
r.bold = True
r.font.size = Pt(20)
r.font.color.rgb = ACCENT
sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
rs = sub.add_run("3D Localization · Qwen3.5-VL ↔ Utonia 표현 정렬")
rs.font.size = Pt(11.5)
rs.italic = True
meta = doc.add_paragraph()
meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
meta.add_run("작성일: 2026-06-04").font.size = Pt(9.5)

# rule line
pr = doc.add_paragraph()
pbdr = OxmlElement("w:pBdr")
bottom = OxmlElement("w:bottom")
bottom.set(qn("w:val"), "single"); bottom.set(qn("w:sz"), "6")
bottom.set(qn("w:space"), "1"); bottom.set(qn("w:color"), "1F3A5F")
pbdr.append(bottom)
pr._p.get_or_add_pPr().append(pbdr)

# ========================= 0. 배경 =========================
heading("0. 공통 배경", 1)
para("Utonia는 여러 도메인의 point cloud로 사전학습된 PT-v3(PointTransformer-v3) "
     "3D 인코더입니다. 본 작업은 이 인코더를 고정/미세조정하여 두 가지 다운스트림 "
     "과제에 적용한 것입니다.")
bullet("3D Localization — 자연어 참조표현(\"a chair near the table\")을 받아 장면 "
       "안의 해당 물체를 3D 박스로 찾는 grounding 과제 (Locate-3D 데이터셋).")
bullet("Qwen3.5-VL ↔ Utonia 표현 정렬(Alignment) — Utonia의 3D point feature를 "
       "Qwen3.5-VL 비전 타워의 2D patch feature와 정렬하여, 공간추론(VSI/re-VSI) "
       "LLM이 소비할 수 있는 3D 표현으로 만드는 과제.")
para("두 작업 모두 \"사전학습된 3D 인코더 위에 어떤 head/loss/감독신호를 얹어야 "
     "다운스트림이 학습되는가\"가 핵심 질문이었고, 상당 부분이 예상과 다른 실패를 "
     "진단하고 원인을 찾는 과정이었습니다.")

# ========================= 1. Localization =========================
heading("1. 3D Localization (참조표현 grounding)", 1)

heading("1.1 시도한 것", 2)
bullet("목표: Utonia 인코더 + Locate-3D 데이터셋(ScanNet + ARKitScenes + "
       "ScanNet++)으로 참조표현 grounding 학습.")
bullet("원 논문(Locate-3D)과 두 가지를 의도적으로 다르게 설계:")
bullet("grounding 방식 — 박스를 직접 회귀하는 DETR식 decoder 대신, entity별 "
       "per-point segmentation을 학습하고 그 mask의 경계상자(AABB)를 박스로 사용.", 1)
bullet("텍스트 분해 분리 — 질의 전체를 한 번에 넣지 않고, 작은 EntityHead 모듈이 "
       "질의에서 entity를 먼저 추출(\"a chair near the table\" → chair, table)하고, "
       "grounding 모델이 각 entity를 localize. 두 모듈을 따로 학습해 추론 시 연결.", 1)

heading("1.2 과정에서 알게 된 것 (가장 중요한 발견)", 2)
rich([("DETR식 set-prediction decoder는 Utonia feature 위에서 끝내 수렴하지 "
       "못함. ", True), ("박스가 정답 근처에도 가지 못하고 매칭 IoU가 ~0.02에 고착.",)])
rich([("근본 원인: ", True),
      ("Hungarian matching이 학습 초기부터 의미를 가지려면 각 query feature가 "
       "텍스트와 정렬(CLIP-aligned)되어 있어야 함. 원 논문이 쓰는 3D-JEPA feature는 "
       "이를 제공하지만, Utonia는 raw 9차원 입력(좌표+색+법선)으로 학습된 인코더라 "
       "텍스트 정렬 신호가 없음. → sparse 매칭은 init부터 무의미한 신호를 주고 "
       "학습이 시작되지 못함.",)])
rich([("반대로 ", ), ("dense per-point 감독(segmentation)", True),
      ("은 Utonia의 사전학습 목표(dense per-point 표현)와 일치 → frozen feature가 "
       "즉시 유용하고 학습이 됨.",)])

heading("1.3 시행착오", 2)
para("DETR 틀 안에서 단일 변수씩 바꿔가며 검증했고, 각 시도가 별도 교훈을 "
     "남김(모두 같은 collapse로 귀결):")
table(["시도", "결과 / 알게 된 점"],
      [["인코더를 함께 학습", "random 초기화 decoder가 사전학습 인코더를 역전파로 망가뜨림"],
       ["인코더 freeze", "공개 Utonia ckpt에 U-Net decoder 가중치가 없음을 발견 → 기본 "
        "설정은 random decoder를 생성하고 그걸 freeze하면 random feature를 영구히 "
        "먹임 (enc_mode 함정)"],
       ["사전학습 인코더 bottleneck 사용", "100% 사전학습 feature지만 해상도가 거칠어"
        "(~0.32 m voxel) 작은 가구에 IoU 상한이 낮음"],
       ["U-Net decoder scratch 학습 / query에 텍스트 주입 / 더 세밀한 중간 feature",
        "보강에도 모두 plateau"]],
      widths=[2.4, 4.0])
rich([("→ 결론: DETR 폐기, segmentation으로 재구성.", True)])
para("")
para("이후 segmentation 방식에서의 시행착오:")
bullet("클래스 불균형 — 정답 영역에 속하는 point가 전체의 수 %뿐이라 평범한 BCE는 "
       "\"전부 음성\" 자명해로 붕괴. → 양성 가중치(bce_pos_weight)와 Dice loss 비중을 "
       "크게 줘서 양성에 gradient를 유지(불균형 대응용으로 잡은 값).")
bullet("데이터 규모가 병목 — 단일 코퍼스(ARKit only)에서 방식 자체는 학습됨"
       "(Acc@0.25 0→0.03)을 확인. 모델이 아니라 데이터가 병목임을 파악 → 통합 "
       "코퍼스로 확장.")
bullet("데이터셋별 정답 형태가 다름 — ARKit은 박스(point-in-box 근사 mask), "
       "ScanNet/ScanNet++는 per-point instance ID. 후자는 instance.npy를 읽어 "
       "GridSample 이후에 진짜 mask/박스를 유도 → ARKit 근사보다 엄밀히 우수한 감독. "
       "두 코퍼스가 박스를 갖는 시점이 달라 transform 파이프라인을 분리.")
rich([("하이퍼파라미터 튜닝의 함정 (중요한 교훈) — ", True),
      ("진짜 mask는 양성 비율이 높으니 양성 가중치/Dice를 낮추고 threshold를 올려 "
       "mask를 타이트하게 만드는 게 \"명백해\" 보였으나, 측정 결과 오히려 악화"
       "(Acc@0.25 0.54 → 0.20). 이유: mask→박스 metric에서는 약간 큰 박스는 IoU>0.25를 "
       "통과하지만 작은 박스는 못 통과 → 원래 값이 사실은 과제에 맞춰 recall을 "
       "선호하도록 잡힌 값이었음. → 되돌림. 교훈: '명백한' 튜닝도 반드시 metric으로 "
       "검증; loss 균형 직관은 결정 규칙(mask→box)을 넘어 전이되지 않음.",)])

heading("1.4 엔지니어링에서 반복적으로 겪은 것", 2)
bullet("checkpoint가 조용히 로드되지 않는 것이 최대 시간 낭비 — 가중치 경로가 실제 "
       "연결 안 되면 경고 없이 from-scratch로 학습. → 시작 시점에 \"가중치 로드 여부 "
       "+ head norm\"을 출력하는 sanity 훅을 추가해 가장 먼저 확인.")
bullet("OOM 내성 — ScanNet++ 장면 크기가 제각각이라 epoch 후반 메모리 단편화로 OOM. "
       "→ 모든 rank가 함께 배치를 건너뛰는 DDP-safe OOM-skip, epoch 중간 checkpoint "
       "저장.")
bullet("인프라 — wandb 없이 JSONL/CSV 로깅, NCCL 불가 환경 대비 gloo 백엔드 지원, "
       "혼합 코퍼스 collate(샘플별 리스트는 유지, point tensor만 concat).")

heading("1.5 결과와 방향", 2)
rich([("결과: ", True),
      ("ARKit+ScanNet 통합에서 val Acc@0.25 = 0.54 (AccAll@0.25 0.50 / AccAll@0.5 "
       "0.42, epoch 15).",)])
rich([("학습 단계 정착: ", True),
      ("① 인코더 frozen 통합 학습 → ② 더 긴 스케줄로 stage-1 checkpoint 생성 → "
       "③ 인코더 unfreeze 미세조정(차등 학습률 + gradient checkpoint, 가장 위험한 "
       "단계라 자주 eval하고 regression 시 조기 종료).",)])
rich([("방향: ", True),
      ("데이터 규모가 추가 성능의 주 레버이며, EntityHead 체인으로 annotation 없는 "
       "원시 장면 + 자유 질의에 대해 end-to-end 추론·시각화가 가능. 향후 인코더 "
       "미세조정 폭과 데이터 추가가 다음 개선 축.",)])

# ========================= 2. Alignment =========================
heading("2. Qwen3.5-VL ↔ Utonia(PTv3) 표현 정렬", 1)
rich([("최종 목표: ", True),
      ("Utonia의 3D point feature를 Qwen3.5-VL이 이미 소비하는 2D vision "
       "manifold(Qwen3.5 ViT patch)에 정렬시켜, 정렬된 3D feature를 Video-3D-LLM의 "
       "3D positional encoding 대체로 써서 공간추론(VSI/re-VSI) 성능을 올리는 것.",)])

heading("2.1 시도한 것 (학습 구조)", 2)
bullet("2D teacher(고정): Qwen3.5-4B의 model.visual. 512×512 이미지(mean=std=0.5 "
       "정규화) → patch token. LM은 버리고 비전 타워만 사용.")
bullet("3D student: Utonia base 채널의 PT-v3 backbone(warm-start), upcast하여 "
       "1332-d per-point feature.")
bullet("정렬: dataset 단에서 매 step (3D point ↔ 2D patch) correspondence를 다시 "
       "계산. point feature를 patch_proj로 사상하고, patch feature와의 "
       "cosine/InfoNCE로 정렬. SSL(mask/unmask)은 선택적으로 병행.")
bullet("핵심 진단 지표: pos_bc(올바른 pair의 batch-centered cosine), neg_bc(잘못된 "
       "pair), discrim_gap_bc = pos_bc − neg_bc, 그리고 R@1 / MRR / CKA.")

heading("2.2 시행착오 — 변종 A→I의 흐름", 2)
para("이 작업은 \"loss를 바꿔봐도 안 풀리던 것의 진짜 원인을 찾는\" 디버깅 서사가 "
     "핵심입니다.")

heading("Phase 1 — SSL 버그 수정 (A, B)", 3)
bullet("원본 모델에 두 버그 발견: ① teacher의 mask/unmask head가 student 가중치로 "
       "초기화되지 않아 SSL 타깃이 random prototype, ② EMA 업데이트가 빈 pass라 "
       "teacher가 갱신 안 됨.")
bullet("A: 두 버그를 피하려 SSL 자체를 제거(정렬만). → pos_bc 0.01, 학습 안 됨.")
bullet("B: 두 버그를 정상 수정 + cosine pull로 정렬. → mean-direction collapse: "
       "모든 3D feature가 Qwen 평균 방향으로 끌려가 pos=neg=0.97, R@1=0.004. "
       "\"학습은 됐지만 의미 없는 해(trivial solution)\".")

heading("Phase 2 — loss 형식 탐색 (C, D, E, F)", 3)
para("cosine pull의 trivial solution을 막으려 contrastive 형식을 바꿔봄:")
bullet("C: per-scene InfoNCE → pos_bc 0.006 정체")
bullet("D: batch-centered cosine(cosine_bc) → 동일 정체")
bullet("E: cross-scene InfoNCE(K~22k, τ=0.5) → loss가 log(K) 천장에서 안 내려옴")
bullet("F: E + K-subsample 1024 + patch_proj를 MLP로 → 여전히 pos_bc 0.01")
rich([("5개 변종 모두 pos_bc ≤ 0.01 동일 천장. ", True),
      ("loss formulation만으로는 안 풀린다는 결론.",)])

heading("Phase 3 — 단일 batch 깊은 진단 도구 작성", 3)
bullet("\"loss 문제가 아닐 가능성\"이 커져 debug_alignment.py 작성: K survival"
       "(91% 정상), gradient flow(소실 아님), 단일 batch overfit은 100 step에 "
       "pos_bc 0.018→0.066으로 됨(학습 자체는 가능), correspondence 정상.")
rich([("핵심 단서: ", True),
      ("batch 안에서는 학습되는데 scene 간 일반화가 안 됨 → patch_proj가 batch마다 "
       "다른 방향으로 끌려감.",)])

heading("Phase 4 — PCA로 진짜 원인 발견", 3)
rich([("effective rank 분석: PTv3 출력 eff_rank≈6, 그런데 ",),
      ("Qwen feature(마지막 block 출력) eff_rank = 1.0", True),
      (". 모든 patch가 한 방향에 줄지어 있음(μ+λ·v). → 정렬할 정보 자체가 없음. "
       "어떤 loss로도 불가능.",)])

heading("Phase 5 — ViT 구조에서 누락 단계 발견", 3)
bullet("모든 ViT block에서 rank<5, patch_embed부터 붕괴. 가설: merger의 post-block "
       "norm을 빼먹음.")
bullet("Qwen3.5 visual의 merger(norm → 2×2 spatial merge → fc1/act/fc2) 구조 확인. "
       "merger.norm만 적용해도 rank 1 → 12.58 (12배 개선).")
p = doc.add_paragraph()
p.paragraph_format.space_after = Pt(8)
p.paragraph_format.left_indent = Inches(0.2)
rr = p.add_run("결론(한 줄): 원본 코드의 ENC2D_forward가 merger.norm을 호출하지 않아, "
               "사실상 rank-1 feature에 정렬 loss를 걸고 있었던 것이 B~F 전부가 같은 "
               "천장에 막힌 진짜 원인. (1줄 수정으로 해결)")
rr.bold = True
# shade the conclusion paragraph
ppr = p._p.get_or_add_pPr()
shd = OxmlElement("w:shd"); shd.set(qn("w:val"), "clear"); shd.set(qn("w:fill"), "EAF0F6")
ppr.append(shd)

heading("2.3 진단 이후의 방향 (G → H → I)", 2)
para("발견을 반영해 모델을 재설계하며 점진 적용:")
table(["변종", "핵심 변경", "의도 / 결과"],
      [["G", "two-tower(common_dim=512, qwen_proj도 학습) + merger.norm(per-patch, "
        "rank≈12) + enc2d_layer_idx=-2 + InfoNCE_batch(K-sub 1024) + MLP patch_proj "
        "+ SSL off",
        "고정 공간을 맞히는 대신 CLIP/SimCLR식 공통 공간으로. SSL이 정렬을 약간 "
        "방해함을 A vs F에서 확인해 끔"],
       ["H", "use_full_merger=True(2×2 merger 전체 적용 → 16×16 grid, LLM hidden "
        "2560-d), enc2d_layer_idx=-1",
        "rank≈12도 부족 → rank 100+ 기대되는 \"진짜 LLM-aligned\" 표현으로"],
       ["I", "H + 가벼운 SSL 복귀(mask 1/16, roll 1/16, unmask 1/8, enc2d 3/4)",
        "정렬 안정 후 PTv3 일반화 유지용 정규화. 다운스트림 평가에 쓴 최종 레시피 계열"]],
      widths=[0.5, 3.3, 2.6])

heading("2.4 성공 판정 기준 (설정한 체크리스트)", 2)
table(["지표", "목표", "의미"],
      [["pos_bc", "≥ 0.2", "올바른 pair cosine"],
       ["neg_bc", "≤ 0.05", "잘못된 pair cosine (낮을수록 변별적)"],
       ["discrim_gap_bc", "≥ 0.15", "둘의 차"],
       ["R@1", "≥ 0.10", "scene 내 정답 patch 1순위 검색"],
       ["MRR", "≥ 0.15", "mean reciprocal rank"],
       ["CKA(proj, qwen)", "≥ 0.05", "표현 정렬도"]],
      widths=[1.8, 1.2, 3.4])
para("pos_bc ≥ 0.15 & R@1 ≥ 0.05 이면 성공으로 보고 downstream(Video-3D-LLM)으로 "
     "진행, 미달이면 full-merger 16×16 강화 또는 DINOv2 sanity check(framework 자체 "
     "동작 확인)로 분기하는 의사결정 기준을 마련.")

heading("2.5 이 작업에서 얻은 일반 교훈", 2)
bullet("\"loss가 안 떨어진다\"의 원인이 loss에 있지 않을 수 있다. 5개 loss 변형을 "
       "모두 시도한 뒤에야 입력 feature(teacher 표현)의 rank 붕괴가 진짜 원인임을 "
       "발견. → feature 건강성(effective rank)을 먼저 진단하는 것이 loss 튜닝보다 우선.")
bullet("외부 모델(Qwen ViT)의 올바른 사용법: 단순 block 출력이 아니라 의도된 "
       "post-processing(merger.norm / 전체 merger)을 거쳐야 의미 있는 patch 표현이 "
       "나옴. teacher를 \"정확히 의도대로\" 추출하는 것이 정렬의 전제.")
bullet("trivial solution 방지: cosine pull 단독은 mean-direction으로 붕괴 → "
       "negatives를 명시적으로 밀어내는 InfoNCE가 안전하되, K가 너무 크면 log(K) "
       "천장에 막히므로 적절한 subsample(≈1024) 필요.")
bullet("양방향 학습 가능한 공통 공간(two-tower)이 단일 head로 고정 공간을 맞히는 "
       "것보다 안정적.")
bullet("진단 인프라의 가치: 단일 batch overfit / layer-by-layer rank probe / "
       "correspondence 시각화 같은 도구가 없었으면 원인을 못 찾았을 것. 학습 중 "
       "held-out scene에서 pos_bc/CKA 추이를 주기적으로 모니터링하는 체계 구축.")

# ========================= 3. 종합 =========================
heading("3. 종합", 1)
bullet("Localization: \"논문을 그대로 따른 DETR 방식이 Utonia feature 특성(텍스트 "
       "비정렬) 때문에 실패 → segmentation으로 재구성하여 성공(Acc@0.25 0.54)\"이 "
       "핵심 서사. 이후는 데이터 규모, 진짜 mask 감독, 그리고 metric 기반 튜닝 검증의 "
       "문제.")
bullet("Alignment: \"loss를 다섯 번 바꿔도 막히던 것의 진짜 원인이 teacher feature의 "
       "rank-1 붕괴(merger.norm 누락)였음을 진단으로 규명 → two-tower + 올바른 "
       "feature 추출 + InfoNCE로 재설계\"가 핵심 서사. 정렬 품질은 아직 목표 지표 "
       "도달을 검증하는 단계이며, 미달 시 full-merger 16×16 / DINOv2 sanity 분기가 "
       "준비됨.")
bullet("공통적으로, 두 작업 모두 \"실패의 표면 증상(수렴 안 됨)을 loss가 아니라 "
       "표현/감독신호 수준에서 진단\"한 것이 진전을 만든 핵심이었습니다.")

doc.save(OUT)
print("saved:", OUT)
