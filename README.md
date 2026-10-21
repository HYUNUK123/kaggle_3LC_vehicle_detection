# 🚗 3LC Multi-Vehicle Detection Challenge

> **Data-Centric AI** 접근으로 접근한 Kaggle 다중 차량 객체 탐지 프로젝트
> UA-DETRAC 데이터셋 · YOLOv8n from-scratch · 3LC 데이터 관리 파이프라인

[![YOLOv8](https://img.shields.io/badge/Model-YOLOv8n-blue)]()
[![Framework](https://img.shields.io/badge/Framework-3LC-orange)]()
[![Metric](https://img.shields.io/badge/mAP@0.5-0.85-green)]()

---

## 📌 Overview

CCTV 기반 도로 영상(UA-DETRAC)에서 **4종 차량(truck / car / van / bus)**을 탐지하는 프로젝트입니다.
모델 아키텍처와 하이퍼파라미터를 바꾸는 대신, **데이터 품질 개선(Data-Centric AI)**에 집중해 성능을 끌어올리는 것을 목표로 했습니다.

| 항목 | 내용 |
|------|------|
| **Task** | Multi-class vehicle detection |
| **Dataset** | UA-DETRAC (train 7,852 / val 982) |
| **Classes** | `0: truck`, `1: car`, `2: van`, `3: bus` |
| **Model** | YOLOv8n (from scratch, `yolov8n.yaml`) |
| **Metric** | mAP@0.5 |
| **Framework** | 3LC + tlc_ultralytics 8.4.6 |
| **Hardware** | Windows · RTX 4060 8GB |

---

## ⚠️ Competition Constraints

대회 규칙상 다음이 **금지**되어, 모델 성능보다 데이터 전략이 핵심 변수였습니다.

- ❌ 사전학습 가중치 사용 금지 (random init으로 from-scratch 학습)
- ❌ 외부 데이터 / 이미지 추가 금지
- ❌ 입력 크기 변경 금지 (**640px 고정**)
- ❌ TTA · Ensemble · Pseudo-labeling · Knowledge Distillation 금지
- ✅ 데이터 필터링 / 정제 / 라벨 수정은 **허용** (대회의 핵심 의도)

> 이 제약 덕분에 "모델을 어떻게 키울까"가 아니라 **"데이터를 어떻게 다듬을까"**가 전체 전략의 중심이 되었습니다.

---

## 🔑 Key Problem: Class Imbalance

`car`가 데이터셋을 압도하고 `truck / van / bus`는 소수 클래스였습니다.

```
car    ████████████████████████  (majority)
truck  ████
van    ███
bus    ██
```

**혼동 행렬(confusion matrix) 분석 결과** 각 소수 클래스의 실패 원인이 서로 달랐습니다.

- `truck` → `van`으로 오분류 (클래스 혼동)
- `van` → `car`와 혼동 (데이터 부족이 아닌 분류 문제)
- 전 클래스 공통: 20%+ 미검출(background) — augmentation으로 해결 불가

---

## 🛠️ Approach — Data-Centric Pipeline

### 1. Deduplication (중복 제거)
- 임베딩 유사도 기반 near-duplicate 제거
- **임계값 0.99가 sweet spot** (0.98은 유효 데이터까지 제거해 성능 하락)
- `ua_detrac_10k-train` (7,852) → `-dedup-99` (~5,100) 생성 후 검증

### 2. Loss Weight Ablation
- `cls` loss weight를 `0.5 → 0.7 → 1.0 → 1.5`로 점진 상향
- 분류 가중치를 높일수록 **van 탐지 성능 개선** (혼동 문제가 핵심이라는 가설과 일치)

### 3. Copy-Paste Augmentation
- Kisantal(2019) 방식으로 소수 클래스(truck ×4, van ×2) within-image 오버샘플
- 3LC `TableWriter`로 증강 테이블 직접 생성
- **결과**: val mAP ↑, 그러나 Kaggle test는 소폭 ↓
  → `car` false positive 증가가 원인으로 추정 (오버샘플의 양면성 확인)

### 4. Optimizer Comparison
| Optimizer | Setting | 결과 |
|-----------|---------|------|
| **SGD** | lr=0.01, 100 epochs | 🥇 Best |
| AdamW | lr=0.001, 50 epochs | 🥈 2위 |

### 5. 3LC Dashboard Iteration (라벨 품질 개선)
```
학습된 모델로 예측 → 3LC에 등록 → Dashboard에서 GT vs 예측 비교
→ 라벨 노이즈 발견 → 수정/제거 → 새 revision → 재학습
```
모델 예측과 GT를 비교해 **라벨 노이즈를 시각적으로 탐지**하는 워크플로우.

---

## 🧰 Scripts

| 파일 | 역할 |
|------|------|
| `diagnose_table.py` | 3LC 테이블에서 클래스별 instance 수, 불균형 비율, bbox 크기 분포 리포트 |
| `copy_paste_table.py` | 소수 클래스 within-image copy-paste 증강 후 새 3LC 테이블 생성 |
| `inspect_table.py` | 3LC row/bbox 스키마 진단 (1회성) |
| `collect_predictions.py` | 학습 모델 예측을 3LC Run에 등록 (라벨 검수용) |
| `train.py` | 학습 엔트리포인트 (config.yaml 기반) |

> 💡 **3LC bbox 포맷 주의**: `bbs`의 `x0, y0, x1, y1` 키는 이름과 달리 실제로는 YOLO `cx, cy, w, h`에 매핑됩니다. (`inspect_table.py`로 검증 완료)

---

## 📊 Results

| 지표 | 값 |
|------|-----|
| **본 프로젝트 mAP@0.5** | ~0.85 |
| 1위 솔루션 | ~0.95 |
| 참고 솔루션 (val+train merge + 라벨 정제) | 0.934 |

**검증된 것**
- ✅ Dedup ablation으로 최적 임계값 도출
- ✅ cls weight ablation으로 소수 클래스 개선
- ✅ 혼동 행렬 분석으로 클래스별 실패 원인 분리

**다음 단계 (미완)**
- 🔲 3LC Dashboard 라벨 품질(iteration) 작업 본격화 — 0.85 → 0.90+ 로 가는 가장 확실한 경로
- 🔲 selective copy-paste (작은 박스만 선별) 적용

---

## 💡 Lessons Learned

1. **제약이 전략을 만든다** — 모델 고정 조건에서 데이터가 유일한 레버였고, 이는 실무의 data-centric 접근과 정확히 일치했다.
2. **val 성능 ↑ ≠ test 성능 ↑** — copy-paste가 val mAP는 올렸지만 test에서는 car FP를 늘렸다. 증강은 양날의 검.
3. **실패 원인을 쪼개라** — "소수 클래스가 약하다"를 혼동 행렬로 분해하니, truck은 데이터 문제·van은 분류 문제로 처방이 달라졌다.

---

## 🗂️ Tech Stack

`Python` · `YOLOv8n (Ultralytics)` · `3LC / tlc_ultralytics` · `PyTorch` · `OpenCV`
