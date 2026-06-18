#!/usr/bin/env python3
"""
클래스 불균형 해소용 Copy-Paste 증강 (Kisantal 2019 방식 기반).

이 대회 코드 흐름에 맞춘 별도 전처리 스크립트.
train.py 는 건드리지 않는다. 흐름:

  1) 3LC train 테이블을 디스크에 YOLO 포맷으로 export  (export_table.py)
  2) [이 스크립트] 희소 클래스를 골라 copy-paste 증강 -> 새 폴더 생성
  3) 증강된 폴더를 다시 3LC 테이블로 등록           (register_table.py)
  4) config.yaml 의 train_table_name 을 새 테이블로 변경 -> train.py 실행

Kisantal 논문 원칙 중 채택한 것:
  - 같은 이미지 안에 복제 (배경/맥락 유지, 마스크 불필요)
  - 붙일 때 기존 객체와 겹치지 않게 보장 (IoU)
  - 잘리지 않은(fully-visible) 객체만 복제 소스로 사용
변경한 것:
  - 'small object'가 아니라 '희소 클래스' 객체를 복제 대상으로 삼음
    (지금 문제는 객체 크기가 아니라 car/bus/van 간 개수 불균형이므로)

사용:
  python copy_paste_aug.py \
      --images dataset/images/train --labels dataset/labels/train \
      --out-images dataset_aug/images/train --out-labels dataset_aug/labels/train \
      --target-classes 1 3 --oversample 3
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2


# ----------------------- 라벨 입출력 -----------------------
def load_labels(label_path: Path):
    """YOLO txt -> [(cls, cx, cy, w, h), ...] 정규화 좌표 (float 유지)"""
    rows = []
    if not label_path.exists():
        return rows
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 5:
            rows.append((int(float(parts[0])),) + tuple(map(float, parts[1:])))
    return rows


def save_labels(label_path: Path, rows):
    lines = [f"{int(c)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
             for c, cx, cy, w, h in rows]
    label_path.write_text("\n".join(lines))


# ----------------------- 좌표/IoU 유틸 -----------------------
def to_pixel(box, img_w, img_h):
    """(cls,cx,cy,w,h) 정규화 -> [x1,y1,x2,y2] 픽셀"""
    _, cx, cy, w, h = box
    return [(cx - w / 2) * img_w, (cy - h / 2) * img_h,
            (cx + w / 2) * img_w, (cy + h / 2) * img_h]


def iou(a, b):
    """두 픽셀 xyxy 박스 IoU"""
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def is_fully_visible(px, img_w, img_h, margin=2):
    """이미지 경계에 잘리지 않은 객체만 True (논문 권장: 깨끗한 소스만)"""
    x1, y1, x2, y2 = px
    return (x1 >= margin and y1 >= margin and
            x2 <= img_w - margin and y2 <= img_h - margin)


# ----------------------- 핵심: 이미지 내 copy-paste -----------------------
def copy_paste_within_image(img, labels, target_classes,
                            n_paste_range=(1, 3),
                            iou_thresh=0.0,
                            max_tries=40):
    """
    한 이미지 안에서 희소 클래스 객체를 IoU 겹침 없이 같은 이미지 다른 위치에 복제.

    img            : BGR 이미지 (numpy)
    labels         : [(cls,cx,cy,w,h), ...] 정규화
    target_classes : 복제할 희소 클래스 id 집합
    반환            : (증강 이미지, 증강 라벨 리스트)
    """
    h_img, w_img = img.shape[:2]
    out_img = img.copy()
    out_labels = list(labels)

    # 현재 이미지의 모든 박스 (픽셀) — 붙일 때마다 갱신됨
    occupied = [to_pixel(b, w_img, h_img) for b in labels]

    # 복제 소스 = 희소 클래스 & 잘리지 않은 객체
    donors = []
    for b in labels:
        if int(b[0]) not in target_classes:
            continue
        px = to_pixel(b, w_img, h_img)
        if is_fully_visible(px, w_img, h_img):
            donors.append((b, px))

    for box, px in donors:
        x1, y1, x2, y2 = (int(round(v)) for v in px)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)
        patch = img[y1:y2, x1:x2]
        if patch.size == 0:
            continue
        pw, ph = x2 - x1, y2 - y1
        if pw >= w_img or ph >= h_img:
            continue  # 객체가 이미지만큼 크면 붙일 자리가 없음

        for _ in range(random.randint(*n_paste_range)):
            placed = False
            for _ in range(max_tries):
                nx1 = random.randint(0, w_img - pw)
                ny1 = random.randint(0, h_img - ph)
                new_box = [nx1, ny1, nx1 + pw, ny1 + ph]
                if all(iou(new_box, o) <= iou_thresh for o in occupied):
                    placed = True
                    break
            if not placed:
                continue  # 빈 자리 못 찾으면 이 복제는 포기 (억지로 겹치지 않음)

            out_img[ny1:ny1 + ph, nx1:nx1 + pw] = patch
            occupied.append(new_box)
            cx = (nx1 + pw / 2) / w_img
            cy = (ny1 + ph / 2) / h_img
            out_labels.append((box[0], cx, cy, pw / w_img, ph / h_img))

    return out_img, out_labels


# ----------------------- 데이터셋 전체 처리 -----------------------
def build_dataset(images_dir, labels_dir, out_images_dir, out_labels_dir,
                  target_classes, oversample, img_ext, seed):
    random.seed(seed)
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    out_images_dir = Path(out_images_dir)
    out_labels_dir = Path(out_labels_dir)
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)
    target_classes = set(target_classes)

    n_orig = n_aug = n_skipped = 0
    pasted_total = 0

    for img_path in sorted(images_dir.glob(f"*{img_ext}")):
        label_path = labels_dir / (img_path.stem + ".txt")
        labels = load_labels(label_path)
        img = cv2.imread(str(img_path))
        if img is None:
            n_skipped += 1
            continue

        # 1) 원본은 항상 그대로 복사 (원본 분포 보존)
        shutil.copy(img_path, out_images_dir / img_path.name)
        save_labels(out_labels_dir / (img_path.stem + ".txt"), labels)
        n_orig += 1

        # 2) 희소 클래스 포함 이미지만 추가 증강본 생성 (= 오버샘플링)
        has_target = any(int(b[0]) in target_classes for b in labels)
        if not has_target:
            continue

        for k in range(oversample - 1):
            aug_img, aug_labels = copy_paste_within_image(
                img, labels, target_classes)
            pasted_total += len(aug_labels) - len(labels)
            stem = f"{img_path.stem}_cp{k}"
            cv2.imwrite(str(out_images_dir / (stem + img_ext)), aug_img)
            save_labels(out_labels_dir / (stem + ".txt"), aug_labels)
            n_aug += 1

    print("=" * 70)
    print("Copy-Paste 증강 완료")
    print("=" * 70)
    print(f"  원본 복사       : {n_orig}")
    print(f"  증강본 생성     : {n_aug}")
    print(f"  추가된 인스턴스 : {pasted_total} (희소 클래스 객체)")
    print(f"  읽기 실패       : {n_skipped}")
    print(f"  출력 이미지     : {out_images_dir}")
    print(f"  출력 라벨       : {out_labels_dir}")
    print("=" * 70)
    print("\n다음 단계:")
    print("  1) python diagnose.py --images <out-images> --labels <out-labels>")
    print("     -> 증강 후 클래스 분포가 개선됐는지 재확인")
    print("  2) 증강 폴더를 3LC 테이블로 등록 후 config.yaml 테이블명 변경")
    print("  3) copy-paste 없는 baseline 과 mAP A/B 비교 (특히 car mAP 하락 여부)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="원본 이미지 폴더 (train)")
    ap.add_argument("--labels", required=True, help="원본 YOLO 라벨 폴더 (train)")
    ap.add_argument("--out-images", required=True, help="증강 이미지 출력 폴더")
    ap.add_argument("--out-labels", required=True, help="증강 라벨 출력 폴더")
    ap.add_argument("--target-classes", type=int, nargs="+", required=True,
                    help="복제할 희소 클래스 id (예: --target-classes 1 3)")
    ap.add_argument("--oversample", type=int, default=3,
                    help="희소 클래스 포함 이미지 총 등장 배수 (기본 3)")
    ap.add_argument("--ext", default=".jpg", help="이미지 확장자")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.oversample < 2:
        raise SystemExit("--oversample 은 2 이상이어야 증강본이 생깁니다.")

    build_dataset(args.images, args.labels, args.out_images, args.out_labels,
                  args.target_classes, args.oversample, args.ext, args.seed)


if __name__ == "__main__":
    main()
