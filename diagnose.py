#!/usr/bin/env python3
"""
클래스 불균형 진단 스크립트.

YOLO 포맷 데이터셋(images/, labels/)을 읽어서:
  - 클래스별 인스턴스 개수 (개수 불균형 확인)
  - 클래스별 bbox 크기 분포 (small object 문제 확인)
  - 클래스를 포함한 이미지 수
를 출력한다.

이 결과로 copy-paste 대상 클래스(target_classes)와
오버샘플링 배수(oversample_factor)를 정한다.

사용:
  python diagnose.py --images dataset/images/train --labels dataset/labels/train
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# data.yaml 의 names 순서와 일치 (0:truck 1:car 2:van 3:bus)
DEFAULT_CLASS_NAMES = {0: "truck", 1: "car", 2: "van", 3: "bus"}


def load_labels(label_path: Path):
    """YOLO txt -> [(cls, cx, cy, w, h), ...] 정규화 좌표"""
    rows = []
    if not label_path.exists():
        return rows
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 5:
            cls = int(float(parts[0]))
            cx, cy, w, h = map(float, parts[1:])
            rows.append((cls, cx, cy, w, h))
    return rows


def diagnose(images_dir: str, labels_dir: str, img_ext: str,
             class_names: dict):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    instance_count = defaultdict(int)          # 클래스별 인스턴스 수
    image_count = defaultdict(int)             # 클래스별 포함 이미지 수
    areas_px = defaultdict(list)               # 클래스별 bbox 면적(픽셀^2)
    side_px = defaultdict(list)                # 클래스별 bbox 짧은 변(픽셀)
    total_images = 0
    images_with_label = 0

    label_files = sorted(labels_dir.glob("*.txt"))
    for label_path in label_files:
        img_path = images_dir / (label_path.stem + img_ext)
        if not img_path.exists():
            continue
        total_images += 1
        rows = load_labels(label_path)
        if not rows:
            continue
        images_with_label += 1

        # 이미지 크기 (면적을 픽셀로 환산하려면 필요)
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h_img, w_img = img.shape[:2]

        classes_here = set()
        for cls, cx, cy, w, h in rows:
            instance_count[cls] += 1
            classes_here.add(cls)
            bw, bh = w * w_img, h * h_img
            areas_px[cls].append(bw * bh)
            side_px[cls].append(min(bw, bh))
        for cls in classes_here:
            image_count[cls] += 1

    # ---------- 출력 ----------
    print("=" * 78)
    print("클래스 불균형 진단 결과")
    print("=" * 78)
    print(f"전체 이미지: {total_images} | 라벨 있는 이미지: {images_with_label}\n")

    all_classes = sorted(instance_count.keys())
    if not all_classes:
        print("라벨을 찾지 못했습니다. 경로/확장자를 확인하세요.")
        return

    total_inst = sum(instance_count.values())
    max_inst = max(instance_count.values())

    print(f"{'class':<14}{'instances':>10}{'share':>9}{'imgs':>8}"
          f"{'imbalance':>11}{'med_area':>11}{'med_side':>10}{'small%':>9}")
    print("-" * 78)
    for cls in all_classes:
        name = class_names.get(cls, f"id{cls}")
        n = instance_count[cls]
        share = 100 * n / total_inst
        imb = max_inst / n  # 다수 클래스 대비 배율
        med_area = float(np.median(areas_px[cls]))
        med_side = float(np.median(side_px[cls]))
        # COCO 기준 small = 면적 < 32x32=1024 픽셀^2
        small_ratio = 100 * np.mean(np.array(areas_px[cls]) < 1024)
        print(f"{name:<14}{n:>10}{share:>8.1f}%{image_count[cls]:>8}"
              f"{imb:>10.1f}x{med_area:>11.0f}{med_side:>9.0f}px"
              f"{small_ratio:>8.1f}%")

    print("-" * 78)
    print(f"{'TOTAL':<14}{total_inst:>10}")
    print()
    print("해석 가이드:")
    print("  - imbalance: 다수 클래스 대비 몇 배 적은지. 3x 이상이면 개수 불균형 대상.")
    print("  - small%   : 면적 32x32px 미만 비율. 높으면 small object 문제도 동반.")
    print("  - 개수 부족(imbalance 큼) -> copy-paste target_classes 후보")
    print("  - small%만 높음        -> copy-paste보다 imgsz 상향이 우선")
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="이미지 폴더")
    ap.add_argument("--labels", required=True, help="YOLO 라벨 폴더")
    ap.add_argument("--ext", default=".jpg", help="이미지 확장자")
    args = ap.parse_args()
    diagnose(args.images, args.labels, args.ext, DEFAULT_CLASS_NAMES)


if __name__ == "__main__":
    main()
