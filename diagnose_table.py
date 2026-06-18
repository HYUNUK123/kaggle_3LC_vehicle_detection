#!/usr/bin/env python3
"""
3LC table 기반 클래스 불균형 진단.

ua_detrac_10k-train-dedup-98 같은 dedup 된 테이블을 직접 읽어서
클래스별 인스턴스 수 / 크기 분포를 출력한다.

bbs 포맷 주의:
  bb_list 의 각 항목은 {x0, y0, x1, y1, label}
  여기서 x0=center_x, y0=center_y, x1=width, y1=height  (전부 0~1 정규화)
  -> 이름은 모서리처럼 보이지만 실제로는 YOLO cx,cy,w,h 다.

Usage:
    python diagnose_table.py
    python diagnose_table.py --table ua_detrac_10k-train-dedup-98
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import tlc
import yaml

WORK_DIR = Path(__file__).resolve().parent

# data.yaml 의 names 순서와 일치 (0:truck 1:car 2:van 3:bus)
CLASS_NAMES = {0: "truck", 1: "car", 2: "van", 3: "bus"}


def _load_config() -> dict:
    cfg_path = WORK_DIR / "config.yaml"
    if not cfg_path.is_file():
        print(f"ERROR: Missing {cfg_path}", file=sys.stderr)
        sys.exit(1)
    with cfg_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def iter_boxes(table: tlc.Table):
    """
    테이블의 모든 행을 돌며 (row_index, img_w, img_h, [(label,cx,cy,w,h),...]) 산출.
    bbs 의 x0,y0,x1,y1 -> cx,cy,w,h 로 그대로 매핑.
    """
    n = len(table)
    for i in range(n):
        row = table.table_rows[i]
        bbs = row.get("bbs") or {}
        img_w = bbs.get("image_width") or row.get("width")
        img_h = bbs.get("image_height") or row.get("height")
        boxes = []
        for bb in bbs.get("bb_list", []):
            label = int(bb["label"])
            cx, cy = float(bb["x0"]), float(bb["y0"])
            w, h = float(bb["x1"]), float(bb["y1"])
            boxes.append((label, cx, cy, w, h))
        yield i, img_w, img_h, boxes


def diagnose(table: tlc.Table):
    instance_count = defaultdict(int)
    image_count = defaultdict(int)
    areas_px = defaultdict(list)
    side_px = defaultdict(list)
    total_images = 0
    images_with_label = 0

    for _, img_w, img_h, boxes in iter_boxes(table):
        total_images += 1
        if not boxes:
            continue
        images_with_label += 1
        classes_here = set()
        for label, cx, cy, w, h in boxes:
            instance_count[label] += 1
            classes_here.add(label)
            bw, bh = w * img_w, h * img_h
            areas_px[label].append(bw * bh)
            side_px[label].append(min(bw, bh))
        for c in classes_here:
            image_count[c] += 1

    print("=" * 78)
    print("클래스 불균형 진단 (3LC table)")
    print("=" * 78)
    print(f"전체 이미지: {total_images} | 라벨 있는 이미지: {images_with_label}\n")

    all_classes = sorted(instance_count.keys())
    if not all_classes:
        print("라벨을 못 찾았습니다. bbs 구조를 다시 확인하세요.")
        return

    total_inst = sum(instance_count.values())
    max_inst = max(instance_count.values())

    print(f"{'class':<14}{'instances':>10}{'share':>9}{'imgs':>8}"
          f"{'imbalance':>11}{'med_area':>11}{'med_side':>10}{'small%':>9}")
    print("-" * 78)
    for c in all_classes:
        name = CLASS_NAMES.get(c, f"id{c}")
        n = instance_count[c]
        share = 100 * n / total_inst
        imb = max_inst / n
        med_area = float(np.median(areas_px[c]))
        med_side = float(np.median(side_px[c]))
        small_ratio = 100 * np.mean(np.array(areas_px[c]) < 1024)  # 32x32px
        print(f"{name:<14}{n:>10}{share:>8.1f}%{image_count[c]:>8}"
              f"{imb:>10.1f}x{med_area:>11.0f}{med_side:>9.0f}px"
              f"{small_ratio:>8.1f}%")

    print("-" * 78)
    print(f"{'TOTAL':<14}{total_inst:>10}\n")
    print("해석:")
    print("  - imbalance 3x 이상  -> copy-paste target-classes 후보")
    print("  - small% 가 높으면   -> 개수보다 크기 문제, imgsz 상향 우선 고려")
    print("=" * 78)


def main() -> int:
    cfg = _load_config()
    tlc_cfg = cfg.get("tlc", {})
    project_name = str(tlc_cfg.get("project_name", "ua_detrac_vehicle_detection"))
    dataset_name = str(tlc_cfg.get("dataset_name", "ua_detrac_10k"))
    default_table = str(tlc_cfg.get("train_table_name", f"{dataset_name}-train"))

    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=default_table,
                    help="진단할 테이블 이름 (기본: config.yaml 의 train_table_name)")
    args = ap.parse_args()

    print(f"테이블 로드: {project_name} / {dataset_name} / {args.table}")
    table = tlc.Table.from_names(
        project_name=project_name,
        dataset_name=dataset_name,
        table_name=args.table,
    ).latest()

    diagnose(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
