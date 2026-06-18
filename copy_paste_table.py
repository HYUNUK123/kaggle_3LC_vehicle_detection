#!/usr/bin/env python3
"""
클래스 불균형 해소용 Copy-Paste 증강 (3LC table 버전).

deduplicate.py 의 패턴을 그대로 따른다:
  - tlc.Table.from_names(...).latest() 로 로드
  - row 는 table.table_rows[i] 로 접근
  - 새 테이블은 TableWriter + row_schema + add_row + finalize 로 생성

흐름:
  ua_detrac_10k-train-dedup-98   (dedup 끝난 테이블)
    -> [이 스크립트] 희소 클래스 copy-paste
    -> ua_detrac_10k-train-dedup-98-cp   (새 테이블)
    -> config.yaml 의 train_table_name 을 이걸로 변경 -> train.py

bbs 포맷 (inspect_table.py 로 확인됨):
  bb_list 항목 = {x0, y0, x1, y1, label}
  x0=center_x, y0=center_y, x1=width, y1=height  (전부 0~1 정규화)

Kisantal 논문 원칙 채택:
  - 같은 이미지 안에 복제 (배경/맥락 유지)
  - 기존 객체와 IoU 겹침 없이 배치
  - 잘리지 않은(fully-visible) 객체만 복제 소스로 사용
변경:
  - 'small object' 가 아니라 '희소 클래스' 객체를 복제 대상으로 삼음

주의:
  - 이 스크립트는 train 테이블에만 적용. val 테이블에는 절대 쓰지 말 것.
  - 새 이미지 파일은 data/train/images_cp/ 에 저장된다 (원본은 그대로 둠).

Usage:
    python copy_paste_table.py --target-classes 0 3 --oversample 3
    python copy_paste_table.py --target-classes 0 2 3 --oversample 3 \
        --source-table ua_detrac_10k-train-dedup-98
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import tlc
import yaml
from tqdm import tqdm

WORK_DIR = Path(__file__).resolve().parent
CLASS_NAMES = {0: "truck", 1: "car", 2: "van", 3: "bus"}


def _load_config() -> dict:
    cfg_path = WORK_DIR / "config.yaml"
    if not cfg_path.is_file():
        print(f"ERROR: Missing {cfg_path}", file=sys.stderr)
        sys.exit(1)
    with cfg_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ----------------------- bbs <-> 내부 표현 -----------------------
def bbs_to_boxes(bbs: dict):
    """3LC bbs dict -> [(label,cx,cy,w,h), ...] 정규화"""
    boxes = []
    for bb in bbs.get("bb_list", []):
        boxes.append((int(bb["label"]),
                      float(bb["x0"]), float(bb["y0"]),
                      float(bb["x1"]), float(bb["y1"])))
    return boxes


def boxes_to_bb_list(boxes):
    """[(label,cx,cy,w,h), ...] -> 3LC bb_list (키 이름 x0,y0,x1,y1 유지)"""
    return [
        {"x0": cx, "y0": cy, "x1": w, "y1": h, "label": int(label)}
        for (label, cx, cy, w, h) in boxes
    ]


# ----------------------- 좌표 / IoU -----------------------
def to_pixel(box, img_w, img_h):
    """(label,cx,cy,w,h) 정규화 -> [x1,y1,x2,y2] 픽셀 (모서리)"""
    _, cx, cy, w, h = box
    return [(cx - w / 2) * img_w, (cy - h / 2) * img_h,
            (cx + w / 2) * img_w, (cy + h / 2) * img_h]


def iou(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def is_fully_visible(px, img_w, img_h, margin=2):
    x1, y1, x2, y2 = px
    return (x1 >= margin and y1 >= margin and
            x2 <= img_w - margin and y2 <= img_h - margin)


# ----------------------- 핵심 copy-paste -----------------------
def copy_paste_within_image(img, boxes, target_classes,
                            n_paste_range=(1, 3),
                            iou_thresh=0.0,
                            max_tries=40):
    """
    한 이미지 안에서 희소 클래스 객체를 IoU 겹침 없이 복제.
    img   : BGR numpy
    boxes : [(label,cx,cy,w,h), ...] 정규화
    반환   : (증강 이미지, 증강 boxes, 추가된 객체 수)
    """
    h_img, w_img = img.shape[:2]
    out_img = img.copy()
    out_boxes = list(boxes)
    occupied = [to_pixel(b, w_img, h_img) for b in boxes]
    added = 0

    donors = []
    for b in boxes:
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
            continue

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
                continue
            out_img[ny1:ny1 + ph, nx1:nx1 + pw] = patch
            occupied.append(new_box)
            cx = (nx1 + pw / 2) / w_img
            cy = (ny1 + ph / 2) / h_img
            out_boxes.append((box[0], cx, cy, pw / w_img, ph / h_img))
            added += 1

    return out_img, out_boxes, added


# ----------------------- 새 테이블 생성 -----------------------
def build_cp_table(source_table: tlc.Table, target_classes, oversample,
                    out_img_dir: Path, seed: int) -> tlc.Table:
    random.seed(seed)
    out_img_dir.mkdir(parents=True, exist_ok=True)
    target_classes = set(target_classes)

    new_table_name = f"{source_table.name}-cp"
    print(f"\n새 테이블: {new_table_name}")
    print(f"증강 이미지 저장 위치: {out_img_dir}")

    writer = tlc.TableWriter(
        project_name=source_table.project_name,
        dataset_name=source_table.dataset_name,
        table_name=new_table_name,
        column_schemas=source_table.row_schema,
        if_exists="overwrite",
        description=(f"Copy-paste augmented from {source_table.name} "
                     f"(targets={sorted(target_classes)}, oversample={oversample})"),
    )

    n = len(source_table)
    n_orig = n_aug = total_added = 0

    for i in tqdm(range(n), desc="Copy-paste"):
        row = source_table.table_rows[i]

        # 1) 원본 행은 항상 그대로 보존
        writer.add_row(row)
        n_orig += 1

        bbs = row.get("bbs") or {}
        boxes = bbs_to_boxes(bbs)
        has_target = any(int(b[0]) in target_classes for b in boxes)
        if not has_target:
            continue

        # 원본 이미지 로드
        img_path = row.get("image")
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"\nWARN: 이미지 로드 실패, 건너뜀: {img_path}")
            continue

        # 2) 증강본 생성 (oversample-1 개)
        src_stem = Path(str(img_path)).stem
        for k in range(oversample - 1):
            aug_img, aug_boxes, added = copy_paste_within_image(
                img, boxes, target_classes)
            total_added += added

            new_name = f"{src_stem}_cp{k}.jpg"
            new_path = out_img_dir / new_name
            cv2.imwrite(str(new_path), aug_img)

            # 새 행 구성 — 원본 행 키를 그대로 복제하고 image/bbs 만 교체
            new_row = dict(row)
            new_row["image"] = str(new_path).replace("\\", "/")
            new_row["bbs"] = {
                "image_width": bbs.get("image_width", img.shape[1]),
                "image_height": bbs.get("image_height", img.shape[0]),
                "bb_list": boxes_to_bb_list(aug_boxes),
            }
            writer.add_row(new_row)
            n_aug += 1

    new_table = writer.finalize()

    print("\n" + "=" * 70)
    print("Copy-Paste 증강 완료")
    print("=" * 70)
    print(f"  원본 행          : {n_orig}")
    print(f"  증강 행          : {n_aug}")
    print(f"  추가된 인스턴스  : {total_added} (희소 클래스)")
    print(f"  새 테이블        : {new_table.name}")
    print(f"  테이블 크기      : {len(new_table)}")
    print(f"  url              : {new_table.url}")
    return new_table


def main() -> int:
    cfg = _load_config()
    tlc_cfg = cfg.get("tlc", {})
    project_name = str(tlc_cfg.get("project_name", "ua_detrac_vehicle_detection"))
    dataset_name = str(tlc_cfg.get("dataset_name", "ua_detrac_10k"))
    default_src = str(tlc_cfg.get("train_table_name", f"{dataset_name}-train"))

    ap = argparse.ArgumentParser()
    ap.add_argument("--target-classes", type=int, nargs="+", required=True,
                    help="복제할 희소 클래스 id (예: --target-classes 0 3)")
    ap.add_argument("--oversample", type=int, default=3,
                    help="희소 클래스 이미지 총 등장 배수 (기본 3, 2 이상)")
    ap.add_argument("--source-table", default=default_src,
                    help="입력 테이블 (기본: config.yaml train_table_name). "
                         "dedup 된 테이블을 지정하세요.")
    ap.add_argument("--out-dir", default="data/train/images_cp",
                    help="증강 이미지 저장 폴더 (competition_starter 기준 상대경로)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.oversample < 2:
        raise SystemExit("--oversample 은 2 이상이어야 증강본이 생깁니다.")

    names = [CLASS_NAMES.get(c, f"id{c}") for c in args.target_classes]
    print(f"입력 테이블 : {args.source_table}")
    print(f"증강 대상   : {args.target_classes} ({', '.join(names)})")
    print(f"oversample  : {args.oversample}")

    source_table = tlc.Table.from_names(
        project_name=project_name,
        dataset_name=dataset_name,
        table_name=args.source_table,
    ).latest()

    out_img_dir = (WORK_DIR / args.out_dir).resolve()
    new_table = build_cp_table(
        source_table, args.target_classes, args.oversample, out_img_dir, args.seed)

    print("\n다음 단계:")
    print(f"  1) python diagnose_table.py --table {new_table.name}")
    print(f"     -> 증강 후 분포 개선 확인")
    print(f"  2) config.yaml 수정:")
    print(f"       tlc:")
    print(f"         train_table_name: {new_table.name}")
    print(f"  3) python train.py  (baseline 과 mAP A/B 비교)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
