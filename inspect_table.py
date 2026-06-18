#!/usr/bin/env python3
"""
3LC train table 의 행 구조(특히 bbox)를 확인하기 위한 일회용 스니펫.
deduplicate.py 와 동일한 방식으로 테이블을 로드한다.

이 스크립트는 아무것도 수정하지 않는다. 출력만 보고 알려주세요.

Usage:
    python inspect_table.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import tlc
import yaml

WORK_DIR = Path(__file__).resolve().parent


def _load_config() -> dict:
    cfg_path = WORK_DIR / "config.yaml"
    if not cfg_path.is_file():
        print(f"ERROR: Missing {cfg_path}", file=sys.stderr)
        sys.exit(1)
    with cfg_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _safe(obj):
    """출력용: JSON 직렬화 안 되는 객체는 타입+repr 로 대체"""
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return f"<{type(obj).__name__}> {repr(obj)[:200]}"


def main() -> int:
    cfg = _load_config()
    tlc_cfg = cfg.get("tlc", {})
    project_name = str(tlc_cfg.get("project_name", "ua_detrac_vehicle_detection"))
    dataset_name = str(tlc_cfg.get("dataset_name", "ua_detrac_10k"))
    train_name = str(tlc_cfg.get("train_table_name", f"{dataset_name}-train"))

    print("=" * 70)
    print("3LC Table 구조 점검")
    print("=" * 70)
    print(f"  project: {project_name}")
    print(f"  dataset: {dataset_name}")
    print(f"  table:   {train_name}")

    table = tlc.Table.from_names(
        project_name=project_name,
        dataset_name=dataset_name,
        table_name=train_name,
    ).latest()

    print(f"  rows:    {len(table)}")
    print(f"  type:    {type(table).__name__}")

    # ---- row_schema 의 컬럼 이름들 ----
    print("\n[row_schema] 컬럼 키:")
    try:
        schema = table.row_schema
        # schema 구조는 버전마다 다르므로 여러 방식 시도
        if hasattr(schema, "values"):
            for k in schema.values.keys():
                print(f"   - {k}")
        else:
            print(f"   schema repr: {repr(schema)[:500]}")
    except Exception as e:
        print(f"   (schema 접근 실패: {e})")

    # ---- 첫 행 통째로 ----
    print("\n[table_rows[0]] 전체 키:")
    row = table.table_rows[0]
    if hasattr(row, "keys"):
        for k in row.keys():
            v = row.get(k)
            print(f"   - {k!r}: type={type(v).__name__}")
    else:
        print(f"   row type={type(row).__name__}, repr={repr(row)[:300]}")

    # ---- bbox 로 보이는 키 찾아서 깊게 출력 ----
    print("\n[bbox 후보 키] 상세:")
    bbox_like = []
    if hasattr(row, "keys"):
        for k in row.keys():
            kl = str(k).lower()
            if any(t in kl for t in ("bb", "box", "label", "annot", "object")):
                bbox_like.append(k)

    if not bbox_like:
        print("   bbox 로 보이는 키를 못 찾음 — 아래 첫 행 전체를 확인하세요.")
    for k in bbox_like:
        v = row.get(k)
        print(f"\n   key = {k!r}")
        print(f"   type = {type(v).__name__}")
        print(f"   value = {_safe(v)}")

    # ---- 첫 행 전체 덤프 (fallback) ----
    print("\n[첫 행 전체 덤프]:")
    if hasattr(row, "keys"):
        dump = {k: _safe(row.get(k)) for k in row.keys()}
        print(json.dumps(dump, indent=2, ensure_ascii=False, default=str)[:2000])
    else:
        print(repr(row)[:2000])

    # ---- 이미지 경로 한 개 확인 ----
    print("\n[이미지 경로 샘플]:")
    img = row.get("image") if hasattr(row, "get") else None
    print(f"   raw type: {type(img).__name__}")
    if hasattr(img, "to_absolute_path"):
        print(f"   to_absolute_path(): {img.to_absolute_path()}")
    else:
        print(f"   repr: {repr(img)[:200]}")

    print("\n" + "=" * 70)
    print("이 출력 전체를 복사해서 알려주세요. (특히 bbox 후보 키 부분)")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
