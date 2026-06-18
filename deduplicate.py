#!/usr/bin/env python3
"""
3LC table에서 cosine similarity 기반으로 중복 이미지 제거.
이미지에서 직접 feature 추출 → embedding 생성 → dedup.

Usage:
    python deduplicate.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import tlc
import yaml
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

WORK_DIR = Path(__file__).resolve().parent


def _load_config() -> dict:
    cfg_path = WORK_DIR / "config.yaml"
    if not cfg_path.is_file():
        print(f"ERROR: Missing {cfg_path}", file=sys.stderr)
        sys.exit(1)
    with cfg_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_image_paths_from_table(table: tlc.Table) -> list[Path]:
    """3LC table에서 이미지 경로 추출"""
    paths = []
    for i in range(len(table)):
        row = table.table_rows[i]
        img = row.get("image")
        
        # 3LC URL 또는 경로 처리
        if hasattr(img, 'to_absolute_path'):
            path = Path(img.to_absolute_path())
        elif isinstance(img, str):
            path = Path(img)
        elif hasattr(img, 'path'):
            path = Path(img.path)
        else:
            path = Path(str(img))
        
        paths.append(path)
    return paths


def extract_simple_embeddings(
    image_paths: list[Path],
    resize_to: int = 64,
    batch_size: int = 32,
) -> np.ndarray:
    """
    간단한 이미지 embedding 추출.
    
    이미지를 작게 리사이즈하고 픽셀값을 flatten한 후 normalize.
    학습/추론 안 쓰는 단순 통계 기반이라 대회 규칙 위반 아님.
    
    Args:
        image_paths: 이미지 경로 리스트
        resize_to: 리사이즈 크기 (작을수록 빠름, 32~128 권장)
    
    Returns:
        embeddings: (N, resize_to*resize_to*3) numpy array
    """
    print(f"\n[Embedding] Extracting from {len(image_paths)} images...")
    print(f"            Method: pixel-based (resize to {resize_to}x{resize_to})")
    
    embeddings = []
    
    for path in tqdm(image_paths, desc="Processing"):
        try:
            img = Image.open(path).convert("RGB")
            img = img.resize((resize_to, resize_to), Image.BILINEAR)
            arr = np.array(img, dtype=np.float32) / 255.0
            
            # 평탄화 + L2 정규화
            vec = arr.flatten()
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            
            embeddings.append(vec)
        except Exception as e:
            print(f"\nWARN: Failed to load {path}: {e}")
            # 실패한 이미지는 0 벡터 (dedup에서 안 걸리도록)
            embeddings.append(np.zeros(resize_to * resize_to * 3, dtype=np.float32))
    
    return np.array(embeddings)


def extract_yolov8_embeddings(
    image_paths: list[Path],
    model_path: str = "runs/detect/yolov8n_baseline/weights/best.pt",
    batch_size: int = 16,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> np.ndarray:
    """
    학습된 YOLOv8n으로 embedding 추출.
    Backbone의 마지막 feature를 사용.
    
    더 정교한 dedup 가능하지만 학습된 모델 필요.
    """
    from ultralytics import YOLO
    
    print(f"\n[Embedding] Loading YOLOv8n from {model_path}")
    
    if not Path(model_path).exists():
        print(f"ERROR: Model not found at {model_path}")
        print(f"       Please train first or use extract_simple_embeddings()")
        sys.exit(1)
    
    model = YOLO(model_path)
    model.model.to(device).eval()
    
    print(f"            Device: {device}")
    print(f"            Extracting features from {len(image_paths)} images...")
    
    embeddings = []
    
    with torch.no_grad():
        for path in tqdm(image_paths, desc="Processing"):
            try:
                img = Image.open(path).convert("RGB")
                img = img.resize((640, 640), Image.BILINEAR)
                arr = np.array(img, dtype=np.float32) / 255.0
                tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
                
                # Backbone 통과
                # YOLOv8 모델의 backbone feature 추출
                features = []
                x = tensor
                for layer in model.model.model[:10]:  # backbone만
                    x = layer(x)
                    features.append(x)
                
                # 마지막 feature를 global pooling으로 vector화
                last_feat = features[-1]  # (1, C, H, W)
                pooled = torch.nn.functional.adaptive_avg_pool2d(last_feat, 1)
                vec = pooled.squeeze().cpu().numpy()  # (C,)
                
                # L2 normalize
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                
                embeddings.append(vec)
                
            except Exception as e:
                print(f"\nWARN: Failed {path}: {e}")
                embeddings.append(np.zeros(256, dtype=np.float32))
    
    return np.array(embeddings)


def dedup_by_cosine_similarity(
    embeddings: np.ndarray,
    threshold: float = 0.989,
) -> np.ndarray:
    """
    Cosine similarity 기반 dedup (greedy).
    
    Returns:
        keep_mask: (N,) bool array
    """
    n = len(embeddings)
    print(f"\n[Dedup] Computing similarity matrix (N={n})...")
    
    memory_mb = (n * n * 4) / (1024 * 1024)
    print(f"        Memory: ~{memory_mb:.0f} MB")
    
    sim_matrix = cosine_similarity(embeddings)
    np.fill_diagonal(sim_matrix, 0)
    
    keep = np.ones(n, dtype=bool)
    n_removed = 0
    
    print(f"[Dedup] Removing duplicates (threshold={threshold})...")
    
    for i in tqdm(range(n), desc="Dedup"):
        if not keep[i]:
            continue
        similar = sim_matrix[i, i+1:] > threshold
        to_remove = np.where(similar)[0] + (i + 1)
        for j in to_remove:
            if keep[j]:
                keep[j] = False
                n_removed += 1
    
    n_kept = keep.sum()
    print(f"\n[Dedup] Result:")
    print(f"        Original: {n}")
    print(f"        Removed:  {n_removed} ({n_removed/n*100:.1f}%)")
    print(f"        Kept:     {n_kept} ({n_kept/n*100:.1f}%)")
    
    return keep


def deduplicate_table(
    project_name: str,
    dataset_name: str,
    table_name: str,
    threshold: float = 0.986,
    method: str = "simple",
    model_path: str = "runs/detect/yolov8n_baseline/weights/best.pt",
) -> Optional[tlc.Table]:
    """
    3LC table에서 중복 제거.
    TableFromYolo 호환 버전.
    """
    print("=" * 70)
    print("3LC Table Deduplication (Cosine Similarity)")
    print("=" * 70)
    
    print(f"\nLoading table:")
    print(f"  project: {project_name}")
    print(f"  dataset: {dataset_name}")
    print(f"  table:   {table_name}")
    
    table = tlc.Table.from_names(
        project_name=project_name,
        dataset_name=dataset_name,
        table_name=table_name,
    ).latest()
    
    print(f"  rows:    {len(table)}")
    print(f"  type:    {type(table).__name__}")
    
    # 이미지 경로 추출
    print(f"\nExtracting image paths...")
    image_paths = get_image_paths_from_table(table)
    print(f"  paths:   {len(image_paths)}")
    
    # Embedding 추출
    if method == "simple":
        embeddings = extract_simple_embeddings(image_paths, resize_to=64)
    elif method == "yolov8":
        embeddings = extract_yolov8_embeddings(image_paths, model_path=model_path)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    print(f"\n  shape:   {embeddings.shape}")
    
    # Dedup 실행
    keep_mask = dedup_by_cosine_similarity(embeddings, threshold=threshold)
    
    keep_indices = np.where(keep_mask)[0].tolist()
    
    print(f"\n[Final] Keeping {len(keep_indices)} / {len(table)} rows")
    
    # 안전장치: 너무 많이 제거됐는지 확인
    keep_ratio = len(keep_indices) / len(table)
    if keep_ratio < 0.3:
        print(f"\n⚠️  WARNING: Only {keep_ratio*100:.1f}% kept!")
        print(f"   This is likely too aggressive.")
        print(f"   Recommended: increase threshold to 0.98 or 0.99")
        
        response = input("\nContinue anyway? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            return None
    
    # ⭐ TableFromYolo는 revise/filter_indices가 없으므로
    # TableWriter로 새 table을 직접 생성
    print(f"\nCreating new table via TableWriter...")
    
    try:
        new_table = _create_filtered_yolo_table(
            original_table=table,
            keep_indices=keep_indices,
            threshold=threshold,
            method=method,
        )
        
        print(f"\nNew table created:")
        print(f"  url:  {new_table.url}")
        print(f"  name: {new_table.name}")
        print(f"  size: {len(new_table)}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"1. Update config.yaml:")
    print(f"   tlc:")
    print(f"     train_table_name: {new_table.name}")
    print(f"2. Re-train: python train.py")
    
    return new_table


def _create_filtered_yolo_table(
    original_table: tlc.Table,
    keep_indices: list[int],
    threshold: float,
    method: str,
) -> tlc.Table:
    """
    TableFromYolo에서 row 일부만 골라 새 table 생성.
    
    Strategy: 원본 table을 읽어서 필요한 row들을 새 TableWriter에 추가.
    """
    keep_set = set(keep_indices)
    
    # 새 table 이름
    new_table_name = f"{original_table.name}-dedup-{int(threshold*100)}"
    
    print(f"  new table name: {new_table_name}")
    
    # 원본 table의 schema 가져오기
    schema = original_table.row_schema
    
    # 새 TableWriter 생성
    writer = tlc.TableWriter(
        project_name=original_table.project_name,
        dataset_name=original_table.dataset_name,
        table_name=new_table_name,
        column_schemas=schema,
        if_exists="overwrite",
        description=f"Deduplicated from {original_table.name} "
                    f"(threshold={threshold}, method={method})",
    )
    
    # keep_indices에 해당하는 row만 새 table에 추가
    print(f"  copying {len(keep_indices)} rows...")
    
    for i in tqdm(keep_indices, desc="Writing"):
        row = original_table.table_rows[i]
        writer.add_row(row)
    
    # 완료
    new_table = writer.finalize()
    return new_table


def main() -> int:
    cfg = _load_config()
    tlc_cfg = cfg.get("tlc", {})
    
    project_name = str(tlc_cfg.get("project_name", "ua_detrac_vehicle_detection"))
    dataset_name = str(tlc_cfg.get("dataset_name", "ua_detrac_10k"))
    train_name = str(tlc_cfg.get("train_table_name", f"{dataset_name}-train"))
    threshold = float(tlc_cfg.get("dedup_threshold", 0.989))
    method = str(tlc_cfg.get("dedup_method", "simple"))
    
    print(f"Config:")
    print(f"  project:   {project_name}")
    print(f"  dataset:   {dataset_name}")
    print(f"  table:     {train_name}")
    print(f"  threshold: {threshold}")
    print(f"  method:    {method}")
    
    new_table = deduplicate_table(
        project_name=project_name,
        dataset_name=dataset_name,
        table_name=train_name,
        threshold=threshold,
        method=method,
    )
    
    return 0 if new_table is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())