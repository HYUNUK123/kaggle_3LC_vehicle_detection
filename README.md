# UA-DETRAC vehicle detection - starter kit

Self-contained folder for **local training** and **Kaggle submission**.  
**YOLOv8n** only · **3LC** tables · CSV submission **`id`**, **`image_id`**, **`prediction_string`**.

Install Python packages per the competition **Environment Setup** page (e.g. `3lc-ultralytics`, `umap-learn`, `torch`, `PyYAML`, `tqdm`). **Git is not required** for this kit.

---

## Quick start (after environment setup)

```bash
python verify_setup.py       # check environment before anything else
3lc login YOUR_API_KEY        # one-time per machine
python register_tables.py     # create 3LC tables (train + val)
python train.py               # train YOLOv8n from scratch
python predict.py             # generate submission.csv
# Upload submission.csv to Kaggle
```

### Expected timings

| Step | GPU (~8 GB VRAM) | CPU only |
|------|-----------------|----------|
| `verify_setup.py` | ~2 sec | ~2 sec |
| `register_tables.py` | ~15 sec | ~15 sec |
| `train.py` (10 epochs) | **~25 min** | **~2–4 hours** |
| `predict.py` (982 images) | ~25 sec | ~5 min |

If training seems stuck at the start, **wait 60 seconds** — PyTorch, 3LC, and data-loader workers need time to initialize. This is normal.

**CPU-only?** See the commented-out overrides at the bottom of `config.yaml` for recommended CPU settings.

---

## Folder layout (what you should have)

Everything below lives in **this same directory** (the kit root):

```
├── README.md
├── config.yaml              # Paths + hyperparameters (edit here only)
├── dataset.yaml             # YOLO paths → data/train, data/val, data/test
├── sample_submission.csv    # Official row ids + image_ids (from competition Data)
├── verify_setup.py          # Pre-flight environment check (run first!)
├── register_tables.py
├── train.py
├── predict.py
└── data/
    ├── train/
    │   ├── images/          # *.jpg (or png, …)
    │   └── labels/          # one .txt per image (YOLO: class xc yc w h)
    ├── val/
    │   ├── images/
    │   └── labels/
    └── test/
        └── images/          # test only — no labels folder for participants
```

**Paths:** All scripts set their working directory to **the folder that contains this README** (`Path(__file__).parent`). Run:

```bash
cd /path/to/this/folder
python register_tables.py
```

On **Kaggle Notebooks**, after copying or unzipping the kit into e.g. `/kaggle/working/starter`:

```python
import os
os.chdir("/kaggle/working/starter")  # path where these files live
```

Then run the same `python ...` commands from that directory.

---

## End-to-end workflow

0. **`python verify_setup.py`** — checks Python, packages, GPU, and data files. Fix any `[FAIL]` items before continuing.
1. **`3lc login YOUR_API_KEY`**
2. **`python register_tables.py`** — creates **train** and **val** 3LC tables only (idempotent; does not overwrite). **Test** images stay on disk and are **not** registered in 3LC.
3. **`python train.py`** — **YOLOv8n from scratch** (`yolov8n.yaml`); weights: `runs/detect/<run_name>/weights/best.pt`.
4. **`python predict.py`** — reads `sample_submission.csv`, writes `submission.csv`. Default pipeline: `memory` (good on ~8 GB GPUs).
5. **Upload `submission.csv`** on Kaggle.

---

## Configuration (`config.yaml`)

| Section | Purpose |
|---------|---------|
| `paths` | `dataset_yaml`, `sample_submission`, `submission_csv`, `test_images`, `runs_detect_root` - all **relative to this kit root** unless you use an absolute path. |
| `tlc` | `project_name`, `dataset_name`, `train_table_name`, `val_table_name`, `image_embeddings_dim` |
| `training` | `model` must stay **`yolov8n`**; epochs, batch, device, etc. There is **no** pretrained-weights switch — **`train.py`** always uses **`yolov8n.yaml`**. |
| `predict` | `pipeline`: **`memory`** or **`txt`**; `batch`, conf, iou |

Do **not** hardcode paths inside the `.py` files; change **`config.yaml`** instead.

---

## Submission CSV

Must match **`sample_submission.csv`**: columns **`id`**, **`image_id`**, **`prediction_string`**.  
Per image, **`prediction_string`** is space-separated **`class conf xc yc w h`** (normalized YOLO), repeated for each box, or the literal **`no box`**.

---

## Before you zip this kit for upload

**Include:** all of the layout above (with real images/labels as distributed by the competition host).  
**Exclude** (optional, smaller zip / no leakage):

- `runs/` (local training outputs)
- `predictions/` (temporary predict output)
- `submission.csv`, `tables_used.txt` (local artifacts)
- any `labels_backup_*` folders if you created exports locally

**No `.git` folder** is needed for participants.

---

## Troubleshooting

| Issue | What to try |
|-------|-------------|
| `dataset.yaml` not found | Run scripts from the kit root; check `paths.dataset_yaml` in `config.yaml`. |
| CUDA OOM on `predict.py` | New terminal after training; `predict.batch: 1`; keep `predict.pipeline: memory`; or `predict.device: cpu`. |
| Wrong submission shape | Re-download **`sample_submission.csv`** from the competition **Data** tab. |
| Crash at "Reducing image embeddings" | `pip install umap-learn`, or set `image_embeddings_dim: 0` in `config.yaml` to skip. |
| Training seems frozen at start | Normal — PyTorch and data loaders take up to 60 sec to initialize. Wait. |
| `CUDA: False` despite having GPU | Install CUDA PyTorch **before** `3lc-ultralytics`. See **Environment Setup**. |
| Not sure if setup is correct | Run **`python verify_setup.py`** to check everything at once. |

---

## License / data use

Follow the official competition **Rules** and data license.
