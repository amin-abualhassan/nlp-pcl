# nlp-pcl

Script-first pipeline for the **“Don’t Patronize Me!” (PCL)** NLP coursework: binary classification (**PCL=1** vs **No PCL=0**).

Main ideas in the final submission pipeline:

- **DeBERTa-v3-large** backbone (`microsoft/deberta-v3-large`)
- Binary classifier head (PCL vs No PCL)
- Optional **7-category auxiliary head** (multi-label) when category labels are available
- **5-fold Stratified CV**
- **OOF threshold tuning** for **F1 on the positive class**
- Fold **ensemble inference** (default: simple mean over folds)
- Outputs in the required format: `dev.txt` / `test.txt` (one `0/1` per line)

Repository:

```text
https://github.com/amin-abualhassan/nlp-pcl
```

If Git LFS is blocked/slow for you, there is also a Google Drive fallback for the model artifacts:

```text
https://drive.google.com/drive/folders/1FAUTcZRYLRKM-L8iZt_-mGZiH9xzGy89?usp=sharing
```

---

## Repo layout (high level)

- `src/pcl_exercise/` : package code (data loading, model, training, inference)
- `configs/` : YAML configs (paths + hyperparams)
- `scripts/` : CLI scripts
- `data/` : local data files for train/dev/test
- `reports/` : EDA + local evaluation outputs used in the report
- `BestModel/` : **the submission model artifact bundle (per spec)**  
  - `BestModel/model_archive/` : split archive parts + checksum (tracked via Git LFS)
  - `BestModel/default.yaml` : config used for the best run
  - `BestModel/run_dir.txt` : points to the selected run directory name
- `models/` : **generated locally** after restoring/extracting the model artifacts (not committed)

At repo root, these are the submission files:

- `dev.txt`
- `test.txt`
- `selection_report.json` (selection metadata)

---

## Quick start (fresh clone → restore models → rebuild dev/test)

### 0) System prereqs (Ubuntu / WSL)

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs
git lfs install
```

### 1) Clone (with or without LFS smudge)

**Normal clone (recommended)**

```bash
cd ~
git clone https://github.com/amin-abualhassan/nlp-pcl.git
cd nlp-pcl
```

If you want a faster checkout and then fetch LFS explicitly:

```bash
cd ~
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/amin-abualhassan/nlp-pcl.git
cd nlp-pcl
git lfs pull
```

Sanity check:

```bash
git lfs ls-files | grep BestModel/model_archive || true
ls -lh BestModel/model_archive/
```

### 2) Create a venv

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

### 3) Install PyTorch (GPU vs CPU)

Check:

```bash
nvidia-smi || true
python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available()); print('torch cuda', torch.version.cuda)"
```

Example CUDA 12.8 wheels:

```bash
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
```

CPU-only:

```bash
pip install torch torchvision torchaudio
```

### 4) Install project deps

```bash
pip install -r requirements.txt
```

(Optional, convenient)

```bash
pip install -e .
```

### 5) Restore the saved models into `./models`

The trained fold models are stored as a split archive under `BestModel/model_archive/`.
This script reconstructs the archive, checks its SHA256, extracts into `./models/`, and deletes the temporary `.tgz`.

```bash
chmod +x scripts/restore_models.sh
bash scripts/restore_models.sh
```

You should end up with something like:

```bash
ls models/outputs/
ls models/outputs/*/models/fold0/model.pt
```

### 6) Rebuild `dev.txt` and `test.txt`

Plain mean over folds (no weighting):

```bash
python3 scripts/build_dev_test.py --fold_weighting none
```

Writes:

- `dev.txt`
- `test.txt`
- `selection_report.json`

### 7) Verify determinism

```bash
python3 scripts/build_dev_test.py --fold_weighting none --report_json selection_report_fresh.json
python3 scripts/build_dev_test.py --fold_weighting none --report_json selection_report_fresh.json
sha256sum dev.txt test.txt selection_report_fresh.json
```

`dev.txt` / `test.txt` should match exactly across machines if the same restored models + same data files are used.

---

## Training (re-running CV)

If you want to re-train the full 5-fold CV run:

```bash
python3 scripts/train_cv.py --config configs/default.yaml
```

For a faster smoke test:

```bash
python3 scripts/train_cv.py --config configs/smoke.yaml
```

---

## Evaluation scripts

### Global evaluation (dev F1)

```bash
python3 scripts/eval_dev_f1.py --dev_csv data/dev_df_2.csv --pred dev.txt
```

### Local evaluation (Stage 5.2)

This produces error analysis + slice analysis + PR/cali plots under a report folder:

```bash
python3 scripts/local_eval_stage5.py \
  --selection_report selection_report.json \
  --dev_csv data/dev_df_2.csv \
  --pred dev.txt \
  --pred_alt dev.txt.bak \
  --cats_tsv data/raw/dontpatronizeme_categories.tsv \
  --out_dir reports/local_eval
```

---

## Notes

- Inference is deterministic for fixed models + data (same inputs → same outputs).
- `BestModel/` exists to satisfy the coursework requirement: “push best model + code in a folder named BestModel”.
- The fold models are shipped as an archive (Git LFS) because raw model folders are too large for normal git.
