# nlp-pcl

This repo is a cleaned-up, script-first version of the “Don’t Patronize Me!” (PCL) Task 4 Subtask 1 pipeline.

The old notebook workflow (and the old `simpletransformers` approach) is kept under `legacy/` for reference, but the main path is now:

- DeBERTa-v3-large backbone
- Binary classifier head (PCL vs No PCL)
- Optional 7-category auxiliary head (multi-label)
- 5-fold Stratified CV
- OOF threshold tuning for **F1 on the positive class**
- Fold ensemble inference
- Outputs in the exact required format: `dev.txt` / `test.txt`

---

## Repo layout

- `src/pcl_exercise/` : package code (data loading, model, training, inference)
- `configs/` : YAML configs (paths + hyperparams)
- `scripts/` : CLI scripts
- `data/` : your local data files for train/dev/test
- `models/` : restored model artifacts (generated locally; don’t commit this folder)
- `outputs_only.tgz.part_*` + `outputs_only.tgz.sha256` : split archive used to rebuild `models/` on any machine

---

## Step 0 : Setup

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

If you want to run on GPU, install the matching PyTorch build for your CUDA first, then install requirements.

---

## Cloning

### This repo: **no Git LFS needed**
Model artifacts are shipped as a split archive (`outputs_only.tgz.part_*`).  
So a normal clone is enough:

```bash
git clone <repo-url>
cd nlp-pcl
```

Then restore the models:

```bash
bash scripts/restore_models.sh
```

After that you should have `models/outputs/.../models/fold*/model.pt` available locally.

### If you ever use a branch/repo that stores models in **Git LFS**
You’ll know it’s LFS if you see `.gitattributes` with `filter=lfs` or `git lfs ls-files` shows model files.

Quick setup:

```bash
# one-time
git lfs install

# after cloning
git lfs pull
```

(Install `git-lfs` via your OS package manager if needed.)

---

## Option A : You already have trained models (restore + predict)

### 1) Restore model artifacts into `./models`

```bash
bash scripts/restore_models.sh
```

### 2) Regenerate `dev.txt` and `test.txt`

This script searches for run directories under:
- `models/outputs/*` (restored)
- `outputs/*` (if you trained locally)

Plain mean over folds (no weighting):

```bash
python3 scripts/build_dev_test.py --fold_weighting none
```

Outputs:
- `dev.txt` (repo root)
- `test.txt` (repo root)
- `selection_report.json` (what it picked + metrics)

### 3) Check dev F1

```bash
python3 scripts/eval_dev_f1.py --dev_csv data/dev_df_2.csv --pred dev.txt
```

---

## Option B : Train from scratch

### 1) Put your data in `data/`

Minimum:

- `data/train_df.csv`
- `data/dev_df_2.csv`
- `data/task4_test.tsv` (only needed to produce test predictions)

Optional (aux labels):
- `data/other/train_semeval_parids-labels.csv`
- `data/other/dev_semeval_parids-labels.csv`
- or span categories TSV (see config)

### 2) Train 5-fold CV + tune OOF threshold

```bash
python3 scripts/train_cv.py --config configs/default.yaml
```

This creates a timestamped run dir under `outputs/...` with:
- `models/fold0..fold4/model.pt`
- `threshold.json` (t*)
- OOF probs/metrics + logs + resolved config

### 3) Predict dev/test for that run

```bash
python3 scripts/predict.py --config configs/default.yaml --run_dir outputs/<RUN_DIR> --split both
```

Or use the “pick best run and write dev/test” script:

```bash
python3 scripts/build_dev_test.py --runs_root outputs --fold_weighting none
```

---

## Fold weighting (optional)

Default is a plain average across folds.

If you want to *downweight* weaker folds using their CV performance on the train split:

```bash
python3 scripts/build_dev_test.py --fold_weighting cv_f1 --gamma 2.0 --min_weight 0.05
```

---

## Data expectations

Your CSV/TSV files must have at least:

- `par_id`
- `text`
- `label`
- `target_flag` and/or a derived `y` (binary), depending on your loader config

Tokenizer errors like `TextEncodeInput must be ...` are almost always a NaN in `text`.
The scripts defensively do `text = fillna("").astype(str)`.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'pcl_exercise'`
Run from the repo root, and either:
- use the scripts as shown (they add `src/` to `sys.path`), or
- install the package editable:

```bash
pip install -e .
```

---

## Sharing note

Before making this repo public, double-check you’re allowed to redistribute any dataset files under `data/`.
If unsure, keep it private and share only code + restore script.
