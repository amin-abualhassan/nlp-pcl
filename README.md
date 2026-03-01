# nlp-pcl (Stage 3 revamp)

This repo is a cleaned-up, script-first version of the “Don’t Patronize Me!” (PCL) Task 4 Subtask 1 pipeline.

The old notebook workflow (and the old `simpletransformers` approach) is kept under `legacy/` for reference, but the main path is now:

- DeBERTa-v3-large backbone
- Binary classifier head (PCL vs No PCL)
- Optional 7-category auxiliary head (multi-label)
- 5-fold Stratified CV (train only)
- OOF threshold tuning for **F1 on the positive class**
- Fold ensemble inference
- Outputs in the exact required format: `dev.txt` / `test.txt`

---

## Repo layout (what’s what)

- `src/pcl_exercise/` — the actual package code (data loading, model, training, inference)
- `configs/` — YAML configs (paths + hyperparams)
- `scripts/` — CLI entry points
- `models/` — restored model artifacts (NOT meant to be committed directly)
- `outputs_only.tgz.part_*` + `outputs_only.tgz.sha256` — split archive that can recreate `models/`
- `data/` — your local data files for training/dev/test

---

## Step 0 — Setup (once)

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

If you plan to train on GPU, install the matching PyTorch build for your CUDA setup (then run the requirements install).

---

## Option A — You already have trained models (fast path)

This is the “just rebuild dev.txt / test.txt from saved folds” workflow.

### 1) Restore model artifacts into `./models`

```bash
bash scripts/restore_models.sh
```

That script:
- rebuilds `outputs_only.tgz` from the `part_*` files
- verifies SHA256
- extracts into `./models`
- cleans up the rebuilt tar

After it runs, you should see something like:

```bash
ls models/outputs/
```

### 2) Regenerate `dev.txt` and `test.txt` from the best run

This script searches for valid run directories under:
- `models/outputs/*` (restored)
- `outputs/*` (if you trained locally)

By default it uses a plain mean over folds (no weighting):

```bash
python3 scripts/build_dev_test.py --fold_weighting none
```

Outputs:
- `dev.txt` in the repo root
- `test.txt` in the repo root
- `selection_report.json` (what it picked + metrics)

### 3) Check dev F1 (sanity)

```bash
python3 scripts/eval_dev_f1.py --dev_csv data/dev_df_2.csv --pred dev.txt
```

---

## Option B — Train from scratch (slower, but reproducible)

### 1) Make sure your data files exist

Minimum:

- `data/train_df.csv`
- `data/dev_df_2.csv`
- `data/task4_test.tsv` (only needed to produce test predictions)

Optional (aux labels):
- `data/other/train_semeval_parids-labels.csv`
- `data/other/dev_semeval_parids-labels.csv`
- or span categories TSV (see config)

### 2) Run training (5-fold CV + OOF threshold)

```bash
python3 scripts/train_cv.py --config configs/default.yaml
```

This creates a timestamped run dir under `outputs/…` with:
- `models/fold0..fold4/model.pt`
- `threshold.json` (t*)
- `oof_probs.npy` and `oof_metrics.json`
- logs + resolved config

### 3) Predict dev/test for that run

If you want the “classic” flow:

```bash
python3 scripts/predict.py --config configs/default.yaml --run_dir outputs/<RUN_DIR> --split both
```

Or, if you prefer the unified “pick best run and write dev/test” approach:

```bash
python3 scripts/build_dev_test.py --runs_root outputs --fold_weighting none
```

---

## Fold weighting (optional)

By default, the ensemble is a plain average across folds. That’s usually the safest choice.

If you want to *downweight* weaker folds using their CV performance on the train split:

```bash
python3 scripts/build_dev_test.py --fold_weighting cv_f1 --gamma 2.0 --min_weight 0.05
```

Notes:
- `gamma` controls how strong the weighting is (2.0 is “noticeable”).
- `min_weight` prevents a fold from getting near-zero weight.

---

## Data expectations (quick notes)

Your CSV/TSV files must have at least:

- `par_id` (unique paragraph id)
- `text`
- `label` (the original label scale)
- `target_flag` and/or a derived `y` (binary) depending on your loader config

If you ever hit tokenizer errors like `TextEncodeInput must be ...`, it’s almost always a NaN in `text`.
The scripts defensively do: `text = fillna("").astype(str)`.

---

## Large files and cloning

This repo keeps model artifacts in a split archive (`outputs_only.tgz.part_*`) so:
- cloning stays manageable
- everyone can restore models locally with `scripts/restore_models.sh`

The extracted `models/` directory is treated as a generated artifact and should be ignored by git.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'pcl_exercise'`
Run from the repo root, and either:
- use the scripts as shown (they add `src/` to `sys.path`), or
- install the package editable:

```bash
pip install -e .
```

### GPU not used
Check:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
nvidia-smi
```

---

## License / sharing note

Before making this repo public, double-check you’re allowed to redistribute any dataset files under `data/`.
If unsure, keep the repo private and only share code + restore scripts.
