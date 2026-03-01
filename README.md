# nlp-pcl

Script-first pipeline for the “Don’t Patronize Me!” (PCL) **Task 4 / Subtask 1** classifier.

Legacy notebooks / old `simpletransformers` workflow are kept under `legacy/` for reference, but the main path here is:

- DeBERTa-v3-large backbone
- Binary classifier head (PCL vs No PCL)
- Optional 7-category auxiliary head (multi-label)
- 5-fold Stratified CV
- OOF threshold tuning for **F1 on the positive class**
- Fold ensemble inference
- Outputs in the exact required format: `dev.txt` / `test.txt`

Repo URL:

```bash
https://github.com/amin-abualhassan/nlp-pcl.git
```

---

## Repo layout (what’s what)

- `src/pcl_exercise/` : package code (data loading, model, training, inference)
- `configs/` : YAML configs (paths + hyperparams)
- `scripts/` : CLI scripts
- `data/` : your local data files for train/dev/test
- `models/` : **generated locally** after restoring/extracting model artifacts (don’t commit this folder)
- `outputs_only.tgz.part_*` + `outputs_only.tgz.sha256` : model archive split into parts (stored via Git LFS)

---

## Quick start (fresh clone → restore models → rebuild dev/test)

### 0) System prereqs (Ubuntu / WSL)

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs
git lfs install
```

### 1) Clone (with or without LFS smudge)

#### Normal clone (recommended)
This should fetch LFS objects automatically (including `outputs_only.tgz.part_*`).

```bash
cd ~
git clone https://github.com/amin-abualhassan/nlp-pcl.git
cd nlp-pcl
```

If the clone feels slow, it’s usually downloading the LFS parts (they’re big). Let it finish.

#### “Fast” clone (skip downloading LFS during checkout)
Useful if you want the repo quickly, then pull LFS explicitly.

```bash
cd ~
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/amin-abualhassan/nlp-pcl.git
cd nlp-pcl
git lfs pull
```

Sanity check (you should see the parts listed):

```bash
git lfs ls-files | grep outputs_only.tgz.part || true
ls -lh outputs_only.tgz.part_*
```

> If you ever cloned as `sudo` and then run git as your normal user, you may get “dubious ownership”.
> Fix:
> ```bash
> git config --global --add safe.directory "$(pwd)"
> ```

### 2) Create a venv

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

### 3) Install PyTorch (GPU vs CPU)

#### GPU (CUDA)
First, check what you have:

```bash
nvidia-smi
python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available()); print('torch cuda', torch.version.cuda)"
```

Example for CUDA 12.8 wheels:

```bash
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
```

If you already have a CUDA torch installed and `cuda? True`, you can keep it.

#### CPU only
```bash
pip install torch torchvision torchaudio
```

### 4) Install project deps

```bash
pip install -r requirements.txt
```

(Optional but convenient)
```bash
pip install -e .
```

### 5) Restore the saved models into `./models`

The trained fold models are shipped as a split archive (parts + sha256). Restore them like this:

```bash
chmod +x scripts/restore_models.sh
bash scripts/restore_models.sh
```

You should end up with:

```bash
ls models/outputs/
ls models/outputs/*/models/fold0/model.pt
```

### 6) Rebuild `dev.txt` and `test.txt` (no retraining)

Plain mean over folds (no weighting):

```bash
python3 scripts/build_dev_test.py --fold_weighting none
```

This writes (repo root):
- `dev.txt`
- `test.txt`
- `selection_report.json`

### 7) Verify you get the same outputs (recommended)

Hashes:

```bash
sha256sum dev.txt test.txt selection_report.json
```

Notes:
- `dev.txt` and `test.txt` should match exactly across machines if everything is the same.
- `selection_report.json` may differ because it can contain absolute paths. That’s fine.

Dev F1 check:

```bash
python3 scripts/eval_dev_f1.py --dev_csv data/dev_df_2.csv --pred dev.txt
```

---

## Training from scratch (optional)

### 1) Put your data in `data/`

Minimum:
- `data/train_df.csv`
- `data/dev_df_2.csv`
- `data/task4_test.tsv` (only needed to produce `test.txt`)

Optional (aux labels):
- `data/other/train_semeval_parids-labels.csv`
- `data/other/dev_semeval_parids-labels.csv`
- or a span categories TSV (see config)

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

If you want to downweight weaker folds using their CV performance on the train split:

```bash
python3 scripts/build_dev_test.py --fold_weighting cv_f1 --gamma 2.0 --min_weight 0.05
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'pcl_exercise'`
Run from the repo root, and either:
- use the scripts as shown (they add `src/` to `sys.path`), or
- install editable:

```bash
pip install -e .
```

### `restore_models.sh` says it can’t find `outputs_only.tgz.part_*`
That means the parts are missing locally (usually because LFS objects weren’t pulled).
Run:

```bash
git lfs pull
ls -lh outputs_only.tgz.part_*
```

---

## Sharing note

Before making the repo public, double-check you’re allowed to redistribute any dataset files under `data/`.
If unsure, keep the repo private and only share code + restore script.
