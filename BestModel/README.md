# nlp-pcl

Script-first pipeline for the “Don’t Patronize Me!” (PCL) **NLP coursework**.

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

You can also find the models folder in this Google Drive folder (if you faced issues with lfs):

```bash
https://drive.google.com/drive/folders/1FAUTcZRYLRKM-L8iZt_-mGZiH9xzGy89?usp=sharing
```

---

## Repo layout

- `src/pcl_exercise/` : package code (data loading, model, training, inference)
- `configs/` : YAML configs (paths + hyperparams)
- `scripts/` : CLI scripts
- `data/` : your local data files for train/dev/test
- `models/` : **generated locally** after restoring/extracting model artifacts (don’t commit this folder)
- `outputs_only.tgz.part_*` + `outputs_only.tgz.sha256` : model archive split into parts (stored via Git LFS)

---

## Quick start

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

### 3) Install PyTorch

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

(Optional)
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

### 6) Rebuild `dev.txt` and `test.txt` using the existing models

Plain mean over folds (no weighting):

```bash
python3 scripts/build_dev_test.py --fold_weighting none
```

This writes (repo root):
- `dev.txt`
- `test.txt`
- `selection_report.json`

### 7) Verify you get the same outputs

Run the prediction script twice:

```bash
python3 scripts/build_dev_test.py --fold_weighting none --report_json selection_report_fresh.json
python3 scripts/build_dev_test.py --fold_weighting none --report_json selection_report_fresh.json
```

It will be something like:

- Selected run: `models/outputs/20260228_165925_deberta_mtl_cv5_lam_sweep_1`
- Threshold `t*`: `0.45`
- DEV ensemble @t*: `f1≈0.6015`, `precision≈0.6158`, `recall≈0.5879`

Then verify the files match across machines using hashes:

```bash
sha256sum dev.txt test.txt selection_report_fresh.json
```

Reference hashes from a clean clone run:

- `dev.txt`  → `5d16a7fb1fe9e85df0d7635d4b45bb47a7de9ce5012d881a47fbe766822f1c10`
- `test.txt` → `2a149df15d9f3f1d505dcd8b80e82398ccc1422be22a5583b209bf115e1c31cc`

Dev F1 check:

```bash
python3 scripts/eval_dev_f1.py --dev_csv data/dev_df_2.csv --pred dev.txt
```

