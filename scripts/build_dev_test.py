#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pcl_exercise.config import load_config
from pcl_exercise.data import load_datasets
from pcl_exercise.metrics import compute_prf
from pcl_exercise.modeling import MultiTaskClassifier
from pcl_exercise.training import TextDataset, make_collate, predict_probs


@dataclass
class RunSpec:
    run_dir: Path
    threshold: float
    backbone_name: str
    pooling: str
    max_length: int
    dropout: float
    batch_size: int
    mixed_precision: str
    folds: List[int]


def repo_root_from_this_file() -> Path:
    return Path(__file__).resolve().parents[1]


def find_run_dirs(repo_root: Path) -> List[Path]:
    roots = [repo_root / "models" / "outputs", repo_root / "outputs"]
    out: List[Path] = []
    for r in roots:
        if not r.exists():
            continue
        for p in sorted(r.glob("*")):
            if not p.is_dir():
                continue
            if (p / "threshold.json").exists() and (p / "models" / "fold0" / "model.pt").exists():
                out.append(p)
    return out


def load_threshold(run_dir: Path) -> float:
    t_path = run_dir / "threshold.json"
    obj = json.loads(t_path.read_text(encoding="utf-8"))
    return float(obj["threshold"])


def load_resolved_cfg(run_dir: Path) -> Dict:
    p = run_dir / "config.resolved.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def list_folds(run_dir: Path) -> List[int]:
    models_dir = run_dir / "models"
    folds: List[int] = []
    for d in models_dir.glob("fold*"):
        if d.is_dir() and (d / "model.pt").exists():
            try:
                folds.append(int(d.name.replace("fold", "")))
            except ValueError:
                pass
    return sorted(folds)


def sanitize_text_column(df, col: str = "text"):
    """Avoid tokenizer crashes from NaN/float."""
    if col not in df.columns:
        return df
    df = df.copy()
    df[col] = df[col].fillna("").astype(str)
    return df


def build_run_spec(run_dir: Path, fallback_cfg) -> RunSpec:
    t_star = load_threshold(run_dir)
    resolved = load_resolved_cfg(run_dir)

    def get(path: List[str], default=None):
        cur = resolved
        for k in path:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    backbone_name = get(["model", "backbone_name"], fallback_cfg.raw["model"]["backbone_name"])
    pooling = get(["model", "pooling"], fallback_cfg.raw["model"]["pooling"])
    max_length = int(get(["model", "max_length"], fallback_cfg.raw["model"]["max_length"]))
    dropout = float(get(["model", "dropout"], fallback_cfg.raw["model"].get("dropout", 0.1)))
    batch_size = int(get(["train", "batch_size"], fallback_cfg.raw["train"]["batch_size"]))
    mixed_precision = str(get(["train", "mixed_precision"], fallback_cfg.raw["train"].get("mixed_precision", "bf16")))

    folds = list_folds(run_dir)
    if not folds:
        raise ValueError(f"No folds found under: {run_dir/'models'}")

    return RunSpec(
        run_dir=run_dir,
        threshold=t_star,
        backbone_name=backbone_name,
        pooling=pooling,
        max_length=max_length,
        dropout=dropout,
        batch_size=batch_size,
        mixed_precision=mixed_precision,
        folds=folds,
    )


def load_tokenizer_for_run(run: RunSpec):
    """
    Prefer saved tokenizer to avoid HF hub/network dependency.
    fold0 tokenizer should be identical across folds.
    """
    tok_dir = run.run_dir / "models" / "fold0" / "tokenizer"
    if tok_dir.exists():
        return AutoTokenizer.from_pretrained(str(tok_dir), use_fast=True)
    return AutoTokenizer.from_pretrained(run.backbone_name, use_fast=True)


def _build_loader(texts: List[str], run: RunSpec) -> DataLoader:
    tokenizer = load_tokenizer_for_run(run)
    collate = make_collate(tokenizer, run.max_length)

    y = np.zeros(len(texts), dtype=np.int64)
    cat = np.zeros((len(texts), 7), dtype=np.int64)
    w = np.ones(len(texts), dtype=np.float32)
    has_aux = np.zeros(len(texts), dtype=np.int64)

    ds = TextDataset(texts=texts, y=y, cat=cat, w_scale=w, has_aux=has_aux)
    dl = DataLoader(ds, batch_size=run.batch_size * 2, shuffle=False, collate_fn=collate)
    return dl


@torch.no_grad()
def _predict_fold_probs(run: RunSpec, fold: int, loader: DataLoader) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultiTaskClassifier(backbone_name=run.backbone_name, dropout=run.dropout, pooling=run.pooling)
    state = torch.load(run.run_dir / "models" / f"fold{fold}" / "model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.to(device)

    probs = predict_probs(model, loader, device, run.mixed_precision)
    probs = np.asarray(probs).reshape(-1)
    return probs.astype(np.float32)


def write_labels_txt(path: Path, yhat: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(int(x)) for x in yhat.tolist()) + "\n", encoding="utf-8")


def compute_metrics_at_threshold(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> Dict[str, float]:
    yhat = (probs >= threshold).astype(int)
    m = compute_prf(y_true.astype(int), yhat)
    return {k: float(v) for k, v in m.items()}


def compute_cv_fold_weights_from_train(
    run: RunSpec,
    train_texts: List[str],
    y_train: np.ndarray,
    *,
    gamma: float = 2.0,
    eps: float = 1e-3,
    min_weight: float = 0.05,
) -> Tuple[Dict[int, float], List[Dict]]:
    """
    Option C:
    - Reconstruct the same StratifiedKFold splits used in training.py (random_state=42).
    - Evaluate each fold model on its *own* val split (from the train set).
    - Convert per-fold val F1 into weights (softly).

    weights are normalized to sum to 1.

    min_weight is a safety floor (fraction of uniform weight) to avoid hard-zeroing.
    """
    n_folds = len(run.folds)
    if n_folds < 2:
        return {run.folds[0]: 1.0}, []

    # training.py: StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_reports: List[Dict] = []
    f1_by_fold: Dict[int, float] = {}

    # Important: fold indices are enumeration order from skf.split(...)
    # Your saved folders are fold0..fold{k-1}. So this should align.
    for fold_idx, (_, va_idx) in enumerate(skf.split(np.zeros(len(y_train)), y_train.astype(int))):
        if fold_idx not in run.folds:
            # In case some folds are missing on disk, skip them
            continue

        va_texts = [train_texts[i] for i in va_idx]
        va_y = y_train[va_idx].astype(int)

        dl = _build_loader(va_texts, run)
        probs = _predict_fold_probs(run, fold_idx, dl)

        m = compute_metrics_at_threshold(va_y, probs, run.threshold)
        f1 = float(m["f1_pos"])
        f1_by_fold[fold_idx] = f1

        fold_reports.append({
            "fold": int(fold_idx),
            "cv_val_metrics_at_tstar": m,
            "cv_val_size": int(len(va_idx)),
        })

    if not f1_by_fold:
        # Fallback: uniform weights
        w = {f: 1.0 / len(run.folds) for f in run.folds}
        return w, fold_reports

    # Soft weighting: w_f ∝ max(eps, f1_f)^gamma
    raw = {f: max(eps, f1_by_fold.get(f, 0.0)) ** gamma for f in run.folds}
    s = float(sum(raw.values()))
    if s <= 0:
        w = {f: 1.0 / len(run.folds) for f in run.folds}
        return w, fold_reports

    weights = {f: raw[f] / s for f in run.folds}

    # Safety floor: don't allow any fold to go below min_weight * uniform
    # Example: with 5 folds and min_weight=0.05 => floor = 0.05*(1/5)=0.01
    uniform = 1.0 / len(run.folds)
    floor = min_weight * uniform
    weights = {f: max(floor, weights[f]) for f in run.folds}
    # renormalize after flooring
    s2 = float(sum(weights.values()))
    weights = {f: weights[f] / s2 for f in run.folds}

    # add weights into report
    for fr in fold_reports:
        f = int(fr["fold"])
        fr["weight"] = float(weights.get(f, 0.0))
        fr["f1_pos"] = float(f1_by_fold.get(f, 0.0))

    return weights, fold_reports


@torch.no_grad()
def predict_ensemble_probs(
    run: RunSpec,
    texts: List[str],
    *,
    fold_weights: Optional[Dict[int, float]] = None,
) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    """
    Returns:
      - ensemble_probs: weighted mean over folds (or plain mean if weights None)
      - fold_probs: dict fold->probs (for reporting/debug)
    """
    dl = _build_loader(texts, run)
    fold_probs: Dict[int, np.ndarray] = {}
    for f in run.folds:
        fold_probs[f] = _predict_fold_probs(run, f, dl)

    stack = np.stack([fold_probs[f] for f in run.folds], axis=0)  # [F, N]

    if fold_weights is None:
        ens = stack.mean(axis=0)
        return ens.astype(np.float32), fold_probs

    w = np.array([float(fold_weights.get(f, 0.0)) for f in run.folds], dtype=np.float32)
    if w.sum() <= 0:
        ens = stack.mean(axis=0)
        return ens.astype(np.float32), fold_probs

    w = w / w.sum()
    ens = (stack * w[:, None]).sum(axis=0)
    return ens.astype(np.float32), fold_probs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--dev_out", type=str, default="dev.txt")
    ap.add_argument("--test_out", type=str, default="test.txt")
    ap.add_argument("--runs_root", type=str, default="")
    ap.add_argument("--report_json", type=str, default="selection_report.json")

    # Option C controls
    ap.add_argument("--fold_weighting", type=str, default="cv_f1", choices=["none", "cv_f1"],
                    help="none=plain mean. cv_f1=weight folds by CV val F1 on train.")
    ap.add_argument("--gamma", type=float, default=2.0,
                    help="Weight sharpness: w ∝ f1^gamma. 1=gentle, 2=stronger, 3=more aggressive.")
    ap.add_argument("--min_weight", type=float, default=0.05,
                    help="Safety floor relative to uniform weight (fraction of uniform).")
    args = ap.parse_args()

    repo_root = repo_root_from_this_file()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (repo_root / cfg_path).resolve()
    cfg = load_config(str(cfg_path))

    ds = load_datasets(
        train_csv=cfg.paths.train_csv,
        dev_csv=cfg.paths.dev_csv,
        test_tsv=cfg.paths.test_tsv,
        category_labels_train_csv=cfg.paths.category_labels_train_csv,
        category_labels_dev_csv=cfg.paths.category_labels_dev_csv,
        span_categories_tsv=cfg.paths.span_categories_tsv,
        span_min_annotators=cfg.paths.span_min_annotators,
    )

    ds.train = sanitize_text_column(ds.train, "text")
    ds.dev = sanitize_text_column(ds.dev, "text")
    ds.test = sanitize_text_column(ds.test, "text")

    train_texts = ds.train["text"].tolist()
    y_train = ds.train["y"].astype(int).to_numpy()

    dev_texts = ds.dev["text"].tolist()
    y_dev = ds.dev["y"].astype(int).to_numpy()
    test_texts = ds.test["text"].tolist()

    if args.runs_root:
        root = Path(args.runs_root)
        if not root.is_absolute():
            root = (repo_root / root).resolve()
        run_dirs: List[Path] = []
        for p in sorted(root.glob("*")):
            if p.is_dir() and (p / "threshold.json").exists() and (p / "models" / "fold0" / "model.pt").exists():
                run_dirs.append(p)
    else:
        run_dirs = find_run_dirs(repo_root)

    if not run_dirs:
        raise SystemExit(
            f"No run dirs found. Looked under {repo_root/'models/outputs'} and {repo_root/'outputs'} "
            f"(or set --runs_root)."
        )

    report: Dict = {"repo_root": str(repo_root), "runs": []}

    best_run: Optional[RunSpec] = None
    best_run_entry: Optional[Dict] = None
    best_f1 = -1.0
    best_dev_probs: Optional[np.ndarray] = None
    best_weights: Optional[Dict[int, float]] = None

    for run_dir in run_dirs:
        entry: Dict = {"run_dir": str(run_dir)}
        try:
            run = build_run_spec(run_dir, cfg)
            entry["threshold"] = float(run.threshold)
            entry["folds"] = run.folds
            entry["fold_weighting"] = args.fold_weighting
            entry["model"] = {
                "backbone_name": run.backbone_name,
                "pooling": run.pooling,
                "max_length": run.max_length,
                "dropout": run.dropout,
                "batch_size": run.batch_size,
                "mixed_precision": run.mixed_precision,
            }

            fold_weights = None
            if args.fold_weighting == "cv_f1":
                fold_weights, fold_reports = compute_cv_fold_weights_from_train(
                    run,
                    train_texts=train_texts,
                    y_train=y_train,
                    gamma=float(args.gamma),
                    min_weight=float(args.min_weight),
                )
                entry["cv_fold_weights"] = {str(k): float(v) for k, v in fold_weights.items()}
                entry["cv_fold_reports"] = fold_reports

            probs, _ = predict_ensemble_probs(run, dev_texts, fold_weights=fold_weights)
            dev_m = compute_metrics_at_threshold(y_dev, probs, run.threshold)
            entry["dev_metrics_ensemble"] = dev_m

            f1 = float(dev_m["f1_pos"])
            prec = float(dev_m["precision_pos"])
            if (f1 > best_f1) or (
                abs(f1 - best_f1) < 1e-12 and best_run_entry and prec > float(best_run_entry["dev_metrics_ensemble"]["precision_pos"])
            ):
                best_f1 = f1
                best_run = run
                best_run_entry = entry
                best_dev_probs = probs
                best_weights = fold_weights

        except Exception as e:
            entry["error"] = str(e)

        report["runs"].append(entry)

    if best_run is None or best_run_entry is None or best_dev_probs is None:
        raise SystemExit("All runs failed. Check selection_report.json for errors.")

    dev_yhat = (best_dev_probs >= best_run.threshold).astype(int)

    test_probs, _ = predict_ensemble_probs(best_run, test_texts, fold_weights=best_weights)
    test_yhat = (test_probs >= best_run.threshold).astype(int)

    dev_out = Path(args.dev_out)
    test_out = Path(args.test_out)
    report_out = Path(args.report_json)

    if not dev_out.is_absolute():
        dev_out = (repo_root / dev_out).resolve()
    if not test_out.is_absolute():
        test_out = (repo_root / test_out).resolve()
    if not report_out.is_absolute():
        report_out = (repo_root / report_out).resolve()

    write_labels_txt(dev_out, dev_yhat)
    write_labels_txt(test_out, test_yhat)

    report["selected"] = {
        "run_dir": str(best_run.run_dir),
        "threshold": float(best_run.threshold),
        "folds": best_run.folds,
        "fold_weighting": args.fold_weighting,
        "dev_metrics_ensemble": best_run_entry["dev_metrics_ensemble"],
        "cv_fold_weights": best_run_entry.get("cv_fold_weights", None),
        "dev_out": str(dev_out),
        "test_out": str(test_out),
    }

    report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n✅ Selection complete (BEST RUN by DEV F1)")
    print(f"Selected run: {best_run.run_dir}")
    print(f"Threshold t*: {best_run.threshold:.4f}")
    m = best_run_entry["dev_metrics_ensemble"]
    print(f"DEV ensemble @t*: f1={m['f1_pos']:.4f} p={m['precision_pos']:.4f} r={m['recall_pos']:.4f}")
    if args.fold_weighting == "cv_f1":
        print("Fold weights (from train CV val F1):")
        for f in best_run.folds:
            print(f"  fold{f}: {float(best_weights.get(f, 0.0)):.4f}")
    print("\nWrote:")
    print(f"  dev:    {dev_out}")
    print(f"  test:   {test_out}")
    print(f"  report: {report_out}")


if __name__ == "__main__":
    main()