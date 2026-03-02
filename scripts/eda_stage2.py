#!/usr/bin/env python3
"""
Stage 2 (EDA) script for the PCL dataset.

Two distinct EDA techniques:

Technique A — Structure + quality checks
  - Token-length profiling using the *same tokenizer* used by the model (local tokenizer preferred)
  - Truncation-rate vs candidate max_length values
  - Basic text-quality artifacts (HTML entities, URLs, non-ascii, etc.)
  - Exact-duplicate checks (within split + train↔dev leakage)

Technique B — Label + metadata structure
  - Label-scale (0–4) distribution + binary mapping summary (y = label>=2 or target_flag)
  - Split shift in keyword/country distributions (train vs dev vs test)
  - Positive rate by keyword/country (train/dev only)
  - Optional 7-category auxiliary label prevalence + co-occurrence (if available)

Run (from repo root):
  python3 scripts/eda_stage2.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Headless plotting
import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from transformers import AutoTokenizer  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pcl_exercise.config import load_config  # noqa: E402
from pcl_exercise.data import load_datasets, CANONICAL_CATEGORIES  # noqa: E402


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def sanitize_text_column(df: pd.DataFrame, col: str = "text") -> pd.DataFrame:
    if col not in df.columns:
        return df
    out = df.copy()
    out[col] = out[col].fillna("").astype(str)
    return out


def _find_local_tokenizer_dir(repo_root: Path) -> Optional[Path]:
    """
    Prefer a local tokenizer saved under:
      - models/outputs/*/models/fold0/tokenizer
      - outputs/*/models/fold0/tokenizer
    Returns the newest-looking one (sorted name).
    """
    candidates: List[Path] = []
    for root in [repo_root / "models" / "outputs", repo_root / "outputs"]:
        if not root.exists():
            continue
        for run_dir in sorted(root.glob("*")):
            tok = run_dir / "models" / "fold0" / "tokenizer"
            if tok.exists() and (tok / "tokenizer.json").exists():
                candidates.append(tok)
    if not candidates:
        return None
    return candidates[-1]


def load_tokenizer(backbone_name: str, tokenizer_dir: Optional[Path]) -> Tuple[Optional[AutoTokenizer], str]:
    """
    Returns (tokenizer_or_None, source_string).
    If tokenizer can't be loaded, returns (None, reason) and the script falls back to whitespace tokens.
    """
    if tokenizer_dir is not None and tokenizer_dir.exists():
        try:
            tok = AutoTokenizer.from_pretrained(str(tokenizer_dir), use_fast=True)
            return tok, f"local:{tokenizer_dir}"
        except Exception as e:
            return None, f"failed_local:{tokenizer_dir} ({e})"

    # Try local cached HF files; then network if available.
    try:
        tok = AutoTokenizer.from_pretrained(backbone_name, use_fast=True, local_files_only=True)
        return tok, f"hf_local_cache:{backbone_name}"
    except Exception:
        pass

    try:
        tok = AutoTokenizer.from_pretrained(backbone_name, use_fast=True)
        return tok, f"hf_download:{backbone_name}"
    except Exception as e:
        return None, f"failed_hf:{backbone_name} ({e})"


def token_lengths(texts: List[str], tokenizer: Optional[AutoTokenizer], batch_size: int = 256) -> np.ndarray:
    """
    Compute token lengths with the provided tokenizer.
    Falls back to whitespace token counts if tokenizer is None.
    """
    if tokenizer is None:
        # approximate: whitespace split
        return np.asarray([len(str(t).split()) for t in texts], dtype=np.int32)

    lens: List[int] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        ids = enc["input_ids"]
        lens.extend([len(x) for x in ids])
    return np.asarray(lens, dtype=np.int32)


def percentiles(arr: np.ndarray, ps: List[float]) -> Dict[str, float]:
    out = {}
    a = np.asarray(arr, dtype=np.float64)
    for p in ps:
        out[f"p{int(p)}"] = float(np.percentile(a, p))
    out["max"] = float(np.max(a)) if a.size else float("nan")
    out["mean"] = float(np.mean(a)) if a.size else float("nan")
    out["std"] = float(np.std(a)) if a.size else float("nan")
    return out


def save_table(df: pd.DataFrame, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_base.with_suffix(".csv"), index=False)
    df.to_markdown(out_base.with_suffix(".md"), index=False)


def plot_hist_overlay(lengths_by_split: Dict[str, np.ndarray], out_png: Path, title: str) -> None:
    plt.figure()
    for name, arr in lengths_by_split.items():
        if arr.size == 0:
            continue
        plt.hist(arr, bins=60, alpha=0.5, density=True, label=f"{name} (n={arr.size})")
    plt.xlabel("Token length")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_line(x: List[int], ys: Dict[str, List[float]], out_png: Path, title: str, ylab: str) -> None:
    plt.figure()
    for name, y in ys.items():
        plt.plot(x, y, marker="o", label=name)
    plt.xlabel("max_length")
    plt.ylabel(ylab)
    plt.title(title)
    plt.legend()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def normalize_text_for_dupes(s: str) -> str:
    s = str(s)
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def duplicate_report(train: pd.DataFrame, dev: pd.DataFrame, text_col: str = "text") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - summary_df: counts
      - examples_df: sample duplicate pairs across train/dev
    """
    tr = train[[text_col, "par_id"]].copy()
    dv = dev[[text_col, "par_id"]].copy()
    tr["norm"] = tr[text_col].apply(normalize_text_for_dupes)
    dv["norm"] = dv[text_col].apply(normalize_text_for_dupes)

    # Within-split duplicates
    tr_dupe_mask = tr["norm"].duplicated(keep=False)
    dv_dupe_mask = dv["norm"].duplicated(keep=False)

    # Cross-split duplicates (same normalized text in both)
    common = set(tr["norm"]).intersection(set(dv["norm"]))
    cross = pd.merge(
        tr[tr["norm"].isin(common)][["par_id", "norm"]],
        dv[dv["norm"].isin(common)][["par_id", "norm"]],
        on="norm",
        suffixes=("_train", "_dev"),
        how="inner",
    )

    summary = pd.DataFrame(
        [
            {"check": "train_size", "value": int(len(tr))},
            {"check": "dev_size", "value": int(len(dv))},
            {"check": "train_within_dup_texts", "value": int(tr_dupe_mask.sum())},
            {"check": "dev_within_dup_texts", "value": int(dv_dupe_mask.sum())},
            {"check": "unique_train_norm_texts", "value": int(tr["norm"].nunique())},
            {"check": "unique_dev_norm_texts", "value": int(dv["norm"].nunique())},
            {"check": "cross_split_dup_pairs", "value": int(len(cross))},
            {"check": "cross_split_unique_dup_texts", "value": int(cross["norm"].nunique() if len(cross) else 0)},
        ]
    )

    examples = cross.head(50).copy()
    return summary, examples


def artifact_table(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """
    Simple “noise/artifact” checks. Outputs counts + percentages.
    """
    t = df[text_col].fillna("").astype(str)

    checks = {
        "has_html_entity (&...;)": t.str.contains(r"&[a-zA-Z]+;", regex=True),
        "has_url (http/www)": t.str.contains(r"(http://|https://|www\.)", regex=True),
        "has_newline": t.str.contains(r"\n"),
        "has_non_ascii": t.apply(lambda x: any(ord(ch) > 127 for ch in x)),
        "has_repeated_punct (!!/??)": t.str.contains(r"(\!\!|\?\?)", regex=True),
        "has_many_spaces": t.str.contains(r"\s{3,}", regex=True),
    }

    rows = []
    n = len(df)
    for name, mask in checks.items():
        c = int(mask.sum())
        rows.append({"check": name, "count": c, "pct": (c / n * 100.0) if n else 0.0})
    return pd.DataFrame(rows).sort_values("pct", ascending=False)


def label_scale_summary(train: pd.DataFrame, dev: pd.DataFrame) -> pd.DataFrame:
    def _one(df: pd.DataFrame, split: str) -> pd.DataFrame:
        out = df.copy()
        if "label" in out.columns:
            out["label"] = out["label"].astype(int)
        if "y" in out.columns:
            out["y"] = out["y"].astype(int)
        g = out.groupby("label")["y"].agg(["count", "mean"]).reset_index()
        g["split"] = split
        g.rename(columns={"mean": "p(y=1)"}, inplace=True)
        return g

    return pd.concat([_one(train, "train"), _one(dev, "dev")], ignore_index=True)


def plot_bar_counts(df: pd.DataFrame, x: str, y: str, hue: Optional[str], out_png: Path, title: str) -> None:
    plt.figure()
    if hue is None:
        plt.bar(df[x].astype(str), df[y].astype(float))
    else:
        # simple grouped bar without seaborn
        cats = sorted(df[x].unique().tolist())
        hues = sorted(df[hue].unique().tolist())
        x_idx = np.arange(len(cats))
        width = 0.8 / max(1, len(hues))
        for i, h in enumerate(hues):
            sub = df[df[hue] == h].set_index(x).reindex(cats).fillna(0.0)
            plt.bar(x_idx + i * width, sub[y].astype(float).to_numpy(), width=width, label=str(h))
        plt.xticks(x_idx + width * (len(hues) - 1) / 2, [str(c) for c in cats], rotation=0)
        plt.legend()

    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def split_shift_tables(
    train: pd.DataFrame,
    dev: pd.DataFrame,
    test: Optional[pd.DataFrame],
    col: str,
    top_k: int = 15,
) -> pd.DataFrame:
    """
    Returns a table with counts per split for the top_k most frequent values in TRAIN.
    """
    def counts(df: pd.DataFrame, name: str) -> pd.Series:
        if df is None or col not in df.columns:
            return pd.Series(dtype=np.int64, name=name)
        return df[col].astype(str).value_counts().rename(name)

    c_tr = counts(train, "train")
    c_dv = counts(dev, "dev")
    c_ts = counts(test, "test") if test is not None else pd.Series(dtype=np.int64, name="test")

    top_vals = c_tr.head(top_k).index.tolist()
    out = pd.DataFrame({col: top_vals})
    out["train"] = out[col].map(c_tr).fillna(0).astype(int)
    out["dev"] = out[col].map(c_dv).fillna(0).astype(int)
    out["test"] = out[col].map(c_ts).fillna(0).astype(int) if len(c_ts) else 0
    out["train_pct"] = out["train"] / max(1, int(len(train))) * 100.0
    out["dev_pct"] = out["dev"] / max(1, int(len(dev))) * 100.0
    out["test_pct"] = out["test"] / max(1, int(len(test))) * 100.0 if test is not None else np.nan
    return out


def aux_category_tables(df: pd.DataFrame, split: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - freq table (category prevalence among positives with aux labels)
      - co-occurrence matrix table (counts)
    """
    cat_cols = [f"cat_{c}" for c in CANONICAL_CATEGORIES]
    have = [c for c in cat_cols if c in df.columns]
    if not have or "has_aux_labels" not in df.columns:
        return (
            pd.DataFrame([{"split": split, "note": "no_aux_columns"}]),
            pd.DataFrame([{"split": split, "note": "no_aux_columns"}]),
        )

    pos = df[(df["y"].astype(int) == 1) & (df["has_aux_labels"].astype(bool))].copy()
    if pos.empty:
        return (
            pd.DataFrame([{"split": split, "note": "no_pos_with_aux"}]),
            pd.DataFrame([{"split": split, "note": "no_pos_with_aux"}]),
        )

    X = pos[have].astype(int).to_numpy()
    freq = X.mean(axis=0)  # prevalence among positives with aux
    freq_df = pd.DataFrame(
        {
            "split": split,
            "category": [c.replace("cat_", "") for c in have],
            "prevalence_pct": (freq * 100.0),
            "count": X.sum(axis=0),
            "n_pos_with_aux": X.shape[0],
        }
    ).sort_values("count", ascending=False)

    co = X.T @ X  # counts of co-occurrence
    co_df = pd.DataFrame(co, index=[c.replace("cat_", "") for c in have], columns=[c.replace("cat_", "") for c in have])
    co_df.insert(0, "split", split)

    return freq_df, co_df


def plot_heatmap(matrix_df: pd.DataFrame, out_png: Path, title: str) -> None:
    """
    matrix_df: square matrix with index/columns as category names (no split column)
    """
    mat = matrix_df.to_numpy(dtype=float)
    labels = matrix_df.columns.tolist()

    plt.figure(figsize=(8, 7))
    im = plt.imshow(mat)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.title(title)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--out_dir", type=str, default="reports/eda")
    ap.add_argument("--tokenizer_dir", type=str, default="", help="Optional path to a tokenizer folder (preferred).")
    ap.add_argument("--top_k", type=int, default=15, help="Top-K keywords/countries to include in tables.")
    ap.add_argument(
        "--max_lengths",
        type=str,
        default="64,96,128,160,192,224,256,320,384,512",
        help="Comma-separated candidate max_length values for truncation-rate plot.",
    )
    args = ap.parse_args()

    out_dir = _ensure_dir(REPO_ROOT / args.out_dir)
    techA = _ensure_dir(out_dir / "techA_structure_quality")
    techB = _ensure_dir(out_dir / "techB_labels_metadata")

    # Load config
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (REPO_ROOT / cfg_path).resolve()
    cfg = load_config(str(cfg_path))

    # Load datasets using the package loader (ensures y + aux columns)
    ds = load_datasets(
        train_csv=cfg.paths.train_csv,
        dev_csv=cfg.paths.dev_csv,
        test_tsv=cfg.paths.test_tsv,
        category_labels_train_csv=cfg.paths.category_labels_train_csv,
        category_labels_dev_csv=cfg.paths.category_labels_dev_csv,
        span_categories_tsv=cfg.paths.span_categories_tsv,
        span_min_annotators=cfg.paths.span_min_annotators,
    )

    train = sanitize_text_column(ds.train, "text")
    dev = sanitize_text_column(ds.dev, "text")
    test = sanitize_text_column(ds.test, "text") if ds.test is not None else None

    manifest: Dict[str, object] = {
        "config": str(cfg_path),
        "paths": {
            "train_csv": str(cfg.paths.train_csv),
            "dev_csv": str(cfg.paths.dev_csv),
            "test_tsv": str(cfg.paths.test_tsv) if cfg.paths.test_tsv else None,
        },
        "outputs": [],
    }

    # --------------------------
    # Technique A — lengths + truncation + artifacts + duplicates
    # --------------------------
    tok_dir: Optional[Path] = None
    if args.tokenizer_dir.strip():
        tok_dir = Path(args.tokenizer_dir).expanduser()
        if not tok_dir.is_absolute():
            tok_dir = (REPO_ROOT / tok_dir).resolve()
    else:
        tok_dir = _find_local_tokenizer_dir(REPO_ROOT)

    tokenizer, tok_src = load_tokenizer(cfg.raw["model"]["backbone_name"], tok_dir)
    manifest["tokenizer_source"] = tok_src
    manifest["tokenizer_dir"] = str(tok_dir) if tok_dir is not None else None

    train_texts = train["text"].astype(str).tolist()
    dev_texts = dev["text"].astype(str).tolist()
    test_texts = test["text"].astype(str).tolist() if test is not None and "text" in test.columns else []

    tr_len = token_lengths(train_texts, tokenizer)
    dv_len = token_lengths(dev_texts, tokenizer)
    ts_len = token_lengths(test_texts, tokenizer) if test_texts else np.asarray([], dtype=np.int32)

    lengths_by_split = {"train": tr_len, "dev": dv_len}
    if ts_len.size:
        lengths_by_split["test"] = ts_len

    # Length percentiles table
    ps = [50, 75, 90, 95, 97, 99]
    rows = []
    for split, arr in lengths_by_split.items():
        rows.append({"split": split, **percentiles(arr, ps)})
    pct_df = pd.DataFrame(rows)
    save_table(pct_df, techA / "token_length_percentiles")

    # Histogram overlay plot
    plot_hist_overlay(lengths_by_split, techA / "token_length_hist_overlay.png", "Token length distribution (model tokenizer)")

    # Truncation rates for candidate max_length
    max_lengths = [int(x.strip()) for x in args.max_lengths.split(",") if x.strip()]
    trunc = {"train_trunc_pct": [], "dev_trunc_pct": []}
    if ts_len.size:
        trunc["test_trunc_pct"] = []
    for L in max_lengths:
        trunc["train_trunc_pct"].append(float((tr_len > L).mean() * 100.0))
        trunc["dev_trunc_pct"].append(float((dv_len > L).mean() * 100.0))
        if ts_len.size:
            trunc["test_trunc_pct"].append(float((ts_len > L).mean() * 100.0))

    trunc_df = pd.DataFrame({"max_length": max_lengths, **trunc})
    save_table(trunc_df, techA / "truncation_rates_by_max_length")
    plot_line(
        max_lengths,
        {k: trunc_df[k].tolist() for k in trunc_df.columns if k != "max_length"},
        techA / "truncation_rates_plot.png",
        "Truncation rate vs max_length",
        "Percent of examples with length > max_length",
    )

    # Length by class (train/dev)
    def _len_by_class(df: pd.DataFrame, arr: np.ndarray, split: str) -> pd.DataFrame:
        out = pd.DataFrame({"split": split, "y": df["y"].astype(int).to_numpy(), "tok_len": arr})
        return out

    len_class = pd.concat([_len_by_class(train, tr_len, "train"), _len_by_class(dev, dv_len, "dev")], ignore_index=True)
    save_table(len_class.groupby(["split", "y"])["tok_len"].describe(percentiles=[0.5, 0.9, 0.95]).reset_index(), techA / "token_length_by_class_desc")

    # Simple boxplot by class (train+dev)
    plt.figure()
    for i, (split, y) in enumerate([("train", 0), ("train", 1), ("dev", 0), ("dev", 1)]):
        sub = len_class[(len_class["split"] == split) & (len_class["y"] == y)]["tok_len"].to_numpy()
        if sub.size == 0:
            sub = np.asarray([np.nan])
        plt.boxplot(sub, positions=[i], widths=0.6, showfliers=False)
    plt.xticks(range(4), ["train y=0", "train y=1", "dev y=0", "dev y=1"], rotation=20, ha="right")
    plt.ylabel("Token length")
    plt.title("Token length by split and class (boxplot, no outliers)")
    plt.tight_layout()
    plt.savefig(techA / "token_length_by_class_boxplot.png", dpi=200)
    plt.close()

    # Artifact checks
    art_train = artifact_table(train)
    art_train.insert(0, "split", "train")
    art_dev = artifact_table(dev)
    art_dev.insert(0, "split", "dev")
    art_all = pd.concat([art_train, art_dev], ignore_index=True)
    save_table(art_all, techA / "text_artifacts_table")

    # Duplicate checks
    dup_summary, dup_examples = duplicate_report(train, dev)
    save_table(dup_summary, techA / "duplicate_summary")
    save_table(dup_examples, techA / "duplicate_examples_train_dev")

    # --------------------------
    # Technique B — label + keyword/country shift + aux categories
    # --------------------------
    # Label scale summary (train/dev)
    if "label" in train.columns and "label" in dev.columns:
        ls = label_scale_summary(train, dev)
        save_table(ls, techB / "label_scale_to_binary_summary")

        # Bar plot: counts by label scale
        def _label_counts(df: pd.DataFrame, split: str) -> pd.DataFrame:
            vc = df["label"].astype(int).value_counts().sort_index()
            return pd.DataFrame({"label": vc.index.astype(int), "count": vc.to_numpy(), "split": split})

        counts = pd.concat([_label_counts(train, "train"), _label_counts(dev, "dev")], ignore_index=True)
        save_table(counts, techB / "label_scale_counts")
        plot_bar_counts(counts, x="label", y="count", hue="split", out_png=techB / "label_scale_counts.png", title="Label scale distribution (0–4)")

    # Keyword/country distribution shift tables
    if "keyword" in train.columns and "keyword" in dev.columns:
        kw_shift = split_shift_tables(train, dev, test, col="keyword", top_k=int(args.top_k))
        save_table(kw_shift, techB / "keyword_shift_topk")
        # Plot train/dev/test percentages for top-k
        plt.figure(figsize=(10, 5))
        x = np.arange(len(kw_shift))
        width = 0.25
        plt.bar(x - width, kw_shift["train_pct"], width=width, label="train")
        plt.bar(x, kw_shift["dev_pct"], width=width, label="dev")
        if test is not None:
            plt.bar(x + width, kw_shift["test_pct"], width=width, label="test")
        plt.xticks(x, kw_shift["keyword"].astype(str).tolist(), rotation=35, ha="right")
        plt.ylabel("Percent of split")
        plt.title(f"Keyword distribution shift (top {int(args.top_k)} by train frequency)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(techB / "keyword_shift_topk.png", dpi=200)
        plt.close()

        # Positive rate by keyword (train/dev only)
        def _pos_rate(df: pd.DataFrame, split: str) -> pd.DataFrame:
            g = df.groupby("keyword")["y"].agg(["count", "mean"]).reset_index()
            g["split"] = split
            g.rename(columns={"mean": "p(y=1)"}, inplace=True)
            return g

        kw_pos = pd.concat([_pos_rate(train, "train"), _pos_rate(dev, "dev")], ignore_index=True)
        save_table(kw_pos.sort_values(["split", "count"], ascending=[True, False]), techB / "keyword_pos_rate_all")

        # Keep a small top-k table for report friendliness
        kw_pos_top = (
            kw_pos[kw_pos["split"] == "train"].sort_values("count", ascending=False).head(int(args.top_k))[
                ["keyword"]
            ].merge(kw_pos, on="keyword", how="left")
        )
        save_table(kw_pos_top, techB / "keyword_pos_rate_topk")

    if "country_code" in train.columns and "country_code" in dev.columns:
        cc_shift = split_shift_tables(train, dev, test, col="country_code", top_k=int(args.top_k))
        save_table(cc_shift, techB / "country_shift_topk")

        plt.figure(figsize=(10, 5))
        x = np.arange(len(cc_shift))
        width = 0.25
        plt.bar(x - width, cc_shift["train_pct"], width=width, label="train")
        plt.bar(x, cc_shift["dev_pct"], width=width, label="dev")
        if test is not None:
            plt.bar(x + width, cc_shift["test_pct"], width=width, label="test")
        plt.xticks(x, cc_shift["country_code"].astype(str).tolist(), rotation=35, ha="right")
        plt.ylabel("Percent of split")
        plt.title(f"Country distribution shift (top {int(args.top_k)} by train frequency)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(techB / "country_shift_topk.png", dpi=200)
        plt.close()

        def _pos_rate(df: pd.DataFrame, split: str) -> pd.DataFrame:
            g = df.groupby("country_code")["y"].agg(["count", "mean"]).reset_index()
            g["split"] = split
            g.rename(columns={"mean": "p(y=1)"}, inplace=True)
            return g

        cc_pos = pd.concat([_pos_rate(train, "train"), _pos_rate(dev, "dev")], ignore_index=True)
        save_table(cc_pos.sort_values(["split", "count"], ascending=[True, False]), techB / "country_pos_rate_all")

        cc_pos_top = (
            cc_pos[cc_pos["split"] == "train"].sort_values("count", ascending=False).head(int(args.top_k))[
                ["country_code"]
            ].merge(cc_pos, on="country_code", how="left")
        )
        save_table(cc_pos_top, techB / "country_pos_rate_topk")

    # Aux category frequency + co-occurrence (train/dev)
    freq_tr, co_tr = aux_category_tables(train, "train")
    freq_dv, co_dv = aux_category_tables(dev, "dev")
    save_table(freq_tr, techB / "aux_category_freq_train")
    save_table(freq_dv, techB / "aux_category_freq_dev")

    # Co-occurrence heatmaps (only if real square matrices)
    def _maybe_heatmap(co_df: pd.DataFrame, split: str) -> None:
        if "note" in co_df.columns:
            return
        # co_df has 'split' first column; remove it
        mat = co_df.drop(columns=["split"])
        plot_heatmap(mat, techB / f"aux_category_cooccurrence_{split}.png", f"Aux category co-occurrence (counts) — {split}")

    _maybe_heatmap(co_tr, "train")
    _maybe_heatmap(co_dv, "dev")

    # Categories per positive example (how multi-label is it?)
    cat_cols = [f"cat_{c}" for c in CANONICAL_CATEGORIES]
    have_cols = [c for c in cat_cols if c in train.columns]
    if have_cols and "has_aux_labels" in train.columns:
        def _k_dist(df: pd.DataFrame, split: str) -> pd.DataFrame:
            pos = df[(df["y"].astype(int) == 1) & (df["has_aux_labels"].astype(bool))].copy()
            if pos.empty:
                return pd.DataFrame([{"split": split, "k": 0, "count": 0}])
            k = pos[have_cols].astype(int).sum(axis=1)
            vc = k.value_counts().sort_index()
            return pd.DataFrame({"split": split, "k": vc.index.astype(int), "count": vc.to_numpy()})

        kdist = pd.concat([_k_dist(train, "train"), _k_dist(dev, "dev")], ignore_index=True)
        save_table(kdist, techB / "aux_num_categories_per_pos")
        plot_bar_counts(kdist, x="k", y="count", hue="split", out_png=techB / "aux_num_categories_per_pos.png",
                        title="How many aux categories per positive paragraph?")

    # Save manifest
    manifest_path = out_dir / "manifest.json"
    manifest["outputs"] = [
        str(p.relative_to(REPO_ROOT)) for p in sorted(out_dir.rglob("*")) if p.is_file()
    ]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n✅ EDA complete.")
    print(f"Outputs written under: {out_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Tokenizer source: {tok_src}")
    print("\nNext: paste back (1) the saved figures you plan to use, and (2) the key tables/rows you want in the report.")


if __name__ == "__main__":
    main()
