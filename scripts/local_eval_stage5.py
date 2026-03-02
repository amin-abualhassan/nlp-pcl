#!/usr/bin/env python3
"""
Stage 5.2 Local Evaluation generator for PCL NLP cw.

Outputs (under --out_dir):
- metrics_overall.md
- confusion_matrix.csv
- threshold_sweep.csv + threshold_sweep.png
- pr_curve.csv + pr_curve.png
- calibration.csv + calibration.png
- score_hist.png
- slices_keyword.csv
- slices_country.csv
- slices_label_scale.csv
- slices_length_bucket.csv
- category_recall.csv
- top_errors_fp.md / top_errors_fn.md
- compare_alt_run.md (if --pred_alt provided)
- metadata_baseline.md (if --train_csv provided and sklearn available)
- bootstrap_ci.md
"""

from __future__ import annotations

import argparse
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------
# Helpers
# ----------------------------

def read_lines_01(path: str | Path) -> np.ndarray:
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    lines = [ln.strip() for ln in lines if ln.strip() != ""]
    bad = [(i + 1, ln) for i, ln in enumerate(lines) if ln not in {"0", "1"}]
    if bad:
        raise ValueError(f"{path}: non 0/1 lines (first 10): {bad[:10]}")
    return np.array([int(x) for x in lines], dtype=np.int64)

def snippet(text: str, n: int = 220) -> str:
    t = re.sub(r"\s+", " ", str(text)).strip()
    if len(t) <= n:
        return t
    return t[: n - 1].rstrip() + "…"

def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def f1_pr(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return f1, p, r

def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}

def apply_threshold(probs: np.ndarray, t: float) -> np.ndarray:
    return (probs >= t).astype(np.int64)

def load_dev_df(dev_csv: Path) -> pd.DataFrame:
    # Your dev_df_2.csv is TSV.
    df = pd.read_csv(dev_csv, sep="\t")
    expected = {"par_id", "keyword", "country_code", "text", "label"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{dev_csv}: missing columns {sorted(missing)}; got {list(df.columns)}")

    if "target_flag" in df.columns:
        df["y"] = df["target_flag"].astype(int)
    else:
        # fallback: label >= 2 -> positive
        df["y"] = (df["label"].astype(int) >= 2).astype(int)

    df["label"] = df["label"].astype(int)
    df["word_count"] = df["text"].astype(str).apply(lambda s: len(str(s).split()))
    return df

def read_selection_report(path: Path) -> Tuple[Path, float]:
    import json
    j = json.loads(path.read_text(encoding="utf-8"))
    sel = j["selected"]
    run_dir = Path(sel["run_dir"])
    t = float(sel["threshold"])
    return run_dir, t

def parse_span_categories_tsv(tsv_path: Path) -> pd.DataFrame:
    """
    File format (after disclaimer lines):
    par_id, art_id, text, keyword, country_code, span_start, span_end, span_text, category, n_annotators
    """
    lines = tsv_path.read_text(encoding="utf-8", errors="replace").splitlines()
    rows = []
    for ln in lines:
        ln = ln.strip()
        if not ln or ln.startswith("-"):
            continue
        parts = ln.split("\t")
        if len(parts) < 10:
            continue
        if not re.match(r"^\d+$", parts[0]):
            continue
        rows.append(parts[:10])

    cols = [
        "par_id", "art_id", "text", "keyword", "country_code",
        "span_start", "span_end", "span_text", "category", "n_annotators"
    ]
    df = pd.DataFrame(rows, columns=cols)
    df["par_id"] = df["par_id"].astype(int)
    df["n_annotators"] = df["n_annotators"].astype(int)
    return df

def bin_by_quantiles(x: pd.Series, q: int = 5) -> pd.Series:
    # Robust bucketing
    try:
        return pd.qcut(x, q=q, duplicates="drop")
    except Exception:
        return pd.cut(x, bins=q)

def bootstrap_f1(y: np.ndarray, p: np.ndarray, t: float, n: int = 2000, seed: int = 42) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    vals = []
    for _ in range(n):
        samp = rng.choice(idx, size=len(y), replace=True)
        yy = y[samp]
        pp = p[samp]
        pred = apply_threshold(pp, t)
        c = confusion(yy, pred)
        f1, _, _ = f1_pr(c["tp"], c["fp"], c["fn"])
        vals.append(f1)
    vals = np.array(vals)
    return float(vals.mean()), float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


# ----------------------------
# Main analysis
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev_csv", type=str, default="data/dev_df_2.csv")
    ap.add_argument("--pred", type=str, default="dev.txt")
    ap.add_argument("--selection_report", type=str, default="selection_report.json")
    ap.add_argument("--probs", type=str, default="", help="Optional: dev_probs.npy; if empty we read from selected run_dir/dev_probs.npy")
    ap.add_argument("--cats_tsv", type=str, default="data/raw/dontpatronizeme_categories.tsv")
    ap.add_argument("--span_min_annotators", type=int, default=2)
    ap.add_argument("--pred_alt", type=str, default="", help="Optional: dev.txt.bak to compare an alternative run")
    ap.add_argument("--train_csv", type=str, default="data/train_df.csv", help="Used for metadata-only baseline (optional)")
    ap.add_argument("--no_metadata_baseline", action="store_true")
    ap.add_argument("--out_dir", type=str, default="reports/local_eval")
    ap.add_argument("--top_k", type=int, default=12)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    safe_mkdir(out_dir)

    dev_df = load_dev_df(Path(args.dev_csv))
    y = dev_df["y"].to_numpy(dtype=np.int64)

    # Locate probs and threshold from selection_report
    run_dir, t_star = read_selection_report(Path(args.selection_report))

    probs_path = Path(args.probs) if args.probs else (run_dir / "dev_probs.npy")
    if not probs_path.exists():
        raise FileNotFoundError(f"dev_probs.npy not found at: {probs_path}")
    probs = np.load(probs_path)

    pred_file = read_lines_01(args.pred)
    if len(pred_file) != len(dev_df):
        raise ValueError(f"{args.pred}: {len(pred_file)} lines, but dev has {len(dev_df)} rows")

    # Sanity: prob->threshold predictions must match dev.txt (should, given your pipeline)
    pred_from_probs = apply_threshold(probs, t_star)
    mismatch = int((pred_from_probs != pred_file).sum())
    if mismatch != 0:
        # Not fatal, but worth flagging in outputs.
        mismatch_note = f"WARNING: {mismatch} predictions differ between dev.txt and probs>=t*.\n"
    else:
        mismatch_note = ""

    # Overall metrics
    c = confusion(y, pred_file)
    f1, ppos, rpos = f1_pr(c["tp"], c["fp"], c["fn"])

    # Save confusion matrix
    cm = pd.DataFrame(
        [[c["tn"], c["fp"]], [c["fn"], c["tp"]]],
        index=["y=0", "y=1"],
        columns=["pred=0", "pred=1"],
    )
    cm.to_csv(out_dir / "confusion_matrix.csv", index=True)

    # Threshold sweep (0..1)
    ts = np.round(np.arange(0.0, 1.0001, 0.01), 2)
    sweep_rows = []
    for t in ts:
        pr = apply_threshold(probs, float(t))
        cc = confusion(y, pr)
        f1t, pt, rt = f1_pr(cc["tp"], cc["fp"], cc["fn"])
        sweep_rows.append((t, f1t, pt, rt, cc["tp"], cc["fp"], cc["fn"], cc["tn"]))
    sweep = pd.DataFrame(sweep_rows, columns=["threshold", "f1_pos", "precision_pos", "recall_pos", "tp", "fp", "fn", "tn"])
    sweep.to_csv(out_dir / "threshold_sweep.csv", index=False)

    # Plot threshold sweep
    plt.figure()
    plt.plot(sweep["threshold"], sweep["f1_pos"], label="F1 (pos)")
    plt.plot(sweep["threshold"], sweep["precision_pos"], label="Precision (pos)")
    plt.plot(sweep["threshold"], sweep["recall_pos"], label="Recall (pos)")
    plt.axvline(t_star, linestyle="--")
    plt.xlabel("threshold")
    plt.ylabel("score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "threshold_sweep.png", dpi=200)
    plt.close()

    # PR curve (manual, from sweep; good enough for report)
    pr_curve = sweep.sort_values("recall_pos")
    pr_curve[["recall_pos", "precision_pos", "threshold"]].to_csv(out_dir / "pr_curve.csv", index=False)

    plt.figure()
    plt.plot(pr_curve["recall_pos"], pr_curve["precision_pos"])
    plt.xlabel("Recall (pos)")
    plt.ylabel("Precision (pos)")
    plt.tight_layout()
    plt.savefig(out_dir / "pr_curve.png", dpi=200)
    plt.close()

    # Calibration curve (10 bins)
    cal = pd.DataFrame({"p": probs, "y": y})
    cal["bin"] = pd.cut(cal["p"], bins=np.linspace(0, 1, 11), include_lowest=True)
    cal_agg = cal.groupby("bin", observed=True).agg(
        mean_p=("p", "mean"),
        frac_pos=("y", "mean"),
        n=("y", "size"),
    ).reset_index(drop=True)
    cal_agg.to_csv(out_dir / "calibration.csv", index=False)

    plt.figure()
    plt.plot(cal_agg["mean_p"], cal_agg["frac_pos"], marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Empirical positive rate")
    plt.tight_layout()
    plt.savefig(out_dir / "calibration.png", dpi=200)
    plt.close()

    # Score histogram (pos vs neg)
    plt.figure()
    plt.hist(probs[y == 0], bins=30, alpha=0.7, label="y=0")
    plt.hist(probs[y == 1], bins=30, alpha=0.7, label="y=1")
    plt.axvline(t_star, linestyle="--")
    plt.xlabel("predicted probability")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "score_hist.png", dpi=200)
    plt.close()

    # Add columns for slicing
    dev_df = dev_df.copy()
    dev_df["pred"] = pred_file
    dev_df["prob"] = probs
    dev_df["is_fp"] = (dev_df["pred"] == 1) & (dev_df["y"] == 0)
    dev_df["is_fn"] = (dev_df["pred"] == 0) & (dev_df["y"] == 1)
    dev_df["is_tp"] = (dev_df["pred"] == 1) & (dev_df["y"] == 1)
    dev_df["is_tn"] = (dev_df["pred"] == 0) & (dev_df["y"] == 0)

    # Slice metrics helper
    def slice_metrics(g: pd.DataFrame) -> pd.Series:
        yy = g["y"].to_numpy()
        pp = g["pred"].to_numpy()
        cc = confusion(yy, pp)
        f1g, pg, rg = f1_pr(cc["tp"], cc["fp"], cc["fn"])
        return pd.Series({
            "n": len(g),
            "pos": int(yy.sum()),
            "precision_pos": pg,
            "recall_pos": rg,
            "f1_pos": f1g,
            "fp": cc["fp"],
            "fn": cc["fn"],
        })

    # Keyword slice (keep only reasonably-supported slices for interpretation)
    kw = dev_df.groupby("keyword").apply(slice_metrics).reset_index()
    kw.to_csv(out_dir / "slices_keyword.csv", index=False)

    # Country slice
    ct = dev_df.groupby("country_code").apply(slice_metrics).reset_index()
    ct.to_csv(out_dir / "slices_country.csv", index=False)

    # Label scale slice (0..4)
    ls = dev_df.groupby("label").apply(slice_metrics).reset_index().sort_values("label")
    ls.to_csv(out_dir / "slices_label_scale.csv", index=False)

    # Length bucket slice
    dev_df["len_bucket"] = bin_by_quantiles(dev_df["word_count"], q=5).astype(str)
    lb = dev_df.groupby("len_bucket").apply(slice_metrics).reset_index()
    lb.to_csv(out_dir / "slices_length_bucket.csv", index=False)

    # Top errors (high-confidence)
    fps = dev_df[dev_df["is_fp"]].sort_values("prob", ascending=False).head(args.top_k)
    fns = dev_df[dev_df["is_fn"]].sort_values("prob", ascending=True).head(args.top_k)

    def dump_errors_md(df_err: pd.DataFrame, title: str, path: Path) -> None:
        lines = [f"# {title}", ""]
        for _, r in df_err.iterrows():
            lines.append(
                f"- par_id={int(r.par_id)} | prob={r.prob:.3f} | keyword={r.keyword} | country={r.country_code} | label_scale={int(r.label)}"
            )
            lines.append(f"  - text: {snippet(r.text)}")
        path.write_text("\n".join(lines), encoding="utf-8")

    dump_errors_md(fps, "Top False Positives (highest prob, y=0)", out_dir / "top_errors_fp.md")
    dump_errors_md(fns, "Top False Negatives (lowest prob, y=1)", out_dir / "top_errors_fn.md")

    # Category analysis from span annotations (filtered to dev + min annotators)
    cats_tsv = Path(args.cats_tsv)
    if cats_tsv.exists():
        cats = parse_span_categories_tsv(cats_tsv)
        cats = cats[(cats["par_id"].isin(set(dev_df["par_id"].astype(int)))) & (cats["n_annotators"] >= args.span_min_annotators)]
        # Aggregate per paragraph -> set(categories)
        pid_to_cats: Dict[int, set] = {}
        for pid, g in cats.groupby("par_id"):
            pid_to_cats[int(pid)] = set(g["category"].tolist())
        dev_df["cats"] = dev_df["par_id"].astype(int).map(lambda pid: sorted(pid_to_cats.get(int(pid), set())))

        pos_with = dev_df[(dev_df["y"] == 1) & (dev_df["cats"].map(len) > 0)]
        rows = []
        all_cats = sorted(set(cats["category"].tolist()))
        for cat in all_cats:
            sub = pos_with[pos_with["cats"].apply(lambda cs: cat in cs)]
            if len(sub) == 0:
                continue
            tp_c = int((sub["pred"] == 1).sum())
            fn_c = int((sub["pred"] == 0).sum())
            rec_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) else 0.0
            rows.append((cat, len(sub), tp_c, fn_c, rec_c))
        cat_recall = pd.DataFrame(rows, columns=["category", "n_pos_with_cat", "tp", "fn", "recall"]).sort_values("recall")
        cat_recall.to_csv(out_dir / "category_recall.csv", index=False)
    else:
        (out_dir / "category_recall.csv").write_text("cats_tsv missing\n", encoding="utf-8")

    # Compare alternative prediction file (e.g., fold-weighted)
    if args.pred_alt:
        pred_alt = read_lines_01(args.pred_alt)
        if len(pred_alt) != len(dev_df):
            raise ValueError(f"{args.pred_alt}: {len(pred_alt)} lines, but dev has {len(dev_df)} rows")

        c_alt = confusion(y, pred_alt)
        f1a, pa, ra = f1_pr(c_alt["tp"], c_alt["fp"], c_alt["fn"])

        # Where alt fixes us vs hurts us
        base_wrong = (pred_file != y)
        alt_wrong = (pred_alt != y)
        alt_fixes = dev_df[base_wrong & (~alt_wrong)].copy()
        alt_hurts = dev_df[(~base_wrong) & alt_wrong].copy()

        alt_fixes = alt_fixes.assign(pred_alt=pred_alt[base_wrong & (~alt_wrong)])
        alt_hurts = alt_hurts.assign(pred_alt=pred_alt[(~base_wrong) & alt_wrong])

        md = []
        md.append("# Compare alternative run vs submitted run")
        md.append("")
        md.append(f"Submitted:  f1={f1:.4f} p={ppos:.4f} r={rpos:.4f} (t*={t_star:.2f})")
        md.append(f"Alt pred:   f1={f1a:.4f} p={pa:.4f} r={ra:.4f}")
        md.append("")
        md.append(f"- Alt fixes (we were wrong, alt is right): {len(alt_fixes)}")
        md.append(f"- Alt hurts (we were right, alt is wrong): {len(alt_hurts)}")
        md.append("")
        md.append("## Alt fixes (examples)")
        for _, r in alt_fixes.sort_values("prob", ascending=False).head(8).iterrows():
            md.append(f"- par_id={int(r.par_id)} | prob={r.prob:.3f} | y={int(r.y)} | ours={int(r.pred)} | alt={int(pred_alt[int(r.name)])}")
            md.append(f"  - {snippet(r.text)}")
        md.append("")
        md.append("## Alt hurts (examples)")
        for _, r in alt_hurts.sort_values("prob", ascending=False).head(8).iterrows():
            md.append(f"- par_id={int(r.par_id)} | prob={r.prob:.3f} | y={int(r.y)} | ours={int(r.pred)} | alt={int(pred_alt[int(r.name)])}")
            md.append(f"  - {snippet(r.text)}")

        (out_dir / "compare_alt_run.md").write_text("\n".join(md), encoding="utf-8")

    # Bootstrap CI for F1 at t*
    mean_f1, lo, hi = bootstrap_f1(y=y, p=probs, t=t_star, n=2000, seed=42)
    (out_dir / "bootstrap_ci.md").write_text(
        "\n".join([
            "# Bootstrap CI (dev F1_pos)",
            "",
            f"- threshold t* = {t_star:.2f}",
            f"- bootstrap mean F1_pos = {mean_f1:.4f}",
            f"- 95% CI ≈ [{lo:.4f}, {hi:.4f}]",
        ]),
        encoding="utf-8",
    )

    # Metadata-only baseline (keyword/country/length) trained on train, evaluated on dev
    if not args.no_metadata_baseline:
        try:
            from sklearn.compose import ColumnTransformer
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import OneHotEncoder
        except Exception:
            (out_dir / "metadata_baseline.md").write_text(
                "sklearn not available; skipped metadata baseline.\n",
                encoding="utf-8",
            )
        else:
            train_path = Path(args.train_csv)
            if not train_path.exists():
                (out_dir / "metadata_baseline.md").write_text(
                    f"{train_path} missing; skipped metadata baseline.\n",
                    encoding="utf-8",
                )
            else:
                train_df = pd.read_csv(train_path, sep="\t")
                if "target_flag" in train_df.columns:
                    train_df["y"] = train_df["target_flag"].astype(int)
                else:
                    train_df["y"] = (train_df["label"].astype(int) >= 2).astype(int)
                train_df["word_count"] = train_df["text"].astype(str).apply(lambda s: len(str(s).split()))

                X_train = train_df[["keyword", "country_code", "word_count"]].copy()
                y_train = train_df["y"].to_numpy(dtype=np.int64)
                X_dev = dev_df[["keyword", "country_code", "word_count"]].copy()

                # Very simple baseline: keyword + country + length (no text)
                pre = ColumnTransformer(
                    transformers=[
                        ("cat", OneHotEncoder(handle_unknown="ignore"), ["keyword", "country_code"]),
                        ("num", "passthrough", ["word_count"]),
                    ],
                    remainder="drop",
                )

                clf = LogisticRegression(max_iter=2000, class_weight="balanced")
                pipe = Pipeline([("pre", pre), ("clf", clf)])
                pipe.fit(X_train, y_train)
                dev_prob_meta = pipe.predict_proba(X_dev)[:, 1]

                # Report both (a) 0.5 threshold and (b) best threshold on dev (upper bound)
                pred05 = (dev_prob_meta >= 0.5).astype(np.int64)
                c05 = confusion(y, pred05)
                f105, p05, r05 = f1_pr(c05["tp"], c05["fp"], c05["fn"])

                # Find best threshold on dev for this baseline (optimistic)
                best = None
                for t in np.linspace(0, 1, 201):
                    prd = (dev_prob_meta >= t).astype(np.int64)
                    cc = confusion(y, prd)
                    f1t, pt, rt = f1_pr(cc["tp"], cc["fp"], cc["fn"])
                    if (best is None) or (f1t > best[0]):
                        best = (f1t, pt, rt, float(t))

                f1b, pb, rb, tb = best

                (out_dir / "metadata_baseline.md").write_text(
                    "\n".join([
                        "# Metadata-only baseline (sanity check)",
                        "",
                        "Features: keyword + country_code + word_count (no text).",
                        "",
                        f"- threshold=0.50: f1={f105:.4f} p={p05:.4f} r={r05:.4f}",
                        f"- best-on-dev threshold={tb:.3f} (optimistic upper bound): f1={f1b:.4f} p={pb:.4f} r={rb:.4f}",
                        "",
                        "Purpose: quantify how far topic priors + metadata can go, and show that strong performance requires modelling the text.",
                    ]),
                    encoding="utf-8",
                )

    # Write a compact overall summary MD
    (out_dir / "metrics_overall.md").write_text(
        "\n".join([
            "# Local evaluation summary (dev)",
            "",
            mismatch_note.rstrip(),
            f"- threshold t* (from selection_report): {t_star:.2f}",
            f"- F1_pos: {f1:.4f}",
            f"- Precision_pos: {ppos:.4f}",
            f"- Recall_pos: {rpos:.4f}",
            "",
            "Confusion counts:",
            f"- TP={c['tp']} FP={c['fp']} FN={c['fn']} TN={c['tn']}",
        ]).strip() + "\n",
        encoding="utf-8",
    )

    print(f"OK: wrote local eval outputs to: {out_dir}")


if __name__ == "__main__":
    main()