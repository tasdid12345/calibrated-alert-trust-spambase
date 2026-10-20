"""Baseline (random forest, raw probabilities) vs improved (class-balanced LightGBM + isotonic calibration) on Spambase (OpenML 44)."""
import json
import socket
import sys
import time

import lightgbm as lgb
import matplotlib
import numpy as np
import openml

matplotlib.use("Agg")  # headless plotting for GitHub Actions
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import brier_score_loss, f1_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

SEED = 42
SEEDS = [42, 43, 44]          # both models are randomized, so both are run with each seed
DATA_ID = 44                  # Spambase on OpenML
SOURCE_URL = "https://www.openml.org/d/44"
MAX_ROWS = 200_000            # safety cap for runtime; Spambase (~4.6k rows) is far below it
TEST_SIZE = 0.25
N_BINS = 10

socket.setdefaulttimeout(60)  # bounds any stalled network call (timeout=60)


def load_data():
    """Download Spambase via the openml package, retrying with exponential backoff."""
    for attempt in range(5):
        try:
            ds = openml.datasets.get_dataset(DATA_ID)
            X, y, _, _ = ds.get_data(target=ds.default_target_attribute, dataset_format="dataframe")
            return X, y
        except Exception as exc:  # network errors, HTTP 429 / 5xx
            wait = 2 ** attempt
            print(f"download failed ({exc}); retrying in {wait}s")
            time.sleep(wait)
    sys.exit("could not download OpenML dataset 44")


def ece(y, p, bins=N_BINS):
    """Expected calibration error with equal-width bins: bin-weighted |mean prob - observed rate|."""
    idx = np.minimum((p * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def evaluate(y, p):
    """Shared evaluation for both models: Brier (main), macro-F1 at 0.5, ECE."""
    return {
        "brier": float(brier_score_loss(y, p)),
        "f1": float(f1_score(y, (p >= 0.5).astype(int), average="macro")),
        "ece": ece(y, p),
    }


def run_baseline(seed, Xtr, ytr):
    """Standard 300-tree random forest; raw predicted probabilities are used as-is."""
    return RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1).fit(Xtr, ytr)


def run_improved(seed, Xtr, ytr):
    """Class-balanced LightGBM with isotonic calibration fitted by 5-fold CV on the training split."""
    base = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                              class_weight="balanced", random_state=seed, n_jobs=-1, verbose=-1)
    return CalibratedClassifierCV(estimator=base, method="isotonic", cv=5).fit(Xtr, ytr)


def main():
    print("loading Spambase (OpenML 44)")
    X, y = load_data()
    X = X.to_numpy(dtype=float)
    classes, y = np.unique(y.astype(str).to_numpy(), return_inverse=True)
    if len(classes) != 2:
        sys.exit("expected a binary target")
    print(f"positive class = '{classes[1]}'")

    if len(X) > MAX_ROWS:  # subsample very large data (not triggered for Spambase)
        keep = np.random.default_rng(SEED).choice(len(X), MAX_ROWS, replace=False)
        X, y = X[keep], y[keep]

    # Single stratified split, fixed seed; the test set is only touched in the final evaluation.
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    print(f"train={len(Xtr)} test={len(Xte)}")

    runs = {"baseline": [], "improved": []}
    for seed in SEEDS:
        print(f"seed {seed}: fitting baseline random forest")
        p_b = run_baseline(seed, Xtr, ytr).predict_proba(Xte)[:, 1]
        print(f"seed {seed}: fitting improved balanced LightGBM + isotonic")
        p_i = run_improved(seed, Xtr, ytr).predict_proba(Xte)[:, 1]
        runs["baseline"].append(evaluate(yte, p_b))
        runs["improved"].append(evaluate(yte, p_i))

    def agg(name, key):
        v = np.array([r[key] for r in runs[name]])
        return float(v.mean()), float(v.std())

    b_brier, b_brier_sd = agg("baseline", "brier")
    i_brier, i_brier_sd = agg("improved", "brier")
    b_f1, b_f1_sd = agg("baseline", "f1")
    i_f1, i_f1_sd = agg("improved", "f1")
    b_ece, b_ece_sd = agg("baseline", "ece")
    i_ece, i_ece_sd = agg("improved", "ece")
    metrics = {
        "baseline_score": b_brier, "improved_score": i_brier,
        "improvement": b_brier - i_brier,  # Brier is lower-better
        "baseline_score_std": b_brier_sd, "improved_score_std": i_brier_sd,
        "baseline_f1": b_f1, "improved_f1": i_f1,
        "baseline_f1_std": b_f1_sd, "improved_f1_std": i_f1_sd,
        "baseline_ece": b_ece, "improved_ece": i_ece,
        "baseline_ece_std": b_ece_sd, "improved_ece_std": i_ece_sd,
        "n_seeds": len(SEEDS),
    }
    if not all(np.isfinite(v) for v in metrics.values()):
        sys.exit("non-finite metric; failing")

    # Optional chart: Brier score, baseline vs improved, with seed std error bars.
    import os
    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["baseline RF", "balanced LGBM\n+ isotonic"], [b_brier, i_brier],
           yerr=[b_brier_sd, i_brier_sd], capsize=6, color=["#999999", "#3572b0"])
    ax.set_ylabel("Brier score (test, lower is better)")
    ax.set_title("Spambase: calibration of alert probabilities")
    fig.tight_layout()
    fig.savefig("figures/brier_comparison.png", dpi=120)

    results = {
        "question": "Can a class-balanced, isotonic-calibrated LightGBM model give more trustworthy alert probabilities than a standard random forest on the public Spambase alert-style dataset?",
        "method": ("Single stratified 75/25 split (seed 42) of OpenML Spambase. Baseline: 300-tree random forest with raw probabilities. "
                   "Improved: class-balanced LightGBM with isotonic calibration (5-fold CV on the training split). "
                   "Both run with seeds 42, 43, 44 and evaluated identically on the test split; no test-set tuning."),
        "dataset": {
            "name": "Spambase (OpenML ID 44)",
            "source_url": SOURCE_URL,
            "n_rows": int(len(X)),
            "n_train": int(len(Xtr)),
            "n_test": int(len(Xte)),
            "same_as_paper": False,
            "baseline_is_papers_method": False,
        },
        "metric_name": "Brier score on test split (lower is better)",
        "higher_is_better": False,
        "baseline": "Standard random forest (300 trees, default settings) using raw predicted probabilities.",
        "improved": "Class-balanced LightGBM (class_weight=balanced) with isotonic probability calibration via 5-fold CV on the training split.",
        "metrics": metrics,
        "notes": ("Spambase is a spam-email stand-in for security alerts, not an alert dataset. Single split; the std values only reflect "
                  "model randomness over 3 seeds, not split variability. No hyperparameter search was run for either model. "
                  "Macro-F1 uses a 0.5 threshold on the probabilities. ECE uses 10 equal-width bins. "
                  "Class-balancing shifts probabilities away from the base rate; isotonic calibration is intended to correct this, "
                  "but results may not transfer to real alert streams."),
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Brier baseline={b_brier:.4f} improved={i_brier:.4f}; results.json written")


if __name__ == "__main__":
    main()
