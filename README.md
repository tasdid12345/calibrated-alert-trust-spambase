# Calibrated LightGBM vs Random Forest for Spam-Style Alert Probabilities

Motivation: "On the Verification of ML Systems and Models" (2025) argues that machine-learning systems used in security settings must be verified, and that their outputs must be trustworthy and not just accurate. For alert triage, a well-calibrated probability is what lets analysts set sensible thresholds, so this experiment checks how calibration differs between two standard classifiers.

Question: Can a class-balanced, isotonic-calibrated LightGBM model give more trustworthy alert probabilities than a standard random forest on the public Spambase alert-style dataset?

Dataset: Spambase (OpenML ID 44), about 4,600 emails with 57 features, spam vs. non-spam. It is used as a stand-in for security alerts. It is not the paper's own dataset.

Baseline: a standard 300-tree random forest that reports raw predicted probabilities.

Improved model: class-balanced LightGBM with isotonic calibration fitted by 5-fold cross-validation on the training split. Both models use the same split, preprocessing, evaluation code, and seeds.

## How to run

```
pip install -r requirements.txt
python analysis.py
```

Results: results are written to `results.json` (and a chart to `figures/`).

## Results (latest run)

On the public Spambase dataset (OpenML ID 44, one 75/25 split), a standard random forest with raw probabilities reached a Brier score of 0.042 on the test set, while a class-balanced, isotonic-calibrated LightGBM reached 0.034, a small improvement (lower is better).

| Metric | Value |
|---|---|
| baseline_score | 0.04156 |
| improved_score | 0.03433 |
| improvement | 0.007221 |
| baseline_score_std | 0.0002923 |
| improved_score_std | 0 |
| baseline_f1 | 0.9483 |
| improved_f1 | 0.9564 |
| baseline_f1_std | 0.001875 |
| improved_f1_std | 0 |
| baseline_ece | 0.04996 |
| improved_ece | 0.01396 |
| baseline_ece_std | 0.001543 |
| improved_ece_std | 0 |
| n_seeds | 3 |

Run on 2026-10-20. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Spambase is a spam-email stand-in for security alerts, not an alert dataset. Single split; the std values only reflect model randomness over 3 seeds, not split variability. No hyperparameter search was run for either model. Macro-F1 uses a 0.5 threshold on the probabilities. ECE uses 10 equal-width bins. Class-balancing shifts probabilities away from the base rate; isotonic calibration is intended to correct this, but results may not transfer to real alert streams.

## Reference

This project is motivated by the following paper:

> G Dolcetti, V Arceri, A Cortesi, E Zaffanella (2025). *On the Verification of ML Systems and Models*. CEUR WORKSHOP. https://iris.unive.it/handle/10278/5118107

Paper author: Prof. Agostino Cortesi, Ca' Foscari University of Venice (cortesi@unive.it)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
