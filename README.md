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

## Reference

This project is motivated by the following paper:

> G Dolcetti, V Arceri, A Cortesi, E Zaffanella (2025). *On the Verification of ML Systems and Models*. CEUR WORKSHOP. https://iris.unive.it/handle/10278/5118107

Paper author: Prof. Agostino Cortesi, Ca' Foscari University of Venice (cortesi@unive.it)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
