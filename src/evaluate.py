"""Metrics suited to an imbalanced risk-prediction problem."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)


def classification_metrics(y_true, probability, threshold: float = 0.5) -> dict[str, float]:
    probability = np.asarray(probability)
    predicted = probability >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "roc_auc": roc_auc_score(y_true, probability),
        "pr_auc": average_precision_score(y_true, probability),
        "accuracy": accuracy_score(y_true, predicted),
        "precision": precision_score(y_true, predicted, zero_division=0),
        "recall_sensitivity": recall_score(y_true, predicted, zero_division=0),
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "f1": f1_score(y_true, predicted, zero_division=0),
        "brier_score": brier_score_loss(y_true, probability),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def calibration_points(y_true, probability, bins: int = 10) -> pd.DataFrame:
    observed, predicted = calibration_curve(y_true, probability, n_bins=bins, strategy="quantile")
    return pd.DataFrame({"mean_predicted_probability": predicted,
                         "observed_fraction_positive": observed})


def subgroup_metrics(metadata: pd.DataFrame, y_true, probability) -> pd.DataFrame:
    rows = []
    for column in [c for c in ["gender", "age", "race"] if c in metadata]:
        values = metadata[column].fillna("Missing").astype(str)
        for group in sorted(values.unique()):
            mask = values.eq(group).to_numpy()
            if mask.sum() < 30 or len(np.unique(np.asarray(y_true)[mask])) < 2:
                continue
            m = classification_metrics(np.asarray(y_true)[mask], np.asarray(probability)[mask])
            rows.append({"attribute": column, "group": group, "n": int(mask.sum()),
                         "prevalence": float(np.asarray(y_true)[mask].mean()), **m})
    return pd.DataFrame(rows)
