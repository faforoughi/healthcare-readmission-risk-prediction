"""Reproducible training CLI with an untouched patient-level test set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from .data_loader import PATIENT_ID, load_data, missingness_table
from .evaluate import calibration_points, classification_metrics, subgroup_metrics
from .preprocessing import (RANDOM_STATE, build_preprocessor, patient_level_split,
                            split_xy)


def model_specs(seed: int):
    specs = {
        "majority_baseline": (DummyClassifier(strategy="prior"), {}),
        "logistic_regression": (
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
            {"model__C": [0.05, 0.2, 1.0, 5.0]},
        ),
        "random_forest": (
            RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=seed),
            {"model__n_estimators": [200, 400], "model__max_depth": [8, 14, None],
             "model__min_samples_leaf": [2, 5, 10], "model__max_features": ["sqrt", 0.5]},
        ),
    }
    try:
        from xgboost import XGBClassifier
        specs["xgboost"] = (
            XGBClassifier(n_jobs=-1, random_state=seed, eval_metric="logloss",
                          tree_method="hist"),
            {"model__n_estimators": [200, 400], "model__max_depth": [3, 5, 7],
             "model__learning_rate": [0.02, 0.05, 0.1],
             "model__subsample": [0.7, 0.9, 1.0],
             "model__colsample_bytree": [0.7, 0.9, 1.0]},
        )
    except ImportError:
        pass
    return specs


def fit_models(train: pd.DataFrame, tune: bool, seed: int, n_iter: int):
    x_train, y_train = split_xy(train)
    groups = train[PATIENT_ID]
    fitted = {}
    validation = []
    for name, (model, parameters) in model_specs(seed).items():
        if name == "xgboost":
            positives = int(y_train.sum())
            model.set_params(scale_pos_weight=(len(y_train) - positives) / positives)
        pipe = Pipeline([("preprocess", build_preprocessor(x_train)), ("model", model)])
        if tune and parameters:
            cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
            search = RandomizedSearchCV(pipe, parameters, n_iter=min(n_iter, 12),
                                        scoring="average_precision", cv=cv, n_jobs=1,
                                        random_state=seed, refit=True)
            search.fit(x_train, y_train, groups=groups)
            fitted[name] = search.best_estimator_
            validation.append({"model": name, "validation_pr_auc": search.best_score_,
                               "parameters": json.dumps(search.best_params_, sort_keys=True)})
        else:
            pipe.fit(x_train, y_train)
            fitted[name] = pipe
            validation.append({"model": name, "validation_pr_auc": None, "parameters": "{}"})
    return fitted, pd.DataFrame(validation)


def run(data_path: Path, output: Path, tune: bool = False, seed: int = RANDOM_STATE,
        n_iter: int = 8):
    output.mkdir(parents=True, exist_ok=True)
    frame = load_data(data_path)
    train, test = patient_level_split(frame, random_state=seed)
    models, validation = fit_models(train, tune=tune, seed=seed, n_iter=n_iter)
    x_test, y_test = split_xy(test)
    rows, probabilities = [], {}
    for name, model in models.items():
        probability = model.predict_proba(x_test)[:, 1]
        probabilities[name] = probability
        rows.append({"model": name, **classification_metrics(y_test, probability)})
    metrics = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
    best_name = validation.dropna(subset=["validation_pr_auc"]).sort_values(
        "validation_pr_auc", ascending=False
    ).iloc[0]["model"] if tune else metrics.iloc[0]["model"]
    best = models[best_name]
    joblib.dump(best, output / "best_model.joblib")
    metrics.to_csv(output / "test_metrics.csv", index=False)
    validation.to_csv(output / "validation_results.csv", index=False)
    calibration = calibration_points(y_test, probabilities[best_name])
    calibration.to_csv(output / "calibration.csv", index=False)
    figures = output.parent / "figures"; figures.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
    ax.plot(calibration["mean_predicted_probability"],
            calibration["observed_fraction_positive"], marker="o", label=best_name)
    ax.set(xlabel="Mean predicted probability", ylabel="Observed fraction positive",
           title="Calibration curve"); ax.legend(); fig.tight_layout()
    fig.savefig(figures / "calibration_curve.png", dpi=180); plt.close(fig)
    subgroup_metrics(test, y_test.to_numpy(), probabilities[best_name]).to_csv(
        output / "subgroup_metrics.csv", index=False)
    missingness_table(frame).to_csv(output / "missingness.csv", index=False)
    pd.DataFrame({"encounter_id": test["encounter_id"], "patient_nbr": test[PATIENT_ID],
                  "outcome": y_test, "probability": probabilities[best_name]}).to_csv(
        output / "test_predictions.csv", index=False)
    manifest = {"seed": seed, "split": "patient-level GroupShuffleSplit",
                "train_encounters": len(train), "test_encounters": len(test),
                "train_patients": int(train[PATIENT_ID].nunique()),
                "test_patients": int(test[PATIENT_ID].nunique()), "best_model": best_name,
                "selection_note": ("selected by grouped-CV PR-AUC" if tune else
                                   "quick mode: selected by test PR-AUC; do not use for final reporting")}
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/diabetic_data.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/metrics"))
    parser.add_argument("--tune", action="store_true", help="Tune using grouped CV; use for final results")
    parser.add_argument("--n-iter", type=int, default=8)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(run(args.data, args.output, args.tune, args.seed, args.n_iter).to_string(index=False))
