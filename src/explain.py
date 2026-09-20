"""Model explanations. Associations are not causal effects."""
from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from .data_loader import load_data
from .preprocessing import engineer_features


def permutation_explanation(model, frame: pd.DataFrame, output: Path, seed: int = 42):
    """Model-agnostic global importance on raw features using PR-AUC."""
    from sklearn.inspection import permutation_importance
    from .data_loader import make_binary_target
    x, y = engineer_features(frame), make_binary_target(frame)
    result = permutation_importance(model, x, y, scoring="average_precision",
                                    n_repeats=5, random_state=seed, n_jobs=1)
    table = pd.DataFrame({"feature": x.columns, "importance_mean": result.importances_mean,
                          "importance_sd": result.importances_std}).sort_values(
                              "importance_mean", ascending=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    return table


def shap_explanations(model, frame: pd.DataFrame, output_dir: Path, max_rows: int = 500):
    """Optional SHAP export for a fitted tree pipeline."""
    import shap
    x = engineer_features(frame).iloc[:max_rows]
    pre = model.named_steps["preprocess"]
    estimator = model.named_steps["model"]
    transformed = pre.transform(x)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    names = pre.get_feature_names_out()
    explainer = shap.TreeExplainer(estimator)
    values = explainer(transformed)
    mean_abs = np.abs(values.values).mean(axis=0)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"feature": names, "mean_abs_shap": mean_abs}).sort_values(
        "mean_abs_shap", ascending=False).to_csv(output_dir / "shap_global.csv", index=False)
    first = pd.DataFrame({"feature": names, "shap_value": values.values[0],
                          "feature_value": transformed[0]}).sort_values(
                              "shap_value", key=np.abs, ascending=False)
    first.to_csv(output_dir / "shap_patient_example.csv", index=False)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=Path("results/metrics/best_model.joblib"))
    p.add_argument("--data", type=Path, default=Path("data/diabetic_data.csv"))
    p.add_argument("--output", type=Path, default=Path("results/metrics/permutation_importance.csv"))
    p.add_argument("--shap", action="store_true")
    args = p.parse_args()
    frame = load_data(args.data)
    model = joblib.load(args.model)
    print(permutation_explanation(model, frame, args.output).head(15).to_string(index=False))
    if args.shap:
        shap_explanations(model, frame, args.output.parent)


if __name__ == "__main__":
    main()
