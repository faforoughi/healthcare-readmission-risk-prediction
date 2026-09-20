"""Compare simple imputation with imputation plus missingness indicators."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from .data_loader import load_data
from .evaluate import classification_metrics
from .preprocessing import build_preprocessor, patient_level_split, split_xy

def run(data: Path, output: Path, seed: int = 42) -> pd.DataFrame:
    train, test = patient_level_split(load_data(data), random_state=seed)
    x_train, y_train = split_xy(train); x_test, y_test = split_xy(test)
    rows = []
    for indicators in (False, True):
        model = Pipeline([("preprocess", build_preprocessor(x_train, indicators)),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed))])
        model.fit(x_train, y_train)
        rows.append({"strategy": "imputation_plus_indicators" if indicators else "simple_imputation",
                     **classification_metrics(y_test, model.predict_proba(x_test)[:, 1])})
    result = pd.DataFrame(rows); output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False); return result

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--data", type=Path, default=Path("data/diabetic_data.csv"))
    p.add_argument("--output", type=Path, default=Path("results/metrics/missingness_experiment.csv"))
    a = p.parse_args(); print(run(a.data, a.output).to_string(index=False))
