"""Leakage-aware splitting and preprocessing pipelines."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data_loader import PATIENT_ID, TARGET, make_binary_target

RANDOM_STATE = 42
DROP_COLUMNS = {
    TARGET, "encounter_id", PATIENT_ID,
    # Identifiers/codes with little portable clinical meaning or leakage risk.
    "payer_code", "medical_specialty", "weight",
}


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.drop(columns=[c for c in DROP_COLUMNS if c in frame], errors="ignore").copy()
    prior = [c for c in ["number_outpatient", "number_emergency", "number_inpatient"] if c in x]
    if prior:
        x["previous_utilization"] = x[prior].fillna(0).sum(axis=1)
    medication_cols = [
        c for c in ["metformin", "repaglinide", "nateglinide", "chlorpropamide",
                    "glimepiride", "acetohexamide", "glipizide", "glyburide",
                    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
                    "miglitol", "troglitazone", "tolazamide", "examide",
                    "citoglipton", "insulin", "glyburide-metformin",
                    "glipizide-metformin", "glimepiride-pioglitazone",
                    "metformin-rosiglitazone", "metformin-pioglitazone"] if c in x
    ]
    if medication_cols:
        x["active_diabetes_medications"] = x[medication_cols].ne("No").sum(axis=1)
    return x


def patient_level_split(frame: pd.DataFrame, test_size: float = 0.20,
                        random_state: int = RANDOM_STATE):
    """Hold out whole patients; no patient can occur in both partitions."""
    y = make_binary_target(frame)
    groups = frame[PATIENT_ID]
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(frame, y, groups))
    train, test = frame.iloc[train_idx].copy(), frame.iloc[test_idx].copy()
    overlap = set(train[PATIENT_ID]).intersection(test[PATIENT_ID])
    if overlap:
        raise RuntimeError("Patient-level split failed: patient overlap detected")
    return train, test


def build_preprocessor(x: pd.DataFrame, add_missing_indicators: bool = True) -> ColumnTransformer:
    numeric = x.select_dtypes(include=np.number).columns.tolist()
    categorical = x.columns.difference(numeric).tolist()
    num = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=add_missing_indicators)),
        ("scale", StandardScaler()),
    ])
    cat = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent", add_indicator=add_missing_indicators)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5,
                                  sparse_output=True)),
    ])
    return ColumnTransformer([("numeric", num, numeric), ("categorical", cat, categorical)])


def split_xy(frame: pd.DataFrame):
    return engineer_features(frame), make_binary_target(frame)
