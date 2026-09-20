# Explainable Machine Learning for 30-Day Hospital Readmission Risk Prediction

## Overview

This repository studies whether information available during a diabetes-related
hospital encounter can predict readmission in fewer than 30 days. It is a
reproducible research project for clinical tabular data, not a deployed clinical
decision-support system.

The outcome is `readmitted == "<30"` (1) versus readmission after 30 days or no
readmission (0). The project compares a majority-class reference, class-weighted
logistic regression, random forest and XGBoost. PR-AUC is the model-selection
metric because the positive class is imbalanced. Evaluation also reports ROC-AUC,
precision, sensitivity, specificity, F1, confusion-matrix counts and Brier score.

## Research questions

1. How well do linear and tree-based models discriminate 30-day readmission?
2. Are predicted probabilities calibrated enough to be interpreted as risks?
3. Does adding missingness indicators change predictive performance?
4. Does performance vary across age, sex and race groups represented in the data?
5. Which recorded variables are associated with model predictions?

## Dataset and attribution

The intended source is the [UCI Diabetes 130-US Hospitals dataset](https://archive.ics.uci.edu/dataset/296/diabetes+130+us+hospitals+for+years+1999+2008),
described by Strack et al. (2014). It contains encounters from 1999–2008. UCI
lists it under CC BY 4.0. See `data/README.md` for acquisition and attribution.
The data are not covered by this repository's MIT code license.

The source repository contained a 5,000-row, 26-column file, which does not match
the documented dimensions of the official UCI data. It can support a software
smoke test, but results from it must not be presented as full-cohort UCI results.

## Leakage controls

- Splitting uses `patient_nbr`, so all encounters for one patient remain in train
  or test, never both.
- Imputers, encoders and scalers are fitted within training folds.
- Grouped cross-validation tunes models without accessing the test set.
- Encounter and patient identifiers are excluded from predictors.
- Class weights are used; no resampling occurs before splitting.

Some fields may reflect discharge workflow. Availability at the intended
prediction time must be defined before prospective use. This is a methodological
study, not a claim that leakage is resolved for every deployment definition.

## Structure

```text
├── data/README.md
├── notebooks/{01_data_exploration,02_preprocessing,03_modeling,04_explainability}.ipynb
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── eda.py
│   ├── train.py
│   ├── evaluate.py
│   ├── missingness_experiment.py
│   └── explain.py
├── tests/test_pipeline.py
└── results/{figures,metrics}/
```

The notebooks are guided inspection layers. Tested modules in `src/` are the
source of truth, avoiding a second copy of the analysis logic.

## Reproducibility

Python 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
python -m src.eda
python -m src.train                         # quick smoke run only
python -m src.train --tune --n-iter 8       # final grouped-CV run
python -m src.missingness_experiment
python -m src.explain --shap                # tree model required for SHAP
```

Seeds, split counts and model-selection mode are recorded in
`results/metrics/run_manifest.json`. Numeric outputs stay in generated CSV files,
instead of being copied into the README as unverifiable approximate results.

## Missing data

The EDA command writes feature-level missing counts. Numeric values use median
imputation; categorical values use most-frequent imputation. A separate experiment
compares that strategy with added missingness indicators. Missingness can encode
care processes and access rather than biology, so any gain needs caution.

## Calibration, explainability and subgroups

Calibration is assessed with quantile-binned calibration points and Brier score.
Global permutation importance works for every model. Optional SHAP exports for a
tree model contain global and one-encounter explanations. They describe model
associations, not causes or treatment effects.

Subgroup outputs cover age, gender and race groups with at least 30 test encounters
and both classes. This is a diagnostic analysis, not a complete fairness audit.

## Limitations

- Retrospective observational data do not establish causation.
- The 1999–2008 cohort is vulnerable to dataset shift.
- No contemporary external validation is included.
- Administrative coding, missingness and label construction may introduce bias.
- Patient grouping prevents overlap but is not temporal or hospital-level validation.
- A clinical threshold requires explicit false-negative and false-positive costs.
- Calibration and subgroup estimates need uncertainty intervals for publication.

## References

- Strack B, et al. *Impact of HbA1c Measurement on Hospital Readmission Rates:
  Analysis of 70,000 Clinical Database Patient Records.* BioMed Research
  International. 2014;2014:781670. https://doi.org/10.1155/2014/781670
- UCI Machine Learning Repository, dataset 296.
