"""
Credit Risk Dashboard - reproducible model training script.

Run from the Credit_Risk_Streamlit project root:
    python train_model.py

This script:
- removes the CSV row-index column (Unnamed: 0)
- removes records with non-positive age (invalid age values)
- creates train / validation / test splits
- fits imputers and scalers only on training data through pipelines
- trains Logistic Regression, Decision Tree, Random Forest and XGBoost
- fits a Platt-calibrated XGBoost model
- selects operating thresholds on the validation set, not the test set
- saves model pipelines and dashboard outputs
"""

from pathlib import Path
import warnings
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    brier_score_loss
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "cs-training.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET = "SeriousDlqin2yrs"
RANDOM_STATE = 42

# Cost assumptions used in the earlier project discussion:
# false positive = 1 cost unit; false negative = 5 cost units.
# Replace these weights if your final report uses different assumptions.
FP_COST = 1
FN_COST = 5

# -------------------------
# 1. Load and clean dataset
# -------------------------
df = pd.read_csv(DATA_PATH)

# This is only the CSV row index, not a borrower predictor.
df = df.drop(columns=["Unnamed: 0"], errors="ignore")

if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' was not found.")

# Invalid age values are excluded; no other outlier treatment is imposed here.
df = df.loc[df["age"] > 0].copy()

# Ensure target is a binary integer.
df[TARGET] = df[TARGET].astype(int)
X = df.drop(columns=[TARGET])
y = df[TARGET]

feature_names = X.columns.tolist()

# -------------------------
# 2. Train / validation / test split
# -------------------------
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y
)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, test_size=(0.15 / 0.85),
    random_state=RANDOM_STATE, stratify=y_train_full
)

# All features are numeric. Median imputation is learned within each pipeline.
logistic_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
    ))
])

tree_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", DecisionTreeClassifier(
        class_weight="balanced", random_state=RANDOM_STATE,
        min_samples_leaf=10
    ))
])

rf_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestClassifier(
        n_estimators=300, class_weight="balanced_subsample",
        min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE
    ))
])

models = {
    "logistic_regression": logistic_pipeline,
    "decision_tree": tree_pipeline,
    "random_forest": rf_pipeline,
}

# XGBoost is included as a required project model.
try:
    from xgboost import XGBClassifier

    neg, pos = np.bincount(y_train)
    xgb_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=3,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=(neg / max(pos, 1)),
            n_jobs=-1,
            random_state=RANDOM_STATE
        ))
    ])
    models["xgboost"] = xgb_pipeline
except ImportError as exc:
    raise ImportError(
        "XGBoost is not installed. Run: pip install xgboost"
    ) from exc

# -------------------------
# 3. Fit base models
# -------------------------
for name, model in models.items():
    print(f"Training {name} ...")
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_DIR / f"{name}.joblib")

# -------------------------
# 4. Platt calibration
# -------------------------
# Sigmoid calibration is Platt scaling. The estimator includes imputation,
# so preprocessing is repeated correctly inside the calibration CV folds.
try:
    calibrated_xgb = CalibratedClassifierCV(
        estimator=models["xgboost"], method="sigmoid", cv=5
    )
except TypeError:
    # Compatibility with older scikit-learn releases.
    calibrated_xgb = CalibratedClassifierCV(
        base_estimator=models["xgboost"], method="sigmoid", cv=5
    )

print("Training Platt-calibrated XGBoost ...")
calibrated_xgb.fit(X_train, y_train)
joblib.dump(calibrated_xgb, MODEL_DIR / "calibrated_xgboost.joblib")

all_models = {**models, "calibrated_xgboost": calibrated_xgb}

# -------------------------
# 5. Select thresholds on validation set
# -------------------------
threshold_rows = []
selected_thresholds = {}

for name, model in all_models.items():
    valid_prob = model.predict_proba(X_valid)[:, 1]
    best_cost = float("inf")
    best_threshold = 0.50

    for threshold in np.arange(0.05, 0.951, 0.01):
        pred = (valid_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y_valid, pred, labels=[0, 1]
        ).ravel()
        cost = FP_COST * fp + FN_COST * fn

        threshold_rows.append({
            "model": name,
            "threshold": round(float(threshold), 2),
            "TN": int(tn), "FP": int(fp),
            "FN": int(fn), "TP": int(tp),
            "expected_cost_units": int(cost),
            "precision": precision_score(y_valid, pred, zero_division=0),
            "recall": recall_score(y_valid, pred, zero_division=0),
            "f1": f1_score(y_valid, pred, zero_division=0),
        })

        if cost < best_cost:
            best_cost = cost
            best_threshold = float(threshold)

    selected_thresholds[name] = best_threshold

pd.DataFrame(threshold_rows).to_csv(
    OUTPUT_DIR / "threshold_analysis.csv", index=False
)
with open(OUTPUT_DIR / "selected_thresholds.json", "w") as f:
    json.dump(selected_thresholds, f, indent=2)

# -------------------------
# 6. Final test evaluation
# -------------------------
metric_rows = []
prediction_frame = pd.DataFrame(index=X_test.index)
prediction_frame["actual"] = y_test

for name, model in all_models.items():
    prob = model.predict_proba(X_test)[:, 1]
    threshold = selected_thresholds[name]
    pred = (prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, pred, labels=[0, 1]
    ).ravel()

    metric_rows.append({
        "model": name,
        "threshold_selected_on_validation": threshold,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, prob),
        "pr_auc": average_precision_score(y_test, prob),
        "brier_score": brier_score_loss(y_test, prob),
        "TN": int(tn), "FP": int(fp),
        "FN": int(fn), "TP": int(tp),
        "expected_cost_units": int(FP_COST * fp + FN_COST * fn),
    })

    prediction_frame[f"{name}_probability"] = prob
    prediction_frame[f"{name}_prediction"] = pred

pd.DataFrame(metric_rows).to_csv(
    OUTPUT_DIR / "model_metrics.csv", index=False
)
prediction_frame.reset_index(names="source_row_index").to_csv(
    OUTPUT_DIR / "test_predictions.csv", index=False
)

# Store dataset and feature metadata for the Streamlit app.
metadata = {
    "target": TARGET,
    "features": feature_names,
    "rows_after_invalid_age_removal": int(len(df)),
    "class_counts": y.value_counts().sort_index().to_dict(),
    "train_rows": int(len(X_train)),
    "validation_rows": int(len(X_valid)),
    "test_rows": int(len(X_test)),
    "false_positive_cost": FP_COST,
    "false_negative_cost": FN_COST,
}
with open(OUTPUT_DIR / "training_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\nTraining complete.")
print(f"Models saved to: {MODEL_DIR}")
print(f"Dashboard outputs saved to: {OUTPUT_DIR}")
print(pd.DataFrame(metric_rows).round(4).to_string(index=False))
