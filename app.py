from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# Page setup and visual theme
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Risk Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --navy:#07182e; --navy2:#102b4c; --blue:#2878c8; --cyan:#35b8c9; }
    .stApp { background: #f4f7fb; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg,#07182e 0%,#102b4c 100%); }
    [data-testid="stSidebar"] * { color: #f4f8ff !important; }
    .hero {
        padding: 1.35rem 1.6rem; border-radius: 16px; color: white;
        background: linear-gradient(110deg,#07182e 0%,#123d68 68%,#176b91 100%);
        margin-bottom: 1rem; box-shadow: 0 8px 24px rgba(7,24,46,.14);
    }
    .hero h1 { margin:0; font-size:2rem; color:white; }
    .hero p { margin:.35rem 0 0; color:#d8e8f8; }
    div[data-testid="stMetric"] {
        background:white; border:1px solid #e4ebf3; padding:14px 16px;
        border-radius:12px; box-shadow:0 3px 12px rgba(18,43,76,.04);
    }
    .section-note { color:#61748b; font-size:.9rem; }
    div.stButton > button, div.stDownloadButton > button {
        background:#123d68; color:white; border:0; border-radius:8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "cs-training.csv"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "Random Forest": "random_forest.joblib",
    "XGBoost": "xgboost.joblib",
    "Calibrated XGBoost": "calibrated_xgboost.joblib",
}


@st.cache_data(show_spinner=False)
def load_data(path_string):
    df = pd.read_csv(path_string)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    if "age" in df.columns:
        df = df.loc[df["age"] > 0].copy()
    return df


@st.cache_data(show_spinner=False)
def load_csv(path_string):
    path = Path(path_string)
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_resource(show_spinner=False)
def load_model(path_string):
    return joblib.load(path_string)


def read_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def metric_value(row, key, default=np.nan):
    try:
        return float(row[key])
    except Exception:
        return default


def get_estimator(model):
    """Return the final estimator inside a sklearn Pipeline when possible."""
    if hasattr(model, "named_steps"):
        try:
            return list(model.named_steps.values())[-1]
        except Exception:
            pass
    return model


def get_feature_importance(model, feature_names):
    """Extract native model importance or absolute logistic coefficients."""
    estimator = get_estimator(model)
    values = None
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_).ravel()
    elif hasattr(estimator, "coef_"):
        values = np.abs(np.asarray(estimator.coef_)).mean(axis=0).ravel()
    if values is None or len(values) != len(feature_names):
        return pd.DataFrame(columns=["Feature", "Importance"])
    result = pd.DataFrame({"Feature": feature_names, "Importance": values})
    return result.sort_values("Importance", ascending=False)


def fmt_pct(value):
    return "—" if pd.isna(value) else f"{value:.1%}"


# ---------------------------------------------------------------------
# Load project artifacts
# ---------------------------------------------------------------------
if not DATA_PATH.exists():
    st.error(
        "Dataset not found. Make sure this app.py is saved in the "
        "Credit_Risk_Streamlit project folder and the dataset is at "
        "data/cs-training.csv."
    )
    st.stop()

data = load_data(str(DATA_PATH))
metrics = load_csv(str(OUTPUTS_DIR / "model_metrics.csv"))
thresholds = read_json(OUTPUTS_DIR / "selected_thresholds.json", {})
metadata = read_json(OUTPUTS_DIR / "training_metadata.json", {})

feature_cols = [
    c for c in data.columns
    if c != "SeriousDlqin2yrs" and pd.api.types.is_numeric_dtype(data[c])
]
target_col = "SeriousDlqin2yrs"
if target_col not in data.columns:
    st.error("The target column 'SeriousDlqin2yrs' is missing from the dataset.")
    st.stop()

available_models = {}
for label, filename in MODEL_FILES.items():
    model_path = MODELS_DIR / filename
    if model_path.exists():
        try:
            available_models[label] = load_model(str(model_path))
        except Exception as exc:
            st.warning(f"Could not load {label}: {exc}")

# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
st.sidebar.markdown("## 🏦 CREDIT RISK")
st.sidebar.caption("INTELLIGENCE & DECISION SUPPORT")
page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Executive Overview",
        "Exploratory Analysis",
        "Data Quality",
        "Model Performance",
        "Threshold & Cost Analysis",
        "Feature Importance",
        "Credit Risk Prediction",
    ],
)
st.sidebar.divider()
st.sidebar.caption("Dataset: Give Me Some Credit")
st.sidebar.caption(f"Records available: {len(data):,}")
st.sidebar.caption("Models are locally loaded from the models/ folder.")

# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <h1>Credit Risk Intelligence Dashboard</h1>
      <p>Portfolio insights · Predictive modelling · Risk-based decision support</p>
    </div>
    """,
    unsafe_allow_html=True,
)

n_records = len(data)
default_rate = data[target_col].mean()
n_defaults = int(data[target_col].sum())
n_nondefaults = int(n_records - n_defaults)

# ---------------------------------------------------------------------
# 1. Executive Overview
# ---------------------------------------------------------------------
if page == "Executive Overview":
    st.subheader("Executive Overview")
    st.markdown(
        '<p class="section-note">A high-level view of portfolio composition and model readiness.</p>',
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total records", f"{n_records:,}")
    c2.metric("Observed defaults", f"{n_defaults:,}")
    c3.metric("Observed default rate", fmt_pct(default_rate))
    c4.metric("Models available", f"{len(available_models)} / 5")

    left, right = st.columns([1.05, 1])
    with left:
        st.markdown("#### Target distribution")
        target_counts = (
            data[target_col].map({0: "No serious delinquency", 1: "Serious delinquency"})
            .value_counts()
            .rename_axis("Outcome")
            .reset_index(name="Customers")
        )
        fig = px.pie(
            target_counts, names="Outcome", values="Customers", hole=.58,
            color_discrete_sequence=["#2878c8", "#35b8c9"]
        )
        fig.update_layout(margin=dict(t=15, b=15, l=5, r=5), legend_title="")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("#### Model snapshot")
        if not metrics.empty:
            display_cols = [
                c for c in ["model", "roc_auc", "pr_auc", "f1", "precision", "recall"]
                if c in metrics.columns
            ]
            st.dataframe(
                metrics[display_cols].sort_values(
                    "roc_auc", ascending=False
                ) if "roc_auc" in display_cols else metrics[display_cols],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Model metrics are not available yet. Run train_model.py first.")

    st.markdown("#### Portfolio profile")
    profile_cols = [c for c in ["age", "MonthlyIncome", "DebtRatio", "RevolvingUtilizationOfUnsecuredLines"] if c in data]
    if profile_cols:
        st.dataframe(data[profile_cols].describe().T.round(2), use_container_width=True)
    st.info(
        "Interpretation: the observed default rate describes this historical dataset. "
        "It should not be interpreted as the current default rate of a live lending portfolio."
    )

# ---------------------------------------------------------------------
# 2. Exploratory Analysis
# ---------------------------------------------------------------------
elif page == "Exploratory Analysis":
    st.subheader("Exploratory Data Analysis")
    st.markdown('<p class="section-note">Explore customer characteristics and how they differ by observed outcome.</p>', unsafe_allow_html=True)
    numeric_features = [c for c in feature_cols if data[c].nunique(dropna=True) > 1]
    selected_feature = st.selectbox("Choose a variable", numeric_features, index=0)
    chart_data = data[[selected_feature, target_col]].dropna().copy()
    chart_data["Outcome"] = chart_data[target_col].map({0: "No serious delinquency", 1: "Serious delinquency"})
    a, b = st.columns(2)
    with a:
        fig = px.histogram(
            chart_data, x=selected_feature, color="Outcome", barmode="overlay",
            histnorm="probability density", opacity=.65,
            color_discrete_sequence=["#2878c8", "#35b8c9"]
        )
        fig.update_layout(margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)
    with b:
        grouped = data.groupby(target_col)[selected_feature].median().reset_index()
        grouped["Outcome"] = grouped[target_col].map({0: "No serious delinquency", 1: "Serious delinquency"})
        fig = px.bar(
            grouped, x="Outcome", y=selected_feature, color="Outcome",
            color_discrete_sequence=["#2878c8", "#35b8c9"]
        )
        fig.update_layout(showlegend=False, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Correlation heatmap")
    corr_cols = feature_cols + [target_col]
    corr = data[corr_cols].corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    fig.update_layout(height=650, margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------
# 3. Data Quality
# ---------------------------------------------------------------------
elif page == "Data Quality":
    st.subheader("Data Quality & Preprocessing")
    missing = data.isna().sum().rename("Missing values").to_frame()
    missing["Missing %"] = (missing["Missing values"] / len(data) * 100).round(2)
    duplicate_count = int(data.duplicated().sum())
    q1, q2, q3 = st.columns(3)
    q1.metric("Rows after age validation", f"{len(data):,}")
    q2.metric("Duplicate rows", f"{duplicate_count:,}")
    q3.metric("Columns", f"{data.shape[1]:,}")
    st.markdown("#### Missing-value assessment")
    st.dataframe(missing.sort_values("Missing values", ascending=False), use_container_width=True)
    fig = px.bar(
        missing.reset_index().rename(columns={"index": "Feature"}),
        x="Feature", y="Missing %", color="Missing %",
        color_continuous_scale=["#35b8c9", "#123d68"]
    )
    fig.update_layout(showlegend=False, xaxis_tickangle=-35, margin=dict(t=20, b=80))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("#### Numeric summary")
    st.dataframe(data[feature_cols].describe().T.round(3), use_container_width=True)
    st.info(
        "The training script removes the row-index column and records with age ≤ 0. "
        "Median imputation is fitted inside each model pipeline using training data, "
        "rather than calculating imputation values from the full dataset."
    )

# ---------------------------------------------------------------------
# 4. Model Performance
# ---------------------------------------------------------------------
elif page == "Model Performance":
    st.subheader("Model Performance Comparison")
    if metrics.empty:
        st.warning("No model_metrics.csv found. Run train_model.py first.")
    else:
        metric_options = [c for c in ["roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1", "brier_score"] if c in metrics]
        chosen_metric = st.selectbox("Select evaluation metric", metric_options, index=0)
        chart = metrics[["model", chosen_metric]].sort_values(chosen_metric, ascending=(chosen_metric == "brier_score"))
        fig = px.bar(
            chart, x="model", y=chosen_metric, color="model",
            color_discrete_sequence=["#123d68", "#2878c8", "#35b8c9", "#5b8fc4", "#79c8d1"],
            text_auto=".3f"
        )
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title=chosen_metric.replace("_", " ").title(), margin=dict(t=25, b=70))
        fig.update_xaxes(tickangle=-20)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("#### Complete test-set metrics")
        st.dataframe(metrics, use_container_width=True, hide_index=True)
        if "roc_auc" in metrics:
            best = metrics.loc[metrics["roc_auc"].idxmax()]
            st.success(f"Highest ROC-AUC in this run: {best['model']} ({best['roc_auc']:.4f}).")
        st.caption("Metrics are based on the held-out test set, using thresholds selected on the validation set.")

# ---------------------------------------------------------------------
# 5. Threshold & Cost Analysis
# ---------------------------------------------------------------------
elif page == "Threshold & Cost Analysis":
    st.subheader("Threshold & Cost Analysis")
    st.markdown(
        "A probability threshold converts a predicted probability into a class decision. "
        "Lower thresholds generally identify more potential defaults, often increasing both true and false positives."
    )
    if not metrics.empty:
        threshold_cols = [c for c in ["model", "threshold_selected_on_validation", "precision", "recall", "f1", "TN", "FP", "FN", "TP", "expected_cost_units"] if c in metrics]
        st.dataframe(metrics[threshold_cols], use_container_width=True, hide_index=True)
        if {"model", "FP", "FN"}.issubset(metrics.columns):
            long = metrics.melt(id_vars="model", value_vars=["FP", "FN"], var_name="Error type", value_name="Count")
            fig = px.bar(long, x="model", y="Count", color="Error type", barmode="group",
                         color_discrete_map={"FP": "#35b8c9", "FN": "#123d68"})
            fig.update_layout(xaxis_title="", margin=dict(t=20, b=70))
            fig.update_xaxes(tickangle=-20)
            st.plotly_chart(fig, use_container_width=True)
        st.info(
            "Expected cost uses the false-positive and false-negative cost weights configured in "
            "train_model.py. Confirm those weights reflect the intended business scenario before "
            "presenting the cost ranking as a lending policy recommendation."
        )
    else:
        st.warning("No threshold analysis is available yet.")

# ---------------------------------------------------------------------
# 6. Feature Importance
# ---------------------------------------------------------------------
elif page == "Feature Importance":
    st.subheader("Feature Importance & Explainability")
    if not available_models:
        st.warning("No saved models found. Run train_model.py first.")
    else:
        selected_model_name = st.selectbox("Choose a model", list(available_models.keys()))
        importance = get_feature_importance(available_models[selected_model_name], feature_cols)
        if importance.empty:
            st.info("Native feature importance is not available for this saved estimator.")
        else:
            fig = px.bar(
                importance.head(15).sort_values("Importance"),
                x="Importance", y="Feature", orientation="h",
                color="Importance", color_continuous_scale=["#35b8c9", "#123d68"]
            )
            fig.update_layout(height=520, margin=dict(t=20, b=20), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(importance, use_container_width=True, hide_index=True)
        st.caption(
            "For tree models, values are the estimator's built-in impurity-based feature importances. "
            "For logistic regression, absolute coefficient magnitudes are shown. These measures are "
            "not causal effects and should be interpreted with care, especially when predictors are correlated."
        )

# ---------------------------------------------------------------------
# 7. Credit Risk Prediction
# ---------------------------------------------------------------------
elif page == "Credit Risk Prediction":
    st.subheader("Individual Credit Risk Assessment")
    if not available_models:
        st.warning("No saved models found. Run train_model.py first.")
        st.stop()

    selected_model_name = st.selectbox("Prediction model", list(available_models.keys()), index=0)
    model = available_models[selected_model_name]
    threshold_default = thresholds.get(selected_model_name.lower().replace(" ", "_"), None)
    if threshold_default is None:
        threshold_default = thresholds.get(selected_model_name, 0.5)
    try:
        threshold_default = float(threshold_default)
    except Exception:
        threshold_default = 0.5

    st.markdown("Enter customer details below. Values are initialized using the dataset median; replace them with the applicant's information.")
    values = {}
    left, right = st.columns(2)
    for i, feature in enumerate(feature_cols):
        series = pd.to_numeric(data[feature], errors="coerce").dropna()
        if series.empty:
            default = 0.0
            low, high = 0.0, 1.0
        else:
            default = float(series.median())
            low = float(series.quantile(.01))
            high = float(series.quantile(.99))
            if not np.isfinite(low) or not np.isfinite(high) or low == high:
                low, high = float(series.min()), float(series.max() if series.max() != series.min() else series.min() + 1)
        col = left if i % 2 == 0 else right
        with col:
            if feature in ["age", "NumberOfOpenCreditLinesAndLoans", "NumberOfTimes90DaysLate",
                           "NumberOfTime30-59DaysPastDueNotWorse", "NumberOfTime60-89DaysPastDueNotWorse",
                           "NumberRealEstateLoansOrLines", "NumberOfDependents"]:
                values[feature] = st.number_input(
                    feature.replace("_", " "), min_value=float(max(0, low)),
                    max_value=float(max(high, max(0, low) + 1)), value=float(np.clip(default, max(0, low), max(high, max(0, low) + 1))),
                    step=1.0
                )
            else:
                values[feature] = st.number_input(
                    feature.replace("_", " "), min_value=float(min(0, low)),
                    max_value=float(max(high, max(0, low) + 1)), value=float(np.clip(default, min(0, low), max(high, max(0, low) + 1))),
                    format="%.4f"
                )

    threshold = st.slider("Decision threshold", 0.01, 0.99, float(np.clip(threshold_default, .01, .99)), .01)
    if st.button("Assess credit risk", type="primary", use_container_width=True):
        input_df = pd.DataFrame([values], columns=feature_cols)
        try:
            probability = float(model.predict_proba(input_df)[0, 1])
            prediction = int(probability >= threshold)
            st.divider()
            p1, p2, p3 = st.columns(3)
            p1.metric("Estimated default probability", f"{probability:.1%}")
            p2.metric("Decision threshold", f"{threshold:.0%}")
            p3.metric("Model decision", "Potential default" if prediction else "No default flag")
            gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=probability * 100,
                number={"suffix": "%"},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#2878c8"},
                       "steps": [{"range": [0, 25], "color": "#d8eef1"},
                                 {"range": [25, 60], "color": "#f6e9c9"},
                                 {"range": [60, 100], "color": "#f2d7d7"}]}
            ))
            gauge.update_layout(height=260, margin=dict(t=25, b=5, l=20, r=20))
            st.plotly_chart(gauge, use_container_width=True)
            st.warning(
                "This is a model-based estimate for demonstration and educational use. "
                "It is not a lending decision, a credit score, or a substitute for a validated "
                "credit policy, affordability assessment, compliance review, or human oversight."
            )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.caption("Check that the saved model was trained with the same feature columns and order used by this dashboard.")

st.sidebar.divider()
st.sidebar.caption("Credit Risk Intelligence · Portfolio analytics prototype")
