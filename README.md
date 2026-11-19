# 🏦 Credit Risk Intelligence Dashboard

**An Interactive Machine Learning Platform for Credit Risk Assessment and Default Prediction**

[🚀 View Live Dashboard](## 🚀 Live Demo

[Open the Interactive Credit Risk Dashboard](https://credit-risk-intelligence-dashboard-ruzv35highkgsbytkzzdff.streamlit.app/) | [💻 View Source Code](https://github.com/boseadityabarna-beep/Credit-Risk-Intelligence-Dashboard)

---

A machine learning-based credit risk analytics platform built using Python and Streamlit. The application compares multiple classification models, evaluates default prediction performance, and provides interactive customer-level risk assessment.

**Technologies:** Python | Streamlit | Scikit-learn | XGBoost | Pandas | Plotly
# Credit Risk Streamlit Dashboard

## Project structure
- `data/cs-training.csv`: original training dataset
- `train_model.py`: preprocessing, training, calibration, threshold selection and export
- `models/`: saved fitted model pipelines (created when training completes)
- `outputs/`: metrics, test predictions, threshold analysis and metadata (created when training completes)
- `app.py`: Streamlit dashboard (to be created next)

## Setup
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
python train_model.py
```

The script uses a stratified train/validation/test split. Imputation and scaling are fitted inside model pipelines. The decision threshold is selected on the validation set; final reported metrics are calculated on the held-out test set.

The false-positive and false-negative cost weights are set to 1 and 5 as a configurable example matching the previously discussed cost ratio. Confirm these weights against the final project methodology before presenting expected-cost figures.
