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
