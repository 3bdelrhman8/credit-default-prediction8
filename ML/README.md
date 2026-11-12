# 💳 Credit Default Prediction

A machine-learning app that estimates the probability that a loan applicant will default,
built with a soft-voting ensemble (Logistic Regression, Random Forest, LightGBM, XGBoost),
SMOTE for class imbalance, and a Streamlit interface.

## Features
- **Single client** form with the most important inputs (the rest is filled with training medians/modes).
- **Batch scoring**: upload a CSV (e.g. `test.csv`) and download predictions.
- Adjustable decision threshold and risk banner.
- Preprocessing shared between the notebook and the app (`src/preprocessing.py`) with unit tests.

## Project structure
```
.
├── streamlit_app.py          # Streamlit UI
├── src/preprocessing.py      # Raw data -> model features
├── models/                   # credit_model.joblib + app_artifacts.joblib (generated)
├── notebooks/
│   ├── Project_nti.ipynb     # EDA + training + validation
│   └── export_cell.py        # cell that exports the model and pins requirements
├── tests/test_preprocessing.py
├── requirements.txt
└── .streamlit/config.toml
```

## 1. Export the model from Colab
1. Train the model in `notebooks/Project_nti.ipynb`.
2. Paste the content of `notebooks/export_cell.py` into a new cell at the end and run it.
3. Download from Colab's file panel:
   - `models/credit_model.joblib`
   - `models/app_artifacts.joblib`
   - `requirements.txt` (pinned to your training versions)
4. Put them in this repo at the same paths (replace the placeholder `requirements.txt`).

> The model must be loaded with the **same scikit-learn / xgboost / lightgbm / imbalanced-learn
> versions** used for training, which is why the export cell pins them.

## 2. Run locally
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## 3. Deploy on Streamlit Community Cloud
1. Push the repo to GitHub (include the `models/*.joblib` files; GitHub's limit is 100 MB per file).
2. Go to <https://share.streamlit.io> and sign in with GitHub.
3. **New app** -> choose the repo, branch `main`, main file `streamlit_app.py`.
4. Under **Advanced settings** pick a Python version close to the one used in Colab
   (run `import sys; print(sys.version)` in Colab to check).
5. Deploy.

## Tests
```bash
pip install pytest
pytest
```

## Disclaimer
Educational project. Do not use as the sole basis for real lending decisions.
