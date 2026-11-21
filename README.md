# Flaking PoC - bearing tap-test FRF

Proof-of-concept detector: **Flaking vs Not flaking** (Not flaking = Other + Good), one model per brand
(NBC, TIMKEN). The model input is **only** the FRF CSV curve (0-561.25 Hz, 450 points, 1.25 Hz step).
The Word report is shown in the app for information only and is never used by the model.

Original data is read-only. Everything here lives in `_analysis/flaking_poc/`.

## Files

| File | What it is |
|---|---|
| `features.py` | CSV/docx loading, feature sets, model wrapper, zone explanation |
| `train_poc.py` | Builds the data set, runs grouped CV, trains final models, checks the hold-out |
| `app.py` | Streamlit demo |
| `models/NBC.joblib`, `models/TIMKEN.joblib` | Final models (one per brand) |
| `tables/` | CV results, out-of-fold predictions, importance, hold-out results, data used |
| `plots/` | Confusion matrices |
| `POC_REPORT.md` | Results in plain English |

## Needed packages

```powershell
pip install numpy pandas scipy scikit-learn matplotlib joblib python-docx streamlit
```

## Rebuild the models (about 10 minutes)

```powershell
cd "E:\Projects\AIML DATA - SET 1\AIML\_analysis\flaking_poc"
```

```powershell
python train_poc.py
```

## Run the demo app

```powershell
cd "E:\Projects\AIML DATA - SET 1\AIML\_analysis\flaking_poc"
```

```powershell
streamlit run app.py
```

The app opens at http://localhost:8501. Then:

1. Pick the brand in the sidebar.
2. Upload one FRF CSV (BOM, empty trailing column and `SIG####(1)` headers are handled; files that are
   not 450 points on the 0-561.25 Hz grid are refused with a clear message).
3. Optionally upload the matching Word report to show bearing no., SIG, MFD and observation.

The result is FLAKING / NOT FLAKING with a confidence %, or **UNSURE** when the flaking probability is
between 35% and 65%. The app also shows the curve against the median Flaking / Not-flaking curves
(10-90% bands), the 5 zones that moved the result most, and the 3 most similar training bearings.

## Demo files (hold-out - never used for training)

Use only bearings from `tables/holdout_demo_set.csv` (in `_analysis/tables/`) for the demo: the models
never saw them. `tables/holdout_results.csv` here lists each hold-out bearing with the true label, the
predicted label and the confidence, so you can pick a few clear hits and, to be honest, at least one miss.
See the hold-out table in `POC_REPORT.md`.
