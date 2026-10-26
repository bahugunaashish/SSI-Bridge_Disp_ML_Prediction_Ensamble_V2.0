# SSI Bridge Displacement ML Prediction Ensemble V2.0

Separate Streamlit application based on the V1.0 repository. It retains the V1.0 prediction, comparison, train/validation/test overfitting, case-study/SHAP, and new SSI case features, and adds a displacement-range analysis view for the new 15-model run. The V1.0 repository is not changed.

## Models and archive

The app loads the cluster `.tar.gz` export from Google Drive at first startup, extracts it to Streamlit's temporary storage, verifies checkpoint SHA-256 hashes, and builds the runtime manifest. The archive must retain the export directory structure and include:

- `checkpoints/fold_1/` through `checkpoints/fold_5/` with the 12 checkpointed predictor files (`.pt` and `gpr.joblib`)
- `all_folds_all_models_predictions.csv` for train/validation/test plots
- `shapley/shapley_sample_values.csv` for the SHAP case-study feature
- `range_aware_ensemble/` with held-out predictions, range metrics, and GPR prediction-level blend checkpoints

The three GPR prediction-level hybrids use the paired neural/GPR checkpoints plus saved blend weights; they do not have independent neural weight files. Train/validation/test outputs are available for the 12 checkpointed predictors, and held-out cross-fitted results are available for all 15 choices.

## Streamlit Secrets

Reuse the existing `google_service_account_json` and `[google_drive].file_id` values for the input HDF5 database. Add the archive file ID provided for this V2 run:

```toml
[google_drive]
file_id = "<existing input HDF5 file ID>"
model_archive_file_id_V2 = "<uploaded tar.gz file ID>"
new_prediction_parent_folder_id = "<existing V1 shared parent folder ID>"
```

`google_service_account_json` must be set as a top-level Streamlit secret. Give that service account Viewer access to the archive and the existing HDF5/source files.

The new-case builder reuses the V1.0 shared Drive parent folder ID. It discovers the immediate child folders named `EQs` and `soil_profiles`; separate child-folder IDs are not required.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Set the same values in `.streamlit/secrets.toml` for local testing. The application reads the archive and input HDF5; it does not retrain models or modify simulation databases.
