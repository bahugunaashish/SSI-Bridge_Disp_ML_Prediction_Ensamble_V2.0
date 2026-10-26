# SSI Bridge Displacement ML Prediction — Ensemble V2.0

This repository is the separate V2.0 Streamlit application for the updated SSI displacement predictors. The V1.0 repository remains unchanged.

The V2.0 app is being prepared to carry forward the V1.0 interface and analysis features while supporting 15 predictor choices: the 12 fold-checkpointed predictors and three GPR prediction-level hybrids.

## Model and analysis data

Keep the cluster export in a dedicated child folder of the existing shared Google Drive folder. It should include the 12 models' fold checkpoints, the GPR hybrid blend checkpoints, held-out predictions for all 15 choices, train/validation/test predictions for the 12 checkpointed predictors, range metrics, SHAP outputs, training logs, and metadata. Preserve its directory structure.

Do not commit model checkpoint binaries or simulation databases to this repository.

## Deployment configuration

The app will read the model/data folder ID and existing input database IDs from Streamlit Secrets. Reuse the service account already granted Viewer access to the shared Drive folder.

## Current status

The repository was empty when initialized. App source and model-bundle integration are being added separately; this README does not mean deployment is ready yet.
