"""Load the V2 cluster archive and perform 15-model checkpoint inference."""
from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import tempfile
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
from google.auth.transport.requests import AuthorizedSession
from google.oauth2 import service_account

from ssi_ml_surrogate_v1.predict_eq_descriptors_first2_v15_gpr import predict

BASE_MODELS = [
    "mlp", "tcn_lstm", "gnn_tcn", "meshgraphnet", "graph_transformer",
    "graph_operator", "hybrid_mlp_meshgraphnet", "hybrid_mlp_graph_transformer",
    "hybrid_tcn_lstm_meshgraphnet", "hybrid_tcn_lstm_graph_transformer",
    "hybrid_gnn_tcn_graph_operator", "gpr",
]
GPR_HYBRIDS = {
    "hybrid_mlp_gpr_predlevel": "mlp",
    "hybrid_meshgraphnet_gpr_predlevel": "meshgraphnet",
    "hybrid_tcn_lstm_meshgraphnet_gpr_predlevel": "hybrid_tcn_lstm_meshgraphnet",
}
_PREDICTION_CACHE = OrderedDict()
_CACHE_LIMIT = 512


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _drive_session(credentials_json: str):
    info = json.loads(credentials_json)
    credentials = service_account.Credentials.from_service_account_info(
        info, scopes=["https://www.googleapis.com/auth/drive.readonly"])
    return AuthorizedSession(credentials)


def _safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, mode="r:*") as tar:
        for member in tar.getmembers():
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError(f"Unsafe path in model archive: {member.name}")
            if member.issym() or member.islnk() or member.isdev():
                raise ValueError(f"Unsupported link/device in model archive: {member.name}")
        tar.extractall(destination)


def _find_data_root(extracted: Path) -> Path:
    candidates = [p.parent for p in extracted.rglob("all_folds_all_models_predictions.csv")]
    for candidate in candidates:
        if (candidate / "checkpoints").is_dir() and (candidate / "range_aware_ensemble").is_dir():
            return candidate
    raise FileNotFoundError(
        "Archive must contain checkpoints/, all_folds_all_models_predictions.csv, "
        "and range_aware_ensemble/.")


def _prediction_manifest(data_root: Path) -> dict:
    fold_dirs = sorted((data_root / "checkpoints").glob("fold_*"),
                       key=lambda p: int(p.name.split("_")[-1]))
    if len(fold_dirs) != 5:
        raise FileNotFoundError(f"Expected five checkpoint folds; found {len(fold_dirs)}")
    records = []
    for fold_dir in fold_dirs:
        for name in BASE_MODELS:
            ext = ".joblib" if name == "gpr" else ".pt"
            path = fold_dir / f"{name}{ext}"
            if not path.is_file():
                raise FileNotFoundError(f"Missing model checkpoint: {path}")
            records.append({"name": name, "fold": fold_dir.name,
                            "file": path.relative_to(data_root).as_posix(),
                            "bytes": path.stat().st_size, "sha256": _digest(path)})
    blend_path = data_root / "range_aware_ensemble" / "gpr_prediction_level_hybrid_final_checkpoint.json"
    if not blend_path.is_file():
        raise FileNotFoundError(f"Missing GPR hybrid blend checkpoint: {blend_path}")
    blend = {"file": blend_path.relative_to(data_root).as_posix(),
             "bytes": blend_path.stat().st_size, "sha256": _digest(blend_path)}
    hybrids = {}
    for name, neural in GPR_HYBRIDS.items():
        hybrids[name] = {"neural_model": neural, "gpr_model": "gpr",
                         "blend_checkpoint": blend}
        for fold_dir in fold_dirs:
            records.append({"name": name, "fold": fold_dir.name,
                            **blend, "components": {"neural_model": neural, "gpr_model": "gpr"}})
    return {"format": "SSI-Bridge-Disp-ML-Prediction-Manifest-v2",
            "base_model_names": BASE_MODELS, "models": records,
            "prediction_level_hybrids": hybrids}


def _write_plot_tables(data_root: Path) -> None:
    """Normalize raw cluster CSVs for the V1 plotting functions, in millimetres."""
    import pandas as pd

    range_csv = data_root / "range_aware_ensemble" / "range_ensemble_predictions.csv"
    raw = pd.read_csv(range_csv)
    output_rows = {}
    for _, row in raw.iterrows():
        key = (str(row["simulation"]), int(row["fold"]))
        record = output_rows.setdefault(key, {"simulation_id": key[0], "fold": f"fold_{key[1]}",
                                                "split": "test"})
        output = str(row["output"])
        record[f"actual_{output}_mm"] = float(row["actual_m"]) * 1000
        for model in BASE_MODELS + list(GPR_HYBRIDS):
            col = f"{model}_predicted_m"
            if col in raw.columns:
                record[f"predicted_{output}_{model}_mm"] = float(row[col]) * 1000
    comparison = []
    for (_, _), record in output_rows.items():
        if not {"actual_Ux_mm", "actual_Uy_mm"}.issubset(record):
            continue
        for model in BASE_MODELS + list(GPR_HYBRIDS):
            ux, uy = record.get(f"predicted_Ux_{model}_mm"), record.get(f"predicted_Uy_{model}_mm")
            if ux is None or uy is None:
                continue
            comparison.append({"simulation_id": record["simulation_id"], "model": model,
                "fold": record["fold"], "split": "test",
                "actual_Ux_mm": record["actual_Ux_mm"], "actual_Uy_mm": record["actual_Uy_mm"],
                "predicted_Ux_mm": ux, "predicted_Uy_mm": uy})
    pd.DataFrame(comparison).to_csv(data_root / "comparison_metrics_V2.csv", index=False)

    # Prefer the prediction-only recovery output when the archive contains it.
    cv_with_train_path = data_root / "all_folds_all_models_predictions_with_train.csv"
    cv_path = (cv_with_train_path if cv_with_train_path.is_file()
               else data_root / "all_folds_all_models_predictions.csv")
    cv = pd.read_csv(cv_path)
    cv = cv[cv["model"].isin(BASE_MODELS)].copy()
    cv_out = pd.DataFrame({
        "simulation_id": cv["simulation"].astype(str), "model": cv["model"],
        "fold": "fold_" + cv["fold"].astype(str).str.replace(r"^fold_", "", regex=True),
        "split": cv["split"].astype(str).str.lower(),
        "actual_Ux_mm": pd.to_numeric(cv["Ux_actual"], errors="coerce") * 1000,
        "actual_Uy_mm": pd.to_numeric(cv["Uy_actual"], errors="coerce") * 1000,
        "predicted_Ux_mm": pd.to_numeric(cv["Ux_predicted"], errors="coerce") * 1000,
        "predicted_Uy_mm": pd.to_numeric(cv["Uy_predicted"], errors="coerce") * 1000,
    })
    cv_out.to_csv(data_root / "overfitting_predictions_V2.csv", index=False)


def prepare_model_bundle(file_id: str, credentials_json: str):
    """Download/extract the cluster tarball once and return (model_root, data_root)."""
    cache = Path(tempfile.gettempdir()) / ("ssi_bridge_v2_" + hashlib.sha256(file_id.encode()).hexdigest()[:16])
    marker = cache / ".ready.json"
    if marker.is_file():
        cached_root = Path(json.loads(marker.read_text(encoding="utf-8"))["data_root"])
        if (cached_root / "manifest.json").is_file():
            return cached_root, cached_root
        marker.unlink(missing_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    archive_path = cache / "cluster_export.tar.gz"
    session = _drive_session(credentials_json)
    response = session.get(f"https://www.googleapis.com/drive/v3/files/{file_id}",
                           params={"alt": "media"}, stream=True, timeout=180)
    response.raise_for_status()
    try:
        with archive_path.open("wb") as target:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    target.write(chunk)
    finally:
        response.close()
    extracted = cache / "export"
    _safe_extract(archive_path, extracted)
    data_root = _find_data_root(extracted)
    manifest = _prediction_manifest(data_root)
    (data_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_plot_tables(data_root)
    (data_root / ".ready").write_text("ready", encoding="utf-8")
    archive_path.unlink(missing_ok=True)
    marker.write_text(json.dumps({"data_root": str(data_root)}), encoding="utf-8")
    return data_root, data_root


def available_models(models_dir):
    root = Path(models_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    result = {}
    for item in manifest["models"]:
        result.setdefault(item["name"], []).append(item)
    for name in result:
        result[name].sort(key=lambda item: int(item["fold"].split("_")[-1]))
    return result


def _verified(root: Path, entry: dict, verify_hash: bool) -> Path:
    path = (root / entry["file"]).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing/unsafe checkpoint: {entry['file']}")
    if verify_hash and _digest(path) != entry["sha256"]:
        raise ValueError(f"Hash mismatch: {entry['file']}")
    return path


def _one_prediction(input_hdf5, case_id, checkpoint, output):
    source, model = Path(input_hdf5).resolve(), Path(checkpoint).resolve()
    ss, ms = source.stat(), model.stat()
    key = (str(source), ss.st_size, ss.st_mtime_ns, str(case_id), str(model), ms.st_size, ms.st_mtime_ns)
    if key in _PREDICTION_CACHE:
        _PREDICTION_CACHE.move_to_end(key)
        y, sigma = _PREDICTION_CACHE[key]
        return y.copy(), None if sigma is None else sigma.copy()
    predict(str(input_hdf5), str(case_id), str(checkpoint), str(output))
    with h5py.File(output, "r") as handle:
        y = np.asarray(handle["predicted_peak_abs_response"][:], dtype=np.float64).reshape(-1)
        sigma = (np.asarray(handle["predictive_std"][:], dtype=np.float64).reshape(-1)
                 if "predictive_std" in handle else None)
    if y.shape != (2,):
        raise ValueError(f"Expected [Ux,Uy], got {y.shape}")
    _PREDICTION_CACHE[key] = (y.copy(), None if sigma is None else sigma.copy())
    if len(_PREDICTION_CACHE) > _CACHE_LIMIT:
        _PREDICTION_CACHE.popitem(last=False)
    return y, sigma


def _blend(neural, gpr, sigma, ux_weight, uy_weight):
    floor = 1e-12
    n, g = np.maximum(neural, floor), np.maximum(gpr, floor)
    effective = np.asarray([ux_weight, uy_weight]) / (1.0 + (np.maximum(sigma, 0) / g) ** 2)
    return np.power(10.0, (1.0 - effective) * np.log10(n) + effective * np.log10(g))


def predict_folds(input_hdf5, case_id, models_dir, model_name, fold=None, verify_hash=True):
    root = Path(models_dir).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    entries = available_models(root).get(model_name, [])
    if fold is not None:
        entries = [entry for entry in entries if entry["fold"] == fold]
    if not entries:
        raise ValueError(f"No checkpoints for model={model_name}, fold={fold}")
    hybrid = manifest.get("prediction_level_hybrids", {}).get(model_name)
    predictions, uncertainties, folds = [], [], []
    with tempfile.TemporaryDirectory(prefix="ssi_prediction_") as tmp:
        for index, entry in enumerate(entries):
            checkpoint = _verified(root, entry, verify_hash)
            if hybrid:
                blend_path = _verified(root, hybrid["blend_checkpoint"], verify_hash)
                blend = json.loads(blend_path.read_text(encoding="utf-8"))
                neural = next(x for x in manifest["models"] if x["name"] == hybrid["neural_model"] and x["fold"] == entry["fold"])
                gpr = next(x for x in manifest["models"] if x["name"] == "gpr" and x["fold"] == entry["fold"])
                n, _ = _one_prediction(input_hdf5, case_id, _verified(root, neural, verify_hash), Path(tmp) / f"n{index}.h5")
                g, sigma = _one_prediction(input_hdf5, case_id, _verified(root, gpr, verify_hash), Path(tmp) / f"g{index}.h5")
                if sigma is None:
                    raise ValueError("GPR checkpoint returned no predictive uncertainty")
                predictions.append(_blend(n, g, sigma,
                    blend["outputs"]["Ux"][model_name], blend["outputs"]["Uy"][model_name]))
                uncertainties.append(sigma)
            else:
                y, sigma = _one_prediction(input_hdf5, case_id, checkpoint, Path(tmp) / f"{index}.h5")
                predictions.append(y)
                if sigma is not None:
                    uncertainties.append(sigma)
            folds.append(entry["fold"])
    values = np.stack(predictions)
    result = {"model": model_name, "folds": folds, "predictions_m": values.tolist(),
              "mean_m": values.mean(axis=0).tolist(),
              "fold_std_m": values.std(axis=0, ddof=1).tolist() if len(values) > 1 else [0.0, 0.0]}
    if uncertainties:
        result["gpr_predictive_std_by_fold_m"] = np.stack(uncertainties).tolist()
    return result
