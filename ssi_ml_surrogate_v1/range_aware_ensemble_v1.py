"""Cross-fitted range-aware log-space ensemble over the trained predictors.

This is a post-training analysis stage. It reads the held-out prediction CSV
written by cross_validate_eq_descriptors_first2_v6_gpr_mape_fix and writes all
outputs to a separate directory. It never opens or changes either HDF5 database.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

MODEL_NAMES = ["mlp", "tcn_lstm", "gnn_tcn", "meshgraphnet",
               "graph_transformer", "graph_operator",
               "hybrid_mlp_meshgraphnet", "hybrid_mlp_graph_transformer",
               "hybrid_tcn_lstm_meshgraphnet", "hybrid_tcn_lstm_graph_transformer",
               "hybrid_gnn_tcn_graph_operator", "gpr"]
GPR_PREDICTION_HYBRIDS = {
    "hybrid_mlp_gpr_predlevel": "mlp",
    "hybrid_meshgraphnet_gpr_predlevel": "meshgraphnet",
    "hybrid_tcn_lstm_meshgraphnet_gpr_predlevel": "hybrid_tcn_lstm_meshgraphnet",
}
OUTPUTS = ["Ux", "Uy"]
PREDICTION_FLOOR_M = 1e-12
MIN_BAND_CASES = 20


def _fit_weights(pred_log: np.ndarray, actual_log: np.ndarray) -> np.ndarray:
    """Fit nonnegative weights summing to one against log displacement."""
    n_models = pred_log.shape[1]
    uniform = np.full(n_models, 1.0 / n_models)
    if len(actual_log) < max(8, 2 * n_models):
        return uniform

    def objective(weights):
        residual = pred_log @ weights - actual_log
        return float(np.mean(residual ** 2) + 1e-4 * np.sum((weights - uniform) ** 2))

    fit = minimize(objective, uniform, method="SLSQP", bounds=[(0.0, 1.0)] * n_models,
                   constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
                   options={"maxiter": 500, "ftol": 1e-10})
    if not fit.success or not np.isfinite(fit.x).all():
        return uniform
    weights = np.clip(fit.x, 0.0, 1.0)
    return weights / weights.sum()


def _fit_experts(train: pd.DataFrame, output: str):
    pred_cols = [f"{output}_{name}_log" for name in MODEL_NAMES]
    ycol = f"{output}_actual"
    usable = train[np.isfinite(train[ycol]) & (train[ycol] > 0)].copy()
    if len(usable) < max(24, 2 * len(MODEL_NAMES)):
        raise ValueError(f"Too few positive cross-fitted cases to train the {output} ensemble: {len(usable)}")
    ylog = np.log10(usable[ycol].to_numpy(dtype=float))
    xlog = usable[pred_cols].to_numpy(dtype=float)
    global_weights = _fit_weights(xlog, ylog)
    edges = np.quantile(ylog, [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
    # Expand tied boundaries slightly; small target sets may have repeated values.
    centers = np.quantile(ylog, [1.0 / 6.0, 0.5, 5.0 / 6.0])
    scale = max(float(np.median(np.abs(ylog - np.median(ylog)))), 0.35)
    experts, counts = [], []
    band_ids = np.digitize(ylog, edges[1:-1], right=True)
    for band in range(3):
        selected = band_ids == band
        counts.append(int(selected.sum()))
        if selected.sum() < MIN_BAND_CASES:
            experts.append(global_weights.copy())
        else:
            experts.append(_fit_weights(xlog[selected], ylog[selected]))
    return {"global_weights": global_weights, "expert_weights": np.asarray(experts),
            "edges_log10_m": edges.tolist(), "centers_log10_m": centers.tolist(),
            "gate_scale_log10": scale, "band_fit_counts": counts, "fit_cases": len(usable)}


def _gate_weights(pred_log: np.ndarray, config: dict) -> np.ndarray:
    centers = np.asarray(config["centers_log10_m"], dtype=float)
    scale = float(config["gate_scale_log10"])
    preliminary = pred_log @ np.asarray(config["global_weights"], dtype=float)
    logits = -0.5 * ((preliminary[:, None] - centers[None, :]) / scale) ** 2
    logits -= logits.max(axis=1, keepdims=True)
    gate = np.exp(logits)
    return gate / gate.sum(axis=1, keepdims=True)


def predict_ensemble(predictions_m: np.ndarray, config: dict,
                     gpr_std_m: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combine positive predictions in log10 space, using GPR uncertainty as a reliability cue."""
    predictions_m = np.asarray(predictions_m, dtype=float)
    logp = np.log10(np.maximum(predictions_m, PREDICTION_FLOOR_M))
    gate = _gate_weights(logp, config)
    expert_w = np.asarray(config["expert_weights"], dtype=float)
    gpr_index = MODEL_NAMES.index("gpr")
    non_gpr = [i for i in range(len(MODEL_NAMES)) if i != gpr_index]
    if gpr_std_m is None:
        certainty = np.ones(len(logp), dtype=float)
    else:
        gpr_std_m = np.asarray(gpr_std_m, dtype=float).reshape(-1)
        valid_sigma = np.isfinite(gpr_std_m) & (gpr_std_m >= 0)
        relative_sigma = np.zeros(len(logp), dtype=float)
        relative_sigma[valid_sigma] = gpr_std_m[valid_sigma] / np.maximum(predictions_m[valid_sigma, gpr_index], PREDICTION_FLOOR_M)
        certainty = np.ones(len(logp), dtype=float)
        certainty[valid_sigma] = 1.0 / (1.0 + relative_sigma[valid_sigma] ** 2)
    per_expert = np.zeros((len(logp), 3), dtype=float)
    effective_gpr_weight = np.zeros(len(logp), dtype=float)
    for expert_i in range(3):
        base_w = expert_w[expert_i]
        adapted_gpr = base_w[gpr_index] * certainty
        neural_mass = np.maximum(0.0, 1.0 - base_w[gpr_index])
        adjusted = np.zeros_like(logp)
        adjusted[:, gpr_index] = adapted_gpr
        if neural_mass > 1e-12:
            adjusted[:, non_gpr] = (1.0 - adapted_gpr[:, None]) * (base_w[non_gpr] / neural_mass)[None, :]
        else:
            adjusted[:, non_gpr] = (1.0 - adapted_gpr[:, None]) / len(non_gpr)
        per_expert[:, expert_i] = np.sum(logp * adjusted, axis=1)
        effective_gpr_weight += gate[:, expert_i] * adjusted[:, gpr_index]
    final_log = np.sum(gate * per_expert, axis=1)
    return np.power(10.0, final_log), gate, effective_gpr_weight


def _predict_gpr_hybrid(neural_m, gpr_m, gpr_std_m, gpr_weight):
    """Blend one neural prediction with GPR in log10 space, downweighting uncertain GPR."""
    neural_m = np.maximum(np.asarray(neural_m, dtype=float), PREDICTION_FLOOR_M)
    gpr_m = np.maximum(np.asarray(gpr_m, dtype=float), PREDICTION_FLOOR_M)
    sigma = np.asarray(gpr_std_m, dtype=float)
    relative = np.zeros_like(gpr_m)
    valid = np.isfinite(sigma) & (sigma >= 0)
    relative[valid] = sigma[valid] / gpr_m[valid]
    effective_gpr_weight = float(gpr_weight) / (1.0 + relative ** 2)
    log_pred = (1.0 - effective_gpr_weight) * np.log10(neural_m) + effective_gpr_weight * np.log10(gpr_m)
    return np.power(10.0, log_pred), effective_gpr_weight


def _fit_gpr_hybrid_weight(train: pd.DataFrame, output: str, neural_name: str) -> float:
    """Fit the GPR share on other folds' held-out predictions only."""
    y = train[f"{output}_actual"].to_numpy(dtype=float)
    n = train[f"{output}_{neural_name}_m"].to_numpy(dtype=float)
    g = train[f"{output}_gpr_m"].to_numpy(dtype=float)
    s = train[f"{output}_gpr_std_m"].to_numpy(dtype=float)
    valid = np.isfinite(y) & (y > 0) & np.isfinite(n) & np.isfinite(g)
    if not valid.any():
        return 0.5
    # Candidate grid is applied with uncertainty adjustment for each case.
    yg = np.log10(np.maximum(y[valid], PREDICTION_FLOOR_M))
    ng = np.log10(np.maximum(n[valid], PREDICTION_FLOOR_M))
    gg = np.log10(np.maximum(g[valid], PREDICTION_FLOOR_M))
    sigma = s[valid]
    rel = np.zeros_like(g[valid]); good = np.isfinite(sigma) & (sigma >= 0)
    rel[good] = sigma[good] / np.maximum(g[valid][good], PREDICTION_FLOOR_M)
    certainty = 1.0 / (1.0 + rel ** 2)
    candidates = np.linspace(0.0, 1.0, 101)
    losses = [np.mean(((1.0 - w * certainty) * ng + (w * certainty) * gg - yg) ** 2)
              + 1e-4 * (w - 0.5) ** 2 for w in candidates]
    return float(candidates[int(np.argmin(losses))])


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    good = np.isfinite(y) & np.isfinite(p) & (y > 0) & (p >= 0)
    y, p = y[good], p[good]
    if not len(y):
        return {"n": 0, "mae_m": "", "rmse_m": "", "mae_log10": "",
                "rmse_log10": "", "median_ape_percent": "", "mean_ape_percent": ""}
    floor = PREDICTION_FLOOR_M
    dlog = np.log10(np.maximum(p, floor)) - np.log10(np.maximum(y, floor))
    ape = np.abs(p - y) / np.maximum(y, floor) * 100.0
    return {"n": int(len(y)), "mae_m": float(np.mean(np.abs(p - y))),
            "rmse_m": float(np.sqrt(np.mean((p - y) ** 2))),
            "mae_log10": float(np.mean(np.abs(dlog))),
            "rmse_log10": float(np.sqrt(np.mean(dlog ** 2))),
            "median_ape_percent": float(np.median(ape)),
            "mean_ape_percent": float(np.mean(ape))}


def _wide_test_rows(source: Path) -> pd.DataFrame:
    raw = pd.read_csv(source)
    required = {"fold", "split", "model", "simulation"}
    if not required.issubset(raw.columns):
        raise ValueError(f"Prediction CSV missing required columns: {sorted(required - set(raw.columns))}")
    raw = raw[raw["split"].astype(str).str.lower() == "test"].copy()
    if raw.empty:
        raise ValueError("Prediction CSV contains no split=test rows")
    rows = []
    for (fold, sim), group in raw.groupby(["fold", "simulation"], sort=True):
        by_model = group.drop_duplicates("model").set_index("model")
        if not set(MODEL_NAMES).issubset(by_model.index):
            continue
        record = {"fold": int(fold), "simulation": str(sim)}
        for output in OUTPUTS:
            actual_col = f"{output}_actual"
            values = by_model[actual_col].dropna().unique()
            if not len(values):
                record[actual_col] = np.nan
            elif not np.allclose(values, values[0], rtol=1e-6, atol=1e-12):
                raise ValueError(f"Actual {output} differs across models for fold={fold}, simulation={sim}")
            else:
                record[actual_col] = float(values[0])
            for model in MODEL_NAMES:
                value = float(by_model.loc[model, f"{output}_predicted"])
                record[f"{output}_{model}_log"] = np.log10(max(value, PREDICTION_FLOOR_M))
                record[f"{output}_{model}_m"] = value
            sigma_col=f"{output}_predictive_std_m"
            sigma_value=by_model.loc["gpr", sigma_col] if sigma_col in by_model.columns else np.nan
            record[f"{output}_gpr_std_m"]=float(sigma_value) if pd.notna(sigma_value) else np.nan
        rows.append(record)
    wide = pd.DataFrame(rows)
    if wide.empty:
        raise ValueError(f"No test simulations have predictions from all {len(MODEL_NAMES)} required predictors")
    return wide


def _range_label(y: float) -> str:
    return f"10^{int(np.floor(np.log10(y)))} to <10^{int(np.floor(np.log10(y))) + 1} m"


def run(predictions_csv: Path, output_dir: Path):
    wide = _wide_test_rows(predictions_csv)
    folds = [int(value) for value in sorted(wide["fold"].unique())]
    out = output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    output_rows, weight_rows, metric_rows = [], [], []
    gpr_hybrid_fold_checkpoints = {}

    # Fold k is evaluated with meta-weights trained only from OOF predictions
    # belonging to the other folds. This keeps the reported test metrics honest.
    for output in OUTPUTS:
        pred_cols = [f"{output}_{name}_log" for name in MODEL_NAMES]
        for fold in folds:
            test = wide[wide.fold == fold].copy()
            train = wide[wide.fold != fold].copy()
            config = _fit_experts(train, output)
            gpr_hybrid_weights = {
                name: _fit_gpr_hybrid_weight(train, output, neural_name)
                for name, neural_name in GPR_PREDICTION_HYBRIDS.items()
            }
            gpr_hybrid_fold_checkpoints.setdefault(str(fold), {})[output] = gpr_hybrid_weights
            model_predictions = test[[f"{output}_{name}_m" for name in MODEL_NAMES]].to_numpy(float)
            gpr_std = test[f"{output}_gpr_std_m"].to_numpy(float)
            ens_pred, gate, effective_gpr = predict_ensemble(model_predictions, config, gpr_std)
            global_log = test[pred_cols].to_numpy(float) @ config["global_weights"]
            uniform_log = test[pred_cols].to_numpy(float).mean(axis=1)
            actual = test[f"{output}_actual"].to_numpy(float)
            gpr_hybrid_predictions = {}
            gpr_hybrid_effective_weights = {}
            for name, neural_name in GPR_PREDICTION_HYBRIDS.items():
                hybrid_pred, effective = _predict_gpr_hybrid(
                    test[f"{output}_{neural_name}_m"].to_numpy(float),
                    test[f"{output}_gpr_m"].to_numpy(float), gpr_std,
                    gpr_hybrid_weights[name])
                gpr_hybrid_predictions[name] = hybrid_pred
                gpr_hybrid_effective_weights[name] = effective
            for i, (_, source_row) in enumerate(test.iterrows()):
                row = {"fold": fold, "simulation": source_row.simulation, "output": output,
                       "actual_m": actual[i], "ensemble_predicted_m": ens_pred[i],
                       "global_log_ensemble_predicted_m": 10.0 ** global_log[i],
                       "uniform_log_ensemble_predicted_m": 10.0 ** uniform_log[i],
                       "gate_low": gate[i, 0], "gate_mid": gate[i, 1], "gate_high": gate[i, 2],
                       "gpr_predictive_std_m": gpr_std[i],
                       "effective_gpr_weight_after_uncertainty": effective_gpr[i],
                       "actual_range": _range_label(actual[i]) if actual[i] > 0 else "zero/nonpositive"}
                for j, model in enumerate(MODEL_NAMES):
                    row[f"{model}_predicted_m"] = model_predictions[i, j]
                for model, pred in gpr_hybrid_predictions.items():
                    row[f"{model}_predicted_m"] = pred[i]
                    row[f"{model}_effective_gpr_weight"] = gpr_hybrid_effective_weights[model][i]
                output_rows.append(row)
            for expert_i, expert in enumerate(("low", "mid", "high")):
                for model_i, model in enumerate(MODEL_NAMES):
                    weight_rows.append({"fold": fold, "output": output, "weights_fit_on": "other_folds_oof",
                                        "expert": expert, "model": model,
                                        "weight": float(config["expert_weights"][expert_i, model_i]),
                                        "fit_cases_in_target_band": config["band_fit_counts"][expert_i]})
            for model, weight in gpr_hybrid_weights.items():
                weight_rows.append({"fold": fold, "output": output, "weights_fit_on": "other_folds_oof",
                                    "expert": "global_gpr_neural_blend", "model": model,
                                    "weight": weight, "fit_cases_in_target_band": int(len(train))})
            for model in MODEL_NAMES:
                pred = test[f"{output}_{model}_m"].to_numpy(float)
                metric_rows.append({"fold": fold, "output": output, "model": model,
                                    "range": "all", **_metrics(actual, pred)})
            for model, pred in gpr_hybrid_predictions.items():
                metric_rows.append({"fold": fold, "output": output, "model": model,
                                    "range": "all", **_metrics(actual, pred)})
            metric_rows.append({"fold": fold, "output": output, "model": "uniform_log_ensemble",
                                "range": "all", **_metrics(actual, 10.0 ** uniform_log)})
            metric_rows.append({"fold": fold, "output": output, "model": "global_log_ensemble",
                                "range": "all", **_metrics(actual, 10.0 ** global_log)})
            metric_rows.append({"fold": fold, "output": output, "model": "range_aware_ensemble",
                                "range": "all", **_metrics(actual, ens_pred)})
            for lo_exp in range(-12, 1):
                lower, upper = 10.0 ** lo_exp, 10.0 ** (lo_exp + 1)
                selected = (actual >= lower) & (actual < upper)
                if not selected.any():
                    continue
                label = f"10^{lo_exp} to <10^{lo_exp + 1} m"
                for model in MODEL_NAMES:
                    metric_rows.append({"fold": fold, "output": output, "model": model, "range": label,
                                        **_metrics(actual[selected], test[f"{output}_{model}_m"].to_numpy(float)[selected])})
                for model, pred in gpr_hybrid_predictions.items():
                    metric_rows.append({"fold": fold, "output": output, "model": model, "range": label,
                                        **_metrics(actual[selected], pred[selected])})
                metric_rows.append({"fold": fold, "output": output, "model": "range_aware_ensemble",
                                    "range": label, **_metrics(actual[selected], ens_pred[selected])})

    predictions = pd.DataFrame(output_rows)
    predictions.to_csv(out / "range_ensemble_predictions.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(out / "range_ensemble_fold_weights.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(out / "range_ensemble_metrics.csv", index=False)
    checkpoint_dir = out / "gpr_prediction_level_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for fold, outputs in gpr_hybrid_fold_checkpoints.items():
        checkpoint = {"format": "SSI-ML-GPR-Prediction-Hybrid-v1", "fold": int(fold),
                      "hybrids": GPR_PREDICTION_HYBRIDS,
                      "fit_data": "other folds' held-out predictions only",
                      "blend_space": "log10 metres",
                      "uncertainty_rule": "effective GPR weight = fitted GPR weight / (1 + (GPR predictive std / GPR mean)^2)",
                      "outputs": outputs}
        (checkpoint_dir / f"fold_{fold}.json").write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")

    # Production meta-weights from all available OOF predictions. These weights
    # are stored with full provenance; fold-specific metrics above remain cross-fit.
    final = {"format": "SSI-ML-RangeAwareLogEnsemble-v2",
             "base_models": MODEL_NAMES, "prediction_floor_m": PREDICTION_FLOOR_M,
             "gpr_prediction_level_hybrids": GPR_PREDICTION_HYBRIDS,
             "training_source": str(predictions_csv.resolve()),
             "fit_data": "all held-out test predictions; one OOF prediction per simulation and model",
             "outputs": {}}
    for output in OUTPUTS:
        final["outputs"][output] = _fit_experts(wide, output)
        final["outputs"][output]["gpr_neural_hybrid_gpr_weights"] = {
            name: _fit_gpr_hybrid_weight(wide, output, neural_name)
            for name, neural_name in GPR_PREDICTION_HYBRIDS.items()
        }
        final["outputs"][output]["expert_weights"] = final["outputs"][output]["expert_weights"].tolist()
        final["outputs"][output]["global_weights"] = final["outputs"][output]["global_weights"].tolist()
        for band, weights in zip(("low", "mid", "high"), final["outputs"][output]["expert_weights"]):
            for model, weight in zip(MODEL_NAMES, weights):
                weight_rows.append({"fold": "all_oof_final", "output": output, "weights_fit_on": "all_oof",
                                    "expert": band, "model": model, "weight": float(weight),
                                    "fit_cases_in_target_band": final["outputs"][output]["band_fit_counts"][("low", "mid", "high").index(band)]})
    (out / "range_ensemble_final_weights.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    final_gpr_checkpoint = {
        "format": "SSI-ML-GPR-Prediction-Hybrid-v1", "fold": "all_oof_final",
        "hybrids": GPR_PREDICTION_HYBRIDS,
        "fit_data": "all available held-out predictions; one out-of-fold prediction per simulation",
        "blend_space": "log10 metres",
        "uncertainty_rule": "effective GPR weight = fitted GPR weight / (1 + (GPR predictive std / GPR mean)^2)",
        "outputs": {output: final["outputs"][output]["gpr_neural_hybrid_gpr_weights"] for output in OUTPUTS},
    }
    (out / "gpr_prediction_level_hybrid_final_checkpoint.json").write_text(
        json.dumps(final_gpr_checkpoint, indent=2), encoding="utf-8")
    pd.DataFrame(weight_rows).to_csv(out / "range_ensemble_fold_weights.csv", index=False)
    report = {
        "purpose": "Cross-fitted evaluation of 15 predictors: seven originals, five hidden-feature neural hybrids, and three GPR prediction-level hybrids.",
        "database_access": "None. The analyzer reads only the existing held-out prediction CSV.",
        "database_writes": "None. All files are written under this separate output directory.",
        "held_out_simulations": int(wide.simulation.nunique()), "folds": folds,
        "prediction_floor_m": PREDICTION_FLOOR_M,
        "ensemble": "The range-aware ensemble uses three log-space experts with simplex weights. Three named GPR-neural hybrids also blend paired predictions in log space; their GPR shares are fit on other folds' held-out predictions and reduced when relative GPR uncertainty is high.",
        "cross_validation": "Each fold's stacking weights are fit only from the other folds' held-out predictions.",
        "metrics": ["MAE/RMSE in metres", "MAE/RMSE in log10 metres", "median and mean absolute percentage error"],
        "files": ["range_ensemble_predictions.csv", "range_ensemble_metrics.csv",
                  "range_ensemble_fold_weights.csv", "range_ensemble_final_weights.json",
                  "gpr_prediction_level_checkpoints/fold_N.json",
                  "gpr_prediction_level_hybrid_final_checkpoint.json"],
        "limitations": ["Requires successful twelve-base-predictor CV output.",
                        "Final weights combine predictions from the trained models; this does not retrain those models.",
                        "Zero actual targets are excluded from log metrics and counted only if present in raw prediction rows."],
    }
    (out / "range_ensemble_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Range-aware ensemble reports written to {out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True,
                        help="all_folds_all_models_predictions.csv from the 12-checkpointed-predictor CV run")
    parser.add_argument("--output", required=True, help="Separate output directory for ensemble reports")
    args = parser.parse_args()
    run(Path(args.predictions), Path(args.output))


if __name__ == "__main__":
    main()
