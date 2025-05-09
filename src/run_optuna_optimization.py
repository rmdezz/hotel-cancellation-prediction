#!/usr/bin/env python3
import os
import sys
import copy
import yaml
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

# ── 0. Path setup ─────────────────────────────────────────────────
try:
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
except NameError:
    ROOT = os.getcwd()
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.full_model_pipeline import make_full_model_pipeline
from src.utils import load_config

# ── 1. Carga de datos y configuración base ─────────────────────────
BASE_CONFIG_PATH = os.path.join(ROOT, "configs", "base_config.yaml")
base_cfg = load_config(BASE_CONFIG_PATH)

df = pd.read_csv(os.path.join(ROOT, base_cfg["data_raw_path"], base_cfg["train_file"]))
y_full = df[base_cfg["target_column"]].map({"Cancelada": 1, "No Cancelada": 0})
if y_full.isna().any():
    raise ValueError("Encontrados valores de target fuera del mapping")

X_full = df.drop(columns=base_cfg["id_columns"] + [base_cfg["target_column"]])

# ── 2. Función objetivo de Optuna (sin pruning) ────────────────────
def objective(trial: optuna.Trial, cfg_base: dict, X: pd.DataFrame, y: pd.Series) -> float:
    cfg = copy.deepcopy(cfg_base)

    # A) Outlier / skew
    out_treat = trial.suggest_categorical("outlier_treatment", ["none", "winsor", "log1p"])
    cfg["preprocessing"]["outlier_treatment_numerical"] = out_treat
    if out_treat == "winsor":
        cfg["preprocessing"]["winsor_quantiles_lower"] = trial.suggest_float("winsor_q_low", 0.001, 0.05)
        cfg["preprocessing"]["winsor_quantiles_upper"] = trial.suggest_float("winsor_q_high", 0.95, 0.999)

    # B) Encoding
    enc = trial.suggest_categorical("encoding_method", ["onehot", "target"])
    cfg["encoding"]["method"] = enc
    if enc == "target":
        cfg["encoding"]["target_smoothing"] = trial.suggest_float("target_smoothing", 1.0, 50.0)

    # C) LightGBM hyperparams (sin early stopping)
    fixed = {
        "objective": "binary",
        "metric": "binary_logloss",
        "random_state": cfg.get("random_seed", 42),
        "n_jobs": -1,
        "verbosity": -1,
    }
    boost = trial.suggest_categorical("boosting_type", ["gbdt", "dart", "goss"])
    params = {
        **fixed,
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.5),
        "max_bin": trial.suggest_int("max_bin", 64, 512, step=32),
        "boosting_type": boost,
    }
    if boost in ("gbdt", "dart"):
        params["subsample"] = trial.suggest_float("subsample", 0.4, 1.0)
        params["subsample_freq"] = trial.suggest_int("subsample_freq", 0, 10)
    else:
        params["subsample"] = 1.0
        params["subsample_freq"] = 0

    cfg["model"]["params"] = params

    # D) Pipeline
    temp_cfg_path = os.path.join(ROOT, f"configs/temp_optuna_{trial.number}.yaml")
    with open(temp_cfg_path, "w") as f:
        yaml.dump(cfg, f)
    pipe = make_full_model_pipeline(temp_cfg_path)

    # E) Cross-validation (sin pruning)
    oof_probs = np.zeros(len(X))
    skf = StratifiedKFold(n_splits=cfg.get("cv_n_splits", 5),
                          shuffle=True,
                          random_state=cfg.get("random_seed", 42))

    for tr_idx, va_idx in skf.split(X, y):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        pipe.fit(X_tr, y_tr)
        oof_probs[va_idx] = pipe.predict_proba(X_va)[:, 1]

    # F) Búsqueda de umbral óptimo
    best_f1, best_thr = 0.0, 0.5
    for thr in np.arange(0.1, 0.9, 0.01):
        preds = (oof_probs >= thr).astype(int)
        f1 = f1_score(y, preds, average="weighted")
        if f1 > best_f1:
            best_f1, best_thr = f1, thr

    os.remove(temp_cfg_path)
    return best_f1

# ── 3. Ejecución del estudio ────────────────────────────────────
if __name__ == "__main__":
    STUDY_NAME = "lgbm_full_opt"
    study = optuna.create_study(
        study_name=STUDY_NAME,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=base_cfg.get("random_seed", 42)),
        pruner=None
    )

    N_TRIALS = 100
    study.optimize(lambda t: objective(t, base_cfg, X_full, y_full), n_trials=N_TRIALS)

    # Resultados
    print(f"\nEstudio '{STUDY_NAME}' completado")
    print(f"Mejor F1 OOF: {study.best_value:.6f}")
    print("Mejores parámetros:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
