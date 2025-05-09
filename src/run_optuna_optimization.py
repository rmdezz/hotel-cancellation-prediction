#!/usr/bin/env python3
import os
import sys
import copy

import numpy as np
import pandas as pd
import optuna
import yaml
import lightgbm as lgb
from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping
from catboost import CatBoostClassifier

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

from optuna.pruners import MedianPruner

# ── 0. Path setup ─────────────────────────────────────────────────
try:
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
except NameError:
    ROOT = os.getcwd()
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.full_model_pipeline import make_full_model_pipeline
from src.utils import load_config

# ── 1. Load data and base config ──────────────────────────────────
BASE_CONFIG_PATH = os.path.join(ROOT, "configs", "base_config.yaml")
base_cfg = load_config(BASE_CONFIG_PATH)

df = pd.read_csv(os.path.join(ROOT, base_cfg["data_raw_path"], base_cfg["train_file"]))
y_full = df[base_cfg["target_column"]].map({"Cancelada": 1, "No Cancelada": 0})
if y_full.isna().any():
    raise ValueError("Target values outside mapping")

X_full = df.drop(columns=base_cfg["id_columns"] + [base_cfg["target_column"]])

# ── 2. Optuna objective with pruning and early stopping ────────────
def objective(trial: optuna.Trial) -> float:
    cfg = copy.deepcopy(base_cfg)

    # A) Numeric outlier treatment
    out_treat = trial.suggest_categorical("outlier_treatment", ["none", "winsor", "log1p"])
    cfg["preprocessing"]["outlier_treatment_numerical"] = out_treat
    if out_treat == "winsor":
        cfg["preprocessing"]["winsor_quantiles_lower"] = trial.suggest_float("winsor_q_low", 0.001, 0.05)
        cfg["preprocessing"]["winsor_quantiles_upper"] = trial.suggest_float("winsor_q_high", 0.95, 0.999)

    # B) Categorical encoding
    enc = trial.suggest_categorical("encoding_method", ["onehot", "target"])
    cfg["encoding"]["method"] = enc
    if enc == "target":
        cfg["encoding"]["target_smoothing"] = trial.suggest_float("target_smoothing", 1.0, 50.0)

    # C) Model hyperparameters
    model_name = cfg["model"]["name"].lower()
    if model_name == "lightgbm":
        fixed = {
            "objective": "binary", "metric": "binary_logloss",
            "random_state": cfg.get("random_seed", 42),
            "n_jobs": -1, "verbosity": -1
        }
        boost = trial.suggest_categorical("boosting_type", ["gbdt", "dart", "goss"])
        params = {
            **fixed,
            "boosting_type": boost,
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
        }
        if boost in ("gbdt", "dart"):
            params["subsample"] = trial.suggest_float("subsample", 0.4, 1.0)
            params["subsample_freq"] = trial.suggest_int("subsample_freq", 0, 10)
        else:
            params["subsample"] = 1.0
            params["subsample_freq"] = 0

    elif model_name == "xgboost":
        params = {
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "early_stopping_rounds": 30,
            "verbosity": 0,
            "n_estimators": trial.suggest_int("xgb_n_estimators", 100, 1000, step=50),
            "learning_rate": trial.suggest_float("xgb_lr", 0.005, 0.1, log=True),
            "max_depth": trial.suggest_int("xgb_max_depth", 3, 12),
            "subsample": trial.suggest_float("xgb_subsample", 0.4, 1.0),
            "colsample_bytree": trial.suggest_float("xgb_colsample", 0.4, 1.0),
            "gamma": trial.suggest_float("xgb_gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("xgb_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("xgb_lambda", 1e-8, 10.0, log=True),
            "n_jobs": -1,
            "random_state": cfg.get("random_seed", 42)
        }

    elif model_name == "catboost":
        params = {
            "iterations": trial.suggest_int("cb_iterations", 100, 1000, step=50),
            "learning_rate": trial.suggest_float("cb_lr", 0.005, 0.1, log=True),
            "depth": trial.suggest_int("cb_depth", 3, 12),
            "l2_leaf_reg": trial.suggest_float("cb_l2", 1e-8, 10.0, log=True),
            "border_count": trial.suggest_int("cb_border", 32, 255),
            "random_strength": trial.suggest_float("cb_rand", 0.0, 10.0),
            "verbose": False, "thread_count": -1,
            "random_seed": cfg.get("random_seed", 42)
        }

    else:
        raise ValueError(f"Unsupported model: {cfg['model']['name']}")

    cfg["model"]["params"] = params

    # D) Build & split pipeline
    full_pipe = make_full_model_pipeline(cfg)
    preproc = Pipeline(full_pipe.steps[:-1])
    model_clf = full_pipe.named_steps["model"]

    # E) K-Fold CV + pruning + early stopping
    oof_probs = np.zeros(len(X_full))
    skf = StratifiedKFold(
        n_splits=cfg.get("cv_n_splits", 5),
        shuffle=True, random_state=cfg.get("random_seed", 42)
    )

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_full, y_full)):
        X_tr, X_va = X_full.iloc[tr_idx], X_full.iloc[va_idx]
        y_tr, y_va = y_full.iloc[tr_idx], y_full.iloc[va_idx]

        X_tr_p = preproc.fit_transform(X_tr, y_tr)
        X_va_p = preproc.transform(X_va)

        # Early stopping callback
        callbacks = [lgb.early_stopping(stopping_rounds=30, verbose=False)] \
                    if model_name == "lightgbm" else None

        fit_kwargs = {}
        if model_name == "lightgbm":
            fit_kwargs = {
                "eval_set": [(X_va_p, y_va)],
                "callbacks": callbacks
            }
        elif model_name == "xgboost":
            fit_kwargs = {
                "eval_set": [(X_va_p, y_va)]
            }
        elif model_name == "catboost":
            fit_kwargs = {
                "eval_set": (X_va_p, y_va),
                "early_stopping_rounds": 30,
                "verbose": False
            }

        model_clf.fit(X_tr_p, y_tr, **fit_kwargs)

        oof_probs[va_idx] = model_clf.predict_proba(X_va_p)[:, 1]
        preds_fold = (oof_probs[va_idx] >= 0.5).astype(int)
        f1_fold = f1_score(y_va, preds_fold, average="weighted")

        trial.report(f1_fold, step=fold)
        if trial.should_prune():
            raise optuna.TrialPruned(f"Pruned at fold {fold} (F1={f1_fold:.4f})")

    # F) Find best threshold
    best_f1 = 0.0
    for thr in np.arange(0.1, 0.9, 0.01):
        preds = (oof_probs >= thr).astype(int)
        f1 = f1_score(y_full, preds, average="weighted")
        if f1 > best_f1:
            best_f1 = f1

    return best_f1

# ── 3. Run optimization ─────────────────────────────────────────────
if __name__ == "__main__":
    STUDY_NAME = "multi_model_opt"
    study = optuna.create_study(
        study_name=STUDY_NAME,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=base_cfg.get("random_seed", 42)),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=2)
    )

    study.optimize(objective, n_trials=100)

    print(f"\nStudy '{STUDY_NAME}' done")
    print(f"Best OOF F1: {study.best_value:.6f}")
    print("Best params:")
    for k,v in study.best_params.items():
        print(f"  {k}: {v}")
