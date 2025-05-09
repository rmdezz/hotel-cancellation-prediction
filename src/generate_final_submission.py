#!/usr/bin/env python3
import os
import sys
import yaml
import pandas as pd

# Asegurar que 'src' esté en el path
try:
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
except NameError:
    ROOT = os.getcwd()
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils import load_config
from src.full_model_pipeline import make_full_model_pipeline

def generate_submission(config_path: str, submission_file_name: str):
    # 1) Carga de configuración
    print(f"Cargando configuración desde: {config_path}")
    cfg = load_config(config_path)

    # 2) Umbral de decisión
    threshold = cfg.get("prediction", {}).get("threshold", 0.5)
    print(f"Usando umbral de decisión = {threshold:.2f}")

    # 3) Carga y prepara los datos de entrenamiento (para reentrenar el modelo final)
    train_df = pd.read_csv(os.path.join(ROOT, cfg["data_raw_path"], cfg["train_file"]))
    mapping = {
        cfg["target_positive_class"]: 1,
        cfg["target_negative_class"]: 0
    }
    y_full = train_df[cfg["target_column"]].map(mapping)
    X_full = train_df.drop(columns=cfg["id_columns"] + [cfg["target_column"]])
    print(f"Datos de train: X_full.shape={X_full.shape}, y_full.shape={y_full.shape}")

    # 4) Entrena el pipeline completo sobre todo el train
    print("Entrenando pipeline final en todo el entrenamiento...")
    final_pipe = make_full_model_pipeline(cfg)
    final_pipe.fit(X_full, y_full)

    # 5) Carga y prepara el test
    print("Cargando datos de test para generar submission...")
    test_df = pd.read_csv(os.path.join(ROOT, cfg["data_raw_path"], cfg.get("test_file", "test_x.csv")))
    # Asumimos que la primera columna de id_columns es la que queremos en la submission
    id_col = cfg["id_columns"][0]
    if id_col not in test_df.columns:
        raise ValueError(f"No encontré la columna ID '{id_col}' en el test set.")
    test_ids = test_df[id_col]
    X_test = test_df.drop(columns=cfg["id_columns"], errors="ignore")

    # 6) Predicciones y aplicación de umbral
    probs = final_pipe.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)
    inv_map = {1: cfg["target_positive_class"], 0: cfg["target_negative_class"]}
    labels = pd.Series(preds).map(inv_map)

    # 7) Guardar submission
    submission = pd.DataFrame({ id_col: test_ids, cfg["target_column"]: labels })
    submission.to_csv(submission_file_name, index=False)
    print(f"Submission guardada en: {submission_file_name}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Genera el CSV de submission final.")
    parser.add_argument("--config", required=True, help="Ruta al YAML de configuración.")
    parser.add_argument("--submission_file", default="submission.csv", help="Nombre de salida.")
    args = parser.parse_args()
    generate_submission(args.config, args.submission_file)
