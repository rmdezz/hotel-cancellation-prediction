import os
import sys
import yaml
import numpy as np
import pandas as pd
import joblib # Para guardar/cargar el modelo si decides hacerlo (opcional aquí)
import argparse
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

# --- Asegurar que 'src' esté en el path ---
try:
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
except NameError:
    ROOT = os.getcwd() # Si se ejecuta directamente desde la raíz o un notebook en raíz
    if os.path.basename(ROOT) == "src": # Si se ejecuta con python -m src.generate_final_submission
        ROOT = os.path.abspath(os.path.join(ROOT, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
print(f"Raíz del proyecto (ROOT) establecida en: {ROOT}")
# --- Fin configuración de Path ---

from src.full_model_pipeline import make_full_model_pipeline # Asumo que toma dict cfg
from src.utils import load_config

def generate_submission(config_path: str, submission_file_name: str):
    """
    Entrena el modelo final con todos los datos de entrenamiento,
    obtiene el umbral óptimo de las predicciones OOF,
    y genera el archivo de submission para el test set.
    """
    print(f"Cargando configuración desde: {config_path}")
    cfg = load_config(config_path)

    # --- PASO 1: Cargar y Preparar Datos de Entrenamiento ---
    print("Cargando datos de entrenamiento...")
    train_df = pd.read_csv(os.path.join(ROOT, cfg["data_raw_path"], cfg["train_file"]))
    
    mapping = {"Cancelada": 1, "No Cancelada": 0}
    y_full = train_df[cfg["target_column"]].map(mapping)
    if y_full.isna().any():
        raise ValueError("Target con NaNs tras mapping.")
    
    cols_to_drop_from_train = cfg.get("id_columns", []) + [cfg["target_column"]]
    X_full = train_df.drop(columns=[col for col in cols_to_drop_from_train if col in train_df.columns])
    print(f"Datos de entrenamiento preparados: X_full.shape={X_full.shape}")

    # --- PASO 2: Obtener Predicciones OOF y Umbral Óptimo ---
    # (Esto implica ejecutar CV en el conjunto de entrenamiento completo)
    print("Ejecutando Validación Cruzada para obtener OOF y umbral óptimo...")
    
    # --- Ajuste para make_full_model_pipeline: Si espera una RUTA ---
    # pipe = make_full_model_pipeline(config_path) 
    # Si espera un dict:
    pipe_for_oof = make_full_model_pipeline(cfg) # Usar cfg directamente si la función lo permite
    # --- Fin del ajuste ---
    
    oof_probs = np.zeros(len(X_full))
    skf = StratifiedKFold(n_splits=cfg.get("cv_n_splits", 5), 
                          shuffle=True, 
                          random_state=cfg.get("random_seed", 42))

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_full, y_full)):
        print(f"  Procesando Fold {fold+1}/{skf.get_n_splits()} para OOF...")
        X_train_fold, X_val_fold = X_full.iloc[tr_idx], X_full.iloc[va_idx]
        y_train_fold = y_full.iloc[tr_idx]
        
        # Clona el pipeline para asegurar que cada fold se entrena desde cero
        # Aunque si 'pipe_for_oof' se crea nuevo antes del bucle y se re-fitea, está bien.
        # current_fold_pipe = clone(pipe_for_oof) # from sklearn.base import clone
        # current_fold_pipe.fit(X_train_fold, y_train_fold)
        # oof_probs[va_idx] = current_fold_pipe.predict_proba(X_val_fold)[:, 1]
        
        # Si pipe_for_oof se crea una vez y se refitea:
        pipe_for_oof.fit(X_train_fold, y_train_fold)
        oof_probs[va_idx] = pipe_for_oof.predict_proba(X_val_fold)[:, 1]


    # Optimizar umbral sobre OOF completo
    best_f1_weighted = 0.0
    optimal_threshold = 0.5
    for threshold_candidate in np.arange(0.1, 0.9, 0.01):
        oof_preds_at_threshold = (oof_probs >= threshold_candidate).astype(int)
        current_f1_weighted = f1_score(y_full, oof_preds_at_threshold, average="weighted")
        if current_f1_weighted > best_f1_weighted:
            best_f1_weighted = current_f1_weighted
            optimal_threshold = threshold_candidate
    
    print(f"Validación Cruzada OOF completada. Mejor F1 Ponderado OOF: {best_f1_weighted:.6f}")
    print(f"Umbral Óptimo (F1 Ponderado) encontrado: {optimal_threshold:.2f}")

    # --- PASO 3: Entrenar Modelo Final en TODOS los Datos de Entrenamiento ---
    print("Entrenando modelo final en todos los datos de entrenamiento...")
    # --- Ajuste para make_full_model_pipeline: Si espera una RUTA ---
    # final_model_pipe = make_full_model_pipeline(config_path) 
    # Si espera un dict:
    final_model_pipe = make_full_model_pipeline(cfg) # Usar cfg directamente
    # --- Fin del ajuste ---
    final_model_pipe.fit(X_full, y_full)
    print("Modelo final entrenado.")
    
    # Opcional: Guardar el modelo final entrenado
    # model_filename = os.path.join(ROOT, "models", "final_pipeline_trained.joblib")
    # os.makedirs(os.path.dirname(model_filename), exist_ok=True)
    # joblib.dump(final_model_pipe, model_filename)
    # print(f"Modelo final guardado en: {model_filename}")

    # --- PASO 4: Cargar Datos de Test y Predecir ---
    print("Cargando datos de test y generando predicciones...")
    test_df = pd.read_csv(os.path.join(ROOT, cfg["data_raw_path"], cfg.get("test_file", "test_x.csv")))
    
    # Guardar IDs para el archivo de submission
    # Asegúrate que 'Id' (o el ID correcto) está en cfg["id_columns"][0] si es una lista
    submission_id_column = cfg.get("id_columns", ["Id"])[0] # Tomar el primer ID de la lista o default a 'Id'
    if submission_id_column not in test_df.columns:
        raise ValueError(f"La columna ID '{submission_id_column}' no se encuentra en el test set.")
    test_ids = test_df[submission_id_column] 
    
    # Preparar X_test (quitar todos los IDs definidos en el config)
    cols_to_drop_from_test = cfg.get("id_columns", [])
    X_test = test_df.drop(columns=[col for col in cols_to_drop_from_test if col in test_df.columns], errors='ignore')
    
    # El pipeline de scikit-learn se encargará de seleccionar las columnas que necesita
    # y manejar columnas extra en X_test (si remainder='passthrough' o 'drop' está bien configurado)

    test_probs = final_model_pipe.predict_proba(X_test)[:, 1]
    test_predictions_labels = (test_probs >= optimal_threshold).astype(int)
    
    mapping_inv = {1: "Cancelada", 0: "No Cancelada"}
    test_predictions_strings = pd.Series(test_predictions_labels).map(mapping_inv)

    # --- PASO 5: Crear Archivo de Submission ---
    submission_df = pd.DataFrame({
        submission_id_column: test_ids,
        cfg["target_column"]: test_predictions_strings
    })
    
    submission_path = os.path.join(ROOT, submission_file_name) # Guardar en la raíz del proyecto
    submission_df.to_csv(submission_path, index=False)
    print(f"Archivo de submission '{submission_file_name}' guardado en: {submission_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Entrena el modelo final y genera el archivo de submission.")
    parser.add_argument("--config", required=True, help="Ruta al archivo de configuración YAML optimizado.")
    parser.add_argument("--submission_file", default="submission.csv", help="Nombre del archivo de submission a generar.")
    args = parser.parse_args()
    
    # Si make_full_model_pipeline necesita la RUTA y no el diccionario:
    # Debes asegurarte de que config_path (args.config) es lo que espera.
    # Si modificaste make_full_model_pipeline para aceptar el dict, entonces carga el cfg aquí
    # y pásalo. Por ahora, el script asume que la función se adapta a lo que se le pase
    # o que el config_path es suficiente.
    
    generate_submission(args.config, args.submission_file)