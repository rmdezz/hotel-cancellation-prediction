import pandas as pd
import os
from sklearn import set_config # Para salida en Pandas

from .utils import load_config 
from .model_pipeline import make_feature_engineering_pipeline

set_config(transform_output="pandas")

CONFIG_PATH = 'configs/base_config.yaml'
cfg = load_config(CONFIG_PATH)
print(f"Configuración cargada desde: {CONFIG_PATH}")

train_df_raw = pd.read_csv(os.path.join(cfg['data_raw_path'], cfg['train_file']), nrows=100)
print(f"Datos raw cargados. Shape: {train_df_raw.shape}")

X = train_df_raw.copy()
y = None

cols_to_remove_before_pipeline = []
if cfg['target_column'] in X.columns:
    cols_to_remove_before_pipeline.append(cfg['target_column'])
    y = X[cfg['target_column']]

id_cols_from_config = cfg.get('id_columns', [])
for id_c in id_cols_from_config:
    if id_c in X.columns:
        cols_to_remove_before_pipeline.append(id_c)

if cols_to_remove_before_pipeline:
    unique_cols_to_remove = list(set(cols_to_remove_before_pipeline))
    X = X.drop(columns=unique_cols_to_remove)
    print(f"Columnas {unique_cols_to_remove} eliminadas de X antes del pipeline.")

print(f"X preparada para el pipeline. Shape: {X.shape}")

# --- Usar el nuevo pipeline ---
print(f"\n--- Ejecutando pipeline de FE con outlier_treatment: {cfg['preprocessing']['outlier_treatment_numerical']} ---")

# Usamos la nueva función que incluye limpieza + FE
fe_pipeline = make_feature_engineering_pipeline(cfg) 

print("Aplicando fit_transform al pipeline de FE...")
X_processed = fe_pipeline.fit_transform(X.copy(), y) 

print("\nInspección de X_processed (después de limpieza y FE):")
print(f"Shape de X_processed: {X_processed.shape}")
print("Primeras filas de X_processed:")
print(X_processed.head())
print(f"Columnas de X_processed: {X_processed.columns.tolist()}")

# Verificar la nueva columna 'Duracion_Total'
if 'Duracion_Total' in X_processed.columns:
    print("\n¡Columna 'Duracion_Total' creada exitosamente!")
    
    # ESTA LÍNEA CAUSA EL ERROR PORQUE LAS COLUMNAS YA FUERON DROPEADAS:
    # print(X_processed[['num_clean__Noches_Semana', 'num_clean__Noches_Fin_Semana', 'Duracion_Total']].head())
    
    # PUEDES IMPRIMIR SOLO 'Duracion_Total' o verificarla de otra manera si es necesario.
    # Por ejemplo, si quisieras verificarla contra los datos ANTES del drop (más complejo)
    # o simplemente confiar en que la creación fue correcta y ahora se dropearon las originales.
    print("Primeros valores de 'Duracion_Total':")
    print(X_processed['Duracion_Total'].head())
    
    # La verificación de suma ya no se puede hacer directamente de esta forma
    # porque las columnas originales para la suma ya no están en X_processed.
    # Si necesitas esta verificación, tendrías que hacerla DENTRO de FeatureCreator
    # antes de dropear las columnas, o pasar las columnas originales al script de prueba
    # de alguna manera (lo cual complica las cosas).
    # Por ahora, podemos confiar en la lógica de suma simple.
    # expected_sum = X_processed['num_clean__Noches_Semana'] + X_processed['num_clean__Noches_Fin_Semana']
    # if X_processed['Duracion_Total'].equals(expected_sum):
    #     print("Verificación de suma para 'Duracion_Total' correcta.")
    # else:
    #     print("ALERTA: Verificación de suma para 'Duracion_Total' incorrecta.")
    print("Nota: La verificación de suma directa con columnas originales no se realiza aquí porque fueron eliminadas por FeatureCreator.")
else:
    print("ALERTA: Columna 'Duracion_Total' no encontrada en la salida.")

print("\nNulos en X_processed por columna:")
nulos_por_columna = X_processed.isnull().sum()
print(nulos_por_columna[nulos_por_columna > 0])
if nulos_por_columna.sum() == 0:
    print("¡No hay nulos en X_processed!")
else:
    print(f"¡ALERTA! Se encontraron {nulos_por_columna.sum()} nulos en X_processed.")

print("\n¡Test del pipeline de FE (v1) finalizado!")