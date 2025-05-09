import yaml
import pandas as pd
import os
from dotenv import load_dotenv # Para cargar variables de entorno si las usas

# Cargar variables de entorno (si tienes un .env para API keys, etc. No usualmente para paths)
load_dotenv()

# Asume que este script se ejecuta desde la raíz del proyecto
# O ajusta los paths relativos si se ejecuta desde scripts/
CONFIG_PATH = "configs/" 

def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config

if __name__ == "__main__":
    # Podrías pasar el nombre del config como argumento de línea de comandos
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--config_name", default="base_config.yaml", type=str)
    # args = parser.parse_args()
    # config = load_config(args.config_name)
    
    config = load_config("base_config.yaml") # Simplificado por ahora
    
    train_path = os.path.join(config['data_raw_path'], config['train_file'])
    df_train = pd.read_csv(train_path)
    
    print(f"Cargado {config['train_file']}, shape: {df_train.shape}")
    print(f"Columna objetivo: {config['target_column']}")
    
    # Aquí llamarías a tus funciones de src/data_preparation.py, src/feature_engineering.py, src/modelling.py
    # Ejemplo:
    # from src.data_preparation import clean_data
    # from src.feature_engineering import create_base_features
    # from src.modelling import train_lightgbm
    
    # cleaned_df = clean_data(df_train, config)
    # featured_df = create_base_features(cleaned_df, config)
    # X, y = ... # separar X e y
    # model, cv_score = train_lightgbm(X, y, config)
    
    print("Pipeline de entrenamiento (simulado) completado.")