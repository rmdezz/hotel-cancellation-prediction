# src/feature_engineering.py

import os
import numpy as np
import pandas as pd
import logging
import argparse

from utils import load_config, setup_logging

def create_features(df):
    # 1. Duración total
    df['Duracion_Total'] = df['Noches_Semana'] + df['Noches_Fin_Semana']

    # 2. Precios por noche y por persona
    df['Precio_por_Noche'] = df['Precio_Promedio_Por_Habitación'] / df['Duracion_Total'].replace(0,1)
    df['Total_Huespedes']   = df['Num_Adultos'] + df['Num_Niños']
    df['Precio_por_Persona'] = df['Precio_Promedio_Por_Habitación'] / df['Total_Huespedes'].replace(0,1)

    # 3. Same-day booking flag
    df['SameDayBooking'] = (df['Tiempo_Antelación'] == 0).astype(int)

    # 4. Bin antelación
    bins   = [-1,0,7,30, df['Tiempo_Antelación'].max()]
    labels = ['same_day','1_7','8_30','>30']
    df['Antelacion_Bin'] = pd.cut(df['Tiempo_Antelación'], bins=bins, labels=labels)

    # 5. Mes cíclico
    df['Mes_sin'] = np.sin(2 * np.pi * df['Mes_Llegada'] / 12)
    df['Mes_cos'] = np.cos(2 * np.pi * df['Mes_Llegada'] / 12)

    # 6. Season bucket
    season_map = {
        12:'Winter',1:'Winter',2:'Winter',
        3:'Spring',4:'Spring',5:'Spring',
        6:'Summer',7:'Summer',8:'Summer',
        9:'Fall',10:'Fall',11:'Fall'
    }
    df['Season'] = df['Mes_Llegada'].map(season_map)

    return df

if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    path = cfg['data_processed_path']
    train = pd.read_parquet(os.path.join(path, 'train_clean.parquet'))
    test  = pd.read_parquet(os.path.join(path, 'test_clean.parquet'))

    train_feat = create_features(train)
    test_feat  = create_features(test)

    train_feat.to_parquet(os.path.join(path, 'train_feat.parquet'), index=False)
    test_feat.to_parquet(os.path.join(path, 'test_feat.parquet'), index=False)
    logging.info("Feature engineering finalizado.")
