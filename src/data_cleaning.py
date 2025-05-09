# src/data_cleaning.py

import os
import numpy as np
import pandas as pd
import logging
import argparse

from utils import load_config, setup_logging

def impute_numeric(df, cols):
    """Imputa numéricos con mediana + flag de missing."""
    for col in cols:
        med = df[col].median()
        df[f"{col}_miss"] = df[col].isnull().astype(int)
        df[col] = df[col].fillna(med)
    return df

def winsorize(df, cols, lower_q=0.01, upper_q=0.99):
    """Clipa valores extremos en los quantiles dados."""
    for col in cols:
        low = df[col].quantile(lower_q)
        high = df[col].quantile(upper_q)
        df[col] = df[col].clip(low, high)
    return df

def impute_categorical(df, cols):
    """Rellena nulos categóricos con 'Missing'."""
    for col in cols:
        df[col] = df[col].fillna('Missing')
    return df

def group_rare(df, col, thresh=0.01):
    """Agrupa categorías con frecuencia < thresh en 'Other'."""
    freq = df[col].value_counts(normalize=True)
    rare = freq[freq < thresh].index
    df[col] = df[col].replace(rare, 'Other')
    return df

if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    path = cfg['data_processed_path']
    train = pd.read_parquet(os.path.join(path, 'train_raw.parquet'))
    test  = pd.read_parquet(os.path.join(path, 'test_raw.parquet'))

    num_cols = cfg.get('numerical_columns', train.select_dtypes(include=np.number).columns.tolist())
    cat_cols = cfg.get('categorical_columns', train.select_dtypes(include='object').columns.tolist())
    # Excluir IDs y target
    cat_cols = [c for c in cat_cols if c not in [cfg['target_column'], cfg['id_column_text']]]

    # Imputación numérica + flags
    train = impute_numeric(train, num_cols)
    test  = impute_numeric(test, num_cols)
    # Winsorización
    train = winsorize(train, ['Tiempo_Antelación', 'Precio_Promedio_Por_Habitación'])
    test  = winsorize(test, ['Tiempo_Antelación', 'Precio_Promedio_Por_Habitación'])
    # Imputación categórica + rarezas
    train = impute_categorical(train, cat_cols)
    test  = impute_categorical(test, cat_cols)
    for col in cat_cols:
        train = group_rare(train, col)
        test  = group_rare(test, col)

    train.to_parquet(os.path.join(path, 'train_clean.parquet'), index=False)
    test.to_parquet(os.path.join(path, 'test_clean.parquet'), index=False)
    logging.info("Data cleaning finalizado.")
