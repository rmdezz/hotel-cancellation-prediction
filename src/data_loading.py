# src/data_loading.py

import os
import pandas as pd
import logging
import argparse

from utils import load_config, setup_logging

def load_raw_data(cfg):
    raw_path = cfg['data_raw_path']
    train = pd.read_csv(os.path.join(raw_path, cfg['train_file']))
    test  = pd.read_csv(os.path.join(raw_path, cfg['test_file']))
    return train, test

def save_parquet(df, path):
    df.to_parquet(path, index=False)
    logging.info(f"Guardado: {path}")

if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    train, test = load_raw_data(cfg)
    os.makedirs(cfg['data_processed_path'], exist_ok=True)
    save_parquet(train, os.path.join(cfg['data_processed_path'], 'train_raw.parquet'))
    save_parquet(test,  os.path.join(cfg['data_processed_path'], 'test_raw.parquet'))
