# src/full_model_pipeline.py

import yaml
from sklearn import set_config
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMClassifier
from category_encoders import TargetEncoder

from src.sklearn_transformers import (
    NumericImputer,
    Winsorizer,
    LogTransformer,
    CategoricalImputer,
    RareGrouper
)
from src.feature_creator import FeatureCreator

def make_full_model_pipeline(cfg: dict) -> Pipeline:
    """
    Construye un pipeline completo:
      1) Limpieza (imputación + outlier treatment)
      2) Feature Engineering
      3) Encoding categórico (target o one-hot)
      4) Modelo LightGBM
    Todo parametrizado via YAML.
    """
    # Para que cada paso devuelva DataFrames con nombres de columna
    set_config(transform_output="pandas")
    
    # 2) Paso de limpieza
    num_cols = cfg['numerical_columns']
    cat_cols = cfg['categorical_columns']

    # 2.1 Pipeline numérica
    num_steps = [
        ('impute', NumericImputer(
            cols=num_cols,
            create_flag=cfg['preprocessing']['create_missing_flags']
        ))
    ]
    ot = cfg['preprocessing']['outlier_treatment_numerical']
    if ot == 'winsor':
        num_steps.append(('winsor', Winsorizer(
            cols=cfg['preprocessing'].get('winsor_cols', num_cols),
            lower_quantile=cfg['preprocessing']['winsor_quantiles_lower'],
            upper_quantile=cfg['preprocessing']['winsor_quantiles_upper']
        )))
    elif ot == 'log1p':
        num_steps.append(('log1p', LogTransformer(
            cols=cfg['preprocessing'].get('log_transform_cols', num_cols)
        )))
    # (si ot == 'none', no añadimos paso extra)

    num_clean = Pipeline(num_steps)

    # 2.2 Pipeline categórica
    cat_clean = Pipeline([
        ('impute', CategoricalImputer(
            cols=cat_cols,
            fill_value=cfg['preprocessing']['categorical_imputation_fill_value']
        )),
        ('rare', RareGrouper(
            cols=cat_cols,
            min_freq=cfg['preprocessing']['rare_grouping_threshold']
        ))
    ])

    # 2.3 ColumnTransformer de limpieza
    cleaner = ColumnTransformer(
        transformers=[
            ('num_clean', num_clean, num_cols),
            ('cat_clean', cat_clean, cat_cols)
        ],
        remainder='passthrough'  # deja pasar todo lo demás (para FE)
    )

    # 3) Feature Engineering
    fe_params = cfg.get('feature_engineering_params', {})
    fe = FeatureCreator(config_params=fe_params)

    # 4) Encoding categórico final
    enc_method = cfg['encoding']['method']
    if enc_method == 'target':
        encoder = ColumnTransformer(
            transformers=[
                ('target_enc',
                 TargetEncoder(
                     smoothing=cfg['encoding']['target_smoothing'],
                     handle_missing='value',
                     handle_unknown='value'
                 ),
                 make_column_selector(dtype_include=['object', 'category'])
                )
            ],
            remainder='passthrough'
        )
    elif enc_method == 'onehot':
        encoder = ColumnTransformer(
            transformers=[
                ('onehot_enc',
                 OneHotEncoder(
                     handle_unknown='ignore',
                     sparse_output=False
                 ),
                 make_column_selector(dtype_include=['object', 'category'])
                )
            ],
            remainder='passthrough'
        )
    else:
        raise ValueError(f"Unknown encoding method: {enc_method}")

    # 5) Modelo
    model_cfg = cfg['model']
    if model_cfg['name'].lower() != 'lightgbm':
        raise ValueError("Currently only LightGBM is supported")
    model = LGBMClassifier(**model_cfg['params'])

    # 6) Pipeline completo
    pipeline = Pipeline([
        ('clean', cleaner),
        ('fe',    fe),
        ('encode', encoder),
        ('model',  model)
    ])

    return pipeline
