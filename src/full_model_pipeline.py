# src/full_model_pipeline.py

import yaml
from sklearn import set_config
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder
from category_encoders import TargetEncoder

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from src.sklearn_transformers import (
    NumericImputer,
    Winsorizer,
    LogTransformer,
    CategoricalImputer,
    RareGrouper
)
from src.feature_creator import FeatureCreator


def make_full_model_pipeline(cfg_path: str) -> Pipeline:
    """
    Construye un Pipeline completo parametrizable mediante YAML.
    Modelos soportados: lightgbm, xgboost, catboost.
    """
    # 0) Pandas DataFrame output
    set_config(transform_output="pandas")

    # 1) Cargar configuración
    if isinstance(cfg_path, str):
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = cfg_path  # si ya le pasas un dict

    # 2) Limpieza numérica + categórica
    num_cols = cfg['numerical_columns']
    cat_cols = cfg['categorical_columns']

    # 2.1) Pipeline numérico
    num_steps = [
        ('imputer', NumericImputer(
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

    num_clean = Pipeline(num_steps)

    # 2.2) Pipeline categórico
    cat_clean = Pipeline([
        ('imputer', CategoricalImputer(
            cols=cat_cols,
            fill_value=cfg['preprocessing']['categorical_imputation_fill_value']
        )),
        ('rare', RareGrouper(
            cols=cat_cols,
            min_freq=cfg['preprocessing']['rare_grouping_threshold']
        ))
    ])

    cleaner = ColumnTransformer(
        transformers=[
            ('num_clean', num_clean, num_cols),
            ('cat_clean', cat_clean, cat_cols)
        ],
        remainder='passthrough'
    )

    # 3) Feature Engineering
    fe = FeatureCreator(config_params=cfg.get('feature_engineering_params', {}))

    # 4) Encoding categórico
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
                ('onehot',
                    OneHotEncoder(handle_unknown='ignore', sparse_output=False),
                    make_column_selector(dtype_include=['object', 'category'])
                )
            ],
            remainder='passthrough'
        )
    else:
        raise ValueError(f"Unknown encoding method: {enc_method}")

    # 5) Instanciar el modelo según cfg
    mdl_cfg = cfg['model']
    name = mdl_cfg['name'].lower()
    params = mdl_cfg.get('params', {})
    if name == 'lightgbm':
        model = LGBMClassifier(**params)
    elif name == 'xgboost':
        model = XGBClassifier(**params)
    elif name == 'catboost':
        model = CatBoostClassifier(verbose=False, **params)
    else:
        raise ValueError(f"Unsupported model: {mdl_cfg['name']}")

    # 6) Pipeline completo
    pipeline = Pipeline([
        ('clean', cleaner),
        ('fe', fe),
        ('encode', encoder),
        ('model', model)
    ])

    return pipeline
