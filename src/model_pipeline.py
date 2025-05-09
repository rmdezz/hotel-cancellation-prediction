from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from .feature_creator import FeatureCreator

from .sklearn_transformers import (
    Winsorizer,
    LogTransformer,
    NumericImputer,
    CategoricalImputer,
    RareGrouper
)

def make_cleaning_pipeline(cfg: dict) -> Pipeline:
    num_cols = cfg['numerical_columns']
    cat_cols = cfg['categorical_columns']
    
    outlier_treatment = cfg['preprocessing']['outlier_treatment_numerical']
    
    # --- Pipeline Numérico (Limpieza) ---
    num_limpieza_steps = [
        ('impute_num', NumericImputer(
            cols=num_cols, 
            create_flag=cfg['preprocessing']['create_missing_flags']
        ))
    ]
    if outlier_treatment == 'winsor':
        num_limpieza_steps.append(('winsor', Winsorizer(
            cols=cfg['preprocessing'].get('winsor_cols', num_cols), 
            lower_quantile=cfg['preprocessing']['winsor_quantiles_lower'],
            upper_quantile=cfg['preprocessing']['winsor_quantiles_upper']
        )))
    elif outlier_treatment == 'log1p':
        num_limpieza_steps.append(('log', LogTransformer(
            cols=cfg['preprocessing'].get('log_transform_cols', num_cols)
        )))
    elif outlier_treatment != 'none':
        raise ValueError(f"Unknown outlier_treatment_numerical: {outlier_treatment}")
    
    numerical_cleaner = Pipeline(num_limpieza_steps)

    # --- Pipeline Categórico (Limpieza) ---
    cat_limpieza_steps = [
        ('impute_cat', CategoricalImputer(
            cols=cat_cols, 
            fill_value=cfg['preprocessing']['categorical_imputation_fill_value']
        )),
        ('rare', RareGrouper(
            cols=cat_cols, 
            min_freq=cfg['preprocessing']['rare_grouping_threshold']
        ))
    ]
    categorical_cleaner = Pipeline(cat_limpieza_steps)

    # --- Preprocesador Combinado (Limpieza Inicial) ---
    # Se procesarán num_cols y cat_cols. El resto se pasarán sin cambios (remainder='passthrough')
    # Es importante que las columnas que no son ni num_cols ni cat_cols (ej. 'Id', 'target')
    # no se incluyan aquí si no quieres que pasen o se eliminen.
    # Por ahora, asumimos que num_cols y cat_cols son las únicas que queremos procesar.
    initial_preprocessor = ColumnTransformer(
        transformers=[
            ('num_clean', numerical_cleaner, num_cols),
            ('cat_clean', categorical_cleaner, cat_cols)
        ],
        remainder='passthrough' # Importante: Verifica qué columnas quieres que pasen.
                               # Si solo quieres las procesadas, usa 'drop'.
                               # Si usas 'passthrough' y tienes 'Id' y 'target' en X, pasarán.
    )
    
    # Por ahora, el pipeline solo hace la limpieza.
    # Esto te permite probar `fit_transform`.
    cleaning_pipeline = Pipeline([
        ('initial_cleaning', initial_preprocessor)
    ])

    # Opcional: Si tienes scikit-learn >= 1.2, para que devuelva DataFrames
    # from sklearn import set_config
    # set_config(transform_output="pandas") # Configuración global
    # O individualmente:
    # initial_preprocessor.set_output(transform="pandas")
    # cleaning_pipeline.set_output(transform="pandas")


    return cleaning_pipeline

def make_feature_engineering_pipeline(cfg: dict) -> Pipeline:
    """
    Crea un pipeline que incluye limpieza y creación de features.
    """
    # Paso 1: Obtener el pipeline de limpieza
    cleaning_pipe = make_cleaning_pipeline(cfg)

    # Paso 2: Definir parámetros para FeatureCreator si los hubiera en el config
    feature_creator_params = cfg.get('feature_engineering_params', {})

    # Crear el pipeline completo de preprocesamiento
    # (Este es el pipeline que usaremos para generar X_train_processed, X_test_processed)
    full_preprocessing_pipeline = Pipeline([
        ('cleaning', cleaning_pipe), # El pipeline de limpieza completo
        ('feature_creation', FeatureCreator(config_params=feature_creator_params)) # Nuestro nuevo transformer
    ])
    
    return full_preprocessing_pipeline