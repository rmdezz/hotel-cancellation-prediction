import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureCreator(BaseEstimator, TransformerMixin):
    def __init__(self, config_params=None):
        self.config_params = config_params if config_params is not None else {}
        
        default_antelacion_bins = [-1, 0, 7, 30, 90, 180, 10000] # -1 para incluir 0 con right=True
        default_antelacion_labels = ['0_SameDay', '1_1-7Days', '2_8-30Days', '3_31-90Days', '4_91-180Days', '5_180+Days']
        default_month_to_season_map = {
            1: 'Invierno', 2: 'Invierno', 3: 'Primavera', 4: 'Primavera', 5: 'Primavera',
            6: 'Verano', 7: 'Verano', 8: 'Verano', 9: 'Otoño', 10: 'Otoño',
            11: 'Otoño', 12: 'Invierno'
        }

        self.antelacion_bins = self.config_params.get('antelacion_bins', default_antelacion_bins)
        self.antelacion_labels = self.config_params.get('antelacion_labels', default_antelacion_labels)
        self.month_to_season_map = self.config_params.get('month_to_season_map', default_month_to_season_map)
        self.cols_to_drop_after_fe_config = self.config_params.get('cols_to_drop_after_fe', [])


    def fit(self, X, y=None):
        # Identificar los prefijos usados por ColumnTransformer en X
        # Esto es un poco más robusto que hardcodear "num_clean__" o "remainder__"
        # Asumimos que las columnas tienen un solo "::" o "__" como separador del prefijo
        self.num_prefix_ = "num_clean__" 
        self.cat_prefix_ = "cat_clean__"
        self.rem_prefix_ = "remainder__"
        
        # Podríamos intentar detectar los prefijos si X es un DataFrame y tiene columnas
        # if isinstance(X, pd.DataFrame) and not X.empty:
        #     sample_col = X.columns[0]
        #     if "__" in sample_col:
        #         self.detected_prefix_separator_ = "__"
        #     elif "::" in sample_col: # Ejemplo si el separador fuera otro
        #         self.detected_prefix_separator_ = "::"
        # # Esto es más avanzado y puede no ser necesario si los prefijos son consistentes

        return self

    def _get_col_name(self, original_col_name, X_cols):
        """Helper para encontrar el nombre de columna con prefijo."""
        # Intenta con prefijos comunes o el detectado
        prefixes_to_try = [self.num_prefix_, self.cat_prefix_, self.rem_prefix_, ""] # "" para columnas sin prefijo (nuevas)
        for p in prefixes_to_try:
            prefixed_name = f"{p}{original_col_name}"
            if prefixed_name in X_cols:
                return prefixed_name
        # Si no se encuentra con prefijo, y el original_col_name está en X_cols (ya es una feature creada)
        if original_col_name in X_cols:
            return original_col_name
        return None # O lanzar error / warning

    def transform(self, X):
        X_transformed = X.copy()
        X_cols = X_transformed.columns # Nombres de columnas de entrada

        # --- Nombres de columnas de entrada (usando _get_col_name) ---
        adultos_col = self._get_col_name('Num_Adultos', X_cols)
        ninos_col = self._get_col_name('Num_Niños', X_cols)
        noches_semana_col = self._get_col_name('Noches_Semana', X_cols)
        noches_fin_semana_col = self._get_col_name('Noches_Fin_Semana', X_cols)
        precio_prom_hab_col = self._get_col_name('Precio_Promedio_Por_Habitación', X_cols)
        antelacion_col = self._get_col_name('Tiempo_Antelación', X_cols)
        mes_llegada_col = self._get_col_name('Mes_Llegada', X_cols)

        # --- 1. Duración Total de la Estancia ---
        if noches_semana_col and noches_fin_semana_col:
            X_transformed['Duracion_Total'] = X_transformed[noches_semana_col] + X_transformed[noches_fin_semana_col]
        else:
            print(f"Warning: Columnas para Duracion_Total no encontradas. {noches_semana_col=}, {noches_fin_semana_col=}")
            X_transformed['Duracion_Total'] = 0 

        duracion_total_col = 'Duracion_Total' # Nombre de la nueva feature

        # --- 2. Total de Huéspedes ---
        if adultos_col and ninos_col:
            X_transformed['Total_Huespedes'] = X_transformed[adultos_col] + X_transformed[ninos_col]
            X_transformed['Total_Huespedes'] = X_transformed['Total_Huespedes'].replace(0, 1)
        else:
            print(f"Warning: Columnas para Total_Huespedes no encontradas. {adultos_col=}, {ninos_col=}")
            X_transformed['Total_Huespedes'] = 1
        
        total_huespedes_col = 'Total_Huespedes'

        # --- 3. Precio por Persona y Noche ---
        if precio_prom_hab_col and total_huespedes_col: # total_huespedes_col ya es el nombre de la nueva feature
            X_transformed['Precio_por_Persona_Noche'] = np.where(
                X_transformed[total_huespedes_col] > 0,
                X_transformed[precio_prom_hab_col] / X_transformed[total_huespedes_col],
                X_transformed[precio_prom_hab_col] 
            )
        else:
            print(f"Warning: Columnas para Precio_por_Persona_Noche no encontradas. {precio_prom_hab_col=}, {total_huespedes_col=}")
            X_transformed['Precio_por_Persona_Noche'] = np.nan # Imputar luego si es necesario


        # --- 4. Precio Total Estancia ---
        if precio_prom_hab_col and duracion_total_col: # duracion_total_col ya es el nombre de la nueva feature
            X_transformed['Precio_Total_Estancia'] = X_transformed[precio_prom_hab_col] * X_transformed[duracion_total_col]
        else:
            print(f"Warning: Columnas para Precio_Total_Estancia no encontradas. {precio_prom_hab_col=}, {duracion_total_col=}")
            X_transformed['Precio_Total_Estancia'] = np.nan

        # --- 5. Flag SameDayBooking ---
        if antelacion_col:
            X_transformed['SameDayBooking'] = (X_transformed[antelacion_col] == 0).astype(int)
        else:
            print(f"Warning: Columna {antelacion_col} para SameDayBooking no encontrada.")
            X_transformed['SameDayBooking'] = 0

        # --- 6. Bins de Antelación ---
        if antelacion_col:
            X_transformed['Antelacion_Bin'] = pd.cut(X_transformed[antelacion_col],
                                                     bins=self.antelacion_bins,
                                                     labels=self.antelacion_labels,
                                                     right=True, include_lowest=True)
        else:
            print(f"Warning: Columna {antelacion_col} para Antelacion_Bin no encontrada.")
            X_transformed['Antelacion_Bin'] = 'Missing_Bin' 

        # --- 7. Features Cíclicas de Mes ---
        if mes_llegada_col:
            X_transformed['Mes_sin'] = np.sin(2 * np.pi * X_transformed[mes_llegada_col] / 12)
            X_transformed['Mes_cos'] = np.cos(2 * np.pi * X_transformed[mes_llegada_col] / 12)
        else:
            print(f"Warning: Columna {mes_llegada_col} para Mes_sin/cos no encontrada.")
            X_transformed['Mes_sin'] = 0
            X_transformed['Mes_cos'] = 0
            
        # --- 8. Season Bucket ---
        if mes_llegada_col:
            X_transformed['Season_Bucket'] = X_transformed[mes_llegada_col].map(self.month_to_season_map).fillna('Desconocido')
        else:
            print(f"Warning: Columna {mes_llegada_col} para Season_Bucket no encontrada.")
            X_transformed['Season_Bucket'] = 'Desconocido'

        # --- Eliminar columnas originales si se especifica en config ---
        actual_cols_to_drop = []
        for original_col_name in self.cols_to_drop_after_fe_config:
            col_to_find_and_drop = self._get_col_name(original_col_name, X_cols) # X_cols son las columnas de entrada a transform
            if col_to_find_and_drop and col_to_find_and_drop in X_transformed.columns: # Asegurarse que aún existe
                actual_cols_to_drop.append(col_to_find_and_drop)
        
        if actual_cols_to_drop:
            X_transformed = X_transformed.drop(columns=actual_cols_to_drop, errors='ignore')
            print(f"Dropped columns after FE: {actual_cols_to_drop}")
            
        return X_transformed

    def get_feature_names_out(self, input_features=None):
        # Esta función es importante para que el pipeline sepa los nombres de salida,
        # especialmente si no se usa set_output(transform="pandas") o si hay transformaciones complejas de nombres.
        # Si transform devuelve un DataFrame, y set_output="pandas" está activo,
        # scikit-learn a menudo puede inferir los nombres.
        # Por ahora, la dejaremos simple. Si transform devuelve un DF, los nombres ya están ahí.
        # Si no, necesitaríamos construir la lista de nombres de salida basada en input_features
        # y las transformaciones hechas.
        # Ejemplo muy básico (asume que transform devuelve DF y queremos esos nombres):
        # No es realmente necesario si se usa set_output="pandas" y transform devuelve un DF.
        pass # Scikit-learn usará los nombres del DataFrame devuelto por transform() si es posible.