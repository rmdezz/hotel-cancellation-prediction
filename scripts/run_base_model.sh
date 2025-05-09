# scripts/run_base_model.sh
#!/bin/bash
set -e

CONFIG=$1

echo "=== 1. Cargando datos raw ==="
python src/data_loading.py --config $CONFIG

echo "=== 2. Limpiando datos ==="
python src/data_cleaning.py --config $CONFIG

echo "=== 3. Ingeniería de features ==="
python src/feature_engineering.py --config $CONFIG

echo ">>> Preprocesamiento y FE completados."
