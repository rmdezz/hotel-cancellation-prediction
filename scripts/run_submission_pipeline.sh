#!/bin/bash
set -e # Detener si hay un error

CONFIG_FILE_PATH=$1

if [ -z "$CONFIG_FILE_PATH" ]; then
    echo "Error: Debes proporcionar la ruta al archivo de configuración."
    echo "Uso: bash scripts/run_submission_pipeline.sh configs/tu_mejor_config.yaml"
    exit 1
fi

echo ">>> Ejecutando pipeline completo para generar submission..."
python src/generate_final_submission.py --config "$CONFIG_FILE_PATH" --submission_file "submission.csv"

echo ">>> Proceso completado. 'submission.csv' generado."