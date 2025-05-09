# src/utils.py

import yaml
import logging

def load_config(path: str) -> dict:
    """Carga un archivo YAML de configuración."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def setup_logging():
    """Configura el logging básico."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s'
    )
