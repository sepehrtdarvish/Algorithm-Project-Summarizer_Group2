import yaml
import os
import logging
from typing import Dict, Any

def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Loads the YAML configuration file.
    Uses absolute path relative to the project root to avoid FileNotFoundError.
    """
    if config_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        config_path = os.path.join(project_root, "config", "config.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {os.path.abspath(config_path)}")
    
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config

def setup_logger(name: str = "ProjectLogger") -> logging.Logger:
    """
    Sets up a standard logger for console output.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

def read_text_file(file_path: str) -> str:
    """
    Reads the input text file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found at {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def save_summary(output_dir: str, filename: str, summary: str):
    """
    Saves the final summary to the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(summary)