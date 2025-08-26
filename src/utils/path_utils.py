# src/utils/path_utils.py

import os

def ensure_dir_exists(dir_path: str):
    os.makedirs(dir_path, exist_ok=True)