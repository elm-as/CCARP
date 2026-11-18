"""Kaggle dataset acquisition and local CSV loading for CCARP."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pandas as pd

from src.core.config import DatasetConfig


def has_local_ccarp_files(raw_dir: Path) -> bool:
    """Return True when both raw CCARP CSV files exist locally with non-zero size."""
    if not raw_dir.exists():
        return False
    app_path = raw_dir / "application_record.csv"
    credit_path = raw_dir / "credit_record.csv"
    return (
        app_path.is_file()
        and credit_path.is_file()
        and app_path.stat().st_size > 0
        and credit_path.stat().st_size > 0
    )


def download_ccarp_dataset(config: DatasetConfig) -> Path:
    """Download and unzip the Kaggle credit-card approval dataset into raw_data_dir."""
    config.raw_data_dir.mkdir(parents=True, exist_ok=True)
    if not config.kaggle_exe.is_file():
        raise FileNotFoundError(
            f"Kaggle executable not found at {config.kaggle_exe}"
        )
    kaggle_json = config.kaggle_config_dir / "kaggle.json"
    if not kaggle_json.is_file():
        raise FileNotFoundError(
            f"kaggle.json not found in {config.kaggle_config_dir}"
        )

    env = os.environ.copy()
    env["KAGGLE_CONFIG_DIR"] = str(config.kaggle_config_dir)
    cmd = [
        str(config.kaggle_exe),
        "datasets",
        "download",
        "-d",
        config.kaggle_slug,
        "-p",
        str(config.raw_data_dir),
        "--unzip",
        "--force",
    ]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Kaggle download failed (code {result.returncode}): {result.stderr.strip()}"
        )
    if not has_local_ccarp_files(config.raw_data_dir):
        raise FileNotFoundError(
            f"Expected CSV files missing after download in {config.raw_data_dir}"
        )
    return config.raw_data_dir


def load_raw_tables(config: DatasetConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load application_record.csv and credit_record.csv, downloading if absent."""
    if not has_local_ccarp_files(config.raw_data_dir):
        download_ccarp_dataset(config)
    app_path = config.raw_data_dir / config.application_filename
    credit_path = config.raw_data_dir / config.credit_filename
    app_df = pd.read_csv(app_path)
    credit_df = pd.read_csv(credit_path, dtype={"STATUS": str})
    return app_df, credit_df
