"""Data loading utilities for TalkPlayData-Challenge."""

import os
from pathlib import Path
from typing import Optional

import pandas as pd
from datasets import Dataset, load_from_disk
from loguru import logger

RAW_DATA_DIR = Path(os.getenv("RAW_DATA_DIR", "./data/raw"))


def load_split(split: str, data_dir: Optional[Path] = None) -> Dataset:
    """Load a HuggingFace dataset split from disk.

    Args:
        split: One of 'train', 'dev', 'blind_a', 'blind_b'
        data_dir: Override default RAW_DATA_DIR

    Returns:
        HuggingFace Dataset object
    """
    data_dir = data_dir or RAW_DATA_DIR
    split_path = data_dir / split
    if not split_path.exists():
        raise FileNotFoundError(
            f"Split '{split}' not found at {split_path}. "
            f"Run: python scripts/download_data.py --split {split}"
        )
    logger.info(f"Loading split '{split}' from {split_path}")
    return load_from_disk(str(split_path))


def load_track_metadata(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load track metadata (1M+ tracks).

    Returns:
        DataFrame with columns: track_id, track_name, artist_name,
                                album_name, tags, release_date
    """
    data_dir = data_dir or RAW_DATA_DIR
    meta_path = data_dir / "track_metadata"

    # Try parquet first (faster), fall back to HF dataset
    parquet_path = meta_path / "metadata.parquet"
    if parquet_path.exists():
        logger.info(f"Loading track metadata from {parquet_path}")
        return pd.read_parquet(parquet_path)

    if meta_path.exists():
        ds = load_from_disk(str(meta_path))
        return ds.to_pandas()

    raise FileNotFoundError(
        f"Track metadata not found at {meta_path}. "
        "Run: python scripts/download_data.py"
    )


def load_user_profiles(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load user profiles (demographics + listening history).

    Returns:
        DataFrame with columns: user_id, age, gender, country, track_history
    """
    data_dir = data_dir or RAW_DATA_DIR
    profiles_path = data_dir / "user_profiles"

    parquet_path = profiles_path / "profiles.parquet"
    if parquet_path.exists():
        logger.info(f"Loading user profiles from {parquet_path}")
        return pd.read_parquet(parquet_path)

    if profiles_path.exists():
        ds = load_from_disk(str(profiles_path))
        return ds.to_pandas()

    raise FileNotFoundError(
        f"User profiles not found at {profiles_path}. "
        "Run: python scripts/download_data.py"
    )
