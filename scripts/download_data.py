#!/usr/bin/env python3
"""
Download TalkPlayData-Challenge dataset from HuggingFace for RecSys 2026.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --split train
    python scripts/download_data.py --split all
"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

DATASET_REPO = "talkpl-ai/talkplay-data-challenge"
DATA_DIR = Path(os.getenv("RAW_DATA_DIR", "./data/raw"))

SPLITS = ["train", "dev", "blind_a"]  # blind_b released 1 week before challenge end


def download_dataset(split: str = "all", use_auth: bool = True):
    """Download TalkPlayData-Challenge splits from HuggingFace."""
    try:
        from datasets import load_dataset
        from huggingface_hub import login
    except ImportError:
        logger.error("Install huggingface deps: uv sync")
        raise

    hf_token = os.getenv("HF_TOKEN")
    if use_auth and hf_token:
        login(token=hf_token)
        logger.info("Logged in to HuggingFace.")
    elif use_auth:
        logger.warning("HF_TOKEN not set in .env. Attempting anonymous download.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    splits_to_download = SPLITS if split == "all" else [split]

    for s in splits_to_download:
        logger.info(f"Downloading split: {s}")
        try:
            ds = load_dataset(DATASET_REPO, split=s, token=hf_token)
            out_dir = DATA_DIR / s
            out_dir.mkdir(exist_ok=True)
            ds.save_to_disk(str(out_dir))
            logger.success(f"✅ Saved {s} split → {out_dir} ({len(ds)} examples)")
        except Exception as e:
            logger.error(f"Failed to download split '{s}': {e}")
            logger.warning(
                "\n"
                "⚠️  HuggingFace is not reachable from the corporate network.\n"
                "   Options to download data:\n"
                "   1. Run this script off the corporate network (home/hotspot)\n"
                "   2. Use huggingface-cli on a personal machine:\n"
                "        huggingface-cli download talkpl-ai/talkplay-data-challenge\n"
                "   3. Visit https://huggingface.co/datasets/talkpl-ai/talkplay-data-challenge\n"
                "      and download manually, then place files in ./data/raw/<split>/\n"
                f"   4. Set HF_TOKEN in .env for private dataset auth if required."
            )

    logger.success("Download complete!")


def main():
    parser = argparse.ArgumentParser(description="Download TalkPlayData-Challenge dataset")
    parser.add_argument(
        "--split",
        default="all",
        choices=["all", "train", "dev", "blind_a"],
        help="Which split to download (default: all)",
    )
    parser.add_argument(
        "--no-auth",
        action="store_true",
        help="Skip HuggingFace authentication",
    )
    args = parser.parse_args()
    download_dataset(split=args.split, use_auth=not args.no_auth)


if __name__ == "__main__":
    main()
