#!/usr/bin/env python3
"""
Download the Caspian Sea Hydroclimatic Dataset from Kaggle.

Usage:
    python download_dataset.py                    # Guide mode (interactive)
    python download_dataset.py --auto            # Auto-detect kaggle CLI
    python download_dataset.py --kaggle-key FILE  # Use kaggle.json credentials

Dataset: https://www.kaggle.com/datasets/chingchonghaha/caspian-sea-hydroclimatic-dataset-1993-2026
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

DATASET_OWNER = "chingchonghaha"
DATASET_SLUG = "caspian-sea-hydroclimatic-dataset-1993-2026"
DATASET_URL = f"https://www.kaggle.com/datasets/{DATASET_OWNER}/{DATASET_SLUG}"
REPO_ROOT = Path(__file__).resolve().parent
DATASET_DIR = REPO_ROOT / "dataset"
RAR_FILE = REPO_ROOT / "dataset.rar"


def check_kaggle_cli():
    """Check if kaggle CLI is installed and configured."""
    if not shutil.which("kaggle"):
        return False, "kaggle CLI not found. Install: pip install kaggle"

    # Check if credentials exist
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    if not kaggle_json.exists():
        return False, f"No kaggle.json found at {kaggle_json}"

    return True, "Ready"


def download_with_kaggle_cli():
    """Download using kaggle CLI."""
    print(f"Downloading dataset: {DATASET_OWNER}/{DATASET_SLUG}")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", f"{DATASET_OWNER}/{DATASET_SLUG}",
         "-p", str(REPO_ROOT)],
        check=True
    )
    return RAR_FILE.exists()


def extract_dataset():
    """Extract dataset.rar to dataset/ directory."""
    if not RAR_FILE.exists():
        print(f"Error: {RAR_FILE} not found")
        return False

    DATASET_DIR.mkdir(exist_ok=True)

    # Check for unrar or 7z
    if shutil.which("unrar"):
        print("Extracting with unrar...")
        subprocess.run(["unrar", "x", "-y", str(RAR_FILE), str(DATASET_DIR)], check=True)
    elif shutil.which("7z"):
        print("Extracting with 7z...")
        subprocess.run(["7z", "x", str(RAR_FILE), f"-o{DATASET_DIR}", "-y"], check=True)
    else:
        print("\nNo extractor found. Install 'unrar' or '7z':")
        print("  sudo apt install unrar")
        print("  or extract dataset.rar manually to dataset/")
        return False

    os.remove(RAR_FILE)
    return True


def setup_kaggle_credentials(key_path=None):
    """Set up kaggle.json credentials."""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"

    if key_path:
        shutil.copy(key_path, kaggle_json)
    else:
        print("\nTo use Kaggle CLI, create an API key:")
        print("1. Go to https://www.kaggle.com/settings/account")
        print("2. Click 'Create New API Token' (downloads kaggle.json)")
        print("3. Place it at ~/.kaggle/kaggle.json")
        print("   mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/")
        print("   chmod 600 ~/.kaggle/kaggle.json")

    if kaggle_json.exists():
        os.chmod(kaggle_json, 0o600)
        return True
    return False


def main():
    auto = "--auto" in sys.argv
    key_path = None

    for i, arg in enumerate(sys.argv):
        if arg == "--kaggle-key" and i + 1 < len(sys.argv):
            key_path = sys.argv[i + 1]

    print("=" * 60)
    print("Caspian Sea Hydroclimatic Dataset Downloader")
    print(f"Dataset: {DATASET_URL}")
    print("=" * 60)

    # Check if already downloaded
    if DATASET_DIR.exists() and any(DATASET_DIR.iterdir()):
        print(f"\nDataset already exists at {DATASET_DIR}")
        print("Delete it first to re-download.")
        return

    # Try kaggle CLI
    cli_ok, msg = check_kaggle_cli()

    if not cli_ok:
        if key_path:
            setup_kaggle_credentials(key_path)
            cli_ok, msg = check_kaggle_cli()
        elif auto:
            setup_kaggle_credentials()
            cli_ok, msg = check_kaggle_cli()

    if cli_ok:
        try:
            if download_with_kaggle_cli():
                extract_dataset()
                print(f"\nDone! Dataset at {DATASET_DIR}")
                return
        except Exception as e:
            print(f"\nKaggle CLI download failed: {e}")

    # Manual fallback
    print(f"\n--- Manual Download Instructions ---")
    print(f"1. Visit: {DATASET_URL}")
    print(f"2. Click 'Download' (requires Kaggle account)")
    print(f"3. Save dataset.rar to: {REPO_ROOT}/")
    print(f"4. Extract: unrar x dataset.rar dataset/")
    print(f"\nOr set up Kaggle CLI for automatic download:")
    setup_kaggle_credentials()


if __name__ == "__main__":
    main()
