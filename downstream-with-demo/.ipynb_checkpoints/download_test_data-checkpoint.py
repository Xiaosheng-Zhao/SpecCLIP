"""
Download test data from HuggingFace Datasets (separate from models)

This script downloads test data from the SpecCLIP-TestData-Small dataset repository.
"""
from huggingface_hub import hf_hub_download, snapshot_download
from pathlib import Path
import json
import os
import sys

# Configuration - load from dataset_config.json if available
DEFAULT_LOCAL_DIR = "./test_data"

def get_dataset_config():
    """Load dataset configuration from dataset_config.json"""
    config_path = Path(__file__).parent / "dataset_config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)

    # Default configuration
    return {
        "repository": "astroshawn/SpecCLIP-TestData-Small",
        "repo_type": "dataset",
        "files": {
            "gaia_lamost_test_only.h5": {
                "description": "Test set with paired Gaia XP and LAMOST LRS spectra"
            }
        }
    }

DATASET_CONFIG = get_dataset_config()
DATASET_REPO_ID = DATASET_CONFIG.get("repository", "astroshawn/SpecCLIP-TestData-Small")


def download_test_data(local_dir=None, filename=None):
    """
    Download test data from HuggingFace Datasets.

    Args:
        local_dir: Local directory to save test data (default: ./test_data)
        filename: Specific file to download (default: download all)

    Returns:
        Path to downloaded file(s)
    """
    if local_dir is None:
        local_dir = DEFAULT_LOCAL_DIR

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("Downloading SpecCLIP Test Data from HuggingFace Datasets")
    print("=" * 70)
    print(f"Dataset Repository: {DATASET_REPO_ID}")
    print(f"Local directory: {local_dir}")

    if filename:
        # Download specific file
        print(f"\nDownloading: {filename}")
        print("This may take several minutes for large files...\n")

        try:
            local_path = hf_hub_download(
                repo_id=DATASET_REPO_ID,
                filename=filename,
                repo_type="dataset",
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
                resume_download=True
            )
            print(f"\n✓ Downloaded: {local_path}")
            return local_path
        except Exception as e:
            print(f"\n✗ Download failed: {e}")
            sys.exit(1)
    else:
        # Download all files
        print("\nDownloading all test data files...")
        print("This may take several minutes for large files...\n")

        try:
            local_path = snapshot_download(
                repo_id=DATASET_REPO_ID,
                repo_type="dataset",
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
                resume_download=True
            )
            print(f"\n✓ All files downloaded to: {local_path}")
            return local_path
        except Exception as e:
            print(f"\n✗ Download failed: {e}")
            sys.exit(1)


def get_test_data_path(local_dir=None, filename="gaia_lamost_test_only.h5"):
    """
    Get the path to test data file, downloading if necessary.

    Args:
        local_dir: Local directory for test data
        filename: Name of the test data file

    Returns:
        Path to test data file
    """
    if local_dir is None:
        local_dir = DEFAULT_LOCAL_DIR

    local_path = Path(local_dir) / filename

    if local_path.exists():
        print(f"✓ Test data found: {local_path}")
        return str(local_path)
    else:
        print(f"Test data not found locally. Downloading...")
        return download_test_data(local_dir, filename)


def update_config_with_test_data(config_path, test_data_path):
    """
    Update a configuration file with the test data path.

    Args:
        config_path: Path to configuration JSON file
        test_data_path: Path to test data file
    """
    if not os.path.exists(config_path):
        print(f"✗ Config file not found: {config_path}")
        return

    with open(config_path, 'r') as f:
        config = json.load(f)

    config['h5_data_path'] = str(test_data_path)

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"✓ Updated {config_path} with test data path")


def main():
    """Main function for command line usage"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download SpecCLIP test data from HuggingFace"
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default=DEFAULT_LOCAL_DIR,
        help=f"Local directory to save test data (default: {DEFAULT_LOCAL_DIR})"
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        help="Specific file to download (default: download all)"
    )
    parser.add_argument(
        "--update-config",
        type=str,
        default=None,
        help="Path to config file to update with test data path"
    )

    args = parser.parse_args()

    # Download test data
    result_path = download_test_data(args.local_dir, args.filename)

    # Update config if requested
    if args.update_config:
        if args.filename:
            test_data_path = result_path
        else:
            test_data_path = Path(result_path) / "gaia_lamost_test_only.h5"

        update_config_with_test_data(args.update_config, test_data_path)

    print("\n" + "=" * 70)
    print("Download Complete!")
    print("=" * 70)

    # Print usage example
    print("\nUsage example:")
    print("```python")
    print("import h5py")
    print("import numpy as np")
    print("")
    if args.filename:
        print(f'with h5py.File("{result_path}", "r") as f:')
    else:
        print(f'with h5py.File("{Path(result_path) / "gaia_lamost_test_only.h5"}", "r") as f:')
    print('    source_ids = np.array(f["test/source_ids"][:])')
    print('    gaia_spectra = np.array(f["test/gaia_spectra"][:])')
    print('    lamost_spectra = np.array(f["test/lamost_spectra"][:])')
    print("```")


if __name__ == "__main__":
    main()