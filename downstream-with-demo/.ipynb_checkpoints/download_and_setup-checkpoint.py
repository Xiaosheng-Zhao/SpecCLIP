"""
Download models and test data from HuggingFace
"""
from huggingface_hub import hf_hub_download, snapshot_download, login
import json
import os
from pathlib import Path
import sys

# Configuration
HF_REPO_ID = "astroshawn/SpecCLIP"
LOCAL_MODEL_DIR = "/home/idies/workspace/Temporary/xzhao/scratch/pretrained_models_new"

def download_all_models(use_snapshot=True, include_test_data=False):
    """
    Download all models from HuggingFace.

    Args:
        use_snapshot: If True, download entire repo at once (faster)
        include_test_data: If True, also download test data from separate dataset repo
    """
    print("\n" + "=" * 70)
    print("Downloading Stellar Spectra Foundation Models from HuggingFace")
    print("=" * 70)
    print(f"Repository: {HF_REPO_ID}")
    print(f"Local directory: {LOCAL_MODEL_DIR}")

    if include_test_data:
        print("\n⚠️  Test data will be downloaded (~2-5 GB)")
        print("   This may take several minutes depending on your connection.")

    # Download models (test data comes from separate repo)
    print("\nDownloading models...")
    try:
        snapshot_download(
            repo_id=HF_REPO_ID,
            local_dir=LOCAL_MODEL_DIR,
            local_dir_use_symlinks=False,
            resume_download=True,
            max_workers=4,
            ignore_patterns=["test_data/*"]  # Test data is in separate repo
        )
        print(f"\n✓ Models downloaded to: {LOCAL_MODEL_DIR}")
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        sys.exit(1)

    # Download test data from separate dataset repository if requested
    test_data_path = None
    if include_test_data:
        from download_test_data import download_test_data
        test_data_dir = Path(LOCAL_MODEL_DIR) / "test_data"
        test_data_path = download_test_data(
            local_dir=str(test_data_dir),
            filename="gaia_lamost_test_only.h5"
        )
        print(f"\n✓ Test data downloaded to: {test_data_path}")

    print("\n✓ Download complete!")
    return LOCAL_MODEL_DIR, test_data_path

def download_test_data_only():
    """
    Download only the test data file from separate dataset repository.
    Useful if you already have the models but need the test data.
    """
    print("\n" + "=" * 70)
    print("Downloading Test Data Only")
    print("=" * 70)

    # Use the separate download_test_data module
    from download_test_data import download_test_data

    test_data_dir = Path(LOCAL_MODEL_DIR) / "test_data"
    test_data_path = download_test_data(
        local_dir=str(test_data_dir),
        filename="gaia_lamost_test_only.h5"
    )

    return test_data_path

def generate_local_configs(model_dir, test_data_path=None):
    """
    Generate local configuration files.

    Args:
        model_dir: Path to downloaded models
        test_data_path: Optional path to test data HDF5 file
    """
    print("\nGenerating local configuration files...")

    config_path = Path(model_dir) / "config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    foundation = config['founation_models']

    # Configuration for LAMOST LRS
    lrs_config = {
        # Encoders
        'xp_encoder_path': str(Path(model_dir) / foundation['encoders']['xp_encoder_ae_768']['path']),
        'lrs_encoder_path': str(Path(model_dir) / foundation['encoders']['lrs_encoder']['path']),
        # Two specclip models
        'specclip_predrecon_path': str(Path(model_dir) / foundation['specclip_models']['specclip_model_predrecon_mlp']['path']),
        'specclip_split_path': str(Path(model_dir) / foundation['specclip_models']['specclip_model_split_mlp']['path']),
        # Individual SBI models for seismic parameters
        'sbi_model_path_age': str(Path(model_dir) / foundation['parameter_estimation']['lamost_lrs']['sbi']['age']['path']),
        'sbi_model_path_nu_max': str(Path(model_dir) / foundation['parameter_estimation']['lamost_lrs']['sbi']['nu_max']['path']),
        'sbi_model_path_mass': str(Path(model_dir) / foundation['parameter_estimation']['lamost_lrs']['sbi']['mass']['path']),
        'sbi_model_path_rad': str(Path(model_dir) / foundation['parameter_estimation']['lamost_lrs']['sbi']['rad']['path']),
        'sbi_model_path_dnu': str(Path(model_dir) / foundation['parameter_estimation']['lamost_lrs']['sbi']['dnu']['path']),
        'sbi_model_path_rv': str(Path(model_dir) / foundation['parameter_estimation']['lamost_lrs']['sbi']['rv']['path']),
        'sbi_model_path_dpi1': str(Path(model_dir) / foundation['parameter_estimation']['lamost_lrs']['sbi']['dpi1']['path']),
        'sbi_model_path_teff': str(Path(model_dir) / foundation['parameter_estimation']['lamost_lrs']['sbi']['teff']['path']),
        'sbi_model_path_logg': str(Path(model_dir) / foundation['parameter_estimation']['lamost_lrs']['sbi']['logg']['path']),
        # Individual MLP models for chemical abundances
        'mlp_model_path_feh': str(Path(model_dir) / foundation['parameter_estimation']['lamost_lrs']['mlp']['feh']['path']),
        'mlp_model_path_ebprp': str(Path(model_dir) / foundation['parameter_estimation']['lamost_lrs']['mlp']['ebprp']['path']),
        'mlp_model_path_afe': str(Path(model_dir) / foundation['parameter_estimation']['lamost_lrs']['mlp']['afe']['path']),
        'mlp_model_path_cfe': str(Path(model_dir) / foundation['parameter_estimation']['lamost_lrs']['mlp']['cfe']['path']),
        'mlp_model_path_nfe': str(Path(model_dir) / foundation['parameter_estimation']['lamost_lrs']['mlp']['nfe']['path']),
        'mlp_model_path_alfe': str(Path(model_dir) / foundation['parameter_estimation']['lamost_lrs']['mlp']['alfe']['path']),
        'mlp_model_path_mgfe': str(Path(model_dir) / foundation['parameter_estimation']['lamost_lrs']['mlp']['mgfe']['path']),
        'mlp_model_path_mnfe': str(Path(model_dir) / foundation['parameter_estimation']['lamost_lrs']['mlp']['mnfe']['path']),
        'mlp_model_path_nife': str(Path(model_dir) / foundation['parameter_estimation']['lamost_lrs']['mlp']['nife']['path']),
        'mlp_model_path_ofe': str(Path(model_dir) / foundation['parameter_estimation']['lamost_lrs']['mlp']['ofe']['path']),
        'mlp_model_path_sife': str(Path(model_dir) / foundation['parameter_estimation']['lamost_lrs']['mlp']['sife']['path']),
        'mlp_model_path_tife': str(Path(model_dir) / foundation['parameter_estimation']['lamost_lrs']['mlp']['tife']['path']),
        'mlp_model_path_crfe': str(Path(model_dir) / foundation['parameter_estimation']['lamost_lrs']['mlp']['crfe']['path']),
        'mlp_model_path_cafe': str(Path(model_dir) / foundation['parameter_estimation']['lamost_lrs']['mlp']['cafe']['path']),
        'stats_dir': str(Path(model_dir) / 'stats')
    }

    # Configuration for Gaia XP
    xp_config = {
        # Encoders
        'xp_encoder_path': str(Path(model_dir) / foundation['encoders']['xp_encoder_ae_768']['path']),
        'lrs_encoder_path': str(Path(model_dir) / foundation['encoders']['lrs_encoder']['path']),
        # Two specclip models
        'specclip_predrecon_path': str(Path(model_dir) / foundation['specclip_models']['specclip_model_predrecon_mlp']['path']),
        'specclip_split_path': str(Path(model_dir) / foundation['specclip_models']['specclip_model_split_mlp']['path']),
        # MLP models for Gaia XP parameters
        'mlp_model_path_afe': str(Path(model_dir) / foundation['parameter_estimation']['gaia_xp']['mlp']['afe']['path']),
        'mlp_model_path_cfe': str(Path(model_dir) / foundation['parameter_estimation']['gaia_xp']['mlp']['cfe']['path']),
        'mlp_model_path_nfe': str(Path(model_dir) / foundation['parameter_estimation']['gaia_xp']['mlp']['nfe']['path']),
        'mlp_model_path_feh': str(Path(model_dir) / foundation['parameter_estimation']['gaia_xp']['mlp']['feh']['path']),
        'mlp_model_path_ebprp': str(Path(model_dir) / foundation['parameter_estimation']['gaia_xp']['mlp']['ebprp']['path']),
        # SBI models for teff and logg
        'sbi_model_path_teff': str(Path(model_dir) / foundation['parameter_estimation']['sbi']['teff']['path']),
        'sbi_model_path_logg': str(Path(model_dir) / foundation['parameter_estimation']['sbi']['logg']['path']),
        'stats_dir': str(Path(model_dir) / 'stats')
    }

    # Configuration for retrieval (uses split model for shared embeddings)
    retrieval_config = {
        'specclip_predrecon_path': str(Path(model_dir) / foundation['specclip_models']['specclip_model_predrecon_mlp']['path']),
        'specclip_split_path': str(Path(model_dir) / foundation['specclip_models']['specclip_model_split_mlp']['path']),
        'xp_encoder_path': str(Path(model_dir) / foundation['encoders']['xp_encoder_ae_768']['path']),
        'lrs_encoder_path': str(Path(model_dir) / foundation['encoders']['lrs_encoder']['path']),
    }

    # Add test data path if available
    if test_data_path:
        retrieval_config['h5_data_path'] = str(test_data_path)
    elif 'test_data' in config:
        default_test_path = Path(model_dir) / "test_data" / "gaia_lamost_test_only.h5"
        if default_test_path.exists():
            retrieval_config['h5_data_path'] = str(default_test_path)

    # Save configurations
    with open('config_lrs.json', 'w') as f:
        json.dump(lrs_config, f, indent=2)
    print("✓ Saved: config_lrs.json")

    with open('config_xp.json', 'w') as f:
        json.dump(xp_config, f, indent=2)
    print("✓ Saved: config_xp.json")

    with open('config_retrieval.json', 'w') as f:
        json.dump(retrieval_config, f, indent=2)
    print("✓ Saved: config_retrieval.json")

    return lrs_config, xp_config, retrieval_config

def main():
    """Main setup function"""
    import argparse

    parser = argparse.ArgumentParser(description="Download stellar foundation models")
    parser.add_argument(
        "--include-test-data",
        action="store_true",
        help="Download test data (~2-5GB, required for retrieval)"
    )
    parser.add_argument(
        "--test-data-only",
        action="store_true",
        help="Download only test data (if models already downloaded)"
    )

    args = parser.parse_args()

    if args.test_data_only:
        # Download only test data
        test_data_path = download_test_data_only()

        # Update retrieval config
        if os.path.exists('config_retrieval.json'):
            with open('config_retrieval.json', 'r') as f:
                config = json.load(f)
            config['h5_data_path'] = test_data_path
            with open('config_retrieval.json', 'w') as f:
                json.dump(config, f, indent=2)
            print("\n✓ Updated config_retrieval.json with test data path")

    else:
        # Download models (and optionally test data)
        result = download_all_models(
            use_snapshot=True,
            include_test_data=args.include_test_data
        )

        model_dir = result[0] if isinstance(result, tuple) else result
        test_data_path = result[1] if isinstance(result, tuple) and len(result) > 1 else None

        # Generate configs
        lrs_config, xp_config, retrieval_config = generate_local_configs(
            model_dir,
            test_data_path
        )

    print("\n" + "=" * 70)
    print("Setup Complete!")
    print("=" * 70)

    if args.test_data_only:
        print("\n✓ Test data downloaded and configured")
    else:
        print(f"\nModels directory: {LOCAL_MODEL_DIR}")
        print("\nConfiguration files created:")
        print("  - config_lrs.json (LAMOST LRS parameters)")
        print("  - config_xp.json (Gaia XP parameters)")
        print("  - config_retrieval.json (spectral retrieval)")

        if args.include_test_data:
            print("\n✓ Test data included - ready for retrieval tasks")
        else:
            print("\n⚠️  Test data not downloaded")
            print("   For retrieval tasks, run with: --include-test-data")
            print("   Or download later with: --test-data-only")

if __name__ == "__main__":
    main()