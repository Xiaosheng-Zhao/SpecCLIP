"""
Functions for Gaia XP stellar parameter estimation
"""
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from specclip.models.specformer_control import SpectralMLPAutoencoder_xp as SpecFormer

import h5py
import torch.nn as nn
from typing import Dict, Tuple, List, Union, Any
import argparse
from scipy.interpolate import interp1d
from sbi.inference import SNPE
import pickle
from pathlib import Path
import pandas as pd
from astropy.io import fits
from scipy.signal import medfilt, savgol_filter

# For clip results
from specclip.models import SpecClipModel_reconstruct_embed768_mlp as SpecClipModel
from specclip.models.specclip_reconstruct_embed768_mlp import GaiaXPHeadWithMLP as GaiaXPHead
from specclip.models.specclip_reconstruct_embed768_mlp import LamostLRSHead

import time

import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

import warnings
from pytorch_lightning.utilities.warnings import PossibleUserWarning

warnings.filterwarnings('ignore', category=PossibleUserWarning, 
                       message='.*is already saved during checkpointing.*')

# Parameter to model mapping for Gaia XP
PREDRECON_XP_PARAMS = ["ebprp", "fe_h", "a_fe", "c_fe", "n_fe"]
SPLIT_XP_PARAMS = ["teff", "logg"]

# Configuration for which embedding type each model was trained with
SBI_MODEL_TRAINING_INFO_XP = {
    # MLP models trained with CLIP embeddings
    'a_fe': 'predrecon_clip',
    'c_fe': 'predrecon_clip',
    'n_fe': 'predrecon_clip',
    'fe_h': 'predrecon_clip',
    'ebprp': 'predrecon_clip',
    

    # trained with XP encoder embeddings
    'fe_h_pretrained': 'xp',
    
    # SBI models 
    # If trained with XP encoder:
    'teff': 'split_clip',
    'logg': 'split_clip',
}

def read_structure_fits(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read LAMOST LRS FITS file in structure format
    Args:
        file_path: Path to the FITS file
    Returns:
        wavelength: Wavelength array
        flux: Flux array
    """
    with fits.open(file_path) as hdul:
        data = hdul[1].data
        
        try:
            wavelength = data['WAVELENGTH']
            flux = data['FLUX']
        except KeyError:
            if 'WAVE' in data.names:
                wavelength = data['WAVE']
            elif 'LAMBDA' in data.names:
                wavelength = data['LAMBDA']
            
            if 'INTENSITY' in data.names:
                flux = data['INTENSITY']
            elif 'SPEC' in data.names:
                flux = data['SPEC']
            else:
                raise KeyError("Cannot find wavelength or flux data columns")
        
        return wavelength[0], flux[0]

def read_csv_spectrum(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read spectrum from CSV file adaptively handling headers.
    """
    try:
        data = pd.read_csv(file_path, header=None)
        if pd.to_numeric(data.iloc[0], errors='coerce').notnull().all():
            cond_len = len(data.iloc[0].values)
            if cond_len>10:
                wavelength = data.iloc[0].values
                flux = data.iloc[1].values
            elif cond_len>=2:
                wavelength = data.iloc[:,0].values
                flux = data.iloc[:,1].values
            return wavelength, flux
    except (pd.errors.ParserError, ValueError):
        pass
        
    try:
        data = pd.read_csv(file_path, header=None,delimiter=r'[,\s;\']+', engine='python')
        if pd.to_numeric(data.iloc[0], errors='coerce').notnull().all():
            wavelength = data.iloc[:,0].values
            flux = data.iloc[:,1].values
            return wavelength, flux
    except (pd.errors.ParserError, ValueError):
        pass

    try:
        data = pd.read_csv(file_path)
        if len(data.iloc[0].values)==1:
            data = pd.read_csv(file_path,delimiter=r'[,\s;\']+', engine='python')
        
        wavelength_cols = ['wavelength', 'WAVELENGTH', 'wave', 'WAVE', 'lambda', 'LAMBDA']
        flux_cols = ['flux', 'FLUX', 'intensity', 'INTENSITY', 'spec', 'SPEC']
        
        wavelength_col = None
        for col in wavelength_cols:
            if col in data.columns:
                wavelength_col = col
                break
                
        flux_col = None
        for col in flux_cols:
            if col in data.columns:
                flux_col = col
                break
                
        if wavelength_col is None or flux_col is None:
            wavelength = data.iloc[:, 0].values
            flux = data.iloc[:, 1].values
        else:
            wavelength = data[wavelength_col].values
            flux = data[flux_col].values
            
        return wavelength, flux
    except Exception as e:
        raise ValueError(f"Unable to read CSV file in either format: {str(e)}")

def read_matrix_fits(fits_path):
    """
    Extract wavelength and flux arrays from a LAMOST spectrum FITS file.
    """
    with fits.open(fits_path) as hdul:
        flux = hdul[0].data[0]
        coeff0 = hdul[0].header['COEFF0']
        coeff1 = hdul[0].header['COEFF1']
        naxis1 = hdul[0].header['NAXIS1']
        log_wave = coeff0 + coeff1 * np.arange(naxis1)
        wavelength = 10**log_wave
        return wavelength, flux
    
def read_spectrum(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Unified interface for reading spectrum files, automatically determines format
    """
    if file_path.endswith('.fits') or file_path.endswith('.fit') or file_path.endswith('.fits.gz') or file_path.endswith('.fit.gz'):
        try:
            with fits.open(file_path) as hdul:
                if isinstance(hdul[0].data, np.ndarray):
                    return read_matrix_fits(file_path)
                elif len(hdul) > 1 and hasattr(hdul[1], 'data'):
                    if isinstance(hdul[1].data, np.ndarray):
                        return read_structure_fits(file_path)
        except Exception as e:
            raise ValueError(f"Cannot read FITS file: {str(e)}")
    elif file_path.endswith('.csv') or file_path.endswith('.txt'):
        return read_csv_spectrum(file_path)
    else:
        raise ValueError("Unsupported file format")

def load_spectrum_data(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load spectrum data from different file formats.
    """
    wavelength, flux = read_spectrum(file_path)
    return wavelength, flux

def interpolate_spectrum(wavelength: np.ndarray,
                        flux: np.ndarray,
                        new_wavelength_range: List[float] = [336, 1020],
                        num_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate spectrum onto a wavelength grid for Gaia XP.
    
    Args:
        wavelength: Input wavelength array
        flux: Input flux array
        new_wavelength_range: Target [min, max] wavelength range
        num_points: Number of points in output arrays
        
    Returns:
        Tuple of (new_wavelength, new_flux) arrays
    """
    if len(wavelength) == 0 or len(flux) == 0:
        raise ValueError("Input arrays cannot be empty")
        
    if len(wavelength) != len(flux):
        raise ValueError(f"Length mismatch: wavelength ({len(wavelength)}) != flux ({len(flux)})")
        
    if np.any(wavelength <= 0):
        raise ValueError("Wavelength values must be positive")
        
    if np.any(~np.isfinite(wavelength)) or np.any(~np.isfinite(flux)):
        raise ValueError("Input arrays contain inf or NaN values")
        
    if not np.all(np.diff(wavelength) > 0):
        raise ValueError("Wavelength array must be strictly increasing")
    
    if wavelength.min() > new_wavelength_range[0] or wavelength.max() < new_wavelength_range[1]:
        raise ValueError(
            f"Target wavelength range {new_wavelength_range} outside of data range"
            f" [{wavelength.min():.2f}, {wavelength.max():.2f}]"
        )
    
    try:
        f = interp1d(wavelength, flux, kind='linear', bounds_error=True)
        new_wave = np.arange(336, 1021, 2)
        new_flux = f(new_wave)
        new_flux /= new_flux[107]  # normalize at 5500 A
        return new_wave, new_flux
    except ValueError as e:
        raise ValueError(f"Interpolation failed: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error during interpolation: {str(e)}")

class ParamRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list = [1024, 512, 64]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

def predict_sbi_parameters_xp(
    model,  # can be LRS or CLIP
    sbi_model_path: str,
    parameters: List[str],
    error_parameter_name: List[str],
    wavelength: np.ndarray,
    flux: np.ndarray,
    device: torch.device,
    num_samples: int = 10000,
    use_clip: bool = False,
    use_split_clip: bool = False,  # NEW: flag for split model
) -> pd.DataFrame:
    """
    Predict parameters using SBI model for a single spectrum
    
    Args:
        model: Pre-trained model (LRS SpecFormer or CLIP model)
        sbi_model_path: Path to saved SBI model
        parameters: List of parameter names to predict
        error_parameter_name: List of error column names
        wavelength: Wavelength array
        flux: Flux array
        device: Computation device
        num_samples: Number of samples for posterior
        use_clip: Use CLIP model (predrecon)
        use_split_clip: Use split CLIP model
        
    Returns:
        DataFrame containing predictions and uncertainties
    """
    # Load SBI model and parameters
    with open(sbi_model_path, 'rb') as f:
        saved_dict = pickle.load(f)
    
    inference = saved_dict['inference']
    if 'mean' in saved_dict:
        means = torch.as_tensor(saved_dict['mean'], device=device)
        stds = torch.as_tensor(saved_dict['std'], device=device)
    else:
        means = torch.as_tensor(saved_dict['means'], device=device)
        stds = torch.as_tensor(saved_dict['stds'], device=device)

    posterior = inference.build_posterior()
    
    # Interpolate spectrum
    new_wavelength, new_flux = interpolate_spectrum(wavelength, flux, num_points=343)
    
    # Convert to tensor
    flux_tensor = torch.tensor(new_flux, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-1)
    wavelength_tensor = torch.tensor(new_wavelength, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-1)
    
    # Get embeddings based on model type
    model.eval()
    with torch.no_grad():
        if use_split_clip:
            # Use split CLIP model - concatenate shared+private
            shared, private = model(flux_tensor, input_type='gaia_spectra')
            embeddings = torch.cat([shared, private], dim=-1)
            embeddings = embeddings.reshape(1,768)
        elif use_clip:
            # Use predrecon CLIP model - unified 768-dim
            outputs = model(flux_tensor, input_type='gaia_spectra')
            embeddings = outputs.reshape(1, 768)
        else:
            # Use LRS SpecFormer model
            outputs = model(flux_tensor)
            embeddings = torch.mean(outputs['embedding'], -2)
    
    # Get predictions from SBI model
    samples = posterior.sample((num_samples,), x=embeddings[0].cpu())
    samples = torch.as_tensor(samples, device=device)
    
    # Unnormalize samples
    samples = samples * stds + means
    
    # Calculate statistics
    results_dict = {}
    for i, parameter in enumerate(parameters):
        parameter_samples = samples[:, i].cpu().numpy()
        results_dict[parameter] = ["{:.2f}".format(float(np.median(parameter_samples)))]
        results_dict[error_parameter_name[i]] = ["{:.2f}".format(float(np.std(parameter_samples)))]
    
    return pd.DataFrame(results_dict)
    
def predict_mlp_parameters(
    model,  # can be XP encoder or CLIP
    mlp_model: nn.Module,
    parameters: List[str],
    error_parameter_name: List[str],
    flux: np.ndarray,
    wavelength: np.ndarray,
    error: np.ndarray,
    device: torch.device,
    train_Teff: bool = False,
    use_clip: bool = False,
    use_split_clip: bool = False,  # NEW
    stats: Dict[str, List[float]]=None
) -> pd.DataFrame:
    """
    Predict parameter for a single spectrum using MLP model
    
    Args:
        model: Pre-trained model (XP SpecFormer or CLIP model)
        mlp_model: Pre-trained MLP model
        parameters: List of parameter names to predict
        error_parameter_name: List of error column names
        flux: Flux array
        wavelength: Wavelength array
        error: Error array (can be None)
        device: Computation device
        train_Teff: Whether this is Teff (needs special scaling)
        use_clip: Use CLIP model (predrecon)
        use_split_clip: Use split CLIP model
        stats: Normalization statistics
        
    Returns:
        DataFrame containing predictions
    """
    new_wavelength, new_flux = interpolate_spectrum(wavelength, flux, num_points=343)
    
    flux_tensor = torch.tensor(new_flux, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-1)
    wavelength_tensor = torch.tensor(new_wavelength, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-1)
    
    model.eval()
    with torch.no_grad():
        if use_split_clip:
            # Use split CLIP model - concatenate shared+private
            shared, private = model(flux_tensor, input_type='gaia_spectra')
            embeddings = torch.cat([shared, private], dim=-1)
            embeddings = embeddings.reshape(1,768)
        elif use_clip:
            # Use predrecon CLIP model - unified 768-dim
            outputs = model(flux_tensor, input_type='gaia_spectra')
            embeddings = outputs.reshape(1, 768)
        else:
            # Use XP SpecFormer model
            outputs = model(flux_tensor)
            embeddings = torch.mean(outputs['embedding'], -2)
        
        # Apply normalization if provided
        if stats is not None:
            mean = torch.as_tensor(stats['mean'], device=device).unsqueeze(0)
            std = torch.as_tensor(stats['std'], device=device).unsqueeze(0)
            embeddings = (embeddings - mean) / std
    
    mlp_model.eval()
    with torch.no_grad():
        prediction = mlp_model(embeddings)

    results_dict = {}
    
    for i, parameter in enumerate(parameters):
        parameter_prediction = prediction[i, 0].cpu().numpy()
        if train_Teff:
            teff_mean = 6777.3634508000005
            teff_std = 2404.0058499580778
            parameter_prediction *= teff_std
            parameter_prediction += teff_mean

        results_dict[parameter] = ["{:.2f}".format(float(parameter_prediction))]
    
    predictions_df = pd.DataFrame(results_dict)
    return predictions_df

def load_mlp_model(model_path: str, device: torch.device, input_dim: int = 768, 
                   hidden_dims: list = [768, 128, 32], output_dim: int = 1) -> nn.Module:
    """Load a saved MLP model from a checkpoint"""
    model = ParamRegressor(input_dim, hidden_dims).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def normalize_features_known_mean_std(features, mean, std):
    return (features - mean) / std

def initialize_encoders(
    xp_encoder_path: str,
    lrs_encoder_path: str,
    embed_dim: int = 768,
    n_head: int = 4,
    model_embed_dim: int = 768,
    dropout: float = 0.1,
    freeze_backbone: bool = True
) -> tuple[nn.Module, nn.Module]:
    """Initialize both encoders with the same configuration as during training."""
    gaia_xp_encoder = GaiaXPHead(
        model_path=xp_encoder_path,
        embed_dim=embed_dim,
        n_head=n_head,
        model_embed_dim=model_embed_dim,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
        load_pretrained_weights=False
    )

    lamost_lrs_encoder = LamostLRSHead(
        model_path=lrs_encoder_path,
        embed_dim=embed_dim,
        n_head=n_head,
        model_embed_dim=model_embed_dim,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
        load_pretrained_weights=False
    )

    return gaia_xp_encoder, lamost_lrs_encoder


def initialize_encoders_split(
    xp_encoder_path: str,
    lrs_encoder_path: str,
    shared_embed_dim: int = 512,
    private_embed_dim: int = 256,
    n_head: int = 4,
    model_embed_dim: int = 768,
    dropout: float = 0.1,
    freeze_backbone: bool = True
) -> tuple[nn.Module, nn.Module]:
    """Initialize both encoders with split architecture."""
    from specclip.models.specclip_reconstruct_split_5122562_mlp_recordloss import (
        GaiaXPHead_split, LamostLRSHead_split
    )

    gaia_xp_encoder = GaiaXPHead_split(
        model_path=xp_encoder_path,
        shared_embed_dim=shared_embed_dim,
        private_embed_dim=private_embed_dim,
        n_head=n_head,
        model_embed_dim=model_embed_dim,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
        load_pretrained_weights=False
    )

    lamost_lrs_encoder = LamostLRSHead_split(
        model_path=lrs_encoder_path,
        shared_embed_dim=shared_embed_dim,
        private_embed_dim=private_embed_dim,
        n_head=n_head,
        model_embed_dim=model_embed_dim,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
        load_pretrained_weights=False
    )

    return gaia_xp_encoder, lamost_lrs_encoder


class StellarParameterPredictorXP:
    """
    A class for predicting stellar parameters from Gaia XP spectra using pre-trained models.
    """
    
    def __init__(self, model_paths_dict=None):
        """Initialize with paths to all models."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        if model_paths_dict is None:
            model_paths_dict = self._get_default_paths()
        
        self.model_paths = model_paths_dict
        
        print("Loading models...")
        self._load_models()
        print("Models loaded successfully!")
        
    def _get_default_paths(self):
        """Return default model paths"""
        return {
            'xp_encoder_path': "/work/zxs/model/pretrained_models/epoch=191-val_loss=0.0000_mt_xp.ckpt",
            'lrs_encoder_path': "/work/zxs/model/pretrained_models/epoch=128-val_loss=0.0000_mt_lrs.ckpt",
            'specclip_predrecon_path': '/work/zxs/model/pretrained_models/specclip_model_predrecon_mlp.ckpt',
            'specclip_split_path': '/work/zxs/model/pretrained_models/specclip_model_split_mlp.ckpt',
            'mlp_model_path_afe': './downstream_models/xp/mlp/afe.pt',
            'mlp_model_path_cfe': './downstream_models/xp/mlp/cfe.pt',
            'mlp_model_path_nfe': './downstream_models/xp/mlp/nfe.pt',
            'mlp_model_path_feh': './downstream_models/xp/mlp/feh.pt',
            'mlp_model_path_feh_clip': './downstream_models/xp/mlp/feh_clip.pt',
            'mlp_model_path_ebprp': './downstream_models/xp/mlp/ebprp.pt',
            'sbi_model_path_teff': './downstream_models/xp/sbi/teff.pkl',
            'sbi_model_path_logg': './downstream_models/xp/sbi/logg.pkl',
            'stats_dir': './stats/',
        }
    
    def _load_models(self):
        """Load all pre-trained models"""
        # Load XP encoder
        self.xp_model = SpecFormer.load_from_checkpoint(
            self.model_paths['xp_encoder_path']
        )
        self.xp_model = self.xp_model.to(self.device)
        self.xp_model.eval()
        
        # Load predrecon CLIP model
        gaia_xp_encoder, lamost_lrs_encoder = initialize_encoders(
            xp_encoder_path=self.model_paths['xp_encoder_path'],
            lrs_encoder_path=self.model_paths['lrs_encoder_path'],
        )
        
        self.clip_model_predrecon = SpecClipModel.load_from_checkpoint(
            checkpoint_path=self.model_paths['specclip_predrecon_path'],
            gaia_xp_encoder=gaia_xp_encoder,
            lamost_lrs_encoder=lamost_lrs_encoder,
            strict=True
        )
        self.clip_model_predrecon = self.clip_model_predrecon.to(self.device)
        self.clip_model_predrecon.eval()
        
        # Legacy alias
        self.clip_model = self.clip_model_predrecon
        
        # Load split CLIP model if available
        if 'specclip_split_path' in self.model_paths:
            from specclip.models.specclip_reconstruct_split_5122562_mlp_recordloss import (
                SpecClipModel_reconstruct_split_5122562_mlp_recordloss as SpecClipModel_split
            )
            gaia_xp_encoder_split, lamost_lrs_encoder_split = initialize_encoders_split(
                xp_encoder_path=self.model_paths['xp_encoder_path'],
                lrs_encoder_path=self.model_paths['lrs_encoder_path'],
            )
            self.clip_model_split = SpecClipModel_split.load_from_checkpoint(
                checkpoint_path=self.model_paths['specclip_split_path'],
                gaia_xp_encoder=gaia_xp_encoder_split,
                lamost_lrs_encoder=lamost_lrs_encoder_split,
                strict=True
            )
            self.clip_model_split = self.clip_model_split.to(self.device)
            self.clip_model_split.eval()
        else:
            self.clip_model_split = None
        
        # Load MLP models
        self.mlp_model_afe = load_mlp_model(
            self.model_paths['mlp_model_path_afe'], 
            self.device, hidden_dims=[1024, 512, 64]
        )
        self.mlp_model_afe.eval()
        
        self.mlp_model_cfe = load_mlp_model(
            self.model_paths['mlp_model_path_cfe'], 
            self.device, hidden_dims=[1024, 512, 64]
        )
        self.mlp_model_cfe.eval()
        
        self.mlp_model_nfe = load_mlp_model(
            self.model_paths['mlp_model_path_nfe'], 
            self.device, hidden_dims=[1024, 512, 64]
        )
        self.mlp_model_nfe.eval()
        
        #self.mlp_model_teff = load_mlp_model(
        #    self.model_paths['sbi_model_path_teff'], 
        #    self.device, hidden_dims=[1024, 512, 64]
        #)
        #self.mlp_model_teff.eval()
        
        #self.mlp_model_logg = load_mlp_model(
        #    self.model_paths['sbi_model_path_logg'], 
        #    self.device, hidden_dims=[1024, 512, 64]
        #)
        #self.mlp_model_logg.eval()
        
        self.mlp_model_feh = load_mlp_model(
            self.model_paths['mlp_model_path_feh'], 
            self.device, hidden_dims=[1024, 512, 64]
        )
        self.mlp_model_feh.eval()
        
        
        self.mlp_model_ebprp = load_mlp_model(
            self.model_paths['mlp_model_path_ebprp'], 
            self.device, hidden_dims=[1024, 512, 64]
        )
        self.mlp_model_ebprp.eval()
        
        self.stats_dir = self.model_paths['stats_dir']
    
    def _get_model_for_parameter(self, param_name):
        """
        Determine which model to use based on how the SBI/MLP was trained
        
        Returns:
            tuple: (model, use_clip, use_split_clip)
        """
        training_type = SBI_MODEL_TRAINING_INFO_XP.get(param_name, 'xp')
        
        if training_type == 'split_clip':
            if self.clip_model_split is None:
                raise ValueError(f"Split CLIP model required for {param_name} but not loaded")
            return self.clip_model_split, True, True
        elif training_type == 'predrecon_clip':
            return self.clip_model_predrecon, True, False
        else:  # 'xp'
            return self.xp_model, False, False
    
    def predict(self, spectrum_file_path, parameter_types=['all'], 
                simple_header=True, return_dataframe=True, display_format='row'):
        """Main prediction function for stellar parameters."""
        if not os.path.exists(spectrum_file_path):
            raise FileNotFoundError(f"Spectrum file not found: {spectrum_file_path}")
        
        try:
            wavelength, flux = load_spectrum_data(spectrum_file_path)
        except Exception as e:
            raise ValueError(f"Error loading spectrum: {str(e)}")
        
        if isinstance(parameter_types, str):
            parameter_types = [parameter_types]
        
        if 'all' in parameter_types:
            parameter_types = ['afe', 'cfe', 'nfe', 'atmospheric', 'ebprp']
        
        prediction_rows = []
        
        if 'afe' in parameter_types:
            try:
                predictions_afe = self._predict_afe_rowwise(wavelength, flux, simple_header)
                prediction_rows.extend(predictions_afe)
            except Exception as e:
                print(f"Warning: [α/Fe] prediction failed: {str(e)}")
        
        if 'cfe' in parameter_types:
            try:
                predictions_cfe = self._predict_cfe_rowwise(wavelength, flux, simple_header)
                prediction_rows.extend(predictions_cfe)
            except Exception as e:
                print(f"Warning: [C/Fe] prediction failed: {str(e)}")
        
        if 'nfe' in parameter_types:
            try:
                predictions_nfe = self._predict_nfe_rowwise(wavelength, flux, simple_header)
                prediction_rows.extend(predictions_nfe)
            except Exception as e:
                print(f"Warning: [N/Fe] prediction failed: {str(e)}")
        
        if 'atmospheric' in parameter_types:
            try:
                predictions_teff = self._predict_teff_rowwise(wavelength, flux, simple_header)
                predictions_logg = self._predict_logg_rowwise(wavelength, flux, simple_header)
                predictions_feh = self._predict_feh_rowwise(wavelength, flux, simple_header)
                prediction_rows.extend(predictions_teff)
                prediction_rows.extend(predictions_logg)
                prediction_rows.extend(predictions_feh)
            except Exception as e:
                print(f"Warning: Atmospheric parameter prediction failed: {str(e)}")
        
        if 'ebprp' in parameter_types:
            try:
                predictions_ebprp = self._predict_ebprp_rowwise(wavelength, flux, simple_header)
                prediction_rows.extend(predictions_ebprp)
            except Exception as e:
                print(f"Warning: E(BP-RP) prediction failed: {str(e)}")
        
        if len(prediction_rows) == 0:
            raise ValueError("No predictions were successfully made")
        
        if display_format == 'row':
            df = pd.DataFrame(prediction_rows)
            cols = ['Parameter']
            if 'Prediction' in df.columns:
                cols.append('Prediction')
            if 'Error' in df.columns:
                cols.append('Error')
            if 'Method' in df.columns:
                cols.append('Method')
            df = df[cols]
        else:
            df = pd.DataFrame(prediction_rows)
        
        if return_dataframe:
            return df
        else:
            return df.to_dict(orient='records')

    def _predict_afe_rowwise(self, wavelength, flux, simple_header):
        """Predict [α/Fe] and return as row"""
        param_name = 'a_fe'
        param_display = '[α/Fe]' if not simple_header else param_name
        
        # Get correct model
        encoder_model, use_clip, use_split_clip = self._get_model_for_parameter(param_name)
        
        with open(os.path.join(self.stats_dir, 'xp/afe_mean_std.json'), 'r') as f:
            stats = json.load(f)
        
        predictions_df = predict_mlp_parameters(
            encoder_model,
            self.mlp_model_afe,
            [param_name],
            [f'{param_name}_err'],
            flux,
            wavelength,
            None,
            self.device,
            train_Teff=False,
            use_clip=use_clip,
            use_split_clip=use_split_clip,
            stats=stats
        )
        
        return [{
            'Parameter': param_display,
            'Prediction': predictions_df[param_name].values[0],
            'Method': 'MLP'
        }]
    
    def _predict_cfe_rowwise(self, wavelength, flux, simple_header):
        """Predict [C/Fe] and return as row"""
        param_name = 'c_fe'
        param_display = '[C/Fe]' if not simple_header else param_name
        
        encoder_model, use_clip, use_split_clip = self._get_model_for_parameter(param_name)
        
        with open(os.path.join(self.stats_dir, 'xp/cfe_mean_std.json'), 'r') as f:
            stats = json.load(f)
        
        predictions_df = predict_mlp_parameters(
            encoder_model,
            self.mlp_model_cfe,
            [param_name],
            [f'{param_name}_err'],
            flux,
            wavelength,
            None,
            self.device,
            train_Teff=False,
            use_clip=use_clip,
            use_split_clip=use_split_clip,
            stats=stats
        )
        
        return [{
            'Parameter': param_display,
            'Prediction': predictions_df[param_name].values[0],
            'Method': 'MLP'
        }]

    def _predict_nfe_rowwise(self, wavelength, flux, simple_header):
        """Predict [N/Fe] and return as row"""
        param_name = 'n_fe'
        param_display = '[N/Fe]' if not simple_header else param_name
        
        encoder_model, use_clip, use_split_clip = self._get_model_for_parameter(param_name)
        
        with open(os.path.join(self.stats_dir, 'xp/nfe_mean_std.json'), 'r') as f:
            stats = json.load(f)
        
        predictions_df = predict_mlp_parameters(
            encoder_model,
            self.mlp_model_nfe,
            [param_name],
            [f'{param_name}_err'],
            flux,
            wavelength,
            None,
            self.device,
            train_Teff=False,
            use_clip=use_clip,
            use_split_clip=use_split_clip,
            stats=stats
        )
        
        return [{
            'Parameter': param_display,
            'Prediction': predictions_df[param_name].values[0],
            'Method': 'MLP'
        }]

    def _predict_teff_rowwise(self, wavelength, flux, simple_header):
        """Predict Teff and return as row"""
        param_name = 'teff'
        param_display = 'T_eff [K]' if not simple_header else param_name
        
        # Get correct model for teff
        encoder_model, use_clip, use_split_clip = self._get_model_for_parameter(param_name)
        
        with open(os.path.join(self.stats_dir, 'xp/teff_mean_std.json'), 'r') as f:
            stats = json.load(f)
        
        predictions_df = predict_mlp_parameters(
            encoder_model,
            self.mlp_model_teff,
            [param_name],
            [f'{param_name}_err'],
            flux,
            wavelength,
            None,
            self.device,
            train_Teff=True,
            use_clip=use_clip,
            use_split_clip=use_split_clip,
            stats=stats
        )
        
        return [{
            'Parameter': param_display,
            'Prediction': predictions_df[param_name].values[0],
            'Method': 'MLP'
        }]

    def _predict_logg_rowwise(self, wavelength, flux, simple_header):
        """Predict log g and return as row"""
        param_name = 'logg'
        param_display = 'log g [cgs]' if not simple_header else param_name
        
        # Get correct model for logg
        encoder_model, use_clip, use_split_clip = self._get_model_for_parameter(param_name)
        
        with open(os.path.join(self.stats_dir, 'xp/logg_mean_std.json'), 'r') as f:
            stats = json.load(f)
        
        predictions_df = predict_mlp_parameters(
            encoder_model,
            self.mlp_model_logg,
            [param_name],
            [f'{param_name}_err'],
            flux,
            wavelength,
            None,
            self.device,
            train_Teff=False,
            use_clip=use_clip,
            use_split_clip=use_split_clip,
            stats=stats
        )
        
        return [{
            'Parameter': param_display,
            'Prediction': predictions_df[param_name].values[0],
            'Method': 'MLP'
        }]

    def _predict_teff_rowwise(self, wavelength, flux, simple_header):
        """Predict effective temperature and return as row"""
        param_name = 'teff'
        param_display = 'T_eff [K]' if not simple_header else param_name
        
        # Get correct model (should use split CLIP)
        encoder_model, use_clip, use_split_clip = self._get_model_for_parameter(param_name)
        
        predictions_df = predict_sbi_parameters_xp(
            encoder_model,
            self.model_paths['sbi_model_path_teff'],
            [param_name],
            [f'{param_name}_err'],
            wavelength,
            flux,
            self.device,
            use_clip=use_clip,
            use_split_clip=use_split_clip
        )
        
        return [{
            'Parameter': param_display,
            'Prediction': predictions_df[param_name].values[0],
            'Error': predictions_df[f'{param_name}_err'].values[0],
            'Method': 'SBI'
        }]

    def _predict_logg_rowwise(self, wavelength, flux, simple_header):
        """Predict surface gravity and return as row"""
        param_name = 'logg'
        param_display = 'log g [cgs]' if not simple_header else param_name
        
        # Get correct model (should use split CLIP)
        encoder_model, use_clip, use_split_clip = self._get_model_for_parameter(param_name)
        
        predictions_df = predict_sbi_parameters_xp(
            encoder_model,
            self.model_paths['sbi_model_path_logg'],
            [param_name],
            [f'{param_name}_err'],
            wavelength,
            flux,
            self.device,
            use_clip=use_clip,
            use_split_clip=use_split_clip
        )
        
        return [{
            'Parameter': param_display,
            'Prediction': predictions_df[param_name].values[0],
            'Error': predictions_df[f'{param_name}_err'].values[0],
            'Method': 'SBI'
        }]

    def _predict_feh_rowwise(self, wavelength, flux, simple_header):
        """Predict [Fe/H] and return as row"""
        param_name = 'fe_h'
        param_display = '[Fe/H] [dex]' if not simple_header else param_name
        
        encoder_model, use_clip, use_split_clip = self._get_model_for_parameter(param_name)
        
        with open(os.path.join(self.stats_dir, 'xp/feh_mean_std.json'), 'r') as f:
            stats = json.load(f)
        
        predictions_df = predict_mlp_parameters(
            encoder_model,
            self.mlp_model_feh,
            [param_name],
            [f'{param_name}_err'],
            flux,
            wavelength,
            None,
            self.device,
            train_Teff=False,
            use_clip=use_clip,
            use_split_clip=use_split_clip,
            stats=stats
        )
        
        return [{
            'Parameter': param_display,
            'Prediction': predictions_df[param_name].values[0],
            'Method': 'MLP'
        }]

    def _predict_ebprp_rowwise(self, wavelength, flux, simple_header):
        """Predict E(BP-RP) and return as row"""
        param_name = 'ebprp'
        param_display = 'E(BP-RP) [mag]' if not simple_header else 'e_bp_rp'
        
        encoder_model, use_clip, use_split_clip = self._get_model_for_parameter(param_name)
        
        with open(os.path.join(self.stats_dir, 'xp/ebprp_mean_std.json'), 'r') as f:
            stats = json.load(f)
        
        predictions_df = predict_mlp_parameters(
            encoder_model,
            self.mlp_model_ebprp,
            ['e_bp_rp'],
            ['e_bp_rp_err'],
            flux,
            wavelength,
            None,
            self.device,
            train_Teff=False,
            use_clip=use_clip,
            use_split_clip=use_split_clip,
            stats=stats
        )
        
        return [{
            'Parameter': param_display,
            'Prediction': predictions_df['e_bp_rp'].values[0],
            'Method': 'MLP'
        }]

    def predict_batch(self, spectrum_file_paths, **kwargs):
        """Predict parameters for multiple spectra."""
        results = []
        for i, file_path in enumerate(spectrum_file_paths):
            print(f"Processing {i+1}/{len(spectrum_file_paths)}: {file_path}")
            try:
                result = self.predict(file_path, **kwargs)
                results.append(result)
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                results.append(None)
        
        return results

    def display_results(self, predictions_df, style='default'):
        """Format predictions for nice display in Jupyter notebooks."""
        if style == 'formatted':
            def format_value(val):
                if pd.isna(val):
                    return '-'
                try:
                    float_val = float(val)
                    if abs(float_val) < 0.01 or abs(float_val) > 10000:
                        return f'{float_val:.2e}'
                    else:
                        return f'{float_val:.2f}'
                except:
                    return val
            
            styled = predictions_df.style.format({
                'Prediction': format_value
            }).set_properties(**{
                'text-align': 'left',
                'font-size': '11pt'
            }).set_table_styles([
                {'selector': 'th', 'props': [('text-align', 'center'), 
                                             ('font-weight', 'bold'),
                                             ('background-color', '#f0f0f0')]},
                {'selector': 'td', 'props': [('padding', '8px')]}
            ])
            
            return styled
            
        elif style == 'highlight':
            def highlight_method(row):
                if 'CLIP' in str(row['Parameter']):
                    return ['background-color: #fff2e6'] * len(row)
                else:
                    return ['background-color: #e6f2ff'] * len(row)
            
            return predictions_df.style.apply(highlight_method, axis=1).set_properties(**{
                'text-align': 'center',
                'font-size': '11pt'
            })
            
        elif style == 'minimal':
            return predictions_df.style.hide(axis='index').set_properties(**{
                'text-align': 'left'
            })
        else:
            return predictions_df.style.set_properties(**{
                'text-align': 'center'
            })


def get_default_config():
    """Get default configuration dictionary for StellarParameterPredictorXP."""
    return {
        'xp_encoder_path': "/work/zxs/model/pretrained_models/epoch=191-val_loss=0.0000_mt_xp.ckpt",
        'lrs_encoder_path': "/work/zxs/model/pretrained_models/epoch=128-val_loss=0.0000_mt_lrs.ckpt",
        'specclip_predrecon_path': '/work/zxs/model/pretrained_models/specclip_model_predrecon_mlp.ckpt',
        'specclip_split_path': '/work/zxs/model/pretrained_models/specclip_model_split_mlp.ckpt',
        'mlp_model_path_afe': './downstream_models/xp/mlp/afe.pt',
        'mlp_model_path_cfe': './downstream_models/xp/mlp/cfe.pt',
        'mlp_model_path_nfe': './downstream_models/xp/mlp/nfe.pt',
        'mlp_model_path_feh': './downstream_models/xp/mlp/feh.pt',
        'mlp_model_path_feh_clip': './downstream_models/xp/mlp/feh_clip.pt',
        'mlp_model_path_ebprp': './downstream_models/xp/mlp/ebprp.pt',
        'sbi_model_path_teff': './downstream_models/xp/sbi/teff.pkl',
        'sbi_model_path_logg': './downstream_models/xp/sbi/logg.pkl',
        'stats_dir': './stats/',
    }

if __name__ == "__main__":
    pass