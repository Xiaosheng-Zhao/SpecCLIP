import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from specclip.models.specformer_control import SpecFormerControl20_wstd as SpecFormer
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
#from specclip.models import SpecClipModel, GaiaXPHead, LamostLRSHead
from specclip.models import SpecClipModel_reconstruct_embed768_mlp as SpecClipModel
from specclip.models.specclip_reconstruct_embed768_mlp import GaiaXPHeadWithMLP as GaiaXPHead
from specclip.models.specclip_reconstruct_embed768_mlp import LamostLRSHead

import time

import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"
torch.backends.cudnn.benchmark = True

import warnings
from pytorch_lightning.utilities.warnings import PossibleUserWarning

# Suppress the specific warning about nn.Module in save_hyperparameters
warnings.filterwarnings('ignore', category=PossibleUserWarning, 
                       message='.*is already saved during checkpointing.*')

# Add constants
PREDRECON_LRS_PARAMS = ["ebprp", "a_fe", "c_fe", "n_fe", "mg_fe", "o_fe", "al_fe",
                        "si_fe", "ca_fe", "ti_fe", "mn_fe", "ni_fe", "cr_fe",
                        "dnu", "nu_max", "mass", "rad", "age", "dpi"]
SPLIT_LRS_PARAMS = ["teff", "logg", "rv", "fe_h"]

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
        # For structure format, data is usually in the first extension
        data = hdul[1].data
        
        # Get wavelength and flux columns
        try:
            wavelength = data['WAVELENGTH']
            flux = data['FLUX']
        except KeyError:
            # Try alternative column names
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
    Tries to read first as array data (no headers) then with headers if needed.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        wavelength: Wavelength array
        flux: Flux array
        
    Raises:
        ValueError: If unable to parse the CSV file in either format
    """
    # First try reading without header
    try: # two rows for .txt or .csv with comma; two columns w/ comma
        data = pd.read_csv(file_path, header=None)
        # Check if first row contains numeric data
        if pd.to_numeric(data.iloc[0], errors='coerce').notnull().all():
            cond_len = len(data.iloc[0].values)
            if cond_len>10: # at least 10 points
                wavelength = data.iloc[0].values
                flux = data.iloc[1].values
            elif cond_len>=2:
                wavelength = data.iloc[:,0].values
                flux = data.iloc[:,1].values
            return wavelength, flux
            
    except (pd.errors.ParserError, ValueError):
        pass
        
    try: # two columns w/o comma
        data = pd.read_csv(file_path, header=None,delimiter=r'[,\s;\']+', engine='python')
        if pd.to_numeric(data.iloc[0], errors='coerce').notnull().all():
            wavelength = data.iloc[:,0].values
            flux = data.iloc[:,1].values
            return wavelength, flux
        
    except (pd.errors.ParserError, ValueError):
        pass

    # If that fails, try reading with header; in columns
    try:
        data = pd.read_csv(file_path)
        if len(data.iloc[0].values)==1:
            data = pd.read_csv(file_path,delimiter=r'[,\s;\']+', engine='python')
        # Look for common column names
        wavelength_cols = ['wavelength', 'WAVELENGTH', 'wave', 'WAVE', 'lambda', 'LAMBDA']
        flux_cols = ['flux', 'FLUX', 'intensity', 'INTENSITY', 'spec', 'SPEC']
        
        # Find wavelength column
        wavelength_col = None
        for col in wavelength_cols:
            if col in data.columns:
                wavelength_col = col
                break
                
        # Find flux column    
        flux_col = None
        for col in flux_cols:
            if col in data.columns:
                flux_col = col
                break
                
        if wavelength_col is None or flux_col is None:
            # If no matching column names, try first two columns
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
    
    Args:
        fits_path: Path to the FITS file
        
    Returns:
        wavelength: Array of wavelengths in Angstroms
        flux: Array of flux values
    """
    with fits.open(fits_path) as hdul:
        # Get the flux data from HDU[0]
        # LAMOST spectra have 6 rows where row 0 is flux
        flux = hdul[0].data[0]
        
        # Get wavelength solution from header
        coeff0 = hdul[0].header['COEFF0']  # log10(wavelength) of first pixel
        coeff1 = hdul[0].header['COEFF1']  # log10(wavelength) step per pixel
        naxis1 = hdul[0].header['NAXIS1']  # number of pixels
        
        # Generate log10(wavelength) array
        log_wave = coeff0 + coeff1 * np.arange(naxis1)
        
        # Convert to wavelength in Angstroms
        wavelength = 10**log_wave
        
        # Verify wavelength range matches metadata
        wavemin = hdul[1].data['WAVEMIN'][0]
        wavemax = hdul[1].data['WAVEMAX'][0]
        #print(f"Header wavelength range: {wavemin:.2f} - {wavemax:.2f} Å")
        #print(f"Computed wavelength range: {wavelength[0]:.2f} - {wavelength[-1]:.2f} Å")
        
        return wavelength, flux
    
def read_spectrum(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Unified interface for reading spectrum files, automatically determines format
    Args:
        file_path: Path to the file
    Returns:
        wavelength: Wavelength array
        flux: Flux array (for multi-exposure data, returns the first exposure)
    """
    if file_path.endswith('.fits') or file_path.endswith('.fit') or file_path.endswith('.fits.gz') or file_path.endswith('.fit.gz'):
        # Try to read FITS file
        try:
            with fits.open(file_path) as hdul:
                if isinstance(hdul[0].data, np.ndarray):
                    # Matrix format
                    return read_matrix_fits(file_path)
                elif len(hdul) > 1 and hasattr(hdul[1], 'data'):
                    if isinstance(hdul[1].data, np.ndarray):
                        # Structure format
                        return read_structure_fits(file_path)
                    
        except Exception as e:
            raise ValueError(f"Cannot read FITS file: {str(e)}")
    elif file_path.endswith('.csv') or file_path.endswith('.txt'):
        # CSV or txt format
        return read_csv_spectrum(file_path)
    else:
        raise ValueError("Unsupported file format")

# Normalization of the spectrum
def gaspp_fitcont2(ww, ff, cfsnr=60):
    """
    Fit continuum over 3850 - 9000 A region.

    Parameters:
    ww : array_like
        Wavelength array.
    ff : array_like
        Flux array.
    cfsnr : float
        Average signal-to-noise ratio in the spectrum.

    Returns:
    cc : ndarray
        Continuum flux array.
    orflx : ndarray
        Original flux array.
    """
    # Keep a copy of the original flux
    orflx = ff.copy()

    # **Convert ff to a supported data type with native byte order**
    ff = ff.astype(np.float64)

    # Smooth the flux with a median filter of size 7
    ff = medfilt(ff, kernel_size=7)

    # Copy ww and ff to ww100 and ff100
    ww100 = ww.copy()
    ff100 = ff.copy()

    # Define wavelength ranges and indices
    wran1ind = np.where((ww100 <= 5700.0) & (ww100 >= 3700.0))[0]
    nran1 = len(wran1ind)
    wran2ind = np.where(ww100 > 6100.0)[0]
    nran2 = len(wran2ind)

    # Prepare wavelength and flux arrays over certain ranges
    if nran1 > 100:
        wran1 = ww100[wran1ind]
        fran1 = ff100[wran1ind]
    else:
        wran1 = ww100
        fran1 = ff100

    if nran2 > 100:
        wran2 = ww100[wran2ind]
        fran2 = ff100[wran2ind]
    else:
        wran2 = ww100
        fran2 = ff100

    # Depending on cfsnr, set nl and apply Savitzky-Golay filter
    if cfsnr < 50:
        nl = int(-1.0 * cfsnr + 55)
        # Ensure nl is odd and at least 3
        if nl % 2 == 0:
            nl += 1
        if nl < 3:
            nl = 3
        ofl = savgol_filter(fran1, window_length=nl, polyorder=4, mode='nearest')
        ofl2 = savgol_filter(fran2, window_length=nl, polyorder=4, mode='nearest')
    else:
        ofl = savgol_filter(fran1, window_length=3, polyorder=2, mode='nearest')
        ofl2 = savgol_filter(fran2, window_length=3, polyorder=2, mode='nearest')

    # For wran1, perform initial polynomial fit and adjustments
    if nran1 > 100:
        # Initial polynomial fit to ofl
        coef0 = np.polyfit(wran1, ofl, 5)
        cfit0 = np.polyval(coef0, wran1)
        # Find the index where cfit0 is maximum
        flxind = np.argmax(cfit0)
        if wran1[flxind] < 4500:
            w11, w12 = 4030, 4160
            w21, w22 = 4270, 4410
            w31, w32 = 4800, 4940
            wbin = 10
            # Indices for specific wavelength ranges
            wind1ind = np.where((ww >= w11) & (ww <= w12))[0]
            wind2ind = np.where((ww >= w21) & (ww <= w22))[0]
            wind3ind = np.where((ww >= w31) & (ww <= w32))[0]
            # Indices for interpolation points
            indw11 = np.where((ww >= w11 - wbin) & (ww <= w11 + wbin))[0]
            indw12 = np.where((ww >= w12 - wbin) & (ww <= w12 + wbin))[0]
            indw21 = np.where((ww >= w21 - wbin) & (ww <= w21 + wbin))[0]
            indw22 = np.where((ww >= w22 - wbin) & (ww <= w22 + wbin))[0]
            indw31 = np.where((ww >= w31 - wbin) & (ww <= w31 + wbin))[0]
            indw32 = np.where((ww >= w32 - wbin) & (ww <= w32 + wbin))[0]
            # Get maximum fluxes at those points
            f11 = fran1[indw11].max() if len(indw11) > 0 else None
            f12 = fran1[indw12].max() if len(indw12) > 0 else None
            f21 = fran1[indw21].max() if len(indw21) > 0 else None
            f22 = fran1[indw22].max() if len(indw22) > 0 else None
            f31 = fran1[indw31].max() if len(indw31) > 0 else None
            f32 = fran1[indw32].max() if len(indw32) > 0 else None
            # Perform interpolation
            if f11 is not None and f12 is not None and len(wind1ind) > 0:
                fran1[wind1ind] = np.interp(ww[wind1ind], [w11, w12], [f11, f12])
            if f21 is not None and f22 is not None and len(wind2ind) > 0:
                fran1[wind2ind] = np.interp(ww[wind2ind], [w21, w22], [f21, f22])
            if f31 is not None and f32 is not None and len(wind3ind) > 0:
                fran1[wind3ind] = np.interp(ww[wind3ind], [w31, w32], [f31, f32])
        # Iteratively fit polynomial and adjust 'ofl'
        ofl = fran1.copy()
        for _ in range(10):
            coef = np.polyfit(wran1, ofl, 5)
            cfit = np.polyval(coef, wran1)
            ofl = np.maximum(cfit, ofl)
        # Final fit
        coef_tot = np.polyfit(wran1, ofl, 5)
        cfit1 = np.polyval(coef_tot, wran1)
    else:
        cfit1 = None

    # For wran2, perform iterative fitting with conditions
    n1000 = 0
    if nran2 > 100:
        while n1000 <= 8:
            coef2 = np.polyfit(wran2, ofl2, 4)
            cfit2 = np.polyval(coef2, wran2)
            ysig2 = np.std(ofl2 - cfit2)
            # Adjust ofl2 where condition is met
            mask = (ofl2 < cfit2) | (ofl2 > cfit2 + 3.0 * ysig2)
            ofl2[mask] = cfit2[mask]
            n1000 += 1
    else:
        cfit2 = None

    # Assemble total wavelength and continuum fit
    totwran = ww.copy()
    totcfit = ff.copy()
    if nran1 > 100:
        totcfit[wran1ind] = cfit1
    if nran2 > 100:
        totcfit[wran2ind] = cfit2
    # Replace non-positive values in totcfit with 1.0
    totcfit[totcfit <= 0.0] = 1.0

    cc = totcfit
    # Return the continuum and original flux
    return cc

def load_spectrum_data(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load spectrum data from different file formats.
    
    Args:
        file_path: Path to input file (.npy, .csv, .fits, etc.)
        
    Returns:
        Tuple of (wavelength, flux) arrays
    """
    wavelength, flux = read_spectrum(file_path)
    # Apply GASPP normalization
    continuum_gaspp = gaspp_fitcont2(wavelength, flux)
    flux_normed = flux / continuum_gaspp
        
    return wavelength, flux_normed

def interpolate_spectrum(wavelength: np.ndarray,
                        flux: np.ndarray,
                        new_wavelength_range: List[float] = [4000, 5600],
                        num_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate spectrum onto a logarithmically-spaced wavelength grid.
    
    Args:
        wavelength: Input wavelength array, must be strictly increasing
        flux: Input flux array, must have same length as wavelength
        new_wavelength_range: Target [min, max] wavelength range for interpolation
        num_points: Number of points in output arrays
        
    Returns:
        Tuple of (new_wavelength, new_flux) arrays
        
    Raises:
        ValueError: If inputs are invalid or wavelength range is outside data
        RuntimeError: If interpolation fails
    """
    
    # Input validation
    if len(wavelength) == 0 or len(flux) == 0:
        raise ValueError("Input arrays cannot be empty")
        
    if len(wavelength) != len(flux):
        raise ValueError(f"Length mismatch: wavelength ({len(wavelength)}) != flux ({len(flux)})")
        
    if np.any(wavelength <= 0):
        raise ValueError("Wavelength values must be positive for log interpolation")
        
    if np.any(~np.isfinite(wavelength)) or np.any(~np.isfinite(flux)):
        raise ValueError("Input arrays contain inf or NaN values")
        
    # Check if wavelength is strictly increasing
    if not np.all(np.diff(wavelength) > 0):
        raise ValueError("Wavelength array must be strictly increasing")
    
    # Validate wavelength range
    if wavelength.min() > new_wavelength_range[0] or wavelength.max() < new_wavelength_range[1]:
        raise ValueError(
            f"Target wavelength range {new_wavelength_range} outside of data range"
            f" [{wavelength.min():.2f}, {wavelength.max():.2f}]"
        )
    
    try:
        # Convert to log space once
        log_wave = np.log10(wavelength)
        
        # Create interpolation function
        f = interp1d(log_wave, flux, kind='linear', bounds_error=True)
        
        # Create new wavelength grid in log space
        #new_log_wave = np.linspace(
        #    np.log10(new_wavelength_range[0]),
        #    np.log10(new_wavelength_range[1]), 
        #    num_points
        #)
        w_start = 3.602 # Starting point accounting for intervals
        w_end = w_start + 1e-4 * 1461
        new_log_wave = np.linspace(w_start, w_end, 1462)
        
        # Interpolate flux
        new_flux = f(new_log_wave)
        
        # Convert wavelength back from log space
        new_wavelength = 10.0 ** new_log_wave
        
        return new_wavelength, new_flux
        
    except ValueError as e:
        raise ValueError(f"Interpolation failed: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error during interpolation: {str(e)}")

class ParamRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list = [768, 256, 64]):
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
        #print (x.shape)
        return self.network(x)

def load_mlp_model(model_path: str, device: torch.device, input_dim: int = 768, hidden_dims: list = [768, 128, 32], output_dim: int = 1) -> nn.Module:
    """
    Load a saved MLP model from a checkpoint
    
    Args:
        model_path: Path to the saved model checkpoint
        device: Device to load the model on
        input_dim: Input dimension of the MLP
        hidden_dim: Hidden dimension of the MLP
        output_dim: Output dimension of the MLP
        
    Returns:
        Loaded MLP model
    """
    # Initialize model architecture
    model = ParamRegressor(input_dim, hidden_dims).to(device)
    
    # Load the saved state
    checkpoint = torch.load(model_path, map_location=device)
    
    # Load the state dict into the model
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model

# normalize features with known mean and std
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

    # Initialize image encoder
    gaia_xp_encoder = GaiaXPHead(
        model_path=xp_encoder_path,
        embed_dim=embed_dim,
        n_head=n_head,
        model_embed_dim=model_embed_dim,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
        load_pretrained_weights=False
    )

    # Initialize spectrum encoder
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
    """
    Initialize both encoders with split architecture for shared/private representations.

    This function is used for the specclip_split model which outputs both shared
    and private embeddings for cross-modal learning.
    """
    from specclip.models.specclip_reconstruct_split_5122562_mlp_recordloss import (
        GaiaXPHead_split, LamostLRSHead_split
    )

    # Initialize Gaia XP (image) encoder
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

    # Initialize LAMOST LRS (spectrum) encoder
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

def predict_mlp_parameters(
    model,  # can be LRS or CLIP
    mlp_model: nn.Module,
    parameters: List[str],
    error_parameter_name: List[str],
    flux: np.ndarray,
    wavelength: np.ndarray,
    error: np.ndarray,
    device: torch.device,
    use_clip: bool = False,
    use_split_clip: bool = False,  # NEW: flag for split model
    stats: Dict[str, List[float]]=None
) -> pd.DataFrame:
    """
    Predict parameter for a single spectrum using MLP model
    
    Args:
        model: Pre-trained model (LRS SpecFormer or CLIP model)
        mlp_model: Pre-trained MLP model
        parameters: List of parameter names to predict
        error_parameter_name: List of error column names
        flux: Flux array
        wavelength: Wavelength array
        error: Error array (can be None)
        device: Computation device
        use_clip: Use CLIP model (predrecon)
        use_split_clip: Use split CLIP model
        stats: Normalization statistics
        
    Returns:
        DataFrame containing predictions and errors
    """
    # Interpolate spectrum
    new_wavelength, new_flux = interpolate_spectrum(wavelength, flux, num_points=1462)
    
    # Convert to tensor and prepare
    flux_tensor = torch.tensor(new_flux, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-1)
    wavelength_tensor = torch.tensor(new_wavelength, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-1)
    
    # Get embeddings based on model type
    model.eval()
    with torch.no_grad():
        if use_split_clip:
            # Use split CLIP model - concatenate shared+private
            shared, private = model(flux_tensor, input_type='lamost_spectra')
            embeddings = torch.cat([shared, private], dim=-1)  # 512+256=768
            
        elif use_clip:
            # Use predrecon CLIP model - unified 768-dim
            outputs = model(flux_tensor, input_type='lamost_spectra')
            embeddings = outputs.reshape(1, 768)
        else:
            # Use LRS SpecFormer model
            outputs = model(flux_tensor)
            embeddings = torch.mean(outputs['embedding'], -2)
        
        # Apply normalization if provided
        if stats is not None:
            mean = torch.as_tensor(stats['mean'], device=device).unsqueeze(0)
            std = torch.as_tensor(stats['std'], device=device).unsqueeze(0)
            embeddings = (embeddings - mean) / std
    
    # Get prediction from MLP
    mlp_model.eval()
    with torch.no_grad():
        prediction = mlp_model(embeddings)

    # Create results dictionary
    results_dict = {}
    for i, parameter in enumerate(parameters):
        parameter_prediction = prediction[i, 0].cpu().numpy()
        results_dict[parameter] = ["{:.2f}".format(float(parameter_prediction))]
        
        if error is not None:
            if isinstance(error, (list, np.ndarray)):
                error_value = error[i] if i < len(error) else None
            else:
                error_value = error
                
            if error_value is not None:
                results_dict[error_parameter_name[i]] = ["{:.2f}".format(float(error_value))]
    
    return pd.DataFrame(results_dict)


def predict_sbi_parameters(
    model,  #  can be LRS or CLIP
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
    new_wavelength, new_flux = interpolate_spectrum(wavelength, flux, num_points=1462)
    
    # Convert to tensor
    flux_tensor = torch.tensor(new_flux, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-1)
    wavelength_tensor = torch.tensor(new_wavelength, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-1)
    
    # Get embeddings based on model type
    model.eval()
    with torch.no_grad():
        if use_split_clip:
            # Use split CLIP model - concatenate shared+private
            shared, private = model(flux_tensor, input_type='lamost_spectra')
            embeddings = torch.cat([shared, private], dim=-1)
            embeddings = embeddings.reshape(1,768)
        elif use_clip:
            # Use predrecon CLIP model - unified 768-dim
            outputs = model(flux_tensor, input_type='lamost_spectra')
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


class StellarParameterPredictor:
    """
    A class for predicting stellar parameters from spectra using pre-trained models.
    """
    
    def __init__(self, model_paths_dict=None):
        """
        Initialize with paths to all models.
        
        Args:
            model_paths_dict: Dictionary containing model paths with keys:
                - 'xp_encoder_path'
                - 'lrs_encoder_path'
                - 'transformer_path'
                - 'sbi_model_path_chemical'
                - 'sbi_model_path_seismic'
                - 'sbi_model_path_rv'
                - 'sbi_model_path_dpi1'
                - 'sbi_model_path_teff'
                - 'sbi_model_path_logg'
                - 'sbi_model_path_feh'
                - 'mlp_model_path_ebprp'
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Use default paths if none provided
        if model_paths_dict is None:
            model_paths_dict = self._get_default_paths()
        
        self.model_paths = model_paths_dict
        
        # Load models
        print("Loading models...")
        self._load_models()
        print("Models loaded successfully!")
        
    def _get_default_paths(self):
        """Return default model paths"""
        return {
            'TBD'
        }
    
    def _load_models(self):
        """Load all pre-trained models"""
        # Load LRS spectrum encoder
        self.lrs_model = SpecFormer.load_from_checkpoint(
            self.model_paths['lrs_encoder_path']
        )
        self.lrs_model = self.lrs_model.to(self.device)
        self.lrs_model.eval()

        # Load predrecon CLIP model (for ebprp, chemical abundances, seismic, dpi)
        gaia_xp_encoder, lamost_lrs_encoder = initialize_encoders(
            xp_encoder_path=self.model_paths['xp_encoder_path'],
            lrs_encoder_path=self.model_paths['lrs_encoder_path'],
        )

        predrecon_path = self.model_paths.get('specclip_predrecon_path',
                                              self.model_paths.get('transformer_path'))
        self.clip_model_predrecon = SpecClipModel.load_from_checkpoint(
            checkpoint_path=predrecon_path,
            gaia_xp_encoder=gaia_xp_encoder,
            lamost_lrs_encoder=lamost_lrs_encoder,
            strict=True
        )
        self.clip_model_predrecon = self.clip_model_predrecon.to(self.device)
        self.clip_model_predrecon.eval()

        # Load split CLIP model (for teff, logg, rv, feh)
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
        
        # Load MLP models for E(BP-RP)
        self.mlp_model_ebprp = load_mlp_model(
            self.model_paths['mlp_model_path_ebprp'], 
            self.device,
            hidden_dims=[1024, 512, 64]
        )
        self.mlp_model_ebprp.eval()

        # Load statistics
        self.stats_dir = self.model_paths['stats_dir']

    def predict_batch(self, spectrum_file_paths, **kwargs):
        """
        Predict parameters for multiple spectra.
        
        Args:
            spectrum_file_paths: List of paths to spectrum files
            **kwargs: Additional arguments passed to predict()
            
        Returns:
            List of DataFrames or dicts with predictions
        """
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

    def predict(self, spectrum_file_path, parameter_types=['all'], 
                simple_header=True, return_dataframe=True, display_format='row'):
        """
        Main prediction function for stellar parameters.
        
        Args:
            spectrum_file_path: Path to spectrum file (.fits, .csv, .txt, .npy)
            parameter_types: List of parameter types to predict. Options:
                - 'all': All parameters
                - 'chemical': Chemical abundances
                - 'seismic': Asteroseismic parameters
                - 'atmospheric': Atmospheric parameters (Teff, logg, [Fe/H])
                - 'RV': Radial velocity
                - 'DPi1': Period spacing
                - 'ebprp': Reddening E(BP-RP)
                Or a list like ['chemical', 'atmospheric']
            simple_header: If True, use simple column names. If False, include units.
            return_dataframe: If True, return pandas DataFrame. If False, return dict.
            display_format: 'row' for row-wise display, 'column' for column-wise (TBD)
            
        Returns:
            pd.DataFrame or dict with predictions and uncertainties
        """
        # Validate input file
        if not os.path.exists(spectrum_file_path):
            raise FileNotFoundError(f"Spectrum file not found: {spectrum_file_path}")
        
        # Load and preprocess spectrum
        try:
            wavelength, flux = load_spectrum_data(spectrum_file_path)
        except Exception as e:
            raise ValueError(f"Error loading spectrum: {str(e)}")
        
        # Handle parameter_types input
        if isinstance(parameter_types, str):
            parameter_types = [parameter_types]
        
        if 'all' in parameter_types:
            parameter_types = ['chemical', 'seismic', 'RV', 'DPi1', 
                             'atmospheric', 'ebprp']
        
        # Store predictions as rows
        prediction_rows = []
        
        # Chemical abundances (SBI)
        if 'chemical' in parameter_types:
            try:
                predictions_chemical = self._predict_chemical_rowwise(
                    wavelength, flux, simple_header
                )
                prediction_rows.extend(predictions_chemical)
            except Exception as e:
                print(f"Warning: Chemical abundance prediction failed: {str(e)}")
        
        # Seismic parameters (SBI)
        if 'seismic' in parameter_types:
            try:
                predictions_seismic = self._predict_seismic_rowwise(
                    wavelength, flux, simple_header
                )
                prediction_rows.extend(predictions_seismic)
            except Exception as e:
                print(f"Warning: Seismic parameter prediction failed: {str(e)}")
        
        # Radial velocity (SBI)
        if 'RV' in parameter_types:
            try:
                predictions_rv = self._predict_rv_rowwise(
                    wavelength, flux, simple_header
                )
                prediction_rows.extend(predictions_rv)
            except Exception as e:
                print(f"Warning: RV prediction failed: {str(e)}")
        
        # DPi1 (SBI)
        if 'DPi1' in parameter_types:
            try:
                predictions_dpi1 = self._predict_dpi1_rowwise(
                    wavelength, flux, simple_header
                )
                prediction_rows.extend(predictions_dpi1)
            except Exception as e:
                print(f"Warning: DPi1 prediction failed: {str(e)}")
        
        # Atmospheric parameters (SBI)
        if 'atmospheric' in parameter_types:
            try:
                predictions_teff = self._predict_teff_rowwise(
                    wavelength, flux, simple_header
                )
                predictions_logg = self._predict_logg_rowwise(
                    wavelength, flux, simple_header
                )
                predictions_feh = self._predict_feh_rowwise(
                    wavelength, flux, simple_header
                )
                prediction_rows.extend(predictions_teff)
                prediction_rows.extend(predictions_logg)
                prediction_rows.extend(predictions_feh)
            except Exception as e:
                print(f"Warning: Atmospheric parameter prediction failed: {str(e)}")
        
        # E(BP-RP) (MLP)
        if 'ebprp' in parameter_types:
            try:
                predictions_ebprp = self._predict_ebprp_rowwise(
                    wavelength, flux, simple_header
                )
                prediction_rows.extend(predictions_ebprp)
            except Exception as e:
                print(f"Warning: E(BP-RP) prediction failed: {str(e)}")
        
        # Create DataFrame
        if len(prediction_rows) == 0:
            raise ValueError("No predictions were successfully made")
        
        if display_format == 'row':
            # Row-wise format: Parameter | Prediction | Error (if SBI)
            df = pd.DataFrame(prediction_rows)
            # Reorder columns to put Parameter first
            cols = ['Parameter']
            if 'Prediction' in df.columns:
                cols.append('Prediction')
            if 'Error' in df.columns:
                cols.append('Error')
            if 'Method' in df.columns:
                cols.append('Method')
            df = df[cols]
        else:
            # TBD
            df = pd.DataFrame(prediction_rows)
        
        if return_dataframe:
            return df
        else:
            return df.to_dict(orient='records')
            
    def _get_model_for_parameter(self, param_name):
        """
        Determine which model and flags to use for a given parameter
        
        Returns:
            tuple: (model, use_clip, use_split_clip)
        """
        if param_name in SPLIT_LRS_PARAMS:
            # Use split CLIP model
            return self.clip_model_split, True, True
        elif param_name in PREDRECON_LRS_PARAMS:
            # Use predrecon CLIP model
            return self.clip_model_predrecon, True, False
        else:
            # Default to LRS model
            return self.lrs_model, False, False
    
    def _predict_chemical_rowwise(self, wavelength, flux, simple_header):
        """Predict chemical abundances and return as rows"""
        param_config = [
            ('a_fe', 'mlp_model_path_afe', '[α/Fe]', 'MLP'),
            ('c_fe', 'mlp_model_path_cfe', '[C/Fe]', 'MLP'),
            ('n_fe', 'mlp_model_path_nfe', '[N/Fe]', 'MLP'),
            ('mg_fe', 'mlp_model_path_mgfe', '[Mg/Fe]', 'MLP'),
            ('o_fe', 'mlp_model_path_ofe', '[O/Fe]', 'MLP'),
            ('al_fe', 'mlp_model_path_alfe', '[Al/Fe]', 'MLP'),
            ('si_fe', 'mlp_model_path_sife', '[Si/Fe]', 'MLP'),
            ('ca_fe', 'mlp_model_path_cafe', '[Ca/Fe]', 'MLP'),
            ('ti_fe', 'mlp_model_path_tife', '[Ti/Fe]', 'MLP'),
            ('mn_fe', 'mlp_model_path_mnfe', '[Mn/Fe]', 'MLP'),
            ('ni_fe', 'mlp_model_path_nife', '[Ni/Fe]', 'MLP'),
            ('cr_fe', 'mlp_model_path_crfe', '[Cr/Fe]', 'MLP'),
        ]

        rows = []
        for param_name, model_key, display_name, method in param_config:
            try:
                model_path = self.model_paths.get(model_key)
                if model_path is None:
                    continue

                param_display = param_name if simple_header else display_name

                # Get correct model for this parameter
                encoder_model, use_clip, use_split_clip = self._get_model_for_parameter(param_name)

                # Load stats
                stats_file = f'lrs/{param_name.replace("_", "")}_mean_std.json'
                try:
                    with open(os.path.join(self.stats_dir, stats_file), 'r') as f:
                        stats = json.load(f)
                except FileNotFoundError:
                    stats = None

                # Load MLP model and predict
                mlp_model = load_mlp_model(model_path, self.device, hidden_dims=[1024, 512, 64])
                predictions_df = predict_mlp_parameters(
                    encoder_model,  # Use correct encoder
                    mlp_model,
                    [param_name],
                    ['fe_h_err'],
                    flux,
                    wavelength,
                    None,
                    self.device,
                    use_clip=use_clip,
                    use_split_clip=use_split_clip,  # Pass split flag
                    stats=stats
                )
                
                rows.append({
                    'Parameter': param_display,
                    'Prediction': predictions_df[param_name].values[0],
                    'Method': method
                })
        
            except Exception as e:
                print(f"Warning: Failed to predict {param_name}: {str(e)}")
                continue

        return rows

    def _predict_seismic_rowwise(self, wavelength, flux, simple_header):
        """Predict asteroseismic parameters and return as rows"""
        param_config = [
            ('dnu', 'sbi_model_path_dnu', 'Δν [μHz]'),
            ('nu_max', 'sbi_model_path_nu_max', 'ν_max [μHz]'),
            ('mass', 'sbi_model_path_mass', 'Mass [M☉]'),
            ('rad', 'sbi_model_path_rad', 'Radius [R☉]'),
            ('age', 'sbi_model_path_age', 'Age [Gyr]'),
        ]

        rows = []
        for param_name, model_key, display_name in param_config:
            try:
                model_path = self.model_paths.get(model_key)
                if model_path is None:
                    continue

                param_display = param_name if simple_header else display_name

                # Get correct model for this parameter
                encoder_model, use_clip, use_split_clip = self._get_model_for_parameter(param_name)

                predictions_df = predict_sbi_parameters(
                    encoder_model,  # Use correct encoder
                    model_path,
                    [param_name],
                    [f'{param_name}_err'],
                    wavelength,
                    flux,
                    self.device,
                    use_clip=use_clip,
                    use_split_clip=use_split_clip  # Pass split flag
                )

                rows.append({
                    'Parameter': param_display,
                    'Prediction': predictions_df[param_name].values[0],
                    'Error': predictions_df[f'{param_name}_err'].values[0],
                    'Method': 'SBI'
                })
            except Exception as e:
                print(f"Warning: Failed to predict {param_name}: {str(e)}")
                continue

        return rows

    def _predict_rv_rowwise(self, wavelength, flux, simple_header):
        """Predict radial velocity and return as row"""
        param_name = 'rv'
        param_display = 'RV [km/s]' if not simple_header else 'rv'
        
        # Get correct model for RV
        encoder_model, use_clip, use_split_clip = self._get_model_for_parameter(param_name)
        
        predictions_df = predict_sbi_parameters(
            encoder_model,
            self.model_paths['sbi_model_path_rv'],
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

    def _predict_dpi1_rowwise(self, wavelength, flux, simple_header):
        """Predict DPi1 and return as row"""
        param_name = 'dpi'
        param_display = 'ΔΠ₁ [s]' if not simple_header else 'dpi'
        
        # Get correct model for DPi1
        encoder_model, use_clip, use_split_clip = self._get_model_for_parameter(param_name)
        
        predictions_df = predict_sbi_parameters(
            encoder_model,
            self.model_paths['sbi_model_path_dpi1'],
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

    def _predict_teff_rowwise(self, wavelength, flux, simple_header):
        """Predict effective temperature and return as row"""
        param_name = 'teff'
        param_display = 'T_eff [K]' if not simple_header else 'teff'
        
        # Get correct model (should use split CLIP)
        encoder_model, use_clip, use_split_clip = self._get_model_for_parameter(param_name)
        
        predictions_df = predict_sbi_parameters(
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
        param_display = 'log g [cgs]' if not simple_header else 'logg'
        
        # Get correct model (should use split CLIP)
        encoder_model, use_clip, use_split_clip = self._get_model_for_parameter(param_name)
        
        predictions_df = predict_sbi_parameters(
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
        """Predict metallicity and return as row"""
        param_name = 'fe_h'
        param_display = '[Fe/H] [dex]' if not simple_header else 'fe_h'

        # Get correct model (should use split CLIP)
        encoder_model, use_clip, use_split_clip = self._get_model_for_parameter(param_name)

        # Load stats
        try:
            with open(os.path.join(self.stats_dir, 'lrs/feh_mean_std.json'), 'r') as f:
                stats = json.load(f)
        except FileNotFoundError:
            stats = None

        # Use MLP model
        mlp_model = load_mlp_model(self.model_paths['mlp_model_path_feh'], 
                                   self.device, hidden_dims=[1024, 512, 64])
        predictions_df = predict_mlp_parameters(
            encoder_model,
            mlp_model,
            [param_name],
            [f'{param_name}_err'],
            flux,
            wavelength,
            None,
            self.device,
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
        """Predict E(BP-RP) using correct CLIP model and return as row"""
        param_name = 'ebprp'
        param_display = 'E(BP-RP) [mag]' if not simple_header else 'e_bp_rp'

        # Get correct model (should use predrecon CLIP)
        encoder_model, use_clip, use_split_clip = self._get_model_for_parameter(param_name)

        # Load stats
        try:
            with open(os.path.join(self.stats_dir, 'lrs/ebprp_mean_std.json'), 'r') as f:
                stats = json.load(f)
        except FileNotFoundError:
            stats = None
        
        predictions_df = predict_mlp_parameters(
            encoder_model,
            self.mlp_model_ebprp,
            ['e_bp_rp'],
            ['e_bp_rp_err'],
            flux,
            wavelength,
            None,
            self.device,
            use_clip=use_clip,
            use_split_clip=use_split_clip,
            stats=stats
        )
        
        return [{
            'Parameter': param_display,
            'Prediction': predictions_df['e_bp_rp'].values[0],
            'Method': 'MLP'
        }]

    def display_results(self, predictions_df, style='default'):
        """
        Format predictions for nice display in Jupyter notebooks.
        
        Args:
            predictions_df: DataFrame with predictions (row-wise format)
            style: Display style ('default', 'highlight', 'minimal', 'formatted')
            
        Returns:
            Styled DataFrame for display
        """
        if style == 'formatted':
            # Apply number formatting for better readability
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
                'Prediction': format_value,
                'Error': format_value
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
            # Highlight rows by method
            def highlight_method(row):
                if row['Method'] == 'SBI':
                    return ['background-color: #e6f2ff'] * len(row)
                else:
                    return ['background-color: #fff2e6'] * len(row)
            
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
    """
    Get default configuration dictionary for StellarParameterPredictor.

    Returns:
        Dictionary with default model paths
    """
    return {
        'xp_encoder_path': "/work/zxs/model/pretrained_models/epoch=191-val_loss=0.0000_mt_xp.ckpt",
        'lrs_encoder_path': "/work/zxs/model/pretrained_models/epoch=128-val_loss=0.0000_mt_lrs.ckpt",
        # Two specclip models for different parameters
        'specclip_predrecon_path': '/work/zxs/model/pretrained_models/specclip_model_predrecon_mlp.ckpt',
        'specclip_split_path': '/work/zxs/model/pretrained_models/specclip_model_split_mlp.ckpt',
        # Legacy key for backward compatibility
        'transformer_path': '/work/zxs/model/pretrained_models/specclip_model_predrecon_mlp.ckpt',
        # Individual SBI models for each parameter
        'sbi_model_path_age': './downstream_models/lrs/sbi/age.pk',
        'sbi_model_path_nu_max': './downstream_models/lrs/sbi/numax.pk',
        'sbi_model_path_mass': './downstream_models/lrs/sbi/mass.pk',
        'sbi_model_path_rad': './downstream_models/lrs/sbi/rad.pk',
        'sbi_model_path_dnu': './downstream_models/lrs/sbi/age.pk',
        'sbi_model_path_rv': './downstream_models/lrs/sbi/rv.pkl',
        'sbi_model_path_dpi1': './downstream_models/lrs/sbi/DP.pkl',
        'sbi_model_path_teff': './downstream_models/lrs/sbi/teff.pkl',
        'sbi_model_path_logg': './downstream_models/lrs/sbi/logg.pkl',
        # Individual MLP models for each parameter
        'mlp_model_path_feh': './downstream_models/lrs/mlp/feh.pt',
        'mlp_model_path_ebprp': './downstream_models/lrs/mlp/ebprp.pt',
        'mlp_model_path_afe': './downstream_models/lrs/mlp/afe.pkl',
        'mlp_model_path_cfe': './downstream_models/lrs/sbi/cfe.pkl',
        'mlp_model_path_nfe': './downstream_models/lrs/sbi/nfe.pkl',
        'mlp_model_path_alfe': './downstream_models/lrs/sbi/cfe.pkl',
        'mlp_model_path_mgfe': './downstream_models/lrs/sbi/cfe.pkl',
        'mlp_model_path_mnfe': './downstream_models/lrs/sbi/cfe.pkl',
        'mlp_model_path_nife': './downstream_models/lrs/sbi/cfe.pkl',
        'mlp_model_path_ofe': './downstream_models/lrs/sbi/cfe.pkl',
        'mlp_model_path_sife': './downstream_models/lrs/sbi/cfe.pkl',
        'mlp_model_path_tife': './downstream_models/lrs/sbi/cfe.pkl',
        'mlp_model_path_crfe': './downstream_models/lrs/sbi/cfe.pkl',
        'mlp_model_path_cafe': './downstream_models/lrs/sbi/cfe.pkl',
        'stats_dir': './stats',
    }