"""
Functions for Gaia XP 
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
