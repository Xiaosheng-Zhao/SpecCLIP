"""
Functions for LAMOST LRS 
"""
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
                        new_wavelength_range: List[float] = [4000, 5598],
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

