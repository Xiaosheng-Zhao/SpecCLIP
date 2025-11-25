"""
Unified stellar parameter prediction interface supporting both LAMOST LRS and Gaia XP spectra.

This module provides a single interface for predicting stellar parameters from different
spectroscopic surveys. It automatically handles survey-specific preprocessing and model selection.

Usage:
    from stellar_params_unified import UnifiedStellarParameterPredictor
    
    # For LAMOST LRS spectra
    predictor = UnifiedStellarParameterPredictor(survey_type='LAMOST_LRS')
    results = predictor.predict('lamost_spectrum.fits')
    
    # For Gaia XP spectra
    predictor = UnifiedStellarParameterPredictor(survey_type='Gaia_XP')
    results = predictor.predict('gaia_xp_spectrum.csv')
"""

#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from predict_lrs_wclip_v0 import StellarParameterPredictor as LRSPredictor
from predict_xp_wclip_v0 import StellarParameterPredictorXP as XPPredictor
import pandas as pd


class UnifiedStellarParameterPredictor:
    """
    Unified predictor supporting both LAMOST LRS and Gaia XP spectra.
    
    This class provides a single interface for stellar parameter prediction across
    different spectroscopic surveys, handling survey-specific data formats and models.
    """
    
    def __init__(self, survey_type='LAMOST_LRS', lrs_config=None, xp_config=None):
        """
        Initialize predictor for specified survey type.
        
        Args:
            survey_type: Survey type, either 'LAMOST_LRS' or 'Gaia_XP'
            lrs_config: Optional configuration dict for LRS models
            xp_config: Optional configuration dict for XP models
            
        Raises:
            ValueError: If survey_type is not recognized
        """
        self.survey_type = survey_type.upper()
        
        if self.survey_type == 'LAMOST_LRS':
            print("Initializing LAMOST LRS predictor...")
            print("This predictor supports:")
            print("  - Chemical abundances (12 elements)")
            print("  - Asteroseismic parameters")
            print("  - Radial velocity")
            print("  - Atmospheric parameters")
            print("  - Reddening")
            self.predictor = LRSPredictor(lrs_config)
            self.available_params = ['chemical', 'seismic', 'RV', 'DPi1', 
                                    'atmospheric', 'ebprp']
            self.wavelength_range = "4000-5600 Å"
            
        elif self.survey_type == 'GAIA_XP':
            print("Initializing Gaia XP predictor...")
            print("This predictor supports:")
            print("  - Alpha element abundance")
            print("  - Carbon and Nitrogen abundances")
            print("  - Atmospheric parameters")
            print("  - Reddening")
            self.predictor = XPPredictor(xp_config)
            self.available_params = ['afe', 'cfe', 'nfe', 'atmospheric', 'ebprp']
            self.wavelength_range = "336-1020 nm"
            
        else:
            raise ValueError(f"Unknown survey type: {survey_type}. "
                           f"Choose 'LAMOST_LRS' or 'Gaia_XP'")
    
    def predict(self, spectrum_file_path, parameter_types=['all'], 
                simple_header=True, return_dataframe=True, display_format='row'):
        """
        Predict stellar parameters from a spectrum.
        
        Args:
            spectrum_file_path: Path to spectrum file (.fits, .csv, .txt)
            parameter_types: Parameters to predict (survey-specific)
                For LAMOST_LRS: ['chemical', 'seismic', 'RV', 'DPi1', 'atmospheric', 'ebprp', 'all']
                For Gaia_XP: ['afe', 'cfe', 'nfe', 'atmospheric', 'ebprp', 'all']
            simple_header: If True, use simple column names. If False, include units and descriptions
            return_dataframe: If True, return pandas DataFrame. If False, return dict
            display_format: 'row' for row-wise display (recommended), 'column' for column-wise
            
        Returns:
            pd.DataFrame or dict with predictions and uncertainties
            
        Example:
            >>> predictor = UnifiedStellarParameterPredictor('LAMOST_LRS')
            >>> results = predictor.predict('spectrum.fits', parameter_types=['atmospheric'])
            >>> print(results)
        """
        return self.predictor.predict(
            spectrum_file_path=spectrum_file_path,
            parameter_types=parameter_types,
            simple_header=simple_header,
            return_dataframe=return_dataframe,
            display_format=display_format
        )
    
    def predict_batch(self, spectrum_file_paths, **kwargs):
        """
        Predict stellar parameters for multiple spectra.
        
        Args:
            spectrum_file_paths: List of paths to spectrum files
            **kwargs: Additional arguments passed to predict()
            
        Returns:
            List of DataFrames (or dicts) with predictions
            
        Example:
            >>> predictor = UnifiedStellarParameterPredictor('Gaia_XP')
            >>> file_list = ['spec1.csv', 'spec2.csv', 'spec3.csv']
            >>> results = predictor.predict_batch(file_list, parameter_types=['atmospheric'])
        """
        return self.predictor.predict_batch(spectrum_file_paths, **kwargs)
    
    def display_results(self, predictions_df, style='default'):
        """
        Format predictions for nice display in Jupyter notebooks.
        
        Args:
            predictions_df: DataFrame with predictions (row-wise format)
            style: Display style options:
                - 'default': Basic formatting
                - 'formatted': Enhanced formatting with number formatting
                - 'highlight': Color-coded by method or parameter type
                - 'minimal': Minimal formatting without index
                
        Returns:
            Styled DataFrame for display
            
        Example:
            >>> results = predictor.predict('spectrum.fits')
            >>> styled = predictor.display_results(results, style='formatted')
            >>> display(styled)  # In Jupyter
        """
        return self.predictor.display_results(predictions_df, style)
    
    def get_available_parameters(self):
        """
        Get list of available parameter types for current survey.
        
        Returns:
            List of parameter type strings
            
        Example:
            >>> predictor = UnifiedStellarParameterPredictor('LAMOST_LRS')
            >>> print(predictor.get_available_parameters())
            ['chemical', 'seismic', 'RV', 'DPi1', 'atmospheric', 'ebprp']
        """
        return self.available_params
    
    def get_info(self):
        """
        Get information about the current predictor configuration.
        
        Returns:
            Dictionary with survey type, wavelength range, and available parameters
        """
        return {
            'survey_type': self.survey_type,
            'wavelength_range': self.wavelength_range,
            'available_parameters': self.available_params,
            'supported_formats': ['.fits', '.fit', '.csv', '.txt']
        }
    
    def switch_survey(self, survey_type, config=None):
        """
        Switch to a different survey type.
        
        This reloads all models for the new survey type. Note that this will
        take time as models need to be loaded into memory.
        
        Args:
            survey_type: 'LAMOST_LRS' or 'Gaia_XP'
            config: Optional configuration dict for the new survey
            
        Example:
            >>> predictor = UnifiedStellarParameterPredictor('LAMOST_LRS')
            >>> # ... do some predictions ...
            >>> predictor.switch_survey('Gaia_XP')
            >>> # Now can predict from Gaia XP spectra
        """
        print(f"\nSwitching from {self.survey_type} to {survey_type.upper()}...")
        self.__init__(
            survey_type=survey_type, 
            lrs_config=config if survey_type.upper() == 'LAMOST_LRS' else None,
            xp_config=config if survey_type.upper() == 'GAIA_XP' else None
        )
        print("Switch complete!")


def get_default_config(survey_type='LAMOST_LRS'):
    """
    Get default configuration for specified survey.
    
    Args:
        survey_type: 'LAMOST_LRS' or 'Gaia_XP'
        
    Returns:
        Dictionary with default model paths for the specified survey
        
    Example:
        >>> config = get_default_config('LAMOST_LRS')
        >>> print(config.keys())
        >>> # Modify paths if needed
        >>> config['spectrum_encoder_path'] = '/custom/path/to/model.ckpt'
        >>> predictor = UnifiedStellarParameterPredictor('LAMOST_LRS', lrs_config=config)
    """
    survey_type = survey_type.upper()
    
    if survey_type == 'LAMOST_LRS':
        from predict_lrs_wclip_v0 import get_default_config as get_lrs_config
        return get_lrs_config()
    elif survey_type == 'GAIA_XP':
        from predict_xp_wclip_v0 import get_default_config as get_xp_config
        return get_xp_config()
    else:
        raise ValueError(f"Unknown survey type: {survey_type}")


def compare_surveys(lrs_spectrum_path=None, xp_spectrum_path=None, 
                   parameter_types='atmospheric', simple_header=False, lrs_config=None, xp_config=None):
    """
    Compare predictions from both LAMOST LRS and Gaia XP for the same star.
    
    This is useful when you have both LRS and XP spectra for the same object
    and want to compare the predictions.
    
    Args:
        lrs_spectrum_path: Path to LAMOST LRS spectrum
        xp_spectrum_path: Path to Gaia XP spectrum
        parameter_types: Parameters to predict (use common parameters like 'atmospheric')
        simple_header: Use simple headers for easier comparison
        
    Returns:
        Dictionary with 'lrs_results' and 'xp_results' DataFrames
        
    Example:
        >>> results = compare_surveys(
        ...     lrs_spectrum_path='lamost_spec.fits',
        ...     xp_spectrum_path='gaia_xp_spec.csv',
        ...     parameter_types='atmospheric'
        ... )
        >>> print("LRS Results:")
        >>> display(results['lrs_results'])
        >>> print("\nXP Results:")
        >>> display(results['xp_results'])
    """
    results = {}
    
    if lrs_spectrum_path is not None:
        print("Predicting from LAMOST LRS spectrum...")
        lrs_predictor = UnifiedStellarParameterPredictor('LAMOST_LRS',lrs_config=lrs_config)
        results['lrs_results'] = lrs_predictor.predict(
            lrs_spectrum_path, 
            parameter_types=parameter_types,
            simple_header=simple_header
        )
    
    if xp_spectrum_path is not None:
        print("\nPredicting from Gaia XP spectrum...")
        xp_predictor = UnifiedStellarParameterPredictor('Gaia_XP',xp_config=xp_config)
        results['xp_results'] = xp_predictor.predict(
            xp_spectrum_path,
            parameter_types=parameter_types,
            simple_header=simple_header
        )
    
    return results


# Convenience functions for quick predictions

def predict_from_lamost(spectrum_path, parameter_types=['all'], **kwargs):
    """
    Quick function to predict from a LAMOST LRS spectrum.
    
    Args:
        spectrum_path: Path to LAMOST spectrum file
        parameter_types: Parameters to predict
        **kwargs: Additional arguments passed to predict()
        
    Returns:
        DataFrame with predictions
    """
    predictor = UnifiedStellarParameterPredictor('LAMOST_LRS',**kwargs)
    return predictor.predict(spectrum_path, parameter_types=parameter_types)#, **kwargs


def predict_from_gaia(spectrum_path, parameter_types=['all'], **kwargs):
    """
    Quick function to predict from a Gaia XP spectrum.
    
    Args:
        spectrum_path: Path to Gaia XP spectrum file
        parameter_types: Parameters to predict
        **kwargs: Additional arguments passed to predict()
        
    Returns:
        DataFrame with predictions
    """
    predictor = UnifiedStellarParameterPredictor('Gaia_XP', **kwargs)
    return predictor.predict(spectrum_path, parameter_types=parameter_types)


if __name__ == "__main__":
    # Example usage
    print("Unified Stellar Parameter Predictor")
    print("=" * 50)
    print("\nSupported surveys:")
    print("  - LAMOST LRS (Low Resolution Spectra)")
    print("  - Gaia XP (BP/RP Spectra)")
    print("\nFor usage examples, see the documentation or run:")
    print("  help(UnifiedStellarParameterPredictor)")
    print("\nQuick start:")
    print("  from stellar_params_unified import UnifiedStellarParameterPredictor")
    print("  predictor = UnifiedStellarParameterPredictor('LAMOST_LRS')")
    print("  results = predictor.predict('spectrum.fits')")
    print("  display(results)")
