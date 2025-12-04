"""
Spectral Retrieval and Cross-Modal Analysis Module

This module provides functionality for:
1. Building embedding databases from test spectra
2. In-modal spectral retrieval
3. Cross-modal spectral retrieval
4. Cross-modal spectral prediction
"""

import os
import h5py
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import warnings

# Suppress common warnings
warnings.filterwarnings('ignore', category=UserWarning)

from specclip.models.specclip_reconstruct_embed768_mlp import GaiaXPHeadWithMLP as GaiaXPHead
from specclip.models.specclip_reconstruct_embed768_mlp import LamostLRSHead
from specclip.models import SpecClipModel_reconstruct_embed768_mlp as SpecClipModel

from scipy.interpolate import interp1d

from scipy.signal import medfilt, savgol_filter

def gaspp_fitcont2(ww, ff, cfsnr=60):
    """
    Fit continuum over 3850 - 9000 A region.
    """
    orflx = ff.copy()
    ff = ff.astype(np.float64)
    ff = medfilt(ff, kernel_size=7)
    
    ww100 = ww.copy()
    ff100 = ff.copy()
    
    wran1ind = np.where((ww100 <= 5700.0) & (ww100 >= 3700.0))[0]
    nran1 = len(wran1ind)
    wran2ind = np.where(ww100 > 6100.0)[0]
    nran2 = len(wran2ind)
    
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
    
    if cfsnr < 50:
        nl = int(-1.0 * cfsnr + 55)
        if nl % 2 == 0:
            nl += 1
        if nl < 3:
            nl = 3
        ofl = savgol_filter(fran1, window_length=nl, polyorder=4, mode='nearest')
        ofl2 = savgol_filter(fran2, window_length=nl, polyorder=4, mode='nearest')
    else:
        ofl = savgol_filter(fran1, window_length=3, polyorder=2, mode='nearest')
        ofl2 = savgol_filter(fran2, window_length=3, polyorder=2, mode='nearest')
    
    if nran1 > 100:
        coef0 = np.polyfit(wran1, ofl, 5)
        cfit0 = np.polyval(coef0, wran1)
        flxind = np.argmax(cfit0)
        if wran1[flxind] < 4500:
            w11, w12 = 4030, 4160
            w21, w22 = 4270, 4410
            w31, w32 = 4800, 4940
            wbin = 10
            wind1ind = np.where((ww >= w11) & (ww <= w12))[0]
            wind2ind = np.where((ww >= w21) & (ww <= w22))[0]
            wind3ind = np.where((ww >= w31) & (ww <= w32))[0]
            indw11 = np.where((ww >= w11 - wbin) & (ww <= w11 + wbin))[0]
            indw12 = np.where((ww >= w12 - wbin) & (ww <= w12 + wbin))[0]
            indw21 = np.where((ww >= w21 - wbin) & (ww <= w21 + wbin))[0]
            indw22 = np.where((ww >= w22 - wbin) & (ww <= w22 + wbin))[0]
            indw31 = np.where((ww >= w31 - wbin) & (ww <= w31 + wbin))[0]
            indw32 = np.where((ww >= w32 - wbin) & (ww <= w32 + wbin))[0]
            f11 = fran1[indw11].max() if len(indw11) > 0 else None
            f12 = fran1[indw12].max() if len(indw12) > 0 else None
            f21 = fran1[indw21].max() if len(indw21) > 0 else None
            f22 = fran1[indw22].max() if len(indw22) > 0 else None
            f31 = fran1[indw31].max() if len(indw31) > 0 else None
            f32 = fran1[indw32].max() if len(indw32) > 0 else None
            if f11 is not None and f12 is not None and len(wind1ind) > 0:
                fran1[wind1ind] = np.interp(ww[wind1ind], [w11, w12], [f11, f12])
            if f21 is not None and f22 is not None and len(wind2ind) > 0:
                fran1[wind2ind] = np.interp(ww[wind2ind], [w21, w22], [f21, f22])
            if f31 is not None and f32 is not None and len(wind3ind) > 0:
                fran1[wind3ind] = np.interp(ww[wind3ind], [w31, w32], [f31, f32])
        ofl = fran1.copy()
        for _ in range(10):
            coef = np.polyfit(wran1, ofl, 5)
            cfit = np.polyval(coef, wran1)
            ofl = np.maximum(cfit, ofl)
        coef_tot = np.polyfit(wran1, ofl, 5)
        cfit1 = np.polyval(coef_tot, wran1)
    else:
        cfit1 = None
    
    n1000 = 0
    if nran2 > 100:
        while n1000 <= 8:
            coef2 = np.polyfit(wran2, ofl2, 4)
            cfit2 = np.polyval(coef2, wran2)
            ysig2 = np.std(ofl2 - cfit2)
            mask = (ofl2 < cfit2) | (ofl2 > cfit2 + 3.0 * ysig2)
            ofl2[mask] = cfit2[mask]
            n1000 += 1
    else:
        cfit2 = None
    
    totwran = ww.copy()
    totcfit = ff.copy()
    if nran1 > 100:
        totcfit[wran1ind] = cfit1
    if nran2 > 100:
        totcfit[wran2ind] = cfit2
    totcfit[totcfit <= 0.0] = 1.0
    
    cc = totcfit
    return cc

class SpectralRetriever:
    """
    Main class for spectral retrieval and cross-modal analysis.
    """
    
    def __init__(
        self,
        xp_encoder_path: str = None,
        lrs_encoder_path: str = None,
        h5_data_path: Optional[str] = None,
        device: Optional[str] = None,
        # New dual model support
        specclip_predrecon_path: Optional[str] = None,
        specclip_split_path: Optional[str] = None,
        use_split_for_retrieval: bool = True
    ):
        """
        Initialize the spectral retriever.

        Args:
            xp_encoder_path: Path to image encoder (Gaia XP) model
            lrs_encoder_path: Path to spectrum encoder (LAMOST LRS) model
            h5_data_path: Optional path to HDF5 file with test data
            device: Device to use ('cuda' or 'cpu')
            specclip_predrecon_path: Path to predrecon model (unified embeddings)
            specclip_split_path: Path to split model (shared+private embeddings)
            use_split_for_retrieval: If True, use split model's shared embedding for retrieval
        """

        self.specclip_predrecon_path = Path(specclip_predrecon_path) if specclip_predrecon_path else None
        self.specclip_split_path = Path(specclip_split_path) if specclip_split_path else None
        self.xp_encoder_path = Path(xp_encoder_path) if xp_encoder_path else None
        self.lrs_encoder_path = Path(lrs_encoder_path) if lrs_encoder_path else None
        self.h5_data_path = Path(h5_data_path) if h5_data_path else None
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_split_for_retrieval = use_split_for_retrieval

        print(f"Using device: {self.device}")
        print("Loading SpecClip model(s)...")
        self._load_models()
        print("Models loaded successfully!")
        
        # Placeholders for embedding databases
        self.gaia_embeddings = None
        self.lamost_embeddings = None
        self.gaia_spectra = None
        self.lamost_spectra = None
        self.source_ids = None
        
        # Wavelength grids
        self.gaia_wavelengths = np.arange(336, 1021, 2) * 10  # nm to Angstrom
        w_start = 3.602
        w_end = w_start + 1e-4 * 1461
        new_log_wave = np.linspace(w_start, w_end, 1462)
        self.lamost_wavelengths = 10 ** new_log_wave
        
    def _load_models(self):
        """Load the SpecClip models (predrecon and optionally split)"""
        # Load predrecon model
        if self.specclip_predrecon_path:
            gaia_xp_encoder, lamost_lrs_encoder = self._initialize_encoders()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model_predrecon = SpecClipModel.load_from_checkpoint(
                    checkpoint_path=str(self.specclip_predrecon_path),
                    gaia_xp_encoder=gaia_xp_encoder,
                    lamost_lrs_encoder=lamost_lrs_encoder,
                    strict=True
                )
            self.model_predrecon = self.model_predrecon.to(self.device)
            self.model_predrecon.eval()
        else:
            self.model_predrecon = None

        # Load split model
        if self.specclip_split_path:
            from specclip.models.specclip_reconstruct_split_5122562_mlp_recordloss import (
                SpecClipModel_reconstruct_split_5122562_mlp_recordloss as SpecClipModel_split,
                GaiaXPHead_split, LamostLRSHead_split
            )
            gaia_xp_encoder_split, lamost_lrs_encoder_split = self._initialize_encoders_split()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model_split = SpecClipModel_split.load_from_checkpoint(
                    checkpoint_path=str(self.specclip_split_path),
                    gaia_xp_encoder=gaia_xp_encoder_split,
                    lamost_lrs_encoder=lamost_lrs_encoder_split,
                    strict=True
                )
            self.model_split = self.model_split.to(self.device)
            self.model_split.eval()
        else:
            self.model_split = None

        # Set default model for retrieval
        if self.use_split_for_retrieval and self.model_split:
            self.model = self.model_split
        elif self.model_predrecon:
            self.model = self.model_predrecon
        else:
            raise ValueError("No model loaded. Provide specclip_predrecon_path or specclip_split_path.")

    def _initialize_encoders_split(
        self,
        shared_embed_dim: int = 512,
        private_embed_dim: int = 256,
        n_head: int = 4,
        model_embed_dim: int = 768,
        dropout: float = 0.1,
        freeze_backbone: bool = True
    ) -> Tuple[nn.Module, nn.Module]:
        """Initialize split encoders for shared/private representations"""
        from specclip.models.specclip_reconstruct_split_5122562_mlp_recordloss import (
            GaiaXPHead_split, LamostLRSHead_split
        )

        gaia_xp_encoder = GaiaXPHead_split(
            model_path=str(self.xp_encoder_path),
            shared_embed_dim=shared_embed_dim,
            private_embed_dim=private_embed_dim,
            n_head=n_head,
            model_embed_dim=model_embed_dim,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
            load_pretrained_weights=False
        )

        lamost_lrs_encoder = LamostLRSHead_split(
            model_path=str(self.lrs_encoder_path),
            shared_embed_dim=shared_embed_dim,
            private_embed_dim=private_embed_dim,
            n_head=n_head,
            model_embed_dim=model_embed_dim,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
            load_pretrained_weights=False
        )

        return gaia_xp_encoder, lamost_lrs_encoder
    
    def _initialize_encoders(
        self,
        embed_dim: int = 768,
        n_head: int = 4,
        model_embed_dim: int = 768,
        dropout: float = 0.1,
        freeze_backbone: bool = True
    ) -> Tuple[nn.Module, nn.Module]:
        """Initialize both encoders"""
        gaia_xp_encoder = GaiaXPHead(
            model_path=str(self.xp_encoder_path),
            embed_dim=embed_dim,
            n_head=n_head,
            model_embed_dim=model_embed_dim,
            dropout=dropout,
            freeze_backbone=freeze_backbone
        )
        
        lamost_lrs_encoder = LamostLRSHead(
            model_path=str(self.lrs_encoder_path),
            embed_dim=embed_dim,
            n_head=n_head,
            model_embed_dim=model_embed_dim,
            dropout=dropout,
            freeze_backbone=freeze_backbone
        )
        
        return gaia_xp_encoder, lamost_lrs_encoder

    def _ensure_2d_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Ensure embedding is 2D with shape (n_samples, n_features)"""
        if embedding.ndim == 1:
            return embedding.reshape(1, -1)
        elif embedding.ndim == 3:
            # Shape like (1, n_features, 1) -> (1, n_features)
            return embedding.squeeze(-1) if embedding.shape[-1] == 1 else embedding.squeeze(1)
        return embedding
    
    def build_embedding_database(
        self,
        h5_data_path: Optional[str] = None,
        batch_size: int = 1000,
        save_path: Optional[str] = None
    ):
        """
        Build embedding database from test spectra in HDF5 file.
        
        Args:
            h5_data_path: Path to HDF5 file (uses self.h5_data_path if None)
            batch_size: Batch size for processing
            save_path: Optional path to save embeddings
            
        Returns:
            Dictionary with embeddings and spectra
        """
        if h5_data_path is None:
            h5_data_path = self.h5_data_path
        
        if h5_data_path is None:
            raise ValueError("No HDF5 data path provided")
        
        print(f"Loading test data from {h5_data_path}")
        
        with h5py.File(h5_data_path, 'r') as f:
            self.source_ids = np.array(f['test/source_ids'][:])
            self.gaia_spectra = np.array(f['test/gaia_spectra'][:])
            self.lamost_spectra = np.array(f['test/lamost_spectra'][:])
        
        print(f"Loaded {len(self.source_ids)} test spectra")
        
        # Generate embeddings
        print("Generating Gaia XP embeddings...")
        self.gaia_embeddings = self._generate_embeddings_batch(
            self.gaia_spectra, 'gaia_spectra', batch_size
        )
        
        print("Generating LAMOST LRS embeddings...")
        self.lamost_embeddings = self._generate_embeddings_batch(
            self.lamost_spectra, 'lamost_spectra', batch_size
        )
        
        print("✓ Embedding database built successfully!")
        
        # Save if requested
        if save_path:
            self.save_embeddings(save_path)
        
        return {
            'gaia_embeddings': self.gaia_embeddings,
            'lamost_embeddings': self.lamost_embeddings,
            'gaia_spectra': self.gaia_spectra,
            'lamost_spectra': self.lamost_spectra,
            'source_ids': self.source_ids
        }
    
    def _generate_embeddings_batch(
        self,
        spectra: np.ndarray,
        input_type: str,
        batch_size: int
    ) -> np.ndarray:
        """Generate embeddings in batches"""
        embeddings_list = []
        total = len(spectra)
        
        for start_idx in tqdm(range(0, total, batch_size), desc=f"Processing {input_type}"):
            end_idx = min(start_idx + batch_size, total)
            batch = spectra[start_idx:end_idx]
            
            batch_tensor = torch.from_numpy(batch.astype(np.float32)).to(self.device)
            
            if len(batch_tensor.shape) == 2:
                batch_tensor = batch_tensor.unsqueeze(-1)
            
            with torch.no_grad():
                batch_embeddings = self.model(batch_tensor, input_type=input_type)
                if self.use_split_for_retrieval and self.model_split:
                    embeddings_list.append(batch_embeddings[0].cpu().numpy()) # only shared information
                else:
                    embeddings_list.append(batch_embeddings.cpu().numpy()) 
        
        return np.concatenate(embeddings_list, axis=0)
    
    def interpolate_lamost_spectrum(
        self,
        wavelength: np.ndarray,
        flux: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolate LAMOST spectrum to model input format.
        Applies GASPP continuum normalization BEFORE interpolation.
        """
        if len(wavelength) == 0 or len(flux) == 0:
            raise ValueError("Input arrays cannot be empty")
    
        if len(wavelength) != len(flux):
            raise ValueError(f"Length mismatch: wavelength ({len(wavelength)}) != flux ({len(flux)})")
    
        if not np.all(np.diff(wavelength) > 0):
            raise ValueError("Wavelength array must be strictly increasing")
    
        # Apply GASPP continuum normalization (same as stellar_params_lrs.py)
        continuum_gaspp = gaspp_fitcont2(wavelength, flux)
        flux_normed = flux / continuum_gaspp
    
        # Now interpolate the normalized flux
        log_wave = np.log10(wavelength)
        f = interp1d(log_wave, flux_normed, kind='linear', bounds_error=True)
    
        w_start = 3.602
        w_end = w_start + 1e-4 * 1461
        new_log_wave = np.linspace(w_start, w_end, 1462)
    
        new_flux = f(new_log_wave)
        new_wavelength = 10.0 ** new_log_wave
    
        return new_wavelength, new_flux
    
    def interpolate_gaia_spectrum(
        self,
        wavelength: np.ndarray,
        flux: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolate Gaia XP spectrum to model input format.
        Same as in stellar_params_xp.py
        """
        if len(wavelength) == 0 or len(flux) == 0:
            raise ValueError("Input arrays cannot be empty")
        
        if len(wavelength) != len(flux):
            raise ValueError(f"Length mismatch")
        
        if not np.all(np.diff(wavelength) > 0):
            raise ValueError("Wavelength array must be strictly increasing")
        
        f = interp1d(wavelength, flux, kind='linear', bounds_error=True)
        new_wave = np.arange(336, 1021, 2)
        new_flux = f(new_wave)
        new_flux /= new_flux[107]  # Normalize at 550nm
        
        return new_wave, new_flux
    
    def encode_spectrum(
        self,
        spectrum: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
        input_type: str
    ) -> np.ndarray:
        """
        Encode a single spectrum to embedding.
        
        Args:
            spectrum: Either flux array (if from test set) or (wavelength, flux) tuple
            input_type: 'gaia_spectra' or 'lamost_spectra'
            
        Returns:
            Embedding vector
        """
        # Handle different input formats
        if isinstance(spectrum, tuple):
            wavelength, flux = spectrum
            # Interpolate to model format
            if input_type == 'lamost_spectra':
                _, flux = self.interpolate_lamost_spectrum(wavelength, flux)
            else:
                _, flux = self.interpolate_gaia_spectrum(wavelength, flux)
            spectrum = flux
        
        # Convert to tensor
        spec_tensor = torch.from_numpy(
            np.array(spectrum, dtype=np.float32)
        ).to(self.device)
        
        if len(spec_tensor.shape) == 1:
            spec_tensor = spec_tensor.unsqueeze(0).unsqueeze(-1)
        elif spec_tensor.dim() == 2:
            # Shape: (1, N) or (N, 1)
            if spec_tensor.shape[0] == 1:
                # (1, N) -> (1, N, 1)
                spec_tensor = spec_tensor.unsqueeze(-1)
            else:
                # (N, 1) -> (1, N, 1)
                spec_tensor = spec_tensor.unsqueeze(0)
        elif spec_tensor.dim() == 3:
            # Already correct shape
            pass
        
        with torch.no_grad():
            embedding = self.model(spec_tensor, input_type=input_type)

        if self.use_split_for_retrieval and self.model_split:
            embedding_np = embedding[0].cpu().numpy() # only shared embeddings
        else:
            embedding_np = embedding.cpu().numpy() 
        return self._ensure_2d_embedding(embedding_np)
        
        #return embedding.cpu().numpy()
    
    def find_similar_spectra(
        self,
        query_spectrum: Union[np.ndarray, Tuple[np.ndarray, np.ndarray], int],
        query_type: str,
        search_type: str,
        top_k: int = 4,
        exclude_self: bool = False
    ) -> Dict:
        """
        Find similar spectra.
        
        Args:
            query_spectrum: Query spectrum (flux array, (wave, flux) tuple, or test index)
            query_type: 'gaia_spectra' or 'lamost_spectra'
            search_type: 'in_modal' or 'cross_modal'
            top_k: Number of matches to return
            exclude_self: Exclude query from results (for test set queries)
            
        Returns:
            Dictionary with matches, scores, and query info
        """
        # Check if embeddings are built
        if self.gaia_embeddings is None or self.lamost_embeddings is None:
            raise ValueError("Embedding database not built. Call build_embedding_database() first.")
        
        # Handle query input
        query_idx = None
        if isinstance(query_spectrum, int):
            # Query is an index in test set
            query_idx = query_spectrum
            if query_type == 'gaia_spectra':
                query_spectrum_data = self.gaia_spectra[query_idx]
                query_embedding = self.gaia_embeddings[query_idx:query_idx+1]
            else:
                query_spectrum_data = self.lamost_spectra[query_idx]
                query_embedding = self.lamost_embeddings[query_idx:query_idx+1]
        else:
            # External spectrum
            query_embedding = self.encode_spectrum(query_spectrum, query_type)
            if isinstance(query_spectrum, tuple):
                _, query_spectrum_data = (self.interpolate_lamost_spectrum(*query_spectrum) 
                                         if query_type == 'lamost_spectra' 
                                         else self.interpolate_gaia_spectrum(*query_spectrum))
            else:
                query_spectrum_data = query_spectrum
        
        # Determine search database
        if search_type == 'in_modal':
            if query_type == 'gaia_spectra':
                search_embeddings = self.gaia_embeddings
                search_spectra = self.gaia_spectra
            else:
                search_embeddings = self.lamost_embeddings
                search_spectra = self.lamost_spectra
        else:  # cross_modal
            if query_type == 'gaia_spectra':
                search_embeddings = self.lamost_embeddings
                search_spectra = self.lamost_spectra
            else:
                search_embeddings = self.gaia_embeddings
                search_spectra = self.gaia_spectra
        
        # Calculate similarities
        # Ensure query_embedding is 2D: (1, embedding_dim)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        similarities = cosine_similarity(query_embedding, search_embeddings)[0]
        
        # Exclude self if needed
        if exclude_self and query_idx is not None:
            similarities[query_idx] = -np.inf
        
        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_scores = similarities[top_indices]
        top_spectra = search_spectra[top_indices]
        
        return {
            'query_spectrum': query_spectrum_data,
            'query_idx': query_idx,
            'query_type': query_type,
            'search_type': search_type,
            'top_indices': top_indices,
            'top_scores': top_scores,
            'top_spectra': top_spectra,
            'source_ids': self.source_ids[top_indices] if self.source_ids is not None else None
        }
    
    def predict_cross_modal(
        self,
        query_spectrum: Union[np.ndarray, Tuple[np.ndarray, np.ndarray], int],
        query_type: str
    ) -> Dict:
        """
        Predict cross-modal spectrum.
        
        Args:
            query_spectrum: Query spectrum
            query_type: 'gaia_spectra' or 'lamost_spectra'
            
        Returns:
            Dictionary with query and predicted spectrum
        """
        # Encode query
        if isinstance(query_spectrum, int):
            query_idx = query_spectrum
            if query_type == 'gaia_spectra':
                query_spectrum_data = self.gaia_spectra[query_idx]
                query_embedding = self.gaia_embeddings[query_idx:query_idx+1]
            else:
                query_spectrum_data = self.lamost_spectra[query_idx]
                query_embedding = self.lamost_embeddings[query_idx:query_idx+1]
        else:
            query_embedding = self.encode_spectrum(query_spectrum, query_type)
            if isinstance(query_spectrum, tuple):
                _, query_spectrum_data = (self.interpolate_lamost_spectrum(*query_spectrum) 
                                         if query_type == 'lamost_spectra' 
                                         else self.interpolate_gaia_spectrum(*query_spectrum))
            else:
                query_spectrum_data = query_spectrum
            
        # Ensure query_embedding is 2D: (batch_size, embedding_dim)
        query_embedding = self._ensure_2d_embedding(query_embedding)
        
        # Generate cross-modal prediction
        query_embedding_tensor = torch.from_numpy(query_embedding.astype(np.float32)).to(self.device)

        # Ensure tensor has correct shape for decoder
        # Decoders expect: (batch_size, embedding_dim) 
        if query_embedding_tensor.dim() == 1:
            query_embedding_tensor = query_embedding_tensor.unsqueeze(0)
        elif query_embedding_tensor.dim() == 3:
            query_embedding_tensor = query_embedding_tensor.squeeze(-1)
        
        with torch.no_grad():
            if query_type == 'lamost_spectra':
                # LAMOST -> Gaia
                prediction = self.model.gaia_xp_cross_decoder(query_embedding_tensor)
                #model = self.model  
                # Show ALL attributes (methods + modules)
                #print(dir(model))

                pred_type = 'gaia_spectra'
            else:
                # Gaia -> LAMOST
                prediction = self.model.lamost_lrs_cross_decoder(query_embedding_tensor)
                pred_type = 'lamost_spectra'
        
        predicted_spectrum = prediction.cpu().numpy().squeeze()
        
        # Get ground truth if from test set
        ground_truth = None
        if isinstance(query_spectrum, int):
            if query_type == 'gaia_spectra':
                ground_truth = self.lamost_spectra[query_spectrum]
            else:
                ground_truth = self.gaia_spectra[query_spectrum]
        
        return {
            'query_spectrum': query_spectrum_data,
            'query_type': query_type,
            'predicted_spectrum': predicted_spectrum,
            'predicted_type': pred_type,
            'ground_truth': ground_truth
        }
    
    def plot_retrieval_results(
        self,
        results: Dict,
        figsize: Tuple[int, int] = (15, 6),
        save_path: Optional[str] = None
    ):
        """
        Plot retrieval results.
        
        Args:
            results: Results from find_similar_spectra()
            figsize: Figure size
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        query_type = results['query_type']
        search_type = results['search_type']
        
        # Determine wavelengths
        if query_type == 'gaia_spectra':
            query_wave = self.gaia_wavelengths
        else:
            query_wave = self.lamost_wavelengths
        
        if search_type == 'cross_modal':
            search_wave = self.lamost_wavelengths if query_type == 'gaia_spectra' else self.gaia_wavelengths
        else:
            search_wave = query_wave
        
        # Plot query spectrum
        axes[0].plot(query_wave, results['query_spectrum'], 'k-', linewidth=2, label='Query')
        axes[0].set_title(f'Query Spectrum ({query_type})')
        axes[0].set_xlabel('Wavelength (Å)')
        axes[0].set_ylabel('Flux')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot retrieved spectra
        axes[1].plot(query_wave, results['query_spectrum'], 'k-', linewidth=2.5, 
                    label='Query', alpha=0.8)
        
        for i, (spectrum, score) in enumerate(zip(results['top_spectra'], results['top_scores'])):
            axes[1].plot(search_wave, spectrum, linewidth=1.5, alpha=0.7, 
                        label=f'Match #{i+1} (score: {score:.3f})')
        
        title = f'{search_type.replace("_", "-").title()} Search Results'
        axes[1].set_title(title)
        axes[1].set_xlabel('Wavelength (Å)')
        axes[1].set_ylabel('Flux')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        plt.show()
    
    def plot_cross_modal_prediction(
        self,
        results: Dict,
        figsize: Tuple[int, int] = (15, 6),
        save_path: Optional[str] = None
    ):
        """
        Plot cross-modal prediction results.
        
        Args:
            results: Results from predict_cross_modal()
            figsize: Figure size
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        query_type = results['query_type']
        pred_type = results['predicted_type']
        
        query_wave = self.gaia_wavelengths if query_type == 'gaia_spectra' else self.lamost_wavelengths
        pred_wave = self.lamost_wavelengths if pred_type == 'lamost_spectra' else self.gaia_wavelengths
        
        # Plot query
        axes[0].plot(query_wave, results['query_spectrum'], 'b-', linewidth=2)
        axes[0].set_title(f'Query Spectrum ({query_type})')
        axes[0].set_xlabel('Wavelength (Å)')
        axes[0].set_ylabel('Flux')
        axes[0].grid(True, alpha=0.3)
        
        # Plot prediction and ground truth
        axes[1].plot(pred_wave, results['predicted_spectrum'], 'r-', linewidth=2, 
                    label='Predicted', alpha=0.8)
        
        if results['ground_truth'] is not None:
            axes[1].plot(pred_wave, results['ground_truth'], 'g-', linewidth=2, 
                        label='Ground Truth', alpha=0.8)
        
        axes[1].set_title(f'Cross-Modal Prediction ({pred_type})')
        axes[1].set_xlabel('Wavelength (Å)')
        axes[1].set_ylabel('Flux')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        plt.show()
    
    def save_embeddings(self, save_path: str):
        """Save embedding database to file"""
        np.savez(
            save_path,
            gaia_embeddings=self.gaia_embeddings,
            lamost_embeddings=self.lamost_embeddings,
            gaia_spectra=self.gaia_spectra,
            lamost_spectra=self.lamost_spectra,
            source_ids=self.source_ids
        )
        print(f"✓ Saved embeddings to {save_path}")
    
    def load_embeddings(self, load_path: str):
        """Load embedding database from file"""
        data = np.load(load_path)
        self.gaia_embeddings = data['gaia_embeddings']
        self.lamost_embeddings = data['lamost_embeddings']
        self.gaia_spectra = data['gaia_spectra']
        self.lamost_spectra = data['lamost_spectra']
        self.source_ids = data['source_ids']
        print(f"✓ Loaded embeddings from {load_path}")
        

def get_default_model_paths():
    """Get default model paths"""
    return {
        'xp_encoder_path': 'xp_encoder_path',
        'lrs_encoder_path': 'lrs_encoder_path',
        # Two specclip models for different use cases
        'specclip_predrecon_path': 'specclip_predrecon_path',
        'specclip_split_path': 'specclip_split_path',
        # Note: h5_data_path is now downloaded separately using get_default_test_data_path()
    }


def get_default_test_data_path(local_dir="./test_data"):
    """
    Get default test data path, downloading from HuggingFace if needed.

    Args:
        local_dir: Local directory for test data

    Returns:
        Path to test data HDF5 file
    """
    import sys
    from pathlib import Path

    # Add parent directory to path to import download_test_data
    parent_dir = str(Path(__file__).parent.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    from download_test_data import get_test_data_path
    return get_test_data_path(local_dir=local_dir)