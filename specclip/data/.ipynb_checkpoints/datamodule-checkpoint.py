# This implementation is adapted from the AstroCLIP framework
# (Liam et al. 2024), with modifications specific to SpecCLIP.
# Original AstroCLIP code: https://github.com/PolymathicAI/AstroCLIP

from typing import Callable, Dict, List, Optional
import lightning as L
import torch
from torch import Tensor
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import CenterCrop
import h5py
from torch.utils.data import Dataset, DataLoader
import numpy as np

from typing import Dict, Tuple, Optional, List
    
class HDF5Dataset(Dataset):
    def __init__(self, 
                 spectra_data: torch.Tensor,
                 ivar_data: Optional[torch.Tensor] = None):
        """
        Simplified dataset class for spectral data with optional inverse variance.
        
        Args:
            spectra_data: Spectra tensor [N, wavelength_points, 1]
            ivar_data: Optional inverse variance data [N, wavelength_points, 1]
        """
        self.spectra = spectra_data
        self.ivar = ivar_data
            
    def __len__(self) -> int:
        return len(self.spectra)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            "spectra": self.spectra[idx],
        }
        if self.ivar is not None:
            item["ivar"] = self.ivar[idx]
        return item

    
class SpectrumDataloader(L.LightningDataModule):
    def __init__(
        self,
        path: str,
        batch_size: int = 512,
        num_workers: int = 10,
        collate_fn: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
    ) -> None:
        """
        Simplified DataLoader for spectroscopic data.
        
        Args:
            path: Path to the HDF5 file containing spectral data
            batch_size: Number of samples per batch
            num_workers: Number of worker processes for data loading
            collate_fn: Optional function to collate samples into batches
        """
        super().__init__()
        self.save_hyperparameters()
        
    def setup(self, stage: str) -> None:
        if ".h5" not in self.hparams.path:
            raise ValueError(f"Warning: Expected .h5 format, but got {self.hparams.path}")
            
        self.dataset = {}
        
        with h5py.File(self.hparams.path, 'r') as f:
            for split in ['train', 'test']:
                # Load spectrum data
                spectra = torch.tensor(f[f'{split}/spectra'][:], dtype=torch.float32)
                
                # Load inverse variance if available
                ivar = None
                if f'{split}/ivar' in f:
                    ivar = torch.tensor(f[f'{split}/ivar'][:], dtype=torch.float32)
                
                # Create dataset
                self.dataset[split] = HDF5Dataset(spectra, ivar)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset["train"],
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            drop_last=True,
            collate_fn=self.hparams.collate_fn,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset["test"],
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=True,
            collate_fn=self.hparams.collate_fn,
        )
    
class MultiSurveySpectralDataset(Dataset):
    def __init__(self, 
                 gaia_spectra: torch.Tensor,
                 lamost_spectra: torch.Tensor):
        """
        Dataset class handling spectral data from multiple surveys (Gaia and LAMOST).
        
        Args:
            gaia_spectra: Gaia spectra tensor [N, wavelength_points_gaia, 1]
            lamost_spectra: LAMOST spectra tensor [N, wavelength_points_lamost, 1]
        """
        self.gaia_spectra = gaia_spectra
        self.lamost_spectra = lamost_spectra
        
    def __len__(self) -> int:
        return len(self.gaia_spectra)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "gaia_spectra": self.gaia_spectra[idx],
            "lamost_spectra": self.lamost_spectra[idx],
        }

class SpecClipDataloader(L.LightningDataModule):
    def __init__(
        self,
        path: str,
        batch_size: int = 512,
        num_workers: int = 10,
        collate_fn: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
    ) -> None:
        """
        DataLoader handling spectroscopic data from Gaia and LAMOST surveys.
        
        Args:
            path: Path to the HDF5 file containing spectral data
            batch_size: Number of samples per batch
            num_workers: Number of worker processes for data loading
            collate_fn: Optional function to collate samples into batches
        """
        super().__init__()
        self.save_hyperparameters()
        
    def setup(self, stage: str) -> None:

        if ".h5" not in self.hparams.path:
            raise ValueError(f"Warning: Expected .h5 format, but got {self.hparams.path}")
        
        self.dataset = {}
            
        with h5py.File(self.hparams.path, 'r') as f:
            for split in ['train', 'test']:
                # Load Gaia and LAMOST spectra
                gaia_spectra = torch.tensor(f[f'{split}/gaia_spectra'][:], dtype=torch.float32)
                lamost_spectra = torch.tensor(f[f'{split}/lamost_spectra'][:], dtype=torch.float32)
                    
                # Create dataset with both Gaia and LAMOST spectra
                self.dataset[split] = MultiSurveySpectralDataset(
                    gaia_spectra=gaia_spectra,
                    lamost_spectra=lamost_spectra
                )


    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset["train"],
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            drop_last=True,
            collate_fn=self.hparams.collate_fn,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset["test"],
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=True,
            collate_fn=self.hparams.collate_fn,
        )


