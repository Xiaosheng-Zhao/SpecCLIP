"""
SpecCLIP: A contrastive learning model for matching Gaia XP and LAMOST LRS spectra.

This package provides tools for training and using SpecCLIP models to create
aligned embeddings of different types of astronomical spectra.
"""

from specclip.models import (
    SpecClipModel,
    SpecClipModel_mlp,
    GaiaXPHead,
    LamostLRSHead,
    GaiaXPHeadWithMLP,
    LamostLRSHeadWithMLP,
)

__version__ = "1.0.0"

__all__ = [
    "SpecClipModel",
    "SpecClipModel_mlp",
    "GaiaXPHead",
    "LamostLRSHead",
    "GaiaXPHeadWithMLP",
    "LamostLRSHeadWithMLP",
]
