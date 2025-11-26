from .specformer_control import SpecFormerControl20_wstd, SpectralMLPAutoencoder_xp, SpectralMLPAutoencoder, SpecFormerControl1_stats
from .specclip import SpecClipModel, LamostLRSHead, GaiaXPHead
from .specclip_mlp import SpecClipModel_mlp, GaiaXPHeadWithMLP, LamostLRSHeadWithMLP
from .specclip_reconstruct_embed768_mlp import SpecClipModel_reconstruct_embed768_mlp
from .specclip_reconstruct_split_5122562_mlp_recordloss import SpecClipModel_reconstruct_split_5122562_mlp_recordloss

__all__ = [
    # SpecFormer models
    'SpecFormerControl20_wstd',
    'SpectralMLPAutoencoder_xp',
    'SpectralMLPAutoencoder',
    'SpecFormerControl1_stats',
    # SpecCLIP models
    'SpecClipModel',
    'SpecClipModel_mlp',
    'GaiaXPHead',
    'LamostLRSHead',
    'GaiaXPHeadWithMLP',
    'LamostLRSHeadWithMLP',
    'SpecClipModel_reconstruct_embed768_mlp',
    'SpecClipModel_reconstruct_split_5122562_mlp_recordloss'
]
