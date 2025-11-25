from .specformer_control import SpecFormerControl20_wstd, SpectralMLPAutoencoder_xp, SpectralMLPAutoencoder, SpecFormerControl1_stats
from .specclip_mlp import SpecClipModel_mlp, GaiaXPHeadWithMLP, LamostLRSHeadWithMLP
from .specclip_reconstruct_embed768_mlp import SpecClipModel_reconstruct_embed768_mlp
from .specclip_reconstruct_split_5122562_mlp import SpecClipModel_reconstruct_split_5122562_mlp
from .specclip_reconstruct_embed768 import SpecClipModel_reconstruct_embed768
from .specclip_reconstruct_split_5122562 import SpecClipModel_reconstruct_split_5122562
from .specclip_reconstruct_embed768_mlp_recordloss import SpecClipModel_reconstruct_embed768_mlp_recordloss
from .specclip_reconstruct_split_5122562_mlp_recordloss import SpecClipModel_reconstruct_split_5122562_mlp_recordloss
from .specclip_reconstruct_split_5122562_recordloss import SpecClipModel_reconstruct_split_5122562_recordloss
from .specclip import SpecClipModel, LamostLRSHead, GaiaXPHead

__all__ = [
    # SpecCLIP models
    'SpecClipModel',
    'SpecClipModel_mlp',
    'GaiaXPHead',
    'LamostLRSHead',
    'GaiaXPHeadWithMLP',
    'LamostLRSHeadWithMLP',
    'SpecClipModel_reconstruct_embed768',
    'SpecClipModel_reconstruct_embed768_mlp',
    'SpecClipModel_reconstruct_split_5122562',
    'SpecClipModel_reconstruct_split_5122562_mlp',
    'SpecClipModel_reconstruct_embed768_mlp_recordloss',
    'SpecClipModel_reconstruct_split_5122562_mlp_recordloss',
    'SpecClipModel_reconstruct_split_5122562_recordloss',
    # SpecFormer models
    'SpecFormerControl20_wstd',
    'SpectralMLPAutoencoder_xp',
    'SpectralMLPAutoencoder',
    'SpecFormerControl1_stats',
]
