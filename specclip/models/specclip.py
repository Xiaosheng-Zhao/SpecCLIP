# =============================================================================
# SpecCLIP-base Pretraining Module
# 
# The SpecCLIP-base model contains only:
#   • contrastive loss,
# to learn a shared latent representations across modalities.
# 
# It uses modality-specific pre-trained encoders:
#   • Gaia XP encoder: masked-transformer objective
#   • LAMOST LRS encoder: masked-transformer objective
#
# Portions of this implementation are adapted from AstroCLIP
# (Liam et al. 2024): https://github.com/waqarsyed/astroclip
# =============================================================================

import os
import sys
from typing import Tuple

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules import MLP, CrossAttentionHead
from .specformer_control import SpecFormerControl1_stats as SpecFormer_xp
from .specformer_control import SpecFormerControl20_wstd as SpecFormer_lm

from torch import Tensor

class SpecClipModel(L.LightningModule):
    def __init__(
        self,
        gaia_xp_encoder: nn.Module,
        lamost_lrs_encoder: nn.Module,
        temperature: float = 15.5,
        lr: float = 1e-4,
        weight_decay: float = 0.05,
        epochs: int = 100,
        eta_min: float = 5e-7,
        logit_scale: float = 15.5,
        learnable_logit_scale: bool = False,
    ):
        """
        The SpecCLIP model that takes two types of spectra (Gaia XP and LAMOST LRS) and embeds them
        into a common space using CLIP loss.
        Note that you must provide the Gaia XP and LAMOST LRS encoders to be used for the embedding.

        Args:
            gaia_xp_encoder (nn.Module): The Gaia XP encoder to be used for embedding.
            lamost_lrs_encoder (nn.Module): The LAMOST LRS encoder to be used for embedding.
            temperature (float): The temperature parameter for the CLIP loss.
            lr (float): The learning rate for the optimizer.
            weight_decay (float): The weight decay for the optimizer.
            epochs (int): The number of epochs for training.
            eta_min (float): The minimum learning rate for the scheduler.
            logit_scale (float): The logit scale for the CLIP loss.
            learnable_logit_scale (bool): Whether the logit scale should be learnable.
        """
        super().__init__()
        self.save_hyperparameters()

        # Define the Gaia XP and LAMOST LRS encoders
        self.gaia_xp_encoder = gaia_xp_encoder
        self.lamost_lrs_encoder = lamost_lrs_encoder

        # Logit scale is fixed to 15.5 and is not a learnable parameter
        if not learnable_logit_scale:
            self.logit_scale = np.log(logit_scale)
        else:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(logit_scale))

        # Use CLIP loss
        self.criterion = CLIPLoss()

    def forward(
        self,
        input: torch.Tensor,
        input_type: str,
    ):
        """
        Forward pass through the appropriate encoder based on input type.

        Args:
            input: Input spectrum tensor
            input_type: Type of input spectrum ('gaia_spectra' or 'lamost_spectra')

        Returns:
            Encoded features
        """
        if input_type == "gaia_spectra":
            return self.gaia_xp_encoder(input)

        elif input_type == "lamost_spectra":
            return self.lamost_lrs_encoder(input)

        else:
            raise ValueError("Input type must be either 'gaia_spectra' or 'lamost_spectra'")

    def training_step(self, batch, batch_idx):
        gaia_xp, lamost_lrs = batch["gaia_spectra"], batch["lamost_spectra"]

        # Get the Gaia XP and LAMOST LRS features
        gaia_xp_features = self.gaia_xp_encoder(gaia_xp)
        lamost_lrs_features = self.lamost_lrs_encoder(lamost_lrs)

        # Calculate the CLIP loss
        loss_withlogit = self.criterion(
            gaia_xp_features, lamost_lrs_features, self.hparams.temperature
        )
        loss_nologit = self.criterion(
            gaia_xp_features, lamost_lrs_features, self.hparams.logit_scale
        )

        # Log the losses
        self.log("train_loss_withlogit", loss_withlogit)
        #self.log("train_loss_nologit", loss_nologit)
        #self.log("scale", self.logit_scale)

        # Return the loss
        return loss_withlogit

    def training_epoch_end(self, outputs):
        if outputs:
            # Determine if the first element is a dictionary or a tensor
            if isinstance(outputs[0], dict):
                # Handle the case where outputs are dictionaries
                losses = [x['loss'] for x in outputs if 'loss' in x]
                avg_train_loss = torch.stack(losses).mean()
            elif isinstance(outputs[0], torch.Tensor):
                # Handle the case where outputs are tensors
                avg_train_loss = torch.stack(outputs).mean()
            else:
                print("Unsupported data type in outputs.")
                return

            # Log the average training loss
            self.log('train_epoch_loss', avg_train_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        else:
            print("No valid outputs received in training_epoch_end")

        # Log the average training loss
        self.log('training_loss', avg_train_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        gaia_xp, lamost_lrs = batch["gaia_spectra"], batch["lamost_spectra"]

        # Get the Gaia XP and LAMOST LRS features
        gaia_xp_features = self.gaia_xp_encoder(gaia_xp)
        lamost_lrs_features = self.lamost_lrs_encoder(lamost_lrs)

        # Calculate the CLIP loss
        val_loss_nologit = self.criterion(
            gaia_xp_features, lamost_lrs_features, self.hparams.logit_scale
        )
        val_loss_withlogit = self.criterion(
            gaia_xp_features, lamost_lrs_features, self.hparams.temperature
        )

        # Log the losses
        #self.log("val_loss_nologit", val_loss_nologit)
        self.log("val_loss_withlogit", val_loss_withlogit)

        return val_loss_withlogit

    def validation_epoch_end(self, outputs):
        if outputs:
            if isinstance(outputs[0], dict):
                losses = [x['val_loss'] for x in outputs if 'val_loss' in x]
                avg_val_loss = torch.stack(losses).mean()
            elif isinstance(outputs[0], torch.Tensor):
                avg_val_loss = torch.stack(outputs).mean()
            else:
                print("Unsupported data type in validation outputs.")
                return

            # Log the average validation loss with epoch
            epoch_info = f"Epoch {self.current_epoch}: "
            self.log('val_epoch_loss', avg_val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            print(epoch_info + f"Average Validation Loss: {avg_val_loss.item()}")
        else:
            print("No valid outputs received in validation_epoch_end")

class CLIPLoss(nn.Module):
    """CLIP contrastive loss for matching Gaia XP and LAMOST LRS spectra"""

    def get_logits(
        self,
        gaia_xp_features: torch.FloatTensor,
        lamost_lrs_features: torch.FloatTensor,
        logit_scale: float,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Calculate logits for CLIP loss between Gaia XP and LAMOST LRS features.

        Args:
            gaia_xp_features: Features from Gaia XP encoder
            lamost_lrs_features: Features from LAMOST LRS encoder
            logit_scale: Scale factor for logits

        Returns:
            Tuple of (logits_per_gaia_xp, logits_per_lamost_lrs)
        """
        # Normalize Gaia XP features
        gaia_xp_features = F.normalize(gaia_xp_features, dim=-1, eps=1e-3)

        # Normalize LAMOST LRS features
        lamost_lrs_features = F.normalize(lamost_lrs_features, dim=-1, eps=1e-3)

        # Calculate the logits for the Gaia XP and LAMOST LRS features
        logits_per_gaia_xp = logit_scale * gaia_xp_features @ lamost_lrs_features.T
        return logits_per_gaia_xp, logits_per_gaia_xp.T

    def forward(
        self,
        gaia_xp_features: torch.FloatTensor,
        lamost_lrs_features: torch.FloatTensor,
        logit_scale: float,
        output_dict: bool = False,
    ) -> torch.FloatTensor:
        """
        Calculate CLIP loss between Gaia XP and LAMOST LRS features.

        Args:
            gaia_xp_features: Features from Gaia XP encoder
            lamost_lrs_features: Features from LAMOST LRS encoder
            logit_scale: Scale factor for logits
            output_dict: Whether to return dict or single loss value

        Returns:
            Contrastive loss value (or dict if output_dict=True)
        """
        # Get the logits for the Gaia XP and LAMOST LRS features
        logits_per_gaia_xp, logits_per_lamost_lrs = self.get_logits(
            gaia_xp_features, lamost_lrs_features, logit_scale
        )

        # Calculate the contrastive loss
        labels = torch.arange(
            logits_per_gaia_xp.shape[0], device=gaia_xp_features.device, dtype=torch.long
        )
        total_loss = (
            F.cross_entropy(logits_per_gaia_xp, labels)
            + F.cross_entropy(logits_per_lamost_lrs, labels)
        ) / 2
        return {"contrastive_loss": total_loss} if output_dict else total_loss

class LamostLRSHead(nn.Module):
    """Encoder head for LAMOST LRS spectra using cross-attention"""

    def __init__(
        self,
        model_path: str,
        embed_dim: int = 1024,
        n_head: int = 4,
        model_embed_dim: int = 768,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
        load_pretrained_weights=True,
    ):
        """
        Cross-attention module for LAMOST LRS spectra that passes them through a pretrained SpecFormer model
        and then through a cross-attention mechanism and MLP to get the final embedding.

        Args:
            model_path (str): Path to the checkpoint of the SpecFormer model.
            embed_dim (int): Dimension of the SpecCLIP embedding.
            n_head (int): Number of heads in the multihead attention.
            model_embed_dim (int): Dimension of the SpecFormer embedding.
            dropout (float): Dropout rate for MLP layers.
            freeze_backbone (bool): Whether to freeze the backbone of the SpecFormer model.
            load_pretrained_weights (bool): Whether to load pretrained weights from checkpoint.
        """
        super().__init__()
        # Load the model from the checkpoint
        checkpoint = torch.load(model_path)
        self.backbone = SpecFormer_lm(**checkpoint["hyper_parameters"])
        if load_pretrained_weights:
            self.backbone.load_state_dict(checkpoint["state_dict"])

        # Freeze backbone if necessary
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Set up cross-attention
        self.cross_attention = CrossAttentionHead(
            embed_dim=embed_dim,
            n_head=n_head,
            model_embed_dim=model_embed_dim,
            dropout=dropout,
        )

        # Set up MLP
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=4 * embed_dim,
            dropout=dropout,
        )

    def forward(
        self, x: torch.tensor, y: torch.tensor = None, return_weights: bool = False
    ):
        """
        Forward pass through LAMOST LRS encoder.

        Args:
            x: Input LAMOST LRS spectrum
            y: Optional auxiliary input (unused)
            return_weights: Whether to return attention weights

        Returns:
            Encoded features (and optionally attention weights)
        """
        # Embed the spectrum using the pretrained model
        with torch.set_grad_enabled(not self.freeze_backbone):
            embedding = self.backbone(x)["embedding"]

        # Pass through cross-attention
        x, attentions = self.cross_attention(embedding)

        # Pass through MLP and residual connection
        x = x + self.mlp(x)

        if return_weights:
            return x.squeeze(), attentions[1]

        return x.squeeze()

class GaiaXPHead(nn.Module):
    """Encoder head for Gaia XP spectra using cross-attention"""

    def __init__(
        self,
        model_path: str,
        embed_dim: int = 1024,
        n_head: int = 4,
        model_embed_dim: int = 768,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
        load_pretrained_weights=True,
    ):
        """
        Cross-attention module for Gaia XP spectra that passes them through a pretrained SpecFormer model
        and then through a cross-attention mechanism and MLP to get the final embedding.

        Args:
            model_path (str): Path to the checkpoint of the SpecFormer model.
            embed_dim (int): Dimension of the SpecCLIP embedding.
            n_head (int): Number of heads in the multihead attention.
            model_embed_dim (int): Dimension of the SpecFormer embedding.
            dropout (float): Dropout rate for MLP layers.
            freeze_backbone (bool): Whether to freeze the backbone of the SpecFormer model.
            load_pretrained_weights (bool): Whether to load pretrained weights from checkpoint.
        """
        super().__init__()
        # Load the model from the checkpoint
        checkpoint = torch.load(model_path)
        self.backbone = SpecFormer_xp(**checkpoint["hyper_parameters"])
        if load_pretrained_weights:
            self.backbone.load_state_dict(checkpoint["state_dict"])

        # Freeze backbone if necessary
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Set up cross-attention
        self.cross_attention = CrossAttentionHead(
            embed_dim=embed_dim,
            n_head=n_head,
            model_embed_dim=model_embed_dim,
            dropout=dropout,
        )

        # Set up MLP
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=4 * embed_dim,
            dropout=dropout,
        )

    def forward(
        self, x: torch.tensor, y: torch.tensor = None, return_weights: bool = False
    ):
        """
        Forward pass through Gaia XP encoder.

        Args:
            x: Input Gaia XP spectrum
            y: Optional auxiliary input (unused)
            return_weights: Whether to return attention weights

        Returns:
            Encoded features (and optionally attention weights)
        """
        # Embed the spectrum using the pretrained model
        with torch.set_grad_enabled(not self.freeze_backbone):
            embedding = self.backbone(x)["embedding"]

        # Pass through cross-attention
        x, attentions = self.cross_attention(embedding)

        # Pass through MLP and residual connection
        x = x + self.mlp(x)

        if return_weights:
            return x.squeeze(), attentions[1]

        return x.squeeze()