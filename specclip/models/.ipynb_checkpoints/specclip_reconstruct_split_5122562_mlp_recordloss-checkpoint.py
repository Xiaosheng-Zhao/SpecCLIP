# =============================================================================
# SpecCLIP-split Pretraining Module
# 
# The SpecCLIP-split model combines:
#   • contrastive loss,
#   • reconstruction loss, and
#   • cross-modal prediction loss,
# to learn split (shared+non-shared) latent representations across modalities.
# 
# It uses modality-specific pre-trained encoders:
#   • Gaia XP encoder: ordinary auto-encoders (OAE)-style reconstruction
#   • LAMOST LRS encoder: masked-transformer (MT, basically self-attention + mask modeling) objective
#
# Portions of this implementation are adapted from AstroCLIP
# (Parker et al. 2024): https://github.com/PolymathicAI/AstroCLIP
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
from .specformer_control import SpectralMLPAutoencoder_xp as SpecFormer_xp
from .specformer_control import SpecFormerControl20_wstd as SpecFormer_lm

from torch import Tensor

class SpecClipModel_reconstruct_split_5122562_mlp_recordloss(L.LightningModule):
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
        reconstruction_weight: float = 0.5,
        cross_modal_weight: float = 0.5,
    ):
        """
        The SpecCLIP-split model that takes Gaia XP and LAMOST LRS spectra and embeds them into a common space using CLIP loss, 
        together with additional decoders for reconstruction and cross-modal predictions with a splitted (shared+non-shared) embedding.
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
            reconstruction_weight (float): Weight for reconstruction loss.
            cross_modal_weight (float): Weight for cross-modal prediction loss.
        """
        super().__init__()
        self.save_hyperparameters()

        # Encoders
        self.gaia_xp_encoder = gaia_xp_encoder  # For Gaia XP
        self.lamost_lrs_encoder = lamost_lrs_encoder  # For LAMOST LRS

        # Decoders
        self.lamost_lrs_decoder = LamostLRSDecoder(shared_dim=512,private_dim=256)
        self.gaia_xp_decoder = GaiaXPDecoder(shared_dim=512,private_dim=256)
        self.lamost_lrs_cross_decoder = CrossModalDecoder(shared_dim=512,out_features=1462)  # LAMOST dim
        self.gaia_xp_cross_decoder = CrossModalDecoder(shared_dim=512, out_features=343)  # Gaia XP dim

        # Loss functions
        if not learnable_logit_scale:
            self.logit_scale = np.log(logit_scale)
        else:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(logit_scale))
        
        self.clip_loss = CLIPLoss()
        self.reconstruction_loss = ReconstructionLoss()

        # Initialize lists to store losses for epoch-end logging
        self.training_step_outputs = []
        self.validation_step_outputs = []

    def forward(self, input: torch.Tensor, input_type: str):
        if input_type == "gaia_spectra":
            shared, private = self.gaia_xp_encoder(input)
            return shared, private
        elif input_type == "lamost_spectra":
            shared, private = self.lamost_lrs_encoder(input)
            return shared, private
        else:
            raise ValueError("Input type must be either 'gaia_spectra' or 'lamost_spectra'")

    def training_step(self, batch, batch_idx):
        gaia_spectra, lamost_spectra = batch["gaia_spectra"], batch["lamost_spectra"]

        # The normalized lamost spectra
        std, mean = lamost_spectra.std(1, keepdim=True), lamost_spectra.mean(1, keepdim=True)
        lamost_spectra_normalized = (lamost_spectra - mean) / std

        # Get embeddings from both encoders
        gaia_shared, gaia_private = self.gaia_xp_encoder(gaia_spectra)
        lamost_shared, lamost_private = self.lamost_lrs_encoder(lamost_spectra)

        # 1. CLIP Loss on shared representations
        clip_loss = self.clip_loss(
            gaia_shared, lamost_shared, self.hparams.temperature
        )

        # 2. Reconstruction Losses using both private and shared representations
        gaia_recon = self.gaia_xp_decoder(gaia_shared, gaia_private)
        lamost_recon = self.lamost_lrs_decoder(lamost_shared, lamost_private)
        
        recon_loss = (
            self.reconstruction_loss(gaia_recon, gaia_spectra[:,:,0]) +
            self.reconstruction_loss(lamost_recon, lamost_spectra_normalized[:,:,0])
        )

        # 3. Cross-Modal Prediction Losses using only shared representations
        gaia_from_lamost = self.gaia_xp_cross_decoder(lamost_shared)
        lamost_from_gaia = self.lamost_lrs_cross_decoder(gaia_shared)
        
        cross_modal_loss = (
            self.reconstruction_loss(gaia_from_lamost, gaia_spectra[:,:,0]) +
            self.reconstruction_loss(lamost_from_gaia, lamost_spectra[:,:,0])
        )

        # Combine losses with weights
        total_loss = (
            clip_loss +
            self.hparams.reconstruction_weight * recon_loss +
            self.hparams.cross_modal_weight * cross_modal_loss
        )

        # Log all losses
        self.log("train_clip_loss", clip_loss)
        self.log("train_recon_loss", recon_loss)
        self.log("train_cross_modal_loss", cross_modal_loss)
        self.log("train_total_loss", total_loss)

        # Store outputs for epoch-end logging
        self.training_step_outputs.append({
            'total_loss': total_loss,
            'train_clip_loss': clip_loss,
            'train_recon_loss': recon_loss,
            'train_cross_modal_loss': cross_modal_loss
        })

        return total_loss

    def training_epoch_end(self, outputs):
        if self.training_step_outputs:
            # Calculate epoch averages for each loss component
            avg_total_loss = torch.stack([x['total_loss'] for x in self.training_step_outputs]).mean()
            avg_clip_loss = torch.stack([x['train_clip_loss'] for x in self.training_step_outputs]).mean()
            avg_recon_loss = torch.stack([x['train_recon_loss'] for x in self.training_step_outputs]).mean()
            avg_cross_modal_loss = torch.stack([x['train_cross_modal_loss'] for x in self.training_step_outputs]).mean()
            
            # Log epoch averages
            self.log('train_epoch_loss', avg_total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('train_epoch_clip_loss', avg_clip_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('train_epoch_recon_loss', avg_recon_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('train_epoch_cross_modal_loss', avg_cross_modal_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            
            print(f"Epoch {self.current_epoch} Training - Total: {avg_total_loss:.4f}, CLIP: {avg_clip_loss:.4f}, Recon: {avg_recon_loss:.4f}, Cross-Modal: {avg_cross_modal_loss:.4f}")
            
            # Clear the list for next epoch
            self.training_step_outputs.clear()
        else:
            print("No valid outputs received in training_epoch_end")

    def validation_step(self, batch, batch_idx):
        gaia_spectra, lamost_spectra = batch["gaia_spectra"], batch["lamost_spectra"]

        # The normalized lamost spectra
        std, mean = lamost_spectra.std(1, keepdim=True), lamost_spectra.mean(1, keepdim=True)
        lamost_spectra_normalized = (lamost_spectra - mean) / std

        # Get embeddings from both encoders
        gaia_shared, gaia_private = self.gaia_xp_encoder(gaia_spectra)
        lamost_shared, lamost_private = self.lamost_lrs_encoder(lamost_spectra)

        # 1. CLIP Loss
        val_clip_loss = self.clip_loss(
            gaia_shared, lamost_shared, self.hparams.temperature
        )

        # 2. Reconstruction Losses
        gaia_recon = self.gaia_xp_decoder(gaia_shared, gaia_private)
        lamost_recon = self.lamost_lrs_decoder(lamost_shared, lamost_private)
        
        val_recon_loss = (
            self.reconstruction_loss(gaia_recon, gaia_spectra[:,:,0]) +
            self.reconstruction_loss(lamost_recon, lamost_spectra_normalized[:,:,0])
        )

        # 3. Cross-Modal Prediction
        gaia_from_lamost = self.gaia_xp_cross_decoder(lamost_shared)
        lamost_from_gaia = self.lamost_lrs_cross_decoder(gaia_shared)
        
        val_cross_modal_loss = (
            self.reconstruction_loss(gaia_from_lamost, gaia_spectra[:,:,0]) +
            self.reconstruction_loss(lamost_from_gaia, lamost_spectra[:,:,0])
        )

        # Combine losses
        val_total_loss = (
            val_clip_loss +
            self.hparams.reconstruction_weight * val_recon_loss +
            self.hparams.cross_modal_weight * val_cross_modal_loss
        )

        # Log validation losses
        self.log("val_clip_loss", val_clip_loss)
        self.log("val_recon_loss", val_recon_loss)
        self.log("val_cross_modal_loss", val_cross_modal_loss)
        self.log("val_total_loss", val_total_loss)

        # Store outputs for epoch-end logging
        self.validation_step_outputs.append({
            'val_total_loss': val_total_loss,
            'val_clip_loss': val_clip_loss,
            'val_recon_loss': val_recon_loss,
            'val_cross_modal_loss': val_cross_modal_loss
        })

        return val_total_loss
    
    def validation_epoch_end(self, outputs):
        if self.validation_step_outputs:
            # Calculate epoch averages for each loss component
            avg_val_total_loss = torch.stack([x['val_total_loss'] for x in self.validation_step_outputs]).mean()
            avg_val_clip_loss = torch.stack([x['val_clip_loss'] for x in self.validation_step_outputs]).mean()
            avg_val_recon_loss = torch.stack([x['val_recon_loss'] for x in self.validation_step_outputs]).mean()
            avg_val_cross_modal_loss = torch.stack([x['val_cross_modal_loss'] for x in self.validation_step_outputs]).mean()
            
            # Log epoch averages
            self.log('val_epoch_loss', avg_val_total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_epoch_clip_loss', avg_val_clip_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_epoch_recon_loss', avg_val_recon_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_epoch_cross_modal_loss', avg_val_cross_modal_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            
            # Print epoch summary
            epoch_info = f"Epoch {self.current_epoch}: "
            print(f"{epoch_info}Average Validation Loss: {avg_val_total_loss.item():.4f}")
            print(f"{epoch_info}Average Val CLIP Loss: {avg_val_clip_loss.item():.4f}")
            print(f"{epoch_info}Average Val Recon Loss: {avg_val_recon_loss.item():.4f}")
            print(f"{epoch_info}Average Val Cross-Modal Loss: {avg_val_cross_modal_loss.item():.4f}")
            
            # Clear the list for next epoch
            self.validation_step_outputs.clear()
        else:
            print("No valid outputs received in validation_epoch_end")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.epochs,
            eta_min=self.hparams.eta_min
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_total_loss"
            }
        }

    def get_embeddings(
        self,
        spectra: torch.Tensor,
        spectra_type: str,
        include_private: bool = False
    ):
        """Get embeddings for inference"""
        if spectra_type == "gaia_spectra":
            shared, private = self.gaia_xp_encoder(spectra)
        elif spectra_type == "lamost_spectra":
            shared, private = self.lamost_lrs_encoder(spectra)
        else:
            raise ValueError("Invalid spectra type")
            
        if include_private:
            return shared, private

# reconstruction loss
class ReconstructionLoss(nn.Module):
    def __init__(self, loss_type: str = "l1"):
        """
        Reconstruction loss for the SpecCLIP model.

        Args:
            loss_type (str): The type of loss to use for reconstruction. Can be either 'l1' or 'l2'.
        """
        super().__init__()
        self.loss_type = loss_type

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "l1":
            return F.l1_loss(x, y)
        elif self.loss_type == "l2":
            return F.mse_loss(x, y)
        else:
            raise ValueError("Loss type must be either 'l1' or 'l2'")

class PredictLoss(nn.Module):
    def __init__(self, loss_type: str = "l1"):
        """
        Reconstruction loss for the SpecCLIP model.

        Args:
            loss_type (str): The type of loss to use for reconstruction. Can be either 'l1' or 'l2'.
        """
        super().__init__()
        self.loss_type = loss_type

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "l1":
            return F.l1_loss(x, y)
        elif self.loss_type == "l2":
            return F.mse_loss(x, y)
        else:
            raise ValueError("Loss type must be either 'l1' or 'l2'")

class LamostLRSDecoder(nn.Module):
    def __init__(
        self, 
        shared_dim: int = 512,
        private_dim: int = 256, 
        out_features: int = 1462  # LAMOST LRS dimension
    ):
        super().__init__()
        # Projection layers for shared and private features
        self.shared_proj = nn.Linear(shared_dim, out_features)
        self.private_proj = nn.Linear(private_dim, out_features)
        
        # Final reconstruction layer
        self.final_layer = nn.Sequential(
            nn.Linear(out_features * 2, out_features * 2),
            nn.GELU(),
            nn.Linear(out_features * 2, out_features),
        )

    def forward(self, shared_features: torch.Tensor, private_features: torch.Tensor):
        # Project both feature sets to spectrum dimension
        shared_proj = self.shared_proj(shared_features)
        private_proj = self.private_proj(private_features)
        
        # Concatenate and generate final reconstruction
        combined = torch.cat([shared_proj, private_proj], dim=-1)
        reconstruction = self.final_layer(combined)
        return reconstruction

class GaiaXPDecoder(nn.Module):
    def __init__(
        self, 
        shared_dim: int = 512,
        private_dim: int = 256, 
        out_features: int = 343  # Gaia XP dimension
    ):
        super().__init__()
        # Projection layers for shared and private features
        self.shared_proj = nn.Linear(shared_dim, out_features)
        self.private_proj = nn.Linear(private_dim, out_features)
        
        # Final reconstruction layer
        self.final_layer = nn.Sequential(
            nn.Linear(out_features * 2, out_features * 2),
            nn.GELU(),
            nn.Linear(out_features * 2, out_features),
        )

    def forward(self, shared_features: torch.Tensor, private_features: torch.Tensor):
        # Project both feature sets to spectrum dimension
        shared_proj = self.shared_proj(shared_features)
        private_proj = self.private_proj(private_features)
        
        # Concatenate and generate final reconstruction
        combined = torch.cat([shared_proj, private_proj], dim=-1)
        reconstruction = self.final_layer(combined)
        return reconstruction
    
class CrossModalDecoder(nn.Module):
    """Decoder for cross-modal prediction using only shared representations"""
    def __init__(
        self,
        shared_dim: int = 512,
        out_features: int = None,  # Set based on target modality
        hidden_expansion: int = 4,
    ):
        super().__init__()
        
        mid_dim = shared_dim * hidden_expansion  # 2048 
        
        if out_features > shared_dim:
            # Path for LAMOST LRS
            self.decoder = nn.Sequential(
                nn.Linear(shared_dim, mid_dim), # 512，512*4=2048
                nn.GELU(),
                nn.BatchNorm1d(mid_dim),
                nn.Dropout(0.1),
                
                nn.Linear(mid_dim, mid_dim), # 2048, 4096
                nn.GELU(),
                nn.BatchNorm1d(mid_dim), 
                nn.Dropout(0.1),
                
                nn.Linear(mid_dim, mid_dim), # 4096, 2048
                nn.GELU(),
                nn.BatchNorm1d(mid_dim), # 2048
                nn.Dropout(0.1),
                
                nn.Linear(mid_dim, out_features) # 987, 1462
            )
        else:
            # Path for Gaia XP
            self.decoder = nn.Sequential(
                nn.Linear(shared_dim, mid_dim), # 512，512*4=2048
                nn.GELU(),
                nn.BatchNorm1d(mid_dim),
                nn.Dropout(0.1),
                
                nn.Linear(mid_dim, mid_dim//2), # 2048, 
                nn.GELU(),
                nn.BatchNorm1d(mid_dim//2),
                nn.Dropout(0.1),
                
                nn.Linear(mid_dim//2, mid_dim//4), # 512, 512
                nn.GELU(),
                nn.BatchNorm1d(mid_dim//4), # 512,
                nn.Dropout(0.1),
                
                nn.Linear(mid_dim//4, out_features) # 512, 343
            )
    
    def forward(self, shared_features: torch.Tensor):
        return self.decoder(shared_features)

class CLIPLoss(nn.Module):
    def get_logits(
        self,
        gaia_xp_features: torch.FloatTensor,
        lamost_lrs_features: torch.FloatTensor,
        logit_scale: float,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
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
        # Get the logits for the lamost lrs and gaia xp features
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
    
class LamostLRSHead_split(nn.Module):
    def __init__(
        self,
        model_path: str,
        shared_embed_dim: int = 512,
        private_embed_dim: int = 256,
        n_head: int = 4,
        model_embed_dim: int = 768,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
        load_pretrained_weights=True,
    ):
        super().__init__()
        # Load the SpecFormer backbone
        checkpoint = torch.load(model_path)
        self.backbone = SpecFormer_lm(**checkpoint["hyper_parameters"])
        if load_pretrained_weights:
            self.backbone.load_state_dict(checkpoint["state_dict"])

        # Freeze backbone if specified
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Shared representation path
        self.shared_attention = CrossAttentionHead(
            embed_dim=shared_embed_dim,
            n_head=n_head,
            model_embed_dim=model_embed_dim,
            dropout=dropout,
        )
        self.shared_mlp = MLP(
            in_features=shared_embed_dim,
            hidden_features=4 * shared_embed_dim,
            dropout=dropout,
        )

        # Private representation path
        self.private_attention = CrossAttentionHead(
            embed_dim=private_embed_dim,
            n_head=n_head//2,  # Reduced heads for private path
            model_embed_dim=model_embed_dim,
            dropout=dropout,
        )
        self.private_mlp = MLP(
            in_features=private_embed_dim,
            hidden_features=4 * private_embed_dim,
            dropout=dropout,
        )

    def forward(
        self, x: torch.tensor, return_weights: bool = False
    ):
        # Get backbone features
        with torch.set_grad_enabled(not self.freeze_backbone):
            embedding = self.backbone(x)["embedding"]

        # Generate shared representation
        shared_x, shared_attn = self.shared_attention(embedding)
        shared_repr = shared_x + self.shared_mlp(shared_x)

        # Generate private representation
        private_x, private_attn = self.private_attention(embedding)
        private_repr = private_x + self.private_mlp(private_x)

        if return_weights:
            return shared_repr.squeeze(), private_repr.squeeze(), shared_attn[1]
        
        return shared_repr.squeeze(), private_repr.squeeze()

class GaiaXPHead_split(nn.Module):
    def __init__(
        self,
        model_path: str,
        shared_embed_dim: int = 512,
        private_embed_dim: int = 256,
        n_head: int = 4,
        model_embed_dim: int = 768,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
        load_pretrained_weights=True,
    ):
        super().__init__()
        # Load the SpecFormer backbone for XP spectra
        checkpoint = torch.load(model_path)
        self.backbone = SpecFormer_xp(**checkpoint["hyper_parameters"])
        if load_pretrained_weights:
            self.backbone.load_state_dict(checkpoint["state_dict"])

        # Freeze backbone if specified
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Initial projection
        self.shared_projection = nn.Linear(model_embed_dim, shared_embed_dim)
        # Feature transformation 
        shared_intermediate_dim = 1160
        self.shared_feature_mlp = nn.Sequential(
            nn.Linear(shared_embed_dim, shared_intermediate_dim),
            nn.LayerNorm(shared_intermediate_dim),
            nn.GELU(),
            nn.Linear(shared_intermediate_dim, shared_embed_dim),
            nn.LayerNorm(shared_embed_dim),
            nn.Dropout(dropout)
        )

        self.private_projection = nn.Linear(model_embed_dim, private_embed_dim)
        private_intermediate_dim = 1160 
        self.private_feature_mlp = nn.Sequential(
            nn.Linear(private_embed_dim, private_intermediate_dim),
            nn.LayerNorm(private_intermediate_dim),
            nn.GELU(),
            nn.Linear(private_intermediate_dim, private_embed_dim),
            nn.LayerNorm(private_embed_dim),
            nn.Dropout(dropout)
        )

        self.shared_mlp = MLP(
            in_features=shared_embed_dim,
            hidden_features=4 * shared_embed_dim,
            dropout=dropout,
        )

        self.private_mlp = MLP(
            in_features=private_embed_dim,
            hidden_features=4 * private_embed_dim,
            dropout=dropout,
        )

    def forward(
        self, x: torch.tensor, return_weights: bool = False
    ):
        with torch.set_grad_enabled(not self.freeze_backbone):
            embedding = self.backbone(x)["latent"]

        # Generate shared representation
        shared_x = self.shared_projection(embedding)
        shared_x = self.shared_feature_mlp(shared_x)
        shared_repr = shared_x + self.shared_mlp(shared_x)

        # Generate private representation
        private_x = self.private_projection(embedding)
        private_x = self.private_feature_mlp(private_x)
        private_repr = private_x + self.private_mlp(private_x)

        if return_weights:
            return shared_repr.squeeze(), private_repr.squeeze(), shared_attn[1]
        
        return shared_repr.squeeze(), private_repr.squeeze()
