import os
import sys
from typing import Tuple

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from dinov2.eval.setup import setup_and_build_model

from ..modules import MLP, CrossAttentionHead
#from .specformer import SpecFormer
from .specformer_control import SpectralMLPAutoencoder_xp as SpecFormer_xp
from .specformer_control import SpecFormerControl20_wstd as SpecFormer_lm

from torch import Tensor

class SpecClipModel_reconstruct_embed768_mlp_recordloss(L.LightningModule):
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
        # add parameter to control the loss: if add the reconstruction loss and predict loss
        add_reconstruct_loss: bool = True,
        add_predict_loss: bool = True,
        #image_decoder: nn.Module = None,
        #spectrum_decoder: nn.Module = None,
        #gaia_xp_cross_decoder: nn.Module = None,
        #lamost_lrs_cross_decoder: nn.Module = None,
        
    ):
        """
        The SpecCLIP model that takes Gaia XP and LAMOST LRS spectra and embeds them into a common space using CLIP loss.
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

        self.gaia_xp_decoder = GaiaXPDecoder(768, 343)
        self.lamost_lrs_decoder = LamostLRSDecoder(768, 1462)
        self.gaia_xp_cross_decoder = GaiaXPCrossDecoder(768, 343)
        self.lamost_lrs_cross_decoder = LamostLRSCrossDecoder(768, 1462)

        # Logit scale is fixed to 15.5 and is not a learnable parameter
        if not learnable_logit_scale:
            self.logit_scale = np.log(logit_scale)
        else:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(logit_scale))

        # Use CLIP loss
        self.criterion = CLIPLoss()
        self.reconstruct_loss = ReconstructionLoss()
        self.predict_loss = PredictLoss()

        # Initialize lists to store losses for epoch-end logging
        self.training_step_outputs = []
        self.validation_step_outputs = []

    def forward(
        self,
        input: torch.Tensor,
        input_type: str,
    ):
        if input_type == "gaia_spectra":
            return self.gaia_xp_encoder(input)

        elif input_type == "lamost_spectra":
            return self.lamost_lrs_encoder(input)

        else:
            raise ValueError("Input type must be either 'gaia_spectra' or 'lamost_spectra'")

    def training_step(self, batch, batch_idx):
        gaia_spectra, lamost_spectra = batch["gaia_spectra"], batch["lamost_spectra"]

        # The normalized lamost spectra
        std, mean = lamost_spectra.std(1, keepdim=True), lamost_spectra.mean(1, keepdim=True)
        lamost_spectra_normalized = (lamost_spectra - mean) / std
        #print ('norm2')

        #print(f"im shape: {im.shape}, sp shape: {sp.shape}")  # Debugging line

        # Get the Gaia XP and LAMOST LRS features
        gaia_xp_features = self.gaia_xp_encoder(gaia_spectra)
        lamost_lrs_features = self.lamost_lrs_encoder(lamost_spectra)

        # Calculate the CLIP loss
        loss_withlogit = self.criterion(
            gaia_xp_features, lamost_lrs_features, self.hparams.temperature
        )
        loss_nologit = self.criterion(
            gaia_xp_features, lamost_lrs_features, self.hparams.logit_scale
        )

        # Initialize loss components
        loss_reconstruct = torch.tensor(0.0, device=self.device)
        loss_predict = torch.tensor(0.0, device=self.device)

        # Add another loss to reconstruct the Gaia XP and LAMOST LRS spectra, respectively: from their features
        if self.hparams.add_reconstruct_loss:
            gaia_xp_reconstruct = self.gaia_xp_decoder(gaia_xp_features)
            lamost_lrs_reconstruct = self.lamost_lrs_decoder(lamost_lrs_features)
            #print ('spectrum reconstruct shape', spectrum_reconstruct.shape)
            loss_reconstruct = self.reconstruct_loss(gaia_spectra[:,:,0], gaia_xp_reconstruct) + self.reconstruct_loss(lamost_spectra_normalized[:,:,0], lamost_lrs_reconstruct)

        # Additional loss to predict one spectrum type from the other's features (cross-prediction)
        if self.hparams.add_predict_loss:
            gaia_xp_cross_reconstruct = self.gaia_xp_cross_decoder(lamost_lrs_features)
            lamost_lrs_cross_reconstruct = self.lamost_lrs_cross_decoder(gaia_xp_features)
            loss_predict = self.predict_loss(gaia_spectra[:,:,0], gaia_xp_cross_reconstruct) + self.predict_loss(lamost_spectra[:,:,0], lamost_lrs_cross_reconstruct)

        # Log the losses (step-level logging)
        self.log("train_loss_withlogit", loss_withlogit, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        if self.hparams.add_reconstruct_loss:
            self.log("train_loss_reconstruct", loss_reconstruct, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        if self.hparams.add_predict_loss:
            self.log("train_loss_predict", loss_predict, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        
        # combine the losses
        if self.hparams.add_reconstruct_loss and self.hparams.add_predict_loss:
            loss = loss_withlogit + loss_reconstruct + loss_predict
        elif self.hparams.add_reconstruct_loss:
            loss = loss_withlogit + loss_reconstruct
        elif self.hparams.add_predict_loss:
            loss = loss_withlogit + loss_predict
        else:
            loss = loss_withlogit

        # Store outputs for epoch-end logging
        output = {
            'loss': loss,
            'train_loss_withlogit': loss_withlogit,
            'train_loss_reconstruct': loss_reconstruct,
            'train_loss_predict': loss_predict
        }
        self.training_step_outputs.append(output)

        # Return the loss
        return loss

    def training_epoch_end(self, outputs):
        if self.training_step_outputs:
            # Calculate epoch averages for each loss component
            avg_total_loss = torch.stack([x['loss'] for x in self.training_step_outputs]).mean()
            avg_clip_loss = torch.stack([x['train_loss_withlogit'] for x in self.training_step_outputs]).mean()
            
            # Log epoch averages
            self.log('train_epoch_loss', avg_total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('train_epoch_loss_withlogit', avg_clip_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            
            if self.hparams.add_reconstruct_loss:
                avg_reconstruct_loss = torch.stack([x['train_loss_reconstruct'] for x in self.training_step_outputs]).mean()
                self.log('train_epoch_loss_reconstruct', avg_reconstruct_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            
            if self.hparams.add_predict_loss:
                avg_predict_loss = torch.stack([x['train_loss_predict'] for x in self.training_step_outputs]).mean()
                self.log('train_epoch_loss_predict', avg_predict_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            
            print(f"Epoch {self.current_epoch} Training - Total: {avg_total_loss:.4f}, CLIP: {avg_clip_loss:.4f}")
            
            # Clear the list for next epoch
            self.training_step_outputs.clear()
        else:
            print("No valid outputs received in training_epoch_end")

    def validation_step(self, batch, batch_idx):
        gaia_spectra, lamost_spectra = batch["gaia_spectra"], batch["lamost_spectra"]

        # The normalized lamost spectra
        std, mean = lamost_spectra.std(1, keepdim=True), lamost_spectra.mean(1, keepdim=True)
        lamost_spectra_normalized = (lamost_spectra - mean) / std
        #print ('norm')

        #print(f"val im shape: {im.shape}, va; sp shape: {sp.shape}")  # Debugging line

        # Get the Gaia XP and LAMOST LRS features
        gaia_xp_features = self.gaia_xp_encoder(gaia_spectra)
        lamost_lrs_features = self.lamost_lrs_encoder(lamost_spectra)

        # Calculate the CLIP loss
        val_loss_nologit = self.criterion(
            gaia_xp_features, lamost_lrs_features, self.hparams.logit_scale
        )
        val_loss_withlogit = self.criterion(
            gaia_xp_features, lamost_lrs_features, self.hparams.temperature
        )

        # Initialize loss components
        loss_reconstruct = torch.tensor(0.0, device=self.device)
        loss_predict = torch.tensor(0.0, device=self.device)

        # Do the similar thing as the training_step to incorporate the other two losses
        # Add another loss to reconstruct the Gaia XP and LAMOST LRS spectra, respectively: from their features
        if self.hparams.add_reconstruct_loss:
            gaia_xp_reconstruct = self.gaia_xp_decoder(gaia_xp_features)
            #print ('image features shape', gaia_xp_features.shape)
            #print ('image reconstruct shape', image_reconstruct.shape)
            #print ('im shape', im.shape)
            #print ('sp shape', sp.shape)
            #print ('spectrum features shape', lamost_lrs_features.shape)
            lamost_lrs_reconstruct = self.lamost_lrs_decoder(lamost_lrs_features)
            #print ('spectrum reconstruct shape', spectrum_reconstruct.shape)
            loss_reconstruct = self.reconstruct_loss(gaia_spectra[:,:,0], gaia_xp_reconstruct) + self.reconstruct_loss(lamost_spectra_normalized[:,:,0], lamost_lrs_reconstruct)

        # Additional loss to predict one spectrum type from the other's features (cross-prediction)
        if self.hparams.add_predict_loss:
            gaia_xp_cross_reconstruct = self.gaia_xp_cross_decoder(lamost_lrs_features)
            lamost_lrs_cross_reconstruct = self.lamost_lrs_cross_decoder(gaia_xp_features)
            loss_predict = self.predict_loss(gaia_spectra[:,:,0], gaia_xp_cross_reconstruct) + self.predict_loss(lamost_spectra[:,:,0], lamost_lrs_cross_reconstruct)

        # Log the losses
        self.log("val_loss_withlogit", val_loss_withlogit)
        #self.log("train_loss_nologit", loss_nologit)
        #self.log("scale", self.logit_scale)
        if self.hparams.add_reconstruct_loss:
            self.log("val_loss_reconstruct", loss_reconstruct)
        if self.hparams.add_predict_loss:
            self.log("val_loss_predict", loss_predict)
        
        # combine the losses
        if self.hparams.add_reconstruct_loss and self.hparams.add_predict_loss:
            val_loss = val_loss_withlogit + loss_reconstruct + loss_predict
        elif self.hparams.add_reconstruct_loss:
            val_loss = val_loss_withlogit + loss_reconstruct
        elif self.hparams.add_predict_loss:
            val_loss = val_loss_withlogit + loss_predict
        else:
            val_loss = val_loss_withlogit

        # Log the losses
        #self.log("val_loss_nologit", val_loss_nologit)
        self.log("val_loss", val_loss)

        # Store outputs for epoch-end logging
        self.validation_step_outputs.append({
            'val_loss': val_loss,
            'val_loss_withlogit': val_loss_withlogit,
            'val_loss_reconstruct': loss_reconstruct,
            'val_loss_predict': loss_predict
        })

        # Return dictionary with all losses for epoch-end logging
        return {
            'val_loss': val_loss,
            'val_loss_withlogit': val_loss_withlogit,
            'val_loss_reconstruct': loss_reconstruct,
            'val_loss_predict': loss_predict
        }

    def validation_epoch_end(self, outputs):
        if self.validation_step_outputs:
            # Calculate epoch averages for each loss component
            avg_val_loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean()
            avg_val_clip_loss = torch.stack([x['val_loss_withlogit'] for x in self.validation_step_outputs]).mean()
            
            # Log epoch averages
            self.log('val_epoch_loss', avg_val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_epoch_loss_withlogit', avg_val_clip_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            
            if self.hparams.add_reconstruct_loss:
                avg_val_reconstruct_loss = torch.stack([x['val_loss_reconstruct'] for x in self.validation_step_outputs]).mean()
                self.log('val_epoch_loss_reconstruct', avg_val_reconstruct_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            
            if self.hparams.add_predict_loss:
                avg_val_predict_loss = torch.stack([x['val_loss_predict'] for x in self.validation_step_outputs]).mean()
                self.log('val_epoch_loss_predict', avg_val_predict_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            
            # Print epoch summary only once
            epoch_info = f"Epoch {self.current_epoch}: "
            print(f"{epoch_info}Average Validation Loss: {avg_val_loss.item():.4f}")
            print(f"{epoch_info}Average Val CLIP Loss: {avg_val_clip_loss.item():.4f}")
            
            if self.hparams.add_reconstruct_loss:
                print(f"{epoch_info}Average Val Reconstruct Loss: {avg_val_reconstruct_loss.item():.4f}")
            
            if self.hparams.add_predict_loss:
                print(f"{epoch_info}Average Val Predict Loss: {avg_val_predict_loss.item():.4f}")
            
            # Clear the list for next epoch
            self.validation_step_outputs.clear()
        else:
            print("No valid outputs received in validation_epoch_end")

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

# The decoders for Gaia XP and LAMOST LRS spectra, and cross-decoders for cross-prediction
class GaiaXPDecoder(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(in_features, 4 * in_features),
            nn.GELU(),
            nn.BatchNorm1d(4 * in_features),
            nn.Dropout(0.1),

            nn.Linear(4 * in_features, 2 * in_features), # 2048, 4096
            nn.GELU(),
            nn.BatchNorm1d(2 * in_features), 
            nn.Dropout(0.1),

            nn.Linear(2 * in_features, in_features), # 2048, 4096
            nn.GELU(),
            nn.BatchNorm1d(in_features), 
            nn.Dropout(0.1),

            nn.Linear(in_features, out_features),
        )

    def forward(self, x: torch.Tensor):
        return self.decoder(x)

class LamostLRSDecoder(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(in_features, 4 * in_features),
            nn.GELU(),
            nn.BatchNorm1d(4 * in_features),
            nn.Dropout(0.1),

            nn.Linear(4 * in_features, 4 * in_features),
            nn.GELU(),
            nn.BatchNorm1d(4 * in_features),
            nn.Dropout(0.1),

            nn.Linear(4 * in_features, 4 * in_features),
            nn.GELU(),
            nn.BatchNorm1d(4 * in_features),
            nn.Dropout(0.1),

            nn.Linear(4 * in_features, out_features),
        )

    def forward(self, x: torch.Tensor):
        return self.decoder(x)

class GaiaXPCrossDecoder(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(in_features, 4 * in_features),
            nn.GELU(),
            nn.BatchNorm1d(4 * in_features),
            nn.Dropout(0.1),

            nn.Linear(4 * in_features, 2 * in_features), # 2048, 4096
            nn.GELU(),
            nn.BatchNorm1d(2 * in_features), 
            nn.Dropout(0.1),

            nn.Linear(2 * in_features, in_features), # 2048, 4096
            nn.GELU(),
            nn.BatchNorm1d(in_features), 
            nn.Dropout(0.1),

            nn.Linear(in_features, out_features),
        )

    def forward(self, x: torch.Tensor):
        return self.decoder(x)

class LamostLRSCrossDecoder(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(in_features, 4 * in_features),
            nn.GELU(),
            nn.BatchNorm1d(4 * in_features),
            nn.Dropout(0.1),

            nn.Linear(4 * in_features, 4 * in_features),
            nn.GELU(),
            nn.BatchNorm1d(4 * in_features),
            nn.Dropout(0.1),

            nn.Linear(4 * in_features, 4 * in_features),
            nn.GELU(),
            nn.BatchNorm1d(4 * in_features),
            nn.Dropout(0.1),

            nn.Linear(4 * in_features, out_features),
        )

    def forward(self, x: torch.Tensor):
        return self.decoder(x)


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
    def __init__(
        self,
        model_path: str,
        embed_dim: int = 768,
        n_head: int = 4,
        model_embed_dim: int = 768,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
        load_pretrained_weights=True,
    ):
        """
        Cross-attention spectrum module that takes a spectrum and passes it through a pretrained SpecFormer model and
        then through a cross-attention mechanism and MLP to get the final embedding.

        Args:
            save_path (str): Path to the checkpoint of the SpecFormer model.
            embed_dim (int): Dimension of the SpecCLIP embedding.
            n_head (int): Number of heads in the multihead attention.
            model_embed_dim (int): Dimension of the SpecFormer embedding.
            dropout (float): Dropout rate for MLP layers.
            freeze_backbone (bool): Whether to freeze the backbone of the SpecFormer model.
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
    def __init__(
        self,
        model_path: str,
        embed_dim: int = 768,
        n_head: int = 4,
        model_embed_dim: int = 768,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
        load_pretrained_weights=True,
    ):
        """
        Cross-attention spectrum module that takes a spectrum and passes it through a pretrained SpecFormer model and
        then through a cross-attention mechanism and MLP to get the final embedding.

        Args:
            save_path (str): Path to the checkpoint of the SpecFormer model.
            embed_dim (int): Dimension of the SpecCLIP embedding.
            n_head (int): Number of heads in the multihead attention.
            model_embed_dim (int): Dimension of the SpecFormer embedding.
            dropout (float): Dropout rate for MLP layers.
            freeze_backbone (bool): Whether to freeze the backbone of the SpecFormer model.
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
    
class LamostLRSHeadWithMLP(nn.Module):
    def __init__(
        self,
        model_path: str,
        embed_dim: int = 768,
        n_head: int = 4,  # Kept for parameter compatibility
        model_embed_dim: int = 768,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
        load_pretrained_weights=True,
    ):
        """
        MLP-based module that replaces cross-attention with equivalent parameter count MLPs.
        Total parameters closely match the original implementation (7.08M).

        Args:
            model_path (str): Path to the checkpoint of the SpecFormer model.
            embed_dim (int): Dimension of the embedding.
            n_head (int): Kept for parameter compatibility (not used).
            model_embed_dim (int): Dimension of the model embedding.
            dropout (float): Dropout rate for MLP layers.
            freeze_backbone (bool): Whether to freeze the backbone of the model.
            load_pretrained_weights (bool): Whether to load pretrained weights.
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

        # Initial projection
        self.projection = nn.Linear(model_embed_dim, embed_dim)
        
        # Feature transformation with precisely calculated parameter count
        intermediate_dim = 1160  # Carefully selected to match parameter count
        self.feature_mlp = nn.Sequential(
            nn.Linear(embed_dim, intermediate_dim),
            nn.LayerNorm(intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout)
        )
        
        # Main MLP - same as original
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=4 * embed_dim,
            dropout=dropout,
        )

    def forward(
        self, x: torch.tensor, y: torch.tensor = None, return_weights: bool = False
    ):
        # Use 'latent' instead of 'embedding'
        with torch.set_grad_enabled(not self.freeze_backbone):
            #embedding = self.backbone(x)["latent"]
            embedding = torch.mean(self.backbone(x)["embedding"], -2)

        # Project to target dimension
        x = self.projection(embedding)
        
        # Apply feature transformation
        x = self.feature_mlp(x)
        
        # Apply main MLP with residual connection
        x = x + self.mlp(x)

        if return_weights:
            # For compatibility, return dummy attention weights
            dummy_attentions = torch.zeros(
                (embedding.size(0), embedding.size(1)), 
                device=embedding.device
            )
            return x.squeeze(), dummy_attentions

        return x.squeeze()
    
class GaiaXPHeadWithMLP(nn.Module):
    def __init__(
        self,
        model_path: str,
        embed_dim: int = 768,
        n_head: int = 4,  # Kept for parameter compatibility
        model_embed_dim: int = 768,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
        load_pretrained_weights=True,
    ):
        """
        MLP-based module that replaces cross-attention with equivalent parameter count MLPs.
        Total parameters closely match the original implementation (7.08M).

        Args:
            model_path (str): Path to the checkpoint of the SpecFormer model.
            embed_dim (int): Dimension of the embedding.
            n_head (int): Kept for parameter compatibility (not used).
            model_embed_dim (int): Dimension of the model embedding.
            dropout (float): Dropout rate for MLP layers.
            freeze_backbone (bool): Whether to freeze the backbone of the model.
            load_pretrained_weights (bool): Whether to load pretrained weights.
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

        # Initial projection
        self.projection = nn.Linear(model_embed_dim, embed_dim)
        
        # Feature transformation with precisely calculated parameter count
        intermediate_dim = 1160  # Carefully selected to match parameter count
        self.feature_mlp = nn.Sequential(
            nn.Linear(embed_dim, intermediate_dim),
            nn.LayerNorm(intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout)
        )
        
        # Main MLP - same as original
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=4 * embed_dim,
            dropout=dropout,
        )

    def forward(
        self, x: torch.tensor, y: torch.tensor = None, return_weights: bool = False
    ):
        # Use 'latent' instead of 'embedding'
        with torch.set_grad_enabled(not self.freeze_backbone):
            embedding = self.backbone(x)["latent"]

        # Project to target dimension
        x = self.projection(embedding)
        
        # Apply feature transformation
        x = self.feature_mlp(x)
        
        # Apply main MLP with residual connection
        x = x + self.mlp(x)

        if return_weights:
            # For compatibility, return dummy attention weights
            dummy_attentions = torch.zeros(
                (embedding.size(0), embedding.size(1)), 
                device=embedding.device
            )
            return x.squeeze(), dummy_attentions

        return x.squeeze()