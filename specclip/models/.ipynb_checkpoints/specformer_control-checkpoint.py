import math

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..modules import LayerNorm, TransformerBlock, _init_by_depth

from scipy.ndimage import gaussian_filter1d, label, find_objects
from typing import Dict, Tuple, Optional, List

class MLPBlock(nn.Module):
    """MLP block with residual connection"""
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        use_residual: bool = True
    ):
        super().__init__()
        self.use_residual = use_residual

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        if self.use_residual:
            return x + self.mlp(self.norm(x))
        else:
            return self.mlp(self.norm(x))

class SpectralMLPAutoencoder_xp(L.LightningModule):
    def __init__(
        self,
        input_dim: int = 1,              # Input dimension (per wavelength point)
        seq_len: int = 343,             # Input sequence length
        hidden_dim: int = 2560,          # Hidden dimension size
        bottleneck_dim: int = 768,       # Bottleneck dimension
        num_encoder_blocks: int = 6,     # Number of encoder MLP blocks
        num_decoder_blocks: int = 6,     # Number of decoder MLP blocks
        expansion_factor: int = 4,       # Expansion factor for MLP hidden layers
        use_masking: bool = True,        # Whether to use masking during training
        mask_ratio: float = 0.15,        # Ratio of tokens to mask (if masking is used)
        mask_chunk_width: int = 20,      # Width of masked chunks
        dropout: float = 0.1,            # Dropout rate
        mask_num_chunks: int = 6,        # Number of chunks to mask
    ):
        super().__init__()
        self.save_hyperparameters()

        # Add 1 to seq_len to account for the std token at the beginning
        self.total_seq_len = seq_len + 2

        # Calculate actual input dimension (input features * total sequence length)
        actual_input_dim = input_dim * self.total_seq_len

        # Initial projection
        self.input_proj = nn.Linear(actual_input_dim, hidden_dim)

        # Encoder blocks
        self.encoder_blocks = nn.ModuleList()
        for _ in range(num_encoder_blocks):
            self.encoder_blocks.append(
                MLPBlock(
                    dim=hidden_dim,
                    hidden_dim=hidden_dim * expansion_factor,
                    dropout=dropout,
                    use_residual=False
                )
            )

        # Bottleneck
        self.bottleneck_down = nn.Linear(hidden_dim, bottleneck_dim)
        self.bottleneck_norm = nn.LayerNorm(bottleneck_dim)
        self.bottleneck_up = nn.Linear(bottleneck_dim, hidden_dim)

        # Decoder blocks
        self.decoder_blocks = nn.ModuleList()
        for _ in range(num_decoder_blocks):
            self.decoder_blocks.append(
                MLPBlock(
                    dim=hidden_dim,
                    hidden_dim=hidden_dim * expansion_factor,
                    dropout=dropout,
                    use_residual=False
                )
            )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, actual_input_dim)

        # Initialize parameters
        self._reset_parameters()

        # Calculate and print model size
        self._calculate_params()

    def _calculate_params(self):
        """Calculate and print the number of trainable parameters"""
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params:,}")

    def _reset_parameters(self):
        """Initialize model parameters"""
        # Initialize linear layers with kaiming initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def preprocess(self, x: Tensor) -> Tensor:
        """
        Normalize input spectra and prepend std token
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, 1]
            
        Returns:
            Processed tensor with std token of shape [batch_size, seq_len+1]
        """
        # Handle shape: input is [batch_size, seq_len, 1]
        if x.dim() == 3 and x.shape[2] == 1:
            x = x.squeeze(-1)  # [batch_size, seq_len]

        batch_size, seq_len = x.shape

        # Compute statistics
        std = x.std(1, keepdim=True)  # [batch_size, 1]
        mean = x.mean(1, keepdim=True)  # [batch_size, 1]

        # Normalize the spectrum
        x_normalized = (x - mean) / std  # [batch_size, seq_len]

        # Create std token
        # Following your original approach, use log10(std) as the std token value
        std_token = std  # [batch_size, 1]
        mean_token = mean  # [batch_size, 1]

        # Concatenate std token with normalized spectrum
        # This is similar to your original approach of prepending the std
        x_with_std = torch.cat([mean_token, std_token, x_normalized], dim=1)  # [batch_size, seq_len+1]

        return x_with_std

    def mask_sequence(self, x: Tensor) -> Tensor:
        """
        Apply random masking to the sequence if masking is enabled
        
        Args:
            x: Input tensor with std token of shape [batch_size, seq_len+1]
            
        Returns:
            Masked tensor of same shape (std token is never masked)
        """
        if not self.hparams.use_masking:
            return x

        # Apply masking, but preserve the std token (first position)
        batch_size, seq_len = x.shape
        masked_x = x.clone()

        # Skip the std token (first position) for masking
        spectrum_only = masked_x[:, 2:]

        # Use either num_chunks approach or ratio-based approach
        if self.hparams.mask_num_chunks > 0:
            # Similar to original code - fixed number of chunks
            for i in range(batch_size):
                for _ in range(self.hparams.mask_num_chunks):
                    # Find a valid start position (exclude the first token)
                    spectrum_len = seq_len - 1
                    chunk_width = min(self.hparams.mask_chunk_width, spectrum_len)

                    # Ensure we don't start a chunk that would go past the end
                    if spectrum_len <= chunk_width:
                        continue

                    start_pos = torch.randint(0, spectrum_len - chunk_width, (1,)).item()
                    # +1 in the position to account for the std token
                    masked_x[i, start_pos+1:start_pos+1+chunk_width] = 0
        else:
            # Ratio-based approach (alternative implementation)
            chunk_width = self.hparams.mask_chunk_width
            spectrum_len = seq_len - 1  # Excluding std token
            num_chunks = max(1, int(spectrum_len * self.hparams.mask_ratio / chunk_width))

            for i in range(batch_size):
                for _ in range(num_chunks):
                    if spectrum_len <= chunk_width:
                        continue
                    start_pos = torch.randint(0, spectrum_len - chunk_width, (1,)).item()
                    # +1 in the position to account for the std token
                    masked_x[i, start_pos+1:start_pos+1+chunk_width] = 0

        return masked_x

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """Forward pass through the model"""
        x = self.preprocess(x)
        return self.forward_without_preprocessing(x)

    def forward_without_preprocessing(self, x: Tensor) -> Dict[str, Tensor]:
        """Forward pass through the autoencoder without preprocessing"""
        batch_size = x.shape[0]

        # Flatten the input
        x_flat = x.reshape(batch_size, -1)

        # Encode
        h = self.input_proj(x_flat)

        # Encoder blocks
        for block in self.encoder_blocks:
            h = block(h)

        # Bottleneck
        latent = self.bottleneck_norm(self.bottleneck_down(h))
        h = self.bottleneck_up(latent)

        # Decoder blocks
        for block in self.decoder_blocks:
            h = block(h)

        # Output projection
        output_flat = self.output_proj(h)

        # Reshape back to original dimensions
        reconstructions = output_flat.reshape(batch_size, self.total_seq_len)

        return {
            "reconstructions": reconstructions,
            "latent": latent
        }

    def training_step(self, batch, batch_idx):
        # Get spectra from batch
        spectra = batch["spectra"]

        #print (spectra.shape)
        #print (spectra[0, :10, :])

        # Preprocess - adds std token
        input_data = self.preprocess(spectra)
        target = input_data.clone()

        # Apply masking if enabled
        if self.hparams.use_masking:
            masked_input = self.mask_sequence(input_data)

            # Forward pass
            output = self.forward_without_preprocessing(masked_input)["reconstructions"]

            # Calculate loss only on masked positions
            mask = (masked_input != target).to(output.dtype)
            loss = F.mse_loss(output * mask, target * mask) / (mask.sum() + 1e-8)
        else:
            # No masking, standard autoencoder
            output = self.forward_without_preprocessing(input_data)["reconstructions"]
            loss = F.mse_loss(output, target)

        # Log metrics
        self.log("training_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        # Get spectra from batch
        spectra = batch["spectra"]

        # Preprocess - adds std token
        input_data = self.preprocess(spectra)
        target = input_data.clone()

        # Apply masking if enabled
        if self.hparams.use_masking:
            masked_input = self.mask_sequence(input_data)

            # Forward pass
            output = self.forward_without_preprocessing(masked_input)["reconstructions"]

            # Calculate loss only on masked positions
            mask = (masked_input != target).to(output.dtype)
            loss = F.mse_loss(output * mask, target * mask) / (mask.sum() + 1e-8)
        else:
            # No masking, standard autoencoder
            output = self.forward_without_preprocessing(input_data)["reconstructions"]
            loss = F.mse_loss(output, target)

        # Log metrics
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_epoch_loss", loss, on_step=False, on_epoch=True,
                prog_bar=True, sync_dist=True)

        return loss

    def training_epoch_end(self, outputs):
        if not outputs:
            print("No valid outputs received in training_epoch_end")
            return

        try:
            # Handle different output types
            if isinstance(outputs[0], dict):
                losses = [x['loss'] for x in outputs if 'loss' in x]
            elif isinstance(outputs[0], torch.Tensor):
                losses = outputs
            else:
                print("Unsupported data type in outputs.")
                return

            # Calculate average loss
            avg_train_loss = torch.stack(losses).mean()

            # Log with consistent naming
            self.log('train_epoch_loss', avg_train_loss,
                    on_step=False, on_epoch=True,
                    prog_bar=True, logger=True, sync_dist=True)

        except Exception as e:
            print(f"Error in training_epoch_end: {str(e)}")

    def validation_epoch_end(self, outputs):
        if not outputs:
            print("No valid outputs received in validation_epoch_end")
            return

        try:
            # Handle different output types
            if isinstance(outputs[0], dict):
                losses = [x['val_loss'] for x in outputs if 'val_loss' in x]
            elif isinstance(outputs[0], torch.Tensor):
                losses = outputs
            else:
                print("Unsupported data type in validation outputs.")
                return

            # Calculate average loss
            avg_val_loss = torch.stack(losses).mean()

            # Log with consistent naming
            self.log('val_epoch_loss', avg_val_loss,
                    on_step=False, on_epoch=True,
                    prog_bar=True, logger=True, sync_dist=True)

        except Exception as e:
            print(f"Error in validation_epoch_end: {str(e)}")

class SpecFormerControl20_wstd(L.LightningModule):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        max_len: int = 400,
        mask_num_chunks: int = 6,
        mask_chunk_width: int = 20, #50
        slice_section_length: int = 20,
        slice_overlap: int = 10,
        dropout: float = 0.1,
        norm_first: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.data_embed = nn.Linear(input_dim, embed_dim)
        self.position_embed = nn.Embedding(max_len, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=embed_dim,
                    num_heads=num_heads,
                    causal=False,
                    dropout=dropout,
                    bias=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layernorm = LayerNorm(embed_dim, bias=True)
        self.head = nn.Linear(embed_dim, input_dim, bias=True)

        self._reset_parameters_datapt()

    def forward(self, x: Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        x = self.preprocess(x)
        return self.forward_without_preprocessing(x)

    def forward_without_preprocessing(self, x: Tensor):
        """Forward pass through the model.
        The training step performs masking before preprocessing,
        thus samples should not be preprocessed again as in forward()"""

        t = x.shape[1]
        if t > self.hparams.max_len:
            raise ValueError(
                f"Cannot forward sequence of length {t}, "
                f"block size is only {self.hparams.max_len}"
            )
        pos = torch.arange(0, t, dtype=torch.long, device=x.device)  # shape (t)

        data_emb = self.data_embed(x)  # to shape (b, t, embedding_dim)
        pos_emb = self.position_embed(pos)  # to shape (t, embedding_dim)

        x = self.dropout(data_emb + pos_emb)
        for block in self.blocks:
            x = block(x)
        x = self.final_layernorm(x)

        reconstructions = self.head(x)

        return {"reconstructions": reconstructions, "embedding": x}

    def training_step(self, batch):
        # slice the input and copy
        input = self.preprocess(batch["spectra"])
        target = torch.clone(input)

        # mask parts of the input
        input = self.mask_sequence(input)
        # forward pass
        output = self.forward_without_preprocessing(input)["reconstructions"]

        # find the mask locations
        locs = (input != target).type_as(output)
        loss = F.mse_loss(output * locs, target * locs, reduction="mean") / locs.mean()
        
        self.log("training_loss", loss, prog_bar=True)
        
        return loss

    #def validation_step(self, batch):
    def validation_step(self, batch, batch_idx):
        #print (batch["spectrum"].shape) #64, 3869, 1
        input = self.preprocess(batch["spectra"])
        target = torch.clone(input)

        # mask parts of the input
        input = self.mask_sequence(input)

        # forward pass
        output = self.forward_without_preprocessing(input)["reconstructions"]

        # find the mask locations
        locs = (input != target).type_as(output)
        loss = F.mse_loss(output * locs, target * locs, reduction="mean") / locs.mean()
        self.log("val_training_loss", loss, prog_bar=True)
        #print ('val losssss',loss)
        return loss
        ##return {"val_training_loss": loss} 

    def training_epoch_end(self, outputs):
        if not outputs:
            print("No valid outputs received in training_epoch_end")
            return

        try:
            # Handle different output types
            if isinstance(outputs[0], dict):
                losses = [x['loss'] for x in outputs if 'loss' in x]
            elif isinstance(outputs[0], torch.Tensor):
                losses = outputs
            else:
                print("Unsupported data type in outputs.")
                return

            # Calculate average loss
            avg_train_loss = torch.stack(losses).mean()
            
            # Log with consistent naming
            self.log('train_epoch_loss', avg_train_loss, 
                    on_step=False, on_epoch=True, 
                    prog_bar=True, logger=True, sync_dist=True)

        except Exception as e:
            print(f"Error in training_epoch_end: {str(e)}")

    def validation_epoch_end(self, outputs):
        if not outputs:
            print("No valid outputs received in validation_epoch_end")
            return

        try:
            # Handle different output types
            if isinstance(outputs[0], dict):
                losses = [x['val_loss'] for x in outputs if 'val_loss' in x]
            elif isinstance(outputs[0], torch.Tensor):
                losses = outputs
            else:
                print("Unsupported data type in validation outputs.")
                return

            # Calculate average loss
            avg_val_loss = torch.stack(losses).mean()
            
            # Log with consistent naming
            self.log('val_epoch_loss', avg_val_loss, 
                    on_step=False, on_epoch=True, 
                    prog_bar=True, logger=True, sync_dist=True)

        except Exception as e:
            print(f"Error in validation_epoch_end: {str(e)}")

    def mask_sequence(self, x: Tensor):
        """Mask batched sequence"""
        return torch.stack([self._mask_seq(el) for el in x])

    def preprocess(self, x):
        std, mean = x.std(1, keepdim=True), x.mean(1, keepdim=True)
        x = (x - mean) / std
        x = self._slice(x)
        x = F.pad(x, pad=(1, 0, 1, 0), mode="constant", value=0)
        x[:, 0, 0] = torch.log10(std.squeeze()) 
        #x[:, 0, 1] = (std.squeeze() - 2) / 8
        return x

    def _reset_parameters_datapt(self):
        # not scaling the initial embeddngs.
        for emb in [self.data_embed, self.position_embed]:
            std = 1 / math.sqrt(self.hparams.embed_dim)
            nn.init.trunc_normal_(emb.weight, std=std, a=-3 * std, b=3 * std)

        # transformer block weights
        self.blocks.apply(lambda m: _init_by_depth(m, self.hparams.num_layers))
        self.head.apply(lambda m: _init_by_depth(m, 1 / 2))

    def _slice(self, x):
        start_indices = np.arange(
            0,
            x.shape[1] - self.hparams.slice_overlap,
            self.hparams.slice_section_length - self.hparams.slice_overlap,
        )
        sections = [
            x[:, start : start + self.hparams.slice_section_length].transpose(1, 2)
            for start in start_indices
        ]

        # If the last section is not of length 'section_length', you can decide whether to keep or discard it
        if sections[-1].shape[1] < self.hparams.slice_section_length:
            sections.pop(-1)  # Discard the last section

        return torch.cat(sections, 1)

    def _mask_seq(self, seq: torch.Tensor) -> torch.Tensor:
        """Randomly masks contiguous sections of an unbatched sequence,
        ensuring separation between chunks is at least chunk_width."""
        len_ = seq.shape[0]
        num_chunks = self.hparams.mask_num_chunks
        chunk_width = self.hparams.mask_chunk_width
        #print ('*************************************',len_)
        # Ensure there's enough space for the chunks and separations
        total_width_needed = num_chunks * chunk_width + (num_chunks - 1) * chunk_width
        if total_width_needed > len_:
            raise ValueError("Sequence is too short to mask")

        masked_seq = seq.clone()

        for i in range(num_chunks):
            start = (i * len_) // num_chunks
            loc = torch.randint(0, len_ // num_chunks - chunk_width, (1,)).item()
            masked_seq[loc + start : loc + start + chunk_width] = 0

        return masked_seq

class SpecFormerControl1_stats(L.LightningModule):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        max_len: int,
        mask_num_chunks: int = 6,
        mask_chunk_width: int = 100,
        slice_section_length: int = 20,
        slice_overlap: int = 10,
        dropout: float = 0.1,
        norm_first: bool = False,
        conv_dim: int = 18,
        include_stats: bool = False,  # New parameter to control stats inclusion
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Store the include_stats parameter
        self.include_stats = include_stats
        
        # Define effective input dimension based on stats inclusion
        effective_input_dim = input_dim + 2 if include_stats else input_dim

        # define a one-dimension convolution
        self.conv1d = nn.Conv1d(input_dim, conv_dim, kernel_size=5, padding=2)
        
        # Adjust data embedding dimension based on stats inclusion
        self.data_embed = nn.Linear(effective_input_dim, embed_dim)
        self.position_embed = nn.Embedding(max_len, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=embed_dim,
                    num_heads=num_heads,
                    causal=False,
                    dropout=dropout,
                    bias=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layernorm = LayerNorm(embed_dim, bias=True)
        self.conv_head = nn.Linear(embed_dim, conv_dim+2, bias=True)

        # Adjust head output dimension based on stats inclusion
        self.head = nn.Linear(embed_dim, effective_input_dim, bias=True)

        self._reset_parameters_datapt()

    def forward(self, x: Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        x = self.preprocess(x)
        return self.forward_without_preprocessing(x)

    def forward_without_preprocessing(self, x: Tensor):
        """Forward pass through the model.
        The training step performs masking before preprocessing,
        thus samples should not be preprocessed again as in forward()"""

        t = x.shape[1]
        if t > self.hparams.max_len:
            raise ValueError(
                f"Cannot forward sequence of length {t}, "
                f"block size is only {self.hparams.max_len}"
            )
        pos = torch.arange(0, t, dtype=torch.long, device=x.device)  # shape (t)

        data_emb = self.data_embed(x)  # to shape (b, t, embedding_dim)
        pos_emb = self.position_embed(pos)  # to shape (t, embedding_dim)

        x = self.dropout(data_emb + pos_emb)
        for block in self.blocks:
            x = block(x)
        x = self.final_layernorm(x)

        reconstructions = self.head(x)

        return {"reconstructions": reconstructions, "embedding": x}

    def preprocess(self, x):
        std, mean = x.std(1, keepdim=True), x.mean(1, keepdim=True)
        x_normalized = (x - mean) / std
        
        if self.include_stats:
            # Original preprocessing with stats
            x = F.pad(x_normalized, pad=(2, 0, 1, 0), mode="constant", value=0)
            x[:, 0, 0] = mean.squeeze() #torch.log10(mean.squeeze())
            x[:, 0, 1] = std.squeeze() #torch.log10(std.squeeze())

        else:
            # Preprocessing without stats - just return normalized data
            x = x_normalized #.unsqueeze(-1)  # Keep the last dimension as 1
            
        return x

    # Rest of the methods remain unchanged
    def training_step(self, batch):
        input = self.preprocess(batch["spectra"])
        target = torch.clone(input)
        input = self.mask_sequence(input)
        output = self.forward_without_preprocessing(input)["reconstructions"]
        locs = (input != target).type_as(output)
        loss = F.mse_loss(output * locs, target * locs, reduction="mean") / locs.mean()
        self.log("training_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input = self.preprocess(batch["spectra"])
        target = torch.clone(input)
        input = self.mask_sequence(input)
        output = self.forward_without_preprocessing(input)["reconstructions"]
        locs = (input != target).type_as(output)
        loss = F.mse_loss(output * locs, target * locs, reduction="mean") / locs.mean()
        self.log("val_training_loss", loss, prog_bar=True)
        return loss

    def training_epoch_end(self, outputs):
        if not outputs:
            print("No valid outputs received in training_epoch_end")
            return

        try:
            # Handle different output types
            if isinstance(outputs[0], dict):
                losses = [x['loss'] for x in outputs if 'loss' in x]
            elif isinstance(outputs[0], torch.Tensor):
                losses = outputs
            else:
                print("Unsupported data type in outputs.")
                return

            # Calculate average loss
            avg_train_loss = torch.stack(losses).mean()
            
            # Log with consistent naming
            self.log('train_epoch_loss', avg_train_loss, 
                    on_step=False, on_epoch=True, 
                    prog_bar=True, logger=True, sync_dist=True)

        except Exception as e:
            print(f"Error in training_epoch_end: {str(e)}")

    def validation_epoch_end(self, outputs):
        if not outputs:
            print("No valid outputs received in validation_epoch_end")
            return

        try:
            # Handle different output types
            if isinstance(outputs[0], dict):
                losses = [x['val_loss'] for x in outputs if 'val_loss' in x]
            elif isinstance(outputs[0], torch.Tensor):
                losses = outputs
            else:
                print("Unsupported data type in validation outputs.")
                return

            # Calculate average loss
            avg_val_loss = torch.stack(losses).mean()
            
            # Log with consistent naming
            self.log('val_epoch_loss', avg_val_loss, 
                    on_step=False, on_epoch=True, 
                    prog_bar=True, logger=True, sync_dist=True)

        except Exception as e:
            print(f"Error in validation_epoch_end: {str(e)}")

    def mask_sequence(self, x: Tensor):
        """Mask batched sequence"""
        return torch.stack([self._mask_seq(el) for el in x])

    def _reset_parameters_datapt(self):
        # not scaling the initial embeddngs.
        for emb in [self.data_embed, self.position_embed]:
            std = 1 / math.sqrt(self.hparams.embed_dim)
            nn.init.trunc_normal_(emb.weight, std=std, a=-3 * std, b=3 * std)

        # transformer block weights
        self.blocks.apply(lambda m: _init_by_depth(m, self.hparams.num_layers))
        self.head.apply(lambda m: _init_by_depth(m, 1 / 2))

    def _slice(self, x):
        start_indices = np.arange(
            0,
            x.shape[1] - self.hparams.slice_overlap,
            self.hparams.slice_section_length - self.hparams.slice_overlap,
        )
        #print ([x[:, start : start + self.hparams.slice_section_length].transpose(1, 2).shape for start in start_indices])
        #print (x.shape) # 64,20; 64,19
        sections = [
            x[:, start : start + self.hparams.slice_section_length].transpose(1, 2)
            for start in start_indices
        ]

        # If the last section is not of length 'section_length', you can decide whether to keep or discard it
        if sections[-1].shape[1] < self.hparams.slice_section_length:
            sections.pop(-1)  # Discard the last section

        return torch.cat(sections, 1)

    def _mask_seq(self, seq: torch.Tensor) -> torch.Tensor:
        """Randomly masks contiguous sections of an unbatched sequence,
        ensuring separation between chunks is at least chunk_width."""
        len_ = seq.shape[0]
        num_chunks = self.hparams.mask_num_chunks
        chunk_width = self.hparams.mask_chunk_width
        #print (len_)
        # Ensure there's enough space for the chunks and separations
        total_width_needed = num_chunks * chunk_width + (num_chunks - 1) * chunk_width
        if total_width_needed > len_:
            raise ValueError("Sequence is too short to mask")

        masked_seq = seq.clone()

        for i in range(num_chunks):
            start = (i * len_) // num_chunks
            loc = torch.randint(0, len_ // num_chunks - chunk_width, (1,)).item()
            masked_seq[loc + start : loc + start + chunk_width] = 0

        return masked_seq

class SpectralMLPAutoencoder(L.LightningModule):
    def __init__(
        self,
        input_dim: int = 1,              # Input dimension (per wavelength point)
        seq_len: int = 1462,             # Input sequence length
        hidden_dim: int = 2560,          # Hidden dimension size
        bottleneck_dim: int = 768,       # Bottleneck dimension
        num_encoder_blocks: int = 6,     # Number of encoder MLP blocks
        num_decoder_blocks: int = 6,     # Number of decoder MLP blocks
        expansion_factor: int = 4,       # Expansion factor for MLP hidden layers
        use_masking: bool = True,        # Whether to use masking during training
        mask_ratio: float = 0.15,        # Ratio of tokens to mask (if masking is used)
        mask_chunk_width: int = 20,      # Width of masked chunks
        dropout: float = 0.1,            # Dropout rate
        mask_num_chunks: int = 6,        # Number of chunks to mask
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Add 1 to seq_len to account for the std token at the beginning
        self.total_seq_len = seq_len + 1
        
        # Calculate actual input dimension (input features * total sequence length)
        actual_input_dim = input_dim * self.total_seq_len
        
        # Initial projection
        self.input_proj = nn.Linear(actual_input_dim, hidden_dim)
        
        # Encoder blocks
        self.encoder_blocks = nn.ModuleList()
        for _ in range(num_encoder_blocks):
            self.encoder_blocks.append(
                MLPBlock(
                    dim=hidden_dim,
                    hidden_dim=hidden_dim * expansion_factor,
                    dropout=dropout,
                    use_residual=False
                )
            )
        
        # Bottleneck
        self.bottleneck_down = nn.Linear(hidden_dim, bottleneck_dim)
        self.bottleneck_norm = nn.LayerNorm(bottleneck_dim)
        self.bottleneck_up = nn.Linear(bottleneck_dim, hidden_dim)
        
        # Decoder blocks
        self.decoder_blocks = nn.ModuleList()
        for _ in range(num_decoder_blocks):
            self.decoder_blocks.append(
                MLPBlock(
                    dim=hidden_dim,
                    hidden_dim=hidden_dim * expansion_factor,
                    dropout=dropout,
                    use_residual=False
                )
            )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, actual_input_dim)
        
        # Initialize parameters
        self._reset_parameters()
        
        # Calculate and print model size
        self._calculate_params()
    
    def _calculate_params(self):
        """Calculate and print the number of trainable parameters"""
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params:,}")
        
    def _reset_parameters(self):
        """Initialize model parameters"""
        # Initialize linear layers with kaiming initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def preprocess(self, x: Tensor) -> Tensor:
        """
        Normalize input spectra and prepend std token
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, 1]
            
        Returns:
            Processed tensor with std token of shape [batch_size, seq_len+1]
        """
        # Handle shape: input is [batch_size, seq_len, 1]
        if x.dim() == 3 and x.shape[2] == 1:
            x = x.squeeze(-1)  # [batch_size, seq_len]
            
        batch_size, seq_len = x.shape
        
        # Compute statistics
        std = x.std(1, keepdim=True).clamp(min=1e-6)  # [batch_size, 1]
        mean = x.mean(1, keepdim=True)  # [batch_size, 1]
        
        # Normalize the spectrum
        x_normalized = (x - mean) / std  # [batch_size, seq_len]
        
        # Create std token
        # Following your original approach, use log10(std) as the std token value
        std_token = torch.log10(std)  # [batch_size, 1]
        
        # Concatenate std token with normalized spectrum
        # This is similar to your original approach of prepending the std
        x_with_std = torch.cat([std_token, x_normalized], dim=1)  # [batch_size, seq_len+1]
        
        return x_with_std
    
    def mask_sequence(self, x: Tensor) -> Tensor:
        """
        Apply random masking to the sequence if masking is enabled
        
        Args:
            x: Input tensor with std token of shape [batch_size, seq_len+1]
            
        Returns:
            Masked tensor of same shape (std token is never masked)
        """
        if not self.hparams.use_masking:
            return x
        
        # Apply masking, but preserve the std token (first position)
        batch_size, seq_len = x.shape
        masked_x = x.clone()
        
        # Skip the std token (first position) for masking
        spectrum_only = masked_x[:, 1:]
        
        # Use either num_chunks approach or ratio-based approach
        if self.hparams.mask_num_chunks > 0:
            # Similar to original code - fixed number of chunks
            for i in range(batch_size):
                for _ in range(self.hparams.mask_num_chunks):
                    # Find a valid start position (exclude the first token)
                    spectrum_len = seq_len - 1
                    chunk_width = min(self.hparams.mask_chunk_width, spectrum_len)
                    
                    # Ensure we don't start a chunk that would go past the end
                    if spectrum_len <= chunk_width:
                        continue
                        
                    start_pos = torch.randint(0, spectrum_len - chunk_width, (1,)).item()
                    # +1 in the position to account for the std token
                    masked_x[i, start_pos+1:start_pos+1+chunk_width] = 0
        else:
            # Ratio-based approach (alternative implementation)
            chunk_width = self.hparams.mask_chunk_width
            spectrum_len = seq_len - 1  # Excluding std token
            num_chunks = max(1, int(spectrum_len * self.hparams.mask_ratio / chunk_width))
            
            for i in range(batch_size):
                for _ in range(num_chunks):
                    if spectrum_len <= chunk_width:
                        continue
                    start_pos = torch.randint(0, spectrum_len - chunk_width, (1,)).item()
                    # +1 in the position to account for the std token
                    masked_x[i, start_pos+1:start_pos+1+chunk_width] = 0
        
        return masked_x
    
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """Forward pass through the model"""
        x = self.preprocess(x)
        return self.forward_without_preprocessing(x)
    
    def forward_without_preprocessing(self, x: Tensor) -> Dict[str, Tensor]:
        """Forward pass through the autoencoder without preprocessing"""
        batch_size = x.shape[0]
        
        # Flatten the input
        x_flat = x.reshape(batch_size, -1)
        
        # Encode
        h = self.input_proj(x_flat)
        
        # Encoder blocks
        for block in self.encoder_blocks:
            h = block(h)
        
        # Bottleneck
        latent = self.bottleneck_norm(self.bottleneck_down(h))
        h = self.bottleneck_up(latent)
        
        # Decoder blocks
        for block in self.decoder_blocks:
            h = block(h)
        
        # Output projection
        output_flat = self.output_proj(h)
        
        # Reshape back to original dimensions
        reconstructions = output_flat.reshape(batch_size, self.total_seq_len)
        
        return {
            "reconstructions": reconstructions,
            "latent": latent
        }
    
    def training_step(self, batch, batch_idx):
        # Get spectra from batch
        spectra = batch["spectra"]
        
        # Preprocess - adds std token
        input_data = self.preprocess(spectra)
        target = input_data.clone()
        
        # Apply masking if enabled
        if self.hparams.use_masking:
            masked_input = self.mask_sequence(input_data)
            
            # Forward pass
            output = self.forward_without_preprocessing(masked_input)["reconstructions"]
            
            # Calculate loss only on masked positions
            mask = (masked_input != target).to(output.dtype)
            loss = F.mse_loss(output * mask, target * mask) / (mask.sum() + 1e-8)
        else:
            # No masking, standard autoencoder
            output = self.forward_without_preprocessing(input_data)["reconstructions"]
            loss = F.mse_loss(output, target)
        
        # Log metrics
        self.log("training_loss", loss, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Get spectra from batch
        spectra = batch["spectra"]
        
        # Preprocess - adds std token
        input_data = self.preprocess(spectra)
        target = input_data.clone()
        
        # Apply masking if enabled
        if self.hparams.use_masking:
            masked_input = self.mask_sequence(input_data)
            
            # Forward pass
            output = self.forward_without_preprocessing(masked_input)["reconstructions"]
            
            # Calculate loss only on masked positions
            mask = (masked_input != target).to(output.dtype)
            loss = F.mse_loss(output * mask, target * mask) / (mask.sum() + 1e-8)
        else:
            # No masking, standard autoencoder
            output = self.forward_without_preprocessing(input_data)["reconstructions"]
            loss = F.mse_loss(output, target)
        
        # Log metrics
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_epoch_loss", loss, on_step=False, on_epoch=True, 
                prog_bar=True, sync_dist=True)
        
        return loss
    
    def training_epoch_end(self, outputs):
        if not outputs:
            print("No valid outputs received in training_epoch_end")
            return

        try:
            # Handle different output types
            if isinstance(outputs[0], dict):
                losses = [x['loss'] for x in outputs if 'loss' in x]
            elif isinstance(outputs[0], torch.Tensor):
                losses = outputs
            else:
                print("Unsupported data type in outputs.")
                return

            # Calculate average loss
            avg_train_loss = torch.stack(losses).mean()
            
            # Log with consistent naming
            self.log('train_epoch_loss', avg_train_loss, 
                    on_step=False, on_epoch=True, 
                    prog_bar=True, logger=True, sync_dist=True)

        except Exception as e:
            print(f"Error in training_epoch_end: {str(e)}")

    def validation_epoch_end(self, outputs):
        if not outputs:
            print("No valid outputs received in validation_epoch_end")
            return

        try:
            # Handle different output types
            if isinstance(outputs[0], dict):
                losses = [x['val_loss'] for x in outputs if 'val_loss' in x]
            elif isinstance(outputs[0], torch.Tensor):
                losses = outputs
            else:
                print("Unsupported data type in validation outputs.")
                return

            # Calculate average loss
            avg_val_loss = torch.stack(losses).mean()
            
            # Log with consistent naming
            self.log('val_epoch_loss', avg_val_loss, 
                    on_step=False, on_epoch=True, 
                    prog_bar=True, logger=True, sync_dist=True)

        except Exception as e:
            print(f"Error in validation_epoch_end: {str(e)}")
