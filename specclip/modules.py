# This implementation is adapted from the AstroCLIP framework
# (Parker et al. 2024), with additional modules and modifications specific to SpecCLIP.
# Original AstroCLIP code: https://github.com/PolymathicAI/AstroCLIP

import math
import numbers
from typing import Callable, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union

class FlexibleAttention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        causal: bool = False,
        dropout: float = 0.1,
        bias: bool = True,
        use_flash: bool = True
    ):
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError(f"embedding_dim must be divisible by num_heads")
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.causal = causal
        self.head_dim = embedding_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Check Flash Attention availability
        self.uses_flash = use_flash and hasattr(F, "scaled_dot_product_attention")
        
        if self.uses_flash:
            # Flash path: separate Q and KV projections
            self.q_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)
            self.kv_proj = nn.Linear(embedding_dim, 2 * embedding_dim, bias=bias)
        else:
            # Non-flash path: unified QKV projection
            self.attention = nn.Linear(embedding_dim, 3 * embedding_dim, bias=bias)
            
        # Output projection (same for both paths)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        
        # Dropout layers
        self.attention_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)
        
        # Causal mask buffer for non-flash path
        if self.causal and not self.uses_flash:
            self.register_buffer("mask", torch.empty((1, 1, 0, 0), dtype=bool))
        
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with scaled normal distribution"""
        std = 0.02
        if self.uses_flash:
            for proj in [self.q_proj, self.kv_proj, self.out_proj]:
                nn.init.normal_(proj.weight, std=std)
                if proj.bias is not None:
                    nn.init.zeros_(proj.bias)
        else:
            nn.init.normal_(self.attention.weight, std=std)
            if self.attention.bias is not None:
                nn.init.zeros_(self.attention.bias)
            nn.init.normal_(self.out_proj.weight, std=std)
            if self.out_proj.bias is not None:
                nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        aux_query: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with optional auxiliary query for cross-attention.
        
        Args:
            x: Input tensor [batch, seq_len, embedding_dim]
            aux_query: Optional auxiliary query for cross-attention
                      - If None: self-attention (Q=K=V from x)
                      - If provided: cross-attention (Q from aux_query, K/V from x)
            need_weights: Whether to return attention weights
            key_padding_mask: Mask for padded positions (not used in Flash path)
            attn_mask: Additional attention mask (not used in Flash path)
        
        Returns:
            output: Attention output [batch, query_len, embedding_dim]
            weights: Attention weights if need_weights=True, else None
        """
        if self.uses_flash:
            return self._forward_flash(x, aux_query, need_weights)
        else:
            return self._forward_standard(
                x, aux_query, key_padding_mask, need_weights, attn_mask
            )

    def _forward_flash(
        self,
        x: torch.Tensor,
        aux_query: Optional[torch.Tensor],
        need_weights: bool
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Flash Attention implementation"""
        B, T, C = x.shape
        H = self.num_heads
        D = self.head_dim
    
        # Query: from aux_query if provided, else from x
        if aux_query is not None:
            q = self.q_proj(aux_query)
            Tq = aux_query.size(1)
        else:
            q = self.q_proj(x)
            Tq = T
    
        # Key and Value: always from x
        kv = self.kv_proj(x)
        k, v = kv.chunk(2, dim=-1)
    
        # Reshape for multi-head attention: [B, H, T, D]
        q = q.view(B, Tq, H, D).transpose(1, 2)
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)
    
        # Apply scaling to query
        q = q * (D ** -0.5)
    
        # Compute attention
        dropout_p = self.dropout if self.training else 0.0
    
        if need_weights:
            # Manual attention computation to extract weights
            attn_weights = torch.matmul(q, k.transpose(-2, -1))
        
            if self.causal:
                causal_mask = torch.triu(
                    torch.ones(Tq, T, dtype=torch.bool, device=q.device),
                    diagonal=1
                )
                attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
        
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = F.dropout(attn_weights, p=dropout_p, training=self.training)
            y = torch.matmul(attn_weights, v)
        else:
            # Use Flash Attention (faster, no weights)
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=self.causal
            )
            attn_weights = None
    
        # Reshape and project: [B, H, Tq, D] -> [B, Tq, C]
        y = y.transpose(1, 2).contiguous().view(B, Tq, C)
        y = self.residual_dropout(self.out_proj(y))
    
        return y, attn_weights

    def _forward_standard(
        self,
        x: torch.Tensor,
        aux_query: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        need_weights: bool,
        attn_mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Standard attention implementation (non-Flash)"""
        B, T, C = x.shape
        H = self.num_heads
        D = self.head_dim
        
        # Project Q, K, V
        if aux_query is not None:
            # Cross-attention: separate projections
            qkv_x = self.attention(x)
            _, k, v = qkv_x.split(self.embedding_dim, dim=2)
            
            qkv_q = self.attention(aux_query)
            q = qkv_q[:, :, :self.embedding_dim]
            Tq = aux_query.size(1)
        else:
            # Self-attention: unified projection
            qkv = self.attention(x)
            q, k, v = qkv.split(self.embedding_dim, dim=2)
            Tq = T
        
        # Reshape: [B, T, H, D] -> [B, H, T, D]
        q = q.view(B, Tq, H, D).transpose(1, 2)
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)
        
        # Compute attention scores
        att = (q @ k.transpose(-2, -1)) * self.scale
        
        # Apply masks
        if self.causal and attn_mask is None:
            if self.mask.shape[2:] != (Tq, T):
                mask = torch.triu(torch.ones(Tq, T, dtype=torch.bool), diagonal=1)
                self.register_buffer("mask", mask.view(1, 1, Tq, T))
            attn_mask = self.mask
        
        if attn_mask is not None:
            att = att.masked_fill(attn_mask, float("-inf"))
        if key_padding_mask is not None:
            att = att.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), 
                float("-inf")
            )
            
        # Softmax and dropout
        att = F.softmax(att, dim=-1)
        att = self.attention_dropout(att)
        
        # Apply attention to values
        y = att @ v
        
        # Reshape: [B, H, Tq, D] -> [B, Tq, C]
        y = y.transpose(1, 2).contiguous().view(B, Tq, C)
        y = self.residual_dropout(self.out_proj(y))
        
        return y, att if need_weights else None


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        causal: bool = False,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        super().__init__()
        
        # Use FlexibleAttention with same state dict structure
        self.attention = FlexibleAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            causal=causal,
            dropout=dropout,
            bias=bias
        )
        
        # Layer normalizations
        self.ln1 = LayerNorm(embedding_dim, bias=bias)
        self.ln2 = LayerNorm(embedding_dim, bias=bias)
        
        # MLP block
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim, bias=bias),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim, bias=bias),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        aux_query: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """
        Forward pass with optional auxiliary query.
        
        Args:
            x: Input features [batch, seq_len, embedding_dim]
            aux_query: Optional auxiliary query (e.g., wavelength embeddings)
                      If None: self-attention
                      If provided: cross-attention
            return_attention: Whether to return attention weights
        
        Returns:
            If return_attention=False: output tensor
            If return_attention=True: (output, attention_weights)
        """
        # Pre-norm for attention
        normed_x = self.ln1(x)
        query = self.ln1(aux_query) if aux_query is not None else normed_x
    
        # Attention with optional weight extraction
        attended, attn_weights = self.attention(
            normed_x,
            aux_query=query,
            need_weights=return_attention
        )
    
        # First residual connection
        # If cross-attention: add to aux_query
        # If self-attention: add to x
        if aux_query is not None:
            x = aux_query + attended
        else:
            x = x + attended
    
        # Pre-norm for MLP + second residual
        x = x + self.mlp(self.ln2(x))
    
        # Return format based on flag
        if return_attention:
            return x, attn_weights
        return x
    
#EnhancedWavelengthAttention = FlexibleAttention
    
class CrossAttentionHead(nn.Module):
    """Cross-attention head with dropout.

    This module is a single head of a cross-attention layer. It takes a query and a key
    tensor, computes the attention weights, and returns the weighted sum of the values
    tensor. The attention weights are also returned.

    :param embed_dim: dimensionality of the input tensors
    :param n_head: number of heads
    :param model_embed_dim: dimensionality of the model tensors
    :param dropout: amount of dropout
    """

    embed_dim: int
    n_head: int
    model_embed_dim: int
    dropout: float

    def __init__(
        self,
        embed_dim: int,
        n_head: int,
        model_embed_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_head,
            batch_first=True,
            kdim=model_embed_dim,
            vdim=model_embed_dim,
        )
        self.layernorm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.tensor):
        batch_size = x.shape[0]
        attentions = self.multihead_attn(
            query=self.query.repeat(batch_size, 1, 1),
            key=x,
            value=x,
            average_attn_weights=False,
        )[0]
        x = self.layernorm(self.dropout(attentions))
        return x, attentions

class MLP(nn.Module):
    """A two-layer MLP.

    This uses a fully-connected layer to encode the input, then applies a non-linearity,
    then uses another fully-connected layer to decode back to the initial dimension, and
    finally applies (optional) dropout.

    :param in_features: size of input layer
    :param hidden_features: size of hidden layer
    :param activation: activation function to use after the expansion; default: GELU
    :param dropout: amount of dropout
    :param bias: whether to use bias in the layers
    """

    in_features: int
    hidden_features: int
    activation: Callable
    dropout: float
    bias: bool

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        activation: Optional[Callable] = None,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.activation = activation if activation is not None else nn.GELU()
        self.dropout = dropout
        self.bias = bias

        self.encoder = nn.Linear(in_features, hidden_features, bias=bias)
        self.decoder = nn.Linear(hidden_features, in_features, bias=bias)
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.activation(x)
        x = self.decoder(x)
        if self.dropout_layer is not None:
            x = self.dropout_layer(x)
        return x

class LayerNorm(nn.Module):
    """Layer normalized with optional bias.

    This is based on PyTorch's :class:`~torch.nn.LayerNorm` module but is needed because
    PyTorch's version does not support disabling the bias.

    :param shape: shape of the input, following an arbitrary number of batch dimensions;
        that is, the input has dimensions `[d1, ..., dk, shape[0], ..., shape[-1]]`
    :param eps: value added to the denominator for numerical stability
    :param bias: whether to include a bias term
    :param dtype: data type to use for the parameters
    """

    normalized_shape: Tuple[int, ...]
    eps: float

    def __init__(
        self,
        shape: Union[int, Tuple[int, ...], torch.Size],
        eps: float = 1e-5,
        bias: bool = True,
        dtype=None,
    ):
        super().__init__()

        self.eps = eps
        if isinstance(shape, numbers.Integral):
            self.normalized_shape = (shape,)
        else:
            self.normalized_shape = tuple(shape)

        self.weight = nn.Parameter(torch.empty(shape))
        self.bias = nn.Parameter(torch.empty(shape)) if bias else None

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, input):
        return F.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps
        )


class TiedLinear(nn.Module):
    """A dense linear layer whose parameters are tied to a tensor provided by the user.

    Using this layer is equivalent to using the functional form,
    :func:`~torch.nn.functional.linear`. The utility of having a module is that it will
    show up in module summaries, which can help to make the structure of the model more
    transparent.

    :param weight: weight tensor
    :param bias: bias tensor; if not provided, there will be no bias
    """

    in_features: int
    """size of each input sample."""

    out_features: int
    """size of each output sample."""

    def __init__(
        self,
        weight: Union[torch.Tensor, nn.Parameter],
        bias: Union[None, torch.Tensor, nn.Parameter],
    ):
        super().__init__()

        if weight.ndim != 2:
            raise ValueError(
                f"weight parameter has {weight.ndim} dimensions, should have 2"
            )
        self.out_features, self.in_features = weight.shape

        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )


def _init_by_depth(module: nn.Module, depth: int) -> None:
    """Initialize the weights of a module based on the depth of the model."""
    if isinstance(module, nn.Linear):
        fan_in = module.weight.size(-1)
        std = 1 / math.sqrt(2 * fan_in * depth)
        nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
