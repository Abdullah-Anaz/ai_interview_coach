from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models


class WindowCrossAttention(nn.Module):
    """
    Implements a window-based cross-attention mechanism for feature fusion[cite: 5].
    """

    def __init__(self, dim: int, window_size: int = 7, num_heads: int = 8) -> None:
        super().__init__()
        if dim <= 0 or window_size <= 0 or num_heads <= 0:
            raise ValueError("Dimensions, window size, and heads must be strictly positive.")
        if dim % num_heads != 0:
            raise ValueError("Dimension must be perfectly divisible by the number of heads.")

        self.dim: int = dim
        self.window_size: int = window_size
        self.num_heads: int = num_heads
        
        head_dim: int = dim // num_heads
        self.scale: float = head_dim ** -0.5

        self.q_proj: nn.Linear = nn.Linear(dim, dim, bias=True)
        self.k_proj: nn.Linear = nn.Linear(dim, dim, bias=True)
        self.v_proj: nn.Linear = nn.Linear(dim, dim, bias=True)
        self.out_proj: nn.Linear = nn.Linear(dim, dim, bias=True)
        self.softmax: nn.Softmax = nn.Softmax(dim=-1)

    def forward(self, x_primary: torch.Tensor, x_secondary: torch.Tensor) -> torch.Tensor:
        """Executes cross-attention computation across partitioned sequences."""
        if not isinstance(x_primary, torch.Tensor) or not isinstance(x_secondary, torch.Tensor):
            raise TypeError("Attention inputs must be strictly typed torch.Tensor objects.")
        if x_primary.dim() != 3 or x_secondary.dim() != 3:
            raise ValueError("Attention inputs must be strictly 3-dimensional (Batch, Sequence, Channels).")

        batch_size: int
        seq_len: int
        channels: int
        batch_size, seq_len, channels = x_primary.shape
        
        head_dim: int = channels // self.num_heads

        q: torch.Tensor = self.q_proj(x_primary).reshape(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
        k: torch.Tensor = self.k_proj(x_secondary).reshape(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
        v: torch.Tensor = self.v_proj(x_secondary).reshape(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)

        attn: torch.Tensor = self.softmax((q @ k.transpose(-2, -1)) * self.scale)
        fused: torch.Tensor = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, channels)
        
        return self.out_proj(fused)


class POSTERPlus(nn.Module):
    """
    POSTER++ two-stream cross-fusion architecture for latent embedding extraction[cite: 5].
    """

    def __init__(self, feature_dim: int = 1024) -> None:
        super().__init__()
        if not isinstance(feature_dim, int) or isinstance(feature_dim, bool):
            raise TypeError("feature_dim must be an integer.")
        if feature_dim <= 0:
            raise ValueError("feature_dim must be strictly positive.")

        self.image_backbone: nn.Module = models.resnet18(weights=None)
        self.image_backbone.fc = nn.Identity()
        
        self.landmark_backbone: nn.Module = models.mobilenet_v2(weights=None)
        self.landmark_backbone.classifier = nn.Identity()

        self.img_proj: nn.Linear = nn.Linear(512, feature_dim)
        self.lmk_proj: nn.Linear = nn.Linear(1280, feature_dim)

        self.cross_attention: WindowCrossAttention = WindowCrossAttention(dim=feature_dim)
        self.layer_norm: nn.LayerNorm = nn.LayerNorm(feature_dim)

    def forward(self, image_tensor: torch.Tensor, landmark_tensor: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Executes the two-stream forward pass to generate a pre-classification latent embedding.
        If no landmark tensor is provided, the image tensor is duplicated across both streams[cite: 5].
        """
        if not isinstance(image_tensor, torch.Tensor):
            raise TypeError("image_tensor must be a strictly typed torch.Tensor.")
        if image_tensor.dim() != 4:
            raise ValueError("image_tensor must be 4-dimensional (Batch, Channels, Height, Width).")

        if landmark_tensor is None:
            landmark_tensor = image_tensor
        elif not isinstance(landmark_tensor, torch.Tensor):
            raise TypeError("landmark_tensor must be a strictly typed torch.Tensor.")

        img_feat: torch.Tensor = self.image_backbone(image_tensor)
        lmk_feat: torch.Tensor = self.landmark_backbone(landmark_tensor)

        img_emb: torch.Tensor = self.img_proj(img_feat).unsqueeze(1)
        lmk_emb: torch.Tensor = self.lmk_proj(lmk_feat).unsqueeze(1)

        fused_emb: torch.Tensor = self.cross_attention(img_emb, lmk_emb)
        fused_emb = self.layer_norm(fused_emb + img_emb)
        
        return fused_emb.squeeze(1)