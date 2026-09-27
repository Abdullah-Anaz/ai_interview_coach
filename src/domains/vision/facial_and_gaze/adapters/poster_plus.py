import logging
from typing import Any, Callable, Dict, List, Optional, cast

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

from thop import profile

logger: logging.Logger = logging.getLogger(__name__)

PredictLatentFn = Callable[[torch.Tensor], Optional[np.ndarray]]


class WindowCrossAttention(nn.Module):
    """
    Implements a window-based cross-attention mechanism for feature fusion.

    Args:
        dim (int): The dimensionality of the input feature vectors.
        window_size (int, optional): The size of the local attention window. Defaults to 7.
        num_heads (int, optional): The number of attention heads. Defaults to 8.
    """

    def __init__(self, dim: int, window_size: int = 7, num_heads: int = 8) -> None:
        super().__init__()
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
        """
        Executes the cross-attention computation across partitioned windows.

        Args:
            x_primary (torch.Tensor): Primary modality tensor.
            x_secondary (torch.Tensor): Secondary modality tensor.

        Returns:
            torch.Tensor: The fused feature tensor.
        """
        batch_size: int
        seq_len: int
        channels: int
        batch_size, seq_len, channels = x_primary.shape
        
        q: torch.Tensor = self.q_proj(x_primary).reshape(batch_size, seq_len, self.num_heads, channels // self.num_heads).transpose(1, 2)
        k: torch.Tensor = self.k_proj(x_secondary).reshape(batch_size, seq_len, self.num_heads, channels // self.num_heads).transpose(1, 2)
        v: torch.Tensor = self.v_proj(x_secondary).reshape(batch_size, seq_len, self.num_heads, channels // self.num_heads).transpose(1, 2)

        attn: torch.Tensor = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.softmax(attn)

        fused: torch.Tensor = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, channels)
        return self.out_proj(fused)


class POSTERPlus(nn.Module):
    """
    POSTER++ two-stream cross-fusion architecture for latent embedding extraction.

    Args:
        feature_dim (int, optional): The dimensionality of the output latent space. Defaults to 1024.
    """

    def __init__(self, feature_dim: int = 1024) -> None:
        super().__init__()
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

        Args:
            image_tensor (torch.Tensor): The raw RGB video frame tensor.
            landmark_tensor (Optional[torch.Tensor], optional): The spatial landmark representation.

        Returns:
            torch.Tensor: The 1024-dimensional latent embedding.
        """
        if landmark_tensor is None:
            landmark_tensor = image_tensor

        img_feat: torch.Tensor = self.image_backbone(image_tensor)
        lmk_feat: torch.Tensor = self.landmark_backbone(landmark_tensor)

        img_emb: torch.Tensor = self.img_proj(img_feat).unsqueeze(1)
        lmk_emb: torch.Tensor = self.lmk_proj(lmk_feat).unsqueeze(1)

        fused_emb: torch.Tensor = self.cross_attention(img_emb, lmk_emb)
        fused_emb = self.layer_norm(fused_emb + img_emb)
        
        return fused_emb.squeeze(1)


def _load_poster_weights(model: nn.Module, weights_path: str, device: torch.device) -> nn.Module:
    """
    Safely loads serialized state dictionary weights into the POSTER++ model architecture.

    Args:
        model (nn.Module): The initialized POSTER++ model instance.
        weights_path (str): The local file system path to the `.pth` weights file.
        device (torch.device): The target compute device.

    Returns:
        nn.Module: The model with loaded weights, or randomly initialized weights if missing.
    """
    try:
        state_dict: Dict[str, Any] = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        logger.info("Successfully loaded POSTER++ latent weights from %s", weights_path)
    except (FileNotFoundError, RuntimeError) as exc:
        logger.warning("POSTER++ weights unavailable at %s. Utilizing initialized state. Error: %s", weights_path, exc)
    except Exception as exc:
        logger.error("Failed to map POSTER++ weights to model architecture: %s", exc)
        
    return model


def create_poster_adapter(config: Dict[str, Any], device: torch.device) -> PredictLatentFn:
    """
    Generates a functional closure for extracting latent embeddings over temporal tensor sequences.

    Args:
        config (Dict[str, Any]): Configuration parameters containing model and weight definitions.
        device (torch.device): The target compute device.

    Returns:
        PredictLatentFn: A callable function accepting a video tensor sequence and returning 
                         aggregated latent embeddings for FACS mapping.
    """
    weights_path: str = config.get("poster_weights_path", "src/models/weights/poster_plus_latent.pth")
    feature_dim: int = config.get("feature_dim", 1024)
    batch_size: int = config.get("inference_batch_size", 16)
    
    model: POSTERPlus = POSTERPlus(feature_dim=feature_dim)
    model = _load_poster_weights(model, weights_path, device)
    model.to(device)
    model.eval()

    # 1. Calculate Parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    config["parameter_count_millions"] = round(total_params / 1_000_000.0, 2)

    # 2. Calculate GFLOPs
    try:
        # POSTER++ expects a standard tensor input
        dummy_input = torch.randn(1, 3, 224, 224, device=device)
        macs, _ = profile(model, inputs=(dummy_input,), verbose=False)
        config["computational_load_gflops"] = round((macs * 2) / 1_000_000_000.0, 2)
    except ImportError:
        logger.warning("thop library not found. Skipping GFLOPs calculation.")
        config["computational_load_gflops"] = 0.0
    except Exception as exc:
        logger.warning("Failed to profile GFLOPs: %s", exc)
        config["computational_load_gflops"] = 0.0
    
    def predict(sequence_tensor: torch.Tensor, fps: float) -> Optional[np.ndarray]:
        """
        Executes sequence-level temporal inference to extract latent embeddings using micro-batching.

        Args:
            sequence_tensor (torch.Tensor): A batch sequence of video frames formatted as (T, C, H, W).
            fps (float): The native frame rate of the video sequence (mandated by the PredictTrajectoryFn interface).

        Returns:
            Optional[np.ndarray]: Array representing the mean latent embedding across the sequence, 
                                  or None if execution fails.
        """
        try:
            with torch.no_grad():
                seq_device: torch.Tensor = sequence_tensor.to(device, dtype=torch.float32)
                num_frames: int = seq_device.size(0)
                all_latents: List[torch.Tensor] = []

                for i in range(0, num_frames, batch_size):
                    batch_frames: torch.Tensor = seq_device[i : i + batch_size]
                    batch_latents: torch.Tensor = model(image_tensor=batch_frames)
                    all_latents.append(batch_latents)

                latent_embeddings: torch.Tensor = torch.cat(all_latents, dim=0)
                temporal_mean_embedding: torch.Tensor = latent_embeddings.mean(dim=0)
                
                return cast(np.ndarray, temporal_mean_embedding.cpu().numpy())
                
        except (RuntimeError, ValueError) as exc:
            logger.error("Tensor shape or dimension mismatch during POSTER++ latent prediction: %s", exc)
            return None
        except Exception as exc:
            logger.error("Unexpected execution failure during POSTER++ sequence prediction: %s", exc)
            return None
            
    return predict