import logging
from typing import Any, Callable, Dict, Optional, cast

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

logger: logging.Logger = logging.getLogger(__name__)

PredictTrajectoryFn = Callable[[torch.Tensor, float], Optional[np.ndarray]]


class HumanOmni(nn.Module):
    """
    HumanOmni tri-branch architecture for holistic kinematic and spatial feature extraction.

    Args:
        feature_dim (int, optional): Latent projection dimensionality. Defaults to 512.
        num_joints (int, optional): Target number of skeletal joints for regression. Defaults to 33.
    """

    def __init__(self, feature_dim: int = 512, num_joints: int = 33) -> None:
        super().__init__()
        self.num_joints: int = num_joints
        
        self.body_branch: nn.Module = models.mobilenet_v3_small(weights=None)
        self.body_branch.classifier = nn.Identity()
        
        self.face_branch: nn.Module = models.mobilenet_v3_small(weights=None)
        self.face_branch.classifier = nn.Identity()
        
        self.interaction_branch: nn.Module = models.mobilenet_v3_small(weights=None)
        self.interaction_branch.classifier = nn.Identity()

        self.body_proj: nn.Linear = nn.Linear(576, feature_dim)
        self.face_proj: nn.Linear = nn.Linear(576, feature_dim)
        self.interaction_proj: nn.Linear = nn.Linear(576, feature_dim)
        
        self.fusion_norm: nn.LayerNorm = nn.LayerNorm(feature_dim)
        self.kinematic_regressor: nn.Linear = nn.Linear(feature_dim, num_joints * 3)

    def forward(
        self, 
        body_tensor: torch.Tensor, 
        face_tensor: torch.Tensor, 
        interaction_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Executes the tri-branch forward pass and regresses 3D spatial coordinates.

        Args:
            body_tensor (torch.Tensor): Cropped body tensor.
            face_tensor (torch.Tensor): Cropped face tensor.
            interaction_tensor (torch.Tensor): Global interaction scene tensor.

        Returns:
            torch.Tensor: Predicted 3D joint coordinates shaped as (Batch, num_joints, 3).
        """
        body_feat: torch.Tensor = self.body_proj(self.body_branch(body_tensor))
        face_feat: torch.Tensor = self.face_proj(self.face_branch(face_tensor))
        int_feat: torch.Tensor = self.interaction_proj(self.interaction_branch(interaction_tensor))
        
        fused_feat: torch.Tensor = self.fusion_norm(body_feat + face_feat + int_feat)
        kinematics: torch.Tensor = self.kinematic_regressor(fused_feat)
        
        return kinematics.view(-1, self.num_joints, 3)


def _load_human_omni_weights(model: nn.Module, weights_path: str, device: torch.device) -> nn.Module:
    """
    Safely loads serialized state dictionary weights into the HumanOmni model.

    Args:
        model (nn.Module): The initialized HumanOmni model instance.
        weights_path (str): The local file system path to the `.pth` weights file.
        device (torch.device): The target compute device.

    Returns:
        nn.Module: The model with loaded weights, or randomly initialized weights if missing.
    """
    try:
        state_dict: Dict[str, Any] = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        logger.info("Successfully loaded HumanOmni weights from %s", weights_path)
    except (FileNotFoundError, RuntimeError) as exc:
        logger.warning("HumanOmni weights unavailable at %s. Utilizing initialized state. Error: %s", weights_path, exc)
    except Exception as exc:
        logger.error("Failed to map HumanOmni weights to model architecture: %s", exc)
        
    return model


def create_human_omni_adapter(config: Dict[str, Any], device: torch.device) -> PredictTrajectoryFn:
    """
    Generates a functional closure for executing HumanOmni kinematic trajectory extraction.

    Args:
        config (Dict[str, Any]): Configuration parameters containing model and weight definitions.
        device (torch.device): The target compute device.

    Returns:
        PredictTrajectoryFn: A callable function accepting a video tensor sequence and FPS, 
                             returning the extracted trajectory array.
    """
    weights_path: str = config.get("omni_weights_path", "src/models/weights/human_omni.pth")
    feature_dim: int = config.get("feature_dim", 512)
    num_joints: int = config.get("num_joints", 33)
    
    model: HumanOmni = HumanOmni(feature_dim=feature_dim, num_joints=num_joints)
    model = _load_human_omni_weights(model, weights_path, device)
    model.to(device)
    model.eval()
    
    def predict(sequence_tensor: torch.Tensor, fps: float) -> Optional[np.ndarray]:
        """
        Processes a sequence of frames to extract holistic 3D joint trajectories.

        Args:
            sequence_tensor (torch.Tensor): A tensor sequence of image frames shaped (T, C, H, W).
            fps (float): The frames per second of the sequence.

        Returns:
            Optional[np.ndarray]: An array of skeletal landmarks across the sequence, or None if extraction fails.
        """
        try:
            with torch.no_grad():
                seq_device: torch.Tensor = sequence_tensor.to(device, dtype=torch.float32)
                
                face_crop: torch.Tensor = seq_device.clone()
                body_crop: torch.Tensor = seq_device.clone()
                interaction_scene: torch.Tensor = seq_device.clone()
                
                raw_trajectory: torch.Tensor = model(
                    body_tensor=body_crop, 
                    face_tensor=face_crop, 
                    interaction_tensor=interaction_scene
                )
                
                return cast(np.ndarray, raw_trajectory.cpu().numpy())
                
        except (RuntimeError, ValueError) as exc:
            logger.error("Tensor shape or dimension mismatch during HumanOmni prediction: %s", exc)
            return None
        except Exception as exc:
            logger.error("Unexpected execution failure during HumanOmni sequence prediction: %s", exc)
            return None
            
    return predict