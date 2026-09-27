from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models


class HumanOmni(nn.Module):
    """
    HumanOmni tri-branch architecture for holistic kinematic and spatial feature extraction[cite: 14].
    """

    def __init__(self, feature_dim: int = 512, num_joints: int = 33) -> None:
        super().__init__()
        
        if not isinstance(feature_dim, int) or feature_dim <= 0:
            raise ValueError("feature_dim must be a strictly positive integer.")
        if not isinstance(num_joints, int) or num_joints <= 0:
            raise ValueError("num_joints must be a strictly positive integer.")
            
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
        face_tensor: Optional[torch.Tensor] = None, 
        interaction_tensor: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Executes the tri-branch forward pass and regresses 3D spatial coordinates[cite: 14].
        Automatically clones the primary tensor across missing auxiliary branches to maintain structural integrity.
        """
        if not isinstance(body_tensor, torch.Tensor):
            raise TypeError("body_tensor must be a strictly typed torch.Tensor.")
        if body_tensor.dim() != 4:
            raise ValueError("body_tensor must be 4-dimensional (Batch, Channels, Height, Width).")

        face_input: torch.Tensor = face_tensor if face_tensor is not None else body_tensor
        interaction_input: torch.Tensor = interaction_tensor if interaction_tensor is not None else body_tensor

        body_feat: torch.Tensor = self.body_proj(self.body_branch(body_tensor))
        face_feat: torch.Tensor = self.face_proj(self.face_branch(face_input))
        int_feat: torch.Tensor = self.interaction_proj(self.interaction_branch(interaction_input))
        
        fused_feat: torch.Tensor = self.fusion_norm(body_feat + face_feat + int_feat)
        kinematics: torch.Tensor = self.kinematic_regressor(fused_feat)
        
        return kinematics.view(-1, self.num_joints, 3)