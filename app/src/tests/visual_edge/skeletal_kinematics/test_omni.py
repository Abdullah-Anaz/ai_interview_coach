import pytest
import torch

from app.src.visual_edge.skeletal_kinematics.omni import HumanOmni


@pytest.fixture
def gpu_device() -> torch.device:
    """Provides a valid hardware device for GPU-bound execution limits."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_humanomni_expected_behaviour_for_full_tri_branch_input(gpu_device: torch.device) -> None:
    """Verifies that the architecture successfully fuses all three spatial modalities into a 3D coordinate matrix."""
    model = HumanOmni(feature_dim=512, num_joints=33).to(gpu_device)
    model.eval()
    
    body: torch.Tensor = torch.randn(4, 3, 224, 224, device=gpu_device)
    face: torch.Tensor = torch.randn(4, 3, 224, 224, device=gpu_device)
    scene: torch.Tensor = torch.randn(4, 3, 224, 224, device=gpu_device)
    
    with torch.no_grad():
        kinematics: torch.Tensor = model(body_tensor=body, face_tensor=face, interaction_tensor=scene)
        
    assert kinematics.shape == (4, 33, 3)
    assert not torch.isnan(kinematics).any()


def test_humanomni_expected_behaviour_for_isolated_body_tensor(gpu_device: torch.device) -> None:
    """Verifies that the architecture automatically clones the primary tensor across missing auxiliary branches."""
    model = HumanOmni(feature_dim=512, num_joints=33).to(gpu_device)
    model.eval()
    
    body: torch.Tensor = torch.randn(2, 3, 224, 224, device=gpu_device)
    
    with torch.no_grad():
        kinematics: torch.Tensor = model(body_tensor=body)
        
    assert kinematics.shape == (2, 33, 3)


def test_humanomni_raises_for_invalid_initialization_dimensions() -> None:
    """Ensures initialization blocks structurally impossible mathematical configurations."""
    with pytest.raises(ValueError, match="feature_dim must be a strictly positive integer."):
        HumanOmni(feature_dim=-512)
        
    with pytest.raises(ValueError, match="num_joints must be a strictly positive integer."):
        HumanOmni(feature_dim=512, num_joints=0)


def test_humanomni_raises_for_invalid_tensor_types(gpu_device: torch.device) -> None:
    """Ensures the network's forward pass blocks untyped python data structures."""
    model = HumanOmni().to(gpu_device)
    
    with pytest.raises(TypeError, match="body_tensor must be a strictly typed torch.Tensor."):
        model(body_tensor=[1, 2, 3])  # type: ignore


def test_humanomni_raises_for_invalid_tensor_shape(gpu_device: torch.device) -> None:
    """Ensures the network specifically checks that temporal feeds maintain 4 dimensions."""
    model = HumanOmni().to(gpu_device)
    
    # Missing batch dimension
    invalid_body: torch.Tensor = torch.randn(3, 224, 224, device=gpu_device) 
    
    with pytest.raises(ValueError, match="body_tensor must be 4-dimensional"):
        model(body_tensor=invalid_body)