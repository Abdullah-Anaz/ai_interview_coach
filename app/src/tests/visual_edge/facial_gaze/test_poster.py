import pytest
import torch

from app.src.visual_edge.facial_gaze.poster import POSTERPlus, WindowCrossAttention


@pytest.fixture
def gpu_device() -> torch.device:
    """Provides a valid hardware device for GPU-bound execution limits."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_windowcrossattention_expected_behaviour_for_valid_tensors(gpu_device: torch.device) -> None:
    """Verifies that the attention mechanism mathematically fuses valid 3D sequence tensors."""
    attention = WindowCrossAttention(dim=1024, num_heads=8).to(gpu_device)
    attention.eval()
    
    primary: torch.Tensor = torch.randn(16, 1, 1024, device=gpu_device)
    secondary: torch.Tensor = torch.randn(16, 1, 1024, device=gpu_device)
    
    output: torch.Tensor = attention(primary, secondary)
    
    assert output.shape == (16, 1, 1024)
    assert not torch.isnan(output).any()


def test_windowcrossattention_raises_for_indivisible_dimensions() -> None:
    """Ensures instantiation fails if head dimensions cannot be symmetrically partitioned."""
    with pytest.raises(ValueError, match="Dimension must be perfectly divisible by the number of heads."):
        # 1023 is not divisible by 8, which will correctly trigger the ValueError
        WindowCrossAttention(dim=1023, num_heads=8)

def test_windowcrossattention_raises_for_invalid_tensor_shape(gpu_device: torch.device) -> None:
    """Ensures execution mathematically rejects 2D or 4D tensor injections."""
    attention = WindowCrossAttention(dim=1024, num_heads=8).to(gpu_device)
    invalid_tensor: torch.Tensor = torch.randn(16, 1024, device=gpu_device)
    
    with pytest.raises(ValueError, match="Attention inputs must be strictly 3-dimensional"):
        attention(invalid_tensor, invalid_tensor)


def test_posterplus_expected_behaviour_for_single_stream_input(gpu_device: torch.device) -> None:
    """Verifies the unified model architecture outputs the precise target dimensionality from an isolated RGB feed."""
    model = POSTERPlus(feature_dim=1024).to(gpu_device)
    model.eval()
    
    synthetic_image: torch.Tensor = torch.randn(4, 3, 224, 224, device=gpu_device)
    
    with torch.no_grad():
        embedding: torch.Tensor = model(synthetic_image)
        
    assert embedding.shape == (4, 1024)
    assert not torch.isnan(embedding).any()


def test_posterplus_expected_behaviour_for_dual_stream_input(gpu_device: torch.device) -> None:
    """Verifies the architecture safely computes explicit dual-modality injections without collapsing."""
    model = POSTERPlus(feature_dim=1024).to(gpu_device)
    model.eval()
    
    synthetic_image: torch.Tensor = torch.randn(2, 3, 224, 224, device=gpu_device)
    synthetic_landmark: torch.Tensor = torch.randn(2, 3, 224, 224, device=gpu_device)
    
    with torch.no_grad():
        embedding: torch.Tensor = model(synthetic_image, synthetic_landmark)
        
    assert embedding.shape == (2, 1024)


def test_posterplus_raises_for_invalid_feature_dim() -> None:
    """Ensures initialization blocks structurally invalid latent dimensions."""
    with pytest.raises(ValueError, match="feature_dim must be strictly positive."):
        POSTERPlus(feature_dim=-1024)
        
    with pytest.raises(TypeError, match="feature_dim must be an integer."):
        POSTERPlus(feature_dim="1024")  # type: ignore


def test_posterplus_raises_for_invalid_tensor_types(gpu_device: torch.device) -> None:
    """Ensures the network's forward pass blocks untyped python data structures."""
    model = POSTERPlus(feature_dim=1024).to(gpu_device)
    
    with pytest.raises(TypeError, match="image_tensor must be a strictly typed torch.Tensor."):
        model([1, 2, 3])  # type: ignore


def test_posterplus_raises_for_invalid_tensor_shape(gpu_device: torch.device) -> None:
    """Ensures the network specifically checks that temporal feeds maintain 4 dimensions."""
    model = POSTERPlus(feature_dim=1024).to(gpu_device)
    invalid_image: torch.Tensor = torch.randn(3, 224, 224, device=gpu_device) # Missing batch dimension
    
    with pytest.raises(ValueError, match="image_tensor must be 4-dimensional"):
        model(invalid_image)