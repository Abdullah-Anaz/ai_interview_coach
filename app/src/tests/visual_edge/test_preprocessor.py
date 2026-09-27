import pytest
import numpy as np
import torch
from typing import Tuple

from app.src.visual_edge.preprocessor import process_facial_frames


@pytest.fixture
def target_device() -> torch.device:
    """Yields the target GPU device if available, otherwise safely falls back for test execution."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def valid_video_array() -> np.ndarray:
    """Generates a strictly compliant 4D numpy uint8 array simulating a batch of RGB frames."""
    return np.random.randint(0, 255, (4, 480, 640, 3), dtype=np.uint8)


@pytest.fixture
def non_uint8_array() -> np.ndarray:
    """Generates a float-based array to test format coercion and OpenCV stability."""
    return np.random.rand(2, 300, 300, 3).astype(np.float32)


def test_process_facial_frames_expected_behaviour_for_standard_input(
    valid_video_array: np.ndarray, 
    target_device: torch.device
) -> None:
    """Tests if facial preprocessing successfully crops, resizes, and normalizes into target bounds."""
    target_bounds: Tuple[int, int] = (224, 224)
    result: torch.Tensor = process_facial_frames(
        valid_video_array, 
        device=target_device, 
        target_size=target_bounds
    )
    
    assert result is not None
    assert isinstance(result, torch.Tensor)
    assert result.device.type == target_device.type
    assert result.dtype == torch.float32
    assert result.shape == (4, 3, 224, 224)
    assert torch.max(result).item() <= 3.0 
    assert torch.min(result).item() >= -3.0


def test_process_facial_frames_expected_behaviour_for_no_face_detected(
    valid_video_array: np.ndarray, 
    target_device: torch.device
) -> None:
    """Tests the fallback routing when OpenCV Haar cascades fail to detect facial geometry."""
    result: torch.Tensor = process_facial_frames(
        valid_video_array, 
        device=target_device, 
        target_size=(256, 256)
    )
    
    assert result.shape == (4, 3, 256, 256)


def test_process_facial_frames_expected_behaviour_for_coercion(
    non_uint8_array: np.ndarray, 
    target_device: torch.device
) -> None:
    """Tests if the pipeline successfully coerces float arrays into OpenCV-compliant uint8 arrays."""
    result: torch.Tensor = process_facial_frames(
        non_uint8_array, 
        device=target_device, 
        target_size=(224, 224)
    )
    
    assert result.shape == (2, 3, 224, 224)
    assert result.dtype == torch.float32


def test_process_facial_frames_raises_for_invalid_dimensions(target_device: torch.device) -> None:
    """Tests architectural safety locks against incorrect spatial matrix shapes."""
    invalid_3d_array: np.ndarray = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    invalid_channels: np.ndarray = np.random.randint(0, 255, (4, 480, 640, 4), dtype=np.uint8)
    
    with pytest.raises(ValueError):
        process_facial_frames(invalid_3d_array, device=target_device)
        
    with pytest.raises(ValueError):
        process_facial_frames(invalid_channels, device=target_device)


def test_process_facial_frames_raises_for_invalid_type(target_device: torch.device) -> None:
    """Tests architectural safety locks against non-numpy data types."""
    with pytest.raises(TypeError):
        process_facial_frames("not_an_array", device=target_device)