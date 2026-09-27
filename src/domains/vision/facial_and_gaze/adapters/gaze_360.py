import logging
from typing import Any, Dict, List, Optional, cast

import numpy as np
import torch

from src.domains.vision.adapters.base_adapter import PredictTrajectoryFn

logger = logging.getLogger(__name__)


def _load_gaze360_model(weights_path: str, device: torch.device) -> torch.nn.Module:
    """
    Handles the side-effect of loading the Gaze360 PyTorch model weights.

    Args:
        weights_path (str): File system path to the model checkpoint.
        device (torch.device): Target hardware device.

    Returns:
        torch.nn.Module: Initialized and evaluation-ready PyTorch model.

    Raises:
        RuntimeError: Triggered if the model checkpoint is missing or corrupted.
    """
    try:
        model: torch.nn.Module = torch.load(weights_path, map_location=device)
        model.eval()
        logger.info("Successfully loaded Gaze360 model from %s", weights_path)
        return model
    except Exception as exc:
        logger.error("Failed to load Gaze360 model from %s: %s", weights_path, exc)
        raise RuntimeError(f"Gaze360 model initialization failure: {exc}")


def _create_temporal_windows(sequence_tensor: torch.Tensor, window_size: int = 7) -> torch.Tensor:
    """
    Transforms a continuous sequence of frames into overlapping temporal windows 
    required by Gaze360's bidirectional LSTM backbone.

    Args:
        sequence_tensor (torch.Tensor): Continuous tensor of shape (T, C, H, W).
        window_size (int): Size of the temporal window. Must be an odd integer. Defaults to 7.

    Returns:
        torch.Tensor: Windowed tensor of shape (T, window_size, C, H, W).
    """
    num_frames, channels, height, width = sequence_tensor.shape
    pad_size: int = window_size // 2

    start_padding: torch.Tensor = sequence_tensor[0:1].expand(pad_size, channels, height, width)
    end_padding: torch.Tensor = sequence_tensor[-1:].expand(pad_size, channels, height, width)
    
    padded_sequence: torch.Tensor = torch.cat([start_padding, sequence_tensor, end_padding], dim=0)
    
    windows: List[torch.Tensor] = [
        padded_sequence[i : i + window_size] for i in range(num_frames)
    ]
    
    return torch.stack(windows)


def create_gaze360_adapter(config: Dict[str, Any], device: torch.device) -> PredictTrajectoryFn:
    """
    Factory function adhering to the base_adapter functional interface.
    Initializes the Gaze360 model and returns a pure prediction closure.

    Args:
        config (Dict[str, Any]): Dictionary containing 'weights_path' and 'batch_size'.
        device (torch.device): Hardware target for tensor execution.

    Returns:
        PredictTrajectoryFn: A pure closure accepting a tensor sequence and FPS.

    Raises:
        ValueError: Triggered if 'weights_path' is omitted from the configuration.
    """
    weights_path: str = config.get("weights_path", "")
    batch_size: int = config.get("batch_size", 32)
    
    if not weights_path:
        raise ValueError("Gaze360 adapter requires a valid 'weights_path' in the config.")

    model: torch.nn.Module = _load_gaze360_model(weights_path, device)

    def predict(sequence_tensor: torch.Tensor, fps: float) -> Optional[np.ndarray]:
        """
        Executes sequence-level temporal inference to extract 3D gaze vectors.

        Args:
            sequence_tensor (torch.Tensor): Continuous frame sequence (T, C, H, W).
            fps (float): Native frame rate of the sequence.

        Returns:
            Optional[np.ndarray]: Mean 3D gaze vector array, or None if execution fails.
        """
        try:
            with torch.no_grad():
                seq_device: torch.Tensor = sequence_tensor.to(device, dtype=torch.float32)
                
                windowed_tensor: torch.Tensor = _create_temporal_windows(seq_device, window_size=7)
                num_windows: int = windowed_tensor.size(0)
                
                all_gaze_vectors: List[torch.Tensor] = []

                for i in range(0, num_windows, batch_size):
                    batch_windows: torch.Tensor = windowed_tensor[i : i + batch_size]
                    batch_predictions: torch.Tensor = model(batch_windows)
                    all_gaze_vectors.append(batch_predictions)

                gaze_trajectory: torch.Tensor = torch.cat(all_gaze_vectors, dim=0)
                mean_gaze_vector: torch.Tensor = gaze_trajectory.mean(dim=0)
                
                return cast(np.ndarray, mean_gaze_vector.cpu().numpy())
                
        except (RuntimeError, ValueError) as exc:
            logger.error("Tensor shape mismatch during Gaze360 sequence prediction: %s", exc)
            return None
        except Exception as exc:
            logger.error("Unexpected execution failure during Gaze360 prediction: %s", exc)
            return None

    return predict