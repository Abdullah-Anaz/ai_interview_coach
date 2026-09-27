import logging
from typing import Any, Dict, List, Optional, cast

import numpy as np
import torch

from thop import profile

import src.domains.vision.facial_and_gaze.adapters.gazetr_model as GazeTR_Architecture

from src.domains.vision.adapters.base_adapter import PredictTrajectoryFn

logger = logging.getLogger(__name__)


def _load_gazetr_model(weights_path: str, device: torch.device) -> torch.nn.Module:
    """
    Handles the side-effect of loading the real GazeTR-Hybrid model weights.

    Args:
        weights_path (str): File system path to the model checkpoint.
        device (torch.device): Target hardware device.

    Returns:
        torch.nn.Module: Initialized and evaluation-ready PyTorch model.

    Raises:
        RuntimeError: Triggered if the model checkpoint is missing or corrupted.
    """
    try:
        # 1. Instantiate the real architecture
        model = GazeTR_Architecture.Model()
        
        # 2. Load the state_dict from the file you downloaded
        state_dict = torch.load(weights_path, map_location=device)
        
        # 3. Inject the weights into the architecture
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        logger.info("Successfully loaded REAL GazeTR-Hybrid weights from %s", weights_path)
        return model
        
    except Exception as exc:
        logger.error("Failed to load GazeTR-Hybrid model: %s", exc)
        raise RuntimeError(f"GazeTR-Hybrid initialization failure: {exc}")


def create_gazetr_adapter(config: Dict[str, Any], device: torch.device) -> PredictTrajectoryFn:
    """
    Factory function adhering to the base_adapter functional interface.
    Initializes the GazeTR model and returns a pure prediction closure.

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
        raise ValueError("GazeTR-Hybrid adapter requires a valid 'weights_path' in the config.")

    model: torch.nn.Module = _load_gazetr_model(weights_path, device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    config["parameter_count_millions"] = round(total_params / 1_000_000.0, 2)

    # 2. Calculate GFLOPs
    try:
        # GazeTR expects a dictionary input with the key "face"
        dummy_input = {"face": torch.randn(1, 3, 224, 224, device=device)}
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
        Executes frame-level inference to extract 3D gaze vectors using micro-batching.

        Args:
            sequence_tensor (torch.Tensor): A batch sequence of video frames (T, C, H, W).
            fps (float): Native frame rate of the sequence (mandated by interface).

        Returns:
            Optional[np.ndarray]: Mean 3D gaze vector array, or None if execution fails.
        """
        try:
            with torch.no_grad():
                seq_device: torch.Tensor = sequence_tensor.to(device, dtype=torch.float32)
                num_frames: int = seq_device.size(0)
                all_gaze_vectors: List[torch.Tensor] = []

                # Micro-batching to prevent OOM errors on large video sequences
                all_latents: List[torch.Tensor] = []
                
                for i in range(0, num_frames, batch_size):
                    batch_frames: torch.Tensor = seq_device[i : i + batch_size]
                    
                    batch_input = {"face": batch_frames}
                    # Bypass the classification head to extract embeddings
                    batch_latents: torch.Tensor = model.extract_latents(batch_input)
                    
                    all_latents.append(batch_latents)

                # Aggregate temporal latent embeddings
                gaze_trajectory: torch.Tensor = torch.cat(all_latents, dim=0)
                mean_latent_vector: torch.Tensor = gaze_trajectory.mean(dim=0)
                
                return cast(np.ndarray, mean_latent_vector.cpu().numpy())
                
        except (RuntimeError, ValueError) as exc:
            logger.error("Tensor shape mismatch during GazeTR sequence prediction: %s", exc)
            return None
        except Exception as exc:
            logger.error("Unexpected execution failure during GazeTR prediction: %s", exc)
            return None

    return predict