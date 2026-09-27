import logging
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, cast

import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

logger: logging.Logger = logging.getLogger(__name__)

PredictTrajectoryFn = Callable[[torch.Tensor, float], Optional[np.ndarray]]


class SpatialGCNLayer(nn.Module):
    """
    Executes a spatial graph convolutional operation over input node features.

    Args:
        in_features (int): The number of input features per node.
        out_features (int): The number of output features per node.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight: nn.Parameter = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Applies the layer transformation to the input tensor.

        Args:
            x (torch.Tensor): Input feature tensor.
            adj (torch.Tensor): Adjacency matrix tensor.

        Returns:
            torch.Tensor: Transformed feature tensor.
        """
        support: torch.Tensor = torch.matmul(x, self.weight)
        output: torch.Tensor = torch.matmul(adj, support)
        return output


class PlanarLiftingGCN(nn.Module):
    """
    Graph Convolutional Network designed to refine 3D spatial coordinates.

    Args:
        num_nodes (int, optional): Total number of nodes in the graph. Defaults to 478.
        in_dim (int, optional): Input feature dimensionality. Defaults to 3.
        hidden_dim (int, optional): Hidden layer dimensionality. Defaults to 64.
    """

    def __init__(self, num_nodes: int = 478, in_dim: int = 3, hidden_dim: int = 64) -> None:
        super().__init__()
        self.adj: nn.Parameter = nn.Parameter(torch.empty(num_nodes, num_nodes))
        nn.init.eye_(self.adj)
        
        self.gcn1: SpatialGCNLayer = SpatialGCNLayer(in_features=in_dim, out_features=hidden_dim)
        self.gcn2: SpatialGCNLayer = SpatialGCNLayer(in_features=hidden_dim, out_features=in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executes the forward pass refining the spatial coordinates.

        Args:
            x (torch.Tensor): Input coordinate tensor.

        Returns:
            torch.Tensor: Refined coordinate tensor.
        """
        adj_norm: torch.Tensor = F.softmax(self.adj, dim=-1)
        hidden: torch.Tensor = F.relu(self.gcn1(x, adj_norm))
        refined: torch.Tensor = self.gcn2(hidden, adj_norm)
        return x + refined


def _tensor_to_numpy_image(tensor_frame: torch.Tensor) -> np.ndarray:
    """
    Converts a PyTorch image tensor to a standard NumPy RGB image format.

    Args:
        tensor_frame (torch.Tensor): The image tensor formatted as (C, H, W).

    Returns:
        np.ndarray: The image array formatted as (H, W, C) with uint8 data type.
    """
    numpy_frame: np.ndarray = tensor_frame.permute(1, 2, 0).cpu().numpy()
    return (numpy_frame * 255.0).astype(np.uint8)


def _load_gcn_model(config: Dict[str, Any], device: torch.device) -> Optional[nn.Module]:
    """
    Initializes and loads the Planar Lifting GCN weights from disk.

    Args:
        config (Dict[str, Any]): Configuration dictionary containing 'gcn_weights_path'.
        device (torch.device): The target compute device for the model.

    Returns:
        Optional[nn.Module]: The loaded GCN model, or None if loading fails or path is omitted.
    """
    gcn_path: str = config.get("gcn_weights_path", "")
    if not gcn_path:
        return None
        
    try:
        model: PlanarLiftingGCN = PlanarLiftingGCN()
        model.load_state_dict(torch.load(gcn_path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        return model
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error("Failed to load planar lifting GCN from %s: %s", gcn_path, exc)
        return None


def _ensure_task_file(task_path: str) -> None:
    """
    Verifies the existence of the MediaPipe task file and downloads it if missing.

    Args:
        task_path (str): The local file system path for the task file.
    """
    path: Path = Path(task_path)
    if not path.exists():
        logger.info("Downloading modern MediaPipe task file to %s...", task_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            url: str = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(url, task_path)
        except (urllib.error.URLError, OSError) as exc:
            logger.error("Failed to download or save MediaPipe task file: %s", exc)
            raise


def create_mediapipe_gcn_adapter(config: Dict[str, Any], device: torch.device) -> PredictTrajectoryFn:
    """
    Generates a pure functional closure for executing MediaPipe face landmarking and GCN refinement.

    Args:
        config (Dict[str, Any]): Configuration parameters for the pipeline.
        device (torch.device): The target compute device.

    Returns:
        PredictTrajectoryFn: A callable function accepting a video tensor sequence and FPS, 
                             returning the extracted trajectory array.
    """
    task_path: str = config.get("landmarker_task_path", "src/models/weights/face_landmarker.task")
    
    try:
        _ensure_task_file(task_path)
    except Exception:
        logger.critical("Cannot initialize MediaPipe adapter without valid task file.")
        raise RuntimeError("MediaPipe task file initialization failed.")

    mp_delegate: Any = (
        mp_python.BaseOptions.Delegate.GPU 
        if device.type == "cuda" 
        else mp_python.BaseOptions.Delegate.CPU
    )

    base_options: Any = mp_python.BaseOptions(
        model_asset_path=task_path,
        delegate=mp_delegate
    )
    
    options: Any = vision.FaceLandmarkerOptions( 
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    detector: Any = vision.FaceLandmarker.create_from_options(options)
    gcn_model: Optional[nn.Module] = _load_gcn_model(config, device)

    def predict(sequence_tensor: torch.Tensor, fps: float) -> Optional[np.ndarray]:
        """
        Processes a sequence of frames to extract and optionally refine facial landmark trajectories.

        Args:
            sequence_tensor (torch.Tensor): A tensor sequence of image frames.
            fps (float): The frames per second of the sequence.

        Returns:
            Optional[np.ndarray]: An array of facial landmarks across the sequence, or None if extraction fails.
        """
        try:
            t_frames: int = sequence_tensor.shape[0]
            trajectories: List[np.ndarray] = []
            
            frame_idx: int
            for frame_idx in range(t_frames):
                frame_np: np.ndarray = _tensor_to_numpy_image(sequence_tensor[frame_idx])
                
                mp_image: Any = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_np)
                detection_result: Any = detector.detect(mp_image)
                
                coords: np.ndarray
                if detection_result.face_landmarks:
                    landmarks: Any = detection_result.face_landmarks[0]
                    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=float)
                else:
                    coords = trajectories[-1] if trajectories else np.zeros((478, 3), dtype=float)
                        
                trajectories.append(coords)
                
            raw_trajectory: np.ndarray = np.stack(trajectories)

            if gcn_model is not None:
                with torch.no_grad():
                    traj_tensor: torch.Tensor = torch.tensor(
                        raw_trajectory, 
                        dtype=torch.float32, 
                        device=device
                    )
                    refined_tensor: torch.Tensor = gcn_model(traj_tensor)
                    return cast(np.ndarray, refined_tensor.cpu().numpy())

            return raw_trajectory

        except (RuntimeError, ValueError) as exc:
            logger.error("Data processing error during sequence prediction: %s", exc)
            return None
        except Exception as exc:
            logger.error("Unexpected error during sequence prediction: %s", exc)
            return None

    return predict