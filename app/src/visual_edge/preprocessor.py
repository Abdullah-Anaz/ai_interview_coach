import os
import urllib.request
import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F
from typing import Tuple


def process_skeletal_frames(
    frames: np.ndarray, 
    device: torch.device = torch.device("cuda:0")
) -> torch.Tensor:
    """
    Transforms raw video frames for the HumanOmni skeletal pipeline via dimension permutation,
    scaling, and ImageNet statistical normalization.

    Args:
        frames (np.ndarray): 4D array of video frames structured as (Time, Height, Width, Channels).
        device (torch.device): Target compute device. Defaults to cuda:0.

    Returns:
        torch.Tensor: Normalized float32 tensor structured as (Batch, Channels, Height, Width).

    Raises:
        TypeError: If frames is not a numpy array.
        ValueError: If frames does not have exactly 4 dimensions or 3 color channels.
    """
    if not isinstance(frames, np.ndarray):
        raise TypeError("frames must be a numpy.ndarray.")
    if frames.ndim != 4:
        raise ValueError(f"frames must have exactly 4 dimensions, got {frames.ndim}.")
    if frames.shape[-1] != 3:
        raise ValueError(f"frames must have exactly 3 color channels, got {frames.shape[-1]}.")

    tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float().to(device)
    tensor = tensor / 255.0
    tensor = F.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    return tensor


def _ensure_haar_cascade() -> str:
    """
    Resolves the absolute path to the OpenCV Haar Cascade, dynamically downloading 
    it to the current working directory if missing from the local site-packages.
    """
    cascade_name = "haarcascade_frontalface_default.xml"
    default_path = os.path.join(cv2.data.haarcascades, cascade_name)
    
    if os.path.exists(default_path):
        return default_path
        
    fallback_path = os.path.join(os.getcwd(), cascade_name)
    if not os.path.exists(fallback_path):
        url = f"https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/{cascade_name}"
        urllib.request.urlretrieve(url, fallback_path)
        
    return fallback_path


def process_facial_frames(
    frames: np.ndarray, 
    device: torch.device = torch.device("cuda:0"),
    target_size: Tuple[int, int] = (224, 224)
) -> torch.Tensor:
    """
    Isolates facial geometry using OpenCV Haar Cascades, resizes to a target dimension, 
    and normalizes the output for the POSTER++ facial gaze pipeline.

    Args:
        frames (np.ndarray): 4D array of video frames structured as (Time, Height, Width, Channels).
        device (torch.device): Target compute device. Defaults to cuda:0.
        target_size (Tuple[int, int]): Spatial bounds for the ResNet/MobileNet encoders.

    Returns:
        torch.Tensor: Normalized float32 tensor structured as (Batch, Channels, Height, Width).

    Raises:
        TypeError: If frames is not a numpy array.
        ValueError: If frames does not have exactly 4 dimensions or 3 color channels.
        RuntimeError: If the OpenCV Haar Cascade XML fails to load or download.
    """
    if not isinstance(frames, np.ndarray):
        raise TypeError("frames must be a numpy.ndarray.")
    if frames.ndim != 4:
        raise ValueError(f"frames must have exactly 4 dimensions, got {frames.ndim}.")
    if frames.shape[-1] != 3:
        raise ValueError(f"frames must have exactly 3 color channels, got {frames.shape[-1]}.")

    cascade_path: str = _ensure_haar_cascade()
    face_cascade: cv2.CascadeClassifier = cv2.CascadeClassifier(cascade_path)
    
    if face_cascade.empty():
        raise RuntimeError(f"Failed to load OpenCV Haar cascade from {cascade_path}")

    cropped_frames = []

    for frame in frames:
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
            
        gray_frame: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        faces: tuple = face_cascade.detectMultiScale(
            gray_frame, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        if len(faces) > 0:
            faces_sorted = sorted(faces, key=lambda rect: rect[2] * rect[3], reverse=True)
            x, y, w, h = faces_sorted[0]
            
            face_crop: np.ndarray = frame[y:y+h, x:x+w]
            
            if face_crop.size == 0:
                face_crop = frame 
        else:
            face_crop = frame 
            
        face_resized: np.ndarray = cv2.resize(face_crop, target_size)
        cropped_frames.append(face_resized)
        
    cropped_array: np.ndarray = np.stack(cropped_frames)
    
    tensor: torch.Tensor = torch.from_numpy(cropped_array).permute(0, 3, 1, 2).float().to(device)
    tensor = tensor / 255.0
    tensor = F.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    return tensor