import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)


def create_face_detector(device: torch.device, margin: int = 20) -> MTCNN:
    """Initializes and returns an MTCNN face detection model instance."""
    return MTCNN(keep_all=False, device=device, margin=margin, post_process=False)


def create_vision_transform(target_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """Creates a standard image resize and tensor conversion pipeline."""
    return transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ])


def extract_raw_frames(video_path: str) -> Tuple[List[np.ndarray], float]:
    """
    Extracts chronological RGB frames and frame rate from a video file.

    Args:
        video_path (str): File system path to the target video.

    Returns:
        Tuple[List[np.ndarray], float]: Chronological list of RGB numpy arrays and native FPS.
    """
    path_obj: Path = Path(video_path)
    if not path_obj.exists():
        raise IOError(f"Video file not found: {video_path}")

    capture: cv2.VideoCapture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise IOError(f"Cannot open video stream: {video_path}")

    fps: float = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0.0 or np.isnan(fps):
        logger.warning("Unreliable FPS in %s. Falling back to 30.0 FPS.", video_path)
        fps = 30.0

    frames: List[np.ndarray] = []
    while True:
        success: bool
        frame: np.ndarray
        success, frame = capture.read()
        if not success:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    capture.release()
    return frames, fps


def crop_face_sequence(
    frames: List[np.ndarray],
    detector: MTCNN,
    transform: transforms.Compose
) -> Optional[torch.Tensor]:
    """
    Detects, crops, and stacks facial crops with temporal fallback interpolation.

    Args:
        frames (List[np.ndarray]): Ordered sequence of RGB frames.
        detector (MTCNN): Initialized facial detector.
        transform (transforms.Compose): Tensor transformation pipeline.

    Returns:
        Optional[torch.Tensor]: Stacked sequence tensor (T, C, H, W) or None if no face found.
    """
    processed_tensors: List[torch.Tensor] = []
    last_valid_box: Optional[np.ndarray] = None

    frame_np: np.ndarray
    for frame_np in frames:
        pil_img: Image.Image = Image.fromarray(frame_np)
        boxes: Optional[np.ndarray]
        boxes, _ = detector.detect(pil_img)

        if boxes is not None and len(boxes) > 0:
            last_valid_box = boxes[0]
        elif last_valid_box is None:
            continue

        x1, y1, x2, y2 = [max(0, float(coord)) for coord in last_valid_box]
        cropped_img: Image.Image = pil_img.crop((x1, y1, x2, y2))
        processed_tensors.append(transform(cropped_img))

    if not processed_tensors:
        return None

    return torch.stack(processed_tensors)


def process_video_pipeline(
    video_path: str,
    detector: MTCNN,
    transform: transforms.Compose
) -> Tuple[Optional[torch.Tensor], float]:
    """
    Composes extraction and facial cropping into an end-to-end functional pipeline.

    Args:
        video_path (str): Path to input video file.
        detector (MTCNN): Face detector instance.
        transform (transforms.Compose): Tensor transform pipeline.

    Returns:
        Tuple[Optional[torch.Tensor], float]: Processed tensor sequence and FPS.
    """
    try:
        raw_frames: List[np.ndarray]
        fps: float
        raw_frames, fps = extract_raw_frames(video_path)

        if not raw_frames:
            return None, 0.0

        tensor_sequence: Optional[torch.Tensor] = crop_face_sequence(
            frames=raw_frames,
            detector=detector,
            transform=transform
        )

        return tensor_sequence, fps

    except Exception as exc:
        logger.error("Pipeline failed for video %s: %s", video_path, exc)
        return None, 0.0