import logging
from pathlib import Path

import torch

from src.domains.vision.facial_and_gaze.adapters.poster_plus import POSTERPlus

logger: logging.Logger = logging.getLogger(__name__)


def generate_initial_poster_weights(output_path: str) -> None:
    """
    Initializes the POSTER++ model and saves its untuned latent state dictionary to disk.

    Args:
        output_path (str): The file system path where the weights file will be saved.
    """
    try:
        target_path: Path = Path(output_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        model: POSTERPlus = POSTERPlus(feature_dim=1024)
        torch.save(model.state_dict(), target_path)
        
        logger.info("Successfully generated initialized POSTER++ latent weights at %s", target_path)
        
    except Exception as exc:
        logger.error("Failed to generate POSTER++ weights: %s", exc)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    generate_initial_poster_weights(
        output_path="src/models/weights/poster_plus_latent.pth"
    )