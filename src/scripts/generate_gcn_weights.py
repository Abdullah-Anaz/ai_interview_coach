import logging
from pathlib import Path

import torch

from src.domains.vision.skeletal_kinematics.adapters.mediapipe_gcn import PlanarLiftingGCN

logger = logging.getLogger(__name__)


def generate_initial_gcn_weights(output_path: str) -> None:
    """
    Initializes the PlanarLiftingGCN model and saves its untuned state dictionary to disk.

    Args:
        output_path (str): The file system path where the weights file will be saved.
    """
    try:
        target_path: Path = Path(output_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        model: PlanarLiftingGCN = PlanarLiftingGCN()
        torch.save(model.state_dict(), target_path)
        
        logger.info("Successfully generated initialized GCN weights at %s", target_path)
        
    except Exception as exc:
        logger.error("Failed to generate GCN weights: %s", exc)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    generate_initial_gcn_weights(
        output_path="src/models/weights/planar_lifting_gcn.pt"
    )