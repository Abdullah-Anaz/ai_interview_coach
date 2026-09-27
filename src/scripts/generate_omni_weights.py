import logging
from pathlib import Path

import torch

from src.domains.vision.skeletal_kinematics.adapters.human_omni import HumanOmni

logger: logging.Logger = logging.getLogger(__name__)


def generate_initial_omni_weights(output_path: str) -> None:
    """
    Initializes the HumanOmni model and saves its untuned state dictionary to disk.

    Args:
        output_path (str): The file system path where the weights file will be saved.
    """
    try:
        target_path: Path = Path(output_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        model: HumanOmni = HumanOmni()
        torch.save(model.state_dict(), target_path)
        
        logger.info("Successfully generated initialized HumanOmni weights at %s", target_path)
        
    except Exception as exc:
        logger.error("Failed to generate HumanOmni weights: %s", exc)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    generate_initial_omni_weights(
        output_path="src/models/weights/human_omni.pth"
    )