import argparse
import logging
from pathlib import Path
from typing import Any, Dict

from src.core.config import load_yaml_config
from src.core.device import get_compute_device
from src.core.registry import get_runner

import src.domains.transcription.transcription_exp
import src.domains.acoustic.acoustic_exp
import src.domains.vision.skeletal_kinematics.vision_skeletal_kinamatics_exp
import src.domains.vision.facial_and_gaze.facial_and_gaze_exp


def configure_logging(log_level: int = logging.INFO) -> None:
    """
    Configures the root logging format and level across all execution modules.

    Args:
        log_level (int): Logging severity level (e.g., logging.INFO).
    """
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments for the experimentation framework.

    Returns:
        argparse.Namespace: The parsed command-line arguments containing the config path.
    """
    parser = argparse.ArgumentParser(
        description="Run a specific domain experiment based on a YAML configuration."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML configuration file defining the experiment parameters.",
    )
    return parser.parse_args()


def execute_experiment(config: Dict[str, Any]) -> None:
    """
    Resolves the execution context and triggers the designated domain runner.

    Args:
        config (Dict[str, Any]): The parsed configuration dictionary containing 
        at minimum the 'domain' key.

    Raises:
        ValueError: If the 'domain' key is missing from the configuration file.
    """
    logger = logging.getLogger(__name__)
    domain_name: Any = config.get("domain")
    if not domain_name:
        logger.critical("Configuration file missing required 'domain' key")
        raise ValueError("Configuration file must specify a 'domain' key.")

    device = get_compute_device()
    logger.info("Allocated compute device: %s", device)

    runner_function = get_runner(str(domain_name))
    runner_function(config, device)


def main() -> None:
    """
    Entry point for the execution script. Configures logging, parses arguments, 
    loads configuration, and dispatches the domain runner.
    """
    configure_logging()
    args = parse_arguments()
    config_data: Dict[str, Any] = load_yaml_config(args.config)
    execute_experiment(config_data)


if __name__ == "__main__":
    main()