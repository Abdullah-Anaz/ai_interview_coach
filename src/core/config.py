import yaml
from pathlib import Path
from typing import Dict, Any


def load_yaml_config(file_path: Path) -> Dict[str, Any]:
    """
    Loads and parses a YAML configuration file into a dictionary.

    Args:
        file_path (Path): The absolute or relative path to the YAML configuration file.

    Returns:
        Dict[str, Any]: A dictionary containing the parsed configuration parameters.

    Raises:
        FileNotFoundError: If the specified YAML file does not exist at the given path.
        yaml.YAMLError: If the YAML file is malformed and cannot be safely parsed.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as file:
        config_data = yaml.safe_load(file)
        
    return config_data