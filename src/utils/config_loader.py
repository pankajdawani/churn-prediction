"""
Configuration loader for the churn prediction project.
Reads YAML config files and returns them as plain Python dicts.
"""
import yaml
from pathlib import Path
from typing import Any, Dict

from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_CONFIG_PATH = "configs/config.yaml"


def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """
    Load a YAML configuration file and return it as a dictionary.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Dictionary containing all configuration values.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file is not valid YAML.
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found at '{config_path}'. "
            "Make sure 'configs/config.yaml' exists in your project root."
        )

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Config loaded from {config_path}")
    return config
