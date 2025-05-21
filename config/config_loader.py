import yaml
from paths import CONFIG_PATH
from pathlib import Path



def load_model_config(model_name: str, path: Path = CONFIG_PATH) -> dict:
    """
    Load model-specific configuration from YAML.

    Args:
        model_name (str): The name of the model (e.g., 'vgg16', 'resnet50').
        path (Path): Path to the YAML config file.

    Returns:
        dict: Configuration for the given model, or empty dict if not found.
    """
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config.get(model_name, {})
