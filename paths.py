from pathlib import Path

# This resolves to the root of your project, no matter where it's moved
PROJECT_ROOT = Path(__file__).resolve().parent

# Subdirectories or files relative to project root
DATA_PATH = PROJECT_ROOT / "mall_dataset"
CONFIG_PATH = PROJECT_ROOT / "config" / "models_parameters.yaml"
FRAMES_PATH = DATA_PATH / "frames"