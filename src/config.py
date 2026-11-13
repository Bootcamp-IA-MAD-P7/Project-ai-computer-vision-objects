"""Project-wide constants for AeroVision AI."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = PROJECT_ROOT / "models" / "best.pt"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs"
DEFAULT_CONF = 0.25
DEFAULT_IMGSZ = 640