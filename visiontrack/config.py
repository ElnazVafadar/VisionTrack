from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    model: str = "yolo11n-seg.pt"
    tracker: str = "bytetrack.yaml"
    confidence: float = 0.35
    iou: float = 0.5
    device: str = ""
    trail_length: int = 30
    line_position: float = 0.5
    output_dir: Path = Path("outputs")
    save_video: bool = True
    show: bool = True
