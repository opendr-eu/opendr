import dataclasses
from pathlib import Path


@dataclasses.dataclass
class LoopClosureDetection:
    config_file: Path
    detection_threshold: float
    id_threshold: int
    num_matches: int
