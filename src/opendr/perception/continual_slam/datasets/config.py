import dataclasses
from pathlib import Path
from typing import Optional, Tuple


@dataclasses.dataclass
class Dataset:
    dataset: str
    config_file: Path
    dataset_path: Optional[Path]
    scales: Optional[Tuple[int, ...]]
    height: Optional[int]
    width: Optional[int]
    frame_ids: Tuple[int, ...]
