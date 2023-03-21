import dataclasses
from pathlib import Path


@dataclasses.dataclass
class DatasetConfig:
    dataset: str
    dataset_path: Path
    height: int
    width: int
