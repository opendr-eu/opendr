import dataclasses
from pathlib import Path
from typing import Optional, Tuple, Union


@dataclasses.dataclass
class Config:
    config_file: Path
    train_set: Optional[Union[Tuple[int, ...], int, str]]
    val_set: Optional[Union[Tuple[int, ...], Tuple[str, ...], int, str]]
    resnet: int
    resnet_pretrained: bool
    scales: Tuple[int, ...]
    learning_rate: float
    scheduler_step_size: int
    batch_size: int
    num_workers: int
    num_epochs: int
    min_depth: Optional[float]
    max_depth: Optional[float]
    disparity_smoothness: float
    velocity_loss_scaling: Optional[float]
    mask_dynamic: bool
    log_path: Path
    save_frequency: int
    save_val_depth: bool
    save_val_depth_batches: int
    multiple_gpus: bool
    gpu_ids: Optional[Tuple[int, ...]]
    load_weights_folder: Optional[Path]
    use_wandb: Optional[bool]
