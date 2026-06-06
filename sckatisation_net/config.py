from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import yaml


@dataclass
class DataConfig:
    data_dir: str
    image_size: int = 224
    batch_size: int = 16
    num_workers: int = 4
    val_split: float = 0.2
    test_split: float = 0.0
    seed: int = 42
    augment_lisc: bool = False


@dataclass
class ModelConfig:
    num_classes: int = 5
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    embed_dim: int = 128
    depth: int = 4
    num_heads: int = 4
    kan_hidden_dim: int = 256
    dropout: float = 0.1
    rbf_grid_size: int = 8


@dataclass
class TrainConfig:
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine_restarts"
    mixed_precision: bool = True
    output_dir: str = "runs/sckatisation_net"
    early_stopping_patience: int = 20
    grad_clip_norm: float = 1.0


@dataclass
class ExperimentConfig:
    data: DataConfig
    model: ModelConfig
    train: TrainConfig


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        raw = yaml.safe_load(text)
    else:
        raw = json.loads(text)
    return ExperimentConfig(
        data=DataConfig(**raw["data"]),
        model=ModelConfig(**raw["model"]),
        train=TrainConfig(**raw["train"]),
    )


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(asdict(config), sort_keys=False), encoding="utf-8")
