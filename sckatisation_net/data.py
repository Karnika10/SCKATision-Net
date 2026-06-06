from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset, random_split


WBC_CLASSES = ["basophil", "eosinophil", "lymphocyte", "monocyte", "neutrophil"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class SimpleImageFolder(Dataset):
    """Minimal torchvision-free ImageFolder replacement.

    Expected directory layout:
        root/class_name/image.png
    """

    def __init__(self, root: str | Path, transform: Callable | None = None):
        self.root = Path(root)
        self.transform = transform
        if not self.root.exists():
            raise FileNotFoundError(f"Image folder not found: {self.root}")

        self.classes = sorted([p.name for p in self.root.iterdir() if p.is_dir()])
        if not self.classes:
            raise ValueError(f"No class subdirectories found under {self.root}")
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}

        self.samples: list[tuple[Path, int]] = []
        for class_name in self.classes:
            class_dir = self.root / class_name
            for path in sorted(class_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((path, self.class_to_idx[class_name]))
        if not self.samples:
            raise ValueError(f"No image files found under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class WBCTransform:
    """PIL-based transforms that keep normalization in [0, 1], as in the manuscript."""

    def __init__(self, image_size: int = 224, train: bool = True, augment: bool = False):
        self.image_size = int(image_size)
        self.train = train
        self.augment = augment

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        if self.train and self.augment:
            if random.random() < 0.5:
                image = ImageOps.mirror(image)
            if random.random() < 0.5:
                image = ImageOps.flip(image)
            angle = random.uniform(-25, 25)
            translate = (
                random.uniform(-0.08, 0.08) * self.image_size,
                random.uniform(-0.08, 0.08) * self.image_size,
            )
            shear = random.uniform(-8, 8)
            # PIL affine matrix maps output coordinates to input coordinates.
            image = image.rotate(angle, resample=Image.Resampling.BILINEAR)
            image = image.transform(
                image.size,
                Image.Transform.AFFINE,
                (1, np.tan(np.deg2rad(shear)), -translate[0], 0, 1, -translate[1]),
                resample=Image.Resampling.BILINEAR,
            )
        arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr).contiguous()


def build_transforms(image_size: int = 224, train: bool = True, augment: bool = False):
    return WBCTransform(image_size=image_size, train=train, augment=augment)


def make_imagefolder_loaders(
    data_dir: str | Path,
    image_size: int = 224,
    batch_size: int = 16,
    num_workers: int = 4,
    val_split: float = 0.2,
    seed: int = 42,
    augment_train: bool = False,
) -> Tuple[DataLoader, DataLoader, list[str]]:
    """Create train/val dataloaders from a class-folder directory."""
    data_dir = Path(data_dir)
    full_train = SimpleImageFolder(data_dir, transform=build_transforms(image_size, True, augment_train))
    full_val = SimpleImageFolder(data_dir, transform=build_transforms(image_size, False, False))
    n_total = len(full_train)
    n_val = int(round(n_total * val_split))
    n_train = n_total - n_val
    if n_train <= 0 or n_val <= 0:
        raise ValueError("val_split produced an empty train or validation set")
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(range(n_total), [n_train, n_val], generator=generator)
    train_ds = torch.utils.data.Subset(full_train, list(train_subset))
    val_ds = torch.utils.data.Subset(full_val, list(val_subset))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, full_train.classes


def make_split_loaders(
    root_dir: str | Path,
    image_size: int = 224,
    batch_size: int = 16,
    num_workers: int = 4,
    augment_train: bool = False,
) -> Tuple[DataLoader, DataLoader | None, DataLoader | None, list[str]]:
    """Create loaders from train/val/test class-folder splits if present."""
    root_dir = Path(root_dir)
    train_dir = root_dir / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"Expected split directory: {train_dir}")
    train_ds = SimpleImageFolder(train_dir, transform=build_transforms(image_size, True, augment_train))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    def optional_loader(split: str):
        split_dir = root_dir / split
        if not split_dir.exists():
            return None
        ds = SimpleImageFolder(split_dir, transform=build_transforms(image_size, False, False))
        return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, optional_loader("val"), optional_loader("test"), train_ds.classes
