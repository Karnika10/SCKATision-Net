#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch import nn

from sckatisation_net.config import load_config
from sckatisation_net.data import make_imagefolder_loaders, make_split_loaders
from sckatisation_net.engine import train_one_epoch, evaluate, save_checkpoint
from sckatisation_net.model import SCKATisionNet
from sckatisation_net.utils import ensure_dir, get_device, save_json, set_seed


def build_optimizer(name: str, model, lr: float, weight_decay: float):
    name = name.lower()
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    raise ValueError(f"Unsupported optimizer: {name}")


def main():
    parser = argparse.ArgumentParser(description="Train SCKATision-Net for WBC classification")
    parser.add_argument("--config", required=True, help="YAML/JSON config path")
    parser.add_argument("--split-folders", action="store_true", help="Use data_dir/train, data_dir/val, data_dir/test layout")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.data.seed)
    device = get_device(args.device)
    out_dir = ensure_dir(cfg.train.output_dir)

    if args.split_folders:
        train_loader, val_loader, _, class_names = make_split_loaders(
            cfg.data.data_dir, cfg.data.image_size, cfg.data.batch_size, cfg.data.num_workers, cfg.data.augment_lisc
        )
        if val_loader is None:
            raise ValueError("--split-folders requires a validation folder at data_dir/val")
    else:
        train_loader, val_loader, class_names = make_imagefolder_loaders(
            cfg.data.data_dir, cfg.data.image_size, cfg.data.batch_size, cfg.data.num_workers,
            cfg.data.val_split, cfg.data.seed, cfg.data.augment_lisc
        )

    cfg.model.num_classes = len(class_names)
    model = SCKATisionNet(cfg.model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(cfg.train.optimizer, model, cfg.train.learning_rate, cfg.train.weight_decay)
    scheduler = None
    if cfg.train.scheduler == "cosine_restarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.train.mixed_precision and device.type == "cuda")

    best_acc = 0.0
    patience = 0
    history = []
    config_dict = asdict(cfg)
    save_json({"class_names": class_names, "config": config_dict}, out_dir / "run_metadata.json")

    for epoch in range(1, cfg.train.epochs + 1):
        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, cfg.train.grad_clip_norm)
        val_stats = evaluate(model, val_loader, criterion, device, class_names)
        if scheduler:
            scheduler.step(epoch)
        row = {"epoch": epoch, "train": train_stats, "val": val_stats}
        history.append(row)
        save_json({"history": history}, out_dir / "history.json")
        print(f"Epoch {epoch:03d}: train_acc={train_stats['accuracy']:.4f} val_acc={val_stats['accuracy']:.4f} val_loss={val_stats['loss']:.4f}")
        if val_stats["accuracy"] > best_acc:
            best_acc = val_stats["accuracy"]
            patience = 0
            save_checkpoint(out_dir / "best.pt", model, optimizer, scheduler, epoch, best_acc, class_names, config_dict)
        else:
            patience += 1
        save_checkpoint(out_dir / "last.pt", model, optimizer, scheduler, epoch, best_acc, class_names, config_dict)
        if patience >= cfg.train.early_stopping_patience:
            print(f"Early stopping after {patience} epochs without improvement.")
            break


if __name__ == "__main__":
    main()
