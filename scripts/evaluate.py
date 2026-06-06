#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch import nn

from sckatisation_net.config import ModelConfig
from sckatisation_net.data import make_split_loaders, build_transforms, SimpleImageFolder
from sckatisation_net.engine import evaluate, load_checkpoint
from sckatisation_net.model import SCKATisionNet
from sckatisation_net.utils import get_device, save_json
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained SCKATision-Net checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True, help="ImageFolder test dir or root with test/ when --split-folders is used")
    parser.add_argument("--split-folders", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", default="evaluation_metrics.json")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = get_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model_cfg = ModelConfig(**ckpt["config"]["model"])
    model = SCKATisionNet(model_cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)

    if args.split_folders:
        _, _, test_loader, class_names = make_split_loaders(args.data_dir, model_cfg.image_size, args.batch_size, args.num_workers)
        if test_loader is None:
            raise ValueError("No test folder found")
    else:
        ds = SimpleImageFolder(args.data_dir, transform=build_transforms(model_cfg.image_size, False, False))
        test_loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        class_names = ds.classes

    metrics = evaluate(model, test_loader, nn.CrossEntropyLoss(), device, class_names)
    save_json(metrics, args.output)
    print(metrics)


if __name__ == "__main__":
    main()
