#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from PIL import Image

from sckatisation_net.config import ModelConfig
from sckatisation_net.data import build_transforms
from sckatisation_net.engine import load_checkpoint
from sckatisation_net.model import SCKATisionNet
from sckatisation_net.utils import get_device


def main():
    parser = argparse.ArgumentParser(description="Predict WBC class for one image")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = get_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ModelConfig(**ckpt["config"]["model"])
    class_names = ckpt.get("class_names", [str(i) for i in range(cfg.num_classes)])
    model = SCKATisionNet(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    image = Image.open(args.image).convert("RGB")
    x = build_transforms(cfg.image_size, False, False)(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu()
    idx = int(probs.argmax())
    print({"class": class_names[idx], "confidence": float(probs[idx]), "probabilities": {c: float(p) for c, p in zip(class_names, probs)}})


if __name__ == "__main__":
    main()
