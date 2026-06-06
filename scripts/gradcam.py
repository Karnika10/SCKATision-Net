#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from sckatisation_net.config import ModelConfig
from sckatisation_net.data import build_transforms
from sckatisation_net.engine import load_checkpoint
from sckatisation_net.model import SCKATisionNet
from sckatisation_net.utils import get_device


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x, class_idx=None):
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1)[0])
        self.model.zero_grad(set_to_none=True)
        logits[:, class_idx].sum().backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.cpu().numpy(), class_idx


def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM heatmap for SCKATision-Net")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="gradcam.png")
    parser.add_argument("--class-idx", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = get_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ModelConfig(**ckpt["config"]["model"])
    model = SCKATisionNet(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    image = Image.open(args.image).convert("RGB")
    transform = build_transforms(cfg.image_size, False, False)
    x = transform(image).unsqueeze(0).to(device)
    cam, class_idx = GradCAM(model, model.scconv.out[-1])(x, args.class_idx)

    img = image.resize((cfg.image_size, cfg.image_size))
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.imshow(cam, alpha=0.45, cmap="jet")
    plt.axis("off")
    plt.title(f"Grad-CAM class index: {class_idx}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, bbox_inches="tight", dpi=200)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
