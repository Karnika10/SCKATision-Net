from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from tqdm import tqdm

from .metrics import compute_metrics
from .utils import AverageMeter, save_json


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None, grad_clip_norm: float = 1.0):
    model.train()
    losses = AverageMeter()
    correct = 0
    total = 0
    pbar = tqdm(loader, desc="train", leave=False)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        use_amp = scaler is not None and device.type == "cuda"
        with torch.autocast(device_type="cuda", enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)
        if scaler is not None and device.type == "cuda":
            scaler.scale(loss).backward()
            if grad_clip_norm:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
        bs = images.size(0)
        losses.update(loss.item(), bs)
        correct += (logits.argmax(1) == targets).sum().item()
        total += bs
        pbar.set_postfix(loss=f"{losses.avg:.4f}", acc=f"{correct/max(total,1):.4f}")
    return {"loss": losses.avg, "accuracy": correct / max(total, 1)}


@torch.no_grad()
def evaluate(model, loader, criterion, device, class_names: list[str] | None = None):
    model.eval()
    losses = AverageMeter()
    y_true, y_pred = [], []
    for images, targets in tqdm(loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        losses.update(loss.item(), images.size(0))
        y_true.extend(targets.cpu().tolist())
        y_pred.extend(logits.argmax(1).cpu().tolist())
    metrics = compute_metrics(y_true, y_pred, class_names)
    metrics["loss"] = losses.avg
    return metrics


def save_checkpoint(path: str | Path, model, optimizer, scheduler, epoch: int, best_metric: float, class_names: list[str], config: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer else None,
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "best_metric": best_metric,
        "class_names": class_names,
        "config": config,
    }, path)


def load_checkpoint(path: str | Path, model, map_location="cpu"):
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state"])
    return checkpoint
