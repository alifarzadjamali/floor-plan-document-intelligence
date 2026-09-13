from __future__ import annotations

import csv
import json
import math
import random
import shutil
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
import yaml
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from floorplan_di.data.segmentation import AugmentationConfig, CachedSegmentationDataset
from floorplan_di.evaluation.segmentation_metrics import ConfusionMatrix
from floorplan_di.models.losses import multiclass_dice_loss
from floorplan_di.models.segformer import build_segformer


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _resize_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return functional.interpolate(
        logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
    )


def calculate_class_weights(manifest_path: Path, cap: float) -> torch.Tensor:
    counts = np.zeros(5, dtype=np.float64)
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        counts += np.asarray(json.loads(line)["pixel_counts"], dtype=np.float64)
    frequencies = counts / counts.sum()
    weights = np.median(frequencies) / np.maximum(frequencies, 1e-12)
    weights = np.minimum(weights, cap)
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    dice_weight: float,
) -> tuple[dict[str, Any], float]:
    model.eval()
    confusion = ConfusionMatrix()
    losses: list[float] = []
    with torch.inference_mode():
        for batch in loader:
            images = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            logits = _resize_logits(model(pixel_values=images).logits, labels)
            loss = criterion(logits, labels) + dice_weight * multiclass_dice_loss(logits, labels)
            losses.append(float(loss.item()))
            confusion.update(labels.cpu().numpy(), logits.argmax(dim=1).cpu().numpy())
    return confusion.compute(), float(np.mean(losses))


def _scheduler(optimizer: AdamW, warmup_steps: int, total_steps: int) -> LambdaLR:
    def factor(step: int) -> float:
        if step < warmup_steps:
            return max(1e-8, (step + 1) / max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return LambdaLR(optimizer, factor)


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: AdamW,
    epoch: int,
    config: dict[str, Any],
    class_weights: torch.Tensor,
    best_score: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "config": config,
            "class_weights": class_weights.cpu(),
            "best_score": best_score,
            "torch_version": torch.__version__,
        },
        path,
    )


def train(config_path: Path, cache_root: Path, output: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed_everything(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Phase 2 training requires the configured CUDA GPU")
    augmentation = AugmentationConfig(**config["augment"])
    train_dataset = CachedSegmentationDataset(
        cache_root / "manifests" / "train.jsonl", True, augmentation, config["seed"]
    )
    val_dataset = CachedSegmentationDataset(cache_root / "manifests" / "val.jsonl")
    loader_options = {
        "batch_size": config["batch_size"],
        "num_workers": config["num_workers"],
        "pin_memory": True,
        "persistent_workers": False,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)
    model = build_segformer(config["model_name"]).to(device)
    class_weights = calculate_class_weights(
        cache_root / "manifests" / "train.jsonl", config["class_weight_cap"]
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    total_steps = len(train_loader) * config["epochs"]
    scheduler = _scheduler(optimizer, config["warmup_epochs"] * len(train_loader), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output / "config.yaml")
    run_details = {
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "class_weights": class_weights.tolist(),
        "train_samples": len(train_dataset),
        "validation_samples": len(val_dataset),
        "selection_metric": config["selection_metric"],
    }
    (output / "environment.json").write_text(
        json.dumps(run_details, indent=2) + "\n", encoding="utf-8"
    )

    best_score = -float("inf")
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, config["epochs"] + 1):
        epoch_start = perf_counter()
        model.train()
        train_dataset.set_epoch(epoch)
        losses: list[float] = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{config['epochs']}"):
            images = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=True):
                logits = _resize_logits(model(pixel_values=images).logits, labels)
                loss = criterion(logits, labels) + config[
                    "dice_loss_weight"
                ] * multiclass_dice_loss(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(float(loss.item()))

        metrics, val_loss = evaluate(
            model, val_loader, device, criterion, config["dice_loss_weight"]
        )
        score = float(metrics[config["selection_metric"]])
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation_loss": val_loss,
            "selection_score": score,
            "elapsed_seconds": perf_counter() - epoch_start,
            "mean_iou_all_classes": metrics["mean_iou_all_classes"],
            "mean_dice_all_classes": metrics["mean_dice_all_classes"],
            "mean_iou_foreground": metrics["mean_iou_foreground"],
            "mean_dice_foreground": metrics["mean_dice_foreground"],
        }
        history.append(row)
        _save_checkpoint(
            output / "checkpoints" / "last.pt",
            model,
            optimizer,
            epoch,
            config,
            class_weights,
            best_score,
        )
        if score > best_score:
            best_score = score
            stale_epochs = 0
            _save_checkpoint(
                output / "checkpoints" / "best.pt",
                model,
                optimizer,
                epoch,
                config,
                class_weights,
                best_score,
            )
            (output / "best_validation_metrics.json").write_text(
                json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
            )
        else:
            stale_epochs += 1
        with (output / "history.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)
        if stale_epochs >= config["early_stopping_patience"]:
            break

    summary = {"best_selection_score": best_score, "epochs_completed": len(history), **run_details}
    (output / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
