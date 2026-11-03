"""Reproducible binary classification training and checkpoint utilities."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import calculate_metrics
from src.utils.reproducibility import set_seed


@dataclass
class TrainingConfig:
    epochs: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    scheduler: str = "cosine"
    scheduler_step_size: int = 5
    scheduler_gamma: float = 0.1
    early_stopping_patience: int = 5
    gradient_clip_norm: float = 1.0
    mixed_precision: bool = True
    seed: int = 42
    checkpoint_dir: str = "outputs/checkpoints"
    class_imbalance_mode: str = "none"
    positive_class_weight: float | None = None


def build_binary_criterion(config: TrainingConfig, device: torch.device) -> nn.Module:
    """Build the configured binary loss without inspecting dataset labels."""
    if config.class_imbalance_mode not in {"none", "weighted_loss"}:
        raise ValueError("class_imbalance_mode must be 'none' or 'weighted_loss'")
    if config.class_imbalance_mode == "weighted_loss":
        if config.positive_class_weight is None or config.positive_class_weight <= 0:
            raise ValueError("weighted_loss requires a positive positive_class_weight")
        return nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(config.positive_class_weight, device=device, dtype=torch.float32)
        )
    return nn.BCEWithLogitsLoss()


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    best_val_loss: float,
    history: list[dict[str, Any]],
    config: TrainingConfig,
) -> None:
    """Persist all state required to resume an experiment."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "best_val_loss": best_val_loss,
            "history": history,
            "config": asdict(config),
        },
        destination,
    )


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Load model and optional optimizer/scheduler state from a checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
    gradient_clip_norm: float,
) -> tuple[float, dict[str, Any]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    labels: list[int] = []
    probabilities: list[float] = []
    samples = 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        try:
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images).reshape(-1)
                loss = criterion(logits, targets)
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
        except torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError(
                "CUDA out of memory. Reduce the configured batch_size or image_size explicitly; "
                "experimental settings were not changed automatically."
            ) from exc
        batch_size = targets.numel()
        total_loss += float(loss.detach()) * batch_size
        samples += batch_size
        labels.extend(targets.detach().cpu().int().tolist())
        probabilities.extend(torch.sigmoid(logits.detach()).cpu().tolist())
    return total_loss / max(samples, 1), calculate_metrics(labels, probabilities)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device | str,
    config: TrainingConfig,
    logger: logging.Logger | None = None,
    resume_from: str | Path | None = None,
) -> dict[str, Any]:
    """Train with AdamW, scheduler, early stopping, and best checkpointing."""
    set_seed(config.seed)
    device = torch.device(device)
    model.to(device)
    criterion = build_binary_criterion(config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler: Any
    if config.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    elif config.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma
        )
    elif config.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=1)
    elif config.scheduler == "none":
        scheduler = None
    else:
        raise ValueError(f"Unsupported scheduler: {config.scheduler}")
    use_amp = bool(config.mixed_precision and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history: list[dict[str, Any]] = []
    start_epoch, best_val_loss, stale_epochs = 0, float("inf"), 0
    if resume_from:
        checkpoint = load_checkpoint(resume_from, model, optimizer, scheduler, device)
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
        history = list(checkpoint.get("history", []))
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best.pt"
    for epoch in range(start_epoch, config.epochs):
        started = time.perf_counter()
        train_loss, train_metrics = _run_epoch(
            model, train_loader, criterion, device, optimizer, scaler, use_amp, config.gradient_clip_norm
        )
        val_loss, val_metrics = _run_epoch(
            model, val_loader, criterion, device, None, scaler, use_amp, config.gradient_clip_norm
        )
        if scheduler is not None:
            scheduler.step(val_loss) if config.scheduler == "plateau" else scheduler.step()
        record = {
            "epoch": epoch,
            "duration_seconds": time.perf_counter() - started,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        history.append(record)
        save_checkpoint(checkpoint_dir / "last.pt", model, optimizer, scheduler, epoch, best_val_loss, history, config)
        if val_loss < best_val_loss:
            best_val_loss, stale_epochs = val_loss, 0
            save_checkpoint(best_path, model, optimizer, scheduler, epoch, best_val_loss, history, config)
        else:
            stale_epochs += 1
        if logger:
            logger.info("epoch=%d train_loss=%.5f val_loss=%.5f val_f1=%.4f", epoch, train_loss, val_loss, val_metrics["f1"])
        if stale_epochs >= config.early_stopping_patience:
            break
    (checkpoint_dir / "history.json").write_text(json.dumps(history, indent=2, allow_nan=False), encoding="utf-8")
    return {"history": history, "best_checkpoint": str(best_path), "best_val_loss": best_val_loss}
