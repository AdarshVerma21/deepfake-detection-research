"""Smoke-test EfficientNet-B0 and XceptionNet on synthetic data."""

from __future__ import annotations

import argparse

try:
    import torch
    from torch import nn
except ModuleNotFoundError as exc:
    raise SystemExit("torch is required for scripts/smoke_test_models.py. Run in Kaggle or another PyTorch environment.") from exc

from src.models.classifiers import create_model


def resolve_device(requested: str) -> torch.device:
    normalized = requested.lower().strip()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if normalized == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device option: {requested}")


def run_smoke_test(device: torch.device, batch_size: int, image_size: int) -> None:
    inputs = torch.randn(batch_size, 3, image_size, image_size, device=device)
    targets = torch.randint(0, 2, (batch_size, 1), device=device, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss()

    for name in ("efficientnet_b0", "xception"):
        model = create_model(name, num_classes=1, pretrained=False).to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(inputs)
            if outputs.shape != (batch_size, 1):
                raise AssertionError(f"{name} produced shape {tuple(outputs.shape)}; expected ({batch_size}, 1)")
            if not torch.isfinite(outputs).all():
                raise AssertionError(f"{name} produced non-finite outputs")
            loss = criterion(outputs, targets)
            if not torch.isfinite(loss):
                raise AssertionError(f"{name} produced non-finite loss")
        print(f"[PASS] {name}: output_shape={tuple(outputs.shape)} loss={loss.item():.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()
    run_smoke_test(resolve_device(args.device), batch_size=args.batch_size, image_size=args.image_size)
