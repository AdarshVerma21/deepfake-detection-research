"""Artifact writing for immutable test-set evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

from src.evaluation.metrics import calculate_metrics


def save_test_artifacts(
    rows: Sequence[dict[str, str]],
    labels: Sequence[int],
    probabilities: Sequence[float],
    metrics_path: str | Path,
    predictions_path: str | Path,
    checkpoint: str | Path,
    test_csv: str | Path,
) -> dict[str, Any]:
    """Save metrics and row-level predictions without changing test rows."""

    if len(rows) != len(labels) or len(rows) != len(probabilities):
        raise ValueError("Test rows, labels, and probabilities must have equal lengths")
    metrics = calculate_metrics(list(labels), list(probabilities))
    metrics["checkpoint"] = str(checkpoint)
    metrics["test_csv"] = str(test_csv)
    metrics_file = Path(metrics_path)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_file.write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    predictions_file = Path(predictions_path)
    predictions_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) + ["probability", "prediction"] if rows else ["probability", "prediction"]
    with predictions_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, probability in zip(rows, probabilities):
            output_row = dict(row)
            output_row["probability"] = f"{probability:.10f}"
            output_row["prediction"] = str(int(probability >= 0.5))
            writer.writerow(output_row)
    return metrics
