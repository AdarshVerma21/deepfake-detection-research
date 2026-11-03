"""Strict preparation and validation of FaceForensics++ split manifests."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.data.faceforensics import MANIFEST_COLUMNS


VALID_SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class ManifestPreparationReport:
    """Files and counts produced from one combined manifest."""

    split_paths: dict[str, Path]
    statistics_path: Path
    counts: dict[str, dict[str, int]]


def _read_rows(manifest_path: Path) -> list[dict[str, str]]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Combined manifest not found: {manifest_path}")
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {manifest_path}")
        missing = [column for column in MANIFEST_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"Manifest is missing required columns: {missing}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Manifest contains no samples: {manifest_path}")
    return rows


def _normalise_frame_path(raw_path: str, output_dir: Path) -> str:
    if not raw_path.strip():
        raise ValueError("Manifest contains an empty frame_path")
    candidate = Path(raw_path)
    resolved = candidate.resolve() if candidate.is_absolute() else (output_dir / candidate).resolve()
    try:
        relative = resolved.relative_to(output_dir.resolve())
    except ValueError as exc:
        raise ValueError(f"frame_path is outside preprocessing output directory: {raw_path}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing image referenced by manifest: {resolved}")
    return relative.as_posix()


def prepare_split_manifests(
    manifest_path: str | Path,
    output_dir: str | Path,
    split_output_dir: str | Path | None = None,
) -> ManifestPreparationReport:
    """Validate a combined manifest and write official train/val/test CSVs.

    No rows are resampled or reassigned. ``frame_path`` values are normalized
    relative to the preprocessing output directory and every referenced image
    must exist.
    """

    manifest = Path(manifest_path)
    preprocessing_root = Path(output_dir).resolve()
    destination = Path(split_output_dir) if split_output_dir is not None else preprocessing_root
    destination.mkdir(parents=True, exist_ok=True)
    rows = _read_rows(manifest)
    split_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    video_splits: dict[str, str] = {}
    source_splits: dict[str, str] = {}
    for row_number, row in enumerate(rows, start=2):
        split = row.get("split", "").strip()
        if split not in VALID_SPLITS:
            raise ValueError(f"Invalid or empty split on manifest row {row_number}: {split!r}")
        try:
            label = int(row.get("label", ""))
        except ValueError as exc:
            raise ValueError(f"Invalid label on manifest row {row_number}") from exc
        if label not in (0, 1):
            raise ValueError(f"Label must be 0 or 1 on manifest row {row_number}: {label}")
        video_id = row.get("video_id", "").strip()
        source_video_id = row.get("source_video_id", "").strip()
        if not video_id or not source_video_id:
            raise ValueError(f"Empty video identity on manifest row {row_number}")
        assigned_video = video_splits.setdefault(video_id, split)
        assigned_source = source_splits.setdefault(source_video_id, split)
        if assigned_video != split:
            raise ValueError(f"video_id crosses splits: {video_id}")
        if assigned_source != split:
            raise ValueError(f"source_video_id crosses splits: {source_video_id}")
        row["label"] = str(label)
        row["frame_path"] = _normalise_frame_path(row["frame_path"], preprocessing_root)
        split_rows[split].append({column: row.get(column, "") for column in MANIFEST_COLUMNS})

    missing_splits = [split for split in VALID_SPLITS if not split_rows[split]]
    if missing_splits:
        raise ValueError(f"Manifest has no rows for official split(s): {missing_splits}")

    split_paths: dict[str, Path] = {}
    counts: dict[str, dict[str, int]] = {}
    for split in VALID_SPLITS:
        path = destination / f"{split}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
            writer.writerows(split_rows[split])
        split_paths[split] = path
        counts[split] = {
            "samples": len(split_rows[split]),
            "real": sum(row["label"] == "0" for row in split_rows[split]),
            "fake": sum(row["label"] == "1" for row in split_rows[split]),
            "videos": len({row["video_id"] for row in split_rows[split]}),
            "source_videos": len({row["source_video_id"] for row in split_rows[split]}),
        }

    statistics_path = destination / "split_statistics.json"
    statistics: dict[str, Any] = {
        "manifest": str(manifest),
        "preprocessing_output_dir": str(preprocessing_root),
        "splits": counts,
        "video_split_leakage": False,
        "source_video_split_leakage": False,
    }
    statistics_path.write_text(json.dumps(statistics, indent=2), encoding="utf-8")
    return ManifestPreparationReport(split_paths, statistics_path, counts)
