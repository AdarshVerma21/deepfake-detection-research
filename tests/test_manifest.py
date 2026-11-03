import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from src.data.faceforensics import MANIFEST_COLUMNS
from src.data.manifest import prepare_split_manifests


def _write_manifest(root: Path, rows: list[dict[str, object]]) -> Path:
    (root / "face_crops").mkdir()
    for row in rows:
        image_path = root / str(row["frame_path"])
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4, 4), "white").save(image_path)
    path = root / "manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _row(video: str, source: str, split: str, label: int, frame: str) -> dict[str, object]:
    return {
        "video_id": video,
        "source_video_id": source,
        "manipulation_type": "original" if label == 0 else "Deepfakes",
        "label": label,
        "split": split,
        "frame_path": frame,
        "frame_index": 0,
        "original_video_path": "source.mp4",
        "fps": 30,
        "frame_count": 1,
        "width": 4,
        "height": 4,
    }


def test_prepare_manifests_preserves_official_splits_and_metadata(tmp_path: Path) -> None:
    rows = [
        _row("train-real", "source-train", "train", 0, "face_crops/train-real.jpg"),
        _row("train-fake", "source-train-fake", "train", 1, "face_crops/train-fake.jpg"),
        _row("val-real", "source-val", "val", 0, "face_crops/val-real.jpg"),
        _row("val-fake", "source-val-fake", "val", 1, "face_crops/val-fake.jpg"),
        _row("test-real", "source-test", "test", 0, "face_crops/test-real.jpg"),
        _row("test-fake", "source-test-fake", "test", 1, "face_crops/test-fake.jpg"),
    ]
    report = prepare_split_manifests(_write_manifest(tmp_path, rows), tmp_path)
    assert set(report.split_paths) == {"train", "val", "test"}
    with report.split_paths["train"].open(newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert written[0]["frame_path"] == "face_crops/train-real.jpg"
    assert written[0]["frame_index"] == "0"
    stats = json.loads(report.statistics_path.read_text(encoding="utf-8"))
    assert stats["splits"]["train"]["real"] == 1


def test_prepare_normalizes_absolute_frame_path(tmp_path: Path) -> None:
    rows = [
        _row("train", "source-train", "train", 0, "face_crops/train.jpg"),
        _row("val", "source-val", "val", 0, "face_crops/val.jpg"),
        _row("test", "source-test", "test", 1, "face_crops/test.jpg"),
    ]
    path = _write_manifest(tmp_path, rows)
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows_from_disk = list(csv.DictReader(handle))
    rows_from_disk[0]["frame_path"] = str(tmp_path / rows_from_disk[0]["frame_path"])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows_from_disk)
    report = prepare_split_manifests(path, tmp_path)
    with report.split_paths["train"].open(newline="", encoding="utf-8") as handle:
        assert next(csv.DictReader(handle))["frame_path"] == "face_crops/train.jpg"


def test_prepare_rejects_missing_image(tmp_path: Path) -> None:
    rows = [_row(split, f"source-{split}", split, label, f"face_crops/{split}-{label}.jpg") for split, label in (
        ("train", 0), ("val", 0), ("test", 1)
    )]
    path = _write_manifest(tmp_path, rows)
    (tmp_path / "face_crops/test-1.jpg").unlink()
    with pytest.raises(FileNotFoundError, match="Missing image"):
        prepare_split_manifests(path, tmp_path)


@pytest.mark.parametrize(
    "change, message",
    [
        ({"label": 2}, "Label must be"),
        ({"split": ""}, "Invalid or empty split"),
    ],
)
def test_prepare_rejects_invalid_labels_and_splits(tmp_path: Path, change: dict[str, object], message: str) -> None:
    rows = [_row("a", "sa", "train", 0, "face_crops/a.jpg"),
            _row("b", "sb", "val", 0, "face_crops/b.jpg"),
            _row("c", "sc", "test", 1, "face_crops/c.jpg")]
    rows[0].update(change)
    with pytest.raises(ValueError, match=message):
        prepare_split_manifests(_write_manifest(tmp_path, rows), tmp_path)


def test_prepare_rejects_video_split_leakage(tmp_path: Path) -> None:
    rows = [
        _row("same", "source-a", "train", 0, "face_crops/a.jpg"),
        _row("same", "source-a", "val", 0, "face_crops/b.jpg"),
        _row("test", "source-test", "test", 1, "face_crops/c.jpg"),
    ]
    with pytest.raises(ValueError, match="video_id crosses"):
        prepare_split_manifests(_write_manifest(tmp_path, rows), tmp_path)


def test_prepare_rejects_source_split_leakage(tmp_path: Path) -> None:
    rows = [
        _row("train", "same-source", "train", 0, "face_crops/a.jpg"),
        _row("val", "same-source", "val", 1, "face_crops/b.jpg"),
        _row("test", "source-test", "test", 1, "face_crops/c.jpg"),
    ]
    with pytest.raises(ValueError, match="source_video_id crosses"):
        prepare_split_manifests(_write_manifest(tmp_path, rows), tmp_path)
