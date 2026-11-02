"""FaceForensics++ c23 discovery, sampling, cropping, and manifest generation.

The default detector is OpenCV's bundled Haar cascade. It is intentionally
simple and CPU-friendly for preparation and testing; production experiments
should record any detector change in their configuration and methodology.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

import cv2
import numpy as np

from src.utils.logging_utils import setup_logger


SUPPORTED_CATEGORIES = {
    "original": Path("original_sequences/youtube/c23"),
    "Deepfakes": Path("manipulated_sequences/Deepfakes/c23"),
    "Face2Face": Path("manipulated_sequences/Face2Face/c23"),
    "FaceSwap": Path("manipulated_sequences/FaceSwap/c23"),
    "NeuralTextures": Path("manipulated_sequences/NeuralTextures/c23"),
}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
UNSUPPORTED_VIDEO_EXTENSIONS = {".flv", ".m4v", ".mpeg", ".mpg", ".wmv"}
MANIFEST_COLUMNS = [
    "video_id",
    "source_video_id",
    "manipulation_type",
    "label",
    "split",
    "frame_path",
    "frame_index",
    "original_video_path",
    "fps",
    "frame_count",
    "width",
    "height",
]


class FaceDetector(Protocol):
    """Interface used by the pipeline and its tests."""

    def detect(self, frame: np.ndarray) -> Sequence[tuple[int, int, int, int]]:
        """Return face boxes as x, y, width, height."""


@dataclass(frozen=True)
class FaceForensicsPipelineConfig:
    """Configuration for one deterministic dataset preparation run."""

    dataset_root: Path
    output_dir: Path
    frames_per_video: int = 5
    crop_size: int = 224
    detector: str = "haar"
    detector_config: str | None = None
    seed: int = 42
    categories: tuple[str, ...] = tuple(SUPPORTED_CATEGORIES)
    split_dir: Path | None = None
    sampling_strategy: str = "uniform"

    def __post_init__(self) -> None:
        if self.frames_per_video < 1:
            raise ValueError("frames_per_video must be at least 1")
        if self.crop_size < 1:
            raise ValueError("crop_size must be at least 1")
        if self.detector != "haar":
            raise ValueError("Only the documented 'haar' detector is currently supported")
        if self.sampling_strategy not in {"uniform", "random"}:
            raise ValueError("sampling_strategy must be 'uniform' or 'random'")
        unknown = set(self.categories) - set(SUPPORTED_CATEGORIES)
        if unknown:
            raise ValueError(f"Unsupported FaceForensics++ categories: {sorted(unknown)}")


@dataclass(frozen=True)
class VideoRecord:
    """A discovered FaceForensics++ video and its dataset label."""

    path: Path
    video_id: str
    source_video_id: str
    manipulation_type: str
    label: int


@dataclass
class PipelineReport:
    """Outputs and counts from a preparation run."""

    manifest_path: Path
    statistics_path: Path
    videos_discovered: int
    videos_processed: int
    samples_written: int
    failures: int
    warnings: int


class HaarCascadeFaceDetector:
    """OpenCV Haar detector, using a supplied or bundled cascade XML file."""

    def __init__(self, cascade_path: str | Path | None = None) -> None:
        path = Path(cascade_path) if cascade_path else Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self._classifier = cv2.CascadeClassifier(str(path))
        if self._classifier.empty():
            raise ValueError(f"Unable to load Haar cascade: {path}")

    def detect(self, frame: np.ndarray) -> Sequence[tuple[int, int, int, int]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return [
            tuple(int(value) for value in box)
            for box in self._classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        ]


def _normalise_id(value: str) -> str:
    return Path(value).stem


def _source_id(video_id: str, manipulation_type: str) -> str:
    if manipulation_type == "original":
        return video_id
    return video_id.split("_", maxsplit=1)[0]


def validate_dataset_root(root: Path, categories: Iterable[str]) -> None:
    """Validate the requested FaceForensics++ c23 category directories."""

    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist or is not a directory: {root}")
    missing = [str(root / SUPPORTED_CATEGORIES[name]) for name in categories if not (root / SUPPORTED_CATEGORIES[name]).is_dir()]
    if missing:
        raise FileNotFoundError("Missing FaceForensics++ category directories: " + ", ".join(missing))


def discover_videos(
    root: Path,
    categories: Iterable[str],
    logger: logging.Logger | None = None,
) -> list[VideoRecord]:
    """Discover supported video files beneath the selected c23 categories."""

    records: list[VideoRecord] = []
    for manipulation_type in categories:
        category_root = root / SUPPORTED_CATEGORIES[manipulation_type]
        for path in sorted(category_root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in VIDEO_EXTENSIONS:
                if logger and path.suffix.lower() in UNSUPPORTED_VIDEO_EXTENSIONS:
                    logger.warning("Unsupported video format skipped: %s", path)
                continue
            video_id = path.stem
            records.append(
                VideoRecord(
                    path=path,
                    video_id=video_id,
                    source_video_id=_source_id(video_id, manipulation_type),
                    manipulation_type=manipulation_type,
                    label=0 if manipulation_type == "original" else 1,
                )
            )
    return records


def extract_video_metadata(path: Path) -> dict[str, float | int]:
    """Read basic metadata and reject unreadable or empty videos."""

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Video cannot be opened: {path}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if frame_count <= 0 or width <= 0 or height <= 0:
            raise ValueError(f"Video has invalid or empty metadata: {path}")
        return {"fps": fps, "frame_count": frame_count, "width": width, "height": height}
    finally:
        capture.release()


def sample_frame_indices(frame_count: int, count: int, seed: int, key: str, strategy: str = "uniform") -> list[int]:
    """Return unique, deterministic frame indices for a video."""

    if frame_count <= 0:
        return []
    sample_count = min(frame_count, count)
    if strategy == "uniform":
        return sorted(set(int(index) for index in np.linspace(0, frame_count - 1, sample_count)))
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
    rng = random.Random(int(digest[:16], 16))
    return sorted(rng.sample(range(frame_count), sample_count))


def preprocess_face(frame: np.ndarray, crop_size: int) -> np.ndarray:
    """Resize a BGR face crop and return an RGB uint8 image for consistent storage."""

    resized = cv2.resize(frame, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


def _valid_crop(frame: np.ndarray, box: Sequence[int]) -> np.ndarray | None:
    if len(box) != 4:
        return None
    x, y, width, height = (int(value) for value in box)
    frame_height, frame_width = frame.shape[:2]
    left, top = max(0, x), max(0, y)
    right, bottom = min(frame_width, x + width), min(frame_height, y + height)
    if right <= left or bottom <= top:
        return None
    return frame[top:bottom, left:right]


def _extract_split_ids(value: Any) -> set[str]:
    if isinstance(value, str):
        return {_normalise_id(value)}
    if isinstance(value, Mapping):
        result: set[str] = set()
        for nested in value.values():
            result.update(_extract_split_ids(nested))
        return result
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        result = set()
        for nested in value:
            result.update(_extract_split_ids(nested))
        return result
    return set()


def load_split_assignments(split_dir: Path | None, logger: logging.Logger) -> dict[str, str]:
    """Load official split JSON files without assuming one particular schema."""

    if split_dir is None:
        logger.warning("No split directory configured; manifest split values will be empty.")
        return {}
    assignments: dict[str, str] = {}
    for split in ("train", "val", "test"):
        candidates = [split_dir / f"{split}.json", split_dir / f"{split}_list.json"]
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            logger.warning("Missing split metadata for '%s' in %s.", split, split_dir)
            continue
        try:
            with path.open("r", encoding="utf-8") as handle:
                ids = _extract_split_ids(json.load(handle))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not read split metadata %s: %s", path, exc)
            continue
        for item in ids:
            if item in assignments and assignments[item] != split:
                raise ValueError(f"Video appears in multiple official splits: {item}")
            assignments[item] = split
    return assignments


class FaceForensicsPipeline:
    """Prepare face crops and a CSV manifest from FaceForensics++ videos."""

    def __init__(
        self,
        config: FaceForensicsPipelineConfig,
        face_detector: FaceDetector | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config
        self.detector = face_detector or HaarCascadeFaceDetector(config.detector_config)
        self.logger = logger or setup_logger("faceforensics", config.output_dir / "logs")
        self._warnings = 0
        self._failures = 0

    def run(self, limit_videos: int | None = None) -> PipelineReport:
        validate_dataset_root(self.config.dataset_root, self.config.categories)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        records = discover_videos(self.config.dataset_root, self.config.categories, self.logger)
        if limit_videos is not None:
            records = records[: max(0, limit_videos)]
        assignments = load_split_assignments(self.config.split_dir, self.logger)
        manifest_path = self.config.output_dir / "manifest.csv"
        crop_root = self.config.output_dir / "face_crops"
        rows: list[dict[str, Any]] = []
        seen: set[tuple[str, int]] = set()
        processed = 0
        for record in records:
            try:
                metadata = extract_video_metadata(record.path)
                indices = sample_frame_indices(
                    int(metadata["frame_count"]),
                    self.config.frames_per_video,
                    self.config.seed,
                    record.video_id,
                    self.config.sampling_strategy,
                )
                capture = cv2.VideoCapture(str(record.path))
                try:
                    wrote_for_video = False
                    for frame_index in indices:
                        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                        ok, frame = capture.read()
                        if not ok or frame is None or frame.size == 0:
                            self._warn("Empty or unreadable frame %s in %s", frame_index, record.path)
                            continue
                        boxes = [box for box in self.detector.detect(frame) if _valid_crop(frame, box) is not None]
                        if not boxes:
                            self._warn("No valid face detected at frame %s in %s", frame_index, record.path)
                            continue
                        box = max(boxes, key=lambda item: int(item[2]) * int(item[3]))
                        crop = _valid_crop(frame, box)
                        if crop is None:
                            self._warn("Invalid face bounding box at frame %s in %s", frame_index, record.path)
                            continue
                        key = (str(record.path), frame_index)
                        if key in seen:
                            self._warn("Duplicate sample skipped: %s frame %s", record.path, frame_index)
                            continue
                        seen.add(key)
                        relative_path = Path("face_crops") / record.manipulation_type / record.video_id / f"{frame_index:06d}.jpg"
                        destination = self.config.output_dir / relative_path
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        if not cv2.imwrite(str(destination), cv2.cvtColor(preprocess_face(crop, self.config.crop_size), cv2.COLOR_RGB2BGR)):
                            raise OSError(f"Could not write face crop: {destination}")
                        rows.append(
                            {
                                "video_id": record.video_id,
                                "source_video_id": record.source_video_id,
                                "manipulation_type": record.manipulation_type,
                                "label": record.label,
                                "split": assignments.get(record.video_id, assignments.get(record.source_video_id, "")),
                                "frame_path": str(relative_path),
                                "frame_index": frame_index,
                                "original_video_path": str(record.path),
                                **metadata,
                            }
                        )
                        wrote_for_video = True
                    processed += int(wrote_for_video)
                finally:
                    capture.release()
            except (OSError, ValueError, cv2.error) as exc:
                self._failure("Failed to process %s: %s", record.path, exc)
        self._write_manifest(manifest_path, rows)
        statistics_path = self._write_statistics(rows, len(records), processed)
        return PipelineReport(
            manifest_path=manifest_path,
            statistics_path=statistics_path,
            videos_discovered=len(records),
            videos_processed=processed,
            samples_written=len(rows),
            failures=self._failures,
            warnings=self._warnings,
        )

    def _warn(self, message: str, *args: Any) -> None:
        self._warnings += 1
        self.logger.warning(message, *args)

    def _failure(self, message: str, *args: Any) -> None:
        self._failures += 1
        self.logger.error(message, *args)

    @staticmethod
    def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

    def _write_statistics(self, rows: list[dict[str, Any]], discovered: int, processed: int) -> Path:
        stats = {
            "videos_discovered": discovered,
            "videos_with_samples": processed,
            "samples_written": len(rows),
            "warnings": self._warnings,
            "failures": self._failures,
            "class_distribution": {
                "real": sum(int(row["label"]) == 0 for row in rows),
                "fake": sum(int(row["label"]) == 1 for row in rows),
            },
            "split_distribution": {
                split: sum(row["split"] == split for row in rows)
                for split in ("train", "val", "test", "")
            },
            "manipulation_distribution": {
                category: sum(row["manipulation_type"] == category for row in rows)
                for category in self.config.categories
            },
            "split_identity_disjointness": "not verified; source-level assignments are preserved when supplied",
        }
        path = self.config.output_dir / "statistics.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(stats, handle, indent=2)
        return path
