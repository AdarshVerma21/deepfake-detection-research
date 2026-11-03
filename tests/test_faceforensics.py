import json
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from src.data.faceforensics import (
    FaceForensicsPipeline,
    FaceForensicsPipelineConfig,
    discover_videos,
    load_split_assignments,
    sample_frame_indices,
)


def _fixture_root(tmp_path: Path) -> Path:
    root = tmp_path / "ff++"
    for relative in (
        "original_sequences/youtube/c23",
        "manipulated_sequences/Deepfakes/c23",
    ):
        (root / relative).mkdir(parents=True)
    (root / "original_sequences/youtube/c23/000.mp4").touch()
    (root / "manipulated_sequences/Deepfakes/c23/000_001.mp4").touch()
    return root


def test_discover_videos_assigns_source_ids(tmp_path: Path) -> None:
    records = discover_videos(_fixture_root(tmp_path), ("original", "Deepfakes"))
    assert [(record.video_id, record.source_video_id, record.label) for record in records] == [
        ("000", "000", 0),
        ("000_001", "000", 1),
    ]


def test_sampling_is_deterministic_and_bounded() -> None:
    first = sample_frame_indices(20, 5, 42, "000", "random")
    assert first == sample_frame_indices(20, 5, 42, "000", "random")
    assert len(first) == 5
    assert all(0 <= index < 20 for index in first)


def test_split_loader_supports_nested_official_schema(tmp_path: Path) -> None:
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    (split_dir / "train.json").write_text(json.dumps([["000", "001"], ["002", "003"]]), encoding="utf-8")
    assignments = load_split_assignments(split_dir, __import__("logging").getLogger("test"))
    assert assignments["000"] == "train"
    assert assignments["003"] == "train"


def test_pipeline_uses_mocked_capture_and_detector(tmp_path: Path, monkeypatch) -> None:
    root = _fixture_root(tmp_path)
    output = tmp_path / "output"
    frame = np.full((32, 48, 3), 127, dtype=np.uint8)

    class FakeCapture:
        def __init__(self, path: str) -> None:
            self.path = path

        def isOpened(self) -> bool:
            return True

        def get(self, property_id: int) -> float:
            return {
                cv2.CAP_PROP_FRAME_COUNT: 3,
                cv2.CAP_PROP_FPS: 30,
                cv2.CAP_PROP_FRAME_WIDTH: 48,
                cv2.CAP_PROP_FRAME_HEIGHT: 32,
            }.get(property_id, 0)

        def set(self, property_id: int, value: int) -> bool:
            return True

        def read(self):
            return True, frame.copy()

        def release(self) -> None:
            return None

    monkeypatch.setattr("src.data.faceforensics.cv2.VideoCapture", FakeCapture)
    detector = type("Detector", (), {"detect": lambda self, image: [(4, 4, 20, 20)]})()
    config = FaceForensicsPipelineConfig(
        dataset_root=root,
        output_dir=output,
        categories=("original",),
        frames_per_video=2,
        crop_size=16,
    )
    report = FaceForensicsPipeline(config, face_detector=detector).run()
    assert report.samples_written == 2
    rows = (output / "manifest.csv").read_text(encoding="utf-8").splitlines()
    assert rows[0].startswith("video_id,source_video_id,manipulation_type")
    assert len(rows) == 3
    assert cv2.imread(str(output / "face_crops/original/000/000000.jpg")).shape[:2] == (16, 16)


def test_pipeline_batches_and_persists_progress(tmp_path: Path, monkeypatch) -> None:
    root = _fixture_root(tmp_path)
    (root / "original_sequences/youtube/c23/001.mp4").touch()
    (root / "original_sequences/youtube/c23/002.mp4").touch()
    output = tmp_path / "output"
    frame = np.full((32, 48, 3), 127, dtype=np.uint8)

    class FakeCapture:
        def __init__(self, path: str) -> None:
            self.path = path

        def isOpened(self) -> bool:
            return True

        def get(self, property_id: int) -> float:
            return {cv2.CAP_PROP_FRAME_COUNT: 1, cv2.CAP_PROP_FPS: 30, cv2.CAP_PROP_FRAME_WIDTH: 48,
                    cv2.CAP_PROP_FRAME_HEIGHT: 32}.get(property_id, 0)

        def set(self, property_id: int, value: int) -> bool:
            return True

        def read(self):
            return True, frame.copy()

        def release(self) -> None:
            return None

    monkeypatch.setattr("src.data.faceforensics.cv2.VideoCapture", FakeCapture)
    detector = type("Detector", (), {"detect": lambda self, image: [(4, 4, 20, 20)]})()
    config = FaceForensicsPipelineConfig(dataset_root=root, output_dir=output, categories=("original",), frames_per_video=1)
    report = FaceForensicsPipeline(config, face_detector=detector).run(batch_size=1)
    progress = json.loads((output / "progress.json").read_text(encoding="utf-8"))
    assert report.videos_processed == 3
    assert progress["completed_videos"] == 3
    assert progress["failed_videos"] == {}


def test_pipeline_resumes_skips_completed_and_retries_failed_without_duplicates(tmp_path: Path, monkeypatch) -> None:
    root = _fixture_root(tmp_path)
    output = tmp_path / "output"
    frame = np.full((32, 48, 3), 127, dtype=np.uint8)
    calls: list[str] = []

    class FakeCapture:
        def __init__(self, path: str) -> None:
            self.path = path

        def isOpened(self) -> bool:
            return True

        def get(self, property_id: int) -> float:
            return {cv2.CAP_PROP_FRAME_COUNT: 1, cv2.CAP_PROP_FPS: 30, cv2.CAP_PROP_FRAME_WIDTH: 48,
                    cv2.CAP_PROP_FRAME_HEIGHT: 32}.get(property_id, 0)

        def set(self, property_id: int, value: int) -> bool:
            return True

        def read(self):
            return True, frame.copy()

        def release(self) -> None:
            return None

    monkeypatch.setattr("src.data.faceforensics.cv2.VideoCapture", FakeCapture)

    class FlakyDetector:
        def detect(self, image):
            calls.append("detect")
            if len(calls) == 1:
                raise RuntimeError("simulated interruption")
            return [(4, 4, 20, 20)]

    config = FaceForensicsPipelineConfig(dataset_root=root, output_dir=output, categories=("original",), frames_per_video=1)
    first = FaceForensicsPipeline(config, face_detector=FlakyDetector()).run(batch_size=1)
    assert first.failures == 1
    second = FaceForensicsPipeline(config, face_detector=FlakyDetector()).run(batch_size=1, resume=True)
    assert second.failures == 0
    with (output / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(__import__("csv").DictReader(handle))
    assert len(rows) == 1
    assert len({(row["video_id"], row["frame_index"]) for row in rows}) == 1
    progress = json.loads((output / "progress.json").read_text(encoding="utf-8"))
    assert progress["completed_videos"] == 1
    assert progress["failed_videos"] == {}
