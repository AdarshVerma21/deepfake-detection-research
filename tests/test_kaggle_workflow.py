import csv
import json
from pathlib import Path

import pytest

from src.data.faceforensics import MANIFEST_COLUMNS
from src.data.kaggle_backup import BackupError, KaggleBackupManager, validate_checkpoint


def _checkpoint(root: Path, videos: int) -> None:
    rows = []
    for index in range(videos):
        relative = Path("face_crops") / f"{index:03d}.jpg"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"crop-{index}".encode())
        rows.append(
            {
                "video_id": f"{index:03d}",
                "source_video_id": f"{index:03d}",
                "manipulation_type": "original",
                "label": 0,
                "split": "train",
                "frame_path": relative.as_posix(),
                "frame_index": 0,
                "original_video_path": f"{index:03d}.mp4",
                "fps": 30,
                "frame_count": 1,
                "width": 224,
                "height": 224,
            }
        )
    with (root / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def test_checkpoint_validation_rejects_duplicate_records(tmp_path: Path) -> None:
    _checkpoint(tmp_path, 1)
    with (tmp_path / "manifest.csv").open("a", encoding="utf-8") as handle:
        handle.write((tmp_path / "manifest.csv").read_text(encoding="utf-8").splitlines()[1] + "\n")
    with pytest.raises(BackupError, match="Duplicate"):
        validate_checkpoint(tmp_path)


def test_package_excludes_credentials_and_upload_verifies_status(tmp_path: Path) -> None:
    _checkpoint(tmp_path, 2)
    (tmp_path / "kaggle.json").write_text("secret", encoding="utf-8")
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        output = "Version: 7\n"
        return type("Result", (), {"returncode": 0, "stdout": output, "stderr": ""})()

    manager = KaggleBackupManager("adarshverma21/private-checkpoints", runner=runner)
    package, fingerprint = manager.create_package(tmp_path, 2)
    assert not (package / "kaggle.json").exists()
    uploaded = manager.upload(package, 2, fingerprint)
    assert uploaded.version == "7"
    assert calls[0][:4] == ["kaggle", "datasets", "version", "-p"]
    assert calls[1][:4] == ["kaggle", "datasets", "status", "-d"]


def test_failed_upload_does_not_report_a_backup(tmp_path: Path) -> None:
    _checkpoint(tmp_path, 1)

    def runner(command, **kwargs):
        return type("Result", (), {"returncode": 1, "stdout": "", "stderr": "denied"})()

    manager = KaggleBackupManager("adarshverma21/private-checkpoints", runner=runner)
    package, fingerprint = manager.create_package(tmp_path, 1)
    with pytest.raises(BackupError, match="upload failed"):
        manager.upload(package, 1, fingerprint)
    assert not (tmp_path / "last_verified_backup.json").exists()


def test_progress_records_verified_version_only_after_upload(tmp_path: Path) -> None:
    progress = tmp_path / "progress.json"
    progress.write_text(json.dumps({"completed_video_ids": ["a"], "configuration": {}}), encoding="utf-8")
    from src.data.kaggle_workflow import KagglePreprocessingWorkflow

    state = json.loads(progress.read_text(encoding="utf-8"))
    KagglePreprocessingWorkflow._record_verified_backup(progress, state, "3", 1, "abc")
    assert json.loads(progress.read_text(encoding="utf-8"))["last_verified_backup"]["version"] == "3"
