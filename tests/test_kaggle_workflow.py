import csv
import json
import shutil
from pathlib import Path

import pytest

from src.data.faceforensics import MANIFEST_COLUMNS
from src.data.kaggle_backup import (
    BackupError,
    KaggleBackupManager,
    _configuration_fingerprint,
    validate_checkpoint,
    validate_progress,
)


def _configuration(dataset_root: Path) -> dict:
    return {
        "frames_per_video": 10,
        "crop_size": 224,
        "detector": "haar",
        "sampling_strategy": "uniform",
        "seed": 42,
        "categories": ["original", "Deepfakes"],
        "dataset_identity": str(dataset_root.resolve()),
    }


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
    (root / "progress.json").write_text(
        json.dumps(
            {
                "completed_video_ids": [f"{index:03d}" for index in range(videos)],
                "completed_videos": videos,
                "configuration": _configuration(root / "dataset"),
            }
        ),
        encoding="utf-8",
    )


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

    package_holder = {}

    def runner(command, **kwargs):
        calls.append(command)
        if command[2] == "download":
            destination = Path(command[command.index("-p") + 1])
            shutil.copytree(package_holder["package"], destination, dirs_exist_ok=True)
            return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()
        output = "ready\n" if command[2] == "status" else ""
        return type("Result", (), {"returncode": 0, "stdout": output, "stderr": ""})()

    manager = KaggleBackupManager(
        "adarshverma21/private-checkpoints", runner=runner, privacy_confirmed=True
    )
    configuration = _configuration(tmp_path / "dataset")
    package, fingerprint = manager.create_package(tmp_path, 2, configuration)
    package_holder["package"] = package
    assert not (package / "kaggle.json").exists()
    uploaded = manager.upload(package, 2, fingerprint, configuration=configuration)
    assert uploaded.version == "latest"
    assert uploaded.configuration == configuration
    assert calls[0][:4] == ["kaggle", "datasets", "version", "-p"]
    assert calls[1][:4] == ["kaggle", "datasets", "status", "-d"]
    assert "--json" not in calls[1]
    assert "--dataset-version-number" not in calls[2]


def test_zero_crop_video_cannot_be_counted_as_completed(tmp_path: Path) -> None:
    _checkpoint(tmp_path, 1)
    (tmp_path / "manifest.csv").write_text(
        (tmp_path / "manifest.csv").read_text(encoding="utf-8").splitlines()[0] + "\n",
        encoding="utf-8",
    )
    with pytest.raises(BackupError, match="disagree"):
        validate_progress(tmp_path, set())


def test_inconsistent_progress_and_manifest_counts_are_rejected(tmp_path: Path) -> None:
    _checkpoint(tmp_path, 1)
    (tmp_path / "progress.json").write_text(
        json.dumps({"completed_video_ids": [], "completed_videos": 1}),
        encoding="utf-8",
    )
    with pytest.raises(BackupError, match="completed_videos"):
        validate_progress(tmp_path, {"000"})


def test_malformed_cli_version_output_fails_safely(tmp_path: Path) -> None:
    _checkpoint(tmp_path, 1)

    def runner(command, **kwargs):
        return type("Result", (), {"returncode": 0, "stdout": "Version: ???", "stderr": ""})()

    manager = KaggleBackupManager(
        "adarshverma21/private-checkpoints", runner=runner, privacy_confirmed=True
    )
    package, fingerprint = manager.create_package(tmp_path, 1)
    with pytest.raises(BackupError, match="plain-text ready"):
        manager.upload(package, 1, fingerprint)


def test_privacy_verification_failure_fails_safely(tmp_path: Path) -> None:
    _checkpoint(tmp_path, 1)

    def runner(command, **kwargs):
        if command[2] == "download":
            destination = Path(command[command.index("-p") + 1])
            shutil.copytree(tmp_path, destination, dirs_exist_ok=True)
            return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()
        return type(
            "Result", (), {"returncode": 0, "stdout": "ready\n", "stderr": ""}
        )()

    manager = KaggleBackupManager("adarshverma21/private-checkpoints", runner=runner)
    package, fingerprint = manager.create_package(tmp_path, 1)
    with pytest.raises(BackupError, match="privacy"):
        manager.upload(package, 1, fingerprint)


def test_interrupted_upload_can_be_verified_without_duplicate(tmp_path: Path) -> None:
    _checkpoint(tmp_path, 1)
    calls = []
    configuration = _configuration(tmp_path / "dataset")
    _, fingerprint = validate_checkpoint(tmp_path)
    (tmp_path / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "checkpoint": {
                    "completed_videos": 1,
                    "fingerprint": fingerprint,
                    "configuration": configuration,
                    "configuration_fingerprint": _configuration_fingerprint(configuration),
                }
            }
        ),
        encoding="utf-8",
    )

    def runner(command, **kwargs):
        calls.append(command)
        if command[2] == "status":
            return type(
                "Result", (), {"returncode": 0, "stdout": "ready\n", "stderr": ""}
            )()
        destination = Path(command[command.index("-p") + 1])
        shutil.copytree(tmp_path, destination, dirs_exist_ok=True)
        return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    manager = KaggleBackupManager(
        "adarshverma21/private-checkpoints", runner=runner, privacy_confirmed=True
    )
    count, fingerprint = validate_checkpoint(tmp_path)
    recovered = manager.verify_latest(count, fingerprint, {"000"}, configuration)
    assert recovered is not None
    assert recovered.version == "latest"
    assert not any(command[2] == "version" for command in calls)


def test_restore_validates_and_replaces_output_atomically(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _checkpoint(source, 1)
    _, fingerprint = validate_checkpoint(source)
    configuration = _configuration(source / "dataset")
    (source / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "checkpoint": {
                    "completed_videos": 1,
                    "fingerprint": fingerprint,
                    "configuration": configuration,
                    "configuration_fingerprint": _configuration_fingerprint(configuration),
                }
            }
        ),
        encoding="utf-8",
    )
    output.mkdir()
    (output / "stale.txt").write_text("stale", encoding="utf-8")

    def runner(command, **kwargs):
        if command[2] == "status":
            return type(
                "Result", (), {"returncode": 0, "stdout": "ready\n", "stderr": ""}
            )()
        destination = Path(command[command.index("-p") + 1])
        shutil.copytree(source, destination, dirs_exist_ok=True)
        return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    KaggleBackupManager(
        "adarshverma21/private-checkpoints", runner=runner, privacy_confirmed=True
    ).restore_latest(output, configuration)
    assert not (output / "stale.txt").exists()
    assert (output / "manifest.csv").is_file()


def test_restore_rejects_mismatching_preprocessing_configuration(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _checkpoint(source, 1)
    _, fingerprint = validate_checkpoint(source)
    checkpoint_configuration = _configuration(source / "dataset")
    (source / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "checkpoint": {
                    "completed_videos": 1,
                    "fingerprint": fingerprint,
                    "configuration": checkpoint_configuration,
                    "configuration_fingerprint": _configuration_fingerprint(checkpoint_configuration),
                }
            }
        ),
        encoding="utf-8",
    )
    requested = dict(checkpoint_configuration, frames_per_video=5)

    def runner(command, **kwargs):
        if command[2] == "status":
            return type("Result", (), {"returncode": 0, "stdout": "ready\n", "stderr": ""})()
        destination = Path(command[command.index("-p") + 1])
        shutil.copytree(source, destination, dirs_exist_ok=True)
        return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    with pytest.raises(BackupError, match="incompatible"):
        KaggleBackupManager(
            "adarshverma21/private-checkpoints", runner=runner, privacy_confirmed=True
        ).restore_latest(output, requested)
    assert not output.exists()


def test_clean_start_refuses_to_reuse_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "pilot"
    output.mkdir()
    from scripts.run_kaggle_preprocessing import ensure_clean_start

    with pytest.raises(SystemExit, match="refusing to modify"):
        ensure_clean_start(output, True)


def test_clean_start_and_resume_are_mutually_exclusive() -> None:
    from scripts.run_kaggle_preprocessing import ensure_start_mode

    with pytest.raises(SystemExit, match="cannot be combined"):
        ensure_start_mode(True, True)


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
