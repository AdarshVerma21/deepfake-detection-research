"""Private Kaggle checkpointing for interruption-safe preprocessing."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Callable, Iterable

from src.data.faceforensics import MANIFEST_COLUMNS


class BackupError(RuntimeError):
    """Raised when a checkpoint cannot be safely packaged or verified."""


class _BackupMismatch(BackupError):
    """Raised when a remote version is valid but is not the requested checkpoint."""


@dataclass(frozen=True)
class BackupUpload:
    version: str
    completed_videos: int
    fingerprint: str


def _safe_files(root: Path) -> Iterable[Path]:
    excluded = {"logs", "__pycache__"}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(part in excluded for part in path.parts):
            continue
        if path.name.startswith(".") or path.suffix in {".tmp", ".key", ".pem", ".secret"}:
            continue
        if path.name.lower() in {".env", ".kaggle", "kaggle.json"} or "credentials" in path.name.lower():
            continue
        yield path


def validate_checkpoint(root: Path) -> tuple[int, str]:
    """Validate manifest/crop integrity and return video count and fingerprint."""

    manifest = root / "manifest.csv"
    if not manifest.is_file():
        raise BackupError(f"Checkpoint is missing {manifest}")
    import csv

    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or any(column not in reader.fieldnames for column in MANIFEST_COLUMNS):
            raise BackupError("Checkpoint manifest has an invalid header")
        rows = list(reader)
    identities: set[tuple[str, str, str]] = set()
    videos: set[str] = set()
    for row in rows:
        key = (row["video_id"], row["frame_index"], row["frame_path"])
        if key in identities:
            raise BackupError(f"Duplicate manifest record: {key}")
        identities.add(key)
        crop = (root / row["frame_path"]).resolve()
        if root.resolve() not in crop.parents or not crop.is_file():
            raise BackupError(f"Missing or out-of-tree crop: {row['frame_path']}")
        videos.add(row["video_id"])
    digest = hashlib.sha256()
    for path in _safe_files(root):
        if path.name == "dataset-metadata.json":
            continue
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return len(videos), digest.hexdigest()


def validate_progress(root: Path, manifest_videos: set[str] | None = None) -> set[str]:
    """Validate progress state independently from the manifest video count."""

    path = root / "progress.json"
    if not path.is_file():
        raise BackupError(f"Checkpoint is missing {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BackupError(f"Unable to read progress state: {path}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("completed_video_ids"), list):
        raise BackupError("Checkpoint progress has no completed_video_ids list")
    completed = {str(item) for item in value["completed_video_ids"]}
    if len(completed) != len(value["completed_video_ids"]):
        raise BackupError("Checkpoint progress contains duplicate completed video IDs")
    recorded = value.get("completed_videos")
    if recorded is not None and recorded != len(completed):
        raise BackupError("Checkpoint progress completed_videos disagrees with completed_video_ids")
    if manifest_videos is not None and completed != manifest_videos:
        raise BackupError(
            "Checkpoint progress and manifest disagree: "
            f"progress={len(completed)}, manifest={len(manifest_videos)}"
        )
    return completed


class KaggleBackupManager:
    """Create complete private-dataset versions and verify their upload."""

    def __init__(
        self,
        dataset: str,
        runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
        privacy_confirmed: bool = False,
    ) -> None:
        if "/" not in dataset:
            raise ValueError("dataset must use owner/slug form")
        self.dataset = dataset
        self._runner = runner or subprocess.run
        self.privacy_confirmed = privacy_confirmed

    def create_package(self, output_dir: Path, completed_videos: int) -> tuple[Path, str]:
        count, fingerprint = validate_checkpoint(output_dir)
        if count != completed_videos:
            raise BackupError(f"Manifest contains {count} videos, expected {completed_videos}")
        if (output_dir / "progress.json").is_file():
            validate_progress(output_dir, manifest_video_ids(output_dir))
        package = Path(tempfile.mkdtemp(prefix="faceforensics-kaggle-"))
        for source in _safe_files(output_dir):
            destination = package / source.relative_to(output_dir)
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)
        metadata = {
            "id": self.dataset,
            "title": self.dataset.rsplit("/", 1)[-1],
            "description": "Private FaceForensics++ c23 preprocessing checkpoints.",
            "licenses": [{"name": "other"}],
            "private": True,
            "checkpoint": {"completed_videos": completed_videos, "fingerprint": fingerprint},
        }
        (package / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return package, fingerprint

    def upload(
        self,
        package: Path,
        completed_videos: int,
        fingerprint: str,
        expected_video_ids: set[str] | None = None,
    ) -> BackupUpload:
        if any("kaggle.json" in path.name.lower() or "credentials" in path.name.lower() for path in package.rglob("*")):
            raise BackupError("Refusing to upload credentials in checkpoint package")
        message = f"FaceForensics preprocessing checkpoint: {completed_videos} videos"
        upload = self._runner(
            ["kaggle", "datasets", "version", "-p", str(package), "-m", message, "--dir-mode", "zip"],
            check=False,
            capture_output=True,
            text=True,
        )
        if upload.returncode != 0:
            raise BackupError(f"Kaggle upload failed: {upload.stderr.strip()}")
        metadata = self._status_metadata("Kaggle upload verification failed")
        verified = self._verify_version(
            self._status_version(metadata),
            completed_videos,
            fingerprint,
            expected_video_ids,
        )
        self._require_manual_privacy_confirmation()
        return verified

    def verify_latest(
        self,
        completed_videos: int,
        fingerprint: str,
        expected_video_ids: set[str] | None = None,
    ) -> BackupUpload | None:
        """Find a completed matching upload after an interruption."""

        metadata = self._status_metadata("Kaggle backup verification failed")
        try:
            verified = self._verify_version(
                self._status_version(metadata),
                completed_videos,
                fingerprint,
                expected_video_ids,
            )
            self._require_manual_privacy_confirmation()
            return verified
        except _BackupMismatch:
            return None

    def _verify_version(
        self,
        version: str,
        completed_videos: int,
        fingerprint: str,
        expected_video_ids: set[str] | None = None,
    ) -> BackupUpload:
        destination = Path(tempfile.mkdtemp(prefix="faceforensics-kaggle-verify-"))
        try:
            command = [
                "kaggle", "datasets", "download", "-d", self.dataset,
                "-p", str(destination), "--unzip",
            ]
            if version != "latest":
                command.extend(["--dataset-version-number", version])
            result = self._runner(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise BackupError(f"Kaggle backup version verification failed: {result.stderr.strip()}")
            roots = [path.parent for path in destination.rglob("manifest.csv")]
            if len(roots) != 1:
                raise BackupError("Kaggle backup version verification found an ambiguous manifest")
            manifest_count, actual_fingerprint = validate_checkpoint(roots[0])
            manifest_videos = manifest_video_ids(roots[0])
            progress_videos = validate_progress(roots[0], manifest_videos)
            if (
                manifest_count != completed_videos
                or len(progress_videos) != completed_videos
                or actual_fingerprint != fingerprint
                or (expected_video_ids is not None and progress_videos != expected_video_ids)
            ):
                raise _BackupMismatch("Kaggle backup version contents do not match the uploaded checkpoint")
            return BackupUpload(version, len(progress_videos), actual_fingerprint)
        finally:
            shutil.rmtree(destination, ignore_errors=True)

    def _status_metadata(self, error_prefix: str) -> dict[str, Any]:
        status = self._runner(
            ["kaggle", "datasets", "status", "-d", self.dataset],
            check=False,
            capture_output=True,
            text=True,
        )
        if status.returncode != 0:
            raise BackupError(f"{error_prefix}: {status.stderr.strip()}")
        return self._parse_status(status.stdout)

    @staticmethod
    def _parse_status(output: str) -> dict[str, Any]:
        if output.strip().lower() == "ready":
            return {"status": "ready"}
        raise BackupError("Kaggle status did not return the expected plain-text ready state")

    @staticmethod
    def _status_version(metadata: dict[str, Any]) -> str:
        if metadata.get("status") == "ready":
            return "latest"
        raise BackupError("Kaggle status did not identify a ready dataset")

    def _require_manual_privacy_confirmation(self) -> None:
        if not self.privacy_confirmed:
            raise BackupError(
                "Kaggle CLI 2.0.2 does not provide an authoritative privacy check; "
                "confirm the dataset is private in Kaggle, then retry with privacy_confirmed=True"
            )

    def restore_latest(self, output_dir: Path) -> None:
        destination = Path(tempfile.mkdtemp(prefix="faceforensics-kaggle-restore-"))
        try:
            metadata = self._status_metadata("Kaggle restore verification failed")
            self._require_manual_privacy_confirmation()
            result = self._runner(
                ["kaggle", "datasets", "download", "-d", self.dataset, "-p", str(destination), "--unzip"],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise BackupError(f"Kaggle restore failed: {result.stderr.strip()}")
            roots = [path.parent for path in destination.rglob("manifest.csv")]
            if len(roots) != 1:
                raise BackupError("Restored Kaggle dataset has no manifest.csv")
            source = roots[0]
            manifest_count, fingerprint = validate_checkpoint(source)
            manifest_videos = manifest_video_ids(source)
            progress_videos = validate_progress(source, manifest_videos)
            metadata_path = source / "dataset-metadata.json"
            if not metadata_path.is_file():
                raise BackupError("Restored backup is missing recorded metadata")
            try:
                recorded = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise BackupError("Restored backup metadata is malformed") from exc
            if not isinstance(recorded, dict):
                raise BackupError("Restored backup metadata is invalid")
            checkpoint = recorded.get("checkpoint", {})
            if not isinstance(checkpoint, dict):
                raise BackupError("Restored backup metadata has no checkpoint record")
            if checkpoint.get("completed_videos") != len(progress_videos) or checkpoint.get("fingerprint") != fingerprint:
                raise BackupError("Restored backup fingerprint or completed-video count is invalid")
            if manifest_count != len(progress_videos):
                raise BackupError("Restored manifest and progress counts disagree")
            output_dir.parent.mkdir(parents=True, exist_ok=True)
            staged = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-restore-", dir=output_dir.parent))
            for path in _safe_files(source):
                target = staged / path.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
            old = output_dir.with_name(f".{output_dir.name}-previous")
            if old.exists():
                shutil.rmtree(old)
            if output_dir.exists():
                os.replace(output_dir, old)
            try:
                os.replace(staged, output_dir)
            except BaseException:
                if old.exists() and not output_dir.exists():
                    os.replace(old, output_dir)
                raise
            shutil.rmtree(old, ignore_errors=True)
        finally:
            shutil.rmtree(destination, ignore_errors=True)


def utc_checkpoint_time() -> str:
    return datetime.now(timezone.utc).isoformat()


def manifest_video_ids(root: Path) -> set[str]:
    import csv

    with (root / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        return {str(row["video_id"]) for row in csv.DictReader(handle)}
