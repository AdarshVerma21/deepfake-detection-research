"""Sequential, verified Kaggle checkpoint orchestration."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from src.data.faceforensics import (
    FaceForensicsPipeline,
    FaceForensicsPipelineConfig,
    discover_videos,
)
from src.data.kaggle_backup import (
    BackupError,
    KaggleBackupManager,
    manifest_video_ids,
    utc_checkpoint_time,
    validate_checkpoint,
    validate_progress,
)


class KagglePreprocessingWorkflow:
    """Process deterministic video ranges and upload only verified checkpoints."""

    def __init__(self, config: FaceForensicsPipelineConfig, backup: KaggleBackupManager) -> None:
        self.config = config
        self.backup = backup

    def run(
        self,
        chunk_size: int = 100,
        start_index: int = 0,
        max_videos: int | None = None,
        resume: bool = False,
    ) -> int:
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")
        if start_index < 0:
            raise ValueError("start_index must be non-negative")
        if max_videos is not None and max_videos < 1:
            raise ValueError("max_videos must be at least 1 when provided")
        progress_path = self.config.output_dir / "progress.json"
        if resume:
            self.backup.restore_latest(self.config.output_dir)
        records = discover_videos(self.config.dataset_root, self.config.categories)
        end_index = min(len(records), start_index + max_videos) if max_videos is not None else len(records)
        processed_limit = end_index
        while True:
            progress = self._read_progress(progress_path)
            completed = {str(value) for value in progress.get("completed_video_ids", [])}
            first = next((index for index in range(start_index, processed_limit) if records[index].video_id not in completed), None)
            if first is None:
                return len(completed.intersection(record.video_id for record in records))
            chunk = records[first : min(first + chunk_size, processed_limit)]
            pipeline = FaceForensicsPipeline(self.config)
            report = pipeline.run(
                batch_size=chunk_size,
                resume=True,
                progress_file=progress_path,
                start_index=first,
                max_videos=len(chunk),
            )
            after = self._read_progress(progress_path)
            completed = {str(value) for value in after.get("completed_video_ids", [])}
            manifest_count, fingerprint = validate_checkpoint(self.config.output_dir)
            manifest_ids = manifest_video_ids(self.config.output_dir)
            progress_ids = validate_progress(self.config.output_dir, manifest_ids)
            if completed != progress_ids:
                raise BackupError("Progress file changed during processing")
            missing = [record.video_id for record in chunk if record.video_id not in completed]
            if report.failures or missing:
                raise BackupError(
                    f"Chunk {first}:{first + len(chunk)} is incomplete; "
                    f"failures={report.failures}, missing={missing}"
                )
            completed_count = len(progress_ids)
            previous = after.get("last_verified_backup", {})
            if previous.get("fingerprint") == fingerprint and previous.get("completed_videos") == completed_count:
                raise BackupError("Checkpoint is unchanged; refusing duplicate upload")
            recovered = self.backup.verify_latest(completed_count, fingerprint)
            if recovered is not None:
                self._record_verified_backup(
                    progress_path, after, recovered.version, recovered.completed_videos, recovered.fingerprint
                )
                continue
            package, _ = self.backup.create_package(self.config.output_dir, completed_count)
            try:
                uploaded = self.backup.upload(package, completed_count, fingerprint)
            finally:
                import shutil

                shutil.rmtree(package, ignore_errors=True)
            self._record_verified_backup(progress_path, after, uploaded.version, uploaded.completed_videos, uploaded.fingerprint)

    @staticmethod
    def _read_progress(path: Path) -> dict[str, Any]:
        if not path.is_file():
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BackupError(f"Unable to read progress state: {path}") from exc
        if not isinstance(value, dict):
            raise BackupError(f"Progress state is not an object: {path}")
        return value

    @staticmethod
    def _record_verified_backup(
        path: Path,
        progress: dict[str, Any],
        version: str,
        completed_videos: int,
        fingerprint: str,
    ) -> None:
        progress["last_verified_backup"] = {
            "version": version,
            "completed_videos": completed_videos,
            "fingerprint": fingerprint,
            "uploaded_at": utc_checkpoint_time(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(progress, handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
