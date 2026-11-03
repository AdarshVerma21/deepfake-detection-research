"""Private Kaggle checkpointing for interruption-safe preprocessing."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Callable, Iterable, Sequence

from src.data.faceforensics import MANIFEST_COLUMNS


class BackupError(RuntimeError):
    """Raised when a checkpoint cannot be safely packaged or verified."""


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
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return len(videos), digest.hexdigest()


class KaggleBackupManager:
    """Create complete private-dataset versions and verify their upload."""

    def __init__(
        self,
        dataset: str,
        runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    ) -> None:
        if "/" not in dataset:
            raise ValueError("dataset must use owner/slug form")
        self.dataset = dataset
        self._runner = runner or subprocess.run

    def create_package(self, output_dir: Path, completed_videos: int) -> tuple[Path, str]:
        count, fingerprint = validate_checkpoint(output_dir)
        if count != completed_videos:
            raise BackupError(f"Manifest contains {count} videos, expected {completed_videos}")
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

    def upload(self, package: Path, completed_videos: int, fingerprint: str) -> BackupUpload:
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
        status = self._runner(
            ["kaggle", "datasets", "status", "-d", self.dataset],
            check=False,
            capture_output=True,
            text=True,
        )
        if status.returncode != 0:
            raise BackupError(f"Kaggle upload verification failed: {status.stderr.strip()}")
        output = f"{upload.stdout}\n{status.stdout}"
        match = re.search(r"(?:version|versionNumber|Version)\D+([0-9]+(?:\.[0-9]+)*)", output)
        if not match:
            raise BackupError("Kaggle upload succeeded but no backup version was reported")
        return BackupUpload(match.group(1), completed_videos, fingerprint)

    def restore_latest(self, output_dir: Path) -> None:
        destination = Path(tempfile.mkdtemp(prefix="faceforensics-kaggle-restore-"))
        try:
            result = self._runner(
                ["kaggle", "datasets", "download", "-d", self.dataset, "-p", str(destination), "--unzip"],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise BackupError(f"Kaggle restore failed: {result.stderr.strip()}")
            roots = [path.parent for path in destination.rglob("manifest.csv")]
            if not roots:
                raise BackupError("Restored Kaggle dataset has no manifest.csv")
            source = roots[0]
            validate_checkpoint(source)
            if output_dir.exists():
                shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            for path in _safe_files(source):
                target = output_dir / path.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
        finally:
            shutil.rmtree(destination, ignore_errors=True)


def utc_checkpoint_time() -> str:
    return datetime.now(timezone.utc).isoformat()
