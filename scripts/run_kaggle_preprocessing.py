"""Run FaceForensics++ preprocessing with verified private Kaggle checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.faceforensics import FaceForensicsPipelineConfig, SUPPORTED_CATEGORIES
from src.data.kaggle_backup import KaggleBackupManager
from src.data.kaggle_workflow import KagglePreprocessingWorkflow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("/kaggle/input/faceforensics"))
    parser.add_argument("--split-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/faceforensics_c23"))
    parser.add_argument("--kaggle-dataset", default="adarshverma21/faceforensics-c23-processed-checkpoints")
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-videos", type=int)
    parser.add_argument("--frames-per-video", type=int, default=10)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--detector", choices=["haar"], default="haar")
    parser.add_argument("--detector-config", type=Path)
    parser.add_argument("--sampling-strategy", choices=["uniform", "random"], default="uniform")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--categories", nargs="+", choices=sorted(SUPPORTED_CATEGORIES), default=list(SUPPORTED_CATEGORIES))
    parser.add_argument(
        "--privacy-confirmed",
        action="store_true",
        help="Confirm in Kaggle that the backup dataset is private; CLI 2.0.2 cannot verify this.",
    )
    parser.add_argument("--resume", action="store_true", help="Restore the latest verified Kaggle backup before continuing.")
    parser.add_argument(
        "--clean-start",
        action="store_true",
        help="Require a new output directory; never delete or reuse an existing pilot checkpoint.",
    )
    return parser


def ensure_clean_start(output_dir: Path, clean_start: bool) -> None:
    if clean_start and output_dir.exists():
        raise SystemExit(
            f"--clean-start requires a new output directory; refusing to modify existing checkpoint: {output_dir}"
        )


def ensure_start_mode(clean_start: bool, resume: bool) -> None:
    if clean_start and resume:
        raise SystemExit("--clean-start cannot be combined with --resume")


def main() -> None:
    args = build_parser().parse_args()
    ensure_start_mode(args.clean_start, args.resume)
    ensure_clean_start(args.output_dir, args.clean_start)
    config = FaceForensicsPipelineConfig(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        split_dir=args.split_dir,
        frames_per_video=args.frames_per_video,
        crop_size=args.crop_size,
        detector=args.detector,
        detector_config=str(args.detector_config) if args.detector_config else None,
        seed=args.seed,
        categories=tuple(args.categories),
        sampling_strategy=args.sampling_strategy,
    )
    completed = KagglePreprocessingWorkflow(
        config, KaggleBackupManager(args.kaggle_dataset, privacy_confirmed=args.privacy_confirmed)
    ).run(
        chunk_size=args.chunk_size,
        start_index=args.start_index,
        max_videos=args.max_videos,
        resume=args.resume,
    )
    print(f"Verified Kaggle backup: {completed} completed videos")


if __name__ == "__main__":
    main()
