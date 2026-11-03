"""Command-line entrypoint for FaceForensics++ c23 preparation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.faceforensics import FaceForensicsPipeline, FaceForensicsPipelineConfig, SUPPORTED_CATEGORIES
from src.data.manifest import prepare_split_manifests


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare FaceForensics++ c23 face crops and a CSV manifest.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/faceforensics_c23"))
    parser.add_argument("--split-dir", type=Path)
    parser.add_argument("--frames-per-video", type=int, default=5)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--detector", choices=["haar"], default="haar")
    parser.add_argument("--detector-config", type=Path)
    parser.add_argument("--sampling-strategy", choices=["uniform", "random"], default="uniform")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--categories", nargs="+", choices=sorted(SUPPORTED_CATEGORIES), default=list(SUPPORTED_CATEGORIES))
    parser.add_argument("--pilot", action="store_true", help="Process at most 10 videos and 5 frames per video.")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--resume", action="store_true", help="Resume from the progress file and skip completed videos.")
    parser.add_argument("--progress-file", type=Path, help="Path to the resumable progress JSON file.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    frames = 5 if args.pilot else args.frames_per_video
    config = FaceForensicsPipelineConfig(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        split_dir=args.split_dir,
        frames_per_video=frames,
        crop_size=args.crop_size,
        detector=args.detector,
        detector_config=str(args.detector_config) if args.detector_config else None,
        seed=args.seed,
        categories=tuple(args.categories),
        sampling_strategy=args.sampling_strategy,
    )
    report = FaceForensicsPipeline(config).run(
        limit_videos=10 if args.pilot else None,
        batch_size=args.batch_size,
        resume=args.resume,
        progress_file=args.progress_file,
    )
    split_report = prepare_split_manifests(report.manifest_path, args.output_dir)
    print(f"Manifest: {report.manifest_path}")
    print(f"Statistics: {report.statistics_path}")
    print(f"Split CSVs: {', '.join(str(path) for path in split_report.split_paths.values())}")
    print(f"Videos: {report.videos_discovered}, samples: {report.samples_written}, failures: {report.failures}")


if __name__ == "__main__":
    main()
