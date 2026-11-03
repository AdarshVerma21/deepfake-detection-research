# FaceForensics++ preprocessing

The preprocessing preserves the official FaceForensics++ c23
train/validation/test assignments, uniform sampling, seed 42, Haar detection,
10 frames per video, and 224x224 crops. The raw dataset is
`adarshverma21/faceforensics-c23-research`; the processed checkpoint dataset is
`adarshverma21/faceforensics-c23-processed-checkpoints`. Both must remain
private.

## Exact Kaggle notebook instructions

1. Attach the raw private dataset to the notebook. Make the extracted videos
   and official split JSON files available under `/kaggle/input/faceforensics`.
2. Upload/clone this repository into `/kaggle/working/repository`, then run:

```bash
cd /kaggle/working/repository
pip install -r requirements.txt
python scripts/run_kaggle_preprocessing.py \
  --dataset-root /kaggle/input/faceforensics \
  --split-dir /kaggle/input/faceforensics/splits \
  --output-dir /kaggle/working/faceforensics_c23 \
  --kaggle-dataset adarshverma21/faceforensics-c23-processed-checkpoints \
  --frames-per-video 10 \
  --crop-size 224 \
  --chunk-size 100
```

The workflow processes deterministic discovery-order ranges, atomically saves
the manifest and progress state, packages all crops without credentials, and
uploads a complete new private Kaggle version after each chunk. It verifies
both the upload command and dataset status before processing another chunk.
It never passes `--public`.

3. If the session is interrupted, start a new session with the same arguments
   and add `--resume`:

```bash
cd /kaggle/working/repository
python scripts/run_kaggle_preprocessing.py \
  --dataset-root /kaggle/input/faceforensics \
  --split-dir /kaggle/input/faceforensics/splits \
  --output-dir /kaggle/working/faceforensics_c23 \
  --kaggle-dataset adarshverma21/faceforensics-c23-processed-checkpoints \
  --frames-per-video 10 --crop-size 224 --chunk-size 100 --resume
```

Resume downloads the latest verified checkpoint, validates `manifest.csv` and
every referenced crop, and resumes at the first incomplete video. A failed
upload or failed verification exits before the next chunk; rerun with
`--resume` after fixing Kaggle access.

Use `--start-index N` and `--max-videos M` for a bounded run. Indices are
zero-based in sorted discovery order. The lower-level
`scripts/prepare_faceforensics.py` command supports these range controls for
local preprocessing but does not upload backups.

## Local preprocessing

```bash
python scripts/prepare_faceforensics.py \
  --dataset-root /path/to/FaceForensics \
  --split-dir /path/to/FaceForensics/splits \
  --output-dir data/processed/faceforensics_c23 \
  --frames-per-video 10 --crop-size 224 --batch-size 100
```

The output directory is never written into the raw dataset. Split manifests
are generated only after the combined manifest and referenced crops validate.
