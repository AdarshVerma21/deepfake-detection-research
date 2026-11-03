# FaceForensics++ preprocessing

The preprocessing pipeline preserves the official FaceForensics++ c23
train/validation/test assignments and the research configuration of 224x224
face crops. It reads the dataset from the supplied input path and never writes
to the dataset directory.

## Kaggle command

Kaggle input datasets are read-only, and `/kaggle/working` is temporary. Store
the extracted FaceForensics++ data and official split JSON files in
`/kaggle/input/faceforensics`, then write generated artifacts to working
storage:

```bash
python scripts/prepare_faceforensics.py \
  --dataset-root /kaggle/input/faceforensics \
  --split-dir /kaggle/input/faceforensics/splits \
  --output-dir /kaggle/working/faceforensics_c23 \
  --progress-file /kaggle/working/faceforensics_c23-state/progress.json \
  --frames-per-video 10 \
  --crop-size 224 \
  --batch-size 100
```

The manifest is atomically replaced once per completed batch. Progress is
atomically replaced only after that manifest checkpoint succeeds. Crops made
by an interrupted batch may remain on disk, but they are not considered
complete until their manifest rows are checkpointed.

## Recovery

Save `/kaggle/working/faceforensics_c23` and the independent progress directory
as a Kaggle dataset, or keep them in the same session. Restore both before
running:

```bash
python scripts/prepare_faceforensics.py \
  --dataset-root /kaggle/input/faceforensics \
  --split-dir /kaggle/input/faceforensics/splits \
  --output-dir /kaggle/working/faceforensics_c23 \
  --progress-file /kaggle/working/faceforensics_c23-state/progress.json \
  --frames-per-video 10 \
  --crop-size 224 \
  --batch-size 100 \
  --resume
```

Resume validation requires matching preprocessing configuration. Videos are
skipped only when their manifest rows exist and every referenced crop file is
present. Missing progress is rebuilt from those records; malformed progress is
ignored and safely rewritten. A missing crop, failed video, or incomplete batch
is retried. Do not upload datasets automatically or provide Kaggle credentials
to this command.
