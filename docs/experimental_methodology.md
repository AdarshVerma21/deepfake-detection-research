# Experimental Methodology

## 1. Dataset selection and access

Request access through the official FaceForensics++ and Celeb-DF repositories and comply with their licenses, terms, and attribution requirements. Do not download data as part of this documentation task. Record dataset release/version, manipulation types, video counts, frame extraction settings, and checksum or manifest information in the experiment artifact.

Use FaceForensics++ as the development benchmark and Celeb-DF v2 only as an external, no-fine-tuning test. The official FaceForensics++ split commonly used by the dataset release contains 720 training, 140 validation, and 140 test source-video identifiers in separate JSON files. These files establish source-video/sequence disjointness, but do not provide human-identity labels and therefore do not establish identity-disjointness. Confirm the exact split files from the downloaded official release before implementation rather than reconstructing them.

## 2. Leakage prevention

- Split at source-video/sequence level before extracting frames, using the official JSON files.
- Keep all frames from one source video in one split.
- Keep manipulated derivatives with their source assignment.
- Treat possible recurrence of the same person across distinct source videos as a residual leakage risk; quantify or discuss it only if reliable identity metadata can be obtained.
- Fit normalization, thresholds, and any calibration only on training/validation data.
- Do not use Celeb-DF for model selection, early stopping, threshold selection, or choosing illustrative examples.
- Maintain immutable manifests containing source video ID, manipulation, split, frame ID, and preprocessing version.

This prevents highly correlated frames from crossing splits, but it does not prove that human identities cannot recur across source videos.

## 3. Face preprocessing

Use one documented face detector and one crop policy for both architectures.
The current default is OpenCV's Haar cascade: it is CPU-friendly but can miss
faces or produce false positives under pose, occlusion, lighting, and
compression, so failed detections are logged and excluded. Resize accepted
crops to the configured 224 x 224 input and apply the same channel
normalization. Avoid test-time face-detector tuning. Report whether multiple
faces are retained, ranked, or excluded.

The combined manifest retains `video_id`, `source_video_id`, manipulation,
numeric label, official split, frame path/index, original video path, and
video metadata. Split CSVs are generated without random resplitting. Paths are
resolved relative to the preprocessing output directory, and missing images,
invalid labels, unknown splits, and video/source-video split leakage are
rejected.

## 4. Model initialization and fair training

Instantiate the official EfficientNet-B0 and XceptionNet implementations with the same two-class head, input size, augmentation policy, optimizer family, learning-rate schedule, batch-size policy, epoch budget, early-stopping rule, and random-seed protocol. Initialize from ImageNet weights only if the choice is applied consistently and documented; include the pretrained checkpoint source and classifier replacement. Record parameter count, trainable parameter count, and approximate inference cost.

Use the validation split only for model selection and checkpoint selection. Repeat the final training protocol across several pre-specified seeds if compute permits. Preserve all configurations, logs, checkpoints, and code version identifiers.

## 5. Evaluation units and metrics

Distinguish three units explicitly:

- **Image-level:** one preprocessed face crop treated as one image observation.
- **Frame-level:** one face crop extracted from one video frame; this is an image-level prediction with a video-frame provenance.
- **Video-level:** one prediction obtained by aggregating frame probabilities within a source video.

Image-level and frame-level metrics are reported separately from video-level
metrics. The Stage 6 test CLI reports frame-level metrics for the untouched
official test CSV; it does not aggregate frames into video predictions. Any
future video-level analysis must define and report its aggregation rule because
frames from the same video are correlated.

Report frame-level and video-level metrics separately:

- ROC-AUC and PR-AUC;
- accuracy, balanced accuracy, precision, recall, and F1;
- log loss and confusion matrices;
- per-video aggregation using a pre-registered rule such as mean frame probability, with a sensitivity analysis for median and top-k mean;
- calibration summaries where feasible.

Use video-level bootstrap confidence intervals as the primary uncertainty estimate because frames within a video are not independent. For paired model comparisons, bootstrap the difference per video and report a two-sided interval; use a paired permutation or DeLong-style comparison only when its assumptions match the metric and sampling unit.

## 6. Robustness experiments

Apply corruption after face preprocessing or at a clearly specified earlier stage, but use the same stage for both models. Evaluate at least JPEG quality levels, Gaussian noise levels, Gaussian/motion blur levels, resizing/downsampling, and selected combinations. Keep corruption seeds fixed across models. Report each severity, absolute metric, clean-to-corrupted change, and an aggregate robustness curve or area-under-degradation measure.

The corruption test set must remain separate from training and validation. If corruption augmentation is later introduced, it becomes a separate experiment and must not be mixed with the baseline comparison.

## 7. Cross-dataset evaluation

Train and select models using FaceForensics++ only. Freeze weights and threshold policy, then evaluate once on the selected Celeb-DF v2 test partition without fine-tuning. Report the external metric and the transfer gap relative to the in-domain test result. Explain differences in source identities, manipulation process, resolution, compression, and label composition; do not describe the gap as architecture failure without ruling out domain shift.

## 8. Explainability

Generate Grad-CAM maps from the last suitable convolutional feature layer for correctly and incorrectly classified real/fake examples, including clean and corrupted cases. Use identical visualization scaling and face-coordinate mapping. Quantify deletion/occlusion faithfulness by masking high-importance versus low-importance regions and measuring confidence change. Include failure cases and explicitly state that Grad-CAM indicates sensitivity, not causal proof.

## 9. Reproducibility and reporting

Publish or preserve the exact configuration, seed list, package/environment lock information, code revision, dataset manifests (not restricted media), preprocessing implementation, split files or references, corruption parameters, aggregation code, and metric scripts. Report missing detections and exclusions rather than silently dropping them. Do not report results until the protocol has been executed and artifacts are retained.
