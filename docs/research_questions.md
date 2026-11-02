# Research Questions

## Refined questions

### RQ1 — Controlled in-domain comparison

**How do EfficientNet-B0 and XceptionNet compare for binary deepfake classification on the official FaceForensics++ source-video-disjoint test split, and what residual identity-overlap risks remain, when training budget, preprocessing, augmentation, sampling, and decision thresholds are matched?**

Report frame-level and video-level ROC-AUC, PR-AUC, accuracy, precision, recall, F1, log loss, parameter count, and inference cost. This refinement prevents an architecture comparison from being confounded by different data or optimization protocols.

### RQ2 — Corruption robustness

**How does performance change as JPEG compression, Gaussian noise, blur, resizing, and selected combinations increase in severity, and does either architecture degrade more gracefully?**

Use fixed corruption definitions and severities applied only after the clean test preprocessing. Report absolute performance and change from the clean baseline, with confidence intervals. Robustness means measured degradation, not an assumption based on architecture size.

### RQ3 — Attention and explanation behavior

**What similarities and differences do Grad-CAM maps reveal in the facial regions used by EfficientNet-B0 and XceptionNet, and are the maps supported by deletion/occlusion faithfulness checks?**

Compare distributions of attention over face regions and representative examples across real, fake, clean, and corrupted inputs. Treat Grad-CAM as an explanatory diagnostic, not as proof that a model reasons causally from a particular artifact.

### RQ4 — External generalization

**How well do models trained on FaceForensics++ generalize to Celeb-DF v2 without fine-tuning, and how does the transfer gap differ between EfficientNet-B0 and XceptionNet?**

Use the same preprocessing and decision rule, report both model-specific and paired differences, and distinguish domain shift from corruption effects. No Celeb-DF media should enter training, model selection, threshold selection, or explanation-example selection.

## Alignment and interpretation

RQ1 supplies the fair baseline, RQ2 tests robustness, RQ3 tests whether differences are interpretable and faithful, and RQ4 tests external validity. The questions do not assert that either model is superior or that the study is novel; those are empirical conclusions to be earned by the completed analysis.
