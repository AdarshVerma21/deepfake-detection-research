# Research Gap

## Established knowledge

The literature establishes that Xception is a strong face-forgery baseline [1], that curated in-domain performance can be much higher than performance under compression or dataset shift [1–6, 9], and that FaceForensics++ and Celeb-DF v2 measure different aspects of difficulty [1, 2]. EfficientNet provides a principled accuracy/efficiency architecture, but its original evidence is not forensic evidence [11]. Grad-CAM can expose model-sensitive regions, but visual plausibility alone does not establish explanation faithfulness [12].

## Defensible gap

A useful, bounded gap is the lack of a consistently controlled comparison of **EfficientNet-B0 and XceptionNet** in which:

- both models use the same face preprocessing, official FaceForensics++ source-video-disjoint split, augmentation, optimizer, budget, and frame sampling; human-identity disjointness is not claimed because it is not established by the released split JSON files;
- clean performance is separated from controlled corruption performance;
- frame-level and video-level aggregation are reported separately;
- Grad-CAM is compared with quantitative occlusion/deletion checks rather than only example images; and
- the same trained models are evaluated without fine-tuning on Celeb-DF v2.

This is a proposed study design, not a claim that no prior paper has ever performed any one of these analyses. A systematic search of the exact protocol and implementation versions would be needed before making a stronger novelty claim.

## Questions that remain open

1. Does EfficientNet-B0 offer a meaningful accuracy/efficiency trade-off relative to Xception under a matched training budget?
2. Are either model's apparent cues stable under JPEG compression, blur, noise, and resizing?
3. Do the two models attend to different facial regions, and are those regions predictive or merely correlated with dataset artifacts?
4. Does strong FF++ performance transfer to Celeb-DF v2 without fine-tuning?
5. Does temporal aggregation improve reliability, or merely hide unstable frame predictions?

## Evidence needed for a defensible contribution

The project should report exact data versions and split files, parameter counts and compute budget, training curves, per-video rather than only per-frame confidence intervals, corruption severity curves, external-transfer results, and pre-specified statistical comparisons. Explainability claims should include Grad-CAM visualizations, deletion/occlusion measurements, representative successes and failures, and a statement of what Grad-CAM cannot prove. Novelty should be stated only after comparing the completed protocol and results against the closest prior experiments.
