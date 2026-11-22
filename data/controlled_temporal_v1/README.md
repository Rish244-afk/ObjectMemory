# ObjectMemory Controlled Temporal Benchmark v1.0

This generated dataset contains 6 sequences × 5 visits = 30 images. Scenarios cover lighting-only changes, cup movement and return-to-baseline, partial book occlusion, bottle removal, persistent phone movement, and small global camera-like translations.

## Contents
- `images/`: generated PNGs.
- `annotations/manifest.json`: sequences, visits, object IDs, bounding boxes, visibility and nuisance tags.
- `annotations/object_annotations.csv`: flattened object-level labels.
- `annotations/expected_events.json`: expected temporal events.
- `metadata/dataset_card.json`: provenance and limitations.

## Use correctly
Use this for integration and controlled temporal-reasoning tests. The illustrations are synthetic, not photographs, and may not be detected well by pretrained YOLO models. Do not use this dataset alone to claim real-world detector accuracy or superiority of YOLOv8n vs YOLO-World. For detector comparison, use a separate licensed, annotated natural-image test set. Ground-truth boxes are renderer-derived, not manually verified.
