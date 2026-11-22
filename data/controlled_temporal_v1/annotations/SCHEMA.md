# Annotation schema
`manifest.json` stores sequence-oriented annotations. Boxes are pixel-coordinate `[x1,y1,x2,y2]`. Each object has a persistent generated `object_id`, class, and visibility. `expected_events.json` records intended scenario events; these are ground-truth scenario labels, not model predictions.
