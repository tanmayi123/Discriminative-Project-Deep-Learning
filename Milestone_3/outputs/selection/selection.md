# Model selection on validation

Rule (fixed before any boost2 result, SHA-256 C6A67BD9E576028736E7152B9561B609CC8E6824BD0981CAD87C084FDC4EF8C0, matches the recorded hash): the higher mAP@0.5:0.95 on the boost2 validation set wins; otherwise boost stays.

Validation set: 292 images, 3465 objects (pass-2 labels).

| model | imgsz | batch | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| yolo26s_boost | 1280 | 4 | 0.960 | **0.886** |
| yolo26s_boost2 | 1600 | 2 | 0.952 | 0.878 |

**Selected: yolo26s_boost.** Difference (boost2 minus boost) -0.0080; paired bootstrap over images (1000 resamples): 95% interval [-0.0142, -0.0024], boost2 higher in 0% of resamples. The interval is context only; the rule decides.

## Thresholds (selected model, same validation set)

| setting | threshold(s) | recall | precision | F1 | wrong IDs |
|---|---|---|---|---|---|
| one threshold (as evaluate) | 0.4 | 0.929 | 0.963 | 0.946 | 39 |
| grids + scenes (scene duplicate removal IoU 0.5) | 0.1 + 0.35 | 0.937 | 0.965 | 0.951 | 46 |

Grid/scene setting **adopted** (rule: F1 gain above 0.003 and recall not lower than with one threshold). Validation images: 136 grids, 156 scenes.

## Test sets with the adopted thresholds (one look)

Single-threshold test numbers for the same model are in boost2/evaluation.md. mAP does not depend on the threshold and is not repeated here.

| test set | threshold | images | objects | recall | precision | wrong IDs | fully correct | small-object recall |
|---|---|---|---|---|---|---|---|---|
| normal | 0.1 | 72 | 1048 | 0.949 | 0.970 | 10 | 0.472 | 0.896 |
| rotated | 0.1 | 216 | 3144 | 0.922 | 0.947 | 58 | 0.347 | 0.747 |
| singles | 0.1 | 1040 | 1048 | 0.951 | 0.968 | 11 | 0.951 | 0.896 |
| scattered | 0.35 | 104 | 1039 | 0.897 | 0.943 | 17 | 0.250 | 0.667 |
| cutout | 0.35 | 104 | 1014 | 0.881 | 0.914 | 61 | 0.288 | 0.590 |
