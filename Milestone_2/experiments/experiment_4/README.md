# Experiment 4 (Mukul, run C in the report): YOLO11s on the 38 SECTION-01 objects

`mukul_milestone2_yolo11.py` is a single local script (CUDA, Apple MPS or CPU) with five resumable steps. `m2_log.txt` is the full log of the reported run.

## Method

1. **Labels:** every photo is searched with **YOLO-World** and **YOLOE**, using plain-word prompts (`PROMPTS` in the script).
   * If the two boxes overlap (IoU ≥ 0.5), the box is accepted.
   * If only one detector is confident (≥ 0.4), its box is kept.
   * Unreliable photos get a second, name-free pass (YOLOE prompt-free plus a low-threshold YOLO-World).
2. **Rescue rule:** if more than 20% of photos are still unreliable, the most confident of them are kept, marked `kept_for_review`, until the TA's limit is met.
3. **Grids:** 2x2 to 5x5 of 224x224 tiles. No Object ID repeats in a grid. Light augmentation is applied to training tiles. The leakage check passed.

   | Split | Grid images | Objects |
   |---|---|---|
   | train | 950 | 11,350 |
   | val | 190 | 2,270 |
   | test | 190 | 2,270 |

4. **Training:** YOLO11s, imgsz 1120, batch 8, 40 epochs (cosine LR, mosaic off), on an Apple GPU (about 4 hours).
5. **Evaluation:** test mAP overall, per grid size and per Object ID, plus example detections.

### Why only 38 objects

A first run on **all 73 objects** left out up to 97% of the photos for some classes (`outputs/left_out_per_object_all73_earlier_run.csv`). The final run therefore uses the 38 SECTION-01 objects, which the TA allowed.

Labelling outcome for those 38 objects (3,605 photos):

| Outcome | Photos |
|---|---|
| Both detectors agree | 1,279 |
| Fallback detectors agree | 945 |
| One model only | 180 |
| Kept for review (rescue rule) | 480 |
| **Left out** | **721 (20.0%)** |

## Results (test, 190 grid images, 2,270 objects)

| Set | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|---|---|---|---|---|
| **Test** | **0.826** | **0.749** | **0.831** | **0.829** |
| 2x2 | 0.809 | 0.737 | 0.852 | 0.785 |
| 3x3 | 0.841 | 0.769 | 0.885 | 0.804 |
| 4x4 | 0.836 | 0.765 | 0.828 | 0.827 |
| 5x4 | 0.838 | 0.759 | 0.873 | 0.803 |
| 5x5 | 0.820 | 0.742 | 0.875 | 0.785 |

* **Weakest objects** (AP@0.5): table tennis racket 0.37, measuring bottle 0.46, mouth guard case 0.52, RGB projector 0.53, purse 0.53.
* Most of these also lost many photos during labelling, or kept low-confidence boxes under the rescue rule.

## Files

* `outputs/`:
  * box labels (all 73 and SECTION-01), class IDs and the grid summary;
  * photos left out per Object ID (final run and the earlier all-73 run);
  * test metrics, mAP by grid size and per-class AP;
  * the detection tables for the example test images.
* `figures/`:
  * example detections (prediction next to the true boxes), one per grid size;
  * label previews of the training grids;
  * the label review sheets;
  * the test PR curve and confusion matrix.
* `training/`: settings (`args.yaml`), per-epoch metrics (`results.csv`), curves, and the validation confusion matrix.

Not included: the cleaned photos, the grid dataset, the detector caches and the weights.
