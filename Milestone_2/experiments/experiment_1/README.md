# Experiment 1 (Tanmayi): YOLOv8s on all 73 objects, two detectors checked by the Milestone 1 classifier

`Group7_IE7615_Milestone2_YOLOv8.ipynb` runs end to end in **Google Colab** (T4 GPU), with every output saved in the notebook.

## Method

1. **Cleaning:** 7,343 files indexed; `OBJ054_017` removed (TA); 16 exact duplicates removed by MD5 before the split; 5 fake OBJ065 backgrounds (byte-identical to object photos) dropped; the five untagged OBJ004 backgrounds treated as backgrounds; 2 photos not 224x224 resized. Result: 6,936 object photos and 376 background photos.
2. **Split:** stratified 70/15/15 by Object ID (seed 42) before anything else; 4,855 / 1,040 / 1,041 photos. No source photo appears in two splits (checked).
3. **Boxes:** every photo is searched with **Grounding DINO** (tiny) and **YOLO-World** (v2 s), prompted with the object's name from the course Object List (brand names and typos replaced by plain words, e.g. "Marshall Speaker" to "speaker").
   * both boxes overlap (IoU ≥ 0.5): the average box is used (2,903 photos);
   * otherwise the **Milestone 1 ConvNeXt-Tiny classifier** scores a crop of each box and keeps the one it is at least 40% sure shows the right object (Grounding DINO 3,319, YOLO-World 166);
   * otherwise the photo is left out, with the reason. **548 of 6,936 left out (7.9%)**; per Object ID in `outputs/excluded_per_object.csv`.
   * When both detectors found a box they agreed on 88.5% of photos. A random sample of 48 accepted boxes is in `outputs/figures/label_check_accepted.png`.
4. **Grids:** 2x2, 3x3, 4x4, 5x4, 5x5 tiles of 224x224; 300 grid images per member (1,200 in total: 800 train, 200 val, 200 test); no Object ID twice in a grid; about 10% background tiles; train-only tile augmentation (flip, colour, blur or low resolution).
5. **Training:** YOLOv8s, imgsz 1120, batch 8, 80 epochs, checkpoints on Google Drive with automatic resume.
6. **Evaluation:** standard metrics overall and per grid size; task metrics at a confidence threshold chosen on validation (0.3); an error breakdown for every test object.

## Results (test, 200 grid images, 2,688 objects)

| Set | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| Validation | 0.905 | 0.888 | 0.907 | 0.847 |
| **Test** | **0.909** | **0.905** | **0.919** | **0.852** |
| Test 2x2 | 0.874 | 0.862 | 0.913 | 0.836 |
| Test 3x3 | 0.875 | 0.861 | 0.903 | 0.843 |
| Test 4x4 | 0.937 | 0.902 | 0.935 | 0.882 |
| Test 5x4 | 0.886 | 0.901 | 0.921 | 0.853 |
| Test 5x5 | 0.908 | 0.897 | 0.926 | 0.859 |

| What happened to the 2,688 test objects | Objects | Share |
|---|---|---|
| Correct (right ID, good box) | 2,404 | 89.4% |
| Missed (no box at all) | 213 | 7.9% |
| Right ID, loose box | 54 | 2.0% |
| Wrong ID (misclassified) | 17 | 0.6% |

* Every one of the 73 Object IDs was found correctly at least sometimes; 13 have recall below 0.8.
* Extra boxes: 88 false alarms (mostly on background tiles) and 44 duplicates (e.g. one box per sneaker).
* Most common wrong IDs: mobile phone predicted as toy car (4), power bank as computer mouse (3).
* Weakest objects: power bank, analog watch, mouth guard case, card wallet, table tennis racket. They are mostly **missed**, not confused.

## Files

| File | Contents |
|---|---|
| `outputs/cleaning_log.csv` | what was removed and why |
| `outputs/excluded_per_object.csv` | photos left out per Object ID, with the reason |
| `outputs/metrics_overall_and_per_grid.csv` | precision, recall, mAP on validation, test, and test per grid size |
| `outputs/task_metrics_test.csv` | object recall, detection precision, fully correct images |
| `outputs/per_class_ap50_test.csv` | AP@0.5 per Object ID |
| `outputs/error_breakdown_test.csv`, `outcome_table_test.csv`, `confused_pairs_test.csv` | error analysis |
| `outputs/figures/` | label checks, grid examples, training curves, PR curve, confusion matrix, example detections per grid size |

Not included: the labelled box file and grid dataset (they come from the course images; the notebook rebuilds them) and the weights.
