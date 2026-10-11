# Run history: how the reported results were produced

This is the exact sequence that was run (October 9 to 10, 2026, RTX 5070 Laptop GPU). It is recorded
because the project improved its data in rounds, so a fresh end-to-end run of the final notebooks
reproduces the **final** pipeline, not every intermediate model.

## 1. Bounding boxes (`M2_Boxes.ipynb`)

| # | Step(s) | Outcome |
|---|---|---|
| 1 | `stage1_test` | 50 photos, all boxed; first speed 0.5 photos/s |
| 2 | (diagnosis) | Batch 16 exceeded GPU memory and spilled into system RAM; set OWLv2 batch to 4 (8 photos/s) |
| 3 | `stage1` | 6,936 photos boxed in 17 min; 1,366 (19.7%) flagged |
| 4 | `stage1b_test` | Fast visual-prompt path verified identical to the library method on 12 photos |
| 5 | review, mode `exemplar` | Hand-drawn example boxes for OBJ004, OBJ010, OBJ011, OBJ039, OBJ040, OBJ043, OBJ048 |
| 6 | `stage1b` | 6,936 photos in 17 min; text and visual agree on 47.9% |
| 7 | `sheets` | Visual-prompt boxes found to be mostly background; stage 2 changed to learn from all unflagged text boxes |
| 8 | `stage23_quick`, `stage23` | Self-trained YOLO26-s (val mAP@0.5 0.950); vote: 3,172 strong, 2,935 auto, 794 review, 35 review-label |
| 9 | review `verify` + `fix`, split `test` | All 1,041 test photos: 873 confirmed in the grid (10 wrong), 178 corrected |
| 10 | `master_strict` | Every test photo reviewed |
| 11 | review `fix`, OBJ040 + OBJ043, train + val | Classes without automatic boxes, boxed by hand |
| 12 | review `verify`, `-Sample 240`, split `train` | Random audit: 233 of 240 automatic training boxes correct (97.1%) |
| 13 | review `fix`, 8 classes losing > 20%, train + val | OBJ003, OBJ004, OBJ007, OBJ020, OBJ032, OBJ060, OBJ061, OBJ068 |
| 14 | review `verify -Redo` + `fix`, OBJ010, all splits | Rule: one box around both shoes |
| 15 | review `verify -Redo` + `fix`, OBJ051, all splits | Rule: the whole tissue box, including the tissue |
| 16 | review `verify -Redo` + `fix`, OBJ065, OBJ048, OBJ034, all splits | Watch case and strap; power bank only; paddle only |
| 17 | `master,master_strict` after each round | Final: 1,968 boxes by a person, 4,660 automatic, 314 photos left out (4.5%) |

## 2. Detection (`M2_Detect.ipynb`)

| # | Step(s) | Data version | Outcome |
|---|---|---|---|
| 1 | `compose_quick,check_quick,train_quick` | grids | Dry run passed (4.5 GB GPU memory at batch 4, imgsz 1120) |
| 2 | `compose,check` | grids (five sizes) | Check passed; every val/test photo placed once |
| 3 | `train_yolo26s` | grids | 60 epochs, 104 min |
| 4 | `train_yolov8s` | grids | Early stop at epoch 56, 70 min |
| 5 | `evaluate`, `detect` | grids | Post-processing added: duplicate removal, then one box per grid cell (chosen on validation) |
| 6 | `compose,check` | grids, OBJ010 corrected | |
| 7 | `finetune_yolo26s`, `finetune_yolov8s` | grids, OBJ010 corrected | 15 epochs each |
| 8 | `compose,check` | **final**: grids + 15% single photos + look-alike groups + all label corrections | Check passed for val, test, val_shapes, test_shapes |
| 9 | `finetune2_yolo26s` | final | 15 epochs from `yolo26s_ft` |
| 10 | `evaluate` | final | All five models compared on validation; **yolo26s_ft2 selected**; natural-size inference chosen on validation |
| 11 | `detect` | final | Object ID tables and drawn images for the test grids |

Every model in the results table was evaluated on the **same final** validation and test images in
step 10, so the comparison between models is fair. Rows 1 to 7 used earlier versions of
`M2_Detect.ipynb` (before single photos and look-alike groups were added); re-running `train_yolo26s`
with the final notebook trains on the final data.

## Reproducing the final model from scratch (about 3 hours of GPU time)

```powershell
# boxes (requires the human review rounds above; the review CSVs are the irreplaceable part)
.\scripts\run_step.ps1 -Notebook M2_Boxes.ipynb  -Steps "stage1,stage1b,stage23"
# ... review rounds ...
.\scripts\run_step.ps1 -Notebook M2_Boxes.ipynb  -Steps "master,master_strict"
# detection
.\scripts\run_step.ps1 -Notebook M2_Detect.ipynb -Steps "compose,check"
.\scripts\run_step.ps1 -Notebook M2_Detect.ipynb -Steps train_yolo26s
.\scripts\run_step.ps1 -Notebook M2_Detect.ipynb -Steps "finetune_yolo26s"
.\scripts\run_step.ps1 -Notebook M2_Detect.ipynb -Steps "finetune2_yolo26s"
.\scripts\run_step.ps1 -Notebook M2_Detect.ipynb -Steps "evaluate,detect"
```
