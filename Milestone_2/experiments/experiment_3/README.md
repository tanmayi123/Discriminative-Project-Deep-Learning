# Experiment 3 (Praniti): YOLOE-26s fine-tuned on 72 objects

`Milestone2_YOLOE2026s.ipynb` runs in **Google Colab**, with its outputs saved in the notebook.

## Method

* **Model:** YOLOE-26s, Ultralytics' open-vocabulary detector built on YOLO26. It is fine-tuned as a detector with the prompt-embedding recipe (`YOLOEPETrainer`): the object names initialise the classification head, then the whole network trains on our boxes.
* **Objects:** 72. OBJ054 is left out entirely.
* **Boxes:** YOLO-World and YOLOE each search every photo using the object's name (`prompts_used.csv`). A box is accepted when the two detectors agree; photos where only one found the object, the two disagree, nothing was found, or the box is too small are left out.

  | Labelling outcome | Photos | Share |
  |---|---|---|
  | Accepted | 3,664 | 53.6% |
  | Accepted (single photo) | 124 | 1.8% |
  | Left out: one model only | 1,627 | 23.8% |
  | Left out: no detection | 903 | 13.2% |
  | Left out: models disagree | 462 | 6.8% |
  | Left out: too small | 62 | 0.9% |

  In total, **3,788 of 6,842 photos are kept and 44.6% are left out**, above the TA's 20% limit. Some classes kept only 4 to 9 of 95 photos (watch, perfume bottle, swimming goggles, iWatch); see `labelling_per_class.csv`.
* **Images:** single photos plus 2x2 to 5x5 grids (`grids_manifest.csv`).
* **Checks:** a background audit, a box-size check, OBJ004 background renames, suspicious labels, and test failure analysis.
* **Threshold:** confidence threshold 0.15, chosen on validation.

## Results (test)

| Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|
| 0.920 | 0.924 | 0.947 | 0.896 |

| Image type | Images | Objects | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|---|---|
| single photo | 60 | 53 | 0.666 | 0.789 | 0.826 | 0.672 |
| grid 2x2 | 23 | 73 | 0.905 | 0.890 | 0.946 | 0.914 |
| grid 3x3 | 48 | 349 | 0.906 | 0.900 | 0.957 | 0.922 |
| grid 4x4 | 33 | 412 | 0.926 | 0.904 | 0.961 | 0.923 |
| grid 5x4 | 21 | 383 | 0.944 | 0.906 | 0.963 | 0.903 |
| grid 5x5 | 35 | 772 | 0.926 | 0.915 | 0.951 | 0.903 |

Other results:
* Robustness (test mAP@0.5): low quality 0.849, small objects 0.861, blur 0.937.
* Object recall 0.918; ID accuracy of found objects 0.991.
* 45% of multi-object test images are fully correct.

**Caveat:** the test images are built only from photos that received an automatic box, so the hardest photos (the 44.6% left out) are not tested. Together with the 44.6% left out, this is why these numbers are not directly comparable with the final pipeline.

## Files

* `outputs/`: every CSV and JSON the notebook wrote (labels, splits, per-class results, metrics, model selection), plus:
  * `figures/`: training curves, confusion matrix, label checks, robustness, size vs accuracy, failures, demos.
  * `training/`: settings and per-epoch results of the run.
  * `eval/`: PR curves and confusion matrices for validation, test, and each test subset.
* Not included: the grid images (`yolo_grids_full.zip`), the demo photos and the weights.
