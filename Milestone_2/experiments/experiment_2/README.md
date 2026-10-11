# Experiment 2 (Santhosh): model comparison behind the final pipeline

These are the training runs that the final pipeline compared before choosing its model. The code that produced them is the final pipeline itself ([`../../final_pipeline/`](../../final_pipeline/)); this folder keeps the runs that were **not** selected, so the final pipeline folder only holds the chosen model. The order of the runs is in [`RUN_HISTORY.md`](../../final_pipeline/RUN_HISTORY.md).

All runs: COCO-pretrained weights, imgsz 1120, the same grid design and box files. Every model was evaluated on the **same final** validation and test images, so this comparison is fair.

| Run | What it is | Val mAP@0.5 | Val mAP@0.5:0.95 | Test mAP@0.5 | Test mAP@0.5:0.95 | Test precision | Test recall |
|---|---|---|---|---|---|---|---|
| `yolo26s` | YOLO26-s, 60 epochs | 0.943 | 0.907 | 0.921 | 0.858 | 0.922 | 0.884 |
| `yolov8s` | YOLOv8-s, early stop at epoch 56 | 0.940 | 0.897 | 0.910 | 0.846 | 0.914 | 0.890 |
| `yolo26s_ft` | YOLO26-s, +15 epochs after label corrections | 0.956 | 0.919 | 0.931 | 0.863 | 0.922 | 0.910 |
| `yolov8s_ft` | YOLOv8-s, +15 epochs after label corrections | 0.952 | 0.906 | 0.932 | 0.859 | 0.925 | 0.894 |
| `yolo26s_ft2` (**selected**, in `final_pipeline/`) | YOLO26-s, +15 more epochs on the final data | 0.972 | 0.930 | 0.947 | 0.875 | 0.931 | 0.918 |

Selection rule: highest **validation** mAP@0.5:0.95; the test set was only used to report.

Findings:
* YOLO26-s and YOLOv8-s are within about one point of each other at every stage.
* Each round of label correction and fine-tuning added more than the choice of architecture.

`training/selftrain_labelling_detector/` is the small detector trained in `M2_Boxes.ipynb` (stage 2) to propose boxes for labelling; it is not a submission model.

## Files

* `training/<run>/`: `args.yaml` (settings), `results.csv` (per-epoch metrics), `results.png` (curves), PR / F1 curves and confusion matrices on validation.
* `evaluation/<run>_test/`: PR / F1 curves and confusion matrices on the test set.

No weights and no images are included.
