# Discriminative Project Deep Learning: IE7615 Milestone 2 (Group 7)

**Task:** multi-object detection. Build images that contain many objects by placing the course's single-object photos side by side (grids of 224x224 photos), custom-train a YOLO detector, and output the **Object ID and location (bounding box) of every object** in an image.

**Group members:** Mukul Sridar Haladhi, Praniti Sunil Kale, Santhoshkumar Chandrasekar, Tanmayi Shurpali

**Final model:** YOLO26-s, fine-tuned twice on grids of all 73 objects (`final_pipeline/`). It was selected on validation data only, then tested once.

| Test set (every test box checked by a person) | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|---|---|---|---|---|
| **YOLO26-s, final (`yolo26s_ft2`)** | **0.947** | **0.875** | **0.931** | **0.918** |
| YOLOv8-s, fine-tuned once (`yolov8s_ft`) | 0.932 | 0.859 | 0.925 | 0.894 |

Single photos (1x1): 14 of 14 test images fully correct. Full method and results: [the report](report/IE7615_Milestone2_Report_Group7.pdf).

---

## Repository structure

```
Milestone_2/
├── README.md
├── requirements.txt
├── final_pipeline/                          # the submitted pipeline (all results in the report)
│   ├── M2_Boxes.ipynb                       # bounding boxes: text prompt, visual prompt, self-trained detector, vote, human review
│   ├── M2_Detect.ipynb                      # grids, leakage checks, YOLO training and fine-tuning, evaluation, detect()
│   ├── README.md, RUN_HISTORY.md, requirements.txt
│   ├── scripts/run_step.ps1                 # runs any notebook step headlessly (Windows)
│   ├── scripts/m2_cross_eval.py             # cross-evaluation of all pipelines on one human-checked test set (report Table 14)
│   ├── app/m2_eye_app.py                    # "The EYE": local demo app (not hosted); app/earlier_demos/ holds two earlier versions
│   └── outputs/
│       ├── results/                         # results.json, results_summary.md, per-class AP, task metrics, left out per Object ID
│       ├── labels/                          # every box file, incl. review/review_SC.csv (the human review, irreplaceable)
│       ├── training/yolo26s_ft2/            # per-epoch metrics, settings, curves of the final model
│       ├── evaluation/yolo26s_ft2_test/     # PR / F1 curves and confusion matrices on the test set
│       ├── figures/                         # example detections, one per grid size
│       ├── dataset/                         # data.yaml and manifest.csv of the grid dataset (no images)
│       └── logs/                            # a log of every step that was run
├── experiments/
│   ├── experiment_1/                        # YOLOv8s, all 73 objects, two detectors + Milestone 1 classifier (Colab)
│   ├── experiment_2/                        # model comparison runs behind the final pipeline (YOLO26-s vs YOLOv8-s)
│   ├── experiment_3/                        # YOLOE-26s fine-tuned, 72 objects (Colab)
│   └── experiment_4/                        # YOLO11s, the 38 SECTION-01 objects (local, Apple GPU)
└── report/
    └── IE7615_Milestone2_Report_Group7.pdf
```

## What the assignment asks, and where it is

| Requirement | Where |
|---|---|
| At least 100 multi-object images per student, made by placing single-object photos side by side | `final_pipeline/M2_Detect.ipynb` (compose step): 1,500 training images (1,275 grids, equal share of the five sizes, plus 225 single photos), 68 validation and 72 test grids, every validation and test photo used exactly once (`outputs/dataset/manifest.csv`) |
| Custom-train YOLO | `final_pipeline/M2_Detect.ipynb` (train and fine-tune steps), `outputs/training/` |
| Output the Object IDs and locations of all objects in an image | `detect` step of `M2_Detect.ipynb`, `outputs/results/predicted_detections.csv`, `outputs/figures/` |
| Object IDs and grid sizes used | all 73 Object IDs; grids 2x2, 3x3, 4x4, 5x4, 5x5 (448x448 to 1120x1120), plus single photos |
| How labels were made and checked | `M2_Boxes.ipynb`; every test box checked by a person; random audit of training boxes (see below) |
| Photos left out, per Object ID with the reason | `outputs/results/left_out_per_object.csv` |
| mAP@0.5, precision, recall, example detections | `outputs/results/results_summary.md`, `outputs/evaluation/`, `outputs/figures/` |

## Final pipeline (what it does)

1. **Data:** reuses the Milestone 1 cleaning and split (duplicates removed before the split, background-only photos separated, `OBJ054_017` excluded, non-224x224 photos resized). No source photo appears in more than one split.
2. **Bounding boxes** (`M2_Boxes.ipynb`), one box per photo:
   1. **Text prompt:** OWLv2 finds the object from its name in the course Object List.
   2. **Visual prompt:** OWLv2 image-guided search, using example crops of the same object.
   3. **Self-trained detector:** a small YOLO trained on the training split only, on boxes where the methods agree, then run on every photo.
   4. **Vote** between the three sources (strong, auto, or needs review).
   5. **Human review** in a local tool: every one of the 1,041 test photos was checked (885 confirmed, 156 corrected); hard classes were boxed by hand; a random audit of 240 automatic training boxes found 233 correct (97.1%). In total 1,968 boxes come from a person and 4,660 are automatic.
   6. **Left out:** 314 of 6,942 photos (4.5%, TA limit 20%): 308 without a reliable box, 6 not a photo of the object. Per Object ID in `left_out_per_object.csv`; the highest is OBJ059 (comb), 20.0%.
3. **Grids** (`M2_Detect.ipynb`): the five course grid sizes at native 224 px per tile, plus 15% single photos in training. Visually similar objects (watches, wallets, power banks, rackets, sunglasses, bottles, ...) are placed together on purpose in half of the training grids so the model learns to tell them apart. 20% of training tiles get one real-photo problem (blur, low resolution, compression, darkness).
4. **Training:** COCO-pretrained YOLO26-s and YOLOv8-s at imgsz 1120, then short fine-tuning rounds after label corrections. All five runs are compared on validation; `yolo26s_ft2` has the highest validation mAP@0.5:0.95 (0.930) and is tested once.
5. **Output:** `detect` gives the Object ID, confidence and box of every object in any image or folder (CSV, JSON and drawn images).

**Task-level results on the test grids** (confidence threshold and post-processing chosen on validation): object recall 0.918, precision 0.943; of the 1,041 objects in the 72 test grids (every test photo used once), 16 got the wrong ID and 69 were missed. With the optional "one box per 224x224 grid cell" post-processing (it uses the fact that each grid cell holds one photo, so it applies to grid images only), 34.7% of test grids are fully correct; without it, 29.2%. mAP, precision and recall above are on the raw model output.

**Weakest Object IDs (test AP@0.5):** OBJ034 table tennis racket (0.59), OBJ061 card wallet (0.62), OBJ014 sunglasses (0.79), OBJ065 analog watch (0.81).

## Experiments

Each member also built a full pipeline on their own. They differ in objects used, labelling method and model. The report (Section 9) calls the final pipeline's runs **runs 1 to 5** (`experiment_2` plus the selected model) and the three alternative pipelines **runs A, B and C**.

The scores in the table below are each pipeline's **own** test set, labelled automatically by that pipeline, so they are not comparable with each other or with the final pipeline (only `experiment_2` uses the final, human-checked test set).

| | Who | Objects | Boxes made by | Photos left out | Model | Test mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|---|---|---|---|---|---|---|---|---|---|
| [experiment_1](experiments/experiment_1/) (run A) | Tanmayi | 73 | Grounding DINO + YOLO-World, Milestone 1 ConvNeXt classifier as tie-break | 7.9% (9 objects above 20%) | YOLOv8s | 0.919 | 0.852 | 0.909 | 0.905 |
| [experiment_2](experiments/experiment_2/) (runs 1 to 4) | Santhosh | 73 | as final pipeline | 4.5% | YOLO26-s / YOLOv8-s comparison | 0.910 to 0.932 | 0.846 to 0.863 | | |
| [experiment_3](experiments/experiment_3/) (run B) | Praniti | 72 | YOLO-World + YOLOE | 44.6% (20.9% no box or disagreement, 23.8% found by one detector only) | YOLOE-26s | 0.947 | 0.896 | 0.920 | 0.924 |
| [experiment_4](experiments/experiment_4/) (run C) | Mukul | 38 (SECTION-01) | YOLO-World + YOLOE, name-free fallback | 20.0% (16 objects above 20%) | YOLO11s | 0.826 | 0.749 | 0.831 | 0.829 |

**Cross-evaluation on the same human-checked test objects** (report, Table 14). Every alternative was scored with the same code on the final test set, after removing objects of classes it cannot predict and objects whose photo it had trained on; the selected model was scored on exactly the same objects:

| Run | Objects | mAP@0.5 | mAP@0.5:0.95 | Selected model, same objects: mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|---|
| A (experiment_1) | 1,019 | 0.883 | 0.782 | 0.944 | 0.872 |
| B (experiment_3) | 628 | 0.779 | 0.697 | 0.923 | 0.842 |
| C (experiment_4) | 543 | 0.819 | 0.720 | 0.949 | 0.872 |

What the experiments taught us, and what went into the final pipeline:
* **Automatic boxes are the bottleneck.** Open-vocabulary detectors miss small, thin or unusual objects (watches, rackets, perfume bottles, goggles). Experiment 3 had to leave out 45% of photos and experiment 4 first tried all 73 objects but left out 60 to 97% of some classes, then moved to the 38 SECTION-01 objects. The final pipeline adds a visual prompt, a self-trained detector and human review, and leaves out 4.5%.
* **Once found, objects are almost always named correctly.** In experiment 1 only 17 of 2,688 test objects (0.6%) got the wrong ID; 7.9% were missed. Missed small objects are the main error in every pipeline.
* **Grid size matters little.** mAP is about the same from 2x2 to 5x5 in every experiment.
* **YOLO26-s and YOLOv8-s are close** (experiment 2); better labels and fine-tuning gave more than the choice of architecture.

## How to run

**Final pipeline** (local machine with an NVIDIA GPU; about 3 hours of GPU time plus the human review): see [`final_pipeline/README.md`](final_pipeline/README.md) and [`final_pipeline/RUN_HISTORY.md`](final_pipeline/RUN_HISTORY.md).

**Experiments:**
* `experiment_1`: Google Colab (T4). Put the dataset zip and the Object List sheet in Google Drive, set the paths in the settings cell, Run all. Labelling about 45 minutes, training about 2 hours.
* `experiment_3`: Google Colab; paths in the first cells.
* `experiment_4`: local Python script (CUDA, Apple MPS or CPU); set `PROJECT` at the top, then `python mukul_milestone2_yolo11.py`.

Install the dependencies with `pip install -r requirements.txt`.

## Not included in this repository

* **The dataset and all generated images** (grid images, label files, photo sheets). They are the course's student-collected data, and the TAs asked us to push code only.
* **Trained model weights (`.pt`).** Running the notebooks recreates them.
* **Privacy.** `OBJ054_017` (a phone screenshot with personal contact details) is excluded from every pipeline and appears in no figure.
