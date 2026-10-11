# Milestone 2: Object Detection with YOLO

Detects every object in an image built from the course dataset (single photos and grids of 2x2, 3x3,
4x4, 5x4 and 5x5 photos) and reports its **Object ID** and **bounding box**.

**Final model:** YOLO26-s, fine-tuned twice. **Test results** (all test boxes verified by a person):
mAP@0.5 = 0.947, mAP@0.5:0.95 = 0.875, precision = 0.931, recall = 0.918; single photos 14 of 14 correct.
Full method and results: the Milestone 2 report.

## Contents

| File | Purpose |
|---|---|
| `M2_Boxes.ipynb` | **Bounding boxes.** Text prompt (OWLv2), visual prompt (OWLv2 image-guided), self-trained YOLO26-s, consensus vote, the review tool (verify / fix / exemplar modes), box viewing sheets, and the master box file |
| `M2_Detect.ipynb` | **Detection.** Grid generation and leakage checks, YOLO training and fine-tuning, evaluation (chosen on validation, tested once), and the `detect` step that outputs Object IDs and boxes |
| `scripts/run_step.ps1` | Runs any step headlessly on Windows (also opens the review window when needed) |
| `requirements.txt` | Python packages and the versions used |
| `RUN_HISTORY.md` | The exact sequence of steps that produced the reported results |
| `app/m2_eye_app.py` | **The EYE**, a local demo app for the final detector (see `app/README.md`; not hosted) |
| `outputs/` | Results, box files (including the human review files), training history, evaluation curves, example detections, dataset descriptors and logs of the reported run (no images, no weights) |

Both notebooks are self-contained (no extra `.py` files). Every step writes a log to
`milestone2_outputs/logs/`, checkpoints its progress, resumes after interruption, and rebuilds only
when its inputs change.

## Setup

1. **Python 3.10+ and an NVIDIA GPU** (developed on an RTX 5070 Laptop GPU, 8 GB).
2. Install PyTorch with CUDA first, then the rest:
   ```
   pip install torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu128
   pip install -r requirements.txt
   ```
3. **Project folder** (outside this repository), containing:
   ```
   <PROJECT_DIR>/
     Dataset - Updated/images_OBJ001 ... images_OBJ073     the shared course dataset (unchanged)
     milestone1_outputs_v2/clean_dataset.csv, split.csv   from Milestone 1 (cleaning + split)
     object_list.csv                                      exported from the course Object List sheet
   ```
4. Set `PROJECT_DIR` in the **Configuration** cell of both notebooks. All outputs go to
   `<PROJECT_DIR>/milestone2_outputs/`.

## Running

Steps run only when named in `M2_RUN`, so a full run of the notebook is safe and quick by default.

```powershell
# from the milestone2 folder
.\scripts\run_step.ps1 -Notebook M2_Boxes.ipynb  -Steps stage1_test
.\scripts\run_step.ps1 -Notebook M2_Detect.ipynb -Steps "compose,check"
```

Or in VS Code / Jupyter: set the variables in the Configuration cell and click *Run All*.

### Box pipeline (`M2_Boxes.ipynb`)

| Step | What it does | Time (RTX 5070) |
|---|---|---|
| `stage1_test`, `stage1` | Text-prompt boxes for every photo (OWLv2) | 17 min |
| `stage1b_test`, `stage1b` | Visual-prompt boxes (OWLv2 image-guided, 3 examples per class) | 17 min |
| `sheets` | Read-only sheets: every photo of a class with all its boxes | 3 min |
| `stage23_quick`, `stage23` | Self-trained YOLO26-s (training split only) and the consensus vote | 40 min |
| *review* (see below) | Human verification and correction | your time |
| `master`, `master_strict` | Master box file; strict check that every test photo was reviewed | 1 min |

**Review tool.** Opens a window on the local screen (VS Code or the run script; not Google Colab):
```powershell
.\scripts\run_step.ps1 -Notebook M2_Boxes.ipynb -Reviewer SC -Mode verify -Splits test           # grid: click wrong boxes
.\scripts\run_step.ps1 -Notebook M2_Boxes.ipynb -Reviewer SC -Mode fix    -Splits "train,val,test" -Classes OBJ010
.\scripts\run_step.ps1 -Notebook M2_Boxes.ipynb -Reviewer SC -Mode verify -Splits train -Sample 240   # random audit
.\scripts\run_step.ps1 -Notebook M2_Boxes.ipynb -Reviewer SC -Mode verify -Splits "train,val,test" -Classes OBJ051 -Redo
.\scripts\run_step.ps1 -Notebook M2_Boxes.ipynb -Reviewer SC -Mode exemplar -Classes "OBJ040,OBJ043"   # example boxes
```
Keys: verify = click wrong boxes, Enter confirms the page; fix = 1/2/3 pick a suggested box or drag a new
one, Enter saves, N not visible, L other object, S skip, B back, Q quit. Every action is saved
immediately to `milestone2_outputs/review/review_<initials>.csv`.

### Detection pipeline (`M2_Detect.ipynb`)

| Step | What it does |
|---|---|
| `compose_quick,check_quick,train_quick` | Dry run (50 images, 3 epochs) |
| `compose`, `check` | Grids (2x2 to 5x5) and single photos; leakage and label checks; left-out report per Object ID |
| `train_yolo26s`, `train_yolov8s` | Train at imgsz 1120 (resumable) |
| `finetune_yolo26s`, `finetune_yolov8s`, `finetune2_yolo26s` | 15-epoch continuations after data improvements |
| `evaluate` | Selects the model on validation, then tests once; writes `detect/results_summary.md` and `results.json` |
| `detect` | Object ID + box for any image or folder |

**Detecting objects in your own images:**
```powershell
.\scripts\run_step.ps1 -Notebook M2_Detect.ipynb -Steps detect -DetectSource "C:\path\to\images"
```
Results: a table printed per image, plus `milestone2_outputs/detect/detections/detections.csv`,
`detections.json` and a copy of each image with the boxes drawn. The confidence threshold,
post-processing (one box per grid cell) and inference method (natural size for small images) are the
ones chosen on validation; `-DetectConf` and `-DetectImgsz` override them.
