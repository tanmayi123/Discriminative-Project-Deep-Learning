========================================================
  The EYE v3  -  Setup & Run Guide
  Group 7  |  Deep Learning Project 1  |  Milestone 2 Extension
========================================================

WHAT IS THIS
------------
The EYE (Explainable YOLO Explorer) is a local web app that lets you
upload any image (or build a random grid from the test dataset) and
run the trained YOLO26-s object detector on it. Everything runs on
your machine - nothing is uploaded anywhere.

Model: yolo26s_boost  (selected by validation mAP@0.5:0.95 = 0.8857)
       imgsz=1280, grid threshold=0.10, scene threshold=0.35


REQUIREMENTS
------------
- Python 3.10 or newer
- NVIDIA GPU strongly recommended (CPU works but is slow)
- CUDA toolkit matching your GPU driver (if using GPU)


STEP 1 - Install packages
--------------------------
Open a terminal in this folder (RUN_EYE/) and run:

    pip install -r requirements.txt

If you want GPU acceleration, install the CUDA version of torch first:
    https://pytorch.org/get-started/locally/
Then run pip install -r requirements.txt.


STEP 2 - Launch The EYE
------------------------
In the same terminal:

    python launch_eye.py

The launcher patches the weight paths for your machine automatically,
then starts the app. Open your browser at:

    http://127.0.0.1:7862

Stop the server with Ctrl+C.


WHAT YOU WILL SEE
-----------------
- Model panel: "yolo26s_boost" is the default (the selected model).
  All 7 models are available in the dropdown:
    yolo26s, yolov8s, yolo26s_ft, yolov8s_ft, yolo26s_ft2  (Milestone 2)
    yolo26s_boost  (Extension - SELECTED, default)
    yolo26s_boost2 (Extension - trained at higher resolution)
- Upload a photo or click the eye to build a random grid.
- Click Detect (or press D) to run inference.
- The review matrix below shows per-object scores, errors, and a
  threshold sweep.


RANDOM GRID FEATURE
-------------------
The "random grid" button picks photos from the test dataset.
This requires the "Dataset - Updated" folder to be present in the
project root. It is NOT included here (too large).
If the dataset is absent, upload your own images instead - detection
still works fully.


FOLDER STRUCTURE (do not move files around)
--------------------------------------------
RUN_EYE/
  launch_eye.py              <- run this
  m2_eye_app_v3.py           <- the app (do not run directly)
  requirements.txt
  object_list.csv
  milestone2_outputs/
    boxes_master.csv
    detect/
      results.json
      per_class_ap_test.csv
  M2 extra/
    selected_model.json
    boost/
      train.json
      evaluation.json
      runs/boost/weights/
        best.pt              <- selected model weights (~20 MB)
    boost2/
      train.json
      evaluation.json
      runs/boost/weights/
        best.pt              <- boost2 weights (~20 MB)


TROUBLESHOOTING
---------------
"Project folder not found"
  -> Make sure you run: python launch_eye.py  (not m2_eye_app_v3.py directly)

"CUDA out of memory"
  -> The model uses ~1.5 GB VRAM at imgsz=1280. Close other GPU apps.

"No module named ultralytics"
  -> Run: pip install -r requirements.txt

Port already in use
  -> Set env var M2_PORT=7863 (or any free port) before launching:
     Windows:  set M2_PORT=7863 && python launch_eye.py
     Mac/Linux: M2_PORT=7863 python launch_eye.py


========================================================
