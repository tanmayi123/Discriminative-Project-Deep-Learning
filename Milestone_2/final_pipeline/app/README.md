# The EYE: Explainable YOLO Explorer (local demo)

`m2_eye_app.py` is a small demo app for the final Milestone 2 detector. It runs **only on your own computer**
(nothing is uploaded or hosted) and uses just the Python standard library for its web server, plus the packages
the project already uses.

What it does:
* builds a test image (single photo or a 2x2 to 5x5 grid, optional look-alike mix and image conditions such as blur or low resolution), or takes an uploaded image;
* runs the selected model and shows every detected **Object ID** with its box and confidence;
* scores the result against the true boxes and explains every error (missed, wrong ID, extra box);
* runs a stress test and a confidence-threshold sweep.

## Run

```
python m2_eye_app.py          # opens http://127.0.0.1:7860   (stop with Ctrl+C)
```

The app looks for the project folder by itself (the nearest folder above it that contains `milestone2_outputs/`);
set `M2_PROJECT_DIR` to point to it explicitly. It needs the outputs of `M2_Detect.ipynb` (`detect/results.json`,
the model weights) and the course dataset, neither of which is in this repository.
Optional: `M2_PORT` (default 7860).

Keys: N random grid, D detect, U upload, Esc close or zoom out.
