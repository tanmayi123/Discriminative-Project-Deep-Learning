"""
launch_eye.py  -  patch absolute paths for THIS machine, then start The EYE v3.

Usage:
    python launch_eye.py

Opens http://127.0.0.1:7862 in your browser.
Stop with Ctrl+C.
"""
import json, os, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


print("=" * 55)
print("  The EYE v3 - Launcher")
print("=" * 55)
print("Patching weight paths for this machine ...")

EXTRA   = HERE / "M2 extra"
DETECT  = HERE / "milestone2_outputs" / "detect"

# ── boost / boost2: patch train.json weights ──────────────────
for boost_name in ("boost", "boost2"):
    train_path = EXTRA / boost_name / "train.json"
    if train_path.exists():
        d = read_json(train_path)
        d["weights"] = str(EXTRA / boost_name / "runs" / "boost" / "weights" / "best.pt")
        write_json(train_path, d)
        print(f"  patched M2 extra/{boost_name}/train.json")

# ── M2 models: patch weights paths inside results.json ───────
results_path = DETECT / "results.json"
if results_path.exists():
    r = read_json(results_path)
    m2_models = {
        "yolo26s":    DETECT / "runs" / "yolo26s"    / "weights" / "best.pt",
        "yolov8s":    DETECT / "runs" / "yolov8s"    / "weights" / "best.pt",
        "yolo26s_ft": DETECT / "runs" / "yolo26s_ft" / "weights" / "best.pt",
        "yolov8s_ft": DETECT / "runs" / "yolov8s_ft" / "weights" / "best.pt",
        "yolo26s_ft2":DETECT / "runs" / "yolo26s_ft2"/ "weights" / "best.pt",
    }
    for model_name, pt_path in m2_models.items():
        if model_name in r.get("models", {}):
            r["models"][model_name]["weights"] = str(pt_path)
    write_json(results_path, r)
    print(f"  patched milestone2_outputs/detect/results.json ({len(m2_models)} models)")

print("Done. Launching The EYE v3 at http://127.0.0.1:7862 ...")
print("Stop with Ctrl+C")
print("=" * 55)

env = os.environ.copy()
env["M2_PROJECT_DIR"] = str(HERE)

subprocess.run(
    [sys.executable, str(HERE / "m2_eye_app_v3.py")],
    env=env,
)
