"""
Milestone 2 cross-evaluation: every model scored on the SAME human-checked test set, with the same code.

Each teammate run made its own random split, so 36-59% of our test photos were in another model's TRAINING
set. For each teammate model, the test objects whose photo that model trained on are masked (their true box
is removed, and so is any prediction whose centre falls in that photo's 224-px cell). That model and ours are
then scored on exactly the remaining, unseen objects.

Usage (from anywhere inside the project folder):
    python m2_cross_eval.py
Expects the three teammate archives extracted under  <project>/teammate_runs/  (any sub-folder layout).
Writes  milestone2_outputs/detect/cross_eval/cross_eval.json  and  cross_eval.md
"""
import json, os, re, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

TILE = 224

# ------------------------------------------------------------------------------------------ locate everything
def find_project():
    if os.environ.get("M2_PROJECT_DIR"): return Path(os.environ["M2_PROJECT_DIR"])
    for start in (Path(__file__).resolve().parent, Path.cwd()):
        for d in (start, *start.parents):
            if (d / "milestone2_outputs").is_dir(): return d
    sys.exit("Project folder not found (the one containing milestone2_outputs/). Set M2_PROJECT_DIR.")

PROJECT = find_project()
OUT = PROJECT / "milestone2_outputs"
DS = OUT / "detect" / "dataset"
TEAM = PROJECT / "teammate_runs"
RES = OUT / "detect" / "cross_eval"; RES.mkdir(parents=True, exist_ok=True)

def one(pattern, what):
    hits = sorted(TEAM.rglob(pattern))
    if not hits: sys.exit(f"Not found under {TEAM}: {what} ({pattern}). Extract the three teammate zips there.")
    return hits[0]

RUNS = {
    "A": dict(label="Run A: YOLOv8-s (Grounding DINO + YOLO-World + classifier labels)", weights=one("yolov8s_m2/weights/best.pt", "run A weights"),
              train=lambda: set(pd.read_csv(one("IE7615_project_M2/single_photo_split.csv", "run A split")).query("split == 'train'").fname.map(str.lower)),
              conf=0.30, names=None),
    "B": dict(label="Run B: YOLOE-26s (YOLO-World + YOLOE labels)", weights=one("B_YOLOE26s/weights/best.pt", "run B weights"),
              train=lambda: set(pd.read_csv(one("IE7615_M2_final/single_train_split.csv", "run B split")).fname.map(str.lower)),
              conf=0.15, names=one("IE7615_M2_final/class_index_to_object_id.csv", "run B class map")),
    "C": dict(label="Run C: YOLO11-s, Section 1 only (YOLO-World + YOLOE labels)", weights=one("m2_yolo11/runs/yolo11s/weights/best.pt", "run C weights"),
              train=lambda: set(pd.read_csv(one("m2_yolo11/box_labels_section1.csv", "run C labels")).query("split == 'train'").fname.map(str.lower)),
              conf=0.25, names=None),
}
r = json.loads((OUT / "detect" / "results.json").read_text())
OURS = dict(label=f"Ours: {r['selected_model']}", weights=Path(r["models"][r["selected_model"]]["weights"]), conf=0.10, cell=True)

# ------------------------------------------------------------------------------------------ our test set (human-checked boxes)
import yaml
data = yaml.safe_load((DS / "data.yaml").read_text())
names = data["names"]; names = names if isinstance(names, list) else [names[k] for k in sorted(names, key=int)]
imgs = sorted((DS / "images" / "test").glob("*.jpg")) + sorted((DS / "images" / "test").glob("*.png"))
if not imgs: sys.exit(f"No test images in {DS / 'images' / 'test'}")
from PIL import Image
GT = {}                                                    # image -> list of (object_id, (x1,y1,x2,y2))
for p in imgs:
    W, H = Image.open(p).size; lab = DS / "labels" / "test" / (p.stem + ".txt"); g = []
    for line in (lab.read_text().split("\n") if lab.exists() else []):
        if not line.strip(): continue
        c, xc, yc, w, h = line.split()[:5]; xc, yc, w, h = float(xc) * W, float(yc) * H, float(w) * W, float(h) * H
        g.append((names[int(c)], (xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2)))
    GT[p.name] = g
# which photo sits in which cell: from the grid manifest (each grid holds distinct Object IDs)
man = pd.read_csv(DS / "manifest.csv")
col = lambda *cands: next((c for c in man.columns if c.lower() in cands), None)
c_img, c_photo, c_obj = col("image", "grid", "grid_image", "file"), col("fname", "photo", "src_fname", "source"), col("object_id", "class", "label")
if not (c_img and c_photo and c_obj): sys.exit(f"manifest.csv columns not recognised: {list(man.columns)}")
man = man[man[c_img].map(lambda s: (Path(str(s)).stem + ".jpg") in GT)]
PHOTO = {(Path(str(a)).stem + ".jpg", o): Path(str(f)).name.lower() for a, f, o in zip(man[c_img], man[c_photo], man[c_obj])}
print(f"test set: {len(GT)} images, {sum(map(len, GT.values()))} objects, manifest rows matched: {len(PHOTO)}")

# ------------------------------------------------------------------------------------------ metrics (same code for every model)
def iou(a, b):
    iw, ih = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0: return 0.0
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - iw * ih
    return iw * ih / u if u > 0 else 0.0
cell = lambda b: (int(((b[0] + b[2]) / 2) // TILE), int(((b[1] + b[3]) / 2) // TILE))

def ap_101(tp, conf, n_gt):
    if n_gt == 0: return None
    if not len(tp): return 0.0
    o = np.argsort(-np.asarray(conf)); tp = np.asarray(tp, float)[o]; fp = 1 - tp
    rec = np.cumsum(tp) / n_gt; prec = np.cumsum(tp) / np.maximum(np.cumsum(tp) + np.cumsum(fp), 1e-9)
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    return float(np.mean([prec[rec >= t].max() if (rec >= t).any() else 0.0 for t in np.linspace(0, 1, 101)]))

def mean_ap(preds, gts, classes):
    """COCO-style mAP@0.5 and mAP@0.5:0.95 over `classes` (101-point interpolation)."""
    out = {}
    for thr in np.round(np.arange(.5, .96, .05), 2):
        aps = []
        for c in classes:
            n_gt = sum(1 for im in gts for o, _ in gts[im] if o == c)
            if n_gt == 0: continue
            P = sorted(((p, im, b) for im in preds for o, p, b in preds[im] if o == c), key=lambda t: -t[0])
            used, tp, cf = set(), [], []
            for p, im, b in P:
                best, bj = 0, None
                for j, (o, g) in enumerate(gts[im]):
                    if o == c and (im, j) not in used:
                        v = iou(b, g)
                        if v > best: best, bj = v, j
                if bj is not None and best >= thr: used.add((im, bj)); tp.append(1)
                else: tp.append(0)
                cf.append(p)
            aps.append(ap_101(tp, cf, n_gt))
        out[thr] = float(np.mean(aps)) if aps else 0.0
    return out[0.5], float(np.mean(list(out.values())))

def task(preds, gts, conf, use_cell):
    """Precision, recall, wrong IDs and fully-correct images at the run's own threshold (IoU >= 0.5)."""
    n_gt = n_pred = n_ok = n_wrong = 0; fully = []
    for im in gts:
        P = [(o, p, b) for o, p, b in preds.get(im, []) if p >= conf]
        if use_cell:
            best = {}
            for q in sorted(P, key=lambda t: -t[1]): best.setdefault(cell(q[2]), q)
            P = list(best.values())
        used, ok, wrong = set(), 0, 0
        for o, p, b in sorted(P, key=lambda t: -t[1]):
            bj, best = None, 0
            for j, (go, g) in enumerate(gts[im]):
                if j not in used and iou(b, g) > best: best, bj = iou(b, g), j
            if bj is not None and best >= .5:
                used.add(bj); ok += gts[im][bj][0] == o; wrong += gts[im][bj][0] != o
        n_gt += len(gts[im]); n_pred += len(P); n_ok += ok; n_wrong += wrong
        if gts[im]: fully.append(int(ok == len(gts[im]) and len(P) == len(gts[im])))
    return dict(precision=n_ok / max(n_pred, 1), recall=n_ok / max(n_gt, 1), wrong_ids=n_wrong, fully_correct=float(np.mean(fully)) if fully else None)

# ------------------------------------------------------------------------------------------ predictions
from ultralytics import YOLO
def load(w):
    try: return YOLO(str(w))
    except Exception:
        from ultralytics import YOLOE
        return YOLOE(str(w))

def predict(run):
    model = load(run["weights"]); mp = {}
    if run.get("names"):
        mp = {int(i): o for i, o in pd.read_csv(run["names"])[["class_index", "object_id"]].values}
    to_id = lambda c: mp.get(int(c)) or str(model.names[int(c)])
    out = {}
    for p in imgs:
        res = model.predict(str(p), imgsz=1120, conf=0.001, max_det=300, verbose=False)[0]
        out[p.name] = [(to_id(c), float(s), tuple(map(float, b))) for c, s, b in
                       zip(res.boxes.cls.cpu().numpy(), res.boxes.conf.cpu().numpy(), res.boxes.xyxy.cpu().numpy())]
    return out, sorted({to_id(c) for c in model.names})

def masked(preds, gts, seen, classes):
    """Drop test objects whose photo the model trained on (and predictions in their cells); keep only `classes`."""
    G, P, kept, dropped = {}, {}, 0, 0
    for im, g in gts.items():
        bad_cells, keep = set(), []
        for o, b in g:
            if PHOTO.get((im, o), "") in seen or o not in classes: bad_cells.add(cell(b)); dropped += 1
            else: keep.append((o, b)); kept += 1
        G[im] = keep
        P[im] = [(o, s, b) for o, s, b in preds.get(im, []) if cell(b) not in bad_cells and o in classes]
    return G, P, kept, dropped

t0 = time.time()
print("predicting with our model ..."); ours_pred, _ = predict(OURS)
ALL = sorted({o for g in GT.values() for o, _ in g})
results = {"ours_full": dict(label=OURS["label"], objects=sum(map(len, GT.values())),
                             mAP50=None, mAP50_95=None, **task(ours_pred, GT, OURS["conf"], True))}
results["ours_full"]["mAP50"], results["ours_full"]["mAP50_95"] = mean_ap(ours_pred, GT, ALL)
for k, run in RUNS.items():
    print(f"predicting with {run['label']} ..."); pred, classes = predict(run)
    seen = run["train"](); cls = [c for c in ALL if c in classes]
    G, Pm, kept, dropped = masked(pred, GT, seen, set(cls)); _, Po, _, _ = masked(ours_pred, GT, seen, set(cls))
    them_map, ours_map = mean_ap(Pm, G, cls), mean_ap(Po, G, cls)
    results[k] = dict(label=run["label"], classes=len(cls), objects_scored=kept, objects_masked=dropped,
                      teammate=dict(mAP50=them_map[0], mAP50_95=them_map[1], **task(Pm, G, run["conf"], False)),
                      ours_same_objects=dict(mAP50=ours_map[0], mAP50_95=ours_map[1], **task(Po, G, OURS["conf"], True)))
(RES / "cross_eval.json").write_text(json.dumps(results, indent=2))
f = lambda v: "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))
lines = ["# Cross-evaluation on our human-checked test set", "",
         f"Our model, all {results['ours_full']['objects']} test objects: mAP@0.5 {f(results['ours_full']['mAP50'])}, mAP@0.5:0.95 {f(results['ours_full']['mAP50_95'])} (re-scored with this script)", "",
         "| run | classes | objects scored (masked) | mAP@0.5 | mAP@0.5:0.95 | precision | recall | wrong IDs | fully correct | ours on the same objects: mAP@0.5 / @0.5:0.95 / fully correct |",
         "|---|---|---|---|---|---|---|---|---|---|"]
for k in RUNS:
    x = results[k]; t = x["teammate"]; o = x["ours_same_objects"]
    lines.append(f"| {x['label']} | {x['classes']} | {x['objects_scored']} ({x['objects_masked']}) | {f(t['mAP50'])} | {f(t['mAP50_95'])} | {f(t['precision'])} | {f(t['recall'])} | {t['wrong_ids']} | {f(t['fully_correct'])} | {f(o['mAP50'])} / {f(o['mAP50_95'])} / {f(o['fully_correct'])} |")
(RES / "cross_eval.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines)); print(f"\ndone in {(time.time() - t0) / 60:.1f} min -> {RES}")
