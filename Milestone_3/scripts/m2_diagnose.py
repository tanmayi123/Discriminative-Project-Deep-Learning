"""
Milestone 2 extension, step 1: DIAGNOSIS on the VALIDATION set only (the test set stays untouched).

For the selected model it reports
  1. why every missed object was missed (below threshold, removed by the one-box-per-cell rule, wrong ID,
     look-alike, box off, very small, or not seen at all), with object sizes;
  2. levers that need no retraining: filling empty cells, test-time augmentation, larger inference size,
     other thresholds;
  3. the rotation baseline: the same validation grids rotated by 90, 180 and 270 degrees.

Usage (from anywhere inside the project folder):   python m2_diagnose.py
Reads the project; writes ONLY to  PROJECT - 1/M2 extra/diagnosis/  (diagnosis.json, diagnosis.md). Nothing existing is changed.
"""
import json, os, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from PIL import Image

def _np(o):
    """JSON fallback: NumPy numbers become plain Python numbers (the detector returns float32/int64 values)."""
    return o.item() if hasattr(o, "item") else str(o)


TILE, CONF, IMGSZ = 224, 0.10, 1120
LOOKALIKE = {"watches": ["OBJ025", "OBJ046", "OBJ065"], "wallets": ["OBJ061", "OBJ067"],
             "power banks": ["OBJ048", "OBJ057"], "rackets": ["OBJ030", "OBJ034", "OBJ035", "OBJ049"],
             "sunglasses / goggles": ["OBJ014", "OBJ045", "OBJ043"], "brush / comb": ["OBJ050", "OBJ059"],
             "bottles": ["OBJ002", "OBJ015", "OBJ063"], "projectors": ["OBJ064", "OBJ070"],
             "plush toys": ["OBJ021", "OBJ024", "OBJ071"], "notebook / keyboard": ["OBJ003", "OBJ012"]}
GROUP_OF = {o: g for g, ids in LOOKALIKE.items() for o in ids}

def find_project():
    if os.environ.get("M2_PROJECT_DIR"): return Path(os.environ["M2_PROJECT_DIR"])
    for start in (Path(__file__).resolve().parent, Path.cwd()):
        for d in (start, *start.parents):
            if (d / "milestone2_outputs").is_dir(): return d
    sys.exit("Project folder not found (the one containing milestone2_outputs/). Set M2_PROJECT_DIR.")

PROJECT = find_project(); OUT = PROJECT / "milestone2_outputs"; DS = OUT / "detect" / "dataset"   # read only
EXTRA = Path(os.environ["M2_EXTRA_DIR"]) if os.environ.get("M2_EXTRA_DIR") else PROJECT / "M2 extra"   # ALL new outputs go here
EXTRA.mkdir(parents=True, exist_ok=True)
RES = EXTRA / "diagnosis"; RES.mkdir(parents=True, exist_ok=True)
r = json.loads((OUT / "detect" / "results.json").read_text()); WEIGHTS = r["models"][r["selected_model"]]["weights"]
_mode = ((r.get("any_shape") or {}).get("selected_inference") or "natural")
import yaml
names = yaml.safe_load((DS / "data.yaml").read_text())["names"]; names = names if isinstance(names, list) else [names[k] for k in sorted(names, key=int)]
imgs = sorted((DS / "images" / "val").glob("*.jpg")) + sorted((DS / "images" / "val").glob("*.png"))
if not imgs: sys.exit(f"No validation images in {DS / 'images' / 'val'}")

def load_gt(p):
    W, H = Image.open(p).size; lab = DS / "labels" / "val" / (p.stem + ".txt"); g = []
    for line in (lab.read_text().split("\n") if lab.exists() else []):
        if line.strip():
            c, xc, yc, w, h = line.split()[:5]; xc, yc, w, h = float(xc) * W, float(yc) * H, float(w) * W, float(h) * H
            g.append((names[int(c)], (xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2)))
    return g, (W, H)

# ------------------------------------------------------------------------------------------ geometry and metrics
def iou(a, b):
    iw, ih = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0: return 0.0
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - iw * ih
    return iw * ih / u if u > 0 else 0.0
cell = lambda b: (int(((b[0] + b[2]) / 2) // TILE), int(((b[1] + b[3]) / 2) // TILE))

def post(P, conf=CONF, use_cell=True):
    P = [q for q in P if q[1] >= conf]
    if not use_cell: return P
    best = {}
    for q in sorted(P, key=lambda t: -t[1]): best.setdefault(cell(q[2]), q)
    return list(best.values())

def match(P, G):
    """Greedy one-to-one matching at IoU >= 0.5: returns per-GT (status, pred) and unmatched preds."""
    used, res = set(), [None] * len(G)
    for k, (o, s, b) in sorted(enumerate(P), key=lambda t: -t[1][1]):
        j = max(range(len(G)), key=lambda j: iou(b, G[j][1]) if j not in used else -1, default=None)
        if j is None or j in used or iou(b, G[j][1]) < .5: continue
        used.add(j); res[j] = ("correct" if G[j][0] == o else "wrong", (o, s, b))
    return res, len(P) - len(used)

def task(preds, gts, conf=CONF, use_cell=True, fill=None):
    n_gt = n_pred = ok = wrong = 0; full = []
    for im, (G, size) in gts.items():
        P = post(preds[im], conf, use_cell)
        if fill is not None:                                       # one guess for every empty, occupied cell
            occ = {cell(b) for _, b in G}; have = {cell(q[2]) for q in P}; raw = sorted(preds[im], key=lambda t: -t[1])
            for c in occ - have:
                q = next((q for q in raw if cell(q[2]) == c and q[1] >= fill), None)
                if q: P.append(q)
        res, extra = match(P, G)
        k = sum(1 for x in res if x and x[0] == "correct"); w = sum(1 for x in res if x and x[0] == "wrong")
        n_gt += len(G); n_pred += len(P); ok += k; wrong += w; full.append(int(k == len(G) and len(P) == len(G)))
    return dict(recall=round(ok / max(n_gt, 1), 4), precision=round(ok / max(n_pred, 1), 4), wrong_ids=wrong,
                missed_or_wrong=n_gt - ok, fully_correct=round(float(np.mean(full)), 4))

def ap_101(tp, conf, n_gt):
    if not len(tp): return 0.0
    o = np.argsort(-np.asarray(conf)); tp = np.asarray(tp, float)[o]; fp = 1 - tp
    rec = np.cumsum(tp) / n_gt; prec = np.cumsum(tp) / np.maximum(np.cumsum(tp) + np.cumsum(fp), 1e-9)
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    return float(np.mean([prec[rec >= t].max() if (rec >= t).any() else 0.0 for t in np.linspace(0, 1, 101)]))

def mean_ap(preds, gts):
    classes = sorted({o for G, _ in gts.values() for o, _ in G}); out = []
    for thr in np.round(np.arange(.5, .96, .05), 2):
        aps = []
        for c in classes:
            n_gt = sum(1 for G, _ in gts.values() for o, _ in G if o == c)
            P = sorted(((s, im, b) for im in preds for o, s, b in preds[im] if o == c), key=lambda t: -t[0]); used, tp, cf = set(), [], []
            for s, im, b in P:
                G = gts[im][0]; bj = max((j for j in range(len(G)) if G[j][0] == c and (im, j) not in used), key=lambda j: iou(b, G[j][1]), default=None)
                hit = bj is not None and iou(b, G[bj][1]) >= thr
                if hit: used.add((im, bj))
                tp.append(int(hit)); cf.append(s)
            aps.append(ap_101(tp, cf, n_gt))
        out.append(float(np.mean(aps)))
    return round(out[0], 4), round(float(np.mean(out)), 4)

# ------------------------------------------------------------------------------------------ predictions
from ultralytics import YOLO
model = YOLO(str(WEIGHTS))
def to_canvas(im):
    """Natural-size inference (as validated for the selected model): images up to 1120 px are placed unscaled on a grey canvas."""
    im = Image.open(im).convert("RGB") if isinstance(im, str) else im
    if im.width > IMGSZ or im.height > IMGSZ: return im
    c = Image.new("RGB", (IMGSZ, IMGSZ), (128, 128, 128)); c.paste(im, (0, 0)); return c
def predict(images, imgsz=IMGSZ, augment=False, natural=None):
    natural = (_mode != "standard") if natural is None else natural
    out = {}
    for key, im in images.items():
        res = model.predict(to_canvas(im) if natural else im, imgsz=imgsz, conf=0.001, max_det=300, augment=augment, verbose=False)[0]
        out[key] = [(str(model.names[int(c)]), float(s), tuple(map(float, b))) for c, s, b in
                    zip(res.boxes.cls.cpu().numpy(), res.boxes.conf.cpu().numpy(), res.boxes.xyxy.cpu().numpy())]
    return out

def rot_box(b, W, H, k):
    """Box after rotating the image by k*90 degrees counter-clockwise (PIL rotate with expand=True)."""
    x1, y1, x2, y2 = b
    if k == 1: return (y1, W - x2, y2, W - x1)
    if k == 2: return (W - x2, H - y2, W - x1, H - y1)
    return (H - y2, x1, H - y1, x2)

t0 = time.time()
GT = {p.name: load_gt(p) for p in imgs}
IM = {p.name: str(p) for p in imgs}
print(f"validation: {len(GT)} images, {sum(len(g) for g, _ in GT.values())} objects | model: {r['selected_model']} | inference: {_mode}")
base = predict(IM)
report = {"model": r["selected_model"], "images": len(GT), "objects": sum(len(g) for g, _ in GT.values())}
report["baseline"] = dict(**task(base, GT), mAP50=None, mAP50_95=None); report["baseline"]["mAP50"], report["baseline"]["mAP50_95"] = mean_ap(base, GT)

# ---- 1. why each object is missed
causes, sizes, per_class = Counter(), defaultdict(list), defaultdict(Counter)
for im, (G, (W, H)) in GT.items():
    P = post(base[im]); res, _ = match(P, G)
    for j, (o, g) in enumerate(G):
        frac = (g[2] - g[0]) * (g[3] - g[1]) / (TILE * TILE)
        if res[j] and res[j][0] == "correct": per_class[o]["correct"] += 1; continue
        same_hi = [q for q in base[im] if q[0] == o and q[1] >= CONF and iou(q[2], g) >= .5]
        same_lo = [q for q in base[im] if q[0] == o and q[1] < CONF and iou(q[2], g) >= .5]
        near = [q for q in P if q[0] == o and .1 <= iou(q[2], g) < .5]
        if res[j] and res[j][0] == "wrong":
            c = "wrong ID, look-alike" if GROUP_OF.get(o) and GROUP_OF.get(o) == GROUP_OF.get(res[j][1][0]) else "wrong ID, other"
        elif same_hi: c = "removed by one-box-per-cell (another box in the cell won)"
        elif near: c = "box placed off (overlap below 0.5)"
        elif same_lo:
            m = max(q[1] for q in same_lo); c = f"below threshold (seen at {'0.05-0.10' if m >= .05 else '0.01-0.05' if m >= .01 else 'under 0.01'})"
        elif frac < .02: c = "not seen, very small (under 2% of its photo)"
        else: c = "not seen"
        causes[c] += 1; sizes[c].append(frac); per_class[o][c] += 1
report["miss_causes"] = {c: dict(count=n, median_size_pct=round(100 * float(np.median(sizes[c])), 2)) for c, n in causes.most_common()}
weak = sorted(per_class.items(), key=lambda kv: kv[1]["correct"] / max(sum(kv[1].values()), 1))[:12]
report["weakest_classes"] = [dict(object=o, correct=v["correct"], total=sum(v.values()), causes={k: n for k, n in v.items() if k != "correct"}) for o, v in weak]

# ---- 2. levers without retraining
report["levers"] = {}
for thr in (.05, .10, .15, .25): report["levers"][f"threshold {thr:.2f}"] = task(base, GT, conf=thr)
for floor in (.001, .01, .03): report["levers"][f"fill empty cells (guess >= {floor})"] = task(base, GT, fill=floor)
tta = predict(IM, augment=True); report["levers"]["test-time augmentation"] = dict(**task(tta, GT), mAP=mean_ap(tta, GT))
for sz in (1280, 1600):
    hi = predict(IM, imgsz=sz, natural=False); report["levers"][f"inference at {sz} px"] = dict(**task(hi, GT), mAP=mean_ap(hi, GT))

# ---- 3. rotation baseline
report["rotation"] = {"0": dict(**task(base, GT), mAP=report["baseline"]["mAP50"])}
for k, deg in ((1, 90), (2, 180), (3, 270)):
    RG, RI = {}, {}
    for p in imgs:
        G, (W, H) = GT[p.name]; im = Image.open(p).convert("RGB").rotate(90 * k, expand=True)
        RI[p.name] = im; RG[p.name] = ([(o, rot_box(b, W, H, k)) for o, b in G], im.size)
    rp = predict(RI); report["rotation"][str(deg)] = dict(**task(rp, RG), mAP=mean_ap(rp, RG)[0])

(RES / "diagnosis.json").write_text(json.dumps(report, indent=2, default=_np))
L = [f"# Diagnosis on validation ({report['images']} grids, {report['objects']} objects), model {report['model']}", "",
     f"Baseline: {report['baseline']}", "", "## Why objects are missed (threshold 0.10, one box per cell)", "", "| cause | count | median size (% of its photo) |", "|---|---|---|"]
L += [f"| {c} | {v['count']} | {v['median_size_pct']} |" for c, v in report["miss_causes"].items()]
L += ["", "## Weakest classes", "", "| object | correct / total | causes |", "|---|---|---|"] + [f"| {w['object']} | {w['correct']} / {w['total']} | {w['causes']} |" for w in report["weakest_classes"]]
L += ["", "## Levers without retraining", "", "| lever | recall | precision | wrong IDs | fully correct | mAP50 / mAP50-95 |", "|---|---|---|---|---|---|"]
L += [f"| {k} | {v['recall']} | {v['precision']} | {v['wrong_ids']} | {v['fully_correct']} | {v.get('mAP', '')} |" for k, v in report["levers"].items()]
L += ["", "## Rotation (validation grids rotated)", "", "| rotation | recall | precision | wrong IDs | fully correct | mAP50 |", "|---|---|---|---|---|---|"]
L += [f"| {k} deg | {v['recall']} | {v['precision']} | {v['wrong_ids']} | {v['fully_correct']} | {v['mAP']} |" for k, v in report["rotation"].items()]
(RES / "diagnosis.md").write_text("\n".join(L) + "\n"); print("\n".join(L)); print(f"\ndone in {(time.time() - t0) / 60:.1f} min -> {RES}")
