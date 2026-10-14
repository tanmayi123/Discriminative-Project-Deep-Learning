"""
The EYE – Explainable YOLO Explorer: the Milestone 2 detector demo (runs locally; nothing is uploaded anywhere).

The eye is the canvas: the test image sits in the pupil, the iris ring is the score (one segment per object,
coloured by its result), and Detect is a blink. The eyebrow holds the build settings (grid, mix, conditions, seed,
model, threshold); the dock on the left holds upload, random grid, stress test, about, files, snapshot and calm
mode; the review matrix below holds the score, every error with its reason, the inspect view and a threshold sweep.

Uses only the Python standard library for the web server, plus the packages the project already uses.
Run:   python m2_eye_app_v2.py     (opens http://127.0.0.1:7861)      Stop: Ctrl+C
The EYE v2: adds the improved model (M2 extra/boost) and rotated, scattered and cut-out test scenes.
It only READS the project; the original EYE (m2_eye_app.py) and model are untouched.
Finds the project folder by itself (the nearest folder above this file that contains milestone2_outputs/);
override with M2_PROJECT_DIR. Optional: M2_PORT (default 7861), M2_HOST=0.0.0.0 to share on a trusted network.
Keys:  N random grid · D detect · U upload · Esc close / zoom out
"""
import json, os, random, re, tempfile, time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

# ============================================================================================ settings
def find_project():
    """The project folder: M2_PROJECT_DIR if set, else the nearest folder (this file's folder or above, or the current
    folder or above) that contains milestone2_outputs/."""
    if os.environ.get("M2_PROJECT_DIR"): return Path(os.environ["M2_PROJECT_DIR"])
    for start in (Path(__file__).resolve().parent, Path.cwd()):
        for d in (start, *start.parents):
            if (d / "milestone2_outputs").is_dir(): return d
    raise SystemExit("Project folder not found. Put this file inside the project folder (the one containing "
                     "milestone2_outputs/), or set M2_PROJECT_DIR to it.")
PROJECT = find_project()
OUT, DATA = PROJECT / "milestone2_outputs", PROJECT / "Dataset - Updated"
RESULTS, MASTER = OUT / "detect" / "results.json", OUT / "boxes_master.csv"
PERCLASS, OBJLIST = OUT / "detect" / "per_class_ap_test.csv", PROJECT / "object_list.csv"
EXTRA = Path(os.environ["M2_EXTRA_DIR"]) if os.environ.get("M2_EXTRA_DIR") else PROJECT / "M2 extra"      # v2: the extension's outputs
BOOST = EXTRA / os.environ.get("M2_BOOST_DIR", "boost"); BOOST_NAME = "yolo26s_" + BOOST.name   # e.g. M2_BOOST_DIR=boost2
TILE, IMGSZ, CONF_FLOOR = 224, 1120, 0.01
GRIDS = {"Single photo (1x1)": (1, 1), "2x2 (4 objects)": (2, 2), "3x3 (9 objects)": (3, 3),
         "4x4 (16 objects)": (4, 4), "5x4 (20 objects)": (5, 4), "5x5 (25 objects)": (5, 5)}
DEGRADE = ["None", "Blur", "Low resolution", "Dark", "Heavy JPEG", "All (the report's low-quality test)"]
# Look-alike groups, defined from the Object List names (same as in training)
LOOKALIKE = {"watches": ["OBJ025", "OBJ046", "OBJ065"], "wallets": ["OBJ061", "OBJ067"],
             "power banks": ["OBJ048", "OBJ057"], "rackets": ["OBJ030", "OBJ034", "OBJ035", "OBJ049"],
             "sunglasses / goggles": ["OBJ014", "OBJ045", "OBJ043"], "brush / comb": ["OBJ050", "OBJ059"],
             "bottles": ["OBJ002", "OBJ015", "OBJ063"], "projectors": ["OBJ064", "OBJ070"],
             "plush toys": ["OBJ021", "OBJ024", "OBJ071"], "notebook / keyboard": ["OBJ003", "OBJ012"]}
GROUP_OF = {o: g for g, ids in LOOKALIKE.items() for o in ids}


# ============================================================================================ loading
def load_settings(path=RESULTS):
    r = json.loads(Path(path).read_text())
    best = r["selected_model"]
    a = r.get("any_shape") or {}
    mode = a.get("selected_inference", "standard")
    cfg = (a.get("candidates") or {}).get(mode) or {}
    models = {k: dict(weights=v["weights"], test_map50=v.get("test_map50"), test_map=v.get("test_map"),
                      val_map50=v.get("val_map50"), val_map=v.get("val_map"),
                      test_precision=v.get("test_precision"), test_recall=v.get("test_recall"))
              for k, v in r["models"].items()}
    model_conf, model_imgsz, boost_eval = {}, {}, None
    tj, ej = BOOST / "train.json", BOOST / "evaluation.json"
    if tj.exists():
        t = json.loads(tj.read_text()); e = json.loads(ej.read_text()) if ej.exists() else {}
        n = (e.get("tests") or {}).get("normal", {}).get("after", {})
        models[BOOST_NAME] = dict(weights=t["weights"], test_map50=n.get("mAP50"), test_map=n.get("mAP50_95"), val_map50=None, val_map=None,
                                  test_precision=n.get("precision"), test_recall=n.get("recall"))
        model_imgsz[BOOST_NAME] = int(t.get("imgsz", 1120)); model_conf[BOOST_NAME] = float((e.get("chosen_on_validation") or {}).get("threshold", 0.10))
        boost_eval = e or None
    return dict(best=best, models=models, mode=mode, selection_rule=r.get("selection_rule", ""), model_conf=model_conf, model_imgsz=model_imgsz, boost_eval=boost_eval,
                shapes=((a.get("test") or {}).get("by_shape") or {}), task=r.get("test_task_metrics", {}),
                conf=float(cfg.get("conf", r["test_task_metrics"]["conf_threshold"])),
                dup=cfg.get("dup", r["test_task_metrics"].get("duplicate_removal_iou")))

def load_objects(path=OBJLIST):
    """Object ID -> (name, section) from the course Object List; columns found by their content."""
    try:
        t = pd.read_csv(path, dtype=str).fillna("")
        idc = max(t.columns, key=lambda c: t[c].str.strip().str.match(r"OBJ\d+$").mean())
        namec = "OBJECT_NAME" if "OBJECT_NAME" in t.columns else next(c for c in t.columns if "name" in c.lower() and "student" not in c.lower() and c != idc)
        secc = next((c for c in t.columns if c.strip().upper().startswith("SECTION")), None)
        sec = lambda s: re.sub(r"\s+", " ", s.strip().title()) if s.strip() else "Unknown"
        return {i.strip(): (n.strip(), sec(t.at[k, secc]) if secc else "Unknown")
                for k, (i, n) in enumerate(zip(t[idc], t[namec])) if i.strip()}
    except Exception:
        return {}

def load_ap(path=PERCLASS):
    try: return pd.read_csv(path).set_index("object_id").ap50.to_dict()
    except Exception: return {}

def load_photos(master=MASTER, split="test", human_only=True):
    m = pd.read_csv(master)
    ok = m.status.isin(["verified", "fixed"]) if human_only else m.status.isin(["verified", "fixed", "unreviewed_auto"])
    m = m[(m.split == split) & ok & m.x1.notna()].copy()
    m["path"] = [str(DATA / f"images_{c}" / f) for c, f in zip(m.object_id, m.fname)]
    return m[m.path.map(os.path.exists)].reset_index(drop=True)


# ============================================================================================ geometry / post-processing
def iou(a, b):
    iw, ih = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0: return 0.0
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - iw * ih
    return iw * ih / u if u > 0 else 0.0

def cell_of(b): return (int(((b[0] + b[2]) / 2) // TILE), int(((b[1] + b[3]) / 2) // TILE))

def postprocess(preds, dup, W, H, use_cell=True):
    """The evaluated post-processing: one box per 224-px cell (grids only), or overlap removal."""
    if dup == "cell":
        if not (use_cell and W % TILE == 0 and H % TILE == 0): return preds
        best = {}
        for p in sorted(preds, key=lambda q: -q[1]): best.setdefault(cell_of(p[2]), p)
        return list(best.values())
    if dup is None: return preds
    keep = []
    for p in sorted(preds, key=lambda q: -q[1]):
        if all(iou(p[2], k[2]) < float(dup) for k in keep): keep.append(p)
    return keep

def select(raw, conf, dup, W, H, use_cell):
    return postprocess([p for p in raw if p[1] >= conf], dup, W, H, use_cell)

def to_canvas(img, size=IMGSZ):
    W, H = img.size
    if W > size or H > size: return img, False
    c = Image.new("RGB", (size, size), (128, 128, 128)); c.paste(img, (0, 0)); return c, True

def run_model(model, img, mode, conf=CONF_FLOOR, imgsz=IMGSZ):
    """All predictions down to CONF_FLOOR (so missed objects can be explained), in input-image pixels, + ms.
    v2: a model trained at imgsz > 1120 gets the image scaled by imgsz/1120 on an imgsz canvas (as in training)."""
    W0, H0 = img.size; s = imgsz / IMGSZ
    if s != 1: img = img.resize((max(1, round(W0 * s)), max(1, round(H0 * s))), Image.Resampling.BICUBIC)
    W, H = img.size
    inp, padded = to_canvas(img, imgsz) if (mode == "native" or s != 1) else (img, False)
    t0 = time.perf_counter()
    r = model.predict(inp, imgsz=imgsz, conf=conf, max_det=100, verbose=False)[0]
    ms = (time.perf_counter() - t0) * 1000
    out = []
    for c, p, b in zip(r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.xyxy.cpu().numpy()):
        x1, y1, x2, y2 = map(float, b)
        if padded:
            if (x1 + x2) / 2 > W or (y1 + y2) / 2 > H: continue
            x2, y2 = min(x2, W), min(y2, H)
        out.append((int(c), float(p), (x1 / s, y1 / s, x2 / s, y2 / s)))           # back to the image's own pixels
    return out, ms


# ============================================================================================ degradation
def degrade(img, kind):
    if kind in (None, "None"): return img
    W, H = img.size
    def lowres(im, f=0.25): return im.resize((max(8, int(W * f)), max(8, int(H * f))), Image.Resampling.BILINEAR).resize((W, H), Image.Resampling.BILINEAR)
    def jpeg(im, q=20):
        import io; b = io.BytesIO(); im.save(b, "JPEG", quality=q); b.seek(0); return Image.open(b).convert("RGB")
    if kind == "Blur": return img.filter(ImageFilter.GaussianBlur(2.0))
    if kind == "Low resolution": return lowres(img)
    if kind == "Dark": return ImageEnhance.Brightness(img).enhance(0.45)
    if kind == "Heavy JPEG": return jpeg(img, 12)
    return jpeg(ImageEnhance.Brightness(lowres(img)).enhance(0.6), 20)      # the report's low-quality test


# ============================================================================================ explaining every outcome
class Info:
    """Lookup of Object ID, name, section, class AP and look-alike group, for a model's class indices."""
    def __init__(self, ids, objects, ap):
        self.ids, self.objects, self.ap = ids, objects, ap
        self.idx = {v: k for k, v in ids.items()}
    def oid(self, c): return self.ids[c] if c is not None else ""
    def name(self, c): return self.objects.get(self.oid(c), ("", ""))[0] if c is not None else ""
    def section(self, c): return self.objects.get(self.oid(c), ("", "Unknown"))[1] if c is not None else ""
    def apv(self, c):
        v = self.ap.get(self.oid(c)); return round(float(v), 3) if v is not None else None
    def group(self, c): return GROUP_OF.get(self.oid(c), "")
    def label(self, c): return f"{self.oid(c)} ({self.name(c)})" if self.name(c) else self.oid(c)

def explain(gts, preds, raw, conf, info):
    """Every true object and every kept box gets one row: correct / box off / wrong ID / missed / extra + reason."""
    rows, used_g, used_p = [], set(), set()
    def row(result, reason, c, gc, p, v, k_gt=None, k_pred=None):
        rows.append(dict(result=result, reason=reason, predicted=info.oid(c), predicted_name=info.name(c),
                         true=info.oid(gc), true_name=info.name(gc),
                         section=info.section(gc if gc is not None else c),
                         class_AP=info.apv(gc if gc is not None else c),
                         confidence=round(p, 2) if p is not None else None, IoU=round(v, 2) if v is not None else None,
                         _gt=k_gt, _pred=k_pred))
    for i in sorted(range(len(preds)), key=lambda k: -preds[k][1]):          # 1. confident matches
        c, p, b = preds[i]; best, bj = 0.0, None
        for j, (gc, gb) in enumerate(gts):
            if j not in used_g and iou(b, gb) > best: best, bj = iou(b, gb), j
        if bj is not None and best >= 0.5:
            used_g.add(bj); used_p.add(i); gc = gts[bj][0]
            if gc == c: row("correct", "", c, gc, p, best, bj, i)
            else:
                why = (f"look-alike ({info.group(gc)}): called {info.label(c)}" if info.group(c) and info.group(c) == info.group(gc)
                       else f"confused with {info.label(c)}")
                if info.section(c) != info.section(gc): why += f"; different section ({info.section(c)})"
                row("wrong ID", why, c, gc, p, best, bj, i)
    for j, (gc, gb) in enumerate(gts):                                        # 2. right ID, box placed off
        if j in used_g: continue
        cand = [(iou(preds[i][2], gb), i) for i in range(len(preds)) if i not in used_p and preds[i][0] == gc]
        cand = [x for x in cand if x[0] >= 0.1]
        if cand:
            v, i = max(cand); used_g.add(j); used_p.add(i)
            row("box off", f"right ID, box placed off (overlap {v:.2f}, needs 0.50)", gc, gc, preds[i][1], v, j, i)
    for j, (gc, gb) in enumerate(gts):                                        # 3. missed: why?
        if j in used_g: continue
        low = [(q[1], q[0]) for q in raw if q[1] < conf and iou(q[2], gb) >= 0.5]
        same = [x for x in low if x[1] == gc]
        area = (gb[2] - gb[0]) * (gb[3] - gb[1]) / (TILE * TILE)
        if same:
            c = max(same)[0]; d = 2                                                  # never "0.10, below 0.10": add digits until they differ
            while d < 5 and f"{c:.{d}f}" == f"{conf:.{d}f}": d += 1
            why = f"seen at confidence {c:.{d}f}, below the threshold {conf:.{d}f}"
        elif low: x = max(low); why = f"seen weakly as {info.label(x[1])} ({x[0]:.2f}), below the threshold"
        elif area < 0.03: why = f"not found: very small object ({100 * area:.1f}% of its photo)"
        else: why = "not found" + (f" (look-alike group: {info.group(gc)})" if info.group(gc) else "")
        row("missed", why, None, gc, None, None, j, None)
    for i, (c, p, b) in enumerate(preds):                                     # 4. extra boxes: why?
        if i in used_p: continue
        near = [k for k in used_p if iou(preds[k][2], b) >= 0.3 or cell_of(preds[k][2]) == cell_of(b)]
        part = [g for g in gts if iou(g[1], b) > 0.05]
        if near: why = f"second box near {info.oid(preds[near[0]][0])}"
        elif part: why = f"box covers only part of {info.oid(part[0][0])}"
        else: why = "box on the background"
        row("extra", why, c, None, p, None, None, i)
    return rows

def tally(rows, n_gt):
    c = Counter(r["result"] for r in rows)
    return dict(n=n_gt, correct=c["correct"], box_off=c["box off"], wrong=c["wrong ID"], missed=c["missed"],
                extra=c["extra"], named=c["correct"] + c["box off"],
                fully=int(c["correct"] == n_gt and c["extra"] == 0 and c["box off"] == 0 and c["wrong ID"] == 0))

def bootstrap_ci(v, n=2000, seed=42):
    v = np.asarray(v, float)
    if not len(v): return (0.0, 0.0)
    rng = np.random.default_rng(seed); m = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


# ============================================================================================ grids
_cache = {}
def load_224(path):
    if path not in _cache:
        with Image.open(path) as im:
            _cache[path] = ImageOps.exif_transpose(im).convert("RGB").resize((TILE, TILE), Image.Resampling.BICUBIC)
    return _cache[path]

def pick_classes(pool_classes, n, mode, rng):
    """Distinct Object IDs for one grid: random, or whole look-alike groups first."""
    pool = sorted(pool_classes)
    if mode == "Look-alike challenge":
        groups = [[o for o in ids if o in pool] for ids in LOOKALIKE.values()]
        groups = [g for g in groups if len(g) >= 2]
        cand = sorted({o for g in groups for o in g})
        if len(cand) < n:
            raise ValueError(f"Only {len(cand)} look-alike objects available in this selection; choose a grid with at most {len(cand)} objects.")
        rng.shuffle(groups); out = []
        for g in groups:
            for o in g:
                if len(out) < n and o not in out: out.append(o)
        return out
    if len(pool) < n:
        raise ValueError(f"Only {len(pool)} different objects available in this selection; choose a grid with at most {len(pool)} objects.")
    return rng.sample(pool, n)

def make_grid(photos, cols, rows, sections=None, mode="Random", seed=None, objects=None):
    rng = random.Random(seed)
    p = photos
    if sections and objects:
        p = p[p.object_id.map(lambda o: objects.get(o, ("", "Unknown"))[1] in sections)]
    classes = pick_classes(set(p.object_id), cols * rows, mode, rng)
    canvas = Image.new("RGB", (cols * TILE, rows * TILE)); gts = []
    for k, oid in enumerate(classes):
        r = p[p.object_id == oid].sample(1, random_state=rng.randint(0, 10 ** 6)).iloc[0]
        x0, y0 = (k % cols) * TILE, (k // cols) * TILE
        canvas.paste(load_224(r.path), (x0, y0))
        gts.append((oid, (x0 + r.x1, y0 + r.y1, x0 + r.x2, y0 + r.y2)))
    return canvas, gts


# ============================================================================================ drawing
def font(size):
    for f in ("arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try: return ImageFont.truetype(f, size)
        except Exception: pass
    return ImageFont.load_default()

COL = {"correct": (230, 30, 30), "box off": (255, 150, 0), "wrong ID": (170, 0, 220), "extra": (120, 120, 120)}
def draw(img, preds, info, gts=None, rows=None):
    im = img.convert("RGB").copy(); d = ImageDraw.Draw(im); f = font(max(11, im.width // 70))
    status = {r["_pred"]: r["result"] for r in (rows or []) if r["_pred"] is not None}
    for gc, gb in (gts or []): d.rectangle(gb, outline=(0, 200, 0), width=3)
    for i, (c, p, b) in enumerate(preds):
        col = COL.get(status.get(i, "correct"), (230, 30, 30)) if rows else (230, 30, 30)
        d.rectangle(b, outline=col, width=3)
        lab = f"{info.oid(c)} {p:.2f}"; tw = d.textlength(lab, font=f)
        d.rectangle([b[0], b[1], b[0] + tw + 6, b[1] + f.size + 4], fill=col); d.text((b[0] + 3, b[1] + 1), lab, fill="white", font=f)
    return im

_refs = {}
def reference(oid, train, size=150):
    """A training photo of the class, cropped to its box: 'what this class looks like'. Large sizes use the original photo."""
    key = (oid, size)
    if key in _refs: return _refs[key]
    t = train[train.object_id == oid].sort_values("fname")
    if not len(t): _refs[key] = None; return None
    r = t.iloc[0]; m = 0.1 * max(r.x2 - r.x1, r.y2 - r.y1)
    if size <= 150:
        im = load_224(r.path); c = im.crop((max(0, r.x1 - m), max(0, r.y1 - m), min(TILE, r.x2 + m), min(TILE, r.y2 + m)))
    else:                                                     # full resolution: boxes are in 224-px coordinates, scale them
        with Image.open(r.path) as im0: im = ImageOps.exif_transpose(im0).convert("RGB")
        sx, sy = im.width / TILE, im.height / TILE
        c = im.crop((max(0, (r.x1 - m) * sx), max(0, (r.y1 - m) * sy), min(im.width, (r.x2 + m) * sx), min(im.height, (r.y2 + m) * sy)))
    _refs[key] = ImageOps.contain(c, (size, size)); return _refs[key]

def one_tile(img, preds, info, train, gts, r, size=150):
    """The object in this image | a training photo of the predicted class (| of the true class), panels of `size` px."""
    pi, gi = r.get("_pred"), r.get("_gt")
    box = gts[gi][1] if gi is not None else preds[pi][2]
    m = 0.25 * max(box[2] - box[0], box[3] - box[1], 20)
    x1 = min(max(0, box[0] - m), img.width - 2); y1 = min(max(0, box[1] - m), img.height - 2)     # clamp: never crash
    x2 = max(min(img.width, box[2] + m), x1 + 2); y2 = max(min(img.height, box[3] + m), y1 + 2)
    crop = img.crop((x1, y1, x2, y2)); crop = ImageOps.contain(crop, (size, size), Image.Resampling.LANCZOS)
    panels = [("in this image", crop)]
    if pi is not None: panels.append((f"model: {info.oid(preds[pi][0])}", reference(info.oid(preds[pi][0]), train, size)))
    if gi is not None and (pi is None or preds[pi][0] != gts[gi][0]):            # gts hold class indices
        panels.append((f"true: {info.oid(gts[gi][0])}", reference(info.oid(gts[gi][0]), train, size)))
    cw, fs = size + 10, max(12, size // 12)
    tile = Image.new("RGB", (cw * len(panels), size + 14 + fs + 8), "white"); dd = ImageDraw.Draw(tile); f = font(fs)
    for k, (cap, im) in enumerate(panels):
        if im is not None: tile.paste(im, (cw * k + (cw - im.width) // 2, 4 + (size - im.height) // 2))
        dd.text((cw * k + 5, size + 10), cap, fill="black", font=f)
    title = (r["result"] + (f": {r['reason']}" if r.get("reason") else "")) if r.get("result") and r["result"] != "detected" else \
            f"{info.label(preds[pi][0])} {preds[pi][1]:.2f} | {info.section(preds[pi][0])} | class AP {info.apv(preds[pi][0])}"
    return tile, title

def gallery(img, preds, info, train, gts=None, rows=None, limit=30):
    """One tile per object, errors first."""
    entries = rows if rows is not None else [dict(result="", reason="", _pred=i, _gt=None) for i in range(len(preds))]
    return [one_tile(img, preds, info, train, gts, r) for r in sorted(entries, key=lambda r: r["result"] == "correct")[:limit]]

def highlight(img_drawn, box):
    """Dim everything except one region (the object's 224-px cell in grids, else the box) and frame it."""
    base = img_drawn.convert("RGBA"); W, H = base.size
    if W % TILE == 0 and H % TILE == 0:
        cx, cy = cell_of(box); reg = (cx * TILE, cy * TILE, cx * TILE + TILE - 1, cy * TILE + TILE - 1)
    else:
        m = 0.15 * max(box[2] - box[0], box[3] - box[1], 20)
        reg = (max(0, box[0] - m), max(0, box[1] - m), min(W - 1, box[2] + m), min(H - 1, box[3] + m))
    shade = Image.new("RGBA", base.size, (0, 0, 0, 150)); mask = Image.new("L", base.size, 255)
    ImageDraw.Draw(mask).rectangle(reg, fill=0)
    out = Image.alpha_composite(base, Image.composite(shade, Image.new("RGBA", base.size, (0, 0, 0, 0)), mask)).convert("RGB")
    ImageDraw.Draw(out).rectangle(reg, outline=(255, 215, 0), width=max(4, W // 180))
    return out

def row_box(rows, k, preds, gts):
    """The box behind table row k: the true box if the row has one, else the predicted box."""
    if rows is None: return preds[k][2]
    r = rows[k]
    return gts[r["_gt"]][1] if r.get("_gt") is not None else preds[r["_pred"]][2]

def det_table(preds, info):
    return pd.DataFrame([dict(object_id=info.oid(c), name=info.name(c), section=info.section(c), class_AP=info.apv(c),
                              look_alike_group=info.group(c), confidence=round(p, 2),
                              x1=int(b[0]), y1=int(b[1]), x2=int(b[2]), y2=int(b[3]))
                         for c, p, b in sorted(preds, key=lambda t: (int(t[2][1] // TILE), t[2][0]))])

def save_downloads(img_drawn, table, stem="detections"):
    d = Path(tempfile.mkdtemp(prefix="m2demo_"))
    pi, pc, pj = d / f"{stem}.png", d / f"{stem}.csv", d / f"{stem}.json"
    img_drawn.save(pi); table.to_csv(pc, index=False); pj.write_text(table.to_json(orient="records", indent=2))
    return [str(pi), str(pc), str(pj)]

def sweep_plot(gts, raw, dup, W, H, use_cell, conf_now, info):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ts = np.round(np.arange(0.05, 0.91, 0.05), 2); series = {k: [] for k in ("correct", "missed", "extra", "wrong ID")}
    for t in ts:
        tl = tally(explain(gts, select(raw, t, dup, W, H, use_cell), raw, t, info), len(gts))
        series["correct"].append(tl["correct"]); series["missed"].append(tl["missed"] + tl["box_off"])
        series["extra"].append(tl["extra"] + tl["box_off"]); series["wrong ID"].append(tl["wrong"])
    fig, ax = plt.subplots(figsize=(6.5, 3.2), dpi=110)
    for k, st in zip(series, ("-o", "-s", "-^", "-x")): ax.plot(ts, series[k], st, markersize=3, label=k)
    ax.axvline(conf_now, color="black", linestyle="--", linewidth=1, label=f"current {conf_now:.2f}")
    ax.set_xlabel("confidence threshold"); ax.set_ylabel("objects"); ax.set_title("This grid at every threshold")
    ax.legend(fontsize=8, ncol=3); ax.grid(alpha=0.3); fig.tight_layout(); return fig



# ============================================================================================ server (Python standard library only)
import base64, io, threading, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("M2_PORT", "7861"))
FAKE = os.environ.get("M2_FAKE_MODEL") == "1"          # tests only: a stand-in model, no GPU needed

def b64img(img, q=90):
    b = io.BytesIO(); img.convert("RGB").save(b, "JPEG", quality=q); return base64.b64encode(b.getvalue()).decode()

def num(v):
    if v is None: return None
    if isinstance(v, (np.floating, float)): return None if np.isnan(v) else round(float(v), 4)
    if isinstance(v, (np.integer,)): return int(v)
    return v

class FakeModel:
    """Tests only: 'detects' the true objects with one miss, one box off, one wrong ID and one extra box."""
    def __init__(self, names): self.names = names; self.truth = []
    def predict(self, inp, imgsz, conf, max_det, verbose):
        c, p, b = [], [], []
        for k, (ci, box) in enumerate(self.truth):
            if k == len(self.truth) - 1 and len(self.truth) > 1: continue                     # missed
            box = list(box)
            if k == 1: box = [box[0], box[1] + 0.45 * (box[3] - box[1]), box[2], box[3] + 0.45 * (box[3] - box[1])]   # box off
            ci2 = (ci + 1) % len(self.names) if k == 2 else ci                                  # wrong ID
            c.append(ci2); p.append(0.93 - 0.02 * k); b.append(box)
        if self.truth: c.append(self.truth[0][0]); p.append(0.31); b.append([5, 5, 40, 40])    # extra (same cell as #0)
        class _B:
            def __init__(s, a): s.a = np.array(a, float)
            def cpu(s): return s
            def numpy(s): return s.a
        class _R: pass
        r = _R(); r.boxes = type("x", (), dict(cls=_B(c), conf=_B(p), xyxy=_B(b if b else np.zeros((0, 4)))))()
        return [r]

class App:
    def __init__(self):
        self.S = load_settings(); self.OBJ = load_objects(); self.AP = load_ap()
        self.test = load_photos(split="test"); self.train = load_photos(split="train", human_only=False)
        self.sections = sorted({v[1] for v in self.OBJ.values()}) or ["Unknown"]
        self.models = {}; self.lock = threading.Lock(); self.warm = dict(state="idle", error=None); self.warmed = set()
        self.grids, self.dets, self.jobs = {}, {}, {}
        self.available = [m for m, v in self.S["models"].items() if FAKE or Path(v["weights"]).exists()] or [self.S["best"]]

    def model(self, name):
        if name not in self.S["models"]: raise ValueError(f"unknown model {name}")
        if name not in self.models:
            if FAKE:
                ids = sorted(set(self.test.object_id) | set(self.OBJ)); self.models[name] = FakeModel({k: o for k, o in enumerate(ids)})
            else:
                from ultralytics import YOLO
                self.models[name] = YOLO(self.S["models"][name]["weights"])
        m = self.models[name]; return m, Info(m.names, self.OBJ, self.AP)

    def warmup(self):
        """Load the selected model onto the GPU and run one dummy detection, so the first real one is fast."""
        self.warm["state"] = "loading"
        try:
            with self.lock:
                model, _ = self.model(self.S["best"])
                if FAKE: model.truth = []
                run_model(model, Image.new("RGB", (TILE * 3, TILE * 3), (90, 90, 90)), self.S["mode"])
                self.warmed.add(self.S["best"])
            self.warm["state"] = "ready"
        except Exception as e:
            self.warm.update(state="error", error=str(e))

    def status(self, _=None): return dict(warm=self.warm["state"], error=self.warm["error"], warmed=sorted(self.warmed))

    @staticmethod
    def _keep(d, n=40):
        while len(d) > n: d.pop(next(iter(d)))

    # ------------------------------------------------------------------ endpoints
    def config(self, _=None):
        S = self.S
        return dict(best=S["best"], mode=S["mode"], dup=S["dup"], conf=S["conf"], models=self.available,
                    model_info={k: {m: num(v.get(m)) for m in ("val_map50", "val_map", "test_map50", "test_map", "test_precision", "test_recall")}
                                for k, v in S["models"].items()},
                    selection_rule=S.get("selection_rule", ""),
                    shapes={k: {m: num(v.get(m)) for m in ("images", "fully_correct", "object_recall", "precision")} for k, v in S.get("shapes", {}).items()},
                    sections=self.sections, grids=list(GRIDS), degrade=DEGRADE, groups=LOOKALIKE,
                    n_test_photos=int(len(self.test)), n_objects=int(self.test.object_id.nunique()), **self.dataset_counts(),
                    model_conf=S["model_conf"], boost=BOOST_NAME if BOOST_NAME in self.available else None, boost_eval=S["boost_eval"],
                    layouts=["grid", "rotated"] + [n for n, f in (("scattered", "scattered"), ("cut-outs", "cutout")) if (BOOST / "tests" / f / "truth.json").exists()])

    def dataset_counts(self):
        """Photos in the cleaned dataset (Milestone 1 file if present, else the box file) and photos with a reliable box."""
        try:
            m = pd.read_csv(MASTER)
            boxed = int((m.x1.notna() & m.status.isin(["verified", "fixed", "unreviewed_auto"])).sum())
            clean = PROJECT / "milestone1_outputs_v2" / "clean_dataset.csv"
            total = int(len(pd.read_csv(clean))) if clean.exists() else int(len(m))
            return dict(n_photos=total, n_boxed=boxed, n_objects_all=int(m.object_id.nunique()))
        except Exception:
            return dict(n_photos=None, n_boxed=None, n_objects_all=None)

    def scenes(self, name):
        """The evaluation's own test scenes (unseen test photos on unseen backgrounds), with their true boxes."""
        f = BOOST / "tests" / name / "truth.json"
        return json.loads(f.read_text()) if f.exists() else []

    def grid(self, q):
        layout = q.get("layout") or "grid"
        if layout in ("scattered", "cut-outs"):
            sc = self.scenes("scattered" if layout == "scattered" else "cutout")
            if not sc: raise ValueError(f"No {layout} scenes yet: run  python m2_boost.py build  in M2 extra.")
            seed = q.get("seed"); seed = int(seed) if str(seed or "").strip().isdigit() else random.randint(0, 10 ** 6)
            t = sc[seed % len(sc)]; img = Image.open(BOOST / "tests" / ("scattered" if layout == "scattered" else "cutout") / "images" / t["file"]).convert("RGB")
            gts_o = [(o, tuple(b)) for o, b in t["boxes"]]; gid = uuid.uuid4().hex[:10]
            self.grids[gid] = dict(img=img, gts_o=gts_o, cols=0, rows=0, seed=seed, kind="scene", size=layout, mix=layout, cells=False); self._keep(self.grids)
            return dict(id=gid, seed=seed, image=b64img(img), w=img.width, h=img.height, cols=0, rows=0, kind="scene", size=layout, mix=layout)
        size = q.get("size") or "3x3 (9 objects)"
        if size not in GRIDS: raise ValueError("unknown grid size")
        cols, rows = GRIDS[size]; mix = q.get("mix") or "Random"
        seed = q.get("seed"); seed = int(seed) if str(seed or "").strip().isdigit() else random.randint(0, 10 ** 6)
        img, gts_o = make_grid(self.test, cols, rows, q.get("sections") or self.sections, mix, seed, self.OBJ)
        if layout == "rotated":                                   # the same grid, turned 90/180/270 degrees: still cells, exact boxes
            k = 1 + seed % 3; W, H = img.size; img = img.rotate(90 * k, expand=True)
            def rb(b):
                x1, y1, x2, y2 = b; w, h = W, H
                for _ in range(k): x1, y1, x2, y2, w, h = y1, w - x2, y2, w - x1, h, w
                return (x1, y1, x2, y2)
            gts_o = [(o, rb(b)) for o, b in gts_o]
            if k % 2: cols, rows = rows, cols
            mix = f"{mix}, rotated {90 * k}"
        gid = uuid.uuid4().hex[:10]
        self.grids[gid] = dict(img=img, gts_o=gts_o, cols=cols, rows=rows, seed=seed, kind="grid", size=size, mix=mix); self._keep(self.grids)
        return dict(id=gid, seed=seed, image=b64img(img), w=img.width, h=img.height, cols=cols, rows=rows, kind="grid", size=size, mix=mix)

    def upload(self, q):
        raw = base64.b64decode(q["image"].split(",")[-1])
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
        if max(img.size) > 3000: img.thumbnail((3000, 3000))
        gid = uuid.uuid4().hex[:10]; W, H = img.size
        grid_like = W % TILE == 0 and H % TILE == 0
        self.grids[gid] = dict(img=img, gts_o=None, cols=W // TILE if grid_like else 0, rows=H // TILE if grid_like else 0,
                               seed=None, kind="upload", name=str(q.get("name", "upload"))); self._keep(self.grids)
        return dict(id=gid, image=b64img(img), w=W, h=H, cols=self.grids[gid]["cols"], rows=self.grids[gid]["rows"], kind="upload")

    def detect(self, q):
        g = self.grids.get(q.get("id"))
        if not g: raise ValueError("Make a grid or upload an image first.")
        name = q.get("model") or self.S["best"]; conf = float(q.get("conf", self.S["model_conf"].get(name, self.S["conf"])))
        use_cell = bool(q.get("cell", True)) and g.get("cells", True); deg = q.get("degrade") or "None"
        imgsz = self.S["model_imgsz"].get(name, IMGSZ)
        if deg not in DEGRADE: raise ValueError("unknown degradation")
        with self.lock:
            model, info = self.model(name)
            img = degrade(g["img"], deg); W, H = img.size
            gts = [(info.idx[o], b) for o, b in g["gts_o"]] if g["gts_o"] else None
            if FAKE: model.truth = [(c, tuple(v * imgsz / IMGSZ for v in b)) for c, b in (gts or [])]   # the stand-in answers in model pixels
            first = name not in self.warmed
            raw, ms = run_model(model, img, self.S["mode"], imgsz=imgsz); self.warmed.add(name)
        preds = select(raw, conf, self.S["dup"], W, H, use_cell)
        if gts is not None:
            rows = explain(gts, preds, raw, conf, info)
            rows = sorted(rows, key=lambda r: ["missed", "wrong ID", "box off", "extra", "correct"].index(r["result"]))
            t = tally(rows, len(gts))
            per = {}
            for r in rows:
                if r["true"]: per.setdefault(r["section"], [0, 0]); per[r["section"]][1] += 1; per[r["section"]][0] += r["result"] == "correct"
            sw = sweep_series(gts, raw, self.S["dup"], W, H, use_cell, info)
        else:
            preds = sorted(preds, key=lambda t: (int(t[2][1] // TILE), t[2][0]))
            rows = [dict(result="detected", reason="", predicted=info.oid(c), predicted_name=info.name(c), true="", true_name="",
                         section=info.section(c), class_AP=info.apv(c), confidence=round(p, 2), IoU=None, _gt=None, _pred=k)
                    for k, (c, p, b) in enumerate(preds)]
            t, per, sw = None, {}, None
        status = {r["_pred"]: r["result"] for r in rows if r.get("_pred") is not None}
        boxes = [dict(k=k, id=info.oid(c), name=info.name(c), section=info.section(c), ap=info.apv(c), group=info.group(c),
                      conf=round(float(p), 3), x1=num(b[0]), y1=num(b[1]), x2=num(b[2]), y2=num(b[3]), result=status.get(k, "detected"))
                 for k, (c, p, b) in enumerate(preds)]
        truths = [dict(k=k, id=info.oid(c), name=info.name(c), x1=num(b[0]), y1=num(b[1]), x2=num(b[2]), y2=num(b[3]))
                  for k, (c, b) in enumerate(gts)] if gts else []
        did = uuid.uuid4().hex[:10]
        self.dets[did] = dict(img=img, preds=preds, gts=gts, rows=rows, info=info, model=name, grid=g, deg=deg); self._keep(self.dets)
        out_rows = [{**{k: num(v) for k, v in r.items() if not k.startswith("_")}, "gt": r.get("_gt"), "pred": r.get("_pred")} for r in rows]
        return dict(det=did, model=name, conf=conf, degrade=deg, ms=round(ms, 1), first_call=first, w=W, h=H, cols=g["cols"], rows=g["rows"],
                    image=b64img(img) if deg != "None" else None, boxes=boxes, truths=truths, table=out_rows,
                    tally=t, per_section=per, sweep=sw, has_truth=gts is not None)

    def tile(self, did, k, size=150):
        d = self.dets.get(did)
        if not d: raise ValueError("unknown detection")
        rows = d["rows"]
        if not 0 <= k < len(rows): raise ValueError("unknown object")
        img, cap = one_tile(d["img"], d["preds"], d["info"], self.train, d["gts"], rows[k], max(120, min(480, int(size))))
        return dict(image=b64img(img, 92), caption=cap)

    def download(self, did, kind):
        d = self.dets.get(did)
        if not d: raise ValueError("unknown detection")
        info = d["info"]
        if kind == "png":
            b = io.BytesIO(); draw(d["img"], d["preds"], info, d["gts"], d["rows"] if d["gts"] else None).save(b, "PNG")
            return b.getvalue(), "image/png", f"detections_{did}.png"
        tab = (pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in d["rows"]]) if d["gts"]
               else det_table(d["preds"], info))
        if kind == "csv": return tab.to_csv(index=False).encode(), "text/csv", f"detections_{did}.csv"
        if kind == "json": return tab.to_json(orient="records", indent=2).encode(), "application/json", f"detections_{did}.json"
        raise ValueError("unknown download type")

    def snapshot(self, did):
        d = self.dets.get(did)
        if not d: raise ValueError("unknown detection")
        info, g = d["info"], d["grid"]
        ann = draw(d["img"], d["preds"], info, d["gts"], d["rows"] if d["gts"] else None)
        ann = ImageOps.contain(ann, (900, 900)); P = 560; W = ann.width + P + 60; H = max(ann.height + 120, 760)
        out = Image.new("RGB", (W, H), (4, 7, 10)); dr = ImageDraw.Draw(out)
        ft, fb, fs = font(30), font(20), font(16)
        dr.text((30, 26), "THE EYE – Explainable YOLO Explorer", fill=(57, 255, 136), font=ft)
        sub = (f"{g['cols']}x{g['rows']} grid · seed {g['seed']} · {g.get('mix', '')}" if g["kind"] == "grid" else f"upload {d['img'].width}x{d['img'].height}")
        dr.text((30, 66), f"{sub} · model {d['model']}" + (f" · {d['deg']}" if d["deg"] != "None" else ""), fill=(127, 165, 148), font=fs)
        out.paste(ann, (30, 100)); x0, y = ann.width + 60, 100
        if d["gts"]:
            t = tally(d["rows"], len(d["gts"]))
            dr.text((x0, y), f"{t['correct']} / {t['n']} correct", fill=(220, 255, 240), font=font(40)); y += 56
            dr.text((x0, y), f"{t['named']} / {t['n']} named correctly · box off {t['box_off']} · wrong ID {t['wrong']} · missed {t['missed']} · extra {t['extra']}",
                    fill=(127, 165, 148), font=fs); y += 40
            errs = [r for r in d["rows"] if r["result"] != "correct"]
            dr.text((x0, y), "ERRORS" if errs else "NO ERRORS", fill=(57, 255, 136), font=fb); y += 32
            col = {"missed": (34, 228, 255), "wrong ID": (255, 79, 216), "box off": (255, 176, 32), "extra": (139, 151, 168)}
            for r in errs[:14]:
                dr.rectangle([x0, y + 5, x0 + 10, y + 15], fill=col.get(r["result"], (200, 200, 200)))
                dr.text((x0 + 20, y), f"{r['true'] or r['predicted']} · {r['result']}", fill=(220, 255, 240), font=fs); y += 22
                words, line = r["reason"].split(), ""
                for w_ in words:
                    if dr.textlength(line + " " + w_, font=fs) > P - 30: dr.text((x0 + 20, y), line.strip(), fill=(127, 165, 148), font=fs); y += 20; line = ""
                    line += " " + w_
                if line.strip(): dr.text((x0 + 20, y), line.strip(), fill=(127, 165, 148), font=fs); y += 20
                y += 8
        else:
            dr.text((x0, y), f"{len(d['preds'])} objects found", fill=(220, 255, 240), font=font(36)); y += 52
            for c, p_, b in d["preds"][:24]:
                dr.text((x0, y), f"{info.oid(c)}  {info.name(c)}  {p_:.2f}", fill=(127, 165, 148), font=fs); y += 22
        b = io.BytesIO(); out.save(b, "PNG"); return b.getvalue()

    def stress_start(self, q):
        jid = uuid.uuid4().hex[:8]; n = max(1, min(500, int(q.get("n", 100))))
        self.jobs[jid] = dict(done=0, total=n, result=None, error=None); self._keep(self.jobs, 10)
        threading.Thread(target=self._stress, args=(jid, q, n), daemon=True).start()
        return dict(job=jid, total=n)

    def _stress(self, jid, q, n):
        J = self.jobs[jid]
        try:
            size = q.get("size") or "5x5 (25 objects)"; cols, rows_n = GRIDS[size]
            name = q.get("model") or self.S["best"]; conf = float(q.get("conf", self.S["model_conf"].get(name, self.S["conf"]))); use_cell = bool(q.get("cell", True))
            deg = q.get("degrade") or "None"; mix = q.get("mix") or "Random"; secs = q.get("sections") or self.sections
            tot, fully, ms_all, miss_by, pairs, reasons = Counter(), [], [], Counter(), Counter(), Counter()
            for g in range(n):
                img, gts_o = make_grid(self.test, cols, rows_n, secs, mix, 1000 + g, self.OBJ)
                with self.lock:
                    model, info = self.model(name)
                    img = degrade(img, deg); W, H = img.size; gts = [(info.idx[o], b) for o, b in gts_o]
                    imgsz = self.S["model_imgsz"].get(name, IMGSZ)
                    if FAKE: model.truth = [(c, tuple(v * imgsz / IMGSZ for v in b)) for c, b in gts]
                    raw, ms = run_model(model, img, self.S["mode"], imgsz=imgsz)
                ms_all.append(ms)
                rws = explain(gts, select(raw, conf, self.S["dup"], W, H, use_cell), raw, conf, info); t = tally(rws, len(gts))
                for k in ("n", "correct", "named", "box_off", "wrong", "missed", "extra"): tot[k] += t[k]
                fully.append(t["fully"])
                for r in rws:
                    if r["result"] in ("missed", "box off", "wrong ID") and r["true"]: miss_by[f"{r['true']} {r['true_name']}".strip()] += 1
                    if r["result"] == "wrong ID": pairs[f"{r['true']} -> {r['predicted']}"] += 1
                    if r["result"] != "correct": reasons[re.sub(r"[\d.]+", "#", r["reason"].split(";")[0]) or r["result"]] += 1
                J["done"] = g + 1
            lo, hi = bootstrap_ci(fully)
            J["result"] = dict(n_grids=n, size=size, model=name, mix=mix, degrade=deg, sections=secs, totals=dict(tot),
                               fully=float(np.mean(fully)), ci=[lo, hi], ms=float(np.median(ms_all)),
                               hardest=miss_by.most_common(8), confusions=pairs.most_common(8), reasons=reasons.most_common(6))
        except Exception as e:
            J["error"] = str(e)

def sweep_series(gts, raw, dup, W, H, use_cell, info):
    ts = [round(float(t), 2) for t in np.arange(0.05, 0.91, 0.05)]; out = dict(t=ts, correct=[], missed=[], extra=[], wrong=[])
    for t in ts:
        tl = tally(explain(gts, select(raw, t, dup, W, H, use_cell), raw, t, info), len(gts))
        out["correct"].append(tl["correct"]); out["missed"].append(tl["missed"] + tl["box_off"])
        out["extra"].append(tl["extra"] + tl["box_off"]); out["wrong"].append(tl["wrong"])
    return out


def make_handler(app):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def _send(self, code, body, ctype="application/json", extra=None):
            data = body if isinstance(body, bytes) else (json.dumps(body) if ctype == "application/json" else body).encode()
            self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items(): self.send_header(k, v)
            self.end_headers(); self.wfile.write(data)
        def do_GET(self):
            try:
                p = self.path.split("?")[0].strip("/").split("/")
                if p == [""]: return self._send(200, PAGE, "text/html; charset=utf-8")
                if p == ["api", "config"]: return self._send(200, app.config())
                if p == ["api", "status"]: return self._send(200, app.status())
                if len(p) == 3 and p[:2] == ["api", "snapshot"]:
                    return self._send(200, app.snapshot(p[2]), "image/png", {"Content-Disposition": f'attachment; filename="snapshot_{p[2]}.png"'})
                if len(p) == 4 and p[:2] == ["api", "tile"]:
                    from urllib.parse import parse_qs
                    q = parse_qs(self.path.split("?", 1)[1]) if "?" in self.path else {}
                    return self._send(200, app.tile(p[2], int(p[3]), int((q.get("size") or ["150"])[0])))
                if len(p) == 3 and p[:2] == ["api", "stress"]:
                    j = app.jobs.get(p[2]); return self._send(200 if j else 404, j or dict(error="unknown job"))
                if len(p) == 4 and p[:2] == ["api", "download"]:
                    data, ctype, fname = app.download(p[2], p[3])
                    return self._send(200, data, ctype, {"Content-Disposition": f'attachment; filename="{fname}"'})
                self._send(404, dict(error="not found"))
            except Exception as e:
                self._send(400, dict(error=str(e)))
        def do_POST(self):
            try:
                q = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                route = {"/api/grid": app.grid, "/api/upload": app.upload, "/api/detect": app.detect, "/api/stress": app.stress_start}.get(self.path)
                if not route: return self._send(404, dict(error="not found"))
                self._send(200, route(q))
            except Exception as e:
                self._send(400, dict(error=str(e)))
    return H

def main():
    app = App()
    host = os.environ.get("M2_HOST", "127.0.0.1")          # "0.0.0.0" = reachable from other devices on your network
    srv = ThreadingHTTPServer((host, PORT), make_handler(app))
    threading.Thread(target=app.warmup, daemon=True).start()          # load the model while the page opens
    url = f"http://127.0.0.1:{PORT}"
    if host != "127.0.0.1":
        import socket
        try: ip = socket.gethostbyname(socket.gethostname())
        except Exception: ip = "<this computer's IP>"
        print(f"Shared on your network: http://{ip}:{PORT}  (only on a trusted network; the app shows dataset photos)")
    print(f"The EYE (Explainable YOLO Explorer) running at {url}  (model {app.S['best']}; press Ctrl+C to stop)")
    if os.environ.get("M2_NO_BROWSER") != "1":
        import webbrowser; threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try: srv.serve_forever()
    except KeyboardInterrupt: pass


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>The EYE v2 – Explainable YOLO Explorer</title>
<script>try { if (localStorage.getItem("m2theme") === "light") document.documentElement.classList.add("light"); } catch (e) { }</script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&family=Barlow:wght@400;500;600&family=Barlow+Condensed:wght@500;600;700&family=Saira+Stencil+One&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#04070a; --tile:rgba(8,16,13,.68); --tile2:rgba(10,22,17,.86);
  --line:rgba(57,255,136,.18); --line2:rgba(57,255,136,.5);
  --acc:#39ff88; --acc-d:#16c47a; --cyan:#22e4ff; --amber:#ffb020; --mag:#ff4fd8; --grey:#8b97a8; --gold:#ffd84a;
  --txt:#dcfff0; --dim:#7fa594; --dim2:#4d6b5e;
  --mono:"JetBrains Mono",ui-monospace,Consolas,Menlo,monospace; --sans:Inter,system-ui,"Segoe UI",Roboto,sans-serif;
}
*{box-sizing:border-box}
html,body{margin:0;min-height:100%;background:var(--bg);color:var(--txt);font-family:var(--sans)}
#bg{position:fixed;inset:0;z-index:0;display:block}
/* ---------------------------------------------------------------- dock */
.dock{position:fixed;left:18px;top:50%;transform:translateY(-50%);z-index:5;display:flex;flex-direction:column;gap:10px;
  padding:12px 9px;background:var(--tile);border:1px solid var(--line);border-radius:18px;backdrop-filter:blur(7px);box-shadow:0 18px 50px rgba(0,0,0,.55)}
.dbtn{width:62px;display:flex;flex-direction:column;align-items:center;gap:5px;padding:9px 4px;border-radius:12px;border:1px solid transparent;background:none;
  color:var(--dim);font:500 9.5px var(--mono);letter-spacing:.08em;text-transform:uppercase;cursor:pointer;transition:all .15s}
.dbtn svg{width:22px;height:22px;stroke:currentColor;fill:none;stroke-width:1.6}
.dbtn:hover{color:var(--acc);border-color:var(--line2);background:rgba(57,255,136,.06)}
.dbtn.on{color:var(--acc)}
.dsep{height:1px;background:var(--line);margin:2px 6px}
.dock.right{left:auto;right:18px}
/* ---------------------------------------------------------------- main column */
.main{position:relative;z-index:1;min-height:100vh;display:flex;flex-direction:column;align-items:center;padding:16px 110px 34px 110px}
.top{width:var(--ew);display:grid;grid-template-columns:1fr auto 1fr;align-items:center;gap:18px;margin-bottom:6px;font-family:var(--mono)}
.top h1,.logo{margin:0;display:flex;flex-direction:column;align-items:center;gap:4px;white-space:nowrap;font-family:var(--mono)}
.logo .t1{font-size:22px;letter-spacing:.46em;font-weight:500;color:var(--acc);text-shadow:0 0 18px rgba(57,255,136,.5);text-transform:uppercase;margin-right:-.46em}
.logo .t1 b{color:var(--txt);font-weight:700}
.logo .t2{font-size:10.5px;letter-spacing:.34em;font-weight:500;color:var(--dim);text-transform:uppercase;margin-right:-.34em}
.logo .t2 i{font-style:normal;color:var(--acc);font-weight:700;text-shadow:0 0 10px rgba(57,255,136,.6)}
/* intro: the title appears in the pupil, letter by letter; the E's blink; then it flies to the top */
.intrologo{position:absolute;left:50%;top:50%;z-index:2;pointer-events:none;transform-origin:50% 50%}
.intrologo{--txt:#e6fff2;--dim:#9fd3b8;--acc:#39ff88}          /* the pupil is dark in both themes */
.intrologo .lt{display:inline-block;opacity:0;transform:translateY(14px) scale(.4)}
.intrologo.go .lt{animation:ltpop .55s cubic-bezier(.3,1.6,.5,1) forwards;animation-delay:var(--d)}
@keyframes ltpop{0%{opacity:0;transform:translateY(14px) scale(.4)}100%{opacity:1;transform:none}}
.intrologo.blink .lt.ee{animation:ltpop .01s forwards,eeblink .36s ease-in-out .02s 2}
@keyframes eeblink{50%{transform:scaleY(.12)}}
.intrologo .t2{opacity:0;transform:translateY(6px);transition:opacity .6s ease,transform .6s ease}
.intrologo.sub .t2{opacity:1;transform:none}
.flylogo{position:fixed;z-index:46;pointer-events:none;transform-origin:50% 50%}
body.intro .pempty{opacity:0} .pempty{transition:opacity .6s ease}
.top .meta{justify-self:end} .top .meta span[data-tip]{cursor:help;border-bottom:1px dotted var(--line2)}
.top .meta{font-size:10.5px;color:var(--dim);white-space:nowrap}
/* ---------------------------------------------------------------- eyebrow */
.brow{position:relative;width:calc(var(--ew) * 1.2);height:220px}
.brow svg.arc{position:absolute;inset:0;width:100%;height:100%;overflow:visible;pointer-events:none}
.browband{fill:url(#browfill);stroke:rgba(57,255,136,.22);stroke-width:1}
.bf1{stop-color:rgba(57,255,136,.13)} .bf2{stop-color:rgba(57,255,136,.04)}
.hair{stroke:rgba(57,255,136,.26);stroke-width:1.1;stroke-linecap:round}
.browedge{fill:none;stroke:rgba(57,255,136,.55);stroke-width:1.6;filter:drop-shadow(0 0 6px rgba(57,255,136,.55))}
.seg{position:absolute;top:0;transform-origin:50% 50%;background:var(--tile);border:1px solid var(--line);border-radius:14px;padding:10px 11px;backdrop-filter:blur(7px);
  box-shadow:0 14px 40px rgba(0,0,0,.5);display:flex;flex-direction:column;gap:7px;transition:border-color .2s,box-shadow .2s}
.seg:hover{border-color:var(--line2);box-shadow:0 14px 44px rgba(0,0,0,.55),0 0 22px rgba(57,255,136,.12)}
.seg h3{margin:0;font:600 10.5px var(--mono);letter-spacing:.22em;color:var(--acc);text-transform:uppercase;display:flex;justify-content:space-between}
.seg h3 span{color:var(--dim2);font-weight:500;letter-spacing:.06em}
.lbl{font:10px var(--mono);color:var(--dim);letter-spacing:.05em}
.chips{display:flex;flex-wrap:wrap;gap:5px}
.chip{border:1px solid var(--line);background:rgba(0,0,0,.25);color:var(--dim);border-radius:999px;padding:4px 8px;font:500 10.5px var(--mono);cursor:pointer;
  transition:all .15s;user-select:none}
.chip:hover{color:var(--txt);border-color:var(--line2)}
.chip.on{color:var(--acc);border-color:var(--acc);background:rgba(57,255,136,.12);box-shadow:0 0 10px rgba(57,255,136,.18)}
select,input[type=text]{width:100%;background:rgba(0,0,0,.35);color:var(--txt);border:1px solid var(--line);border-radius:9px;padding:6px 8px;font:500 11px var(--mono);outline:none}
select:focus,input:focus{border-color:var(--acc)}
input[type=range]{width:100%;accent-color:var(--acc);margin:0}
.toggle{display:flex;align-items:center;gap:6px;font:10px var(--mono);color:var(--dim);cursor:pointer;user-select:none}
.toggle input{accent-color:var(--acc);margin:0}
.mini{display:flex;gap:5px}
.seg>.mbtn{flex:none}
.mbtn{flex:1;white-space:nowrap;border:1px solid var(--line);background:rgba(0,0,0,.3);color:var(--dim);border-radius:8px;padding:5px 4px;font:500 9.5px var(--mono);letter-spacing:.08em;
  text-transform:uppercase;cursor:pointer}
.mbtn:hover{color:var(--acc);border-color:var(--acc)}
/* ---------------------------------------------------------------- the eye */
.eyewrap{position:relative;width:var(--ew);height:var(--eh);margin:0 auto}
.eyesvg,.lidsvg{position:absolute;inset:0;width:100%;height:100%;overflow:visible}
.lidsvg{z-index:3;pointer-events:none}
.sclera{fill:url(#scl)}
.almond{fill:none;stroke:var(--acc);stroke-width:2;filter:drop-shadow(0 0 10px rgba(57,255,136,.55))}
.irisbase{fill:url(#irisg)}
.irisline{stroke:rgba(57,255,136,.06);stroke-width:1}
.gauge path{fill:none;stroke-width:var(--gw);stroke-linecap:butt;opacity:0;transition:opacity .25s}
.gauge path.show{opacity:1}
.gauge .tick{stroke:rgba(57,255,136,.18)}
.spin{transform-box:fill-box;transform-origin:center;animation:spin 1.6s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.glint{fill:rgba(255,255,255,.7);filter:blur(1px)}
.lid .skin{fill:#071210}
.lid .edge{stroke:var(--acc);stroke-width:2.5;filter:drop-shadow(0 0 8px rgba(57,255,136,.8))}
.pupil{position:absolute;z-index:2;border-radius:50%;overflow:hidden;background:radial-gradient(circle at 50% 50%,#020504 60%,#000 100%);
  box-shadow:0 0 0 var(--pring,4px) rgba(57,255,136,.55),0 0 18px rgba(57,255,136,.35),0 0 40px rgba(57,255,136,.18) inset;cursor:zoom-in;transition:transform .35s cubic-bezier(.2,.9,.3,1.4)}
.pupil.narrow{transform:scale(.9)}
.lens{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);display:flex;align-items:center;justify-content:center}
.zoomer{position:relative;transition:transform .55s cubic-bezier(.2,.8,.2,1);transform-origin:0 0}
.zoomer img{display:block;width:100%;height:100%}
.zoomer svg{position:absolute;inset:0;width:100%;height:100%;overflow:visible}
.pempty{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;text-align:center;font:11px/1.7 var(--mono);color:var(--dim2);padding:22%}
.scan{position:absolute;left:-5%;right:-5%;height:3px;top:0;background:linear-gradient(90deg,transparent,var(--acc),transparent);
  box-shadow:0 0 22px 4px rgba(57,255,136,.55);animation:scan .75s cubic-bezier(.5,0,.5,1) forwards;pointer-events:none;z-index:2}
@keyframes scan{from{top:0;opacity:1}to{top:100%;opacity:.2}}
.cover{position:absolute;background:#010302;animation:uncover .4s ease-out forwards;pointer-events:none}
@keyframes uncover{from{opacity:1}to{opacity:0}}
/* boxes (shared by pupil + stage) */
.box rect.f{fill:none;stroke-width:2.6;vector-effect:non-scaling-stroke}
.det{opacity:0;transform-box:fill-box;transform-origin:center;animation:boxin .38s cubic-bezier(.2,.9,.3,1.2) forwards;animation-delay:var(--d,0s)}
@keyframes boxin{from{opacity:0;transform:scale(.88)}to{opacity:1;transform:scale(1)}}
.det.err rect.f{animation:pulse .5s ease-in-out 2;animation-delay:calc(var(--d,0s) + .42s)}
@keyframes pulse{50%{stroke-width:6}}
.det.missed rect.f{animation:blink .45s steps(1) 4;animation-delay:calc(var(--d,0s) + .42s)}
@keyframes blink{50%{opacity:.15}}
.b-correct rect.f,.b-detected rect.f{stroke:var(--acc);filter:drop-shadow(0 0 4px rgba(57,255,136,.6))}
.b-box-off rect.f{stroke:var(--amber);stroke-dasharray:9 5}
.b-wrong-ID rect.f{stroke:var(--mag);filter:drop-shadow(0 0 5px rgba(255,79,216,.6))}
.b-extra rect.f{stroke:var(--grey);stroke-dasharray:2 4}
.b-missed rect.f{stroke:var(--cyan);stroke-dasharray:6 4}
.truth rect{fill:none;stroke:rgba(255,255,255,.55);stroke-width:1.4;stroke-dasharray:4 4;vector-effect:non-scaling-stroke}
.lab rect{fill:var(--c)}.lab text{font:700 11px var(--mono);fill:#02130a}
.b-correct,.b-detected{--c:var(--acc)}.b-box-off{--c:var(--amber)}.b-wrong-ID{--c:var(--mag)}.b-extra{--c:var(--grey)}.b-missed{--c:var(--cyan)}
.box{cursor:pointer}
.box rect.f{pointer-events:all}            /* the whole box area is clickable, not only its outline */
.hl{fill:rgba(0,0,0,.62);fill-rule:evenodd;pointer-events:none}
.hlf{fill:none;stroke:var(--gold);stroke-width:4;vector-effect:non-scaling-stroke;filter:drop-shadow(0 0 8px rgba(255,216,74,.8))}
/* ---------------------------------------------------------------- detect */
.detrow{display:flex;flex-direction:column;align-items:center;gap:8px;margin:8px 0 2px}
/* the iris ring is the DETECT button */
.detect{cursor:pointer;outline:none}
.detect .ringhit{fill:transparent;pointer-events:all}
.detect .ringglow{fill:none;stroke:rgba(57,255,136,0);pointer-events:none;transition:stroke .25s}
.detect:hover .ringglow,.detect:focus-visible .ringglow{stroke:rgba(57,255,136,.20)}
.detect:active .ringglow{stroke:rgba(57,255,136,.32)}
.detect .detlabel{font-family:var(--mono);font-weight:800;letter-spacing:.5em;fill:#04381f;pointer-events:none;user-select:none;
  paint-order:stroke;stroke:rgba(170,255,205,.75);stroke-width:3.2px;stroke-linejoin:round}
.detect:hover .detlabel{fill:#022814;stroke:rgba(200,255,225,.95)}
.detect.dis{pointer-events:none} .detect.dis .detlabel{opacity:.35}
.status{font:10.5px var(--mono);color:var(--dim);letter-spacing:.06em;position:relative;z-index:2}
.next{animation:nextglow 1.6s ease-in-out infinite}
@keyframes nextglow{50%{box-shadow:0 0 0 1px var(--acc),0 0 30px rgba(57,255,136,.45)}}
.detect.next .ringglow{animation:nextring 1.6s ease-in-out infinite}
@keyframes nextring{0%,100%{stroke:rgba(57,255,136,.04)}50%{stroke:rgba(57,255,136,.30)}}
/* ---------------------------------------------------------------- review matrix */
.review{width:var(--ew);display:grid;grid-template-columns:repeat(4,1fr);gap:14px;position:relative;z-index:2;margin-top:-6px}
.review .tile{transform-origin:50% 0}
.review{transition:grid-template-columns .4s cubic-bezier(.2,.8,.2,1)}
.review.wide{grid-template-columns:.95fr 2.1fr .95fr .62fr}       /* Results widens when a row is expanded; the sweep gives way */
body.calm .review{transition:none}
.beam{position:absolute;left:0;top:0;width:100%;pointer-events:none;z-index:1;overflow:visible}
.beam .cone{fill:url(#beamg)} .beam .cardbeam{fill:url(#beamg)} .beam .bg1{stop-color:rgba(57,255,136,.20)} .beam .bg2{stop-color:rgba(57,255,136,.06)} .beam .bg3{stop-color:rgba(57,255,136,0)}
.beam .ray{stroke:url(#rayg);stroke-width:1.4;opacity:.5;animation:shimmer 3.2s ease-in-out infinite}
.beam .rg1{stop-color:rgba(57,255,136,.55)} .beam .rg2{stop-color:rgba(57,255,136,0)}
@keyframes shimmer{0%,100%{opacity:.18}50%{opacity:.6}}
.beam.beam-pulse .cone{animation:beamflash 1.1s ease-out}
@keyframes beamflash{0%{opacity:1}25%{opacity:2.2;filter:brightness(1.8)}100%{opacity:1}}
.tile.fall{animation:fall .75s cubic-bezier(.25,1.25,.4,1) both;animation-delay:var(--fd,0s)}
@keyframes fall{0%{translate:0 -70px;opacity:0}60%{opacity:1}100%{translate:0 0;opacity:1}}
.rtitle{grid-column:1 / -1;font:500 10.5px var(--mono);letter-spacing:.3em;color:var(--dim2);display:flex;align-items:center;gap:10px}
.rtitle::after{content:"";flex:1;height:1px;background:var(--line)}
.tile{position:relative;height:236px;background:var(--tile);border:1px solid var(--line);border-radius:16px;padding:13px 14px;display:flex;flex-direction:column;
  overflow:hidden;backdrop-filter:blur(7px);box-shadow:0 18px 50px rgba(0,0,0,.55);transition:border-color .25s,box-shadow .25s}
.tile:hover{border-color:var(--line2)}
.th{display:flex;align-items:center;gap:9px;margin-bottom:9px}
.th .ix{font:500 10px var(--mono);color:var(--dim2)}
.th h3{margin:0;font:600 11px var(--mono);letter-spacing:.22em;text-transform:uppercase;color:var(--acc)}
.th .sp{flex:1}.th .hint{font:10px var(--mono);color:var(--dim2)}
.tb{flex:1;display:flex;flex-direction:column;gap:8px;min-height:0}
.big{font:700 34px var(--mono);line-height:1}
.score{display:flex;align-items:baseline;gap:6px}.score .of{font:500 14px var(--mono);color:var(--dim)}
.bar{height:6px;border-radius:9px;background:rgba(255,255,255,.06);overflow:hidden}
.bar i{display:block;height:100%;width:0;background:linear-gradient(90deg,var(--acc-d),var(--acc));box-shadow:0 0 12px var(--acc);transition:width .9s cubic-bezier(.2,.8,.2,1)}
.verdict{font:600 11px var(--mono);letter-spacing:.1em}
.stats{display:grid;grid-template-columns:repeat(2,1fr);gap:4px 10px;font:11px var(--mono);color:var(--dim)}.stats b{color:var(--txt)}
.list{flex:1;overflow:auto;display:flex;flex-direction:column;gap:5px;padding-right:3px;min-height:0}
.list::-webkit-scrollbar{width:5px}.list::-webkit-scrollbar-thumb{background:var(--line);border-radius:5px}
.item{display:flex;gap:8px;align-items:flex-start;padding:6px 8px;border-radius:9px;border:1px solid transparent;cursor:pointer;background:rgba(0,0,0,.2);font:10.5px/1.35 var(--mono)}
.item:hover{border-color:var(--line2)}.item.sel{border-color:var(--acc);background:rgba(57,255,136,.08)}
.dot{flex:0 0 8px;height:8px;border-radius:2px;margin-top:3px;background:var(--c)}
.r-correct,.r-detected{--c:var(--acc)}.r-box-off{--c:var(--amber)}.r-wrong-ID{--c:var(--mag)}.r-extra{--c:var(--grey)}.r-missed{--c:var(--cyan)}
.item .t{color:var(--txt)}.item .w{color:var(--dim)}
.insp{flex:1;display:flex;flex-direction:column;gap:6px;min-height:0}
.ftabs{display:flex;gap:4px} .chip.sm{padding:2px 7px;font-size:9.5px}
.enl{background:none;border:1px solid var(--line);color:var(--dim);border-radius:7px;padding:1px 7px;font:12px var(--mono);cursor:pointer;line-height:1.4}
.enl:hover{color:var(--acc);border-color:var(--acc)}
.item .ib{flex:1;min-width:0}
.item .sent{color:var(--txt);line-height:1.4}
.xbtn{flex:0 0 auto;align-self:flex-start;width:20px;height:20px;border-radius:6px;border:1px solid var(--line2);background:none;color:var(--acc);font:700 12px var(--mono);
  cursor:pointer;line-height:1;padding:0}
.xbtn:hover{background:rgba(57,255,136,.12)}
.xtab{width:100%;border-collapse:collapse;margin-top:6px;font:9.5px/1.45 var(--mono)}
.xtab td{padding:2px 4px;border-top:1px solid var(--line);vertical-align:top;word-break:break-word}
.xtab td:first-child{color:var(--dim);white-space:nowrap;width:1%}
.xtab td:last-child{color:var(--txt)}
.inav{display:flex;align-items:center;gap:6px;font:10px var(--mono);color:var(--dim)}
.inav .sp{flex:1}
.insp .cap{color:var(--txt)}
.sctl{display:flex;align-items:center;gap:5px;font:10px var(--mono);color:var(--dim);letter-spacing:.06em;text-transform:uppercase}
.sctl select,.sctl input{width:auto;padding:4px 6px;font-size:10.5px;border-radius:7px}
.stagebar{flex-wrap:wrap;row-gap:8px}
.legend span[data-tip]{cursor:help}
.bigview{width:min(1100px,95vw)}
.bigimg{width:100%;max-height:58vh;object-fit:contain;border-radius:10px;background:#fff}
.bignav{display:flex;align-items:center;gap:10px;margin:10px 0;font:11px var(--mono);color:var(--dim)}
.bignav .sp{flex:1}
.insp img{width:100%;flex:1;min-height:0;object-fit:contain;border-radius:9px;background:#fff}
.insp .cap{font:10px/1.45 var(--mono);color:var(--dim)}
.sweep{width:100%;flex:1}
.glitch{animation:glitch .36s steps(1,end) 1}
@keyframes glitch{
  0%{clip-path:inset(0 0 0 0);translate:0}
  12%{clip-path:inset(8% 0 62% 0);translate:-4px 0;box-shadow:-3px 0 0 rgba(255,46,120,.65),3px 0 0 rgba(34,228,255,.65)}
  24%{clip-path:inset(58% 0 12% 0);translate:4px 0}
  36%{clip-path:inset(30% 0 44% 0);translate:-2px 1px;box-shadow:2px 0 0 rgba(255,46,120,.6),-2px 0 0 rgba(34,228,255,.6)}
  48%{clip-path:inset(72% 0 4% 0);translate:3px 0}
  60%{clip-path:inset(0 0 0 0);translate:-1px 0}
  100%{clip-path:inset(0 0 0 0);translate:0}}
/* ---------------------------------------------------------------- hints */
.hintdot{position:absolute;z-index:6;width:26px;height:26px;border-radius:50%;background:var(--acc);color:#02130a;font:700 12px var(--mono);
  display:flex;align-items:center;justify-content:center;box-shadow:0 0 18px rgba(57,255,136,.7);pointer-events:none;animation:bob 1.6s ease-in-out infinite;transition:opacity .6s}
.hintdot span{position:absolute;left:32px;white-space:nowrap;font:500 10.5px var(--mono);color:var(--acc);text-shadow:0 0 8px #000}
@keyframes bob{50%{transform:translateY(-4px)}}
.hidehints .hintdot{opacity:0}
/* ---------------------------------------------------------------- overlays (expanded pupil, stress, downloads, about) */
.ov{position:fixed;inset:0;z-index:30;background:rgba(1,3,4,.76);backdrop-filter:blur(5px);display:none;align-items:center;justify-content:center}
.ov.show{display:flex;animation:fade .2s ease-out}
@keyframes fade{from{opacity:0}to{opacity:1}}
.panel{background:var(--tile2);border:1px solid var(--line2);border-radius:18px;box-shadow:0 30px 90px rgba(0,0,0,.7),0 0 40px rgba(57,255,136,.08);animation:pop .28s cubic-bezier(.2,.9,.25,1.2)}
@keyframes pop{from{transform:scale(.94);opacity:0}to{transform:scale(1);opacity:1}}
.stage{width:min(1480px,96vw);height:min(920px,93vh);display:grid;grid-template-columns:1fr 390px;grid-template-rows:auto 1fr;gap:16px;padding:16px}
.stagebar{grid-column:1 / -1;display:flex;align-items:center;gap:12px;font:11px var(--mono);color:var(--dim)}
.stagebar h2{margin:0;font:600 12px var(--mono);letter-spacing:.24em;color:var(--acc)}.stagebar .sp{flex:1}
.figbox{position:relative;display:flex;align-items:center;justify-content:center;min-height:0;background:#010302;border-radius:12px;border:1px solid var(--line);overflow:hidden}
.fig{position:relative}.fig img{display:block;width:100%;height:100%}.fig svg{position:absolute;inset:0;width:100%;height:100%;overflow:visible}
.side{display:flex;flex-direction:column;gap:12px;min-height:0}
.legend{display:flex;flex-wrap:wrap;gap:6px 12px;font:10.5px var(--mono);color:var(--dim)}
.legend span{display:flex;align-items:center;gap:6px}.legend i{width:16px;height:0;border-top:2px solid var(--c)}
.legend i.dash{border-top-style:dashed}.legend i.dots{border-top-style:dotted}
.ministats{font:11px/1.6 var(--mono);color:var(--dim)}.ministats b{color:var(--txt)}
.tabs{display:flex;gap:6px}
.btn{border:1px solid var(--line2);background:rgba(0,0,0,.3);color:var(--txt);border-radius:10px;padding:8px 12px;font:600 10.5px var(--mono);letter-spacing:.12em;
  text-transform:uppercase;cursor:pointer;text-decoration:none;text-align:center}
.btn:hover{border-color:var(--acc);box-shadow:0 0 14px rgba(57,255,136,.2)}
.btn.primary{background:linear-gradient(135deg,#39ff88,#16c47a);color:#02130a;border-color:transparent}
.btn:disabled{opacity:.45}
.dl{display:flex;gap:7px}.dl a{flex:1}
.close{background:none;border:1px solid var(--line);color:var(--dim);border-radius:9px;padding:6px 10px;font:11px var(--mono);cursor:pointer}
.close:hover{color:var(--txt);border-color:var(--acc)}
.small{width:min(900px,94vw);max-height:90vh;overflow:auto;padding:22px 24px}
.small h2{margin:0 0 12px;font:600 13px var(--mono);letter-spacing:.24em;color:var(--acc)}
.small p,.small li{font-size:13px;line-height:1.6;color:#bfe9d6}
.small table{width:100%;border-collapse:collapse;font:11.5px var(--mono);margin-top:6px}
.small td,.small th{border-bottom:1px solid var(--line);padding:6px 4px;text-align:left;vertical-align:top}.small th{color:var(--acc);font-weight:500}
.row{display:flex;align-items:center;gap:10px}
.prog{height:8px;border-radius:9px;background:rgba(255,255,255,.07);overflow:hidden;margin:12px 0}
.prog i{display:block;height:100%;width:0;background:linear-gradient(90deg,var(--acc-d),var(--acc));transition:width .3s}
.kpis{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:14px 0}
.kpi{border:1px solid var(--line);border-radius:12px;padding:12px;background:rgba(0,0,0,.25)}
.kpi .v{font:700 24px var(--mono)}.kpi .k{font:10.5px var(--mono);color:var(--dim);margin-top:4px}
kbd{font:600 10px var(--mono);border:1px solid var(--line2);border-radius:5px;padding:1px 5px;color:var(--acc)}
#tip{position:fixed;z-index:60;pointer-events:none;background:rgba(3,10,7,.95);border:1px solid var(--line2);border-radius:10px;padding:8px 10px;
  font:11px/1.5 var(--mono);color:var(--txt);max-width:300px;display:none;box-shadow:0 10px 30px rgba(0,0,0,.6)}
.htitle{position:fixed;z-index:40;pointer-events:none;font:500 15px var(--mono);letter-spacing:.24em;text-transform:uppercase;color:var(--acc);
  text-shadow:0 0 16px rgba(57,255,136,.7),0 0 2px #000;white-space:nowrap;opacity:0;transition:opacity .25s;padding:6px 12px;border-radius:10px;
  background:rgba(2,8,5,.55);border:1px solid rgba(57,255,136,.25);backdrop-filter:blur(4px)}
.htitle.show{opacity:1}
.top h1,.rtitle,#status{transition:opacity .2s}.top h1.away,.rtitle.away,#status.away{opacity:0}
.htitle::after{content:"_";margin-left:2px;animation:cur .8s steps(1) infinite}
@keyframes cur{50%{opacity:0}}
body.calm .htitle{transition:none}
#toast{position:fixed;left:50%;bottom:26px;transform:translateX(-50%);z-index:70;background:rgba(40,5,15,.95);border:1px solid #ff4f7a;color:#ffd9e2;
  padding:10px 16px;border-radius:10px;font:12px var(--mono);display:none}
.shock{position:fixed;z-index:50;width:20px;height:20px;margin:-10px 0 0 -10px;border-radius:50%;pointer-events:none;border:2px solid var(--acc);
  box-shadow:0 0 30px rgba(57,255,136,.8),inset 0 0 20px rgba(57,255,136,.5);animation:shock .7s ease-out forwards}
.shock.strong{border-width:3px;animation-duration:.9s}
@keyframes shock{from{transform:scale(.2);opacity:1}to{transform:scale(var(--s,22));opacity:0}}
.flash{position:fixed;inset:0;z-index:49;pointer-events:none;background:radial-gradient(circle at var(--x) var(--y),rgba(57,255,136,.18),transparent 45%);animation:fadeout .5s forwards}
@keyframes fadeout{to{opacity:0}}
/* wake-up intro: only the (closed) eye is visible until it has opened */
.top,.brow .seg,.brow .arc,.dock,.detrow,.review>*{transition:opacity .7s ease,translate .7s cubic-bezier(.2,.8,.2,1)}
body.intro .top,body.intro .brow .seg,body.intro .brow .arc,body.intro .dock,body.intro .detrow,body.intro .review>*,body.intro .hintdot{opacity:0;translate:0 16px;pointer-events:none}
.beam{transition:opacity 1.2s ease .5s} body.intro .beam{opacity:0}
body.intro .dock{translate:0 0}
body.intro #eyesvg,body.intro .pupil{animation:eyeglow 3.6s ease-in-out}   /* not on the lids' layer: a filter there hid the upper lid */
@keyframes eyeglow{0%{opacity:.35;filter:brightness(.6)}100%{opacity:1;filter:none}}
/* ================= light theme: only overrides; the dark styles above are unchanged ================= */
html.light{--bg:#d7ebdf;--tile:rgba(255,255,255,.93);--tile2:rgba(252,254,253,.98);--line:rgba(10,110,64,.22);--line2:rgba(10,130,74,.5);
  --acc:#0b9e56;--acc-d:#087a43;--cyan:#0094b5;--amber:#d48300;--mag:#c22ea1;--grey:#69768a;--gold:#c99a00;--txt:#10261c;--dim:#4b695b;--dim2:#8ba699}
html.light .zoomer,html.light .fig{--acc:#39ff88;--amber:#ffb020;--mag:#ff4fd8;--cyan:#22e4ff;--grey:#8b97a8;--gold:#ffd84a}   /* dark image areas keep the vivid colours */
html.light .dock,html.light .seg,html.light .tile{box-shadow:inset 0 1px 0 rgba(255,255,255,.95),0 2px 6px rgba(20,70,45,.08),0 12px 30px rgba(20,70,45,.14)}
html.light .seg:hover,html.light .tile:hover{box-shadow:inset 0 1px 0 rgba(255,255,255,.95),0 4px 10px rgba(20,70,45,.10),0 16px 38px rgba(20,70,45,.18),0 0 0 1px rgba(11,158,86,.22)}
html.light .tile::after,html.light .seg::after{content:none}
html.light .almond{filter:drop-shadow(0 6px 16px rgba(20,70,45,.22))}
html.light .irisline{stroke:rgba(225,255,238,.12)}
html.light .pupil{background:radial-gradient(circle at 50% 46%,#123323 0%,#081a12 62%,#030b07 100%);box-shadow:0 0 0 var(--pring,4px) rgba(150,240,190,.6),0 0 14px rgba(11,158,86,.3),inset 0 0 50px rgba(0,0,0,.55)}
html.light .pupil::before{content:"";position:absolute;inset:0;border-radius:50%;pointer-events:none;z-index:0;
  background:radial-gradient(ellipse 46% 26% at 34% 20%,rgba(255,255,255,.16),rgba(255,255,255,0) 70%)}
.lens{z-index:1}
html.light .pempty{color:#8fb3a1}
html.light .panel{box-shadow:0 24px 70px rgba(20,70,45,.18)}
html.light .chip,html.light .mbtn,html.light .btn,html.light .kpi{background:rgba(255,255,255,.75)}
html.light select,html.light input[type=text]{background:rgba(255,255,255,.9)}
html.light .chip.on{background:rgba(11,158,86,.12);box-shadow:none}
html.light .item{background:rgba(10,80,50,.045)} html.light .item.sel{background:rgba(11,158,86,.10)}
html.light .bar,html.light .prog{background:rgba(0,0,0,.08)}
html.light .logo .t1{text-shadow:0 0 14px rgba(11,158,86,.22)} html.light .logo .t2 i{text-shadow:none}
html.light .beam .bg1{stop-color:rgba(11,158,86,.16)} html.light .beam .bg2{stop-color:rgba(11,158,86,.05)} html.light .beam .rg1{stop-color:rgba(11,140,76,.45)}
html.light .browband{stroke:rgba(11,110,64,.22)} html.light .bf1{stop-color:rgba(11,120,70,.14)} html.light .bf2{stop-color:rgba(11,120,70,.05)}
html.light .hair{stroke:rgba(11,100,58,.30)} html.light .browedge{stroke:rgba(11,140,76,.55);filter:none}
html.light .lid .skin{fill:#e6eee9}
html.light .ov{background:rgba(232,240,235,.74)}
html.light .small p,html.light .small li{color:#2a4739}
html.light #tip{background:rgba(255,255,255,.97);color:var(--txt);box-shadow:0 10px 30px rgba(20,70,45,.16)}
html.light .htitle{background:rgba(255,255,255,.85);text-shadow:none}
html.light .hintdot span{text-shadow:0 0 6px #fff,0 0 2px #fff}
html.light .dbtn:hover{background:rgba(11,158,86,.07)}
body.calm *,body.calm *::before,body.calm *::after{animation:none!important;transition:none!important}
body.calm .det{opacity:1}
body.calm .scan,body.calm .cover,body.calm .shock,body.calm .flash{display:none}
@media (max-width:1100px){.review{grid-template-columns:repeat(2,1fr)} .main{padding-left:24px;padding-right:24px} .dock{position:static;transform:none;flex-direction:row;flex-wrap:wrap;justify-content:center;margin:10px auto}
  .dbtn{width:58px}}
/* ================= the journey: how the EYE was built (scoped to #jy) ================= */
#jy{    
  --bp:#173a63; --bp2:#1e4a7d; --bpline:rgba(255,255,255,.12); --chalk:#f4f7fb;
  --hivis:#f4c430; --hivis-ink:#3d2e00; --kraft:#efe2c4; --kraft2:#e4d3ac; --kraft-ink:#3b2f1c; --kraft-dim:#6f5d3d;
   --roadline:#f4c430; 
  --sans:"Barlow",system-ui,"Segoe UI",Roboto,sans-serif; --cond:"Barlow Condensed","Arial Narrow",system-ui,sans-serif; --stencil:"Saira Stencil One",Impact,"Arial Black",sans-serif;--page:#24272c;--page2:#1c1f23;--ink:#eef1f4;--ink2:#aab3bd;--line:rgba(238,241,244,.16);--road:#14171a;--shadow:0 18px 44px rgba(0,0,0,.45);color:var(--ink);font-family:var(--sans);font-size:16px;line-height:1.5;position:relative;z-index:2}
html.light #jy{--page:#e7e3da;--page2:#dcd7cb;--ink:#1d232b;--ink2:#4b5560;--line:rgba(29,35,43,.18);--road:#3a3f46;--shadow:0 18px 40px rgba(29,35,43,.18);}
#jy button{font:inherit;color:inherit}
#jy :focus-visible{outline:3px solid var(--hivis);outline-offset:3px}
#jy .wrap{max-width:1320px;margin:0 auto;padding:10px 0 36px}
#jy header{display:flex;align-items:flex-end;gap:24px;flex-wrap:wrap;margin-bottom:22px}
#jy header .titles{flex:1;min-width:280px}
#jy h1{font-family:var(--cond);font-weight:700;font-size:clamp(34px,5vw,58px);line-height:.95;margin:0;letter-spacing:-.01em}
#jy .sub{margin:10px 0 0;color:var(--ink2);max-width:62ch}
#jy .hdr-tools{display:flex;gap:8px}
#jy .tbtn{border:1.5px solid var(--line);background:transparent;border-radius:999px;padding:7px 14px;cursor:pointer;font-family:var(--cond);font-weight:600;letter-spacing:.02em}
#jy .tbtn:hover{border-color:var(--ink2)}
#jy /* station map */
.map{display:flex;align-items:center;gap:0;margin:6px 0 18px;overflow-x:auto;padding:6px 2px 10px;scrollbar-width:none}
#jy .map::-webkit-scrollbar{display:none}
#jy .mstop{display:flex;align-items:center;flex:1 0 auto}
#jy .mstop:last-child .mline{display:none}
#jy .mdot{width:38px;height:38px;border-radius:50%;border:2px solid var(--ink2);background:var(--page);display:grid;place-items:center;cursor:pointer;
  font-family:var(--stencil);font-size:15px;color:var(--ink2);transition:background .25s,color .25s,border-color .25s,transform .25s;flex:0 0 auto}
#jy .mdot:hover{border-color:var(--ink)}
#jy .mdot.on{background:var(--hivis);border-color:var(--hivis);color:var(--hivis-ink);transform:scale(1.12)}
#jy .mdot.done{border-color:var(--ink);color:var(--ink)}
#jy .mline{height:2px;flex:1;min-width:14px;background:repeating-linear-gradient(90deg,var(--ink2) 0 8px,transparent 8px 14px);margin:0 4px;opacity:.6}
#jy .mname{font-family:var(--cond);font-weight:600;font-size:13px;color:var(--ink2);margin-left:8px;white-space:nowrap;display:none}
#jy .mstop.on .mname{display:inline;color:var(--ink)}
#jy /* track */
.trackwrap{position:relative}
#jy .track{position:relative;display:flex;gap:0;overflow-x:auto;scroll-snap-type:x mandatory;scroll-behavior:smooth;padding:8px 0 0;scrollbar-width:thin}
#jy .stop{flex:0 0 auto;width:calc(min(380px,82vw) + 36px);padding-right:36px;scroll-snap-align:start;position:relative}
#jy .stop:last-child{width:min(380px,82vw);padding-right:0}
#jy .card{width:min(380px,82vw);height:550px;perspective:1600px;cursor:pointer;background:none;border:0;padding:0;text-align:left;display:block}
#jy .inner{position:relative;width:100%;height:100%;transform-style:preserve-3d;transition:transform .7s cubic-bezier(.3,.7,.2,1)}
#jy .card.flipped .inner{transform:rotateY(180deg)}
#jy .face{position:absolute;inset:0;border-radius:14px;backface-visibility:hidden;-webkit-backface-visibility:hidden;overflow:hidden;box-shadow:var(--shadow)}
#jy /* front: blueprint */
.front{background-color:var(--bp);color:var(--chalk);display:flex;flex-direction:column;
  background-image:linear-gradient(var(--bpline) 1px,transparent 1px),linear-gradient(90deg,var(--bpline) 1px,transparent 1px);background-size:22px 22px}
#jy .front .tape{height:12px;background:repeating-linear-gradient(-45deg,var(--hivis) 0 12px,#1b1b1b 12px 24px)}
#jy .front .num{font-family:var(--stencil);font-size:64px;line-height:1;color:var(--hivis);margin:18px 22px 0}
#jy .front .scene{display:block;width:calc(100% - 28px);margin:6px 14px 0;height:auto}
#jy .front h2{font-family:var(--cond);font-weight:700;font-size:30px;line-height:1.05;margin:4px 22px 0;letter-spacing:.005em}
#jy .front .line{margin:8px 22px 0;font-size:17px;line-height:1.4;color:#d6e2f1;max-width:32ch}
#jy .front .hint{margin:auto 22px 18px;font-family:var(--cond);font-weight:600;font-size:14px;color:var(--hivis);letter-spacing:.03em;display:flex;align-items:center;gap:8px}
#jy .front .hint svg{width:16px;height:16px}
#jy .sc .w{stroke:var(--chalk);stroke-width:2;fill:none;stroke-linecap:round;stroke-linejoin:round}
#jy .sc .wf{fill:var(--chalk)}
#jy .sc .y{fill:var(--hivis)}
#jy .sc .ys{stroke:var(--hivis);stroke-width:2;fill:none;stroke-linecap:round}
#jy .sc .d{stroke:var(--chalk);stroke-width:1.5;fill:none;stroke-dasharray:4 4}
#jy .sc .t{fill:var(--chalk);font-family:var(--cond);font-weight:600;font-size:13px}
#jy .sc .ty{fill:var(--hivis-ink);font-family:var(--cond);font-weight:700;font-size:13px}
#jy .sc .faint{stroke:rgba(244,247,251,.35);stroke-width:1.5;fill:none}
#jy /* back: site report on kraft paper */
.back{transform:rotateY(180deg);background:var(--kraft);color:var(--kraft-ink);padding:20px 22px 18px;display:flex;flex-direction:column;
  background-image:repeating-linear-gradient(0deg,transparent 0 27px,rgba(59,47,28,.09) 27px 28px)}
#jy .back .bhead{display:flex;align-items:baseline;gap:12px;border-bottom:2px solid var(--kraft-ink);padding:18px 0 8px;margin-bottom:10px}
#jy .back .bnum{font-family:var(--stencil);font-size:30px;color:var(--kraft-ink)}
#jy .back h3{font-family:var(--cond);font-weight:700;font-size:23px;margin:0;line-height:1.1}
#jy .back ul{margin:0;padding:0 0 0 18px;font-size:14.5px;line-height:1.45}
#jy .back li{margin:0 0 6px}
#jy .back li::marker{color:var(--kraft-dim)}
#jy .back .figs{display:flex;gap:8px;margin-top:auto;padding-top:10px}
#jy .back .fig{flex:1;background:var(--kraft2);border-radius:8px;padding:8px 10px}
#jy .back .fig b{display:block;font-family:var(--cond);font-weight:700;font-size:22px;line-height:1.1}
#jy .back .fig span{font-size:12.5px;color:var(--kraft-dim);line-height:1.3;display:block}
#jy .back .why{margin:10px 0 0;font-size:13.5px;color:var(--kraft-dim);font-style:italic}
#jy .back .stamp{position:absolute;right:16px;top:14px;font-family:var(--cond);font-weight:700;font-size:12px;color:#8a2a1a;border:2px solid #8a2a1a;border-radius:6px;padding:2px 7px;transform:rotate(6deg);opacity:.75}
#jy /* road + truck */
.road{position:relative;height:58px;margin-top:22px;margin-right:-36px}
#jy .stop:last-child .road{margin-right:0}
#jy .road::before{content:"";position:absolute;left:0;right:0;top:16px;height:26px;background:var(--road);border-radius:4px}
#jy .road::after{content:"";position:absolute;left:0;right:0;top:28px;height:2px;background:repeating-linear-gradient(90deg,var(--roadline) 0 18px,transparent 18px 34px)}
#jy .post{position:absolute;left:calc(min(380px,82vw) / 2 - 14px);top:-6px;width:28px;height:28px;border-radius:50%;background:var(--hivis);color:var(--hivis-ink);
  font-family:var(--stencil);font-size:13px;display:grid;place-items:center;z-index:2;box-shadow:0 0 0 3px var(--page)}
#jy .truck{position:absolute;top:10px;left:0;width:64px;height:38px;z-index:3;filter:drop-shadow(0 0 8px var(--glow));transition:transform .9s cubic-bezier(.4,.1,.2,1);pointer-events:none}
#jy .nav{display:flex;align-items:center;gap:12px;margin-top:14px}
#jy .nbtn{width:46px;height:46px;border-radius:50%;border:2px solid var(--ink);background:transparent;cursor:pointer;display:grid;place-items:center}
#jy .nbtn:hover{background:var(--hivis);border-color:var(--hivis);color:var(--hivis-ink)}
#jy .nbtn svg{width:20px;height:20px}
#jy .count{font-family:var(--cond);font-weight:600;font-size:18px;min-width:80px}
#jy .keys{margin-left:auto;color:var(--ink2);font-size:14px}
#jy .keys kbd{font-family:var(--cond);font-weight:600;border:1.5px solid var(--line);border-radius:5px;padding:0 6px;margin:0 2px}
#jy footer{margin-top:26px;color:var(--ink2);font-size:14px;border-top:1.5px solid var(--line);padding-top:12px}
@media (prefers-reduced-motion: reduce){#jy .inner,#jy .truck,#jy .mdot{transition:none}
#jy .track{scroll-behavior:auto}}
@media (max-width:640px){#jy .card{height:640px}
#jy .keys{display:none}}
#jy{width:100%;max-width:1320px;margin-top:84px;scroll-margin-top:24px}
#jy header{flex-wrap:nowrap;align-items:center}
#jy .jyeye{flex:0 0 auto;width:clamp(120px,13vw,190px);opacity:0;transform:translateY(-70vh) scale(1.8)}
#jy .jyeye svg{display:block;width:100%;height:auto;overflow:visible}
#jy .jyeye.landed{animation:jyland 1.15s cubic-bezier(.3,1.3,.45,1) forwards}
@keyframes jyland{0%{opacity:0;transform:translateY(-70vh) scale(1.8)}55%{opacity:1}100%{opacity:1;transform:none}}
#jy .msc{fill:#0d2219} #jy .mout{fill:none;stroke:var(--acc);stroke-width:2.2;filter:drop-shadow(0 0 6px rgba(57,255,136,.55))}
#jy .ji1{stop-color:#062a18} #jy .ji2{stop-color:#0b4a2b} #jy .ji3{stop-color:#021a0e}
#jy .mring{fill:none;stroke:rgba(57,255,136,.55);stroke-width:5} #jy .mpup{fill:#030b07} #jy .mglint{fill:rgba(255,255,255,.75)}
#jy .mlid rect{fill:#071210} #jy .mlid.u{transform:translateY(-58px)} #jy .mlid.l{transform:translateY(58px)}
#jy .jyeye.blink .mlid.u{animation:jyblu .42s ease-in-out 2} #jy .jyeye.blink .mlid.l{animation:jybll .42s ease-in-out 2}
@keyframes jyblu{0%,100%{transform:translateY(-58px)}50%{transform:translateY(0)}} @keyframes jybll{0%,100%{transform:translateY(58px)}50%{transform:translateY(0)}}
#jy #jy-iris-g{transition:transform .25s ease-out}
html.light #jy .msc{fill:#f6faf7} html.light #jy .mout{stroke:#0b9e56;filter:drop-shadow(0 4px 10px rgba(20,70,45,.2))}
html.light #jy .ji1{stop-color:#0f5536} html.light #jy .ji2{stop-color:#1d7f4e} html.light #jy .ji3{stop-color:#0b452b}
html.light #jy .mring{stroke:rgba(150,240,190,.55)} html.light #jy .mlid rect{fill:#e6eee9}
#jy .more{flex:1 1 auto;min-height:0;overflow:auto;margin:4px 0 2px;padding-right:4px;font-size:13px;line-height:1.4;border-top:1.5px dashed rgba(59,47,28,.35);padding-top:6px}
#jy .more p{margin:6px 0 0}
#jy .mt{width:100%;border-collapse:collapse;font-size:12.5px}
#jy .mt th{text-align:left;font-family:var(--cond);font-weight:700;font-size:13px;border-bottom:1.5px solid var(--kraft-ink);padding:2px 4px}
#jy .mt td{padding:3px 4px;border-bottom:1px solid rgba(59,47,28,.18);vertical-align:top}
#jy .mt tr.win td{font-weight:600;background:rgba(244,196,48,.28)}
#jy .back a{color:var(--kraft-ink);text-decoration:underline;text-decoration-thickness:1.5px;text-underline-offset:2px}
#jy .back a:hover{color:#8a2a1a}
#jy .card{cursor:pointer}
@media (prefers-reduced-motion: reduce){#jy .jyeye{opacity:1;transform:none} #jy .jyeye.landed{animation:none} #jy .jyeye.blink .mlid{animation:none}}
body.calm #jy .jyeye{opacity:1;transform:none;animation:none}
#jy h1{font-size:clamp(26px,3.2vw,42px);line-height:1.12;max-width:34ch}
.howbtn{justify-self:start;border:1px solid var(--line2);background:var(--tile);color:var(--acc);border-radius:999px;padding:8px 14px;font:600 11px var(--mono);letter-spacing:.18em;text-transform:uppercase;cursor:pointer;backdrop-filter:blur(6px)}
.howbtn:hover{border-color:var(--acc);box-shadow:0 0 16px rgba(57,255,136,.25)}

/* ================= the journey, re-skinned in The EYE's design language (overrides) ================= */
#jy{--sans:Inter,system-ui,"Segoe UI",Roboto,sans-serif;--cond:Inter,system-ui,"Segoe UI",Roboto,sans-serif;--stencil:"JetBrains Mono",ui-monospace,Consolas,monospace;
  --ink:var(--txt);--ink2:var(--dim);--line:rgba(57,255,136,.18);--page:#04070a;
  --bp:#07130e;--bpline:rgba(57,255,136,.075);--chalk:#bff7d6;--hivis:#39ff88;--hivis-ink:#02130a;
  --kraft:#08140f;--kraft2:rgba(57,255,136,.07);--kraft-ink:#dcfff0;--kraft-dim:#7fa594;
  --road:#07110c;--roadline:#39ff88;--truckline:#02130a;--glow:rgba(57,255,136,.55);--shadow:0 18px 44px rgba(0,0,0,.5)}
html.light #jy{--ink:#10261c;--ink2:#4b695b;--line:rgba(10,110,64,.22);--page:#d7ebdf;
  --bp:#ffffff;--bpline:rgba(11,110,64,.07);--chalk:#0b6a3e;--hivis:#0b9e56;--hivis-ink:#ffffff;
  --kraft:#ffffff;--kraft2:rgba(11,158,86,.08);--kraft-ink:#10261c;--kraft-dim:#4b695b;
  --road:#cfe3d7;--roadline:#0b9e56;--truckline:#ffffff;--glow:rgba(11,158,86,.0);--shadow:inset 0 1px 0 rgba(255,255,255,.95),0 2px 6px rgba(20,70,45,.08),0 12px 30px rgba(20,70,45,.14)}
#jy h1{font:600 clamp(22px,2.6vw,34px)/1.18 var(--sans);letter-spacing:-.01em;color:var(--ink)}
#jy .sub{color:var(--ink2);font-size:14px}
#jy .tbtn{border:1px solid var(--line2);background:var(--tile);color:var(--acc);font:600 10.5px var(--mono);letter-spacing:.16em;text-transform:uppercase;padding:8px 14px;backdrop-filter:blur(6px)}
#jy .tbtn:hover{border-color:var(--acc);box-shadow:0 0 16px var(--glow)}
#jy .mdot{border:1px solid var(--line2);background:var(--tile2);color:var(--ink2);font:600 12px var(--mono)}
#jy .mdot:hover{border-color:var(--acc);color:var(--acc)}
#jy .mdot.on{background:var(--acc);border-color:var(--acc);color:var(--hivis-ink);box-shadow:0 0 16px var(--glow)}
#jy .mdot.done{border-color:var(--acc);color:var(--acc)}
#jy .mline{background:repeating-linear-gradient(90deg,var(--line2) 0 6px,transparent 6px 12px);opacity:1}
#jy .mname{font:600 10px var(--mono);letter-spacing:.14em;text-transform:uppercase;color:var(--acc)}
#jy .face{border:1px solid var(--line)}
#jy .front{color:var(--chalk)}
#jy .front .tape{height:3px;background:linear-gradient(90deg,transparent,var(--hivis) 20%,var(--hivis) 80%,transparent);box-shadow:0 0 14px var(--glow)}
#jy .front .num{font:700 50px/1 var(--stencil);color:var(--hivis);text-shadow:0 0 18px var(--glow);margin:20px 22px 0}
#jy .front h2{font:600 24px/1.15 var(--sans);letter-spacing:-.01em;color:var(--kraft-ink)}
#jy .front .line{color:var(--kraft-dim);font-size:15px}
#jy .front .hint{font:600 10.5px var(--mono);letter-spacing:.16em;text-transform:uppercase;color:var(--hivis)}
#jy .sc .t{font-family:var(--sans);font-weight:600;font-size:12px}
#jy .sc .ty{font-family:var(--sans);font-weight:700;font-size:12px}
#jy .sc .faint{stroke:var(--bpline);stroke-width:2}
#jy .back{background:var(--kraft);background-image:repeating-linear-gradient(0deg,transparent 0 27px,var(--bpline) 27px 28px);color:var(--kraft-ink)}
#jy .back .bhead{border-bottom:1px solid var(--line2)}
#jy .back .bnum{font:700 26px var(--stencil);color:var(--hivis);text-shadow:0 0 12px var(--glow)}
#jy .back h3{font:600 19px/1.2 var(--sans)}
#jy .back ul{font-size:13.5px}
#jy .back li::marker{color:var(--hivis)}
#jy .back .stamp{color:var(--hivis);border:1px solid var(--hivis);font:600 9.5px var(--mono);letter-spacing:.12em;text-transform:uppercase;opacity:.9}
#jy .back .fig{background:var(--kraft2);border:1px solid var(--line)}
#jy .back .fig b{font:700 18px var(--mono);color:var(--hivis)}
#jy .back .fig span{color:var(--kraft-dim)}
#jy .back .why{color:var(--kraft-dim)}
#jy .back a{color:var(--hivis)} #jy .back a:hover{color:var(--ink)}
#jy .more{border-top:1px dashed var(--line2)}
#jy .mt th{border-bottom:1px solid var(--line2);font:600 10.5px var(--mono);letter-spacing:.08em;text-transform:uppercase;color:var(--hivis)}
#jy .mt td{border-bottom:1px solid var(--line)}
#jy .mt tr.win td{background:var(--kraft2);color:var(--hivis)}
#jy .road::before{background:var(--road);border:1px solid var(--line)}
#jy .road::after{background:repeating-linear-gradient(90deg,var(--roadline) 0 18px,transparent 18px 34px);filter:drop-shadow(0 0 4px var(--glow))}
#jy .post{background:var(--hivis);color:var(--hivis-ink);font:700 12px var(--mono);box-shadow:0 0 0 3px var(--page),0 0 14px var(--glow)}
#jy .nbtn{border:1px solid var(--line2);color:var(--acc);background:var(--tile)}
#jy .nbtn:hover{background:var(--acc);border-color:var(--acc);color:var(--hivis-ink);box-shadow:0 0 16px var(--glow)}
#jy .count{font:600 13px var(--mono);letter-spacing:.08em;color:var(--ink)}
#jy .keys{font-family:var(--mono);font-size:11px;letter-spacing:.04em}
#jy footer{border-top:1px solid var(--line);font-size:13px}
#jy .card:focus-visible{outline:2px solid var(--acc);outline-offset:4px;border-radius:14px}

body.intro #jy,body.intro .howbtn{opacity:0;pointer-events:none} #jy,.howbtn{transition:opacity .7s ease}
</style>
</head>
<body class="intro">
<canvas id="bg"></canvas>

<nav class="dock" id="dock">
  <button class="dbtn" id="d-up" title="Upload an image [U]"><svg viewBox="0 0 24 24"><path d="M12 16V4M7 9l5-5 5 5M4 20h16"/></svg>upload</button>
  <button class="dbtn" id="d-rand" title="New random test grid [N]"><svg viewBox="0 0 24 24"><rect x="3" y="3" width="18" height="18" rx="3"/><circle cx="8" cy="8" r="1.4"/><circle cx="16" cy="16" r="1.4"/><circle cx="12" cy="12" r="1.4"/></svg>random</button>
  <button class="dbtn" id="d-stress" title="Stress test"><svg viewBox="0 0 24 24"><path d="M3 12h4l3-8 4 16 3-8h4"/></svg>stress</button>
  <button class="dbtn" id="d-about" title="About"><svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7.5v.5"/></svg>about</button>
  <input type="file" id="file" accept="image/*" hidden>
</nav>

<nav class="dock right" id="dock2">
  <button class="dbtn" id="d-dl" title="Downloads"><svg viewBox="0 0 24 24"><path d="M12 4v12M7 11l5 5 5-5M4 20h16"/></svg>files</button>
  <button class="dbtn" id="d-snap" title="Snapshot for reports"><svg viewBox="0 0 24 24"><rect x="3" y="6" width="18" height="14" rx="3"/><circle cx="12" cy="13" r="3.5"/><path d="M8 6l2-3h4l2 3"/></svg>snapshot</button>
  <button class="dbtn on" id="d-titles" title="Hover titles on / off"><svg viewBox="0 0 24 24"><path d="M5 6h14M12 6v13M8 19h8"/></svg>titles</button>
  <button class="dbtn" id="d-theme" title="Light / dark theme"><svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg><span id="themelbl">light</span></button>
  <button class="dbtn" id="d-calm" title="Calm mode: no motion"><svg viewBox="0 0 24 24"><path d="M20 14.5A8 8 0 0 1 9.5 4 8 8 0 1 0 20 14.5z"/></svg>calm</button>
</nav>

<main class="main" id="main">
  <svg class="beam" id="beam" aria-hidden="true"></svg>
  <div class="top"><button class="howbtn" id="howbtn" type="button">How was I built?</button><h1 class="logo"><span class="t1">The <b>EYE</b></span><span class="t2"><i>E</i>xplainable <i>Y</i>OLO <i>E</i>xplorer</span></h1><span class="meta" id="meta"><span id="m-obj" data-tip="">-</span> · <span id="m-photos" data-tip="">-</span> · <span id="m-boxed" data-tip="">-</span></span></div>

  <section class="brow" id="brow">
    <svg class="arc" id="arcsvg"><defs><linearGradient id="browgrad" x1="0" x2="1"><stop offset="0" stop-color="#39ff88" stop-opacity="0"/>
      <stop offset=".5" stop-color="#39ff88"/><stop offset="1" stop-color="#39ff88" stop-opacity="0"/></linearGradient></defs><path class="arcl" id="arcpath"/></svg>
    <div class="seg" data-i="0" id="s-grid"><h3>Grid <span>build</span></h3><div class="chips" id="sizes"></div><div class="lbl" id="sizeinfo"></div></div>
    <div class="seg" data-i="1" id="s-mix"><h3>Mix</h3><div class="chips" id="secs"></div><div class="chips" id="mixes"></div><div class="chips" id="layouts" title="layout of the test image"></div></div>
    <div class="seg" data-i="2" id="s-cond"><h3>Conditions</h3><div class="chips" id="degs"></div></div>
    <div class="seg" data-i="3" id="s-seed"><h3>Seed</h3><input type="text" id="seed" placeholder="empty = random">
      <div class="mini"><button class="mbtn" id="dice">random</button><button class="mbtn" id="keep">keep last</button></div><div class="lbl" id="lastseed"></div></div>
    <div class="seg" data-i="4" id="s-model"><h3>Model</h3><select id="model"></select><div class="lbl" id="mapv"></div></div>
    <div class="seg" data-i="5" id="s-thr"><h3>Threshold <span id="thrv">0.10</span></h3><input type="range" id="conf" min="0.05" max="0.9" step="0.05">
      <label class="toggle"><input type="checkbox" id="cell" checked>one box per cell</label><button class="mbtn" id="resetthr">reset to validated</button></div>
  </section>

  <section class="eyewrap" id="eye">
    <svg class="eyesvg" id="eyesvg"></svg>
    <div class="pupil" id="pupil"><div class="logo intrologo" id="intrologo"><span class="t1"><span class="lt" style="--d:0s">T</span><span class="lt" style="--d:.09s">h</span><span class="lt" style="--d:.18s">e</span> <b><span class="lt ee" style="--d:.36s">E</span><span class="lt" style="--d:.45s">Y</span><span class="lt ee" style="--d:.54s">E</span></b></span><span class="t2"><i>E</i>xplainable <i>Y</i>OLO <i>E</i>xplorer</span></div><div class="lens" id="lens"><div class="pempty" id="pempty"><div>press <b>random</b> or <kbd>N</kbd><br>to build a test grid</div></div></div></div>
    <svg class="lidsvg" id="lidsvg"></svg>
  </section>

  <div class="detrow"><div class="status" id="status">warming up the model...</div></div>

  <section class="review" id="review">
    <div class="tile" id="t-score"><div class="th"><span class="ix">01</span><h3>Score</h3><span class="sp"></span><span class="hint" id="ms"></span></div>
      <div class="tb"><div class="score"><span class="big" id="sc">-</span><span class="of" id="scof"></span></div><div class="bar"><i id="scbar"></i></div>
        <div class="verdict" id="verdict"></div><div class="stats" id="stats"></div></div></div>
    <div class="tile" id="t-err"><div class="th"><span class="ix">02</span><h3>Results</h3><span class="sp"></span>
        <span class="ftabs" id="errtabs"><span class="chip sm on" data-f="err">errors</span><span class="chip sm" data-f="all">all</span></span></div>
      <div class="tb"><div class="list" id="errs"><div class="lbl">after detection: one sentence per object; press + for every detail, click to zoom the pupil onto it</div></div></div></div>
    <div class="tile" id="t-insp"><div class="th"><span class="ix">03</span><h3>Inspect</h3><span class="sp"></span><button class="enl" id="insp-big" title="enlarge">⤢</button></div>
      <div class="tb"><div class="insp" id="insp"><div class="lbl">after detection: click any object (or use ‹ ›) to see it next to training photos of the model's class and the true class</div></div></div></div>
    <div class="tile" id="t-sweep"><div class="th"><span class="ix">04</span><h3>Threshold sweep</h3><span class="sp"></span><button class="enl" id="sweep-big" title="enlarge">⤢</button></div>
      <div class="tb"><svg class="sweep" id="sweep" viewBox="0 0 360 170" preserveAspectRatio="none"></svg></div></div>
  </section>
  <section id="jy" aria-label="How the EYE was built">
<div class="wrap">
  <header>
    <div class="jyeye" id="jy-eye" aria-hidden="true"><svg viewBox="0 0 200 110">
      <defs><clipPath id="jy-alm"><path d="M2 55 C46 -10,154 -10,198 55 C154 120,46 120,2 55 Z"/></clipPath>
        <radialGradient id="jy-iris" cx="50%" cy="50%" r="50%"><stop offset=".72" class="ji1"/><stop offset=".9" class="ji2"/><stop offset="1" class="ji3"/></radialGradient></defs>
      <path class="msc" d="M2 55 C46 -10,154 -10,198 55 C154 120,46 120,2 55 Z"/>
      <g clip-path="url(#jy-alm)"><g id="jy-iris-g"><circle cx="100" cy="55" r="42" fill="url(#jy-iris)"/>
        <circle class="mring" cx="100" cy="55" r="42"/><circle class="mpup" cx="100" cy="55" r="30"/><circle class="mglint" cx="113" cy="41" r="4"/></g>
        <g class="mlid u"><rect x="0" y="-60" width="200" height="115"/></g><g class="mlid l"><rect x="0" y="55" width="200" height="115"/></g></g>
      <path class="mout" d="M2 55 C46 -10,154 -10,198 55 C154 120,46 120,2 55 Z"/></svg></div>
    <div class="titles">
      <h1>&ldquo;For simplicity, and since you don&rsquo;t know much biology and human anatomy, I will tell you how I was built as if I were a building.&rdquo;</h1>
      <p class="sub">Milestone 2, Group 7: from 7,343 raw files to a detector for 73 everyday objects. Ten stations, left to right. Flip any card for the full site report.</p>
    </div>
    <div class="hdr-tools">
      <button class="tbtn" id="jy-flipall" type="button">Flip all</button>
      <button class="tbtn" id="jy-up" type="button">Back to the eye</button>
    </div>
  </header>

  <nav class="map" id="jy-map" aria-label="Stations"></nav>

  <div class="trackwrap">
    <div class="track" id="jy-track" tabindex="0" aria-label="Journey stations"></div>
  </div>

  <div class="nav">
    <button class="nbtn" id="jy-prev" type="button" aria-label="Previous station"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M15 5l-7 7 7 7"/></svg></button>
    <button class="nbtn" id="jy-next" type="button" aria-label="Next station"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M9 5l7 7-7 7"/></svg></button>
    <span class="count" id="jy-count">1 / 10</span>
    <span class="keys"><kbd>←</kbd><kbd>→</kbd> move · <kbd>Enter</kbd> flip</span>
  </div>

  <footer>IE7615 Deep Learning, Northeastern University. Detector: YOLO26-s, fine-tuned twice, chosen on validation and tested once on boxes checked by a person.</footer>
</div>
  </section>
</main>

<div class="ov" id="ov-stage"><div class="panel stage">
  <div class="stagebar"><h2 id="stitle">PUPIL</h2>
    <label class="sctl">grid <select id="sc-size"></select></label>
    <label class="sctl">seed <input type="text" id="sc-seed" placeholder="random" size="7"></label><button class="close" id="sc-dice" title="empty seed = a new random grid">random seed</button>
    <label class="sctl">model <select id="sc-model"></select></label>
    <label class="sctl">threshold <select id="sc-thr"></select></label>
    <span class="sp"></span><span id="ssub"></span>
    <button class="btn" id="s-new">Random</button><button class="btn primary" id="s-det">Detect</button><button class="close" id="s-close">close [esc]</button></div>
  <div class="figbox" id="figbox"><div class="fig" id="fig"></div></div>
  <div class="side">
    <div class="legend" id="legend"><span class="b-correct" data-tip="Correct: right Object ID, and the box overlaps the true box by at least half (IoU ≥ 0.5)"><i></i>correct</span>
      <span class="b-box-off" data-tip="Box off: right Object ID, but the box overlaps the true box by less than half (IoU < 0.5)"><i class="dash"></i>box off</span>
      <span class="b-wrong-ID" data-tip="Wrong ID: the box is in the right place, but names a different object (look-alikes are flagged)"><i></i>wrong ID</span>
      <span class="b-extra" data-tip="Extra: a box that matches no true object (a second box, a partial box, background, or an unlabelled object in the photo)"><i class="dots"></i>extra</span>
      <span class="b-missed" data-tip="Missed: no box for this true object; the reason says whether it was seen below the threshold or is very small"><i class="dash"></i>missed</span>
      <span style="--c:rgba(255,255,255,.6)" data-tip="True box: the object's correct box, checked by a person"><i class="dash"></i>true box</span></div>
    <div class="ministats" id="sstats"></div>
    <div class="tabs"><span class="chip on" data-f="all">all</span><span class="chip" data-f="err">errors</span></div>
    <div class="list" id="srows"></div><div class="dl" id="sdl"></div></div></div></div>

<div class="ov" id="ov-big"><div class="panel small bigview" id="bigpanel"></div></div>

<div class="ov" id="ov-stress"><div class="panel small"><h2>STRESS TEST</h2>
  <p>Runs many random grids of held-out test photos with the settings on the eyebrow (grid, mix, conditions, model, threshold) and reports the totals. Grids use seeds 1000, 1001, ... so a run can be repeated exactly.</p>
  <div class="row"><span class="lbl">GRIDS</span><input type="range" id="sn" min="10" max="300" step="10" value="100"><span class="lbl" id="snv">100</span></div>
  <div class="row" style="margin-top:10px"><button class="btn primary" id="srun">Run</button><button class="close" id="sclose">close</button></div>
  <div class="prog"><i id="sprog"></i></div><div id="sres"></div></div></div>

<div class="ov" id="ov-dl"><div class="panel small"><h2>FILES</h2><p>The last detection, in the milestone's output format, plus a snapshot image for reports and slides.</p>
  <div class="dl" id="dls"></div><div class="row" style="margin-top:14px"><button class="close" id="dlclose">close</button></div></div></div>

<div class="ov" id="ov-about"><div class="panel small"><h2>ABOUT</h2>
  <p>The detector reports the <b>Object ID and bounding box</b> of every object. Random grids and the stress test use only <b>held-out test photos</b> whose true boxes were checked by a person.</p>
  <p><b>How to use:</b> ① set up the test on the eyebrow · ② press <b>RANDOM</b> (or upload) and <b>DETECT</b> · ③ review below. Click the pupil for the large view; click an error to zoom onto it.</p>
  <table><tr><th>result</th><th>meaning</th></tr>
  <tr><td style="color:var(--acc)">correct</td><td>right Object ID and the box overlaps the true box by at least half (IoU >= 0.5)</td></tr>
  <tr><td style="color:var(--amber)">box off</td><td>right Object ID, but the box overlaps by less than half</td></tr>
  <tr><td style="color:var(--mag)">wrong ID</td><td>box in the right place, another object named (look-alikes flagged)</td></tr>
  <tr><td style="color:var(--cyan)">missed</td><td>no box; the reason says whether it was seen below the threshold or is very small</td></tr>
  <tr><td style="color:var(--grey)">extra</td><td>a box matching no object</td></tr></table>
  <h2 style="margin-top:18px">BEFORE VS AFTER</h2><div id="aboutboost"></div>
  <h2 style="margin-top:18px">MODELS</h2><div id="aboutmodels"></div>
  <h2 style="margin-top:18px">BY INPUT TYPE</h2><div id="aboutshapes"></div>
  <p>The <b>iris</b> is the score: one segment per object, coloured by its result. Keys: <kbd>N</kbd> random grid · <kbd>D</kbd> detect · <kbd>U</kbd> upload · <kbd>Esc</kbd> close or zoom out.</p>
  <div class="row"><button class="close" id="aclose">close</button></div></div></div>

<div id="tip"></div><div id="toast"></div><div class="htitle" id="htitle"></div>

<script>
"use strict";
const $ = (s, r = document) => r.querySelector(s), $$ = (s, r = document) => [...r.querySelectorAll(s)];
const st = { cfg: null, model: null, conf: .1, cell: true, size: "3x3 (9 objects)", sections: [], mix: "Random", layout: "grid", degrade: "None",
             grid: null, det: null, sel: null, filter: "all", lastSeed: null, zoomed: null, busy: false, warm: "loading", hinted: false };
const calm = () => document.body.classList.contains("calm");
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const cls = r => String(r || "detected").replace(/ /g, "-");
const isLight = () => document.documentElement.classList.contains("light");
const PAL = {
  dark:  { bg: "#04070a", px: ["57,255,136", "34,228,255", "220,255,240"], pxA: 1, pxS: 0, speed: 1, scl: ["#0d2219", "#06120d", "#030806"],
           iris: ["#062a18", "#0b4a2b", "#0b4a2b", "#021a0e"], lidShade: false,
           res: { "correct": "#39ff88", "detected": "#39ff88", "box off": "#ffb020", "wrong ID": "#ff4fd8", "missed": "#22e4ff", "extra": "#8b97a8" },
           sw: { bg: "rgba(0,0,0,.25)", text: "#7fa594", hint: "#4d6b5e", cur: "#ffd84a" } },
  light: { bg: "#d7ebdf", px: ["8,96,52", "0,98,118", "30,70,50"], pxA: 1.5, pxS: 0, speed: 2.2, scl: ["#ffffff", "#eef4f0", "#cfddd5"],
           iris: ["#0f5536", "#1d7f4e", "#43a873", "#0b452b"], lidShade: true,
           res: { "correct": "#0bb862", "detected": "#0bb862", "box off": "#e08a00", "wrong ID": "#d02fab", "missed": "#0096b8", "extra": "#69768a" },
           sw: { bg: "rgba(0,0,0,.04)", text: "#4b695b", hint: "#8ba699", cur: "#c99a00" } } };
const pal = () => PAL[isLight() ? "light" : "dark"];
const RCOL = new Proxy({}, { get: (_, k) => PAL.dark.res[k] });       // the iris is dark in both themes: vivid colours
async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const j = await r.json(); if (!r.ok || j.error) throw new Error(j.error || r.statusText); return j;
}
function toast(m) { const t = $("#toast"); t.textContent = m; t.style.display = "block"; clearTimeout(t._h); t._h = setTimeout(() => t.style.display = "none", 4200); }
const sleep = ms => new Promise(r => setTimeout(r, ms));

/* ================================================================ pixel field */
const BG = (() => {
  const cv = $("#bg"), cx = cv.getContext("2d"); let W, H, px = [], waves = [], mouse = { x: -999, y: -999 }, running = true; const t0 = performance.now();
  function init() {
    const dpr = Math.min(2, devicePixelRatio || 1); W = innerWidth; H = innerHeight; cv.width = W * dpr; cv.height = H * dpr; cx.setTransform(dpr, 0, 0, dpr, 0, 0);
    px = []; for (let i = 0, n = Math.round(W * H / 460); i < n; i++) px.push({ x: Math.random() * W, y: Math.random() * H, s: Math.random() < .8 ? 2 : 3,
      a: .05 + Math.random() * .28, p: Math.random() * 6.28, f: .4 + Math.random() * 1.6, v: .02 + Math.random() * .08,
      ci: Math.random() < .9 ? 0 : (Math.random() < .5 ? 1 : 2) });
    draw(performance.now());
  }
  function draw(now) {
    const t = (now - t0) / 1000, P = pal(); cx.fillStyle = P.bg; cx.fillRect(0, 0, W, H); waves = waves.filter(w => t - w.t < 1.6);
    for (const q of px) {
      if (!calm()) { q.y += q.v * P.speed; if (q.y > H) { q.y = 0; q.x = Math.random() * W; } }
      let x = q.x, y = q.y, a = q.a * (calm() ? 1 : .65 + .35 * Math.sin(t * q.f + q.p));
      const dm = Math.hypot(x - mouse.x, y - mouse.y); if (dm < 110) a += (1 - dm / 110) * .55;
      for (const w of waves) { const r = (t - w.t) * w.v, d = Math.hypot(x - w.x, y - w.y), b = Math.abs(d - r);
        if (b < 48) { const k = (1 - b / 48) * (1 - (t - w.t) / 1.6) * w.amp; x += (x - w.x) / (d || 1) * k * 14; y += (y - w.y) / (d || 1) * k * 14; a += k * .7; } }
      cx.fillStyle = `rgba(${P.px[q.ci]},${Math.min(1, a * P.pxA).toFixed(3)})`; cx.fillRect(x, y, q.s + P.pxS, q.s + P.pxS);
    }
    if (running && !calm()) requestAnimationFrame(draw);
  }
  addEventListener("resize", init); addEventListener("mousemove", e => { mouse.x = e.clientX; mouse.y = e.clientY; });
  document.addEventListener("visibilitychange", () => { running = !document.hidden; if (running) requestAnimationFrame(draw); });
  return { init, wave(x, y, amp = 1) { waves.push({ x, y, t: (performance.now() - t0) / 1000, v: 900, amp }); }, restart() { calm() ? draw(performance.now()) : requestAnimationFrame(draw); } };
})();

/* ================================================================ glitch + shock */
function glitchable(el) { el.addEventListener("mouseenter", () => { if (calm() || el.classList.contains("glitch")) return; el.classList.add("glitch"); setTimeout(() => el.classList.remove("glitch"), 380); }); }
function shock(target, strong = false) {
  if (calm() || !target) return; const r = target.getBoundingClientRect(), x = r.left + r.width / 2, y = r.top + r.height / 2;
  const s = document.createElement("div"); s.className = "shock" + (strong ? " strong" : ""); Object.assign(s.style, { left: x + "px", top: y + "px" });
  s.style.setProperty("--s", strong ? 34 : 22); document.body.appendChild(s); setTimeout(() => s.remove(), 1000);
  const f = document.createElement("div"); f.className = "flash"; f.style.setProperty("--x", x + "px"); f.style.setProperty("--y", y + "px"); document.body.appendChild(f); setTimeout(() => f.remove(), 600);
  BG.wave(x, y, strong ? 1.3 : .8);
  $$(".seg,.tile,.dock").forEach(t => { const b = t.getBoundingClientRect(), dx = b.left + b.width / 2 - x, dy = b.top + b.height / 2 - y, d = Math.hypot(dx, dy) || 1;
    const k = Math.max(0, 1 - d / 950) * (strong ? 14 : 9);
    t.animate([{ translate: "0 0" }, { translate: `${dx / d * k}px ${dy / d * k}px` }, { translate: "0 0" }], { duration: 520, delay: d / 2.2, easing: "cubic-bezier(.2,.9,.3,1)" }); });
}

/* ================================================================ geometry: eyebrow + eye */
let G = {};
function layout() {
  const avail = Math.min(innerWidth - (innerWidth > 1100 ? 250 : 48), 1040), EW = Math.max(560, avail), EH = Math.round(EW * .47);
  document.documentElement.style.setProperty("--ew", EW + "px"); document.documentElement.style.setProperty("--eh", EH + "px");
  const cx = EW / 2, cy = EH / 2, RI = EH * .5, RP = RI * .76, GW = RI * .1, RG = (RI + RP) / 2 + GW * .1;
  G = { EW, EH, cx, cy, RI, RP, GW, RG, LENS: Math.floor(RP * Math.SQRT2 * .98) };
  // brow: one tapered band that follows an arch, hair strokes, the six controls set into it along the curve
  const brow = $("#brow"), segs = $$(".seg"), n = segs.length, gap = 9;
  const BW = Math.round(Math.min(EW * 1.2, innerWidth - (innerWidth > 1100 ? 240 : 24)));
  brow.style.width = BW + "px";
  const U = .78, sw = (U * BW - gap * (n - 1)) / n;                                    // the panels cover |u| <= U
  segs.forEach(sg => { sg.style.width = sw + "px"; sg.style.transform = "none"; });
  const segH = Math.max(...segs.map(sg => sg.offsetHeight)), A = Math.round(EW * .09), half = BW / 2;
  const prof = u => Math.pow(Math.max(0, 1 - Math.pow(Math.abs(u), 3.4)), .5);          // thickness profile: full through the middle, tapering tails
  const PAD = 13, uOut = (U * BW / 2 - sw / 2) / half;
  const yTop = u => 6 + A * u * u;
  // panels hang from the top edge, so the band must reach PAD below every panel's lowest corner
  // even thickness along the body (just enough for the tilted panels), smooth taper in the tails
  const ext = segs.map((_, i) => { const xc = half - U * BW / 2 + sw / 2 + i * (sw + gap), u = (xc - half) / half, a = Math.min(.21, Math.abs(Math.atan(2 * A * u / half)));
    return Math.abs(Math.sin(a)) * sw + Math.cos(a) * segH; });
  const T0 = PAD * 2 + segH / 2 + Math.max(...ext) / 2 - 4, DROP = 18;           // the lower edge curves down a little more towards the tails
  const body = v => T0 + DROP * Math.min(1, v / U) ** 2;
  const th = u => { const v = Math.abs(u); if (v <= U) return body(v); const k = (v - U) / (1 - U); return Math.max(1.5, body(U) * Math.pow(.5 + .5 * Math.cos(Math.PI * Math.min(1, k)), .85)); };
  const Tc = T0, yMid = u => yTop(u) + PAD + segH / 2;
  const H = Math.ceil(Math.max(6 + Tc, A + 18)) + 4; brow.style.height = H + "px";
  brow.style.marginBottom = Math.round(6 + Tc + 14 - H) + "px";            // the eye's top sits 14 px below the brow's lower edge
  segs.forEach((sg, i) => {
    const xc = half - U * BW / 2 + sw / 2 + i * (sw + gap), u = (xc - half) / half, du = .01;
    const slope = (yTop(u + du) - yTop(u - du)) / (2 * du * half), ang = Math.max(-12, Math.min(12, Math.atan(slope) * 180 / Math.PI));
    Object.assign(sg.style, { left: (xc - sw / 2) + "px", top: (yMid(u) - segH / 2) + "px", height: segH + "px", transform: `rotate(${ang}deg)` });
  });
  const N = 90, topPts = [], botPts = [];
  for (let k = 0; k <= N; k++) { const u = -1 + 2 * k / N, x = half + u * half; topPts.push(`${x.toFixed(1)},${yTop(u).toFixed(1)}`); botPts.unshift(`${x.toFixed(1)},${(yTop(u) + th(u)).toFixed(1)}`); }
  let seed = 7; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;     // same hairs on every resize
  let hairs = "";
  for (let k = 0; k < 420; k++) {
    const u = (rnd() * 2 - 1) * .985, t = th(u); if (t < 3) continue;
    const x = half + u * half, y = yTop(u) + (.12 + .76 * rnd()) * t, len = 7 + rnd() * 11 * (.4 + .6 * t / T0);
    const base = Math.atan(2 * A * u / half), lean = (u >= 0 ? -1 : 1) * (.42 + .25 * rnd());   // hair grows outwards from the centre
    const a2 = base + lean + (u >= 0 ? 0 : Math.PI), dx = Math.cos(a2) * len, dy = Math.sin(a2) * len;
    hairs += `<line class="hair" x1="${x.toFixed(1)}" y1="${y.toFixed(1)}" x2="${(x + dx).toFixed(1)}" y2="${(y + dy).toFixed(1)}"/>`;
  }
  $("#arcsvg").setAttribute("viewBox", `0 0 ${BW} ${H}`);
  $("#arcsvg").innerHTML = `<defs><linearGradient id="browfill" x1="0" y1="0" x2="0" y2="1"><stop offset="0" class="bf1"/><stop offset="1" class="bf2"/></linearGradient></defs>
    <polygon class="browband" points="${topPts.join(" ")} ${botPts.join(" ")}"/>${hairs}
    <polyline class="browedge" points="${topPts.join(" ")}"/>`;
  // eye
  const almond = `M 0 ${cy} C ${EW * .22} ${-EH * .16}, ${EW * .78} ${-EH * .16}, ${EW} ${cy} C ${EW * .78} ${EH * 1.16}, ${EW * .22} ${EH * 1.16}, 0 ${cy} Z`;
  let lines = ""; for (let k = 0; k < 72; k++) { const a = k / 72 * Math.PI * 2; lines += `<line class="irisline" x1="${cx + Math.cos(a) * RP}" y1="${cy + Math.sin(a) * RP}" x2="${cx + Math.cos(a) * RI}" y2="${cy + Math.sin(a) * RI}"/>`; }
  $("#eyesvg").setAttribute("viewBox", `0 0 ${EW} ${EH}`);
  $("#eyesvg").innerHTML = `<defs><clipPath id="alm"><path d="${almond}"/></clipPath>
      <radialGradient id="scl" cx="50%" cy="50%" r="60%"><stop offset="0" stop-color="${pal().scl[0]}"/><stop offset=".7" stop-color="${pal().scl[1]}"/><stop offset="1" stop-color="${pal().scl[2]}"/></radialGradient>
      <radialGradient id="irisg" cx="50%" cy="50%" r="50%"><stop offset=".78" stop-color="${pal().iris[0]}"/><stop offset="${isLight() ? .88 : .93}" stop-color="${pal().iris[1]}"/>${isLight() ? `<stop offset=".96" stop-color="${pal().iris[2]}"/>` : ""}<stop offset="1" stop-color="${pal().iris[3]}"/></radialGradient>
      <linearGradient id="lidsh" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="rgba(20,60,40,.22)"/><stop offset=".32" stop-color="rgba(20,60,40,0)"/></linearGradient></defs>
    <path class="sclera" d="${almond}"/>${pal().lidShade ? `<path d="${almond}" fill="url(#lidsh)"/>` : ""}
    <g clip-path="url(#alm)"><g id="irisgrp"><circle class="irisbase" cx="${cx}" cy="${cy}" r="${RI}"/>${lines}
      <g class="gauge" id="gauge" style="--gw:${GW}px"></g>
      <g class="detect" id="detect" role="button" tabindex="0" aria-label="Detect">
        <path class="ringhit" fill-rule="evenodd" d="M ${cx - RI} ${cy} a ${RI} ${RI} 0 1 0 ${2 * RI} 0 a ${RI} ${RI} 0 1 0 ${-2 * RI} 0 Z M ${cx - RP} ${cy} a ${RP} ${RP} 0 1 1 ${2 * RP} 0 a ${RP} ${RP} 0 1 1 ${-2 * RP} 0 Z"/>
        <circle class="ringglow" cx="${cx}" cy="${cy}" r="${(RI + RP) / 2}" style="stroke-width:${RI - RP}px"/>
        <path id="dettext" d="M ${cx - (RI + RP) / 2 - 1} ${cy} A ${(RI + RP) / 2 + 1} ${(RI + RP) / 2 + 1} 0 0 0 ${cx + (RI + RP) / 2 + 1} ${cy}" fill="none"/>
        <text class="detlabel" style="font-size:${Math.max(10, (RI - RP) * .37).toFixed(1)}px" dy="${((RI - RP) * .13).toFixed(1)}"><textPath href="#dettext" startOffset="50%" text-anchor="middle">DETECT</textPath></text>
      </g>
      <circle cx="${cx}" cy="${cy}" r="${RI}" fill="none" stroke="rgba(57,255,136,.35)" stroke-width="1.5"/>
      <circle class="glint" cx="${cx + RP * .55}" cy="${cy - RP * .62}" r="${RP * .07}"/></g></g>`;
  $("#lidsvg").setAttribute("viewBox", `0 0 ${EW} ${EH}`);
  $("#lidsvg").innerHTML = `<defs><clipPath id="alm2"><path d="${almond}"/></clipPath></defs>
    <g clip-path="url(#alm2)"><g class="lid upper" id="lidtop" transform="translate(0 ${-(cy + 6) * lidF})"><rect class="skin" x="0" y="-4" width="${EW}" height="${cy + 4}"/><line class="edge" x1="0" y1="${cy}" x2="${EW}" y2="${cy}"/></g>
      <g class="lid lower" id="lidbot" transform="translate(0 ${(cy + 6) * lidF})"><rect class="skin" x="0" y="${cy}" width="${EW}" height="${cy + 4}"/><line class="edge" x1="0" y1="${cy}" x2="${EW}" y2="${cy}"/></g></g>
    <path class="almond" d="${almond}"/>`;
  const p = $("#pupil"); Object.assign(p.style, { width: 2 * RP + "px", height: 2 * RP + "px", left: cx - RP + "px", top: cy - RP + "px" });
  p.style.setProperty("--pring", Math.max(3, RI * .028).toFixed(1) + "px");
  Object.assign($("#lens").style, { width: G.LENS + "px", height: G.LENS + "px" });
  setStep();
  const btn = $("#detect"); btn.onclick = e => { e.stopPropagation(); detect(btn); };
  btn.onkeydown = e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); detect(btn); } };
  renderGauge(); renderPupil(false); layoutReview(); placeHints();
}
function layoutReview() {
  /* report cards on a gentle curve below the eye (middle lower, ends higher), tilted along it */
  const rev = $("#review"), tiles = $$("#review .tile"); if (!tiles.length) return;
  tiles.forEach(t => t.style.transform = "none");
  const R = rev.getBoundingClientRect(), half = R.width / 2, A = Math.round(G.EH * .12);
  let maxDy = 0;
  tiles.forEach(t => { const r = t.getBoundingClientRect(), u = (r.left + r.width / 2 - R.left - half) / half;
    const dy = A * (1 - u * u), ang = Math.max(-8, Math.min(8, Math.atan(-2 * A * u / half) * 180 / Math.PI));
    t.style.transform = `translateY(${dy.toFixed(1)}px) rotate(${ang.toFixed(2)}deg)`; maxDy = Math.max(maxDy, dy); });
  rev.style.paddingBottom = Math.ceil(maxDy + 10) + "px";
  /* the light: radiating from the pupil's lower half, one beam onto each card, rays spread along the pupil's edge */
  const M = $("#main").getBoundingClientRect(), Pp = $("#pupil").getBoundingClientRect(), beam = $("#beam");
  const pcx = Pp.left + Pp.width / 2 - M.left, pcy = Pp.top + Pp.height / 2 - M.top, pr = Pp.width / 2;
  const onPupil = deg => [pcx + pr * Math.cos(deg * Math.PI / 180), pcy + pr * Math.sin(deg * Math.PI / 180)];
  const cards = tiles.map(t => t.getBoundingClientRect()), bottom = R.bottom - M.top, Hb = bottom + 40;
  beam.setAttribute("viewBox", `0 0 ${M.width} ${Hb}`); beam.style.height = Hb + "px";
  const nC = cards.length, a0 = 150, a1 = 30;                          // the pupil's lower half, left to right (degrees, y down)
  let beams = "", rays = "";
  cards.forEach((c, i) => {
    const s0 = a0 + (a1 - a0) * (i / nC) - 4, s1 = a0 + (a1 - a0) * ((i + 1) / nC) + 4;  // this card's slice of the pupil edge
    const [x0, y0] = onPupil(s0), [x1, y1] = onPupil(s1), top = c.top - M.top + 6;
    beams += `<polygon class="cardbeam" points="${x0},${y0} ${x1},${y1} ${c.right - M.left - 6},${top} ${c.left - M.left + 6},${top}"/>`;
  });
  const NR = 17;
  for (let k = 0; k < NR; k++) { const f = k / (NR - 1), [x1, y1] = onPupil(a0 + (a1 - a0) * f);
    const L = cards[0].left - M.left, Rr = cards[nC - 1].right - M.left, x2 = L + (Rr - L) * f;
    const ci = Math.min(nC - 1, Math.floor(f * nC)), y2 = cards[ci].top - M.top + 4, delay = ((k * .37) % 1 * 3.2).toFixed(2);
    rays += `<line class="ray" x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" style="animation-delay:-${delay}s"/>`; }
  const gy0 = pcy + G.RI * .55, gy1 = Math.max(...cards.map(c => c.top - M.top)) + 10;
  const irisR = G.RI + 2;                                                  // the light starts at the iris's outer edge
  beam.innerHTML = `<defs><mask id="beammask" maskUnits="userSpaceOnUse" x="0" y="0" width="${M.width}" height="${Hb}">
      <rect width="${M.width}" height="${Hb}" fill="#fff"/><circle cx="${pcx}" cy="${pcy}" r="${irisR}" fill="#000"/></mask>
      <linearGradient id="beamg" gradientUnits="userSpaceOnUse" x1="0" y1="${gy0}" x2="0" y2="${gy1}">
      <stop offset="0" class="bg1"/><stop offset=".6" class="bg2"/><stop offset="1" class="bg3"/></linearGradient>
      <linearGradient id="rayg" gradientUnits="userSpaceOnUse" x1="0" y1="${gy0}" x2="0" y2="${gy1}"><stop offset="0" class="rg1"/><stop offset="1" class="rg2"/></linearGradient></defs>
    <g mask="url(#beammask)"><g class="cone">${beams}</g>${rays}</g>`;
}
function followReview() {
  /* keep the cards' curve and the light beams aligned while the column widths animate */
  const t0 = performance.now(), step = now => { layoutReview(); if (now - t0 < 450 && !calm()) requestAnimationFrame(step); };
  calm() ? layoutReview() : requestAnimationFrame(step);
}
function dropCards() {
  /* the eye sheds light: the beam brightens and the cards fall into place, one after another */
  if (calm()) return; const beam = $("#beam"); beam.classList.remove("beam-pulse"); void beam.getBoundingClientRect(); beam.classList.add("beam-pulse");
  $$("#review .tile").forEach((t, i) => { t.classList.remove("fall"); void t.offsetWidth; t.style.setProperty("--fd", (.12 + i * .11) + "s"); t.classList.add("fall"); });
  setTimeout(() => $$("#review .tile").forEach(t => t.classList.remove("fall")), 1500);
}

/* ================================================================ iris = score gauge */
function arc(r, a0, a1) { const p = a => [G.cx + r * Math.cos(a), G.cy + r * Math.sin(a)], [x0, y0] = p(a0), [x1, y1] = p(a1);
  return `M ${x0} ${y0} A ${r} ${r} 0 ${a1 - a0 > Math.PI ? 1 : 0} 1 ${x1} ${y1}`; }
function renderGauge(animate) {
  const g = $("#gauge"); if (!g) return; const det = st.det;
  if (st.warm === "loading" || st.busy) { g.innerHTML = `<path class="show spin" d="${arc(G.RG, 0, Math.PI * 1.4)}" stroke="rgba(57,255,136,.55)" style="stroke-dasharray:6 8"/>`; return; }
  if (!det) { let t = ""; for (let k = 0; k < 60; k++) { const a = k / 60 * Math.PI * 2; t += `<line class="tick" x1="${G.cx + Math.cos(a) * (G.RG - G.GW * .3)}" y1="${G.cy + Math.sin(a) * (G.RG - G.GW * .3)}" x2="${G.cx + Math.cos(a) * (G.RG + G.GW * .3)}" y2="${G.cy + Math.sin(a) * (G.RG + G.GW * .3)}"/>`; }
    g.innerHTML = t; return; }
  const items = det.has_truth ? det.table.map(r => r.result) : det.boxes.map(() => "detected"), n = Math.max(1, items.length), gapA = n > 1 ? .035 : 0, start = -Math.PI / 2;
  g.innerHTML = items.map((r, i) => { const a0 = start + i / n * 2 * Math.PI + gapA / 2, a1 = start + (i + 1) / n * 2 * Math.PI - gapA / 2;
    return `<path d="${arc(G.RG, a0, Math.max(a0 + .001, a1))}" stroke="${RCOL[r] || "#39ff88"}" data-i="${i}"/>`; }).join("");
  const ps = $$("#gauge path"); if (!animate || calm()) ps.forEach(p => p.classList.add("show"));
  else ps.forEach((p, i) => setTimeout(() => p.classList.add("show"), 700 + i * (n > 20 ? 35 : 60)));
}

/* ================================================================ pupil content */
function svgFor(det, w, h, animate) {
  if (!det) return ""; const step = det.boxes.length > 30 ? .025 : .045, base = animate ? .78 : 0; let k = 0; const out = [];
  if (det.has_truth) out.push(...det.truths.map(t => `<g class="truth"><rect x="${t.x1}" y="${t.y1}" width="${t.x2 - t.x1}" height="${t.y2 - t.y1}"/></g>`));
  const fs = Math.max(11, w / 70), lab = (b, txt) => { const tw = txt.length * fs * .62 + 8;
    return `<g class="lab"><rect x="${b.x1}" y="${Math.max(0, b.y1 - fs - 4)}" width="${tw}" height="${fs + 4}" rx="2"/><text x="${b.x1 + 4}" y="${Math.max(fs, b.y1 - 4)}" style="font-size:${fs}px">${esc(txt)}</text></g>`; };
  for (const b of det.boxes) { const r = cls(b.result), err = ["box-off", "wrong-ID", "extra"].includes(r) ? " err" : "", d = animate ? (base + step * k++).toFixed(3) + "s" : "0s";
    out.push(`<g class="box det b-${r}${err}" data-b="${b.k}" style="--d:${d}"><rect class="f" x="${b.x1}" y="${b.y1}" width="${b.x2 - b.x1}" height="${b.y2 - b.y1}"/>${lab(b, b.id + " " + b.conf.toFixed(2))}</g>`); }
  if (det.has_truth) det.table.forEach((r, i) => { if (r.result !== "missed") return; const t = det.truths[r.gt], d = animate ? (base + step * k++).toFixed(3) + "s" : "0s";
    out.push(`<g class="box det missed b-missed" data-row="${i}" style="--d:${d}"><rect class="f" x="${t.x1}" y="${t.y1}" width="${t.x2 - t.x1}" height="${t.y2 - t.y1}"/>${lab(t, t.id + " missed")}</g>`); });
  return `<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">${out.join("")}<g class="hlg"></g></svg>`;
}
const imgSrc = () => "data:image/jpeg;base64," + ((st.det && st.det.image) || st.grid.image);
function renderPupil(animate, cascade) {
  const lens = $("#lens"); if (!st.grid) return;
  const g = st.grid, s = Math.min(G.LENS / g.w, G.LENS / g.h), w = g.w * s, h = g.h * s;
  lens.innerHTML = `<div class="zoomer" id="zoomer" style="width:${w}px;height:${h}px"><img src="${imgSrc()}" alt="">${svgFor(st.det, g.w, g.h, animate && !calm())}</div>`;
  if (animate && st.det && !calm()) { const sc = document.createElement("div"); sc.className = "scan"; $("#zoomer").appendChild(sc); setTimeout(() => sc.remove(), 900); }
  if (cascade && g.kind === "grid" && !calm()) coverCells($("#zoomer"), g, .03);
  bindBoxes($("#zoomer")); st.zoomed = null; if (st.sel != null) highlightIn($("#zoomer"), st.sel);
}
function coverCells(el, g, step) { const cw = 100 / g.cols, ch = 100 / g.rows;
  for (let r = 0; r < g.rows; r++) for (let c = 0; c < g.cols; c++) { const d = document.createElement("div"); d.className = "cover";
    Object.assign(d.style, { left: c * cw + "%", top: r * ch + "%", width: cw + .3 + "%", height: ch + .3 + "%", animationDelay: (r + c) * step + "s" }); el.appendChild(d); setTimeout(() => d.remove(), 1600); } }
function bindBoxes(root) { $$(".box", root).forEach(el => { el.onmouseenter = e => showTip(e, el); el.onmousemove = moveTip; el.onmouseleave = hideTip;
  el.onclick = ev => { ev.stopPropagation(); const i = el.dataset.row != null ? +el.dataset.row : st.det.table.findIndex(r => r.pred === +el.dataset.b); if (i >= 0) select(i, true); }; }); }
function regionOf(i) {
  const det = st.det, g = st.grid, r = det.table[i]; if (!r) return null; const b = r.gt != null && det.truths[r.gt] ? det.truths[r.gt] : det.boxes[r.pred]; if (!b) return null;
  if (g.cols && g.rows) { const cx = Math.floor((b.x1 + b.x2) / 2 / 224), cy = Math.floor((b.y1 + b.y2) / 2 / 224); return [cx * 224, cy * 224, cx * 224 + 224, cy * 224 + 224]; }
  const m = .15 * Math.max(b.x2 - b.x1, b.y2 - b.y1, 20); return [b.x1 - m, b.y1 - m, b.x2 + m, b.y2 + m];
}
function highlightIn(root, i) { const hl = $(".hlg", root), g = st.grid, R = regionOf(i); if (!hl || !R) return; const [x1, y1, x2, y2] = R;
  hl.innerHTML = `<path class="hl" d="M0 0H${g.w}V${g.h}H0Z M${x1} ${y1}V${y2}H${x2}V${y1}Z"/><rect class="hlf" x="${x1}" y="${y1}" width="${x2 - x1}" height="${y2 - y1}"/>`; }
function zoomTo(i) {
  const z = $("#zoomer"); if (!z) return; const R = regionOf(i), g = st.grid; if (!R) return;
  const s = z.clientWidth / g.w, cw = (R[2] - R[0]) * s, ch = (R[3] - R[1]) * s, k = Math.min(3.2, G.LENS * .62 / Math.max(cw, ch));
  if (k <= 1.05) { z.style.transform = ""; st.zoomed = null; return; }
  const mx = (R[0] + R[2]) / 2 * s, my = (R[1] + R[3]) / 2 * s, W = z.clientWidth, H = z.clientHeight;
  z.style.transform = `translate(${W / 2 - mx * k}px,${H / 2 - my * k}px) scale(${k})`; st.zoomed = i;
}
function zoomOut() { const z = $("#zoomer"); if (z) z.style.transform = ""; st.zoomed = null; }

/* ================================================================ eye life: follow cursor, look around, blink, dilate */
let follow = { x: 0, y: 0 };
addEventListener("mousemove", e => {
  if (calm() || st.busy || st.zoomed != null) return; const r = $("#eye").getBoundingClientRect(), dx = e.clientX - (r.left + r.width / 2), dy = e.clientY - (r.top + r.height / 2), d = Math.hypot(dx, dy) || 1;
  const k = Math.min(1, d / 600) * 9; follow = { x: dx / d * k, y: dy / d * k * .7 }; applyFollow();
});
function applyFollow() { $("#irisgrp") && ($("#irisgrp").style.transform = `translate(${follow.x}px,${follow.y}px)`); $("#pupil").style.translate = `${follow.x}px ${follow.y}px`; }
function lookAround() { if (calm()) return; const seq = [[-14, 0], [14, -3], [-6, 3], [0, 0]], els = [$("#irisgrp"), $("#pupil")];
  els[0].animate(seq.map(([x, y]) => ({ transform: `translate(${x}px,${y}px)` })), { duration: 900, easing: "ease-in-out" });
  els[1].animate(seq.map(([x, y]) => ({ translate: `${x}px ${y}px` })), { duration: 900, easing: "ease-in-out" }); }
/* lids: positioned with the SVG transform attribute and animated frame by frame (requestAnimationFrame).
   The browser's own animation engine stopped painting the upper lid while it moved. */
let lidF = 0, lidRun = 0;
const EASE = { in: k => k * k * k, out: k => 1 - Math.pow(1 - k, 3), inout: k => k < .5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2 };
function setLids(f) { const cy = G.cy + 6; lidF = f; $("#lidtop")?.setAttribute("transform", `translate(0 ${-cy * f})`); $("#lidbot")?.setAttribute("transform", `translate(0 ${cy * f})`); }
function lidsTo(f, ms, ease = "inout") {          // f = 0 closed ... 1 fully open
  const id = ++lidRun, f0 = lidF, t0 = performance.now(), E = EASE[ease] || EASE.inout;
  if (ms <= 1 || calm()) { setLids(f); return Promise.resolve(); }
  return new Promise(res => { const step = now => { if (id !== lidRun) return res();       // a newer movement took over
      const k = Math.min(1, (now - t0) / ms); setLids(f0 + (f - f0) * E(k)); k < 1 ? requestAnimationFrame(step) : res(); };
    requestAnimationFrame(step); });
}
function moveLids(closed, ms) { return lidsTo(closed ? 0 : 1, ms, closed ? "in" : "out"); }
async function blinkClose() { if (calm()) return; await moveLids(true, 190); }
async function blinkOpen() { if (calm()) return; await moveLids(false, 280); }

/* ================================================================ steps, guidance, hints */
function setStep() {
  const s = !st.grid ? 1 : !st.det ? 2 : 3;
  $("#d-rand").classList.toggle("next", s === 1); $("#detect")?.classList.toggle("next", s === 2);
  $$("#t-score,#t-err").forEach(t => t.classList.toggle("next", s === 3 && !st.reviewedOnce));
  if (s === 3 && !st.reviewedOnce) setTimeout(() => { st.reviewedOnce = true; $$("#t-score,#t-err").forEach(t => t.classList.remove("next")); }, 4500);
}
function placeHints() {
  $$(".hintdot").forEach(h => h.remove()); return;                       // numbered hints removed by request
  if (st.hinted) return; const main = $("#main").getBoundingClientRect();
  const add = (el, n, text, dx = -36, dy = 0) => { const r = el.getBoundingClientRect(), h = document.createElement("div"); h.className = "hintdot";
    h.innerHTML = n + `<span>${text}</span>`; Object.assign(h.style, { left: (r.left - main.left + dx) + "px", top: (r.top - main.top + r.height / 2 - 13 + dy) + "px" }); $("#main").appendChild(h); };
  add($("#s-grid"), 1, "set up the test here", -8, -64); add($("#detect"), 2, "press random, then detect", $("#detect").offsetWidth + 14); add($("#review"), 3, "review", -36, -100);
}
function retireHints() { if (st.hinted) return; st.hinted = true; $("#main").classList.add("hidehints"); setTimeout(() => $$(".hintdot").forEach(h => h.remove()), 700); }

/* ================================================================ controls on the eyebrow */
function chips(el, items, isOn, onClick) { el.innerHTML = ""; items.forEach(v => { const c = document.createElement("span"); c.className = "chip" + (isOn(v) ? " on" : ""); c.textContent = v.label || v; c.onclick = () => onClick(v); el.appendChild(c); }); }
function renderControls() {
  const c = st.cfg;
  $("#model").innerHTML = c.models.map(m => `<option ${m === st.model ? "selected" : ""}>${esc(m)}</option>`).join("");
  const mi = c.model_info[st.model] || {}; $("#mapv").textContent = `test mAP@0.5 ${mi.test_map50 != null ? mi.test_map50.toFixed(3) : "-"}` + (st.model === c.best ? " · selected" : "");
  $("#conf").value = st.conf; $("#thrv").textContent = (+st.conf).toFixed(2); $("#cell").checked = st.cell;
  chips($("#sizes"), c.grids.map(g => ({ label: g.startsWith("Single") ? "1x1" : g.split(" ")[0], v: g })), x => x.v === st.size, x => { st.size = x.v; renderControls(); });
  $("#sizeinfo").textContent = st.size.replace("Single photo (1x1)", "single photo");
  chips($("#secs"), c.sections, s => st.sections.includes(s), s => { st.sections = st.sections.includes(s) ? st.sections.filter(x => x !== s) : [...st.sections, s]; if (!st.sections.length) st.sections = [s]; renderControls(); });
  chips($("#mixes"), [{ label: "random", v: "Random" }, { label: "look-alikes", v: "Look-alike challenge" }], x => x.v === st.mix, x => { st.mix = x.v; renderControls(); });
  chips($("#layouts"), (c.layouts || ["grid"]).map(v => ({ label: v, v })), x => x.v === st.layout, x => { st.layout = x.v; renderControls(); });
  chips($("#degs"), c.degrade.map(d => ({ label: d.startsWith("All") ? "all" : d.toLowerCase().replace("low resolution", "low-res").replace("heavy jpeg", "jpeg"), v: d })), x => x.v === st.degrade, x => { st.degrade = x.v; renderControls(); });
}
const pickModel = m => { st.model = m; const t = (st.cfg.model_conf || {})[m]; st.conf = t != null ? t : st.cfg.conf; renderControls();
  if ($("#ov-stage").classList.contains("show")) syncStageControls(); };
$("#model").onchange = e => pickModel(e.target.value);
$("#conf").oninput = e => { st.conf = +e.target.value; $("#thrv").textContent = st.conf.toFixed(2); };
$("#cell").onchange = e => st.cell = e.target.checked;
const validatedConf = () => { const t = (st.cfg.model_conf || {})[st.model]; return t != null ? t : st.cfg.conf; };   // each model's own validated threshold
$("#resetthr").onclick = e => { shock(e.currentTarget); st.conf = validatedConf(); renderControls(); };
$("#dice").onclick = e => { shock(e.currentTarget); $("#seed").value = ""; };
$("#keep").onclick = e => { shock(e.currentTarget); if (st.lastSeed != null) $("#seed").value = st.lastSeed; };

/* ================================================================ review tiles */
function countUp(el, to, ms = 900) { if (calm()) { el.textContent = to; return; } const t0 = performance.now();
  const f = n => { const k = Math.min(1, (n - t0) / ms); el.textContent = Math.round(to * (1 - Math.pow(1 - k, 3))); if (k < 1) requestAnimationFrame(f); }; requestAnimationFrame(f); }
function renderReview() {
  const det = st.det;
  if (!det) { $("#sc").textContent = "-"; $("#scof").textContent = ""; $("#scbar").style.width = "0"; $("#verdict").textContent = ""; $("#stats").innerHTML = ""; $("#ms").textContent = "";
    $("#errs").innerHTML = `<div class="lbl">after detection: one sentence per object; press + for every detail, click to zoom the pupil onto it</div>`; $("#sweep").innerHTML = ""; return; }
  $("#ms").textContent = det.ms + " ms" + (det.first_call ? " (incl. model loading)" : "");
  if (det.tally) { const t = det.tally; countUp($("#sc"), t.correct); $("#scof").textContent = "/ " + t.n; setTimeout(() => $("#scbar").style.width = (100 * t.correct / t.n) + "%", 30);
    $("#verdict").innerHTML = t.fully ? `<span style="color:var(--acc)">FULLY CORRECT</span>` : `<span style="color:var(--amber)">${t.named}/${t.n} NAMED CORRECTLY</span>`;
    $("#stats").innerHTML = `<span>box off <b>${t.box_off}</b></span><span>wrong ID <b>${t.wrong}</b></span><span>missed <b>${t.missed}</b></span><span>extra <b>${t.extra}</b></span>` +
      Object.entries(det.per_section).map(([s, v]) => `<span>${esc(s.replace("Section - ", "sec "))} <b>${v[0]}/${v[1]}</b></span>`).join(""); }
  else { countUp($("#sc"), det.boxes.length); $("#scof").textContent = "found"; $("#scbar").style.width = det.boxes.length ? "100%" : "0";
    $("#verdict").innerHTML = det.boxes.length ? "UPLOAD · NO TRUE BOXES" : `<span style="color:var(--amber)">NOTHING DETECTED</span>`; $("#stats").innerHTML = ""; }
  st.exp = new Set(); renderLists(); if ($("#review").classList.contains("wide")) { $("#review").classList.remove("wide"); followReview(); }
  $("#insp").innerHTML = det.table.length ? `<div class="lbl">click any object in the eye or the list, or use ‹ › to step through all ${det.table.length}</div>
      <div class="inav"><button class="enl" id="ip">‹</button><span>– / ${det.table.length}</span><button class="enl" id="in">›</button></div>` : `<div class="lbl">nothing to inspect</div>`;
  if (det.table.length) { $("#ip").onclick = () => select(det.table.length - 1, true); $("#in").onclick = () => select(0, true); }
  $("#sweep").innerHTML = det.sweep ? sweepSvg(det.sweep, det.conf, 360, 170) : `<text x="10" y="30" font-size="11" fill="${pal().sw.hint}" font-family="monospace">needs true boxes (random grids)</text>`;
}
function sweepSvg(s, cur, W, H) {
  const mx = Math.max(1, ...s.correct, ...s.missed, ...s.extra, ...s.wrong), x = t => 10 + (t - .05) / .85 * (W - 20), y = v => H - 22 - v / mx * (H - 36);
  const P = pal(), R = P.res;
  const line = (k, c, d) => `<polyline fill="none" stroke="${c}" stroke-width="2" ${d ? 'stroke-dasharray="4 3"' : ""} points="${s.t.map((t, i) => x(t) + "," + y(s[k][i])).join(" ")}"/>`;
  return `<rect width="${W}" height="${H}" fill="${P.sw.bg}" rx="8"/>` + line("correct", R["correct"]) + line("missed", R["missed"], 1) + line("extra", R["extra"], 1) + line("wrong", R["wrong ID"]) +
    `<line x1="${x(cur)}" x2="${x(cur)}" y1="8" y2="${H - 20}" stroke="${P.sw.cur}" stroke-dasharray="3 3"/>` +
    `<text x="10" y="${H - 6}" font-size="9" fill="${P.sw.text}" font-family="monospace">threshold 0.05→0.90 · green correct · cyan missed · grey extra · pink wrong</text>`;
}
/* ---------- one clear sentence per object, and every detail on demand ---------- */
let GROUPS = {};                                                   // Object ID -> look-alike group (from the server's config)
const f2 = v => v == null ? "-" : (+v).toFixed(2);
const lab = (id, name) => id ? `${id}${name ? " (" + name + ")" : ""}` : "";
function sentence(r) {
  const T = lab(r.true, r.true_name), P = lab(r.predicted, r.predicted_name);
  switch (r.result) {
    case "correct":  return `${T} found correctly, confidence ${f2(r.confidence)}.`;
    case "box off":  return `${T} was named correctly, but its box is placed off (overlap ${f2(r.IoU)}; at least 0.50 is needed).`;
    case "wrong ID": return `${T} was called ${P}` + (String(r.reason).startsWith("look-alike") ? ", a look-alike." : ".");
    case "missed": { const why = String(r.reason), g = why.match(/^not found \(look-alike group: (.+)\)$/);
      if (why === "not found") return `${T} was not found.`;
      if (g) return `${T} was not found (it belongs to the look-alike group ${g[1]}).`;
      if (why.startsWith("not found: ")) return `${T} was not found: ${why.slice(11)}.`;
      return `${T} was missed: ${why}.`; }
    case "extra":    return `Extra box: ${P} at confidence ${f2(r.confidence)}, ${String(r.reason)}.`;
    default:         return `${P} detected, confidence ${f2(r.confidence)}.`;
  }
}
function detailTable(r) {
  const det = st.det, pb = r.pred != null ? det.boxes[r.pred] : null, tb = r.gt != null ? det.truths[r.gt] : null;
  const bx = b => b ? `${Math.round(b.x1)}, ${Math.round(b.y1)} → ${Math.round(b.x2)}, ${Math.round(b.y2)}` : "-";
  const rows = [["result", r.result], ["reason", r.reason || "-"], ["true object", lab(r.true, r.true_name) || "-"], ["predicted", lab(r.predicted, r.predicted_name) || "-"],
    ["section", r.section || "-"], ["look-alike group", GROUPS[r.true || r.predicted] || "-"], ["class AP (test)", r.class_AP ?? "-"],
    ["confidence", f2(r.confidence)], ["IoU with true box", f2(r.IoU)], ["predicted box (px)", bx(pb)], ["true box (px)", bx(tb)]];
  return `<table class="xtab">${rows.map(([k, v]) => `<tr><td>${esc(k)}</td><td>${esc(v)}</td></tr>`).join("")}</table>`;
}
st.exp = new Set(); st.tileFilter = "err";
function rowList(el, filter, zoom) {
  const det = st.det; if (!det) return;
  const rows = det.table.map((r, i) => [r, i]).filter(([r]) => filter === "err" ? !["correct", "detected"].includes(r.result) : true);
  if (!rows.length) {
    el.innerHTML = det.has_truth ? `<div class="lbl" style="color:var(--acc)">no errors: every object found with the right ID${filter === "err" ? ". Choose “all” to see each object" : ""}</div>`
      : `<div class="lbl" style="color:var(--amber)">nothing detected above the threshold ${det.conf.toFixed(2)}. Uploads have no true boxes, so errors cannot be judged; try a lower threshold, or an upright photo</div>`;
    return; }
  el.innerHTML = rows.map(([r, i]) => `<div class="item r-${cls(r.result)}${i === st.sel ? " sel" : ""}" data-i="${i}"><span class="dot"></span>
      <div class="ib"><div class="sent">${esc(sentence(r))}</div>${st.exp.has(i) ? detailTable(r) : ""}</div>
      <button class="xbtn" data-x="${i}" title="${st.exp.has(i) ? "hide details" : "show every detail"}">${st.exp.has(i) ? "−" : "+"}</button></div>`).join("");
  $$(".item", el).forEach(it => it.onclick = () => select(+it.dataset.i, zoom));
  $$(".xbtn", el).forEach(b => b.onclick = e => { e.stopPropagation(); const i = +b.dataset.x; st.exp.has(i) ? st.exp.delete(i) : st.exp.add(i); renderLists(); });
}
function renderLists() {
  if (!st.det) return;
  rowList($("#errs"), st.tileFilter, true);
  const shown = new Set($$("#errs .item").map(it => +it.dataset.i)), wide = [...st.exp].some(i => shown.has(i));
  if ($("#review").classList.contains("wide") !== wide) { $("#review").classList.toggle("wide", wide); followReview(); }
  if ($("#ov-stage").classList.contains("show")) rowList($("#srows"), st.filter, false);
  const nErr = st.det.table.filter(r => !["correct", "detected"].includes(r.result)).length;
  $$("#errtabs .chip").forEach(c => { c.classList.toggle("on", c.dataset.f === st.tileFilter); c.textContent = (c.dataset.f === "err" ? "errors " : "all ") + (c.dataset.f === "err" ? nErr : st.det.table.length); });
}
$$("#errtabs .chip").forEach(c => c.onclick = () => { st.tileFilter = c.dataset.f; renderLists(); });

/* ---------- inspect: every object, with ‹ › and an enlarged view ---------- */
async function showInspect(i) {
  const det = st.det; if (!det || !det.table.length) return; const n = det.table.length; st.inspI = i;
  $("#insp").innerHTML = `<div class="lbl">loading...</div>`;
  try { const t = await api(`/api/tile/${det.det}/${i}`); if (st.det !== det || st.inspI !== i) return;
    $("#insp").innerHTML = `<img src="data:image/jpeg;base64,${t.image}" alt="" title="click to enlarge" style="cursor:zoom-in"><div class="cap">${esc(sentence(det.table[i]))}</div>
      <div class="inav"><button class="enl" id="ip">‹</button><span>${i + 1} / ${n}</span><button class="enl" id="in">›</button><span class="sp"></span></div>`;
    $("#ip").onclick = () => select((i - 1 + n) % n, true); $("#in").onclick = () => select((i + 1) % n, true); $("#insp img").onclick = () => openInspectBig(i);
  } catch (e) { $("#insp").innerHTML = `<div class="lbl">${esc(e.message)}</div>`; }
}
async function openInspectBig(i) {
  const det = st.det; if (!det || !det.table.length) return toast("Detect first, then inspect.");
  i = i ?? st.inspI ?? st.sel ?? 0; const n = det.table.length, r = det.table[i];
  $("#bigpanel").innerHTML = `<h2>INSPECT · ${i + 1} / ${n}</h2><div class="lbl">loading...</div>`; $("#ov-big").classList.add("show");
  try { const t = await api(`/api/tile/${det.det}/${i}?size=360`);
    $("#bigpanel").innerHTML = `<h2>INSPECT · ${i + 1} / ${n}</h2><img class="bigimg" src="data:image/jpeg;base64,${t.image}" alt="">
      <div class="bignav"><button class="close" id="bp">‹ previous</button><span class="sp"></span><span>${esc(sentence(r))}</span><span class="sp"></span><button class="close" id="bn">next ›</button></div>
      ${detailTable(r)}<div class="row" style="margin-top:12px"><button class="close" id="bclose">close [esc]</button></div>`;
    $("#bp").onclick = () => { const k = (i - 1 + n) % n; select(k, true); openInspectBig(k); };
    $("#bn").onclick = () => { const k = (i + 1) % n; select(k, true); openInspectBig(k); };
    $("#bclose").onclick = () => $("#ov-big").classList.remove("show");
  } catch (e) { $("#bigpanel").innerHTML = `<p>${esc(e.message)}</p>`; }
}
function openSweepBig() {
  const det = st.det; if (!det || !det.sweep) return toast("The sweep needs a random grid with true boxes: press RANDOM, then DETECT.");
  const R = pal().res;
  $("#bigpanel").innerHTML = `<h2>THRESHOLD SWEEP</h2>
    <p>How this grid's result changes with the confidence threshold. The dashed yellow line is the current threshold (${det.conf.toFixed(2)}).</p>
    <svg viewBox="0 0 900 420" style="width:100%;height:auto">${sweepSvg(det.sweep, det.conf, 900, 420)}</svg>
    <div class="legend" style="margin-top:8px"><span style="--c:${R["correct"]}"><i></i>correct objects</span><span style="--c:${R["missed"]}"><i class="dash"></i>missed (incl. box off)</span>
      <span style="--c:${R["extra"]}"><i class="dash"></i>extra boxes (incl. box off)</span><span style="--c:${R["wrong ID"]}"><i></i>wrong IDs</span></div>
    <p class="lbl" style="margin-top:8px">A low threshold finds more objects but allows more extra boxes; a high threshold is stricter but misses more. 0.10 was chosen on validation.</p>
    <div class="row" style="margin-top:12px"><button class="close" id="bclose">close [esc]</button></div>`;
  $("#ov-big").classList.add("show"); $("#bclose").onclick = () => $("#ov-big").classList.remove("show");
}
$("#insp-big").onclick = () => openInspectBig(); $("#sweep-big").onclick = openSweepBig;
$("#sweep").style.cursor = "zoom-in"; $("#sweep").onclick = openSweepBig;

async function select(i, zoom) {
  if (st.sel === i) { clearSel(); return; }
  st.sel = i; $$("#srows .item,#errs .item").forEach(el => el.classList.toggle("sel", +el.dataset.i === i));
  highlightIn($("#zoomer"), i); if ($("#ov-stage").classList.contains("show")) highlightIn($("#fig"), i);
  if (zoom && !$("#ov-stage").classList.contains("show")) zoomTo(i);
  showInspect(i);
}
function clearSel() { st.sel = null; zoomOut(); $$(".hlg").forEach(h => h.innerHTML = ""); $$(".item.sel").forEach(e => e.classList.remove("sel")); }

/* ================================================================ tooltips */
function showTip(e, el) { const det = st.det; if (!det) return; let html;
  if (el.dataset.row != null) { const r = det.table[+el.dataset.row]; html = `<b>${esc(r.true)}</b> ${esc(r.true_name)}<br>missed · ${esc(r.reason)}<br>${esc(r.section)}`; }
  else { const b = det.boxes[+el.dataset.b], r = det.table.find(x => x.pred === b.k) || {};
    html = `<b>${esc(b.id)}</b> ${esc(b.name)} · ${b.conf.toFixed(2)}<br>${esc(b.section)}${b.ap != null ? " · class AP " + b.ap : ""}${b.group ? " · look-alikes: " + esc(b.group) : ""}` +
      (r.result && !["correct", "detected"].includes(r.result) ? `<br><span style="color:var(--amber)">${esc(r.result)}: ${esc(r.reason)}</span>` : ""); }
  const t = $("#tip"); t.innerHTML = html; t.style.display = "block"; moveTip(e); }
function moveTip(e) { const t = $("#tip"); t.style.left = Math.min(innerWidth - 320, e.clientX + 14) + "px"; t.style.top = (e.clientY + 14) + "px"; }
function hideTip() { $("#tip").style.display = "none"; }
$$("[data-tip]").forEach(el => { el.onmouseenter = e => { const t = $("#tip"); t.textContent = el.dataset.tip; t.style.display = "block"; moveTip(e); };
  el.onmousemove = moveTip; el.onmouseleave = hideTip; });

/* ================================================================ expanded pupil (large view) */
function fit(fig, box, w, h) { const s = Math.min(box.clientWidth / w, box.clientHeight / h); fig.style.width = w * s + "px"; fig.style.height = h * s + "px"; }
function openStage() { if (!st.grid) return toast("Press RANDOM (or upload) first."); $("#ov-stage").classList.add("show"); renderStage(); }
function renderStage() {
  const g = st.grid, det = st.det, fig = $("#fig");
  $("#stitle").textContent = g.kind === "grid" ? `GRID ${g.cols}x${g.rows}` : g.kind === "scene" ? `SCENE · ${String(g.mix).toUpperCase()}` : "UPLOAD";
  $("#ssub").textContent = (g.kind === "grid" ? `seed ${g.seed} · ${g.mix.toLowerCase()}` : g.kind === "scene" ? `scene ${g.seed} · unseen background` : `${g.w} x ${g.h}`) + (det ? ` · detected with ${det.model} at ${det.conf.toFixed(2)}${det.degrade !== "None" ? " · " + det.degrade.toLowerCase() : ""}` : " · not detected yet");
  syncStageControls();
  fig.innerHTML = `<img src="${imgSrc()}" alt="">${svgFor(det, g.w, g.h, false)}`; requestAnimationFrame(() => fit(fig, $("#figbox"), g.w, g.h)); bindBoxes(fig);
  const rows = $("#srows");
  if (!det) { $("#sstats").innerHTML = "Press <b>Detect</b> to run the model."; rows.innerHTML = ""; $("#sdl").innerHTML = ""; return; }
  const t = det.tally;
  $("#sstats").innerHTML = t ? `<b>${t.correct} of ${t.n}</b> correct · <b>${t.named} of ${t.n}</b> named correctly · <b>${det.ms} ms</b>` : `<b>${det.boxes.length}</b> objects found · <b>${det.ms} ms</b>`;
  rowList(rows, st.filter, false); $("#sdl").innerHTML = dlLinks(det.det); if (st.sel != null) highlightIn(fig, st.sel);
}
function syncStageControls() {
  const c = st.cfg; if (!c) return;
  $("#sc-size").innerHTML = c.grids.map(g => `<option value="${esc(g)}" ${g === st.size ? "selected" : ""}>${g.startsWith("Single") ? "1x1" : g.split(" ")[0]}</option>`).join("");
  $("#sc-model").innerHTML = c.models.map(m => `<option ${m === st.model ? "selected" : ""}>${esc(m)}</option>`).join("");
  const thr = []; for (let v = .05; v <= .901; v += .05) thr.push(+v.toFixed(2));
  if (!thr.includes(+(+st.conf).toFixed(2))) thr.push(+(+st.conf).toFixed(2));
  const vc = +(+validatedConf()).toFixed(2); if (!thr.includes(vc)) thr.push(vc);
  $("#sc-thr").innerHTML = thr.sort((a, b) => a - b).map(v => `<option value="${v}" ${Math.abs(v - st.conf) < 1e-9 ? "selected" : ""}>${v.toFixed(2)}${Math.abs(v - vc) < 1e-9 ? " (validated)" : ""}</option>`).join("");
  $("#sc-seed").value = $("#seed").value;
}
$("#sc-size").onchange = e => { st.size = e.target.value; renderControls(); };
$("#sc-model").onchange = e => pickModel(e.target.value);
$("#sc-thr").onchange = e => { st.conf = +e.target.value; renderControls(); };
$("#sc-seed").oninput = e => { $("#seed").value = e.target.value; };
$("#sc-dice").onclick = e => { shock(e.currentTarget); $("#sc-seed").value = ""; $("#seed").value = ""; };
const dlLinks = id => ["png", "csv", "json"].map(k => `<a class="btn" href="/api/download/${id}/${k}">${k}</a>`).join("") + `<a class="btn" href="/api/snapshot/${id}">snapshot</a>`;

/* ================================================================ actions */
async function randomGrid(src) {
  shock(src); if (st.busy) return; setBusy(true, "building a grid of held-out test photos...");
  try { st.grid = await api("/api/grid", { size: st.size, sections: st.sections, mix: st.mix, layout: st.layout, seed: $("#seed").value.trim() || null });
    st.det = null; st.sel = null; st.lastSeed = st.grid.seed; $("#lastseed").textContent = "last seed " + st.grid.seed;
    $("#pempty") && $("#pempty").remove(); lookAround(); renderPupil(false, true); renderGauge(); renderReview(); $("#insp").innerHTML = `<div class="lbl">select an error after detecting</div>`;
    if ($("#ov-stage").classList.contains("show")) renderStage();
    $("#status").textContent = `grid ready · seed ${st.grid.seed} · press DETECT or D`;
  } catch (e) { toast(e.message); $("#status").textContent = "error"; } finally { setBusy(false); setStep(); }
}
async function detect(src) {
  shock(src || $("#detect"), true); if (st.busy) return; if (!st.grid) return toast("Press RANDOM (or upload) first.");
  setBusy(true, "detecting..."); $("#pupil").classList.add("narrow"); zoomOut(); renderGauge();
  try {
    const req = api("/api/detect", { id: st.grid.id, model: st.model, conf: st.conf, cell: st.cell, degrade: st.degrade });
    await blinkClose(); const [det] = await Promise.all([req, sleep(calm() ? 0 : 160)]);
    st.det = det; st.sel = null; st.busy = false; renderPupil(true); renderGauge(true); await blinkOpen();
    $("#pupil").classList.remove("narrow"); renderReview(); dropCards(); retireHints();
    if ($("#ov-stage").classList.contains("show")) renderStage();
    $("#status").textContent = `${det.model} · ${det.ms} ms${det.first_call ? " (incl. model loading)" : ""}`;
  } catch (e) { toast(e.message); $("#status").textContent = "error"; await blinkOpen(); $("#pupil").classList.remove("narrow"); }
  finally { setBusy(false); renderGauge(false); setStep(); }
}
function setBusy(b, msg) { st.busy = b; ["#s-det", "#s-new"].forEach(s => $(s).disabled = b); $("#detect")?.classList.toggle("dis", b); if (msg) $("#status").textContent = msg; if (!b && st.det) renderGauge(false); else if (b) renderGauge(); }
$("#d-rand").onclick = e => randomGrid(e.currentTarget); $("#s-new").onclick = e => randomGrid(e.currentTarget);
$("#s-det").onclick = e => detect(e.currentTarget);
$("#pupil").onclick = () => { if (st.zoomed != null) return zoomOut(); openStage(); };
$("#s-close").onclick = () => { $("#ov-stage").classList.remove("show"); hideTip(); };
$$(".tabs .chip").forEach(c => c.onclick = () => { $$(".tabs .chip").forEach(x => x.classList.remove("on")); c.classList.add("on"); st.filter = c.dataset.f; renderStage(); });
/* upload, stress, files, about, snapshot, calm */
$("#d-up").onclick = e => { shock(e.currentTarget); $("#file").click(); };
$("#file").onchange = e => { const f = e.target.files[0]; if (!f) return; const rd = new FileReader();
  rd.onload = async () => { setBusy(true, "uploading..."); try { st.grid = await api("/api/upload", { image: rd.result, name: f.name }); st.det = null; st.sel = null;
      $("#pempty") && $("#pempty").remove(); lookAround(); renderPupil(false); renderGauge(); renderReview(); $("#status").textContent = "image ready · press DETECT or D"; }
    catch (err) { toast(err.message); } finally { setBusy(false); setStep(); e.target.value = ""; } }; rd.readAsDataURL(f); };
$("#d-dl").onclick = e => { shock(e.currentTarget); if (!st.det) return toast("Detect first, then download."); $("#dls").innerHTML = dlLinks(st.det.det); $("#ov-dl").classList.add("show"); };
$("#d-snap").onclick = e => { shock(e.currentTarget); if (!st.det) return toast("Detect first, then take a snapshot."); location.href = `/api/snapshot/${st.det.det}`; };
$("#dlclose").onclick = () => $("#ov-dl").classList.remove("show");
function renderAbout() {
  const c = st.cfg; if (!c) return; const f = v => v == null ? "-" : (+v).toFixed(3), pc = v => v == null ? "-" : (100 * v).toFixed(1) + "%";
  const desc = { yolo26s: "YOLO26-s, trained 60 epochs", yolov8s: "YOLOv8-s, trained 56 epochs (early stop)", yolo26s_ft: "YOLO26-s + fine-tune 1 (OBJ010 rule)",
                 yolov8s_ft: "YOLOv8-s + fine-tune 1 (OBJ010 rule)", yolo26s_ft2: "YOLO26-s + fine-tune 2 (single photos, look-alikes, label fixes)",
                 yolo26s_boost: "improved: human-checked labels, rotation, scattered and cut-out scenes, higher resolution" };
  const E = c.boost_eval, tn = { normal: "normal grids", rotated: "rotated grids", scattered: "scattered scenes", cutout: "cut-outs, any background" };
  $("#aboutboost").innerHTML = E && E.tests ? `<table><tr><th>test set</th><th>objects</th><th>recall</th><th>precision</th><th>wrong IDs</th><th>fully correct</th><th>mAP@0.5</th></tr>` +
    Object.entries(E.tests).map(([k, v]) => { const b = v.before, a = v.after, arrow = (x, y, p) => `${p ? pc(x) : f(x)} → <b>${p ? pc(y) : f(y)}</b>`;
      return `<tr><td>${esc(tn[k] || k)}</td><td>${a.objects}</td><td>${arrow(b.recall, a.recall, 1)}</td><td>${arrow(b.precision, a.precision, 1)}</td><td>${b.wrong_ids} → <b>${a.wrong_ids}</b></td>` +
             `<td>${arrow(b.fully_correct, a.fully_correct, 1)}</td><td>${arrow(b.mAP50, a.mAP50)}</td></tr>`; }).join("") +
    `</table><p class="lbl" style="margin-top:8px">Before: the selected Milestone 2 model. After: the improved model, its threshold (${f(E.chosen_on_validation && E.chosen_on_validation.threshold)}) chosen on validation; each test set used once, on human-checked boxes.</p>`
    : `<p class="lbl">The improved model has not been trained and evaluated yet (M2 extra/boost).</p>`;
  const order = Object.keys(c.model_info).sort((a, b) => (c.model_info[a].val_map ?? 0) - (c.model_info[b].val_map ?? 0));
  $("#aboutmodels").innerHTML = `<table><tr><th>model</th><th>val mAP@0.5</th><th>val mAP@.5:.95</th><th>test mAP@0.5</th><th>test mAP@.5:.95</th><th>precision</th><th>recall</th></tr>` +
    order.map(m => { const v = c.model_info[m], sel = m === c.best;
      return `<tr style="${sel ? "color:var(--acc);font-weight:700" : ""}"><td>${esc(m)}${sel ? " ✓" : ""}<br><span style="color:var(--dim);font-weight:400">${esc(desc[m] || "")}</span></td>` +
        `<td>${f(v.val_map50)}</td><td>${f(v.val_map)}</td><td>${f(v.test_map50)}</td><td>${f(v.test_map)}</td><td>${f(v.test_precision)}</td><td>${f(v.test_recall)}</td></tr>`; }).join("") +
    `</table><p class="lbl" style="margin-top:8px">Selected on validation (${esc(c.selection_rule || "highest validation mAP50-95")}), then tested once on human-verified test boxes.</p>`;
  const sh = c.shapes || {}, names = { "1x1": "single photo", "2x2": "2x2 grid", "3x3": "3x3 grid", "4x4": "4x4 grid", "5x4": "5x4 grid", "5x5": "5x5 grid" };
  $("#aboutshapes").innerHTML = Object.keys(sh).length ? `<table><tr><th>input</th><th>images</th><th>objects found (recall)</th><th>precision</th><th>fully correct images</th></tr>` +
    Object.keys(names).filter(k => sh[k]).map(k => `<tr><td>${names[k]}</td><td>${sh[k].images ?? "-"}</td><td>${pc(sh[k].object_recall)}</td><td>${pc(sh[k].precision)}</td><td>${pc(sh[k].fully_correct)}</td></tr>`).join("") +
    `</table><p class="lbl" style="margin-top:8px">Selected model on a second test set built only from test photos. "Fully correct" is strict: every object right and no extra box.</p>` : `<p class="lbl">not available</p>`;
}
$("#d-about").onclick = e => { shock(e.currentTarget); renderAbout(); $("#ov-about").classList.add("show"); }; $("#aclose").onclick = () => $("#ov-about").classList.remove("show");
$("#d-calm").onclick = e => { document.body.classList.toggle("calm"); e.currentTarget.classList.toggle("on", calm()); follow = { x: 0, y: 0 }; applyFollow(); BG.restart(); };
if (matchMedia("(prefers-reduced-motion: reduce)").matches) { document.body.classList.add("calm"); $("#d-calm").classList.add("on"); }
$("#d-stress").onclick = e => { shock(e.currentTarget); $("#ov-stress").classList.add("show"); }; $("#sclose").onclick = () => $("#ov-stress").classList.remove("show");
$("#sn").oninput = e => $("#snv").textContent = e.target.value;
$("#srun").onclick = async e => {
  shock(e.currentTarget, true); const n = +$("#sn").value; $("#srun").disabled = true; $("#sprog").style.width = "0"; $("#sres").innerHTML = `<p class="lbl">running ${n} grids...</p>`;
  try { const { job } = await api("/api/stress", { size: st.size, n, sections: st.sections, mix: st.mix, degrade: st.degrade, model: st.model, conf: st.conf, cell: st.cell });
    let j; do { await sleep(400); j = await api(`/api/stress/${job}`); $("#sprog").style.width = (100 * j.done / j.total) + "%"; } while (!j.result && !j.error);
    if (j.error) throw new Error(j.error); const R = j.result, T = R.totals, pct = v => (100 * v / Math.max(1, T.n)).toFixed(1) + "%";
    const tbl = (h, rows) => `<table><tr><th>${h}</th><th>times</th></tr>${rows.length ? rows.map(r => `<tr><td>${esc(r[0])}</td><td>${r[1]}</td></tr>`).join("") : `<tr><td>none</td><td></td></tr>`}</table>`;
    $("#sres").innerHTML = `<div class="kpis"><div class="kpi"><div class="v">${pct(T.correct)}</div><div class="k">objects correct (${T.correct}/${T.n})</div></div>` +
      `<div class="kpi"><div class="v">${(100 * R.fully).toFixed(1)}%</div><div class="k">fully correct grids · 95% CI ${(100 * R.ci[0]).toFixed(1)}-${(100 * R.ci[1]).toFixed(1)}%</div></div>` +
      `<div class="kpi"><div class="v">${pct(T.named)}</div><div class="k">named correctly · ${R.ms.toFixed(0)} ms per grid</div></div></div>` +
      `<p class="lbl">${R.n_grids} × ${esc(R.size)} · ${esc(R.mix)} · ${esc(R.degrade)} · ${esc(R.model)} · box off ${T.box_off} · wrong ID ${T.wrong} · missed ${T.missed} · extra ${T.extra}</p>` +
      tbl("most often missed / wrong", R.hardest) + tbl("most common confusions", R.confusions) + tbl("failure reasons", R.reasons);
  } catch (err) { $("#sres").innerHTML = `<p style="color:#ff8fa8">${esc(err.message)}</p>`; } finally { $("#srun").disabled = false; }
};
/* hover titles: what each region is for */
const TITLES = [["#dock", '"What you want to see?"', "right"], ["#brow", '"How you want to see?"', "heading"],
                ["#eye", '"Eye will be your eyes!"', "eye"], ["#review", '"My Report Card"', "cards"], ["#dock2", '"Want me to help more?"', "left"]];
let titlesOn = (() => { try { return localStorage.getItem("m2titles") !== "off"; } catch (e) { return true; } })();
$("#d-titles").classList.toggle("on", titlesOn);
let typing = null, hoverEl = null;
function showTitle(el, text, where) {
  if (!titlesOn || document.body.classList.contains("intro")) return; const h = $("#htitle"), r = el.getBoundingClientRect(); hoverEl = el; clearInterval(typing);
  $(".top h1").classList.toggle("away", where === "heading"); $("#status").classList.toggle("away", where === "cards");
  h.textContent = ""; h.classList.add("show");
  const place = () => { const w = h.offsetWidth, hh = h.offsetHeight; let x, y;
    if (where === "right") { x = r.right + 16; y = r.top + r.height / 2 - hh / 2; }
    else if (where === "left") { x = r.left - w - 16; y = r.top + r.height / 2 - hh / 2; }
    else if (where === "eye") {                         // on the white of the eye, between its left corner and the iris, never over the iris
      const ir = $("#irisgrp").getBoundingClientRect(), right = ir.left - 14, left = r.left + r.width * .05;
      x = Math.min((left + right) / 2 - w / 2, right - w); y = r.top + r.height / 2 - hh / 2; }
    else if (where === "heading") { const t = $(".top h1").getBoundingClientRect(); x = t.left + t.width / 2 - w / 2; y = t.top + t.height / 2 - hh / 2; }   // takes the heading's place
    else if (where === "cards") {                       // centred in the gap between the eye's bottom and the cards, in place of the status line
      const e = $("#eye").getBoundingClientRect(), cx0 = e.left + e.width / 2; x = cx0 - w / 2;
      const under = $$("#review .tile").map(t => t.getBoundingClientRect()).filter(t => t.right > x && t.left < x + w);
      const cardsTop = under.length ? Math.min(...under.map(t => t.top)) : e.bottom + 60; y = (e.bottom + cardsTop) / 2 - hh / 2; }
    else { x = r.left + r.width / 2 - w / 2; y = r.top - hh - 8; }
    h.style.left = Math.max(8, Math.min(innerWidth - w - 8, x)) + "px"; h.style.top = Math.max(8, y) + "px"; };
  if (calm()) { h.textContent = text; place(); return; }
  let k = 0; typing = setInterval(() => { h.textContent = text.slice(0, ++k); place(); if (k >= text.length) clearInterval(typing); }, 34);
}
function hideTitle(el) { if (hoverEl !== el) return; clearInterval(typing); $("#htitle").classList.remove("show"); hoverEl = null;
  $(".top h1").classList.remove("away"); $("#status").classList.remove("away"); }
TITLES.forEach(([sel, text, where]) => { const el = $(sel); el.addEventListener("mouseenter", () => showTitle(el, text, where)); el.addEventListener("mouseleave", () => hideTitle(el)); });
$("#d-titles").onclick = e => { titlesOn = !titlesOn; e.currentTarget.classList.toggle("on", titlesOn); try { localStorage.setItem("m2titles", titlesOn ? "on" : "off"); } catch (err) { }
  if (!titlesOn) { clearInterval(typing); $("#htitle").classList.remove("show"); hoverEl = null; $(".top h1").classList.remove("away"); $("#status").classList.remove("away"); } };
/* light / dark theme (the choice is remembered) */
function syncThemeBtn() { $("#themelbl").textContent = isLight() ? "dark" : "light"; }
$("#d-theme").onclick = e => {
  shock(e.currentTarget); document.documentElement.classList.toggle("light"); syncThemeBtn();
  try { localStorage.setItem("m2theme", isLight() ? "light" : "dark"); } catch (err) { }
  layout(); BG.restart();                                                     // eye colours + pixel field
  if (st.det) $("#sweep").innerHTML = st.det.sweep ? sweepSvg(st.det.sweep, st.det.conf, 360, 170) : $("#sweep").innerHTML;
  if ($("#ov-stage").classList.contains("show")) renderStage();
};
syncThemeBtn();
$("#howbtn").onclick = e => { shock(e.currentTarget); window.jyRelayout?.(); $("#jy").scrollIntoView({ behavior: calm() ? "auto" : "smooth" }); };
/* keys */
addEventListener("keydown", e => {
  if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName)) { if (e.key === "Escape") document.activeElement.blur(); return; }
  const k = e.key.toLowerCase();
  if (k === "escape") { const o = $$(".ov.show"); if (o.length) o.forEach(x => x.classList.remove("show")); else clearSel(); }
  else if (k === "n") randomGrid($("#d-rand")); else if (k === "d") detect($("#detect")); else if (k === "u") $("#file").click();
});
let rz; addEventListener("resize", () => { clearTimeout(rz); rz = setTimeout(() => { layout(); if ($("#ov-stage").classList.contains("show")) renderStage(); }, 120); });

/* ================================================================ start */
async function pollWarm() {
  try { const s = await api("/api/status"); st.warm = s.warm;
    if (s.warm === "ready") { $("#status").textContent = st.grid ? $("#status").textContent : "model ready · press RANDOM or N"; renderGauge(); return; }
    if (s.warm === "error") { st.warm = "ready"; $("#status").textContent = "model loads on first detection"; renderGauge(); return; }
  } catch (e) { }
  setTimeout(pollWarm, 700);
}
(async () => {
  $$(".seg,.tile,.dbtn").forEach(glitchable);
  try { st.cfg = await api("/api/config"); st.model = st.cfg.boost || st.cfg.best; st.conf = ((st.cfg.model_conf || {})[st.model] ?? st.cfg.conf); st.sections = [...st.cfg.sections];
    Object.entries(st.cfg.groups || {}).forEach(([g, ids]) => ids.forEach(o => GROUPS[o] = g));
    const c = st.cfg, fmtn = v => v == null ? "-" : Number(v).toLocaleString("en-US"), nObj = c.n_objects_all ?? c.n_objects;
    $("#m-obj").textContent = `${nObj} objects`;
    $("#m-obj").dataset.tip = `${nObj} Object IDs (classes): the everyday objects photographed by the students of the course`;
    $("#m-photos").textContent = `${fmtn(c.n_photos)} photos`;
    $("#m-photos").dataset.tip = `${fmtn(c.n_photos)} photos in the cleaned dataset, after background-only shots and exact duplicates were removed`;
    $("#m-boxed").textContent = `${fmtn(c.n_boxed)} with boxes`;
    $("#m-boxed").dataset.tip = `${fmtn(c.n_boxed)} photos with a reliable bounding box: the ones used to train and test the model` +
      (c.n_photos && c.n_boxed ? ` (${fmtn(c.n_photos - c.n_boxed)} left out because no reliable box could be made)` : "");
    renderControls(); }
  catch (e) { toast("Cannot reach the server: " + e.message); }
  layout(); BG.init(); setStep(); pollWarm(); document.fonts?.ready.then(() => layoutReview()); await wake();
})();
async function wake() {
  /* the eye wakes up with its name in the pupil; the name flies to the top; then everything else appears */
  const logo = $("#intrologo"), h1 = $(".top h1");
  if (calm() || matchMedia("(prefers-reduced-motion: reduce)").matches) { logo.remove(); document.body.classList.remove("intro"); setLids(1); return; }
  const fitLogo = () => { const pw = $("#pupil").clientWidth, lw = logo.offsetWidth || 1; logo.style.transform = `translate(-50%,-50%) scale(${(pw * .66 / lw).toFixed(3)})`; };
  fitLogo();
  await sleep(600);
  const open1 = lidsTo(.32, 1200, "inout");
  setTimeout(() => logo.classList.add("go"), 380);                           // letters pop in as the lids part
  await open1; await sleep(250);
  await lidsTo(.14, 500, "inout"); await sleep(180);
  const open2 = lidsTo(1, 1400, "out");
  setTimeout(() => logo.classList.add("sub"), 500);
  await open2;
  logo.classList.add("blink"); await sleep(820);                              // the E's blink, like a pair of eyes
  /* fly: a copy of the title travels from the pupil to the heading's place and shrinks into it */
  const from = logo.getBoundingClientRect(), to = h1.getBoundingClientRect();
  const fly = document.createElement("div"); fly.className = "logo flylogo"; fly.innerHTML = h1.innerHTML;
  Object.assign(fly.style, { left: to.left + "px", top: to.top + "px", width: to.width + "px", height: to.height + "px" });
  document.body.appendChild(fly); logo.style.visibility = "hidden";
  const k = from.width / to.width, dx = (from.left + from.width / 2) - (to.left + to.width / 2), dy = (from.top + from.height / 2) - (to.top + to.height / 2);
  const anim = fly.animate([{ transform: `translate(${dx}px,${dy}px) scale(${k})` }, { transform: `translate(${dx * .35}px,${dy * .55 - 24}px) scale(${(k + 1) / 2})`, offset: .55 }, { transform: "none" }],
                           { duration: 1050, easing: "cubic-bezier(.45,0,.2,1)", fill: "forwards" });
  await anim.finished.catch(() => {});
  const top = $(".top"); top.style.transition = "none"; top.style.opacity = "1"; top.style.translate = "0 0";   // the real heading takes over in place
  fly.remove(); logo.remove(); void top.offsetWidth; top.style.transition = ""; await sleep(60);
  const order = [...$$(".brow .seg").map((_, i) => `.brow .seg[data-i="${i}"]`), ".brow .arc", "#dock", "#dock2", ".detrow", ...$$(".review > *").map((_, i) => `.review > :nth-child(${i + 1})`)];
  order.forEach((sel, n) => $$(sel).forEach(el => el.style.transitionDelay = (n * 70) + "ms"));
  document.body.classList.remove("intro");
  setTimeout(() => { order.forEach(sel => $$(sel).forEach(el => el.style.transitionDelay = "")); top.style.opacity = ""; top.style.translate = ""; }, order.length * 70 + 900);
  setTimeout(layoutReview, order.length * 70 + 950);
}
</script>

<script>
(() => {
"use strict";


/* ---------- small drawing helpers (blueprint line art) ---------- */
const W = (d, c = "w") => `<path class="${c}" d="${d}"/>`;
const R = (x, y, w, h, c = "w", rx = 2) => `<rect class="${c}" x="${x}" y="${y}" width="${w}" height="${h}" rx="${rx}"/>`;
const C = (x, y, r, c = "w") => `<circle class="${c}" cx="${x}" cy="${y}" r="${r}"/>`;
const T = (x, y, s, c = "t", a = "middle") => `<text class="${c}" x="${x}" y="${y}" text-anchor="${a}">${s}</text>`;
function worker(x, y, o = {}) {           /* a builder in a hard hat; y = feet */
  const arm = o.arm || "M-10 -26 L-16 -14", arm2 = o.arm2 || "M10 -26 L16 -14";
  return `<g transform="translate(${x},${y})${o.flip ? " scale(-1,1)" : ""}">
    ${W("M0 -40 L0 -18")}${W("M0 -18 L-8 0")}${W("M0 -18 L8 0")}${W("M0 -34 L-10 -26")}${W(arm)}${W("M0 -34 L10 -26")}${W(arm2)}
    ${C(0, -47, 7)}<path class="y" d="M-9 -49 A9 9 0 0 1 9 -49 L11 -48 L-11 -48 Z"/>${o.extra || ""}</g>`;
}
const ground = (y = 168) => W(`M8 ${y} L292 ${y}`) + W(`M8 ${y + 6} L40 ${y + 6} M60 ${y + 6} L120 ${y + 6} M150 ${y + 6} L230 ${y + 6} M250 ${y + 6} L292 ${y + 6}`, "faint");
const photo = (x, y, s = 16, cross = false) => R(x, y, s, s * .78) + (cross ? W(`M${x + 3} ${y + 3} L${x + s - 3} ${y + s * .78 - 3} M${x + s - 3} ${y + 3} L${x + 3} ${y + s * .78 - 3}`) : C(x + s * .5, y + s * .4, s * .18));
const svg = inner => `<svg class="scene sc" viewBox="0 0 300 190" aria-hidden="true">${inner}</svg>`;

/* ---------- the ten scenes ---------- */
const scenes = [
  /* 1 clearing the site: broom, wheelbarrow of rejected photos, neat stack kept */
  svg(ground() + worker(70, 168, { arm2: "M10 -26 L26 -6", extra: W("M26 -6 L40 6") + `<path class="y" d="M34 2 L48 10 L44 14 L30 6 Z"/>` }) +
    photo(112, 156, 14, true) + photo(132, 160, 12, true) + photo(96, 160, 12, true) +
    W("M168 150 L222 150 L214 168 L176 168 Z") + C(190, 172, 5) + W("M222 150 L238 140") + photo(176, 136, 14, true) + photo(194, 134, 14, true) + photo(186, 124, 14, true) +
    photo(250, 150, 18) + photo(250, 135, 18) + photo(250, 120, 18) + photo(250, 105, 18) + T(259, 96, "6,942") +
    T(150, 40, "Clearing 385 empty shots and duplicates") ),
  /* 2 architects at the drafting table: plan with boxes, three drafters */
  svg(ground() + W("M90 120 L210 104 L218 130 L98 146 Z") + W("M120 140 L116 168 M196 128 L202 168") +
    `<g transform="rotate(-7 154 125)">` + R(112, 110, 22, 18, "d", 1) + C(123, 119, 4) + R(142, 108, 28, 20, "d", 1) + C(156, 118, 5) + R(178, 106, 20, 16, "d", 1) + C(188, 114, 3) + `</g>` +
    worker(56, 168, { arm2: "M10 -26 L34 -48" }) + worker(244, 168, { flip: true, arm2: "M10 -26 L30 -50" }) + worker(270, 168, { flip: true, arm2: "M10 -26 L22 -40" }) +
    T(150, 40, "Three drafters, one plan per photo") + `<path class="ys" d="M146 66 l6 6 l12 -14"/>` ),
  /* 3 surveyor: tripod, tape measure bracketing a pair of shoes, rule book */
  svg(ground() + W("M60 168 L72 120 L84 168 M72 120 L72 168") + R(62, 108, 22, 12) + C(73, 114, 3, "wf") + worker(42, 168, { arm2: "M10 -26 L22 -30" }) +
    W("M130 160 Q140 146 156 150 L160 160 Z") + W("M166 160 Q176 146 192 150 L196 160 Z") +
    R(122, 134, 82, 30, "d", 2) + W("M122 120 L204 120") + W("M122 116 L122 124 M204 116 L204 124 M143 118 L143 122 M163 118 L163 122 M183 118 L183 122", "w") +
    R(228, 126, 44, 38, "w", 2) + W("M236 136 L264 136 M236 144 L264 144 M236 152 L256 152") + `<rect class="y" x="228" y="126" width="6" height="38"/>` +
    T(163, 108, "one box, both shoes") + T(150, 40, "Every object gets one written rule") ),
  /* 4 bricklaying: a wall of photo bricks growing from 2x2 to 5x5 */
  svg(ground() + (() => { let s = ""; const sizes = [2, 3, 4, 5]; let x = 84;
      sizes.forEach((n, k) => { const cell = 46 / n, base = 168; for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) s += R(x + c * cell, base - (r + 1) * cell, cell - 1.5, cell - 1.5, "w", 1);
        s += T(x + n * cell / 2, base - n * cell - 6, `${n}×${n}`); x += n * cell + 10; }); return s; })() +
    worker(54, 168, { arm2: "M10 -26 L28 -28", extra: `<rect class="y" x="26" y="-34" width="14" height="9" rx="1"/>` }) + T(150, 40, "Photos laid side by side like bricks") ),
  /* 5 foundation: chute pouring concrete onto a slab labelled COCO, rebar */
  svg(ground(170) + R(70, 140, 200, 30, "w", 2) + (() => { let s = ""; for (let x = 90; x < 262; x += 20) if (x < 118 || x > 222) s += W(`M${x} 140 L${x - 6} 170`, "faint"); return s; })() +
    T(170, 160, "COCO, 118k photos") + W("M210 60 L240 60 L250 82 L226 110") + W("M226 110 L214 136", "ys") + W("M222 110 L210 136", "ys") +
    `<path class="y" d="M196 136 Q210 126 226 136 Z"/>` + worker(44, 170, { arm2: "M10 -26 L24 -18", extra: W("M24 -18 L24 0") }) + T(150, 40, "A foundation of general vision") ),
  /* 6 tower crane lifting a floor onto a rising building: 60 floors */
  svg(ground() + W("M70 168 L70 46 M84 168 L84 46") + (() => { let s = ""; for (let y = 56; y < 168; y += 16) s += W(`M70 ${y} L84 ${y + 16} M84 ${y} L70 ${y + 16}`, "faint"); return s; })() +
    W("M40 46 L250 46 M70 46 L77 30 L84 46") + R(40, 46, 18, 12) + W("M200 46 L200 76") + `<rect class="y" x="174" y="76" width="52" height="7" rx="1"/>` +
    (() => { let s = ""; for (let k = 0; k < 6; k++) s += R(160, 160 - k * 13, 76, 12, "w", 1); return s; })() + T(198, 102, "floor 60") +
    T(150, 24, "Raising the detector, one epoch per floor") ),
  /* 7 finishing crew: ladder, paint roller, two coats */
  svg(ground() + R(150, 60, 110, 108, "w", 2) + `<rect class="y" x="150" y="60" width="54" height="108" opacity=".85"/>` + W("M204 60 L204 168", "d") +
    T(177, 118, "coat 1", "ty") + T(232, 118, "coat 2") + W("M96 168 L118 70 M122 168 L140 76") + W("M100 150 L124 150 M105 128 L128 128 M110 106 L133 106 M115 86 L136 86") +
    worker(126, 104, { arm2: "M10 -26 L22 -36", extra: W("M22 -36 L30 -40") + `<rect class="y" x="28" y="-48" width="8" height="16" rx="2"/>` }) + T(150, 40, "Two finishing coats: fine-tune 1 and 2") ),
  /* 8 judging five scale models: one wins the rosette */
  svg(ground() + W("M40 140 L262 140 M52 140 L52 168 M250 140 L250 168") +
    [[56, 30], [96, 42], [136, 36], [176, 58], [216, 46]].map(([x, h], k) => R(x, 140 - h, 26, h, "w", 1) + W(`M${x + 6} ${140 - h + 8} L${x + 6} ${140 - h + 14} M${x + 18} ${140 - h + 8} L${x + 18} ${140 - h + 14}`, "faint") +
      T(x + 13, 156, ["A", "B", "C", "D", "E"][k])).join("") + T(150, 176, "D: YOLO26-s, fine-tuned twice") +
    C(189, 72, 9, "y") + W("M184 80 L180 96 M194 80 L198 96", "ys") + worker(274, 140, { flip: true, arm2: "M10 -26 L20 -38", extra: R(16, -46, 12, 14) }) +
    T(150, 40, "Five models, judged on validation") ),
  /* 9 final inspection: certificate with stamp 0.947 */
  svg(ground() + R(106, 54, 116, 104, "w", 3) + W("M118 72 L210 72 M118 84 L196 84 M118 96 L204 96") + C(188, 126, 28, "y") +
    T(188, 124, "0.947", "ty") + T(188, 138, "mAP@0.5", "ty") + T(130, 128, "1,041", "t") + T(130, 142, "objects") +
    worker(64, 168, { arm2: "M10 -26 L28 -32", extra: R(26, -42, 12, 15) }) + worker(250, 168, { flip: true }) + T(150, 40, "Tested once, on boxes checked by hand") ),
  /* 10 grand opening: building with the eye, ribbon, scissors */
  svg(ground() + R(108, 58, 120, 110, "w", 3) + W("M126 92 Q168 66 210 92 Q168 118 126 92 Z") + C(168, 92, 11) + C(168, 92, 4, "wf") + R(152, 128, 32, 40, "w", 1) +
    W("M60 146 L276 146", "ys") + W("M88 140 L100 152 M100 140 L88 152") + C(86, 138, 3) + C(86, 154, 3) +
    worker(52, 168, { arm2: "M10 -26 L30 -26" }) + worker(254, 168, { flip: true, arm2: "M10 -26 L20 -44" }) + worker(282, 168, { flip: true, arm2: "M10 -26 L18 -46" }) +
    T(150, 40, "Open to visitors: The EYE") )
];

/* ---------- content: front line + back report (real numbers from the milestone) ---------- */
const S = [
  { t: "Clear the site", f: "Cleaned 7,343 raw files down to 6,942 usable photos of 73 objects.",
    b: ["Removed 385 background-only shots and exact duplicates before splitting, so no copy can sit in two splits.",
        "Kept the split from Milestone 1: 70 / 15 / 15, stratified by object, seed 42.",
        "Excluded a phone screenshot (OBJ054_017) and five OBJ004 photos that show no toy car."],
    g: [["6,942", "photos kept"], ["73", "objects, 89 to 106 photos each"], ["4,859 / 1,041 / 1,042", "train / val / test"]], w: "A clean site means nothing leaks between training and testing." },
  { t: "Draw the plans", f: "Three methods drew a box on every photo; a person checked the test set.",
    b: ["<a href=\"https://arxiv.org/abs/2306.09683\" target=\"_blank\" rel=\"noopener\">OWLv2</a> found each object from its name (17 minutes for 6,936 photos, 16× faster after a GPU memory fix).",
        "A visual search with example crops (hand-drawn for 7 objects) mostly boxed the floor, so it became one vote among three.",
        "A <a href=\"https://en.wikipedia.org/wiki/You_Only_Look_Once\" target=\"_blank\" rel=\"noopener\">YOLO</a> helper trained on training photos only cast the third vote: 3,172 strong, 2,935 auto, 829 to review.",
        "Every test photo was checked by hand; the automatic boxes were 98.9% right."],
    g: [["1,968", "boxes drawn or confirmed by a person"], ["98.9%", "auto boxes correct (test)"], ["97.1%", "random training audit"]], w: "Trustworthy test boxes make every reported number trustworthy." },
  { t: "Survey and set the rules", f: "Checked every object's boxes side by side and fixed five inconsistent definitions.",
    b: ["Sneakers: one box around both shoes, not one shoe.",
        "Tissue: the whole box, not just the paper (AP 0.30 → 0.995).",
        "Analog watch: case and strap (0.59 → 0.81). Power bank: the silver one only (0.71 → 0.95). Paddle: not the tennis racket.",
        "314 photos (4.5%) without a reliable box were left out; no object lost more than 20%."],
    g: [["5", "label rules corrected"], ["0.30 → 0.995", "tissue AP after the fix"], ["6,628", "photos with boxes"]], w: "Consistent labels mattered more than the choice of model." },
  { t: "Lay the bricks", f: "Built 1,500 training images from photos placed side by side, 2×2 to 5×5.",
    b: ["Every photo is a 224-pixel brick; grids range from 448 to 1,120 pixels, and each box is shifted to its cell.",
        "15% single photos, so single images are detected too; half of the grids put look-alikes side by side.",
        "Flips, colour changes and 20% blur, low resolution, compression or darkness.",
        "Validation and test grids use every photo exactly once; a check confirms no photo crosses splits."],
    g: [["1,500", "training images"], ["5", "grid sizes, 4 to 25 objects"], ["10", "look-alike groups"]], w: "The grids are the milestone's input format, built exactly to spec." },
  { t: "Pour the foundation", f: "Started from models that already know how to see: pretrained on COCO.",
    b: ["<a href=\"https://arxiv.org/abs/1405.0312\" target=\"_blank\" rel=\"noopener\">COCO</a>: about 118,000 everyday photos in 80 categories, giving edges, shapes and 'objectness' for free.",
        "Two <a href=\"https://en.wikipedia.org/wiki/You_Only_Look_Once\" target=\"_blank\" rel=\"noopener\">YOLO</a> candidates: <a href=\"https://docs.ultralytics.com/models/yolov8/\" target=\"_blank\" rel=\"noopener\">YOLOv8-s</a>, the brief's model, and <a href=\"https://docs.ultralytics.com/models/yolo26/\" target=\"_blank\" rel=\"noopener\">YOLO26-s</a>, the newest from Ultralytics.",
        "Both are <a href=\"https://en.wikipedia.org/wiki/Convolutional_neural_network\" target=\"_blank\" rel=\"noopener\">convolutional neural networks</a> built for <a href=\"https://en.wikipedia.org/wiki/Object_detection\" target=\"_blank\" rel=\"noopener\">object detection</a>: one look, every box at once.",
        "Input size 1,120 pixels, so photos in 5×5 grids keep their native 224 pixels."],
    g: [["118k", "COCO photos behind the start"], ["2", "architectures compared"], ["1,120 px", "input size"]], w: "Learning 73 new objects is far easier on a strong foundation." },
  { t: "Raise the floors", f: "Trained both detectors for up to 60 epochs on a laptop GPU.",
    b: ["RTX 5070 laptop GPU, 8 GB: batch size 4 at 1,120 pixels, cosine learning rate, early stopping after 12 flat epochs.",
        "YOLO26-s: 60 epochs in 104 minutes. YOLOv8-s: stopped after 56 epochs, 70 minutes.",
        "Chosen on validation: confidence 0.10, one box per 224-pixel cell, and small images placed at natural size on the canvas."],
    g: [["60", "epochs (YOLO26-s)"], ["104 min", "training time"], ["0.921", "first test mAP@0.5"]], w: "The structure stands; now it needs finishing." },
  { t: "Two finishing coats", f: "Fine-tuned twice, 15 epochs each, after improving the data.",
    b: ["Coat 1: both models, after the sneaker rule fix.",
        "Coat 2: YOLO26-s, after adding single photos, look-alike groups and four more label fixes.",
        "In total about 135,000 image passes for the final model."],
    g: [["15 + 15", "fine-tune epochs"], ["0.921 → 0.947", "test mAP@0.5"], ["90", "epochs in total"]], w: "Better data, not more epochs, produced most of the gain." },
  { t: "Judge the models", f: "Compared all five runs on validation and chose before looking at the test set.",
    b: ["Rule set in advance: highest validation mAP@0.5:0.95; threshold and post-processing chosen the same way.",
        "Winner: <a href=\"https://docs.ultralytics.com/models/yolo26/\" target=\"_blank\" rel=\"noopener\">YOLO26-s</a> fine-tuned twice. All five below; scroll for the table."],
    more: "<table class=\"mt\"><tr><th>model</th><th>val mAP@.5:.95</th><th>test mAP@0.5</th></tr>" +
      "<tr><td>A <a href=\"https://docs.ultralytics.com/models/yolov8/\" target=\"_blank\" rel=\"noopener\">YOLOv8-s</a></td><td>0.897</td><td>0.910</td></tr><tr><td>B <a href=\"https://docs.ultralytics.com/models/yolo26/\" target=\"_blank\" rel=\"noopener\">YOLO26-s</a></td><td>0.907</td><td>0.921</td></tr>" +
      "<tr><td>C YOLOv8-s, fine-tuned</td><td>0.906</td><td>0.932</td></tr><tr><td>E YOLO26-s, fine-tuned</td><td>0.919</td><td>0.931</td></tr>" +
      "<tr class=\"win\"><td>D YOLO26-s, fine-tuned twice</td><td>0.930</td><td>0.947</td></tr></table>",
    g: [["5", "models judged"], ["0.930", "winning validation score"], ["0", "peeks at the test set"]], w: "Choosing on validation keeps the final test an honest exam." },
  { t: "Final inspection", f: "Tested once on 1,041 objects whose boxes were checked by a person.",
    b: ["mAP@0.5 = 0.947, mAP@0.5:0.95 = 0.875, precision 0.931, recall 0.918.",
        "Single photos: 14 of 14 correct. Wrong Object IDs: only 16 of 1,041 objects.",
        "Known limits: tiny objects, look-alikes, co-occurring objects, degraded and rotated photos. Scroll for details."],
    more: "<table class=\"mt\"><tr><th>grid</th><th>objects found</th><th>fully correct</th></tr>" +
      "<tr><td>2×2</td><td>98.4%</td><td>87.5%</td></tr><tr><td>3×3</td><td>89.7%</td><td>35.7%</td></tr><tr><td>4×4</td><td>91.1%</td><td>14.3%</td></tr>" +
      "<tr><td>5×4</td><td>91.8%</td><td>21.4%</td></tr><tr><td>5×5</td><td>92.0%</td><td>7.1%</td></tr><tr class=\"win\"><td>all 72 grids</td><td>91.8%</td><td>34.7%</td></tr></table>" +
      "<p>Fully correct is strict: every object right and no extra box (95% CI 23.6 to 45.8%).</p>" +
      "<p>48 of 73 objects reach AP ≥ 0.95. Weakest: paddle 0.59, card wallet 0.62, mouth-guard case 0.77, sunglasses 0.79, all look-alikes or small.</p>" +
      "<p>Heavily degraded images (¼ resolution, dark, JPEG 20): mAP@0.5 drops to 0.428.</p>",
    g: [["0.947", "mAP@0.5"], ["14 / 14", "single photos"], ["16", "wrong IDs in 1,041"]], w: "A strong result, measured the honest way." },
  { t: "Open to visitors", f: "Explained every result in The EYE, the Explainable YOLO Explorer.",
    b: ["Random test grids, look-alike challenges and degraded images on demand, with every error explained.",
        "Stress tests over hundreds of grids, per-object inspection and snapshots for the report.",
        "A 20-page report, and the code on GitHub: M2_Boxes.ipynb (the boxes) and M2_Detect.ipynb (grids, training, testing), code only, no data."],
    g: [["10", "construction stations"], ["1", "detector you can question"], ["20", "report pages"]], w: "A model people can see into is a model people can trust." }
];

/* ---------- build ---------- */
const track = document.getElementById("jy-track"), map = document.getElementById("jy-map");
const truckSVG = `<svg class="truck" id="jy-truck" viewBox="0 0 64 38" aria-hidden="true"><rect x="2" y="8" width="38" height="20" rx="2" style="fill:var(--hivis);stroke:var(--truckline)" stroke-width="1.5"/>
  <path d="M40 14 H52 L60 22 V28 H40 Z" style="fill:var(--hivis);stroke:var(--truckline)" stroke-width="1.5"/><rect x="44" y="16" width="8" height="6" style="fill:var(--bp)"/>
  <path d="M6 12 L36 12" style="stroke:var(--truckline)" stroke-width="1" stroke-dasharray="3 3"/><circle cx="14" cy="30" r="5" style="fill:var(--truckline)"/><circle cx="50" cy="30" r="5" style="fill:var(--truckline)"/>
  <circle cx="14" cy="30" r="2" style="fill:var(--hivis)"/><circle cx="50" cy="30" r="2" style="fill:var(--hivis)"/></svg>`;
S.forEach((s, i) => {
  const stop = document.createElement("div"); stop.className = "stop"; stop.dataset.i = i;
  stop.innerHTML = `<div class="card" role="button" tabindex="0" aria-label="Station ${i + 1}: ${s.t}. Press to flip for the full report." aria-pressed="false">
    <div class="inner">
      <div class="face front"><div class="tape"></div><div class="num">${String(i + 1).padStart(2, "0")}</div>${scenes[i]}
        <h2>${s.t}</h2><p class="line">${s.f}</p>
        <div class="hint"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M4 12a8 8 0 0 1 14-5.3M20 12a8 8 0 0 1-14 5.3"/><path d="M18 3v4h-4M6 21v-4h4"/></svg>Flip for the site report</div></div>
      <div class="face back"><span class="stamp">Station ${i + 1} of 10</span>
        <div class="bhead"><span class="bnum">${String(i + 1).padStart(2, "0")}</span><h3>${s.t}</h3></div>
        <ul>${s.b.map(x => `<li>${x}</li>`).join("")}</ul>${s.more ? `<div class="more">${s.more}</div>` : ""}
        <div class="figs">${s.g.map(([v, l]) => `<div class="fig"><b>${v}</b><span>${l}</span></div>`).join("")}</div>
        <p class="why">${s.w}</p></div>
    </div></div>
    <div class="road"><div class="post">${i + 1}</div></div>`;
  track.appendChild(stop);
  const m = document.createElement("div"); m.className = "mstop"; m.dataset.i = i;
  m.innerHTML = `<button class="mdot" type="button" aria-label="Go to station ${i + 1}: ${s.t}">${i + 1}</button><span class="mname">${s.t}</span><span class="mline"></span>`;
  map.appendChild(m);
});
track.insertAdjacentHTML("beforeend", truckSVG);
const stops = [...track.querySelectorAll(".stop")], cards = [...track.querySelectorAll(".card")], truck = document.getElementById("jy-truck");
let cur = 0, progT = 0;
function setCurrent(i, scroll = true) {
  cur = Math.max(0, Math.min(stops.length - 1, i));
  map.querySelectorAll(".mstop").forEach((m, k) => { m.classList.toggle("on", k === cur); m.querySelector(".mdot").classList.toggle("on", k === cur); m.querySelector(".mdot").classList.toggle("done", k < cur); });
  document.getElementById("jy-count").textContent = `${cur + 1} / ${stops.length}`;
  const st = stops[cur], post = st.querySelector(".post"), road = st.querySelector(".road");
  const x = st.offsetLeft + post.offsetLeft - 70, y = road.offsetTop + 8;          // parks just before the post
  truck.style.top = y + "px"; truck.style.transform = `translateX(${x}px)`;
  if (scroll) { progT = Date.now(); } if (scroll) track.scrollTo({ left: st.offsetLeft - 2, behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
  { const m = map.querySelectorAll(".mstop")[cur];                          // scroll only the station map (never the page)
    map.scrollTo({ left: m.offsetLeft - map.clientWidth / 2 + m.offsetWidth / 2, behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" }); }
}
function flip(card) { const on = !card.classList.contains("flipped"); card.classList.toggle("flipped", on); card.setAttribute("aria-pressed", on); }
cards.forEach((c, i) => {
  c.addEventListener("click", e => { if (e.target.closest("a") || e.target.closest(".more") && e.target.closest(".more").scrollHeight > e.target.closest(".more").clientHeight && e.detail === 0) return;
    if (e.target.closest("a")) return; if (i !== cur) setCurrent(i); flip(c); });
  c.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); e.stopPropagation(); if (i !== cur) setCurrent(i); flip(c); } });
});
map.querySelectorAll(".mdot").forEach((d, i) => d.addEventListener("click", () => setCurrent(i)));
document.getElementById("jy-prev").onclick = () => setCurrent(cur - 1);
document.getElementById("jy-next").onclick = () => setCurrent(cur + 1);
document.getElementById("jy-flipall").onclick = () => { const anyFront = cards.some(c => !c.classList.contains("flipped")); cards.forEach(c => { c.classList.toggle("flipped", anyFront); c.setAttribute("aria-pressed", anyFront); });
  document.getElementById("jy-flipall").textContent = anyFront ? "Show fronts" : "Flip all"; };
let jyVisible = false;
const jeye = document.getElementById("jy-eye"), jiris = document.getElementById("jy-iris-g");
function landEye() {
  jeye.classList.remove("landed", "blink"); void jeye.offsetWidth; jeye.classList.add("landed");
  setTimeout(() => { jeye.classList.add("blink"); setTimeout(() => jeye.classList.remove("blink"), 900); }, 1250);
}
new IntersectionObserver(es => es.forEach(en => jyVisible = en.isIntersecting), { threshold: .25 }).observe(document.getElementById("jy"));
new IntersectionObserver(es => es.forEach(en => { if (en.isIntersecting) landEye(); else if (en.boundingClientRect.top > 0) jeye.classList.remove("landed"); }),
                         { threshold: .6 }).observe(document.querySelector("#jy header"));
addEventListener("mousemove", e => { if (!jyVisible || document.body.classList.contains("calm")) return; const r = jeye.getBoundingClientRect();
  const dx = e.clientX - (r.left + r.width / 2), dy = e.clientY - (r.top + r.height / 2), d = Math.hypot(dx, dy) || 1, k = Math.min(1, d / 500) * 10;
  jiris.style.transform = `translate(${(dx / d * k).toFixed(1)}px,${(dy / d * k * .6).toFixed(1)}px)`; });
document.addEventListener("keydown", e => {
  if (!jyVisible || ["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName) || document.querySelector(".ov.show")) return;
  if (e.key === "ArrowRight") { e.preventDefault(); setCurrent(cur + 1); }
  else if (e.key === "ArrowLeft") { e.preventDefault(); setCurrent(cur - 1); }
  else if ((e.key === "Enter" || e.key === " ") && document.activeElement && !document.activeElement.classList.contains("card") && document.activeElement.tagName !== "BUTTON") { e.preventDefault(); flip(cards[cur]); }
});
let st; track.addEventListener("scroll", () => { clearTimeout(st); st = setTimeout(() => {
  if (Date.now() - progT < 1100) return;                                   // scrolling started by the arrows, map or keys
  const l = track.scrollLeft, max = track.scrollWidth - track.clientWidth, c = stops[cur];
  const visible = c.offsetLeft >= l - 4 && c.offsetLeft + c.offsetWidth - 36 <= l + track.clientWidth + 4;
  if (l >= max - 4 && visible) return;                                     // end of the road: keep the chosen station
  let best = 0, bd = 1e9; stops.forEach((s, k) => { const d = Math.abs(s.offsetLeft - l); if (d < bd) { bd = d; best = k; } });
  if (best !== cur) setCurrent(best, false); }, 140); });
document.getElementById("jy-up").onclick = () => document.getElementById("main").scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
addEventListener("resize", () => setCurrent(cur, false));
document.fonts && document.fonts.ready.then(() => setCurrent(cur, false));
window.jyRelayout = () => setCurrent(cur, false);
setCurrent(0, false);

})();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
