"""
Milestone 2 detector demo (runs locally in your browser; nothing is uploaded anywhere).

Tabs
  Upload an image    any image -> boxes, Object ID / name / section / class AP, zoom gallery with reference
                     photos, optional degradation, downloads (annotated image, CSV, JSON), inference time
  Random test grid   a NEW grid of held-out TEST photos (single photo ... 5x5), optionally only one section or
                     only look-alike objects -> true (green) vs model (red), every error explained, per-section
                     scores, threshold sweep chart
  Stress test        many random grids at once -> totals, fully-correct rate with 95% CI, most-missed classes,
                     most common confusions
  About              what everything means

Shared controls: model (any trained run), confidence threshold, one box per 224-px cell.
Defaults are the settings chosen on validation (milestone2_outputs/detect/results.json).

Run:  pip install gradio     then     python m2_demo_app.py      (opens http://127.0.0.1:7860)
"""
import json, os, random, re, tempfile, time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

# ============================================================================================ settings
PROJECT = Path(os.environ.get("M2_PROJECT_DIR",
          r"C:\Users\csk23\Documents\Zanthosh\Northeastern\SEMESTERs\4th Sem\Deep Learning\PROJECT - 1"))
OUT, DATA = PROJECT / "milestone2_outputs", PROJECT / "Dataset - Updated"
RESULTS, MASTER = OUT / "detect" / "results.json", OUT / "boxes_master.csv"
PERCLASS, OBJLIST = OUT / "detect" / "per_class_ap_test.csv", PROJECT / "object_list.csv"
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
    models = {k: dict(weights=v["weights"], test_map50=v.get("test_map50"), test_map=v.get("test_map"))
              for k, v in r["models"].items()}
    return dict(best=best, models=models, mode=mode,
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

def to_canvas(img):
    W, H = img.size
    if W > IMGSZ or H > IMGSZ: return img, False
    c = Image.new("RGB", (IMGSZ, IMGSZ), (128, 128, 128)); c.paste(img, (0, 0)); return c, True

def run_model(model, img, mode, conf=CONF_FLOOR):
    """All predictions down to CONF_FLOOR (so missed objects can be explained), in input-image pixels, + ms."""
    W, H = img.size
    inp, padded = to_canvas(img) if mode == "native" else (img, False)
    t0 = time.perf_counter()
    r = model.predict(inp, imgsz=IMGSZ, conf=conf, max_det=100, verbose=False)[0]
    ms = (time.perf_counter() - t0) * 1000
    out = []
    for c, p, b in zip(r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.xyxy.cpu().numpy()):
        x1, y1, x2, y2 = map(float, b)
        if padded:
            if (x1 + x2) / 2 > W or (y1 + y2) / 2 > H: continue
            x2, y2 = min(x2, W), min(y2, H)
        out.append((int(c), float(p), (x1, y1, x2, y2)))
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
        if same: why = f"seen at confidence {max(same)[0]:.2f}, below the threshold {conf:.2f}"
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
def reference(oid, train):
    """A training photo of the class, cropped to its box: 'what this class looks like'."""
    if oid in _refs: return _refs[oid]
    t = train[train.object_id == oid].sort_values("fname")
    if not len(t): _refs[oid] = None; return None
    r = t.iloc[0]; im = load_224(r.path); m = 0.1 * max(r.x2 - r.x1, r.y2 - r.y1)
    c = im.crop((max(0, r.x1 - m), max(0, r.y1 - m), min(TILE, r.x2 + m), min(TILE, r.y2 + m)))
    _refs[oid] = ImageOps.contain(c, (150, 150)); return _refs[oid]

def gallery(img, preds, info, train, gts=None, rows=None, limit=30):
    """One tile per object: the object in this image | a training photo of the predicted class (| of the true class)."""
    items = []
    entries = rows if rows is not None else [dict(result="", reason="", _pred=i, _gt=None) for i in range(len(preds))]
    order = sorted(entries, key=lambda r: r["result"] == "correct")
    for r in order[:limit]:
        pi, gi = r.get("_pred"), r.get("_gt")
        box = gts[gi][1] if gi is not None else preds[pi][2]
        m = 0.25 * max(box[2] - box[0], box[3] - box[1], 20)
        x1 = min(max(0, box[0] - m), img.width - 2); y1 = min(max(0, box[1] - m), img.height - 2)     # clamp: never crash
        x2 = max(min(img.width, box[2] + m), x1 + 2); y2 = max(min(img.height, box[3] + m), y1 + 2)
        crop = ImageOps.contain(img.crop((x1, y1, x2, y2)), (150, 150))
        panels = [("in this image", crop)]
        if pi is not None: panels.append((f"model: {info.oid(preds[pi][0])}", reference(info.oid(preds[pi][0]), train)))
        if gi is not None and (pi is None or preds[pi][0] != gts[gi][0]):            # gts hold class indices
            panels.append((f"true: {info.oid(gts[gi][0])}", reference(info.oid(gts[gi][0]), train)))
        tile = Image.new("RGB", (160 * len(panels), 182), "white"); dd = ImageDraw.Draw(tile); f = font(12)
        for k, (cap, im) in enumerate(panels):
            if im is not None: tile.paste(im, (160 * k + (160 - im.width) // 2, 4 + (150 - im.height) // 2))
            dd.text((160 * k + 5, 162), cap, fill="black", font=f)
        title = (r["result"] + (f": {r['reason']}" if r["reason"] else "")) if r["result"] else \
                f"{info.label(preds[pi][0])} {preds[pi][1]:.2f} | {info.section(preds[pi][0])} | class AP {info.apv(preds[pi][0])}"
        items.append((tile, title))
    return items

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


# ============================================================================================ the app
def build_app():
    import gradio as gr
    from ultralytics import YOLO
    S = load_settings(); OBJ = load_objects(); AP = load_ap()
    test, train = load_photos(split="test"), load_photos(split="train", human_only=False)
    SECTIONS = sorted({v[1] for v in OBJ.values()}) or ["Unknown"]
    available = [m for m, v in S["models"].items() if Path(v["weights"]).exists()] or [S["best"]]
    MODELS = {}
    def get(name):
        if name not in MODELS: MODELS[name] = YOLO(S["models"][name]["weights"])
        m = MODELS[name]; return m, Info(m.names, OBJ, AP)

    def header(name):
        v = S["models"][name]
        note = "" if name == S["best"] else " (settings were tuned for the selected model)"
        return (f"**Model {name}**: test mAP@0.5 = {v['test_map50']} | inference: {S['mode']} | post-processing: {S['dup']} | "
                f"validated threshold {S['conf']}{note}")

    # ---------------------------------------------------------------- tab 1
    def detect_upload(img, model_name, conf, use_cell, deg):
        if img is None: raise gr.Error("Upload an image first.")
        model, info = get(model_name)
        img = degrade(img.convert("RGB"), deg); W, H = img.size
        raw, ms = run_model(model, img, S["mode"])
        preds = sorted(select(raw, conf, S["dup"], W, H, use_cell), key=lambda t: (int(t[2][1] // TILE), t[2][0]))
        drawn = draw(img, preds, info); tab = det_table(preds, info)
        kind = f"{W // TILE}x{H // TILE} grid of 224-px photos" if (W % TILE == 0 and H % TILE == 0) else "not a 224-px grid (one-box-per-cell not applied)"
        n = len(preds)
        txt = f"**{n} object{'s' if n != 1 else ''} found** | image {W} x {H} ({kind}) | inference {ms:.0f} ms" + (f" | degradation: {deg}" if deg != "None" else "")
        return drawn, txt, tab, gallery(img, preds, info, train), save_downloads(drawn, tab), dict(drawn=drawn, preds=preds, rows=None, gts=None)

    # ---------------------------------------------------------------- tab 2
    def new_grid(size, sections, mode, seed_text):
        cols, rows_n = GRIDS[size]
        seed = int(seed_text) if str(seed_text).strip().isdigit() else random.randint(0, 10 ** 6)
        try: img, gts_o = make_grid(test, cols, rows_n, sections or SECTIONS, mode, seed, OBJ)
        except ValueError as e: raise gr.Error(str(e))
        txt = (f"**New {size} grid ready** (seed **{seed}**, {mode.lower()}). Press **Detect** to run the model; "
               f"change the model, threshold or degradation and press Detect again to compare on this same grid.")
        return img, txt, dict(img=img, gts_o=gts_o, seed=seed, size=size), None, None, [], None, None

    def detect_grid(grid, deg, model_name, conf, use_cell):
        if not grid: raise gr.Error("Make a grid first.")
        model, info = get(model_name); seed = grid["seed"]
        img = degrade(grid["img"], deg); W, H = img.size
        gts = [(info.idx[o], b) for o, b in grid["gts_o"]]
        raw, ms = run_model(model, img, S["mode"])
        preds = select(raw, conf, S["dup"], W, H, use_cell)
        rows = explain(gts, preds, raw, conf, info); t = tally(rows, len(gts))
        verdict = "**Fully correct**" if t["fully"] else "Not fully correct"
        per_sec = Counter(); per_ok = Counter()
        for r in rows:
            if r["true"]: per_sec[r["section"]] += 1; per_ok[r["section"]] += r["result"] == "correct"
        sec_txt = " | ".join(f"{s_}: {per_ok[s_]} of {per_sec[s_]}" for s_ in sorted(per_sec))
        summary = (f"{verdict} (**{model_name}**): **{t['correct']} of {t['n']}** objects correct, **{t['named']} of {t['n']}** named correctly "
                   f"| box off: {t['box_off']} | wrong ID: {t['wrong']} | missed: {t['missed']} | extra: {t['extra']}  \n"
                   f"By section: {sec_txt}  \nSeed **{seed}** | threshold {conf:.2f} | inference {ms:.0f} ms"
                   + (f" | degradation: {deg}" if deg != "None" else "") +
                   "  \nGreen = true box (checked by a person). Model boxes: red = correct, orange = box off, purple = wrong ID, grey = extra. "
                   "Click a table row to highlight that object.")
        rows = sorted(rows, key=lambda r: ["missed", "wrong ID", "box off", "extra", "correct"].index(r["result"]))
        drawn = draw(img, preds, info, gts, rows)
        df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
        fig = sweep_plot(gts, raw, S["dup"], W, H, use_cell, conf, info)
        return (drawn, summary, df, gallery(img, preds, info, train, gts, rows), fig,
                save_downloads(drawn, df, f"grid_{seed}_{model_name}"), dict(drawn=drawn, preds=preds, rows=rows, gts=gts))

    def on_row(state, evt: gr.SelectData):
        if not state: return None
        k = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        n = len(state["rows"]) if state["rows"] is not None else len(state["preds"])
        if k is None or k >= n: return state["drawn"]
        return highlight(state["drawn"], row_box(state["rows"], k, state["preds"], state["gts"]))

    def clear_hl(state): return state["drawn"] if state else None

    # ---------------------------------------------------------------- tab 3
    def stress(size, n_grids, sections, mode, deg, model_name, conf, use_cell, progress=gr.Progress()):
        model, info = get(model_name); cols, rows_n = GRIDS[size]; n_grids = int(n_grids)
        tot, fully, ms_all, miss_by, conf_pairs, reasons = Counter(), [], [], Counter(), Counter(), Counter()
        for g in progress.tqdm(range(n_grids), desc="grids"):
            try: img, gts_o = make_grid(test, cols, rows_n, sections or SECTIONS, mode, 1000 + g, OBJ)
            except ValueError as e: raise gr.Error(str(e))
            img = degrade(img, deg); W, H = img.size; gts = [(info.idx[o], b) for o, b in gts_o]
            raw, ms = run_model(model, img, S["mode"]); ms_all.append(ms)
            rows = explain(gts, select(raw, conf, S["dup"], W, H, use_cell), raw, conf, info); t = tally(rows, len(gts))
            for k in ("n", "correct", "named", "box_off", "wrong", "missed", "extra"): tot[k] += t[k]
            fully.append(t["fully"])
            for r in rows:
                if r["result"] in ("missed", "box off", "wrong ID") and r["true"]: miss_by[f"{r['true']} {r['true_name']}"] += 1
                if r["result"] == "wrong ID": conf_pairs[f"{r['true']} -> {r['predicted']}"] += 1
                if r["result"] != "correct": reasons[re.sub(r"[\d.]+", "#", r["reason"].split(";")[0]) or r["result"]] += 1
        lo, hi = bootstrap_ci(fully); n = max(1, tot["n"])
        summary = (f"### {n_grids} random {size} grids ({mode.lower()}, sections: {', '.join(sections or SECTIONS)}, degradation: {deg}, model {model_name})\n"
                   f"- **Objects correct: {tot['correct']} of {tot['n']} ({100 * tot['correct'] / n:.1f}%)**; named correctly: {100 * tot['named'] / n:.1f}%\n"
                   f"- Box off: {tot['box_off']} | wrong ID: {tot['wrong']} | missed: {tot['missed']} | extra boxes: {tot['extra']}\n"
                   f"- **Fully correct grids: {100 * np.mean(fully):.1f}%** (95% CI {100 * lo:.1f} to {100 * hi:.1f}%)\n"
                   f"- Inference: {np.median(ms_all):.0f} ms per grid (median)")
        hard = pd.DataFrame(miss_by.most_common(10), columns=["object (missed, box off or wrong ID)", "times"])
        pairs = pd.DataFrame(conf_pairs.most_common(10), columns=["true -> predicted", "times"])
        why = pd.DataFrame(reasons.most_common(8), columns=["failure reason (numbers removed)", "times"])
        return summary, hard, pairs, why

    # ---------------------------------------------------------------- layout
    with gr.Blocks(title="Milestone 2 detector demo") as app:
        gr.Markdown("# Milestone 2: object detector demo")
        with gr.Row():
            model_dd = gr.Dropdown(available, value=S["best"], label="Model", scale=1)
            conf = gr.Slider(0.05, 0.9, value=S["conf"], step=0.05, label="Confidence threshold (validated value pre-set)", scale=2)
            cell = gr.Checkbox(value=True, label="One box per 224-px cell (grids only)", scale=1)
        head = gr.Markdown(header(S["best"])); model_dd.change(header, model_dd, head)

        with gr.Tab("Upload an image"):
            with gr.Row():
                with gr.Column(scale=1):
                    up = gr.Image(type="pil", label="Image (single photo, grid, or any image)")
                    deg1 = gr.Dropdown(DEGRADE, value="None", label="Degrade the image first (robustness test)")
                    b1 = gr.Button("Detect", variant="primary")
                with gr.Column(scale=2):
                    o1 = gr.Image(label="Detections"); t1 = gr.Markdown()
            d1 = gr.Dataframe(label="Object ID, name, section, class AP (test) and box of every object (pixels)")
            g1 = gr.Gallery(label="Each detected object | a training photo of the predicted class", columns=4, height="auto")
            f1 = gr.File(label="Downloads: annotated image, CSV, JSON", file_count="multiple")
            st1 = gr.State(None); c1 = gr.Button("Clear highlight", size="sm")
            b1.click(detect_upload, [up, model_dd, conf, cell, deg1], [o1, t1, d1, g1, f1, st1])
            d1.select(on_row, st1, o1); c1.click(clear_hl, st1, o1)

        with gr.Tab("Random test grid"):
            with gr.Row():
                with gr.Column(scale=1):
                    size = gr.Dropdown(list(GRIDS), value="3x3 (9 objects)", label="Grid size")
                    secs = gr.CheckboxGroup(SECTIONS, value=SECTIONS, label="Objects from section(s)")
                    mode = gr.Radio(["Random", "Look-alike challenge"], value="Random", label="Mix")
                    deg2 = gr.Dropdown(DEGRADE, value="None", label="Degradation")
                    seed = gr.Textbox(value="", label="Seed (empty = new grid; a number = repeatable grid)")
                    b2a = gr.Button("Make a new grid")
                    b2 = gr.Button("Detect", variant="primary")
                    c2 = gr.Button("Clear highlight", size="sm")
                with gr.Column(scale=2):
                    o2 = gr.Image(label="Grid (after Detect: green = true; red correct, orange box off, purple wrong ID, grey extra)"); t2 = gr.Markdown()
            d2 = gr.Dataframe(label="Every object, with the reason for every error")
            with gr.Row():
                p2 = gr.Plot(label="Threshold sweep for this grid")
                g2 = gr.Gallery(label="Errors first: object | model's class | true class (training photos)", columns=2, height="auto")
            f2 = gr.File(label="Downloads: annotated grid, CSV, JSON", file_count="multiple")
            grid_st, det_st = gr.State(None), gr.State(None)
            b2a.click(new_grid, [size, secs, mode, seed], [o2, t2, grid_st, d2, g2, f2, p2, det_st])
            b2.click(detect_grid, [grid_st, deg2, model_dd, conf, cell], [o2, t2, d2, g2, p2, f2, det_st])
            d2.select(on_row, det_st, o2); c2.click(clear_hl, det_st, o2)

        with gr.Tab("Stress test"):
            gr.Markdown("Runs many random grids of held-out test photos with the controls above, and reports the totals. "
                        "Grids are fixed by seed 1000, 1001, ... so a run can be repeated exactly.")
            with gr.Row():
                size3 = gr.Dropdown(list(GRIDS), value="5x5 (25 objects)", label="Grid size")
                n3 = gr.Slider(10, 300, value=100, step=10, label="Number of grids")
                secs3 = gr.CheckboxGroup(SECTIONS, value=SECTIONS, label="Section(s)")
                mode3 = gr.Radio(["Random", "Look-alike challenge"], value="Random", label="Mix")
                deg3 = gr.Dropdown(DEGRADE, value="None", label="Degradation")
            b3 = gr.Button("Run stress test", variant="primary")
            t3 = gr.Markdown()
            with gr.Row():
                h3 = gr.Dataframe(label="Most often missed / wrong objects"); c3 = gr.Dataframe(label="Most common confusions"); w3 = gr.Dataframe(label="Most common failure reasons")
            b3.click(stress, [size3, n3, secs3, mode3, deg3, model_dd, conf, cell], [t3, h3, c3, w3])

        with gr.Tab("About"):
            gr.Markdown(f"""
**What the app shows.** Object ID and bounding box of every object, from the model selected on validation ({S['best']}).
Random and stress-test grids use only **held-out test photos** whose true boxes were checked by a person.

**Results.** *correct*: right Object ID and the box overlaps the true box by at least half (IoU >= 0.5).
*box off*: right Object ID, but the box overlaps by less than half (counted as missed + extra in the strict metrics).
*wrong ID*: the box is in the right place but names another object (look-alikes are flagged).
*missed*: no box; the reason says whether the model saw it below the threshold, or the object is very small.
*extra*: a box that matches no object (a second box near an object, a partial box, or background).
*Fully correct*: every object correct and no extra box (the strict all-or-nothing measure in the report).

**Controls.** Confidence threshold: how sure the model must be to show a box (0.1 chosen on validation).
One box per cell: each 224-px cell holds one photo, so only the most confident box per cell is kept (grids only).
Degradation: blur, low resolution, darkness, heavy JPEG, or all of them (the report's low-quality test).
Look-alike challenge: grids made only from groups of similar objects (watches, wallets, rackets, ...).
Seed: the same number always gives the same grid. Class AP: test AP@0.5 of that class (final model).
**Make a new grid** shows the grid without detection; **Detect** runs the chosen model on that same grid, so you can
switch models, threshold or degradation and press Detect again to compare. Click any table row to highlight its object.
Inference time excludes the first call, which also loads the model onto the GPU.
""")
    return app


if __name__ == "__main__":
    build_app().launch(inbrowser=True)
