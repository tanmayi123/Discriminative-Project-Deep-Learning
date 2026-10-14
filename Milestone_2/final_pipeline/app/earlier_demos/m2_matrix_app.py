"""
M2 // DETECTION MATRIX: the Milestone 2 detector demo (runs locally; nothing is uploaded anywhere).

Twelve equal tiles float over a field of tiny pixels: model, threshold, grid size, mix (sections and look-alikes),
conditions (degradations), canvas, action, score, errors, inspect, seed and tools (upload, stress test, downloads,
calm mode, about). Detect opens the expanded canvas, where boxes fade in one by one; every error is explained,
clicking a row or a box highlights its object, and a threshold sweep shows the trade-off for the current grid.

Uses only the Python standard library for the web server (no Gradio or FastAPI needed), plus the packages the
project already uses (ultralytics, pandas, numpy, pillow, matplotlib).

Run:   python m2_matrix_app.py          then the browser opens http://127.0.0.1:7860
Stop:  Ctrl+C in its window
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



# ============================================================================================ server (Python standard library only)
import base64, io, threading, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("M2_PORT", "7860"))
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
        self.models = {}; self.lock = threading.Lock()
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

    @staticmethod
    def _keep(d, n=40):
        while len(d) > n: d.pop(next(iter(d)))

    # ------------------------------------------------------------------ endpoints
    def config(self, _=None):
        S = self.S
        return dict(best=S["best"], mode=S["mode"], dup=S["dup"], conf=S["conf"], models=self.available,
                    model_info={k: dict(test_map50=num(v["test_map50"]), test_map=num(v["test_map"])) for k, v in S["models"].items()},
                    sections=self.sections, grids=list(GRIDS), degrade=DEGRADE, groups=LOOKALIKE,
                    n_test_photos=int(len(self.test)), n_objects=int(self.test.object_id.nunique()))

    def grid(self, q):
        size = q.get("size") or "3x3 (9 objects)"
        if size not in GRIDS: raise ValueError("unknown grid size")
        cols, rows = GRIDS[size]; mix = q.get("mix") or "Random"
        seed = q.get("seed"); seed = int(seed) if str(seed or "").strip().isdigit() else random.randint(0, 10 ** 6)
        img, gts_o = make_grid(self.test, cols, rows, q.get("sections") or self.sections, mix, seed, self.OBJ)
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
        name = q.get("model") or self.S["best"]; conf = float(q.get("conf", self.S["conf"]))
        use_cell = bool(q.get("cell", True)); deg = q.get("degrade") or "None"
        if deg not in DEGRADE: raise ValueError("unknown degradation")
        with self.lock:
            model, info = self.model(name)
            img = degrade(g["img"], deg); W, H = img.size
            gts = [(info.idx[o], b) for o, b in g["gts_o"]] if g["gts_o"] else None
            if FAKE: model.truth = gts or []
            raw, ms = run_model(model, img, self.S["mode"])
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
        return dict(det=did, model=name, conf=conf, degrade=deg, ms=round(ms, 1), w=W, h=H, cols=g["cols"], rows=g["rows"],
                    image=b64img(img) if deg != "None" else None, boxes=boxes, truths=truths, table=out_rows,
                    tally=t, per_section=per, sweep=sw, has_truth=gts is not None)

    def tile(self, did, k):
        d = self.dets.get(did)
        if not d: raise ValueError("unknown detection")
        rows = d["rows"] if d["gts"] is not None else None
        items = gallery(d["img"], d["preds"], d["info"], self.train, d["gts"], rows, limit=10 ** 6) if rows is not None \
            else gallery(d["img"], d["preds"], d["info"], self.train)
        # gallery() puts errors first; map back to table order
        if rows is not None:
            order = sorted(range(len(rows)), key=lambda i: rows[i]["result"] == "correct")
            pos = {i: n for n, i in enumerate(order)}; tile_img, cap = items[pos[k]]
        else:
            tile_img, cap = items[k]
        return dict(image=b64img(tile_img, 92), caption=cap)

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

    def stress_start(self, q):
        jid = uuid.uuid4().hex[:8]; n = max(1, min(500, int(q.get("n", 100))))
        self.jobs[jid] = dict(done=0, total=n, result=None, error=None); self._keep(self.jobs, 10)
        threading.Thread(target=self._stress, args=(jid, q, n), daemon=True).start()
        return dict(job=jid, total=n)

    def _stress(self, jid, q, n):
        J = self.jobs[jid]
        try:
            size = q.get("size") or "5x5 (25 objects)"; cols, rows_n = GRIDS[size]
            name = q.get("model") or self.S["best"]; conf = float(q.get("conf", self.S["conf"])); use_cell = bool(q.get("cell", True))
            deg = q.get("degrade") or "None"; mix = q.get("mix") or "Random"; secs = q.get("sections") or self.sections
            tot, fully, ms_all, miss_by, pairs, reasons = Counter(), [], [], Counter(), Counter(), Counter()
            for g in range(n):
                img, gts_o = make_grid(self.test, cols, rows_n, secs, mix, 1000 + g, self.OBJ)
                with self.lock:
                    model, info = self.model(name)
                    img = degrade(img, deg); W, H = img.size; gts = [(info.idx[o], b) for o, b in gts_o]
                    if FAKE: model.truth = gts
                    raw, ms = run_model(model, img, self.S["mode"])
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
                if len(p) == 4 and p[:2] == ["api", "tile"]: return self._send(200, app.tile(p[2], int(p[3])))
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
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), make_handler(app))
    url = f"http://127.0.0.1:{PORT}"
    print(f"Milestone 2 Matrix app running at {url}  (model {app.S['best']}; press Ctrl+C to stop)")
    if os.environ.get("M2_NO_BROWSER") != "1":
        import webbrowser; threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try: srv.serve_forever()
    except KeyboardInterrupt: pass


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>M2 // Detection Matrix</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#04070a; --tile:rgba(8,16,13,.66); --tile2:rgba(10,22,17,.82);
  --line:rgba(57,255,136,.18); --line2:rgba(57,255,136,.5);
  --acc:#39ff88; --acc-d:#1fae5c; --cyan:#22e4ff; --amber:#ffb020; --mag:#ff4fd8; --grey:#8b97a8;
  --txt:#dcfff0; --dim:#7fa594; --dim2:#4d6b5e;
  --mono:"JetBrains Mono",ui-monospace,Consolas,Menlo,monospace; --sans:Inter,system-ui,"Segoe UI",Roboto,sans-serif;
  --t:min(calc((100vw - 140px) / 4), calc((100vh - 150px) / 3), 300px);
}
*{box-sizing:border-box}
html,body{margin:0;height:100%;background:var(--bg);color:var(--txt);font-family:var(--sans)}
body{overflow:auto}
#bg{position:fixed;inset:0;z-index:0;display:block}
.wrap{position:relative;z-index:1;min-height:100vh;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:22px 24px 26px}
.brand{width:calc(var(--t) * 4 + 54px);display:flex;align-items:baseline;justify-content:space-between;margin-bottom:14px;font-family:var(--mono)}
.brand h1{margin:0;font-size:15px;letter-spacing:.32em;font-weight:500;color:var(--acc);text-shadow:0 0 14px rgba(57,255,136,.45)}
.brand h1 b{color:var(--txt);font-weight:700}
.brand .meta{font-size:11px;color:var(--dim);letter-spacing:.08em}
.matrix{display:grid;grid-template-columns:repeat(4,var(--t));grid-auto-rows:var(--t);gap:18px}
.tile{position:relative;background:var(--tile);border:1px solid var(--line);border-radius:16px;padding:14px 15px 13px;
  display:flex;flex-direction:column;overflow:hidden;backdrop-filter:blur(7px);-webkit-backdrop-filter:blur(7px);
  box-shadow:0 18px 50px rgba(0,0,0,.55), inset 0 1px 0 rgba(255,255,255,.03);transition:border-color .25s, box-shadow .25s}
.tile::after{content:"";position:absolute;inset:0;border-radius:16px;pointer-events:none;
  background:linear-gradient(140deg,rgba(57,255,136,.07),transparent 38%,transparent 70%,rgba(34,228,255,.05))}
.tile:hover{border-color:var(--line2);box-shadow:0 18px 55px rgba(0,0,0,.6),0 0 26px rgba(57,255,136,.12)}
.tile.glitch{animation:glitch .36s steps(1,end) 1}
@keyframes glitch{
  0%{clip-path:inset(0 0 0 0);transform:translate(0)}
  12%{clip-path:inset(8% 0 62% 0);transform:translate(-4px,0);box-shadow:-3px 0 0 rgba(255,46,120,.65),3px 0 0 rgba(34,228,255,.65)}
  24%{clip-path:inset(58% 0 12% 0);transform:translate(4px,0)}
  36%{clip-path:inset(30% 0 44% 0);transform:translate(-2px,1px);box-shadow:2px 0 0 rgba(255,46,120,.6),-2px 0 0 rgba(34,228,255,.6)}
  48%{clip-path:inset(72% 0 4% 0);transform:translate(3px,0)}
  60%{clip-path:inset(0 0 0 0);transform:translate(-1px,0)}
  100%{clip-path:inset(0 0 0 0);transform:translate(0)}}
.th{display:flex;align-items:center;gap:9px;margin-bottom:10px}
.th .ix{font:500 10px var(--mono);color:var(--dim2);letter-spacing:.1em}
.th h3{margin:0;font:600 11.5px var(--mono);letter-spacing:.22em;text-transform:uppercase;color:var(--acc)}
.th .sp{flex:1}
.th .hint{font:10px var(--mono);color:var(--dim2)}
.tb{flex:1;display:flex;flex-direction:column;gap:9px;min-height:0}
.lbl{font:10.5px var(--mono);color:var(--dim);letter-spacing:.06em}
.big{font:700 30px var(--mono);color:var(--txt);line-height:1}
.chips{display:flex;flex-wrap:wrap;gap:6px}
.chip{border:1px solid var(--line);background:rgba(0,0,0,.25);color:var(--dim);border-radius:999px;padding:5px 10px;
  font:500 11px var(--mono);cursor:pointer;transition:all .15s;user-select:none}
.chip:hover{color:var(--txt);border-color:var(--line2)}
.chip.on{color:var(--acc);border-color:var(--acc);background:rgba(57,255,136,.12);box-shadow:0 0 12px rgba(57,255,136,.18)}
select,input[type=text],input[type=number]{width:100%;background:rgba(0,0,0,.35);color:var(--txt);border:1px solid var(--line);
  border-radius:10px;padding:8px 10px;font:500 12px var(--mono);outline:none}
select:focus,input:focus{border-color:var(--acc)}
input[type=range]{width:100%;accent-color:var(--acc)}
.row{display:flex;align-items:center;gap:8px}
.toggle{display:flex;align-items:center;gap:8px;font:11px var(--mono);color:var(--dim);cursor:pointer;user-select:none}
.toggle input{accent-color:var(--acc)}
.btn{border:1px solid var(--line2);background:rgba(0,0,0,.3);color:var(--txt);border-radius:12px;padding:11px 12px;
  font:600 12px var(--mono);letter-spacing:.16em;text-transform:uppercase;cursor:pointer;transition:all .15s;width:100%}
.btn:hover{border-color:var(--acc);box-shadow:0 0 16px rgba(57,255,136,.2)}
.btn.primary{background:linear-gradient(135deg,#39ff88,#16c47a);color:#02130a;border-color:transparent;box-shadow:0 0 22px rgba(57,255,136,.35)}
.btn.primary:hover{box-shadow:0 0 32px rgba(57,255,136,.55)}
.btn.sm{padding:7px 10px;font-size:10.5px;letter-spacing:.1em;border-radius:9px}
.btn:disabled{opacity:.45;cursor:default}
.preview{position:relative;flex:1;border-radius:11px;overflow:hidden;background:#010302;border:1px solid var(--line);cursor:zoom-in;min-height:0}
.preview img{position:absolute;inset:0;width:100%;height:100%;object-fit:contain}
.preview .empty{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;text-align:center;padding:16px;
  font:11px/1.6 var(--mono);color:var(--dim2)}
.preview .badge{position:absolute;left:8px;bottom:8px;font:10px var(--mono);color:var(--acc);background:rgba(0,0,0,.6);padding:3px 7px;border-radius:6px}
.score{display:flex;align-items:baseline;gap:6px}
.score .of{font:500 14px var(--mono);color:var(--dim)}
.bar{height:6px;border-radius:9px;background:rgba(255,255,255,.06);overflow:hidden}
.bar i{display:block;height:100%;width:0;background:linear-gradient(90deg,var(--acc-d),var(--acc));box-shadow:0 0 12px var(--acc);transition:width .9s cubic-bezier(.2,.8,.2,1)}
.stats{display:grid;grid-template-columns:repeat(2,1fr);gap:4px 10px;font:11px var(--mono);color:var(--dim)}
.stats b{color:var(--txt);font-weight:600}
.verdict{font:600 11px var(--mono);letter-spacing:.1em}
.list{flex:1;overflow:auto;display:flex;flex-direction:column;gap:5px;padding-right:3px;min-height:0}
.list::-webkit-scrollbar{width:5px}.list::-webkit-scrollbar-thumb{background:var(--line);border-radius:5px}
.item{display:flex;gap:8px;align-items:flex-start;padding:6px 8px;border-radius:9px;border:1px solid transparent;cursor:pointer;
  background:rgba(0,0,0,.2);font:11px/1.35 var(--mono)}
.item:hover{border-color:var(--line2)}
.item.sel{border-color:var(--acc);background:rgba(57,255,136,.08)}
.dot{flex:0 0 8px;height:8px;border-radius:2px;margin-top:3px}
.r-correct{--c:var(--acc)} .r-box-off{--c:var(--amber)} .r-wrong-ID{--c:var(--mag)} .r-extra{--c:var(--grey)} .r-missed{--c:var(--cyan)} .r-detected{--c:var(--acc)}
.dot{background:var(--c)}
.item .t{color:var(--txt)} .item .w{color:var(--dim)}
.insp{flex:1;display:flex;flex-direction:column;gap:6px;min-height:0}
.insp img{width:100%;flex:1;min-height:0;object-fit:contain;border-radius:9px;background:#fff}
.insp .cap{font:10.5px/1.45 var(--mono);color:var(--dim)}
.tools{display:grid;grid-template-columns:1fr 1fr;gap:7px}
/* ---------------------------------------------------------------- overlays */
.ov{position:fixed;inset:0;z-index:30;background:rgba(1,3,4,.74);backdrop-filter:blur(5px);display:none;align-items:center;justify-content:center}
.ov.show{display:flex;animation:fade .2s ease-out}
@keyframes fade{from{opacity:0}to{opacity:1}}
.panel{background:var(--tile2);border:1px solid var(--line2);border-radius:18px;box-shadow:0 30px 90px rgba(0,0,0,.7),0 0 40px rgba(57,255,136,.08);
  animation:pop .28s cubic-bezier(.2,.9,.25,1.2)}
@keyframes pop{from{transform:scale(.94);opacity:0}to{transform:scale(1);opacity:1}}
.stage{width:min(1480px,96vw);height:min(920px,93vh);display:grid;grid-template-columns:1fr 390px;gap:16px;padding:16px}
.stagebar{grid-column:1 / -1;display:flex;align-items:center;gap:12px;font:11px var(--mono);color:var(--dim)}
.stagebar h2{margin:0;font:600 12px var(--mono);letter-spacing:.24em;color:var(--acc)}
.stagebar .sp{flex:1}
.stage{grid-template-rows:auto 1fr}
.figbox{position:relative;display:flex;align-items:center;justify-content:center;min-height:0;background:#010302;border-radius:12px;border:1px solid var(--line);overflow:hidden}
.fig{position:relative}
.fig img{display:block;width:100%;height:100%}
.fig svg{position:absolute;inset:0;width:100%;height:100%;overflow:visible}
.scan{position:absolute;left:-2%;right:-2%;height:3px;top:0;background:linear-gradient(90deg,transparent,var(--acc),transparent);
  box-shadow:0 0 22px 4px rgba(57,255,136,.55);animation:scan .75s cubic-bezier(.5,0,.5,1) forwards;pointer-events:none}
@keyframes scan{from{top:0;opacity:1}to{top:100%;opacity:.2}}
.cover{position:absolute;background:#010302;animation:uncover .4s ease-out forwards;pointer-events:none}
@keyframes uncover{from{opacity:1}to{opacity:0}}
.side{display:flex;flex-direction:column;gap:12px;min-height:0}
.legend{display:flex;flex-wrap:wrap;gap:6px 12px;font:10.5px var(--mono);color:var(--dim)}
.legend span{display:flex;align-items:center;gap:6px}
.legend i{width:16px;height:0;border-top:2px solid var(--c)}
.legend i.dash{border-top-style:dashed}.legend i.dots{border-top-style:dotted}
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
.b-correct{--c:var(--acc)}.b-detected{--c:var(--acc)}.b-box-off{--c:var(--amber)}.b-wrong-ID{--c:var(--mag)}.b-extra{--c:var(--grey)}.b-missed{--c:var(--cyan)}
.box{cursor:pointer}
.hl{fill:rgba(0,0,0,.62);fill-rule:evenodd;pointer-events:none}
.hlf{fill:none;stroke:#ffd84a;stroke-width:4;vector-effect:non-scaling-stroke;filter:drop-shadow(0 0 8px rgba(255,216,74,.8))}
.ministats{font:11px/1.6 var(--mono);color:var(--dim)}
.ministats b{color:var(--txt)}
.sweep{width:100%;height:120px}
.dl{display:flex;gap:7px}
.dl a{flex:1;text-align:center;text-decoration:none}
.close{background:none;border:1px solid var(--line);color:var(--dim);border-radius:9px;padding:6px 10px;font:11px var(--mono);cursor:pointer}
.close:hover{color:var(--txt);border-color:var(--acc)}
.tabs{display:flex;gap:6px}
.small{width:min(760px,94vw);max-height:90vh;overflow:auto;padding:22px 24px}
.small h2{margin:0 0 12px;font:600 13px var(--mono);letter-spacing:.24em;color:var(--acc)}
.small p,.small li{font-size:13px;line-height:1.6;color:#bfe9d6}
.small table{width:100%;border-collapse:collapse;font:11.5px var(--mono);margin-top:6px}
.small td,.small th{border-bottom:1px solid var(--line);padding:6px 4px;text-align:left}
.small th{color:var(--acc);font-weight:500}
.prog{height:8px;border-radius:9px;background:rgba(255,255,255,.07);overflow:hidden;margin:12px 0}
.prog i{display:block;height:100%;width:0;background:linear-gradient(90deg,var(--acc-d),var(--acc));transition:width .3s}
.kpis{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:14px 0}
.kpi{border:1px solid var(--line);border-radius:12px;padding:12px;background:rgba(0,0,0,.25)}
.kpi .v{font:700 24px var(--mono)}.kpi .k{font:10.5px var(--mono);color:var(--dim);margin-top:4px}
#tip{position:fixed;z-index:60;pointer-events:none;background:rgba(3,10,7,.95);border:1px solid var(--line2);border-radius:10px;padding:8px 10px;
  font:11px/1.5 var(--mono);color:var(--txt);max-width:300px;display:none;box-shadow:0 10px 30px rgba(0,0,0,.6)}
#toast{position:fixed;left:50%;bottom:26px;transform:translateX(-50%);z-index:70;background:rgba(40,5,15,.95);border:1px solid #ff4f7a;color:#ffd9e2;
  padding:10px 16px;border-radius:10px;font:12px var(--mono);display:none}
.shock{position:fixed;z-index:50;width:20px;height:20px;margin:-10px 0 0 -10px;border-radius:50%;pointer-events:none;
  border:2px solid var(--acc);box-shadow:0 0 30px rgba(57,255,136,.8), inset 0 0 20px rgba(57,255,136,.5);animation:shock .7s ease-out forwards}
.shock.strong{border-width:3px;animation-duration:.9s}
@keyframes shock{from{transform:scale(.2);opacity:1}to{transform:scale(var(--s,22));opacity:0}}
.flash{position:fixed;inset:0;z-index:49;pointer-events:none;background:radial-gradient(circle at var(--x) var(--y),rgba(57,255,136,.18),transparent 45%);animation:fade-out .5s forwards}
@keyframes fade-out{to{opacity:0}}
.busy{position:relative}.busy::before{content:"";position:absolute;inset:0;border-radius:12px;background:linear-gradient(90deg,transparent,rgba(255,255,255,.25),transparent);animation:sweep 1s linear infinite}
@keyframes sweep{from{transform:translateX(-100%)}to{transform:translateX(100%)}}
body.calm .tile.glitch,body.calm .det,body.calm .det rect.f,body.calm .scan,body.calm .cover,body.calm .ov.show,body.calm .panel{animation:none!important}
body.calm .det{opacity:1}
body.calm *,body.calm *::before,body.calm *::after{transition:none!important}
body.calm .scan,body.calm .cover,body.calm .shock,body.calm .flash{display:none}
@media (max-width:900px){:root{--t:calc((100vw - 66px) / 2)} .matrix{grid-template-columns:repeat(2,var(--t))} .brand{width:calc(var(--t) * 2 + 18px)}
  .stage{grid-template-columns:1fr;grid-template-rows:auto 60vh auto;height:auto;max-height:94vh;overflow:auto}}
</style>
</head>
<body>
<canvas id="bg"></canvas>
<div class="wrap">
  <div class="brand"><h1>M2 <b>//</b> DETECTION MATRIX</h1><span class="meta" id="meta">loading...</span></div>
  <div class="matrix" id="matrix">
    <section class="tile" id="t-model"><div class="th"><span class="ix">01</span><h3>Model</h3></div>
      <div class="tb"><select id="model"></select>
        <div><div class="lbl">TEST mAP@0.5</div><div class="big" id="mapv">-</div></div>
        <div class="lbl" id="modelnote"></div></div></section>
    <section class="tile" id="t-thr"><div class="th"><span class="ix">02</span><h3>Threshold</h3><span class="sp"></span><span class="hint" id="thrv">0.10</span></div>
      <div class="tb"><input type="range" id="conf" min="0.05" max="0.9" step="0.05">
        <div class="lbl">how sure the model must be to show a box</div>
        <label class="toggle"><input type="checkbox" id="cell" checked> one box per 224-px cell</label>
        <button class="btn sm" id="resetthr">reset to validated</button></div></section>
    <section class="tile" id="t-grid"><div class="th"><span class="ix">03</span><h3>Grid</h3></div>
      <div class="tb"><div class="chips" id="sizes"></div><div class="lbl" id="sizeinfo"></div></div></section>
    <section class="tile" id="t-mix"><div class="th"><span class="ix">04</span><h3>Mix</h3></div>
      <div class="tb"><div class="lbl">SECTIONS</div><div class="chips" id="secs"></div>
        <div class="lbl">OBJECTS</div><div class="chips" id="mixes"></div></div></section>
    <section class="tile" id="t-cond"><div class="th"><span class="ix">05</span><h3>Conditions</h3></div>
      <div class="tb"><div class="chips" id="degs"></div><div class="lbl">applied to the image before detection (robustness test)</div></div></section>
    <section class="tile" id="t-canvas"><div class="th"><span class="ix">06</span><h3>Canvas</h3><span class="sp"></span><span class="hint">click to expand</span></div>
      <div class="tb"><div class="preview" id="preview"><div class="empty">no image yet<br>make a grid or upload</div></div></div></section>
    <section class="tile" id="t-act"><div class="th"><span class="ix">07</span><h3>Action</h3></div>
      <div class="tb" style="justify-content:center;gap:12px">
        <button class="btn" id="newgrid">New grid</button>
        <button class="btn primary" id="detect">Detect</button>
        <div class="lbl" id="status" style="text-align:center">ready</div></div></section>
    <section class="tile" id="t-score"><div class="th"><span class="ix">08</span><h3>Score</h3><span class="sp"></span><span class="hint" id="ms"></span></div>
      <div class="tb"><div class="score"><span class="big" id="sc">-</span><span class="of" id="scof"></span></div>
        <div class="bar"><i id="scbar"></i></div><div class="verdict" id="verdict"></div>
        <div class="stats" id="stats"></div></div></section>
    <section class="tile" id="t-err"><div class="th"><span class="ix">09</span><h3>Errors</h3><span class="sp"></span><span class="hint" id="errn"></span></div>
      <div class="tb"><div class="list" id="errs"><div class="lbl">errors and their reasons appear here</div></div></div></section>
    <section class="tile" id="t-insp"><div class="th"><span class="ix">10</span><h3>Inspect</h3></div>
      <div class="tb"><div class="insp" id="insp"><div class="lbl">select an object: its crop next to a training photo of the model's class (and the true class)</div></div></div></section>
    <section class="tile" id="t-seed"><div class="th"><span class="ix">11</span><h3>Seed</h3></div>
      <div class="tb"><input type="text" id="seed" placeholder="empty = random">
        <div class="lbl">the same seed always rebuilds the same grid</div>
        <div class="row"><button class="btn sm" id="dice">random</button><button class="btn sm" id="keep">keep last</button></div>
        <div class="lbl" id="lastseed"></div></div></section>
    <section class="tile" id="t-tools"><div class="th"><span class="ix">12</span><h3>Tools</h3></div>
      <div class="tb"><div class="tools">
        <button class="btn sm" id="upbtn">Upload</button><button class="btn sm" id="stressbtn">Stress test</button>
        <button class="btn sm" id="dlbtn">Downloads</button><button class="btn sm" id="aboutbtn">About</button></div>
        <label class="toggle"><input type="checkbox" id="calm"> calm mode (no motion)</label>
        <input type="file" id="file" accept="image/*" hidden></div></section>
  </div>
</div>

<!-- expanded canvas -->
<div class="ov" id="ov-stage"><div class="panel stage">
  <div class="stagebar"><h2 id="stitle">CANVAS</h2><span id="ssub"></span><span class="sp"></span>
    <button class="btn sm" id="s-new" style="width:auto">New grid</button><button class="btn sm primary" id="s-det" style="width:auto">Detect</button>
    <button class="close" id="s-close">close [esc]</button></div>
  <div class="figbox" id="figbox"><div class="fig" id="fig"></div></div>
  <div class="side">
    <div class="legend">
      <span class="b-correct"><i></i>correct</span><span class="b-box-off"><i class="dash"></i>box off</span>
      <span class="b-wrong-ID"><i></i>wrong ID</span><span class="b-extra"><i class="dots"></i>extra</span>
      <span class="b-missed"><i class="dash"></i>missed</span><span style="--c:rgba(255,255,255,.6)"><i class="dash"></i>true box</span></div>
    <div class="ministats" id="sstats"></div>
    <div class="tabs"><span class="chip on" data-f="all">all</span><span class="chip" data-f="err">errors</span></div>
    <div class="list" id="srows"></div>
    <svg class="sweep" id="sweep" viewBox="0 0 360 120" preserveAspectRatio="none"></svg>
    <div class="dl" id="sdl"></div>
  </div></div></div>

<div class="ov" id="ov-stress"><div class="panel small">
  <h2>STRESS TEST</h2>
  <p>Runs many random grids of held-out test photos with the current settings (grid, mix, conditions, model, threshold) and reports the totals. Grids use seeds 1000, 1001, ... so a run can be repeated exactly.</p>
  <div class="row"><span class="lbl" style="white-space:nowrap">GRIDS</span><input type="range" id="sn" min="10" max="300" step="10" value="100"><span class="hint" id="snv">100</span></div>
  <div class="row" style="margin-top:10px"><button class="btn primary" id="srun">Run</button><button class="close" id="sclose">close</button></div>
  <div class="prog"><i id="sprog"></i></div><div id="sres"></div></div></div>

<div class="ov" id="ov-dl"><div class="panel small"><h2>DOWNLOADS</h2><p>The last detection, in the milestone's output format.</p><div class="dl" id="dls"></div>
  <div class="row" style="margin-top:14px"><button class="close" id="dlclose">close</button></div></div></div>

<div class="ov" id="ov-about"><div class="panel small"><h2>ABOUT</h2>
  <p>The detector reports the <b>Object ID and bounding box</b> of every object. Random grids and the stress test use only <b>held-out test photos</b> whose true boxes were checked by a person.</p>
  <table><tr><th>result</th><th>meaning</th></tr>
  <tr><td style="color:var(--acc)">correct</td><td>right Object ID and the box overlaps the true box by at least half (IoU >= 0.5)</td></tr>
  <tr><td style="color:var(--amber)">box off</td><td>right Object ID, but the box overlaps by less than half</td></tr>
  <tr><td style="color:var(--mag)">wrong ID</td><td>box in the right place, another object named (look-alikes flagged)</td></tr>
  <tr><td style="color:var(--cyan)">missed</td><td>no box; the reason says whether it was seen below the threshold or is very small</td></tr>
  <tr><td style="color:var(--grey)">extra</td><td>a box matching no object</td></tr></table>
  <p><b>Fully correct</b>: every object correct and no extra box. <b>Threshold</b>: how sure the model must be (0.1 chosen on validation). <b>One box per cell</b>: each 224-px cell holds one photo. <b>Seed</b>: the same number rebuilds the same grid. <b>Calm mode</b> turns off all motion.</p>
  <div class="row"><button class="close" id="aclose">close</button></div></div></div>

<div id="tip"></div><div id="toast"></div>

<script>
"use strict";
const $ = (s, r = document) => r.querySelector(s), $$ = (s, r = document) => [...r.querySelectorAll(s)];
const st = { cfg: null, model: null, conf: 0.1, cell: true, size: "3x3 (9 objects)", sections: [], mix: "Random", degrade: "None",
             grid: null, det: null, sel: null, filter: "all", lastSeed: null, animatePending: false };
const calm = () => document.body.classList.contains("calm");
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const cls = r => String(r || "detected").replace(/ /g, "-");

async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const j = await r.json(); if (!r.ok || j.error) throw new Error(j.error || r.statusText); return j;
}
function toast(m) { const t = $("#toast"); t.textContent = m; t.style.display = "block"; clearTimeout(t._h); t._h = setTimeout(() => t.style.display = "none", 4200); }

/* ---------------------------------------------------------------- pixel field background */
const BG = (() => {
  const cv = $("#bg"), cx = cv.getContext("2d"); let W, H, px = [], waves = [], mouse = { x: -999, y: -999 }, running = true, t0 = performance.now();
  function init() {
    const dpr = Math.min(2, window.devicePixelRatio || 1); W = innerWidth; H = innerHeight; cv.width = W * dpr; cv.height = H * dpr;
    cv.style.width = W + "px"; cv.style.height = H + "px"; cx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const n = Math.round(W * H / 700); px = [];
    for (let i = 0; i < n; i++) px.push({ x: Math.random() * W, y: Math.random() * H, s: Math.random() < .86 ? 1 : 2,
      a: .05 + Math.random() * .28, p: Math.random() * 6.28, f: .4 + Math.random() * 1.6, v: .02 + Math.random() * .08,
      c: Math.random() < .9 ? "57,255,136" : (Math.random() < .5 ? "34,228,255" : "220,255,240") });
    draw(performance.now());
  }
  function draw(now) {
    const t = (now - t0) / 1000; cx.fillStyle = "#04070a"; cx.fillRect(0, 0, W, H);
    waves = waves.filter(w => t - w.t < 1.6);
    for (const q of px) {
      if (!calm()) { q.y += q.v; if (q.y > H) { q.y = 0; q.x = Math.random() * W; } }
      let x = q.x, y = q.y, a = q.a * (calm() ? 1 : .65 + .35 * Math.sin(t * q.f + q.p));
      const dm = Math.hypot(x - mouse.x, y - mouse.y); if (dm < 110) a += (1 - dm / 110) * .55;
      for (const w of waves) {
        const r = (t - w.t) * w.v, d = Math.hypot(x - w.x, y - w.y), band = Math.abs(d - r);
        if (band < 48) { const k = (1 - band / 48) * (1 - (t - w.t) / 1.6) * w.amp; x += (x - w.x) / (d || 1) * k * 14; y += (y - w.y) / (d || 1) * k * 14; a += k * .7; }
      }
      cx.fillStyle = `rgba(${q.c},${Math.min(1, a).toFixed(3)})`; cx.fillRect(x, y, q.s, q.s);
    }
    if (running && !calm()) requestAnimationFrame(draw);
  }
  addEventListener("resize", init); addEventListener("mousemove", e => { mouse.x = e.clientX; mouse.y = e.clientY; });
  document.addEventListener("visibilitychange", () => { running = !document.hidden; if (running) requestAnimationFrame(draw); });
  return { init, wave(x, y, amp = 1) { waves.push({ x, y, t: (performance.now() - t0) / 1000, v: 900, amp }); if (calm()) draw(performance.now()); },
           restart() { if (!calm()) requestAnimationFrame(draw); else draw(performance.now()); } };
})();

/* ---------------------------------------------------------------- glitch on hover, shockwave on press */
$$(".tile").forEach(t => t.addEventListener("mouseenter", () => {
  if (calm() || t.classList.contains("glitch")) return; t.classList.add("glitch"); setTimeout(() => t.classList.remove("glitch"), 380);
}));
function shock(ev, strong = false) {
  if (calm()) return; const r = ev.currentTarget.getBoundingClientRect(), x = r.left + r.width / 2, y = r.top + r.height / 2;
  const s = document.createElement("div"); s.className = "shock" + (strong ? " strong" : ""); s.style.left = x + "px"; s.style.top = y + "px";
  s.style.setProperty("--s", strong ? 34 : 22); document.body.appendChild(s); setTimeout(() => s.remove(), 1000);
  const f = document.createElement("div"); f.className = "flash"; f.style.setProperty("--x", x + "px"); f.style.setProperty("--y", y + "px");
  document.body.appendChild(f); setTimeout(() => f.remove(), 600);
  BG.wave(x, y, strong ? 1.3 : .8);
  $$(".tile").forEach(t => {
    const b = t.getBoundingClientRect(), dx = b.left + b.width / 2 - x, dy = b.top + b.height / 2 - y, d = Math.hypot(dx, dy) || 1;
    const k = Math.max(0, 1 - d / 900) * (strong ? 14 : 9), delay = d / 2.2;
    t.animate([{ transform: "translate(0,0)" }, { transform: `translate(${dx / d * k}px,${dy / d * k}px)` }, { transform: "translate(0,0)" }],
              { duration: 520, delay, easing: "cubic-bezier(.2,.9,.3,1)" });
  });
}

/* ---------------------------------------------------------------- controls */
function chips(el, items, isOn, onClick) {
  el.innerHTML = ""; items.forEach(v => { const c = document.createElement("span"); c.className = "chip" + (isOn(v) ? " on" : "");
    c.textContent = v.label || v; c.onclick = () => { onClick(v); }; el.appendChild(c); });
}
function renderControls() {
  const c = st.cfg;
  $("#model").innerHTML = c.models.map(m => `<option ${m === st.model ? "selected" : ""}>${esc(m)}</option>`).join("");
  const mi = c.model_info[st.model] || {}; $("#mapv").textContent = mi.test_map50 != null ? mi.test_map50.toFixed(3) : "-";
  $("#modelnote").textContent = st.model === c.best ? "selected on validation" : "settings tuned for " + c.best;
  $("#conf").value = st.conf; $("#thrv").textContent = (+st.conf).toFixed(2); $("#cell").checked = st.cell;
  chips($("#sizes"), c.grids.map(g => ({ label: g.split(" ")[0] === "Single" ? "1x1" : g.split(" ")[0], v: g })), x => x.v === st.size, x => { st.size = x.v; renderControls(); });
  $("#sizeinfo").textContent = st.size;
  chips($("#secs"), c.sections, s => st.sections.includes(s), s => { st.sections = st.sections.includes(s) ? st.sections.filter(x => x !== s) : [...st.sections, s]; if (!st.sections.length) st.sections = [s]; renderControls(); });
  chips($("#mixes"), [{ label: "random", v: "Random" }, { label: "look-alikes", v: "Look-alike challenge" }], x => x.v === st.mix, x => { st.mix = x.v; renderControls(); });
  chips($("#degs"), c.degrade.map(d => ({ label: d.startsWith("All") ? "all" : d.toLowerCase(), v: d })), x => x.v === st.degrade, x => { st.degrade = x.v; renderControls(); });
}
$("#model").onchange = e => { st.model = e.target.value; renderControls(); };
$("#conf").oninput = e => { st.conf = +e.target.value; $("#thrv").textContent = st.conf.toFixed(2); };
$("#cell").onchange = e => st.cell = e.target.checked;
$("#resetthr").onclick = e => { shock(e); st.conf = st.cfg.conf; renderControls(); };
$("#dice").onclick = e => { shock(e); $("#seed").value = ""; };
$("#keep").onclick = e => { shock(e); if (st.lastSeed != null) $("#seed").value = st.lastSeed; };
$("#calm").onchange = e => { document.body.classList.toggle("calm", e.target.checked); BG.restart(); };
if (matchMedia("(prefers-reduced-motion: reduce)").matches) { $("#calm").checked = true; document.body.classList.add("calm"); }

/* ---------------------------------------------------------------- canvas rendering (preview + stage) */
function svgFor(det, w, h, animate) {
  if (!det) return "";
  const step = det.boxes.length > 30 ? .025 : .045, base = animate ? .78 : 0; let k = 0, out = [];
  if (det.has_truth) out.push(...det.truths.map(t => `<g class="truth"><rect x="${t.x1}" y="${t.y1}" width="${t.x2 - t.x1}" height="${t.y2 - t.y1}"/></g>`));
  const fs = Math.max(11, w / 70), lab = (b, txt) => { const tw = txt.length * fs * .62 + 8;
    return `<g class="lab"><rect x="${b.x1}" y="${Math.max(0, b.y1 - fs - 4)}" width="${tw}" height="${fs + 4}" rx="2"/><text x="${b.x1 + 4}" y="${Math.max(fs, b.y1 - 4)}" style="font-size:${fs}px">${esc(txt)}</text></g>`; };
  for (const b of det.boxes) {
    const r = cls(b.result), err = ["box-off", "wrong-ID", "extra"].includes(r) ? " err" : "", d = animate ? (base + step * k++).toFixed(3) + "s" : "0s";
    out.push(`<g class="box det b-${r}${err}" data-b="${b.k}" style="--d:${d}"><rect class="f" x="${b.x1}" y="${b.y1}" width="${b.x2 - b.x1}" height="${b.y2 - b.y1}"/>${lab(b, b.id + " " + b.conf.toFixed(2))}</g>`);
  }
  if (det.has_truth) det.table.forEach((r, i) => { if (r.result !== "missed") return; const t = det.truths[r.gt], d = animate ? (base + step * k++).toFixed(3) + "s" : "0s";
    out.push(`<g class="box det missed b-missed" data-row="${i}" style="--d:${d}"><rect class="f" x="${t.x1}" y="${t.y1}" width="${t.x2 - t.x1}" height="${t.y2 - t.y1}"/>${lab(t, t.id + " missed")}</g>`); });
  return `<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">${out.join("")}<g id="hlg"></g></svg>`;
}
function imgSrc() { return "data:image/jpeg;base64," + ((st.det && st.det.image) || st.grid.image); }
function renderPreview(cascade) {
  const p = $("#preview"); if (!st.grid) return;
  const g = st.grid, det = st.det;
  p.innerHTML = `<div class="fig" id="pfig" style="position:absolute;inset:0;margin:auto"><img src="${imgSrc()}">${svgFor(det, g.w, g.h, false)}</div>` +
    `<span class="badge">${g.kind === "grid" ? g.cols + "x" + g.rows + " · seed " + g.seed : "upload · " + g.w + "x" + g.h}${det ? " · " + det.model : ""}</span>`;
  fit($("#pfig"), p, g.w, g.h);
  if (cascade && g.kind === "grid" && !calm()) coverCells($("#pfig"), g, .03);
}
function fit(fig, box, w, h) {
  const bw = box.clientWidth, bh = box.clientHeight, s = Math.min(bw / w, bh / h);
  fig.style.width = w * s + "px"; fig.style.height = h * s + "px";
}
function coverCells(fig, g, step) {
  const cw = 100 / g.cols, ch = 100 / g.rows;
  for (let r = 0; r < g.rows; r++) for (let c = 0; c < g.cols; c++) {
    const d = document.createElement("div"); d.className = "cover";
    Object.assign(d.style, { left: c * cw + "%", top: r * ch + "%", width: cw + .2 + "%", height: ch + .2 + "%", animationDelay: ((r + c) * step) + "s" });
    fig.appendChild(d); setTimeout(() => d.remove(), 1500);
  }
}
function openStage(animate) {
  if (!st.grid) return toast("Make a grid or upload an image first.");
  $("#ov-stage").classList.add("show"); renderStage(animate);
}
function renderStage(animate) {
  const g = st.grid, det = st.det, fig = $("#fig"), box = $("#figbox");
  $("#stitle").textContent = g.kind === "grid" ? `GRID ${g.cols}x${g.rows}` : "UPLOAD";
  $("#ssub").textContent = g.kind === "grid" ? `seed ${g.seed} · ${g.mix.toLowerCase()}` + (det ? ` · ${det.model} · threshold ${det.conf.toFixed(2)}${det.degrade !== "None" ? " · " + det.degrade.toLowerCase() : ""}` : " · not detected yet")
                                             : `${g.w} x ${g.h}` + (det ? ` · ${det.model}` : "");
  fig.innerHTML = `<img src="${imgSrc()}">${svgFor(det, g.w, g.h, animate && !calm())}`;
  requestAnimationFrame(() => fit(fig, box, g.w, g.h));
  if (animate && det && !calm()) { const s = document.createElement("div"); s.className = "scan"; fig.appendChild(s); setTimeout(() => s.remove(), 900); }
  if (animate === "cascade" && g.kind === "grid" && !calm()) coverCells(fig, g, .035);
  $$("#fig .box").forEach(el => {
    el.onmouseenter = e => showTip(e, el); el.onmousemove = e => moveTip(e); el.onmouseleave = hideTip;
    el.onclick = () => { const i = el.dataset.row != null ? +el.dataset.row : st.det.table.findIndex(r => r.pred === +el.dataset.b); if (i >= 0) select(i); };
  });
  renderSide(); if (st.sel != null) highlight(st.sel);
}
function renderSide() {
  const det = st.det, rows = $("#srows");
  if (!det) { $("#sstats").innerHTML = "Press <b>Detect</b> to run the model on this image."; rows.innerHTML = ""; $("#sweep").innerHTML = ""; $("#sdl").innerHTML = ""; return; }
  const t = det.tally;
  $("#sstats").innerHTML = t ? `<b>${t.correct} of ${t.n}</b> correct · <b>${t.named} of ${t.n}</b> named correctly · inference <b>${det.ms} ms</b><br>` +
    Object.entries(det.per_section).map(([s, v]) => `${esc(s)}: <b>${v[0]}/${v[1]}</b>`).join(" · ")
    : `<b>${det.boxes.length}</b> objects found · inference <b>${det.ms} ms</b> · no true boxes for uploads`;
  rows.innerHTML = det.table.map((r, i) => (st.filter === "err" && ["correct", "detected"].includes(r.result)) ? "" :
    `<div class="item r-${cls(r.result)}${i === st.sel ? " sel" : ""}" data-i="${i}"><span class="dot"></span><div><span class="t">${esc(r.true || r.predicted)} · ${esc(r.result)}</span>` +
    `${r.reason ? `<br><span class="w">${esc(r.reason)}</span>` : `<br><span class="w">${esc(r.true_name || r.predicted_name)}${r.confidence != null ? " · " + r.confidence : ""}</span>`}</div></div>`).join("");
  $$("#srows .item").forEach(el => el.onclick = () => select(+el.dataset.i));
  $("#sweep").innerHTML = det.sweep ? sweepSvg(det.sweep, det.conf) : "";
  $("#sdl").innerHTML = dlLinks(det.det);
}
function sweepSvg(s, cur) {
  const W = 360, H = 120, mx = Math.max(1, ...s.correct, ...s.missed, ...s.extra, ...s.wrong), x = t => 8 + (t - .05) / .85 * (W - 16), y = v => H - 14 - v / mx * (H - 26);
  const line = (k, c, dash) => `<polyline fill="none" stroke="${c}" stroke-width="1.8" ${dash ? 'stroke-dasharray="4 3"' : ""} points="${s.t.map((t, i) => x(t) + "," + y(s[k][i])).join(" ")}"/>`;
  return `<rect x="0" y="0" width="${W}" height="${H}" fill="rgba(0,0,0,.25)" rx="8"/>` + line("correct", "#39ff88") + line("missed", "#22e4ff", 1) + line("extra", "#8b97a8", 1) + line("wrong", "#ff4fd8") +
    `<line x1="${x(cur)}" x2="${x(cur)}" y1="6" y2="${H - 12}" stroke="#ffd84a" stroke-dasharray="3 3"/>` +
    `<text x="8" y="${H - 2}" font-size="9" fill="#7fa594" font-family="monospace">threshold 0.05 → 0.90 · green correct · cyan missed · grey extra · pink wrong ID</text>`;
}
function dlLinks(id) { return ["png", "csv", "json"].map(k => `<a class="btn sm" href="/api/download/${id}/${k}">${k}</a>`).join(""); }
function highlight(i) {
  const det = st.det, g = st.grid, hl = $("#fig #hlg"); if (!hl || !det) return; const r = det.table[i]; if (!r) return;
  const b = r.gt != null && det.truths[r.gt] ? det.truths[r.gt] : det.boxes[r.pred]; if (!b) return;
  let x1, y1, x2, y2;
  if (g.cols && g.rows) { const cx = Math.floor((b.x1 + b.x2) / 2 / 224), cy = Math.floor((b.y1 + b.y2) / 2 / 224); x1 = cx * 224; y1 = cy * 224; x2 = x1 + 224; y2 = y1 + 224; }
  else { const m = .15 * Math.max(b.x2 - b.x1, b.y2 - b.y1, 20); x1 = b.x1 - m; y1 = b.y1 - m; x2 = b.x2 + m; y2 = b.y2 + m; }
  hl.innerHTML = `<path class="hl" d="M0 0H${g.w}V${g.h}H0Z M${x1} ${y1}V${y2}H${x2}V${y1}Z"/><rect class="hlf" x="${x1}" y="${y1}" width="${x2 - x1}" height="${y2 - y1}"/>`;
}
async function select(i) {
  st.sel = i; $$("#srows .item, #errs .item").forEach(el => el.classList.toggle("sel", +el.dataset.i === i)); highlight(i);
  const r = st.det.table[i]; $("#insp").innerHTML = `<div class="lbl">loading...</div>`;
  try { const t = await api(`/api/tile/${st.det.det}/${i}`);
    $("#insp").innerHTML = `<img src="data:image/jpeg;base64,${t.image}"><div class="cap">${esc(t.caption)}</div>`; }
  catch (e) { $("#insp").innerHTML = `<div class="lbl">${esc(e.message)}</div>`; }
}
function clearHighlight() { st.sel = null; const hl = $("#fig #hlg"); if (hl) hl.innerHTML = ""; $$(".item.sel").forEach(e => e.classList.remove("sel")); }

/* ---------------------------------------------------------------- tooltips */
function showTip(e, el) {
  const det = st.det; if (!det) return; let html;
  if (el.dataset.row != null) { const r = det.table[+el.dataset.row]; html = `<b>${esc(r.true)}</b> ${esc(r.true_name)}<br>missed · ${esc(r.reason)}<br>${esc(r.section)}`; }
  else { const b = det.boxes[+el.dataset.b], r = det.table.find(x => x.pred === b.k) || {};
    html = `<b>${esc(b.id)}</b> ${esc(b.name)} · ${b.conf.toFixed(2)}<br>${esc(b.section)}${b.ap != null ? " · class AP " + b.ap : ""}${b.group ? " · look-alikes: " + esc(b.group) : ""}` +
           (r.result && !["correct", "detected"].includes(r.result) ? `<br><span style="color:var(--amber)">${esc(r.result)}: ${esc(r.reason)}</span>` : ""); }
  const t = $("#tip"); t.innerHTML = html; t.style.display = "block"; moveTip(e);
}
function moveTip(e) { const t = $("#tip"); t.style.left = Math.min(innerWidth - 320, e.clientX + 14) + "px"; t.style.top = (e.clientY + 14) + "px"; }
function hideTip() { $("#tip").style.display = "none"; }

/* ---------------------------------------------------------------- score + errors tiles */
function countUp(el, to, ms = 900) {
  if (calm()) { el.textContent = to; return; } const t0 = performance.now();
  const f = now => { const k = Math.min(1, (now - t0) / ms); el.textContent = Math.round(to * (1 - Math.pow(1 - k, 3))); if (k < 1) requestAnimationFrame(f); }; requestAnimationFrame(f);
}
function renderScore() {
  const det = st.det; if (!det) { $("#sc").textContent = "-"; $("#scof").textContent = ""; $("#scbar").style.width = "0"; $("#verdict").textContent = ""; $("#stats").innerHTML = ""; $("#ms").textContent = ""; return; }
  $("#ms").textContent = det.ms + " ms";
  if (det.tally) {
    const t = det.tally; countUp($("#sc"), t.correct); $("#scof").textContent = "/ " + t.n;
    setTimeout(() => $("#scbar").style.width = (100 * t.correct / t.n) + "%", 30);
    $("#verdict").innerHTML = t.fully ? `<span style="color:var(--acc)">FULLY CORRECT</span>` : `<span style="color:var(--amber)">${t.named}/${t.n} NAMED CORRECTLY</span>`;
    $("#stats").innerHTML = `<span>box off <b>${t.box_off}</b></span><span>wrong ID <b>${t.wrong}</b></span><span>missed <b>${t.missed}</b></span><span>extra <b>${t.extra}</b></span>`;
  } else {
    countUp($("#sc"), det.boxes.length); $("#scof").textContent = "found"; $("#scbar").style.width = "100%"; $("#verdict").textContent = "UPLOAD · NO TRUE BOXES"; $("#stats").innerHTML = "";
  }
  const errs = det.table.map((r, i) => [r, i]).filter(([r]) => det.has_truth ? r.result !== "correct" : true);
  $("#errn").textContent = det.has_truth ? errs.length + " errors" : errs.length + " found";
  $("#errs").innerHTML = errs.length ? errs.map(([r, i]) => `<div class="item r-${cls(r.result)}" data-i="${i}"><span class="dot"></span><div><span class="t">${esc(r.true || r.predicted)} · ${esc(r.result)}</span><br><span class="w">${esc(r.reason || r.predicted_name)}</span></div></div>`).join("")
                                     : `<div class="lbl" style="color:var(--acc)">no errors in this grid</div>`;
  $$("#errs .item").forEach(el => el.onclick = () => { openStage(false); select(+el.dataset.i); });
}

/* ---------------------------------------------------------------- actions */
async function newGrid(ev) {
  if (ev) shock(ev); setBusy(true, "building grid...");
  try {
    st.grid = await api("/api/grid", { size: st.size, sections: st.sections, mix: st.mix, seed: $("#seed").value.trim() || null });
    st.det = null; st.sel = null; st.lastSeed = st.grid.seed; $("#lastseed").textContent = "last seed: " + st.grid.seed;
    renderPreview(true); renderScore(); $("#insp").innerHTML = `<div class="lbl">select an object after detecting</div>`;
    if ($("#ov-stage").classList.contains("show")) renderStage("cascade");
    $("#status").textContent = "grid ready · press detect";
  } catch (e) { toast(e.message); $("#status").textContent = "error"; } finally { setBusy(false); }
}
async function detect(ev) {
  if (ev) shock(ev, true);
  if (!st.grid) return toast("Make a grid or upload an image first.");
  setBusy(true, "detecting...");
  try {
    st.det = await api("/api/detect", { id: st.grid.id, model: st.model, conf: st.conf, cell: st.cell, degrade: st.degrade });
    st.sel = null; renderPreview(false); renderScore(); openStage(true);
    $("#status").textContent = `${st.det.model} · ${st.det.ms} ms`;
  } catch (e) { toast(e.message); $("#status").textContent = "error"; } finally { setBusy(false); }
}
function setBusy(b, msg) { ["#newgrid", "#detect", "#s-new", "#s-det"].forEach(s => { $(s).disabled = b; $(s).classList.toggle("busy", b); }); if (msg) $("#status").textContent = msg; }
$("#newgrid").onclick = newGrid; $("#detect").onclick = detect; $("#s-new").onclick = newGrid; $("#s-det").onclick = detect;
$("#preview").onclick = () => openStage(false);
$("#s-close").onclick = () => { $("#ov-stage").classList.remove("show"); hideTip(); };
$$(".tabs .chip").forEach(c => c.onclick = () => { $$(".tabs .chip").forEach(x => x.classList.remove("on")); c.classList.add("on"); st.filter = c.dataset.f; renderSide(); });
addEventListener("keydown", e => { if (e.key === "Escape") $$(".ov.show").forEach(o => o.classList.remove("show")); });
addEventListener("resize", () => { if (st.grid) { renderPreview(false); if ($("#ov-stage").classList.contains("show")) fit($("#fig"), $("#figbox"), st.grid.w, st.grid.h); } });

/* upload */
$("#upbtn").onclick = e => { shock(e); $("#file").click(); };
$("#file").onchange = async e => {
  const f = e.target.files[0]; if (!f) return; const rd = new FileReader();
  rd.onload = async () => { setBusy(true, "uploading...");
    try { st.grid = await api("/api/upload", { image: rd.result, name: f.name }); st.det = null; st.sel = null; renderPreview(false); renderScore(); $("#status").textContent = "image ready · press detect"; }
    catch (err) { toast(err.message); } finally { setBusy(false); e.target.value = ""; } };
  rd.readAsDataURL(f);
};
/* downloads + about */
$("#dlbtn").onclick = e => { shock(e); if (!st.det) return toast("Detect first, then download."); $("#dls").innerHTML = dlLinks(st.det.det); $("#ov-dl").classList.add("show"); };
$("#dlclose").onclick = () => $("#ov-dl").classList.remove("show");
$("#aboutbtn").onclick = e => { shock(e); $("#ov-about").classList.add("show"); };
$("#aclose").onclick = () => $("#ov-about").classList.remove("show");
/* stress test */
$("#stressbtn").onclick = e => { shock(e); $("#ov-stress").classList.add("show"); };
$("#sclose").onclick = () => $("#ov-stress").classList.remove("show");
$("#sn").oninput = e => $("#snv").textContent = e.target.value;
$("#srun").onclick = async e => {
  shock(e, true); const n = +$("#sn").value; $("#srun").disabled = true; $("#sprog").style.width = "0"; $("#sres").innerHTML = `<p class="lbl">running ${n} grids...</p>`;
  try {
    const { job } = await api("/api/stress", { size: st.size, n, sections: st.sections, mix: st.mix, degrade: st.degrade, model: st.model, conf: st.conf, cell: st.cell });
    let j; do { await new Promise(r => setTimeout(r, 400)); j = await api(`/api/stress/${job}`); $("#sprog").style.width = (100 * j.done / j.total) + "%"; } while (!j.result && !j.error);
    if (j.error) throw new Error(j.error); const R = j.result, T = R.totals, pct = v => (100 * v / Math.max(1, T.n)).toFixed(1) + "%";
    const tbl = (h, rows) => `<table><tr><th>${h}</th><th>times</th></tr>${rows.length ? rows.map(r => `<tr><td>${esc(r[0])}</td><td>${r[1]}</td></tr>`).join("") : `<tr><td>none</td><td></td></tr>`}</table>`;
    $("#sres").innerHTML = `<div class="kpis"><div class="kpi"><div class="v">${pct(T.correct)}</div><div class="k">objects correct (${T.correct}/${T.n})</div></div>` +
      `<div class="kpi"><div class="v">${(100 * R.fully).toFixed(1)}%</div><div class="k">fully correct grids · 95% CI ${(100 * R.ci[0]).toFixed(1)}-${(100 * R.ci[1]).toFixed(1)}%</div></div>` +
      `<div class="kpi"><div class="v">${pct(T.named)}</div><div class="k">named correctly · ${R.ms.toFixed(0)} ms per grid</div></div></div>` +
      `<p class="lbl">${R.n_grids} × ${esc(R.size)} · ${esc(R.mix)} · ${esc(R.degrade)} · ${esc(R.model)} · box off ${T.box_off} · wrong ID ${T.wrong} · missed ${T.missed} · extra ${T.extra}</p>` +
      tbl("most often missed / wrong", R.hardest) + tbl("most common confusions", R.confusions) + tbl("failure reasons", R.reasons);
  } catch (err) { $("#sres").innerHTML = `<p style="color:#ff8fa8">${esc(err.message)}</p>`; } finally { $("#srun").disabled = false; }
};

/* ---------------------------------------------------------------- start */
(async () => {
  try {
    st.cfg = await api("/api/config"); st.model = st.cfg.best; st.conf = st.cfg.conf; st.sections = [...st.cfg.sections];
    $("#meta").textContent = `${st.cfg.n_objects} objects · ${st.cfg.n_test_photos} held-out test photos · ${st.cfg.best}`;
    renderControls(); BG.init();
  } catch (e) { toast("Cannot reach the server: " + e.message); BG.init(); }
})();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
