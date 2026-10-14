"""
Milestone 2 extension: a person checks EVERY box.

    python m2_label_check.py prepare     # the current model gives a second opinion on every photo (~5-10 min)
    python m2_label_check.py review      # opens the review page in the browser (stop and resume any time)
    python m2_label_check.py apply       # writes the checked box files (never touches boxes_master.csv)

Review page tabs
  Suspects        photos where the model strongly disagrees with the automatic box: check these carefully
  Box size        every split, test included: the label may cover only PART of the object, or be too large
  Rules           every photo of the five rule objects (both shoes, whole tissue box, watch with strap, ...), all splits
  Duplicates      near-identical photos of one object in different splits: D duplicate (the train/val copy is dropped) or N different
  Sweep           every other automatic box, 30 per page: click the wrong ones, then "page correct"
  To fix          photos flagged in the sweep: redraw or exclude
  Left out        photos without a usable box: draw one (or accept the model's suggestion)
  Second objects  another course object visible in the photo: add its box, or dismiss
  Apply           progress and the button that writes the checked box files

Editor keys: K keep label · M use model box · S save drawn box · X exclude · A add second object · arrows next/previous
             Z (or Backspace) back to your most recent decisions, to change any of them
Outputs ONLY in  PROJECT - 1/M2 extra/label_check/ : decisions.csv (every click), boxes_master_checked.csv, extra_boxes.csv, summary.json.
Nothing existing is changed (boxes_master.csv is only read).
"""
import io, json, os, re, sys, threading, time, webbrowser, csv, random
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import pandas as pd
from PIL import Image, ImageOps

def _np(o):
    """JSON fallback: NumPy numbers become plain Python numbers (the detector returns float32/int64 values)."""
    return o.item() if hasattr(o, "item") else str(o)


TILE = 224
def find_project():
    if os.environ.get("M2_PROJECT_DIR"): return Path(os.environ["M2_PROJECT_DIR"])
    for start in (Path(__file__).resolve().parent, Path.cwd()):
        for d in (start, *start.parents):
            if (d / "milestone2_outputs").is_dir(): return d
    sys.exit("Project folder not found (the one containing milestone2_outputs/). Set M2_PROJECT_DIR.")
PROJECT = find_project(); OUT = PROJECT / "milestone2_outputs"; DATA = PROJECT / "Dataset - Updated"
MASTER = OUT / "boxes_master.csv"                                    # read only, never written
EXTRA = Path(os.environ["M2_EXTRA_DIR"]) if os.environ.get("M2_EXTRA_DIR") else PROJECT / "M2 extra"   # ALL new outputs go here
EXTRA.mkdir(parents=True, exist_ok=True)
LC = EXTRA / "label_check"; LC.mkdir(parents=True, exist_ok=True)
PASS2 = os.environ.get("M2_PASS") == "2"          # second pass: the improved model checks the CORRECTED boxes
OPINION, DECISIONS = LC / ("model_opinion_pass2.json" if PASS2 else "model_opinion.json"), LC / "decisions.csv"
PASS2_START = LC / "pass2_started.txt"
FAKE = os.environ.get("M2_FAKE_MODEL") == "1"
AUTO, HUMAN, LEFT = ["unreviewed_auto"], ["verified", "fixed"], ["unreviewed_none", "pending_fix"]
RULE_ALL = "box the WHOLE object: every visible part, tight to its edges"
RULES = {"OBJ010": "one box around BOTH shoes", "OBJ051": "the WHOLE tissue box, including the tissue",
         "OBJ065": "the watch case AND the strap", "OBJ048": "only the silver power bank", "OBJ034": "the paddle only, never the tennis racket"}

def load_objects():
    try:
        t = pd.read_csv(PROJECT / "object_list.csv", dtype=str).fillna("")
        idc = max(t.columns, key=lambda c: t[c].str.strip().str.match(r"OBJ\d+$").mean())
        namec = "OBJECT_NAME" if "OBJECT_NAME" in t.columns else next(c for c in t.columns if "name" in c.lower() and "student" not in c.lower() and c != idc)
        return {i.strip(): n.strip() for i, n in zip(t[idc], t[namec]) if i.strip()}
    except Exception: return {}
NAMES = load_objects()
def photo_path(obj, fname): return DATA / f"images_{obj}" / fname
_cache = {}
def load_224(p):
    p = str(p)
    if p not in _cache:
        with Image.open(p) as im: _cache[p] = ImageOps.exif_transpose(im).convert("RGB").resize((TILE, TILE), Image.Resampling.BICUBIC)
        if len(_cache) > 800: _cache.pop(next(iter(_cache)))
    return _cache[p]
def iou(a, b):
    iw, ih = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0: return 0.0
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - iw * ih
    return iw * ih / u if u > 0 else 0.0

def master():
    src = LC / "boxes_master_checked.csv" if PASS2 else MASTER
    if PASS2 and not src.exists(): sys.exit("Pass 2 needs the first pass applied (label_check/boxes_master_checked.csv).")
    m = pd.read_csv(src); m = m[m.status.isin(AUTO + HUMAN + LEFT)].copy()
    m["key"] = m.object_id + "/" + m.fname
    return m[[photo_path(o, f).exists() for o, f in zip(m.object_id, m.fname)]].reset_index(drop=True)

# ======================================================================================== near-duplicates across splits
DUPS = LC / "near_duplicates.json"
import numpy as np
_D = None
def phash(path):
    """64-bit perceptual hash: low frequencies of a 32x32 greyscale DCT, above or below their median."""
    global _D
    if _D is None:
        n = np.arange(32); _D = np.sqrt(2 / 32) * np.cos(np.pi * (2 * n[None, :] + 1) * n[:, None] / 64); _D[0] /= np.sqrt(2)
    with Image.open(path) as im: g = np.asarray(ImageOps.exif_transpose(im).convert("L").resize((32, 32), Image.Resampling.LANCZOS), float)
    c = (_D @ g @ _D.T)[:8, :8].flatten()[1:]
    return int("".join("1" if v > np.median(c) else "0" for v in c), 2)
def near_duplicates(m, max_dist=10):
    """Pairs of photos of the SAME object in DIFFERENT splits whose hashes differ in at most max_dist of 63 bits."""
    t0 = time.time(); h = {}
    for k, r in enumerate(m.itertuples()):
        try: h[r.key] = phash(photo_path(r.object_id, r.fname))
        except Exception: pass
        if k % 500 == 0: print(f"  hashing {k} / {len(m)} ({time.time() - t0:.0f} s)", end="\r")
    print(); order = {"test": 0, "val": 1, "train": 2}; pairs = []
    for obj, g in m.groupby("object_id"):
        rows = [r for r in g.itertuples() if r.key in h]
        for i, a in enumerate(rows):
            for b in rows[i + 1:]:
                if a.split == b.split: continue
                d = bin(h[a.key] ^ h[b.key]).count("1")
                if d <= max_dist:
                    keep, drop = sorted((a, b), key=lambda r: order.get(r.split, 3))      # keep the test photo; drop the training/val copy
                    pairs.append(dict(key=f"{keep.key}|{drop.key}", keep=keep.key, drop=drop.key, dist=d, splits=f"{keep.split} / {drop.split}"))
    pairs.sort(key=lambda x: x["dist"]); DUPS.write_text(json.dumps(pairs, default=_np)); return pairs

# ======================================================================================== prepare: the model's opinion
def prepare():
    m = master(); print(f"photos: {len(m)} ({m.status.value_counts().to_dict()})")
    rows = list(m.itertuples()); opinion = {}
    if FAKE:
        rng = random.Random(0); names = sorted(m.object_id.unique())
        for r in rows:
            b = (r.x1, r.y1, r.x2, r.y2) if pd.notna(r.x1) else (60, 60, 160, 170)
            j = lambda v: max(0, min(TILE, v + rng.uniform(-6, 6)))
            P = [(r.object_id, rng.uniform(.5, .97), tuple(j(v) for v in b))]
            if rng.random() < .06: P = [(rng.choice(names), .8, P[0][2])]
            elif rng.random() < .08:                                 # the model boxes more than the label (a partial label)
                x1, y1, x2, y2 = b; w, h = x2 - x1, y2 - y1; P = [(r.object_id, .9, (max(0, x1 - .4 * w), max(0, y1 - .4 * h), min(TILE, x2 + .4 * w), min(TILE, y2 + .4 * h)))]
            if rng.random() < .05: P.append((rng.choice(names), .7, (5, 5, 60, 60)))
            opinion[r.key] = P
    else:
        from ultralytics import YOLO
        if PASS2:
            tj = json.loads((EXTRA / "boost" / "train.json").read_text()); model = YOLO(tj["weights"]); IMG = int(tj.get("imgsz", 5 * TILE))
            log_ = print; log_(f"pass 2: second opinion from the improved model ({tj['weights']}, {IMG} px)")
        else:
            r_ = json.loads((OUT / "detect" / "results.json").read_text()); model = YOLO(r_["models"][r_["selected_model"]]["weights"]); IMG = 5 * TILE
        sc = IMG / (5 * TILE)
        t0 = time.time(); batch = []
        def run(batch):
            grids = []
            for chunk in batch:
                g = Image.new("RGB", (5 * TILE, 5 * TILE), (128, 128, 128))
                for k, r in enumerate(chunk): g.paste(load_224(photo_path(r.object_id, r.fname)), ((k % 5) * TILE, (k // 5) * TILE))
                grids.append(g)
            if sc != 1: grids = [g.resize((IMG, IMG), Image.Resampling.BICUBIC) for g in grids]          # as the improved model was trained
            for chunk, res in zip(batch, model.predict(grids, imgsz=IMG, conf=0.01, max_det=300, verbose=False)):
                per = {k: [] for k in range(len(chunk))}
                for c, s, b in zip(res.boxes.cls.cpu().numpy(), res.boxes.conf.cpu().numpy(), res.boxes.xyxy.cpu().numpy() / sc):
                    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2; k = int(cy // TILE) * 5 + int(cx // TILE)
                    if k in per:
                        ox, oy = (k % 5) * TILE, (k // 5) * TILE
                        per[k].append((str(model.names[int(c)]), float(s), (max(0.0, float(b[0]) - ox), max(0.0, float(b[1]) - oy), min(float(TILE), float(b[2]) - ox), min(float(TILE), float(b[3]) - oy))))
                for k, r in enumerate(chunk): opinion[r.key] = sorted(per[k], key=lambda q: -q[1])[:8]
        chunks = [rows[i:i + 25] for i in range(0, len(rows), 25)]
        for i in range(0, len(chunks), 8):
            run(chunks[i:i + 8]); print(f"  {min(len(rows), (i + 8) * 25)} / {len(rows)} photos ({time.time() - t0:.0f} s)", end="\r")
        print()
    OPINION.write_text(json.dumps({k: [(o, round(float(s), 4), [round(float(v), 1) for v in b]) for o, s, b in P] for k, P in opinion.items()}, default=_np))
    q = queues(m, opinion); print("suspects:", len(q["suspects"]), "| sweep:", len(q["sweep"]), "| left out:", len(q["leftout"]), "| second objects:", len(q["second"]),
                                  "| box size:", len(q["extent"]), f"(test photos: {sum(1 for x in q['extent'] if x['split'] == 'test')})", "| rules:", len(q["rules"]))
    per = pd.Series([x["obj"] for x in q["extent"]]).value_counts(); tot = m.object_id.value_counts()
    if len(per):
        print("objects with the most box-size disagreements (a sign of an inconsistent box definition):")
        for o, n in per.head(10).items(): print(f"   {o} {NAMES.get(o, '')}: {n} of {tot.get(o, 0)} photos ({100 * n / max(tot.get(o, 1), 1):.0f}%)")
    print(f"saved {OPINION}")
    if PASS2:
        PASS2_START.write_text(time.strftime("%Y-%m-%d %H:%M:%S")); print("pass 2 ready: review Suspects, Box size and Second objects (duplicates were done in pass 1)"); return
    print("near-duplicate check (perceptual hashes, same object, different splits):")
    pairs = near_duplicates(m); c = pd.Series([p["splits"] for p in pairs]).value_counts().to_dict() if pairs else {}
    print(f"candidate pairs: {len(pairs)} {c}  (review them in the Duplicates tab)")

def queues(m, opinion):
    """Rank every photo: suspects, the rest of the automatic boxes, left-out photos, second-object candidates."""
    Q = dict(suspects=[], sweep=[], leftout=[], second=[], extent=[], rules=[])
    area = lambda b: max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    inter = lambda a, b: max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    for r in m.itertuples():
        P = [(o, s, tuple(b)) for o, s, b in opinion.get(r.key, [])]
        item = dict(key=r.key, fname=r.fname, obj=r.object_id, name=NAMES.get(r.object_id, ""), split=r.split, status=r.status,
                    box=[round(r.x1, 1), round(r.y1, 1), round(r.x2, 1), round(r.y2, 1)] if pd.notna(r.x1) else None)
        same = [q for q in P if q[0] == r.object_id]
        if same: item["model"] = dict(obj=same[0][0], conf=round(same[0][1], 3), box=[round(v, 1) for v in same[0][2]])
        if r.status in LEFT:
            item["reason"] = "no usable box: draw one" + (" (or accept the model's suggestion)" if same and same[0][1] >= .1 else "")
            Q["leftout"].append(item); continue
        L = tuple(item["box"]); why, score = None, 0
        other = [q for q in P if q[0] != r.object_id and q[1] >= .4 and iou(q[2], L) >= .5]
        if other and (not same or same[0][1] < .3):
            why, score = f"the model says {other[0][0]} ({NAMES.get(other[0][0], '')}) here, at {other[0][1]:.2f}", .9 * other[0][1]
            item["model"] = dict(obj=other[0][0], conf=round(other[0][1], 3), box=[round(v, 1) for v in other[0][2]])
        elif same and same[0][1] >= .4 and iou(same[0][2], L) < .5:
            v = iou(same[0][2], L); why, score = f"the model's box differs (overlap {v:.2f}, confidence {same[0][1]:.2f})", same[0][1] * (1 - v)
        elif not [q for q in same if q[1] >= .1 and iou(q[2], L) >= .3]:
            why, score = "the model finds nothing at this box", .5
        if r.status in AUTO or PASS2:
            if why: item["reason"], item["score"] = why, round(score, 3); Q["suspects"].append(item)
            elif not PASS2: Q["sweep"].append(item)
        if r.object_id in RULES and not PASS2: Q["rules"].append(dict(item))
        if not ((r.status in AUTO or PASS2) and why) and same and same[0][1] >= .5:         # box size: partial or oversized labels, every split
            M = tuple(same[0][2]); aL, aM = area(L), area(M)
            if aL > 0 and aM > 0:
                if inter(L, M) / aL >= .85 and aL / aM < .6:
                    Q["extent"].append(dict(item, reason=f"the label may cover only PART of the object (it is {100 * aL / aM:.0f}% of the model's box)", score=round(same[0][1] * (1 - aL / aM), 3)))
                elif inter(L, M) / aM >= .85 and aM / aL < .6:
                    Q["extent"].append(dict(item, reason=f"the label may be TOO LARGE (the model's box is {100 * aM / aL:.0f}% of it)", score=round(same[0][1] * (1 - aM / aL), 3)))
        for o, s, b in P:                                         # another course object, clearly elsewhere in the photo
            if o != r.object_id and s >= .5 and iou(b, L) < .2:
                Q["second"].append(dict(item, second=dict(obj=o, name=NAMES.get(o, ""), conf=round(s, 3), box=[round(v, 1) for v in b]),
                                        reason=f"the model sees {o} ({NAMES.get(o, '')}) at {s:.2f}")); break
    Q["suspects"].sort(key=lambda x: -x["score"]); Q["sweep"].sort(key=lambda x: (x["obj"], x["fname"])); Q["second"].sort(key=lambda x: -x["second"]["conf"])
    Q["extent"].sort(key=lambda x: -x["score"]); Q["rules"].sort(key=lambda x: (x["obj"], x["split"], x["fname"]))
    return Q

# ======================================================================================== decisions (append-only log)
FIELDS = ["time", "key", "action", "x1", "y1", "x2", "y2", "extra_obj"]
_lock = threading.Lock()
def log(rows):
    with _lock:
        new = not DECISIONS.exists()
        with open(DECISIONS, "a", newline="") as f:
            w = csv.DictWriter(f, FIELDS)
            if new: w.writeheader()
            for r in rows: w.writerow({k: r.get(k, "") for k in FIELDS} | {"time": time.strftime("%Y-%m-%d %H:%M:%S")})
def state():
    """Latest decision per photo, plus the list of second-object boxes (added minus removed)."""
    dec, extras = {}, {}
    if DECISIONS.exists():
        for r in csv.DictReader(open(DECISIONS)):
            if r["action"] == "extra": extras.setdefault(r["key"], []).append([r["extra_obj"], *map(float, (r["x1"], r["y1"], r["x2"], r["y2"]))])
            elif r["action"] == "extra_clear": extras.pop(r["key"], None)
            elif r["action"] == "second_dismiss": dec.setdefault("__second__", set()).add(r["key"])
            else: dec[r["key"]] = r
    return dec, extras

def apply():
    """Replays every decision in time order. 'ok' with a box confirms that box (pass 2 shows the corrected box);
    'ok' without a box confirms the original label (pass 1); 'fix' sets a new box; 'exclude' removes the photo."""
    m = pd.read_csv(MASTER); dec, extras = state(); dec.pop("__second__", None)
    m["key"] = m.object_id + "/" + m.fname; n = dict(ok=0, fix=0, exclude=0, flag=0)
    rows = sorted(csv.DictReader(open(DECISIONS)), key=lambda r: r["time"]) if DECISIONS.exists() else []
    per = {}
    for r in rows:
        if r["action"] in ("ok", "fix", "exclude", "flag"): per.setdefault(r["key"], []).append(r)
    idx = {k: i for i, k in enumerate(m.key)}
    for k, rs in per.items():
        if k not in idx: continue
        i = idx[k]; orig = [float(m.at[i, c]) if pd.notna(m.at[i, c]) else None for c in ("x1", "y1", "x2", "y2")]; box, status = list(orig), m.at[i, "status"]
        for r in rs:
            has = all(r.get(c) not in ("", None) for c in ("x1", "y1", "x2", "y2"))
            if r["action"] == "ok":
                box = [float(r[c]) for c in ("x1", "y1", "x2", "y2")] if has else list(orig)
                status = "verified" if (None not in box and None not in orig and all(abs(x - y) < .51 for x, y in zip(box, orig))) else ("fixed" if None not in box else status)
            elif r["action"] == "fix": box = [float(r[c]) for c in ("x1", "y1", "x2", "y2")]; status = "fixed"
            elif r["action"] == "exclude": status = "excluded_human"
        last = rs[-1]["action"]; n[last] += 1
        if status != m.at[i, "status"] or box != orig:
            if None not in box: m.loc[i, ["x1", "y1", "x2", "y2"]] = box
            m.at[i, "status"] = status
    dupd = {k for k, d in dec.items() if d["action"] == "dup_yes"}
    drops = {k.split("|")[1] for k in dupd}; n["duplicates_removed"] = 0
    for i, k in enumerate(m.key):
        if k in drops: m.at[i, "status"] = "excluded_duplicate"; n["duplicates_removed"] += 1
    m.drop(columns="key").to_csv(LC / "boxes_master_checked.csv", index=False)
    ex = [dict(object_id_photo=k.split("/")[0], fname=k.split("/")[1], object_id=e[0], x1=e[1], y1=e[2], x2=e[3], y2=e[4]) for k, es in extras.items() for e in es]
    pd.DataFrame(ex, columns=["object_id_photo", "fname", "object_id", "x1", "y1", "x2", "y2"]).to_csv(LC / "extra_boxes.csv", index=False)
    usable = m[m.status.isin(HUMAN)]; left = m[m.status.isin(AUTO)]
    s = dict(photos=int(len(m)), human_checked_boxes=int(len(usable)), still_unchecked_auto=int(len(left)),
             still_without_box=int(m.status.isin(LEFT).sum()), excluded_by_person=int((m.status == "excluded_human").sum()),
             near_duplicates_removed=int((m.status == "excluded_duplicate").sum()),
             decisions=n, second_object_boxes=len(ex), every_usable_box_checked=bool(len(left) == 0))
    (LC / "summary.json").write_text(json.dumps(s, indent=2, default=_np)); return s

# ======================================================================================== review server
PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Label check</title>
<style>
:root{--bg:#04070a;--tile:rgba(10,22,17,.9);--line:rgba(57,255,136,.2);--acc:#39ff88;--txt:#dcfff0;--dim:#7fa594;--amb:#ffb020;--cy:#22e4ff;--mag:#ff4fd8;--red:#ff5d5d}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--txt);font:13px Inter,system-ui,sans-serif}
header{display:flex;gap:6px;align-items:center;padding:10px 16px;border-bottom:1px solid var(--line);position:sticky;top:0;background:var(--bg);z-index:5;flex-wrap:wrap}
h1{font:600 13px ui-monospace,Consolas,monospace;letter-spacing:.2em;color:var(--acc);margin:0 16px 0 0}
.tab{border:1px solid var(--line);background:none;color:var(--dim);border-radius:999px;padding:6px 12px;cursor:pointer;font:600 11px ui-monospace,monospace;letter-spacing:.08em}
.tab.on{background:var(--acc);color:#02130a;border-color:var(--acc)} .sp{flex:1} .prog{color:var(--dim);font:11px ui-monospace,monospace}
main{padding:14px 16px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:10px}
.th{position:relative;border:2px solid transparent;border-radius:10px;overflow:hidden;cursor:pointer;background:#000}
.th.bad{border-color:var(--red)} .th.done{opacity:.45} .th canvas{display:block;width:100%;height:auto}
.th .cap{font:10px ui-monospace,monospace;padding:4px 6px;color:var(--dim);background:var(--tile);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.bar{display:flex;gap:8px;align-items:center;margin:10px 0}
.btn{border:1px solid var(--line);background:var(--tile);color:var(--acc);border-radius:8px;padding:8px 14px;cursor:pointer;font:600 12px ui-monospace,monospace}
.btn.pri{background:var(--acc);color:#02130a} .btn:hover{border-color:var(--acc)}
.ed{display:flex;gap:18px;align-items:flex-start;flex-wrap:wrap}
#cv{border:1px solid var(--line);border-radius:10px;cursor:crosshair;background:#000}
.info{min-width:300px;max-width:440px} .info .k{color:var(--dim);font:11px ui-monospace,monospace;text-transform:uppercase;letter-spacing:.1em;margin-top:10px}
.why{color:var(--amb);margin:6px 0} kbd{border:1px solid var(--line);border-radius:4px;padding:0 5px;font:11px ui-monospace,monospace;color:var(--acc)}
select{background:var(--tile);color:var(--txt);border:1px solid var(--line);border-radius:6px;padding:6px;font:12px ui-monospace,monospace;max-width:100%}
.leg span{margin-right:12px;font:11px ui-monospace,monospace} .sum td{padding:4px 12px 4px 0;border-bottom:1px solid var(--line)}
</style></head><body>
<header><h1>LABEL CHECK</h1>
 <button class="tab" data-t="suspects">Suspects</button><button class="tab" data-t="sweep">Sweep</button><button class="tab" data-t="tofix">To fix</button>
 <button class="tab" data-t="extent">Box size</button><button class="tab" data-t="rules">Rules</button><button class="tab" data-t="leftout">Left out</button><button class="tab" data-t="second">Second objects</button><button class="tab" data-t="dups">Duplicates</button><button class="tab" data-t="recent">↶ Recent</button><button class="tab" data-t="apply">Apply</button>
 <span class="sp"></span><span class="prog" id="prog"></span></header>
<main id="main">loading...</main>
<script>
const $ = s => document.querySelector(s);
let S = null, tab = "suspects", idx = 0, page = 0, drawn = null, mode = "box", flagged = new Set();
const PER = 30, Z = 2;   // sweep page size; editor zoom (224 -> 448 px)
async function api(u, body) { const r = await fetch(u, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {}); return r.json(); }
async function load() { S = await api("/api/state"); flagged = new Set(S.flagged); render(); }
const done = it => S.decided[it.key] && ["ok", "fix", "exclude"].includes(S.decided[it.key]);
function list() {
  if (tab === "tofix") return S.all.filter(it => S.decided[it.key] === "flag");
  if (tab === "second") return S.q.second.filter(it => !S.second_done.includes(it.key));
  if (tab === "recent") return S.recent;
  if (tab === "extent") return S.q.extent;
  return S.q[tab] || [];
}
function progress() {
  const auto = S.q.suspects.length + S.q.sweep.length, ad = S.q.suspects.concat(S.q.sweep).filter(done).length, ld = S.q.leftout.filter(done).length;
  const ed = S.q.extent.filter(done).length, rd = S.q.rules.filter(done).length;
  const tofix = S.all.filter(it => S.decided[it.key] === "flag").length;
  $("#prog").textContent = `automatic ${ad}/${auto} · box size ${ed}/${S.q.extent.length} · rules ${rd}/${S.q.rules.length} · left out ${ld}/${S.q.leftout.length} · to fix ${tofix} · second objects ${S.n_extra}`;
}
function drawPhoto(cv, it, z, extra) {
  const img = new Image(); img.onload = () => {
    const c = cv.getContext("2d"); c.drawImage(img, 0, 0, 224 * z, 224 * z);
    const R = (b, col, dash, w) => { if (!b) return; c.setLineDash(dash || []); c.strokeStyle = col; c.lineWidth = w || 2; c.strokeRect(b[0] * z, b[1] * z, (b[2] - b[0]) * z, (b[3] - b[1]) * z); };
    if (it.model && extra) R(it.model.box, "#22e4ff", [6, 4], 2);
    if (it.second && extra) R(it.second.box, "#ff4fd8", [6, 4], 2);
    R(it.box, "#39ff88", [], extra ? 2.5 : 2);
    if (S.fixes[it.key]) R(S.fixes[it.key], "#ffffff", [], 3);                        // the box you saved
    (S.extras[it.key] || []).forEach(e => R(e.slice(1), "#ffb020", [], 2.5));
    if (extra && drawn) R(drawn, "#ffffff", [3, 3], 2);
  }; img.src = `/img/${encodeURIComponent(it.key)}`;
}
function render() {
  document.querySelectorAll(".tab").forEach(b => b.classList.toggle("on", b.dataset.t === tab)); progress();
  if (tab === "apply") return renderApply();
  if (tab === "sweep" || tab === "rules") return renderSweep();
  if (tab === "dups") return renderDups();
  const L = list(); if (!L.length) { $("#main").innerHTML = `<p>Nothing here${tab === "tofix" ? ": flag photos in the Sweep first" : tab === "recent" ? ": no decisions yet" : ""}. ✓</p>`; return; }
  idx = Math.max(0, Math.min(idx, L.length - 1)); const it = L[idx]; drawn = null; mode = tab === "second" ? "extra" : "box";
  const st = S.decided[it.key], opts = Object.entries(S.names).map(([o, n]) => `<option value="${o}" ${(it.second && it.second.obj === o) ? "selected" : ""}>${o} ${n}</option>`).join("");
  $("#main").innerHTML = `<div class="bar"><button class="btn" id="pv">‹ previous</button><span>${idx + 1} / ${L.length}</span><button class="btn" id="nx">next ›</button>${tab === "recent" ? "" : `<button class="btn" id="nu">next unchecked »</button>`}<button class="btn" id="bk" title="your most recent decisions (Z)">↶ back (Z)</button>
     <span class="sp"></span><span class="leg"><span style="color:#39ff88">━ label</span><span style="color:#22e4ff">┅ model</span><span style="color:#ff4fd8">┅ second object (model)</span><span style="color:#ffb020">━ added second object</span><span style="color:#fff">━ your saved box</span><span>⬚ box being drawn</span></span></div>
   <div class="ed"><canvas id="cv" width="${224 * Z}" height="${224 * Z}"></canvas><div class="info">
     ${tab === "recent" ? `<div class="why" style="color:#22e4ff">Your decision ${idx + 1} from the latest: <b>${it.last}</b> (${it.when}). Choose again to change it.</div>` : ""}
     <div class="k">photo</div><div>${it.obj} ${it.name} · ${it.fname} · ${it.split}</div>
     <div class="why">${it.reason || ""}</div><div class="k">rule for this object</div><div style="color:#ffd84a">${S.rules[it.obj] || S.rule_all}</div>
     ${it.model ? `<div class="k">model's box</div><div>${it.model.obj} at confidence ${it.model.conf}${it.model.conf < .3 ? " <span style='color:#ff5d5d'>(weak: check carefully)</span>" : ""}</div>` : ""}
     ${st ? `<div class="k">decision so far</div><div>${st}</div>` : ""}
     <div class="k">${tab === "second" ? "add the second object" : "decide"}</div>
     ${tab === "second" ? `<p>Drag a box around the second object, choose its Object ID, then <kbd>A</kbd> add. Or dismiss.</p><select id="xo">${opts}</select>
        <div class="bar"><button class="btn pri" id="bA">A add box</button><button class="btn" id="bM2">use model box</button><button class="btn" id="bD">dismiss</button><button class="btn" id="bC">clear added</button></div>`
     : `<p>Drag on the photo to draw a new box.</p><div class="bar" style="flex-wrap:wrap">${it.box ? `<button class="btn pri" id="bK">K keep label</button>` : ""}
        ${it.model ? `<button class="btn" id="bM">M use model box</button>` : ""}<button class="btn" id="bS">S save my box</button><button class="btn" id="bX">X exclude photo</button>
        ${tab === "recent" && (S.extras[it.key] || []).length ? `<button class="btn" id="bC">clear added second object</button>` : ""}</div>`}
     <p style="color:var(--dim)">Keys: <kbd>K</kbd> <kbd>M</kbd> <kbd>S</kbd> <kbd>X</kbd> <kbd>A</kbd> · <kbd>←</kbd> <kbd>→</kbd> · <kbd>Z</kbd> back</p></div></div>`;
  const cv = $("#cv"); drawPhoto(cv, it, Z, true); bindDraw(cv, it);
  $("#pv").onclick = () => { idx--; render(); }; $("#nx").onclick = () => { idx++; render(); }; $("#bk").onclick = goBack;
  if ($("#nu")) $("#nu").onclick = () => { const k = L.findIndex((x, i) => i > idx && (tab === "second" || !done(x))); const f = k >= 0 ? k : L.findIndex(x => tab === "second" || !done(x)); if (f >= 0) { idx = f; render(); } };
  const B = (id, f) => { const e = $(id); if (e) e.onclick = f; };
  B("#bK", () => decide(it, "ok", it.box)); B("#bM", () => decide(it, "fix", it.model.box)); B("#bS", () => drawn ? decide(it, "fix", drawn) : alert("Draw a box first."));
  B("#bX", () => decide(it, "exclude")); B("#bA", () => addExtra(it, drawn)); B("#bM2", () => addExtra(it, it.second.box)); B("#bD", () => dismiss(it));
  B("#bC", async () => { await api("/api/decide", { key: it.key, action: "extra_clear" }); await reload(); });
}
function bindDraw(cv, it) {
  let s = null; const pos = e => { const r = cv.getBoundingClientRect(), k = r.width / 224, q = r.height / 224;
    return [Math.max(0, Math.min(224, (e.clientX - r.left) / k)), Math.max(0, Math.min(224, (e.clientY - r.top) / q))]; };
  cv.onmousedown = e => { e.preventDefault(); s = pos(e); };
  window.onmousemove = e => { if (!s) return; const p = pos(e); drawn = [Math.min(s[0], p[0]), Math.min(s[1], p[1]), Math.max(s[0], p[0]), Math.max(s[1], p[1])]; drawPhoto(cv, it, Z, true); };
  window.onmouseup = () => { if (!s) return; s = null; if (drawn && (drawn[2] - drawn[0] < 4 || drawn[3] - drawn[1] < 4)) drawn = null; };
}
async function reload() { const k = idx; await load(); idx = k; render(); }
async function decide(it, action, box) {
  await api("/api/decide", Object.assign({ key: it.key, action }, box ? { x1: box[0], y1: box[1], x2: box[2], y2: box[3] } : {}));
  S.decided[it.key] = action; if (action === "fix") S.fixes[it.key] = box; else delete S.fixes[it.key];
  if (tab === "recent") { await load(); idx = 0; return render(); }       // the changed photo is now the latest decision
  if (tab !== "tofix") idx++; S.recent.unshift(Object.assign({}, it, { last: action, when: "just now" })); render();
}
async function goBack() { await load(); tab = "recent"; idx = 0; render(); }
async function addExtra(it, box) {
  if (!box) return alert("Draw a box around the second object first."); const o = $("#xo").value;
  await api("/api/decide", { key: it.key, action: "extra", x1: box[0], y1: box[1], x2: box[2], y2: box[3], extra_obj: o });
  await api("/api/decide", { key: it.key, action: "second_dismiss" }); await reload();
}
async function dismiss(it) { await api("/api/decide", { key: it.key, action: "second_dismiss" }); await reload(); }
const pageBy = { sweep: 0, rules: 0 };
function renderSweep() {
  const L = tab === "rules" ? S.q.rules : S.q.sweep; page = pageBy[tab]; const pages = Math.ceil(L.length / PER) || 1; page = Math.max(0, Math.min(page, pages - 1)); pageBy[tab] = page;
  const items = L.slice(page * PER, (page + 1) * PER);
  $("#main").innerHTML = `<div class="bar"><button class="btn" id="pp">‹ page</button><span>page ${page + 1} / ${pages}</span><button class="btn" id="np">page ›</button>
     <button class="btn" id="fu">first unchecked page »</button><button class="btn" id="bk2" title="your most recent decisions (Z)">↶ back (Z)</button><span class="sp"></span><span style="color:var(--dim)">Click every WRONG box (it turns red), then:</span><button class="btn pri" id="ok">✓ rest of page correct (Enter)</button></div>
     <div style="color:#ffd84a;margin:0 0 6px">Every object: ${S.rule_all}.${tab === "rules" ? " These are all photos of the five rule objects, in every split: make each one follow its rule." : ""}</div>
     ${[...new Set(items.map(it => it.obj))].filter(o => S.rules[o]).map(o => `<div style="color:#ffd84a;margin:0 0 8px">Rule for ${o}: ${S.rules[o]}</div>`).join("")}
     <div class="grid">${items.map((it, i) => `<div class="th ${flagged.has(it.key) ? "bad" : ""} ${done(it) ? "done" : ""}" data-k="${it.key}"><canvas width="224" height="224" id="c${i}"></canvas><div class="cap">${it.obj} ${it.name} · ${it.fname}</div></div>`).join("")}</div>`;
  items.forEach((it, i) => drawPhoto($("#c" + i), it, 1, false));
  document.querySelectorAll(".th").forEach(t => t.onclick = () => { const k = t.dataset.k; flagged.has(k) ? flagged.delete(k) : flagged.add(k); t.classList.toggle("bad"); });
  $("#pp").onclick = () => { pageBy[tab] = page - 1; render(); }; $("#np").onclick = () => { pageBy[tab] = page + 1; render(); };
  $("#bk2").onclick = goBack;
  $("#fu").onclick = () => { const i = L.findIndex(it => !done(it) && S.decided[it.key] !== "flag"); if (i >= 0) { pageBy[tab] = Math.floor(i / PER); render(); } };
  $("#ok").onclick = async () => {
    const final = it => ["fix", "exclude"].includes(S.decided[it.key]);          // never overwrite a fix or an exclusion
    const ok = items.filter(it => !final(it) && !flagged.has(it.key)).map(it => it.key), bad = items.filter(it => !final(it) && flagged.has(it.key)).map(it => it.key);
    await api("/api/page", { ok, bad }); ok.forEach(k => S.decided[k] = "ok"); bad.forEach(k => S.decided[k] = "flag"); pageBy[tab] = page + 1; render(); window.scrollTo(0, 0);
  };
}
let di = 0;
function renderDups() {
  const L = S.dups; if (!L.length) { $("#main").innerHTML = "<p>No near-duplicate pairs across splits. ✓</p>"; return; }
  di = Math.max(0, Math.min(di, L.length - 1)); const p = L[di], st = S.decided[p.key];
  const nd = L.filter(x => S.decided[x.key]).length;
  $("#main").innerHTML = `<div class="bar"><button class="btn" id="dp">‹ previous</button><span>${di + 1} / ${L.length} (${nd} decided)</span><button class="btn" id="dn">next ›</button>
      <button class="btn" id="du">next undecided »</button></div>
    <p>Same scene, taken moments apart? A test photo whose near-twin was used for training measures memory, not recognition.
       Hash distance <b>${p.dist}</b> of 63 (0 = identical) · ${p.splits}${st ? ` · decided: <b>${st === "dup_yes" ? "duplicate" : "different"}</b>` : ""}</p>
    <div class="ed"><div><div class="k">kept (${p.a.split})</div><canvas id="ca" width="358" height="358"></canvas><div class="cap">${p.a.obj} · ${p.a.fname}</div></div>
      <div><div class="k">dropped if duplicate (${p.b.split})</div><canvas id="cb" width="358" height="358"></canvas><div class="cap">${p.b.obj} · ${p.b.fname}</div></div></div>
    <div class="bar"><button class="btn pri" id="dy">D duplicate: drop the ${p.b.split} copy</button><button class="btn" id="dno">N different photos</button></div>`;
  drawPhoto($("#ca"), p.a, 1.6, false); drawPhoto($("#cb"), p.b, 1.6, false);
  const go = async a => { await api("/api/decide", { key: p.key, action: a }); S.decided[p.key] = a; di++; renderDups(); };
  $("#dy").onclick = () => go("dup_yes"); $("#dno").onclick = () => go("dup_no");
  $("#dp").onclick = () => { di--; renderDups(); }; $("#dn").onclick = () => { di++; renderDups(); };
  $("#du").onclick = () => { const k = L.findIndex((x, i) => i > di && !S.decided[x.key]); if (k >= 0) { di = k; renderDups(); } };
}
async function renderApply() {
  $("#main").innerHTML = `<p>Writes <b>boxes_master_checked.csv</b> and <b>extra_boxes.csv</b> in M2 extra/label_check. boxes_master.csv is never changed. You can apply as often as you like.</p>
    <button class="btn pri" id="ap">Apply now</button><div id="res" style="margin-top:14px"></div>`;
  $("#ap").onclick = async () => { const s = await api("/api/apply", {}); $("#res").innerHTML = `<table class="sum">${Object.entries(s).map(([k, v]) => `<tr><td>${k}</td><td>${typeof v === "object" ? JSON.stringify(v) : v}</td></tr>`).join("")}</table>`; };
}
document.addEventListener("keydown", e => {
  if (e.target.tagName === "SELECT") return;
  const k = e.key.toLowerCase(), click = id => { const b = $(id); if (b) { e.preventDefault(); b.click(); } };
  if (k === "z" || e.key === "Backspace") { e.preventDefault(); return goBack(); }
  if (tab === "dups") { if (k === "d") click("#dy"); if (k === "n") click("#dno"); if (e.key === "ArrowRight") click("#dn"); if (e.key === "ArrowLeft") click("#dp"); return; }
  if ((tab === "sweep" || tab === "rules") && e.key === "Enter") return click("#ok");
  if (tab === "sweep" || tab === "rules") { if (e.key === "ArrowRight") click("#np"); if (e.key === "ArrowLeft") click("#pp"); return; }
  const keys = { k: "#bK", m: tab === "second" ? "#bM2" : "#bM", s: "#bS", x: "#bX", a: "#bA", arrowright: "#nx", arrowleft: "#pv" };
  if (keys[k]) click(keys[k]);
});
document.querySelectorAll(".tab").forEach(b => b.onclick = () => { tab = b.dataset.t; idx = 0; render(); });
load();
</script></body></html>"""

class App:
    def __init__(self):
        if not OPINION.exists(): sys.exit("Run  python m2_label_check.py prepare  first.")
        self.m = master(); self.opinion = json.loads(OPINION.read_text()); self.q = queues(self.m, self.opinion)
        self.dups = json.loads(DUPS.read_text()) if DUPS.exists() else []
        self.items = {}                                            # every photo, so any past decision can be reopened
        for r in self.m.itertuples():
            it = dict(key=r.key, fname=r.fname, obj=r.object_id, name=NAMES.get(r.object_id, ""), split=r.split, status=r.status,
                      box=[round(r.x1, 1), round(r.y1, 1), round(r.x2, 1), round(r.y2, 1)] if pd.notna(r.x1) else None)
            same = [q for q in self.opinion.get(r.key, []) if q[0] == r.object_id]
            if same: it["model"] = dict(obj=same[0][0], conf=round(same[0][1], 3), box=[round(v, 1) for v in same[0][2]])
            self.items[r.key] = it
        for qn in ("suspects", "sweep", "leftout", "second", "extent", "rules"):
            for it in self.q[qn]: self.items[it["key"]] = dict(self.items.get(it["key"], {}), **it)
    def recent(self, n=200):
        seen, out = set(), []
        if DECISIONS.exists():
            for r in reversed(list(csv.DictReader(open(DECISIONS)))):
                if r["key"] not in seen and r["key"] in self.items:
                    seen.add(r["key"]); out.append(dict(self.items[r["key"]], last=r["action"], when=r["time"]))
                    if len(out) >= n: break
        return out
    def state(self):
        dec, extras = state(); sec = dec.pop("__second__", set())
        if PASS2 and PASS2_START.exists():                          # pass 2: earlier decisions do not count as "done" here
            t0 = PASS2_START.read_text().strip(); dec = {k: d for k, d in dec.items() if d["time"] >= t0}
        all_items = {it["key"]: it for qn in ("suspects", "sweep", "leftout", "extent", "rules") for it in self.q[qn]}
        fixes = {k: [round(float(d[c]), 1) for c in ("x1", "y1", "x2", "y2")] for k, d in dec.items() if d["action"] == "fix" and d["x1"]}
        return dict(q=self.q, all=list(all_items.values()), decided={k: d["action"] for k, d in dec.items()}, fixes=fixes, flagged=[k for k, d in dec.items() if d["action"] == "flag"],
                    extras=extras, n_extra=sum(map(len, extras.values())), second_done=sorted(sec), names=dict(sorted(NAMES.items())), rules=RULES, rule_all=RULE_ALL,
                    recent=self.recent(), dups=[dict(p, a=self.items.get(p["keep"]), b=self.items.get(p["drop"])) for p in self.dups if p["keep"] in self.items and p["drop"] in self.items])

def serve(port=int(os.environ.get("M2_LABEL_PORT", 7870))):
    app = App()
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def _send(self, code, body, ctype="application/json"):
            b = body if isinstance(body, bytes) else json.dumps(body, default=_np).encode()
            self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b))); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(b)
        def do_GET(self):
            try:
                if self.path == "/": return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
                if self.path == "/api/state": return self._send(200, app.state())
                if self.path.startswith("/img/"):
                    from urllib.parse import unquote
                    obj, fname = unquote(self.path[5:]).split("/", 1)
                    if not re.fullmatch(r"OBJ\d{3}", obj) or "/" in fname or "\\" in fname or ".." in fname: return self._send(400, {"error": "bad path"})
                    buf = io.BytesIO(); load_224(photo_path(obj, fname)).save(buf, "JPEG", quality=90); return self._send(200, buf.getvalue(), "image/jpeg")
                self._send(404, {"error": "not found"})
            except Exception as e: self._send(500, {"error": str(e)})
        def do_POST(self):
            try:
                d = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                if self.path == "/api/decide":
                    if d.get("action") not in ("ok", "fix", "exclude", "flag", "extra", "extra_clear", "second_dismiss", "dup_yes", "dup_no"): return self._send(400, {"error": "bad action"})
                    log([d]); return self._send(200, {"ok": True})
                if self.path == "/api/page":
                    log([dict(key=k, action="ok") for k in d.get("ok", [])] + [dict(key=k, action="flag") for k in d.get("bad", [])]); return self._send(200, {"ok": True})
                if self.path == "/api/apply": return self._send(200, apply())
                self._send(404, {"error": "not found"})
            except Exception as e: self._send(500, {"error": str(e)})
    srv = ThreadingHTTPServer(("127.0.0.1", port), H); url = f"http://127.0.0.1:{port}"
    print(f"Label check running at {url}  (Ctrl+C to stop; every click is saved)")
    if not os.environ.get("M2_NO_BROWSER"): threading.Timer(1, lambda: webbrowser.open(url)).start()
    srv.serve_forever()

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "review"
    if cmd == "prepare": prepare()
    elif cmd == "review": serve()
    elif cmd == "apply": print(json.dumps(apply(), indent=2, default=_np))
    elif cmd == "merge":                                          # python m2_label_check.py merge other_decisions.csv [...]
        rows = []
        for f in sys.argv[2:]: rows += list(csv.DictReader(open(f)))
        rows.sort(key=lambda r: r["time"]); have = list(csv.DictReader(open(DECISIONS))) if DECISIONS.exists() else []
        allr = sorted(have + rows, key=lambda r: r["time"])
        with open(DECISIONS, "w", newline="") as f:
            w = csv.DictWriter(f, FIELDS); w.writeheader(); [w.writerow({k: r.get(k, "") for k in FIELDS}) for r in allr]
        print(f"merged {len(rows)} decisions from {len(sys.argv) - 2} file(s); {len(allr)} in total, in time order")
    else: sys.exit(__doc__)
