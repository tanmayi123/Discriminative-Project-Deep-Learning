"""
Milestone 2 extension: an improved detector that finds (almost) every object: in grids, rotated, scattered,
on any background, small, and in poor light.

    python m2_boost.py masks       # optional: SAM cut-outs of every object (needs internet once for the SAM weights)
    python m2_boost.py build       # training set, validation sets, test sets + preview sheets to check by eye
    python m2_boost.py train       # fine-tunes the selected model (falls back to a smaller batch/resolution if needed);
                                   # if interrupted, run it again: it RESUMES from the last finished epoch
    python m2_boost.py evaluate    # before vs after: choices on validation, then ONE look at each test set
    python m2_boost.py all         # build -> train -> evaluate (run masks first if you want cut-outs)

Uses the label check if it exists (M2 extra/label_check/boxes_master_checked.csv + extra_boxes.csv).
Everything is written ONLY to  PROJECT - 1/M2 extra/boost/ . The existing model, results, box file and EYE are only read:
training starts from a copy of the selected model's weights and saves its own new weights under M2 extra/boost/runs/.
Settings (environment variables): M2_IMGSZ (1600), M2_NTRAIN (3000), M2_EPOCHS (30), M2_BATCH (4), M2_SEED (0),
M2_BOOST_DIR (boost): output folder, e.g. boost2 for a second model; cut-outs and the photo cache are shared with "boost".
"""
import csv, hashlib, json, math, os, random, shutil, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

def _np(o):
    """JSON fallback: NumPy numbers become plain Python numbers (the detector returns float32/int64 values)."""
    return o.item() if hasattr(o, "item") else str(o)


TILE, NOMINAL = 224, 1120                       # photos are 224 px; composites are built at the old 1120-px scale, then rendered
IMGSZ = int(os.environ.get("M2_IMGSZ", 1600)); NTRAIN = int(os.environ.get("M2_NTRAIN", 3000))
EPOCHS = int(os.environ.get("M2_EPOCHS", 30)); BATCH = int(os.environ.get("M2_BATCH", 4)); SEED = int(os.environ.get("M2_SEED", 0))
FAKE = os.environ.get("M2_FAKE_MODEL") == "1"   # tests only: a stand-in for SAM and YOLO
LOOKALIKE = {"watches": ["OBJ025", "OBJ046", "OBJ065"], "wallets": ["OBJ061", "OBJ067"], "power banks": ["OBJ048", "OBJ057"],
             "rackets": ["OBJ030", "OBJ034", "OBJ035", "OBJ049"], "sunglasses / goggles": ["OBJ014", "OBJ045", "OBJ043"],
             "brush / comb": ["OBJ050", "OBJ059"], "bottles": ["OBJ002", "OBJ015", "OBJ063"], "projectors": ["OBJ064", "OBJ070"],
             "plush toys": ["OBJ021", "OBJ024", "OBJ071"], "notebook / keyboard": ["OBJ003", "OBJ012"]}
GROUP_OF = {o: g for g, ids in LOOKALIKE.items() for o in ids}

def find_project():
    if os.environ.get("M2_PROJECT_DIR"): return Path(os.environ["M2_PROJECT_DIR"])
    for start in (Path(__file__).resolve().parent, Path.cwd()):
        for d in (start, *start.parents):
            if (d / "milestone2_outputs").is_dir(): return d
    sys.exit("Project folder not found (the one containing milestone2_outputs/). Set M2_PROJECT_DIR.")
PROJECT = find_project(); OUT = PROJECT / "milestone2_outputs"; DATA = PROJECT / "Dataset - Updated"; DET = OUT / "detect"   # all read only
EXTRA = Path(os.environ["M2_EXTRA_DIR"]) if os.environ.get("M2_EXTRA_DIR") else PROJECT / "M2 extra"   # ALL new outputs go here
EXTRA.mkdir(parents=True, exist_ok=True)
B = EXTRA / os.environ.get("M2_BOOST_DIR", "boost"); B.mkdir(parents=True, exist_ok=True)     # e.g. M2_BOOST_DIR=boost2 for a second model
B1 = EXTRA / "boost"                                                         # the first improved model (kept; compared in evaluate)
LC = EXTRA / "label_check"; DSET = B / "dataset"; TESTS = B / "tests"
MASKS = B1 / "masks" if (B1 / "masks").exists() else B / "masks"            # cut-outs are shared between runs (no need to recompute)
import yaml
NAMES = yaml.safe_load((DET / "dataset" / "data.yaml").read_text())["names"]
NAMES = NAMES if isinstance(NAMES, list) else [NAMES[k] for k in sorted(NAMES, key=int)]
CLS = {o: i for i, o in enumerate(NAMES)}
def log(*a): print(*a, flush=True)
def photo_path(obj, fname): return DATA / f"images_{obj}" / fname
CACHE = (B1 if (B1 / "cache224").exists() else B) / "cache224"
def load_224(p):
    """The photo at 224 px (as in every M2 step). Cached on disk once, so building thousands of images stays fast."""
    p = Path(p); c = CACHE / p.parent.name / (p.stem + ".png")
    if c.exists():
        try: return Image.open(c).convert("RGB")
        except Exception: pass
    with Image.open(p) as im: img = ImageOps.exif_transpose(im).convert("RGB").resize((TILE, TILE), Image.Resampling.BICUBIC)
    try: c.parent.mkdir(parents=True, exist_ok=True); img.save(c)
    except Exception: pass
    return img

# ======================================================================================== labels (checked if available)
USABLE = ["verified", "fixed", "unreviewed_auto"]
def labels():
    checked = LC / "boxes_master_checked.csv"
    m = pd.read_csv(checked if checked.exists() else OUT / "boxes_master.csv")
    if not checked.exists(): log("NOTE: no label check found; using boxes_master.csv (run m2_label_check.py first for 100% checked labels)")
    m = m[m.status.isin(USABLE) & m.x1.notna() & m.object_id.isin(CLS)].copy(); m["key"] = m.object_id + "/" + m.fname
    m = m[[photo_path(o, f).exists() for o, f in zip(m.object_id, m.fname)]]
    extras = {}
    if (LC / "extra_boxes.csv").exists():
        for r in pd.read_csv(LC / "extra_boxes.csv").itertuples():
            if r.object_id in CLS: extras.setdefault(f"{r.object_id_photo}/{r.fname}", []).append((r.object_id, (r.x1, r.y1, r.x2, r.y2)))
    return m.reset_index(drop=True), extras

def backgrounds():
    """The background-only photos removed in Milestone 1 (real rooms, desks, floors), split by name into train/val/test."""
    f = PROJECT / "milestone1_outputs_v2" / "excluded_background_only.csv"; out = {"train": [], "val": [], "test": []}
    if f.exists():
        for r in pd.read_csv(f).itertuples():
            p = photo_path(r.class_name, r.fname)
            if p.exists():
                h = int(hashlib.md5(r.fname.encode()).hexdigest(), 16) % 100
                out["test" if h < 20 else "val" if h < 35 else "train"].append(p)
    log(f"real backgrounds: train {len(out['train'])}, val {len(out['val'])}, test {len(out['test'])}")
    return out

# ======================================================================================== geometry
def rot90_box(b, W, H, k):
    """Box after rotating a WxH image by k*90 degrees counter-clockwise (PIL rotate with expand=True)."""
    x1, y1, x2, y2 = b
    for _ in range(k % 4): x1, y1, x2, y2, W, H = y1, W - x2, y2, W - x1, H, W
    return (x1, y1, x2, y2)
def rot_any_box(b, W, H, deg):
    """Bounding box of a box after rotating a WxH image by deg (counter-clockwise, expand=True)."""
    a = math.radians(deg); c, s = math.cos(a), math.sin(a)
    nW, nH = abs(W * c) + abs(H * s), abs(W * s) + abs(H * c)
    pts = [(b[0], b[1]), (b[2], b[1]), (b[0], b[3]), (b[2], b[3])]; xs, ys = [], []
    for x, y in pts:
        dx, dy = x - W / 2, y - H / 2
        xs.append(nW / 2 + dx * c + dy * s); ys.append(nH / 2 - dx * s + dy * c)
    return (min(xs), min(ys), max(xs), max(ys))
def iou(a, b):
    iw, ih = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0: return 0.0
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - iw * ih
    return iw * ih / u if u > 0 else 0.0
def overlap_small(a, b):
    """Intersection as a share of the smaller box."""
    iw, ih = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0: return 0.0
    return iw * ih / max(1e-6, min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])))

# ======================================================================================== photometric changes (poor light, colour casts, blur, compression, noise)
def photometric(im, rng, p=0.35):
    if rng.random() > p: return im
    k = rng.choice(["dark", "dark", "cast", "blur", "jpeg", "noise", "bright"])
    if k == "dark":
        im = ImageEnhance.Brightness(im).enhance(rng.uniform(.3, .65))
        g = rng.uniform(1.1, 1.6); im = im.point(lambda v: int(255 * (v / 255) ** g))
    elif k == "bright": im = ImageEnhance.Brightness(im).enhance(rng.uniform(1.2, 1.5))
    elif k == "cast":
        a = np.asarray(im, float) * np.array([rng.uniform(.7, 1.3) for _ in range(3)]); im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    elif k == "blur": im = im.filter(ImageFilter.GaussianBlur(rng.uniform(.6, 1.6)))
    elif k == "jpeg":
        import io; buf = io.BytesIO(); im.save(buf, "JPEG", quality=rng.randint(15, 40)); buf.seek(0); im = Image.open(buf).convert("RGB")
    else:
        a = np.asarray(im, float) + np.random.default_rng(rng.randint(0, 10 ** 9)).normal(0, rng.uniform(3, 10), (im.height, im.width, 3))
        im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    return im

def synthetic_background(W, H, rng):
    k = rng.choice(["solid", "gradient", "noise", "stripes"]); c1 = tuple(rng.randint(20, 235) for _ in range(3)); c2 = tuple(rng.randint(20, 235) for _ in range(3))
    if k == "solid": return Image.new("RGB", (W, H), c1)
    if k == "gradient":
        t = np.linspace(0, 1, W)[None, :, None] if rng.random() < .5 else np.linspace(0, 1, H)[:, None, None]
        a = np.array(c1)[None, None] * (1 - t) + np.array(c2)[None, None] * t; return Image.fromarray(np.broadcast_to(a, (H, W, 3)).astype(np.uint8))
    if k == "noise":
        small = np.random.default_rng(rng.randint(0, 10 ** 9)).integers(0, 255, (max(2, H // 40), max(2, W // 40), 3), dtype=np.uint8)
        return Image.fromarray(small).resize((W, H), Image.Resampling.BICUBIC)
    im = Image.new("RGB", (W, H), c1); d = ImageDraw.Draw(im); w = rng.randint(12, 60)
    for x in range(-H, W, 2 * w): d.polygon([(x, 0), (x + w, 0), (x + w + H, H), (x + H, H)], fill=c2)
    return im

def background(W, H, rng, pool):
    if pool and rng.random() < .6:
        p = rng.choice(pool)
        with Image.open(p) as im: im = ImageOps.exif_transpose(im).convert("RGB")
        s = max(W / im.width, H / im.height) * rng.uniform(1, 1.4); im = im.resize((int(im.width * s) + 1, int(im.height * s) + 1))
        x, y = rng.randint(0, im.width - W), rng.randint(0, im.height - H); im = im.crop((x, y, x + W, y + H))
        return photometric(im, rng, .3)
    return synthetic_background(W, H, rng)

# ======================================================================================== pieces: a photo tile or a cut-out, with its boxes, transformed
class Piece:
    """An RGBA image plus its labelled boxes [(object_id, box)] in its own pixel coordinates."""
    def __init__(self, img, boxes): self.img, self.boxes = img, boxes
    def rot90(self, k):
        W, H = self.img.size; self.img = self.img.rotate(90 * k, expand=True); self.boxes = [(o, rot90_box(b, W, H, k)) for o, b in self.boxes]; return self
    def tilt(self, deg, exact_alpha=False):
        W, H = self.img.size; self.img = self.img.rotate(deg, expand=True, resample=Image.Resampling.BICUBIC)
        if exact_alpha:                                            # cut-outs: the visible object IS the box
            bb = self.img.getchannel("A").point(lambda v: 255 if v > 40 else 0).getbbox()
            self.boxes = [(self.boxes[0][0], bb)] if bb else []
        else: self.boxes = [(o, rot_any_box(b, W, H, deg)) for o, b in self.boxes]
        return self
    def scale(self, s):
        W, H = self.img.size; nW, nH = max(8, int(W * s)), max(8, int(H * s)); self.img = self.img.resize((nW, nH), Image.Resampling.BICUBIC)
        fx, fy = nW / W, nH / H; self.boxes = [(o, (b[0] * fx, b[1] * fy, b[2] * fx, b[3] * fy)) for o, b in self.boxes]; return self

def tile_piece(r, extras, rng, aug=True):
    img = load_224(photo_path(r.object_id, r.fname))
    if aug: img = photometric(img, rng, .3)
    boxes = [(r.object_id, (r.x1, r.y1, r.x2, r.y2))] + extras.get(r.key, [])
    return Piece(img.convert("RGBA"), boxes)

def cutout_piece(r, rng):
    mp = MASKS / r.object_id / (Path(r.fname).stem + ".png")
    if not mp.exists(): return None
    img = photometric(load_224(photo_path(r.object_id, r.fname)), rng, .3).convert("RGBA"); img.putalpha(Image.open(mp).convert("L"))
    bb = img.getchannel("A").getbbox()
    if not bb: return None
    img = img.crop(bb); return Piece(img, [(r.object_id, (0, 0, img.width, img.height))])

# ======================================================================================== composites (built at the nominal 1120-px scale)
def grid_image(rows, extras, rng, n_cols, n_rows, aug=True):
    W, H = n_cols * TILE, n_rows * TILE; canvas = Image.new("RGB", (W, H), (128, 128, 128)); boxes = []
    for i, r in enumerate(rows):
        p = tile_piece(r, extras, rng, aug)
        if aug and rng.random() < .25: p.rot90(rng.randint(1, 3))
        if aug and rng.random() < .25:
            p.tilt(rng.uniform(-10, 10)); s = TILE / max(p.img.size); p.scale(s)              # fits back in its cell
        x0, y0 = (i % n_cols) * TILE + (TILE - p.img.width) // 2, (i // n_cols) * TILE + (TILE - p.img.height) // 2
        canvas.paste(p.img, (x0, y0), p.img)
        boxes += [(o, (b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0)) for o, b in p.boxes]
    return canvas, boxes

def place(pieces, W, H, bg, rng, max_overlap):
    """Put pieces at random positions. A piece is pasted ON TOP of earlier ones, so its whole footprint (not just its object)
    may cover at most max_overlap of any earlier object; otherwise an earlier label would describe a hidden object."""
    canvas = bg.copy(); earlier, covered, boxes = [], [], []
    area = lambda b: max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
    inter = lambda a, b: max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    for p in pieces:
        if p.img.width >= W or p.img.height >= H: continue
        alpha_bb = p.img.getchannel("A").getbbox() or (0, 0, p.img.width, p.img.height)
        for _ in range(40):
            x, y = rng.randint(0, W - p.img.width), rng.randint(0, H - p.img.height)
            foot = (x + alpha_bb[0], y + alpha_bb[1], x + alpha_bb[2], y + alpha_bb[3])
            # the TOTAL covered share of each earlier object stays within max_overlap (several pieces cannot add up)
            if all(covered[j] + inter(foot, e) <= max_overlap * area(e) for j, e in enumerate(earlier)): break
        else: continue
        canvas.paste(p.img, (x, y), p.img)
        for j, e in enumerate(earlier): covered[j] += inter(foot, e)
        bbs = [(o, (max(0, b[0] + x), max(0, b[1] + y), min(W, b[2] + x), min(H, b[3] + y))) for o, b in p.boxes]
        earlier += [b for _, b in bbs]; covered += [0.0] * len(bbs); boxes += bbs
    return canvas, boxes

def scatter_image(rows, extras, rng, bgpool, aug=True, W=NOMINAL, H=NOMINAL):
    pieces = []
    for r in rows:
        p = tile_piece(r, extras, rng, aug)
        if rng.random() < .3: p.rot90(rng.randint(1, 3))
        if rng.random() < .3: p.tilt(rng.uniform(-25, 25))
        pieces.append(p.scale(rng.uniform(.35, 1.6)))
    return place(pieces, W, H, background(W, H, rng, bgpool), rng, max_overlap=.10)

def cutout_image(rows, rng, bgpool, W=NOMINAL, H=NOMINAL):
    pieces = []
    for r in rows:
        p = cutout_piece(r, rng)
        if p is None: continue
        if rng.random() < .6: p.tilt(rng.uniform(0, 360), exact_alpha=True)
        pieces.append(p.scale(rng.uniform(.4, 2.0)))
    return place(pieces, W, H, background(W, H, rng, bgpool), rng, max_overlap=.2)

def render(img, imgsz=None):
    """THE one scaling used for training, validation and inference: x imgsz/1120, then onto an imgsz canvas if it fits."""
    imgsz = imgsz or IMGSZ; s = imgsz / NOMINAL
    big = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.Resampling.BICUBIC) if s != 1 else img
    if big.width <= imgsz and big.height <= imgsz:
        c = Image.new("RGB", (imgsz, imgsz), (128, 128, 128)); c.paste(big, (0, 0)); return c, s
    return big, s

def write_yolo(img, boxes, folder, name, render_it=True):
    """Save an image and its YOLO label file; boxes are in the image's nominal coordinates."""
    (folder / "images").mkdir(parents=True, exist_ok=True); (folder / "labels").mkdir(parents=True, exist_ok=True)
    out, s = render(img) if render_it else (img, 1.0); W, H = out.size; out.save(folder / "images" / f"{name}.jpg", quality=92)
    with open(folder / "labels" / f"{name}.txt", "w") as f:
        for o, b in boxes:
            x1, y1, x2, y2 = (v * s for v in b)
            if x2 - x1 < 2 or y2 - y1 < 2: continue
            f.write(f"{CLS[o]} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}\n")

def sheet(items, path, n=12, cols=4, side=420):
    """A preview sheet: images with their boxes drawn, to check by eye."""
    items = items[:n]; rows = (len(items) + cols - 1) // cols
    S = Image.new("RGB", (cols * side, max(1, rows) * side), "white")
    for i, (img, boxes) in enumerate(items):
        k = side / max(img.size); im = img.resize((int(img.width * k), int(img.height * k))); d = ImageDraw.Draw(im)
        for o, b in boxes: d.rectangle([v * k for v in b], outline=(57, 255, 136), width=2); d.text((b[0] * k + 2, b[1] * k + 1), o, fill=(255, 255, 0))
        S.paste(im, ((i % cols) * side, (i // cols) * side))
    S.save(path, quality=88)

# ======================================================================================== masks (SAM cut-outs)
def cmd_masks():
    m, _ = labels(); MASKS.mkdir(parents=True, exist_ok=True); t0 = time.time(); acc, rej = [], []
    if not FAKE:
        from ultralytics import SAM
        os.chdir(EXTRA); sam = SAM(os.environ.get("M2_SAM", "sam2.1_b.pt"))   # the SAM weights are downloaded into M2 extra
    for k, r in enumerate(m.itertuples()):
        img = load_224(photo_path(r.object_id, r.fname)); box = [r.x1, r.y1, r.x2, r.y2]
        if FAKE:
            mk = Image.new("L", (TILE, TILE), 0); ImageDraw.Draw(mk).ellipse(box, fill=255); mask = np.asarray(mk) > 0
        else:
            res = sam(img, bboxes=[box], verbose=False)[0]
            if res.masks is None: rej.append((r, "no mask")); continue
            mask = res.masks.data[0].cpu().numpy() > .5
            if mask.shape != (TILE, TILE): mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((TILE, TILE))) > 127
        x1, y1, x2, y2 = [int(round(v)) for v in box]; inbox = np.zeros_like(mask); inbox[y1:y2, x1:x2] = True
        area_box = max(1, (x2 - x1) * (y2 - y1)); fill = (mask & inbox).sum() / area_box; outside = (mask & ~inbox).sum() / max(1, mask.sum())
        ys, xs = np.nonzero(mask & inbox); mb = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1) if len(xs) else (0, 0, 0, 0)
        if fill < .2 or outside > .25 or iou(mb, box) < .6: rej.append((r, f"fill {fill:.2f}, outside {outside:.2f}, fit {iou(mb, box):.2f}")); continue
        (MASKS / r.object_id).mkdir(exist_ok=True); Image.fromarray(((mask & inbox) * 255).astype(np.uint8)).save(MASKS / r.object_id / (Path(r.fname).stem + ".png")); acc.append(r)
        if k % 250 == 0: log(f"  {k} / {len(m)} photos ({time.time() - t0:.0f} s)")
    log(f"cut-outs: {len(acc)} accepted, {len(rej)} rejected (their normal photos are still used)")
    rng = random.Random(1); show = rng.sample(acc, min(48, len(acc)))
    S = Image.new("RGB", (8 * 140, 6 * 140), (40, 40, 40))
    for i, r in enumerate(show):
        p = cutout_piece(r, random.Random(0))
        if p: p.img.thumbnail((136, 136)); S.paste(p.img, ((i % 8) * 140 + 2, (i // 8) * 140 + 2), p.img)
    S.save(B / "masks_check.jpg", quality=88)
    pd.DataFrame([dict(object_id=r.object_id, fname=r.fname, reason=w) for r, w in rej]).to_csv(B / "masks_rejected.csv", index=False)
    log(f"check by eye: {B / 'masks_check.jpg'}  (rejected list: masks_rejected.csv)")

# ======================================================================================== build
def corrected_grids(split, m, extras):
    """The original M2 grids of a split with CORRECTED truth: checked boxes, second objects, excluded photos ignored."""
    man = pd.read_csv(DET / "dataset" / "manifest.csv"); col = lambda *c: next(x for x in man.columns if x.lower() in c)
    ci, cf, co = col("image", "grid", "file"), col("fname", "photo"), col("object_id", "class", "label")
    stem = lambda s: Path(str(s)).stem; full = pd.read_csv((LC / "boxes_master_checked.csv") if (LC / "boxes_master_checked.csv").exists() else OUT / "boxes_master.csv")
    full["key"] = full.object_id + "/" + full.fname; st = dict(zip(full.key, full.status)); bx = {r.key: (r.x1, r.y1, r.x2, r.y2) for r in full.itertuples()}
    photo = {(stem(a), o): f for a, f, o in zip(man[ci], man[cf], man[co])}; out = []
    for p in sorted((DET / "dataset" / "images" / split).glob("*.jpg")):
        W, H = Image.open(p).size; lab = DET / "dataset" / "labels" / split / (p.stem + ".txt"); boxes, ignore = [], []
        for line in (lab.read_text().split("\n") if lab.exists() else []):
            if not line.strip(): continue
            c, xc, yc, w, h = line.split()[:5]; o = NAMES[int(c)]; xc, yc = float(xc) * W, float(yc) * H
            cx, cy = int(xc // TILE), int(yc // TILE); ox, oy = cx * TILE, cy * TILE; f = photo.get((p.stem, o)); k = f"{o}/{f}"
            if f is None:                                         # not in the manifest: keep the original box
                ww, hh = float(w) * W, float(h) * H; boxes.append((o, (xc - ww / 2, yc - hh / 2, xc + ww / 2, yc + hh / 2))); continue
            if st.get(k) not in USABLE: ignore.append((ox, oy, ox + TILE, oy + TILE)); continue
            b = bx[k]; boxes.append((o, (b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy)))
            boxes += [(eo, (eb[0] + ox, eb[1] + oy, eb[2] + ox, eb[3] + oy)) for eo, eb in extras.get(k, [])]
        out.append(dict(name=p.stem, path=str(p), size=[W, H], boxes=boxes, ignore=ignore, grid=True))
    return out

def save_test(name, items):
    d = TESTS / name; (d / "images").mkdir(parents=True, exist_ok=True); rec = []
    for it in items:
        it["img"].save(d / "images" / f"{it['name']}.jpg", quality=95)
        rec.append(dict(name=it["name"], file=f"{it['name']}.jpg", size=list(it["img"].size), grid=it["grid"], ignore=it.get("ignore", []),
                        boxes=[[o, [round(v, 2) for v in b]] for o, b in it["boxes"]], sizes=it.get("sizes", [])))
    (d / "truth.json").write_text(json.dumps(rec, default=_np)); return rec

def cmd_build():
    rng = random.Random(SEED); np.random.seed(SEED); m, extras = labels(); bg = backgrounds()
    have_masks = MASKS.exists() and any(MASKS.rglob("*.png"))
    log(f"photos: {len(m)} ({m.split.value_counts().to_dict()}) | second objects: {sum(map(len, extras.values()))} | cut-outs available: {have_masks}")
    for d in (DSET, TESTS):
        if d.exists(): shutil.rmtree(d)
    tr = m[m.split == "train"]; va = m[m.split == "val"]; te = m[m.split == "test"]
    # sampling weights: look-alikes and the weakest validation classes appear more often
    weak = set()
    dj = EXTRA / "diagnosis" / "diagnosis.json"
    if dj.exists(): weak = {w["object"] for w in json.loads(dj.read_text()).get("weakest_classes", [])[:10]}
    w = np.array([1.6 if (o in GROUP_OF or o in weak) else 1.0 for o in tr.object_id]); w /= w.sum()
    order = list(range(len(tr))); rng.shuffle(order); pos = [0]
    def draw(n):                                                   # every photo at least once, then weighted extra draws
        out = []
        for _ in range(n):
            if pos[0] < len(order): out.append(tr.iloc[order[pos[0]]]); pos[0] += 1
            else: out.append(tr.iloc[np.random.choice(len(tr), p=w)])
        return [type("R", (), dict(r._asdict() if hasattr(r, "_asdict") else r.to_dict()))() for r in out]
    def by_group(n):                                               # a grid with look-alikes side by side
        g = rng.choice(list(LOOKALIKE)); ids = [o for o in LOOKALIKE[g] if o in set(tr.object_id)]; rows = draw(max(0, n - len(ids)))
        for o in ids:
            c = tr[tr.object_id == o]
            if len(c): r = c.iloc[rng.randrange(len(c))]; rows.append(type("R", (), r.to_dict())())
        rng.shuffle(rows); return rows[:n]
    for d in (tr, va, te): d["key"] = d.object_id + "/" + d.fname
    sizes = [(2, 2), (3, 3), (4, 4), (5, 4), (5, 5)]; mix = {"grid": .40, "scatter": .35 if have_masks else .55, "cutout": .20 if have_masks else 0, "empty": .05}
    kinds = rng.choices(list(mix), weights=list(mix.values()), k=NTRAIN); preview = {k: [] for k in mix}; counts = {k: 0 for k in mix}; t0 = time.time()
    for i, kind in enumerate(kinds):
        if kind == "grid":
            if rng.random() < .15: c, rr = 1, 1
            else: c, rr = rng.choice(sizes)
            rows = by_group(c * rr) if (c * rr > 1 and rng.random() < .5) else draw(c * rr)
            img, boxes = grid_image(rows, extras, rng, c, rr)
            if rng.random() < .2:
                k = rng.randint(1, 3); W, H = img.size; img = img.rotate(90 * k, expand=True); boxes = [(o, rot90_box(b, W, H, k)) for o, b in boxes]
        elif kind == "scatter": img, boxes = scatter_image(draw(rng.randint(3, 16)), extras, rng, bg["train"])
        elif kind == "cutout": img, boxes = cutout_image(draw(rng.randint(3, 14)), rng, bg["train"])
        else: img, boxes = background(NOMINAL, NOMINAL, rng, bg["train"]), []
        img = photometric(img.convert("RGB"), rng, .2)
        write_yolo(img, boxes, DSET / "train", f"{kind}_{i:05d}"); counts[kind] += 1
        if len(preview[kind]) < 6: preview[kind].append((img, boxes))
        if i % 300 == 0: log(f"  training images {i} / {NTRAIN} ({time.time() - t0:.0f} s)")
    sheet(sum(preview.values(), []), B / "preview_train.jpg", n=24, cols=6)
    log(f"training images: {counts} | photos used at least once: {min(pos[0], len(tr))} / {len(tr)}")
    # validation (used during training, rendered) and the same sets at nominal scale (for choosing the threshold)
    vrng = random.Random(SEED + 1); vg = corrected_grids("val", m, extras); val_items = []
    for g in vg:
        img = Image.open(g["path"]).convert("RGB"); val_items.append(dict(name=g["name"], img=img, boxes=g["boxes"], ignore=g["ignore"], grid=True))
        k = vrng.randint(1, 3); W, H = img.size
        val_items.append(dict(name=g["name"] + f"_r{90 * k}", img=img.rotate(90 * k, expand=True), boxes=[(o, rot90_box(b, W, H, k)) for o, b in g["boxes"]],
                              ignore=[rot90_box(b, W, H, k) for b in g["ignore"]], grid=True))
    vrows = [type("R", (), r.to_dict())() for _, r in va.sample(frac=1, random_state=SEED).iterrows()]
    for j in range(0, len(vrows), 10):
        img, boxes = scatter_image(vrows[j:j + 10], extras, vrng, bg["val"]); val_items.append(dict(name=f"val_scatter_{j // 10:03d}", img=img, boxes=boxes, grid=False))
        if have_masks and j % 20 == 0:
            img, boxes = cutout_image(vrows[j:j + 10], vrng, bg["val"]); val_items.append(dict(name=f"val_cutout_{j // 10:03d}", img=img, boxes=boxes, grid=False))
    for it in val_items: write_yolo(it["img"], it["boxes"], DSET / "val", it["name"])
    save_test("val_all", val_items)
    (DSET / "data.yaml").write_text(yaml.safe_dump(dict(path=str(DSET), train="train/images", val="val/images", names={i: n for i, n in enumerate(NAMES)}), sort_keys=False))
    # test sets: normal (corrected), rotated (every grid at 90/180/270), scattered and cut-out scenes on unseen backgrounds
    trng = random.Random(SEED + 2); tg = corrected_grids("test", m, extras)
    normal = [dict(name=g["name"], img=Image.open(g["path"]).convert("RGB"), boxes=g["boxes"], ignore=g["ignore"], grid=True) for g in tg]
    save_test("normal", normal); rotated = []
    for g in normal:
        W, H = g["img"].size
        for k in (1, 2, 3):
            rotated.append(dict(name=f"{g['name']}_r{90 * k}", img=g["img"].rotate(90 * k, expand=True), boxes=[(o, rot90_box(b, W, H, k)) for o, b in g["boxes"]],
                                ignore=[rot90_box(b, W, H, k) for b in g["ignore"]], grid=True))
    save_test("rotated", rotated)
    trows = [type("R", (), r.to_dict())() for _, r in te.sample(frac=1, random_state=SEED).iterrows()]; scat, cut = [], []
    for j in range(0, len(trows), 10):
        img, boxes = scatter_image(trows[j:j + 10], extras, trng, bg["test"]); scat.append(dict(name=f"scatter_{j // 10:03d}", img=img, boxes=boxes, grid=False))
        if have_masks:
            img, boxes = cutout_image(trows[j:j + 10], trng, bg["test"]); cut.append(dict(name=f"cutout_{j // 10:03d}", img=img, boxes=boxes, grid=False))
    save_test("scattered", scat)
    if cut: save_test("cutout", cut)
    sheet([(it["img"], it["boxes"]) for it in (rotated[:3] + scat[:6] + cut[:3])], B / "preview_tests.jpg", n=12)
    info = dict(imgsz=IMGSZ, ntrain=NTRAIN, counts=counts, val_images=len(val_items), tests=dict(normal=len(normal), rotated=len(rotated), scattered=len(scat), cutout=len(cut)),
                labels="checked" if (LC / "boxes_master_checked.csv").exists() else "boxes_master (not checked)", masks=have_masks)
    (B / "build.json").write_text(json.dumps(info, indent=2, default=_np)); log(json.dumps(info, indent=2, default=_np))
    log(f"check by eye: {B / 'preview_train.jpg'} and {B / 'preview_tests.jpg'}")

# ======================================================================================== train
def cmd_train():
    r = json.loads((DET / "results.json").read_text()); start = r["models"][r["selected_model"]]["weights"]
    if not (DSET / "data.yaml").exists(): sys.exit("Run  python m2_boost.py build  first.")
    if FAKE:
        (B / "runs" / "boost" / "weights").mkdir(parents=True, exist_ok=True); (B / "runs" / "boost" / "weights" / "best.pt").write_text("fake")
        (B / "train.json").write_text(json.dumps(dict(imgsz=IMGSZ, batch=BATCH, weights=str(B / "runs" / "boost" / "weights" / "best.pt"), start=start), default=_np)); log("fake training done"); return
    from ultralytics import YOLO
    import torch
    os.chdir(EXTRA)                                                     # anything ultralytics writes by itself lands in M2 extra
    run = B / "runs" / "boost"; last, best = run / "weights" / "last.pt", run / "weights" / "best.pt"
    if (B / "train.json").exists() and best.exists() and os.environ.get("M2_RETRAIN") != "1":
        sys.exit("Training already finished (M2 extra/boost/train.json). To train again from scratch, set M2_RETRAIN=1.")
    if last.exists() and not (B / "train.json").exists() and os.environ.get("M2_RETRAIN") != "1":
        # an interrupted run: continue from the last completed epoch, with exactly the same settings
        import yaml as _y
        a = _y.safe_load((run / "args.yaml").read_text()) if (run / "args.yaml").exists() else {}
        log(f"resuming the interrupted training from {last} (imgsz {a.get('imgsz')}, batch {a.get('batch')}, epochs {a.get('epochs')})")
        YOLO(str(last)).train(resume=True)
        (B / "train.json").write_text(json.dumps(dict(imgsz=int(a.get("imgsz", IMGSZ)), batch=a.get("batch"), epochs=a.get("epochs"), weights=str(best),
                                                      start=str(B / "start_weights.pt"), resumed=True), indent=2, default=_np)); log(f"done: {best}"); return
    copy = B / "start_weights.pt"; shutil.copy2(start, copy); start = str(copy)   # train from a copy: the original is never opened for writing
    plan = [(IMGSZ, b) for b in (BATCH, max(1, BATCH // 2), 1)] + [(1280, 2), (1120, 2)]; seen = set()
    for imgsz, batch in plan:
        if (imgsz, batch) in seen: continue
        seen.add((imgsz, batch)); log(f"training: from a copy of {r['selected_model']} ({Path(start).name}), imgsz {imgsz}, batch {batch}, epochs {EPOCHS}")
        try:
            model = YOLO(start)
            model.train(data=str(DSET / "data.yaml"), imgsz=imgsz, epochs=EPOCHS, batch=batch, patience=10, cos_lr=True, lr0=0.002, warmup_epochs=1,
                        mosaic=0.0, close_mosaic=0, degrees=0.0, fliplr=0.5, flipud=0.0, scale=0.2, translate=0.05, hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,
                        project=str(B / "runs"), name="boost", exist_ok=True, seed=SEED, plots=True, workers=2, amp=True, verbose=True)
            best = B / "runs" / "boost" / "weights" / "best.pt"
            (B / "train.json").write_text(json.dumps(dict(imgsz=imgsz, batch=batch, epochs=EPOCHS, weights=str(best), start=start), indent=2, default=_np)); log(f"done: {best}"); return
        except torch.cuda.OutOfMemoryError:
            log("  out of GPU memory, trying a smaller setting"); torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" not in str(e).lower(): raise
            log("  out of GPU memory, trying a smaller setting"); torch.cuda.empty_cache()
    sys.exit("Training did not fit in GPU memory even at 1120 px, batch 2.")

# ======================================================================================== evaluate
def ap_101(tp, conf, n_gt):
    if not len(tp): return 0.0
    o = np.argsort(-np.asarray(conf)); tp = np.asarray(tp, float)[o]; fp = 1 - tp
    rec = np.cumsum(tp) / n_gt; prec = np.cumsum(tp) / np.maximum(np.cumsum(tp) + np.cumsum(fp), 1e-9); prec = np.maximum.accumulate(prec[::-1])[::-1]
    return float(np.mean([prec[rec >= t].max() if (rec >= t).any() else 0.0 for t in np.linspace(0, 1, 101)]))

def mean_ap(preds, truth):
    classes = sorted({o for t in truth for o, _ in t["boxes"]}); out = []
    for thr in np.round(np.arange(.5, .96, .05), 2):
        aps = []
        for c in classes:
            n_gt = sum(1 for t in truth for o, _ in t["boxes"] if o == c)
            P = sorted(((s, t["name"], b) for t in truth for o, s, b in preds[t["name"]] if o == c), key=lambda x: -x[0]); used, tp, cf = set(), [], []
            G = {t["name"]: t["boxes"] for t in truth}
            for s, n, b in P:
                bj = max((j for j, (o, g) in enumerate(G[n]) if o == c and (n, j) not in used), key=lambda j: iou(b, G[n][j][1]), default=None)
                hit = bj is not None and iou(b, G[n][bj][1]) >= thr
                if hit: used.add((n, bj))
                tp.append(int(hit)); cf.append(s)
            aps.append(ap_101(tp, cf, n_gt))
        out.append(float(np.mean(aps)) if aps else 0.0)
    return round(out[0], 4), round(float(np.mean(out)), 4)

def post(P, t, conf, use_cell, fill=None):
    cell = lambda b: (int(((b[0] + b[2]) / 2) // TILE), int(((b[1] + b[3]) / 2) // TILE))
    ign = {cell(b) for b in t.get("ignore", [])}
    P = [q for q in P if not (t["grid"] and cell(q[2]) in ign)]
    K = [q for q in P if q[1] >= conf]
    if use_cell and t["grid"]:
        best = {}
        for q in sorted(K, key=lambda x: -x[1]): best.setdefault(cell(q[2]), q)
        K = list(best.values())
        if fill is not None:
            have = {cell(q[2]) for q in K}
            for c in {cell(b) for _, b in t["boxes"]} - have:
                q = next((q for q in sorted(P, key=lambda x: -x[1]) if cell(q[2]) == c and q[1] >= fill), None)
                if q: K.append(q)
    return K

def task(preds, truth, conf, use_cell=True, fill=None):
    n_gt = n_pred = ok = wrong = 0; full = []; small_gt = small_ok = 0
    for t in truth:
        G = t["boxes"]; P = post(preds[t["name"]], t, conf, use_cell, fill); used, k, w = set(), 0, 0; hit = set()
        for o, s, b in sorted(P, key=lambda x: -x[1]):
            j = max((j for j in range(len(G)) if j not in used), key=lambda j: iou(b, G[j][1]), default=None)
            if j is None or iou(b, G[j][1]) < .5: continue
            used.add(j); k += G[j][0] == o; w += G[j][0] != o
            if G[j][0] == o: hit.add(j)
        for j, (o, g) in enumerate(G):
            small = (g[2] - g[0]) * (g[3] - g[1]) < .02 * TILE * TILE
            small_gt += small; small_ok += small and j in hit
        n_gt += len(G); n_pred += len(P); ok += k; wrong += w; full.append(int(k == len(G) and len(P) == len(G)))
    return dict(recall=round(ok / max(n_gt, 1), 4), precision=round(ok / max(n_pred, 1), 4), wrong_ids=wrong, missed=n_gt - ok - wrong,
                fully_correct=round(float(np.mean(full)) if full else 0.0, 4), small_recall=round(small_ok / max(small_gt, 1), 4) if small_gt else None, objects=n_gt, images=len(truth))

class Detector:
    def __init__(self, weights, imgsz, tta=False):
        self.imgsz, self.tta = imgsz, tta
        if FAKE: self.model = None
        else:
            from ultralytics import YOLO
            self.model = YOLO(str(weights))
    def __call__(self, img, truth_boxes=None):
        canvas, s = render(img, self.imgsz)
        if FAKE: raw = fake_predict(truth_boxes or [], s, self.imgsz)          # tests: answers in RENDERED coordinates
        else:
            res = self.model.predict(canvas, imgsz=self.imgsz, conf=0.001, max_det=300, augment=self.tta, verbose=False)[0]
            raw = [(str(self.model.names[int(c)]), float(p), tuple(map(float, b))) for c, p, b in
                   zip(res.boxes.cls.cpu().numpy(), res.boxes.conf.cpu().numpy(), res.boxes.xyxy.cpu().numpy())]
        return [(o, p, tuple(v / s for v in b)) for o, p, b in raw]                # back to the image's own coordinates

def fake_predict(boxes, s, imgsz):
    """Tests only: the true boxes, made in RENDERED coordinates, with decisions that depend on the image, never on the scale,
    so a model run at 1120 and at 1600 px must give exactly the same results (any difference = a scaling error)."""
    key = "|".join(f"{o}:{b[0]:.1f},{b[1]:.1f}" for o, b in boxes); rng = random.Random(int(hashlib.md5(key.encode()).hexdigest(), 16)); out = []
    for o, b in boxes:
        if rng.random() < .1: continue
        d = [rng.uniform(-2, 2) for _ in range(4)]; j = lambda v, i: v * s + d[i] * s
        out.append((o if rng.random() > .03 else NAMES[0], rng.uniform(.5, .99), (j(b[0], 0), j(b[1], 1), j(b[2], 2), j(b[3], 3))))
    return out

def run(det, name):
    truth = json.loads((TESTS / name / "truth.json").read_text()); preds = {}
    for t in truth:
        img = Image.open(TESTS / name / "images" / t["file"]).convert("RGB"); preds[t["name"]] = det(img, [(o, tuple(b)) for o, b in t["boxes"]])
    for t in truth: t["boxes"] = [(o, tuple(b)) for o, b in t["boxes"]]; t["ignore"] = [tuple(b) for b in t.get("ignore", [])]
    return preds, truth

def cmd_evaluate():
    r = json.loads((DET / "results.json").read_text()); tj = json.loads((B / "train.json").read_text())
    before = Detector(r["models"][r["selected_model"]]["weights"], NOMINAL); after = Detector(tj["weights"], tj["imgsz"])
    prev = None
    if B.resolve() != B1.resolve() and (B1 / "train.json").exists():
        t1 = json.loads((B1 / "train.json").read_text()); e1 = json.loads((B1 / "evaluation.json").read_text()) if (B1 / "evaluation.json").exists() else {}
        prev = (Detector(t1["weights"], t1["imgsz"]), float((e1.get("chosen_on_validation") or {}).get("threshold", .10)))
    report = dict(before=dict(model=r["selected_model"], imgsz=NOMINAL, conf=0.10), after=dict(model="boost", imgsz=tj["imgsz"]))
    # ---- choices on VALIDATION only: threshold, filling empty cells, test-time augmentation
    pv, tv = run(after, "val_all")
    grid_v = [t for t in tv if t["grid"]]
    f1 = lambda m: 2 * m["recall"] * m["precision"] / max(1e-9, m["recall"] + m["precision"])
    cand = {c: task(pv, tv, c) for c in (.05, .10, .15, .20, .25, .30, .40)}; conf = max(cand, key=lambda c: f1(cand[c]))
    base = task(pv, grid_v, conf); fills = {f: task(pv, grid_v, conf, fill=f) for f in (.01, .03)}
    fbest = max(fills, key=lambda f: fills[f]["recall"]); use_fill = fills[fbest]["recall"] > base["recall"] + .002 and fills[fbest]["precision"] >= base["precision"] - .01
    tta_ok = False
    if not FAKE:
        pt, _ = run(Detector(tj["weights"], tj["imgsz"], tta=True), "val_all"); tta_ok = f1(task(pt, tv, conf)) > f1(cand[conf]) + .003
    report["chosen_on_validation"] = dict(threshold=conf, fill_empty_cells=fbest if use_fill else None, tta=tta_ok, thresholds={str(c): v for c, v in cand.items()})
    if tta_ok: after = Detector(tj["weights"], tj["imgsz"], tta=True)
    log(f"chosen on validation: threshold {conf}, fill empty cells {fbest if use_fill else 'no'}, test-time augmentation {'yes' if tta_ok else 'no'}")
    # ---- single photos: every test photo on its own (built here if the build did not make them)
    if not (TESTS / "singles" / "truth.json").exists():
        m, extras = labels(); te = m[m.split == "test"]; items = []
        for r in te.itertuples():
            boxes = [(r.object_id, (r.x1, r.y1, r.x2, r.y2))] + extras.get(r.key, [])
            items.append(dict(name=f"single_{r.object_id}_{Path(r.fname).stem}", img=load_224(photo_path(r.object_id, r.fname)), boxes=boxes, grid=True))
        save_test("singles", items); log(f"single-photo test set: {len(items)} photos")
    # ---- one look at each test set
    report["tests"] = {}
    for name in [n for n in ("normal", "rotated", "scattered", "cutout", "singles") if (TESTS / n / "truth.json").exists()]:
        pb, tb = run(before, name); pa, ta = run(after, name)
        rb = dict(**task(pb, tb, .10)); rb["mAP50"], rb["mAP50_95"] = mean_ap(pb, tb)
        ra = dict(**task(pa, ta, conf, fill=fbest if use_fill else None)); ra["mAP50"], ra["mAP50_95"] = mean_ap(pa, ta)
        report["tests"][name] = dict(before=rb, after=ra)
        if prev:
            pp, tp = run(prev[0], name); rp = dict(**task(pp, tp, prev[1])); rp["mAP50"], rp["mAP50_95"] = mean_ap(pp, tp); report["tests"][name]["previous"] = rp
        log(f"{name:>10}: before recall {rb['recall']:.3f} mAP50 {rb['mAP50']:.3f}" + (f" | previous {report['tests'][name]['previous']['recall']:.3f} {report['tests'][name]['previous']['mAP50']:.3f}" if prev else "") +
            f" | after recall {ra['recall']:.3f} mAP50 {ra['mAP50']:.3f}")
    (B / "evaluation.json").write_text(json.dumps(report, indent=2, default=_np))
    L = ["# Before vs after", "", f"Before: {report['before']['model']} (1120 px, threshold 0.10). After: boost model ({tj['imgsz']} px, threshold {conf}"
         f"{', empty cells filled' if use_fill else ''}{', test-time augmentation' if tta_ok else ''}; chosen on validation).", "",
         "| test set | images | objects | recall before -> after | precision | wrong IDs | fully correct | small-object recall | mAP@0.5 | mAP@0.5:0.95 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for n, v in report["tests"].items():
        b, a = v["before"], v["after"]; f = lambda x: "-" if x is None else f"{x:.3f}"
        L.append(f"| {n} | {a['images']} | {a['objects']} | {f(b['recall'])} -> **{f(a['recall'])}** | {f(b['precision'])} -> {f(a['precision'])} | {b['wrong_ids']} -> {a['wrong_ids']} | "
                 f"{f(b['fully_correct'])} -> {f(a['fully_correct'])} | {f(b['small_recall'])} -> {f(a['small_recall'])} | {f(b['mAP50'])} -> {f(a['mAP50'])} | {f(b['mAP50_95'])} -> {f(a['mAP50_95'])} |")
    if prev:
        L += ["", f"## The first improved model ({B1.name}) vs this one ({B.name}), same test sets and truth", "",
              "| test set | recall | precision | wrong IDs | fully correct | small-object recall | mAP@0.5 | mAP@0.5:0.95 |", "|---|---|---|---|---|---|---|---|"]
        for n, v in report["tests"].items():
            p_, a = v["previous"], v["after"]; f = lambda x: "-" if x is None else f"{x:.3f}"
            L.append(f"| {n} | {f(p_['recall'])} -> **{f(a['recall'])}** | {f(p_['precision'])} -> {f(a['precision'])} | {p_['wrong_ids']} -> {a['wrong_ids']} | {f(p_['fully_correct'])} -> {f(a['fully_correct'])} | "
                     f"{f(p_['small_recall'])} -> {f(a['small_recall'])} | {f(p_['mAP50'])} -> {f(a['mAP50'])} | {f(p_['mAP50_95'])} -> {f(a['mAP50_95'])} |")
    (B / "evaluation.md").write_text("\n".join(L) + "\n"); log("\n".join(L))

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "masks": cmd_masks()
    elif cmd == "build": cmd_build()
    elif cmd == "train": cmd_train()
    elif cmd == "evaluate": cmd_evaluate()
    elif cmd == "all": cmd_build(); cmd_train(); cmd_evaluate()
    else: sys.exit(__doc__)
