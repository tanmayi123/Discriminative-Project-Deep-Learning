# =============================================================================
# Discriminative Project - Milestone 2 (Mukul): multi-object detection with YOLO11s
#
#   STEP 1  labels : a tight box around the object in every single photo
#                    (YOLO-World + YOLOE, cross-checked; name-free fallback for misses)
#   STEP 2  grids  : 224x224 photos side by side in 2x2, 3x3, 4x4, 5x4, 5x5 grids + YOLO labels
#   STEP 3  train  : YOLO11s at imgsz=1120 (resumes automatically if interrupted)
#   STEP 4  test   : mAP@0.5, mAP@0.5:0.95, precision, recall, per-class AP, per-grid-size mAP
#   STEP 5  demo   : Object IDs + locations of every object in test images, saved as pictures
#
# Every step saves its results and is skipped on the next run if they already exist,
# so the script can be re-run safely after an interruption.
# =============================================================================
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import re, math, json, time, random, shutil
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image, ImageOps, ImageDraw, ImageEnhance
from tqdm.auto import tqdm

# ============================== SETTINGS ==============================
PROJECT   = Path("/Applications/Documents/Sem 4/Deep Learning/Project")
M1_OUT    = PROJECT / "mukul_outputs"             # Milestone 1 outputs: split.csv
M2_DIR    = PROJECT / "m2_yolo11"                 # everything for Milestone 2 goes here
PHOTO_DIR = M2_DIR / "photos_224"                 # cleaned 224x224 copies (boxes refer to these pixels)
GRID_DIR  = M2_DIR / "grid_dataset"               # YOLO dataset: images/ + labels/ + data.yaml
RUNS_DIR  = M2_DIR / "runs"
RESULTS   = M2_DIR / "results"

SUBSET    = "section1"        # "section1" = the 38 Section 1 objects (TA's suggestion) | "all" = all 73
SEED, IMG = 42, 224
EXCLUDE   = {"OBJ054_017.jpg"}                    # TA: not a photo of the object

# labeling
LABEL_DEVICE = "cpu"          # Apple's GPU driver crashes with YOLO-World's text model -> labeling on CPU
AGREE_IOU = 0.50              # two boxes "agree" if they overlap at least this much
SOLO_CONF = 0.40              # if only one model is confident (>= this), its box is kept
MAX_DROP  = 0.20              # TA: leave out at most 20% of the object photos used

# grids: (columns, rows) -> number of images per split. 5x4 = 1120x896, 5x5 = 1120x1120
GRID_COUNTS = {
    "train": {(2, 2): 300, (3, 3): 250, (4, 4): 150, (5, 4): 150, (5, 5): 100},
    "val":   {(2, 2): 60,  (3, 3): 50,  (4, 4): 30,  (5, 4): 30,  (5, 5): 20},
    "test":  {(2, 2): 60,  (3, 3): 50,  (4, 4): 30,  (5, 4): 30,  (5, 5): 20},
}

# training (TA tip: imgsz close to the largest grid, batch 8)
MODEL     = "yolo11s.pt"
IMGSZ     = 1120
EPOCHS    = 40
PATIENCE  = 10                # stop early if validation mAP does not improve for this many epochs
BATCH     = 8                 # lower to 4 if you run out of memory
TRAIN_DEVICE = "auto"         # "auto" = NVIDIA GPU if present, else Mac GPU (mps), else CPU
CONF      = 0.25              # boxes below this confidence are ignored in the demo

# the 38 Section 1 Object IDs (SECTION-01 in the shared sheet)
SECTION1 = ["OBJ001", "OBJ003", "OBJ004", "OBJ011", "OBJ012", "OBJ015", "OBJ019", "OBJ020", "OBJ021", "OBJ022",
            "OBJ027", "OBJ030", "OBJ031", "OBJ032", "OBJ034", "OBJ035", "OBJ036", "OBJ037", "OBJ038", "OBJ039",
            "OBJ040", "OBJ041", "OBJ042", "OBJ043", "OBJ045", "OBJ046", "OBJ048", "OBJ052", "OBJ054", "OBJ055",
            "OBJ057", "OBJ060", "OBJ062", "OBJ063", "OBJ064", "OBJ065", "OBJ066", "OBJ073"]

# object names from the shared sheet, and the text prompts given to the open-vocabulary detectors
OBJECT_NAMES = {
    "OBJ001": "Basketball", "OBJ002": "Water bottle", "OBJ003": "Notebook", "OBJ004": "Toy car",
    "OBJ005": "Computer mouse", "OBJ006": "Capybara toy", "OBJ007": "Apple", "OBJ008": "Cat",
    "OBJ009": "Marshall speaker", "OBJ010": "Nike sneakers", "OBJ011": "Mug", "OBJ012": "Computer keyboard",
    "OBJ013": "Mobile phone", "OBJ014": "Sunglasses", "OBJ015": "Water tumbler", "OBJ016": "Sandisk portable SSD",
    "OBJ017": "Controller", "OBJ018": "Rubik's cube", "OBJ019": "Plant", "OBJ020": "Airpod",
    "OBJ021": "Toy bunny", "OBJ022": "Guitar", "OBJ023": "Bracelet", "OBJ024": "Plush doll",
    "OBJ025": "Watch", "OBJ026": "Red Sox cap", "OBJ027": "Tennis ball", "OBJ028": "Banana",
    "OBJ029": "Orange", "OBJ030": "Tennis racket", "OBJ031": "UNO card box", "OBJ032": "Sound card",
    "OBJ033": "Duck", "OBJ034": "Table tennis racket", "OBJ035": "Badminton racket", "OBJ036": "Pencil case",
    "OBJ037": "Candle", "OBJ038": "Perfume bottle", "OBJ039": "Scissors", "OBJ040": "Clutcher",
    "OBJ041": "Table fan", "OBJ042": "Mouth guard case", "OBJ043": "Swimming goggles", "OBJ044": "Anime figurine",
    "OBJ045": "Wayfarer sunglasses", "OBJ046": "iWatch", "OBJ047": "Backpack", "OBJ048": "Wireless powerbank",
    "OBJ049": "Pickleball bat", "OBJ050": "Hairbrush", "OBJ051": "Tissue", "OBJ052": "Spoon",
    "OBJ053": "Moisturizer", "OBJ054": "Purse", "OBJ055": "Scientific calculator", "OBJ056": "Trader Joe's bag",
    "OBJ057": "Power bank", "OBJ058": "Footstool", "OBJ059": "Comb", "OBJ060": "Sleep mask",
    "OBJ061": "Card wallet", "OBJ062": "Headphones", "OBJ063": "Measuring bottle", "OBJ064": "RGB projector",
    "OBJ065": "Analog watch", "OBJ066": "Hair oil", "OBJ067": "Men's wallet", "OBJ068": "Neck pillow",
    "OBJ069": "Knife", "OBJ070": "Astronaut projector", "OBJ071": "Sloth plushie", "OBJ072": "Dog",
    "OBJ073": "Bowl",
}
PROMPTS = {   # generic words work better than brand names
    "OBJ001": ["basketball"],                    "OBJ002": ["water bottle"],
    "OBJ003": ["notebook"],                      "OBJ004": ["toy car"],
    "OBJ005": ["computer mouse"],                "OBJ006": ["capybara plush toy", "stuffed animal"],
    "OBJ007": ["apple"],                         "OBJ008": ["cat"],
    "OBJ009": ["bluetooth speaker", "speaker"],  "OBJ010": ["sneaker", "shoe"],
    "OBJ011": ["mug", "cup"],                    "OBJ012": ["computer keyboard", "keyboard"],
    "OBJ013": ["mobile phone", "smartphone"],    "OBJ014": ["sunglasses"],
    "OBJ015": ["water tumbler", "bottle"],       "OBJ016": ["portable hard drive", "small electronic device"],
    "OBJ017": ["game controller"],               "OBJ018": ["rubik's cube"],
    "OBJ019": ["potted plant", "plant"],         "OBJ020": ["earbuds case", "airpods case"],
    "OBJ021": ["toy bunny", "stuffed animal"],   "OBJ022": ["guitar"],
    "OBJ023": ["bracelet"],                      "OBJ024": ["plush doll", "stuffed animal"],
    "OBJ025": ["watch", "wristwatch"],           "OBJ026": ["baseball cap", "cap"],
    "OBJ027": ["tennis ball"],                   "OBJ028": ["banana"],
    "OBJ029": ["orange"],                        "OBJ030": ["tennis racket"],
    "OBJ031": ["card box", "box"],               "OBJ032": ["sound card", "small electronic device"],
    "OBJ033": ["rubber duck", "duck"],           "OBJ034": ["table tennis paddle", "ping pong paddle"],
    "OBJ035": ["badminton racket"],              "OBJ036": ["pencil case"],
    "OBJ037": ["candle"],                        "OBJ038": ["perfume bottle"],
    "OBJ039": ["scissors"],                      "OBJ040": ["hair claw clip", "hair clip"],
    "OBJ041": ["table fan", "electric fan"],     "OBJ042": ["mouth guard case", "small plastic case"],
    "OBJ043": ["swimming goggles", "goggles"],   "OBJ044": ["anime figurine", "figurine"],
    "OBJ045": ["sunglasses"],                    "OBJ046": ["smartwatch", "watch"],
    "OBJ047": ["backpack"],                      "OBJ048": ["power bank"],
    "OBJ049": ["pickleball paddle", "paddle"],   "OBJ050": ["hairbrush"],
    "OBJ051": ["tissue box", "tissue packet"],   "OBJ052": ["spoon"],
    "OBJ053": ["moisturizer bottle", "lotion bottle"], "OBJ054": ["purse", "handbag"],
    "OBJ055": ["scientific calculator", "calculator"], "OBJ056": ["tote bag", "shopping bag"],
    "OBJ057": ["power bank"],                    "OBJ058": ["footstool", "stool"],
    "OBJ059": ["comb"],                          "OBJ060": ["sleep mask", "eye mask"],
    "OBJ061": ["card wallet", "wallet"],         "OBJ062": ["headphones"],
    "OBJ063": ["measuring bottle", "bottle"],    "OBJ064": ["projector", "small electronic device"],
    "OBJ065": ["analog watch", "wristwatch"],    "OBJ066": ["hair oil bottle", "bottle"],
    "OBJ067": ["wallet"],                        "OBJ068": ["neck pillow", "travel pillow"],
    "OBJ069": ["knife"],                         "OBJ070": ["astronaut figure", "toy"],
    "OBJ071": ["sloth plush toy", "stuffed animal"], "OBJ072": ["dog"],
    "OBJ073": ["bowl"],
}
Image.MAX_IMAGE_PIXELS = None
for d in (M2_DIR, RESULTS):
    d.mkdir(parents=True, exist_ok=True)


# ============================== shared helpers ==============================
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0

def yolo_line(cid, x1, y1, x2, y2, W, H):
    """YOLO label: class cx cy w h, normalised to 0-1."""
    return f"{cid} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}"

def read_yolo_labels(txt, W, H):
    out = []
    for line in Path(txt).read_text().splitlines():
        if line.strip():
            c, cx, cy, w, h = line.split(); cx, cy, w, h = float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
            out.append((int(c), cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return out

def pick_device(setting):
    import torch
    if setting != "auto":
        return setting
    return 0 if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


# =============================================================================
# STEP 1: tight boxes for every single photo
# =============================================================================
def load_photos():
    """Index the dataset, reuse the Milestone 1 split, apply the subset, make clean 224x224 copies."""
    data_dir = sorted(PROJECT.rglob("images_OBJ001"))[0].parent
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    rows = []
    for folder in sorted(d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith((".", "__"))):
        cname = re.search(r"OBJ\d+", folder.name).group(0)
        for f in sorted(folder.iterdir()):
            if f.suffix.lower() in exts and not f.name.startswith("."):
                rows.append({"src": str(f), "fname": f.name, "class_name": cname})
    df = pd.DataFrame(rows)
    assert set(df.class_name) == set(PROMPTS), "Object IDs in the dataset do not match the prompt list"

    # Milestone 1 split: background photos and exact duplicates already removed, no photo in two splits
    split = pd.read_csv(M1_OUT / "split.csv")
    photos = df.merge(split[["fname", "class_name", "split"]], on=["fname", "class_name"])
    assert len(photos) == len(split), "split.csv does not match the dataset"
    photos = photos[~photos.fname.isin(EXCLUDE)]
    keep_ids = SECTION1 if SUBSET == "section1" else sorted(PROMPTS)
    photos = photos[photos.class_name.isin(keep_ids)].reset_index(drop=True)
    photos["path"] = [str(PHOTO_DIR / c / Path(f).with_suffix(".jpg").name) for c, f in zip(photos.class_name, photos.fname)]

    resized = 0
    for src, dst in zip(photos.src, photos.path):
        dst = Path(dst)
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            if im.size != (IMG, IMG):                               # TA: e.g. one OBJ003 photo is not 224x224
                im = im.resize((IMG, IMG), Image.BICUBIC); resized += 1
            im.save(dst, quality=95)
    print(f"[1] subset '{SUBSET}': {photos.class_name.nunique()} Object IDs | {len(photos)} photos "
          f"| excluded by the TA: {sorted(EXCLUDE)} | resized this run: {resized}")
    return photos

def best_box(result, max_area=None):
    """Highest-confidence box -> (x1, y1, x2, y2, conf), or None. Optionally ignores boxes covering almost the whole photo."""
    b = result.boxes
    if b is None or len(b) == 0:
        return None
    order = b.conf.argsort(descending=True).tolist()
    for i in order:
        x1, y1, x2, y2 = (float(v) for v in b.xyxy[i])
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(IMG, x2), min(IMG, y2)
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        if max_area and (x2 - x1) * (y2 - y1) > max_area * IMG * IMG:
            continue
        return (x1, y1, x2, y2, float(b.conf[i]))
    return None

def run_detector(photos, tag, load, set_prompts, conf=0.05, max_area=None):
    """Detect the object in the given photos, one Object ID at a time. Cached per Object ID:
    if the prompts or the photo list change, only that Object ID is re-run."""
    cdir = M2_DIR / f"raw_{tag}"; cdir.mkdir(exist_ok=True)
    model, parts = None, []
    for cname, grp in tqdm(photos.groupby("class_name"), desc=f"[1] {tag}"):
        cache = cdir / f"{cname}.csv"
        key = "|".join(PROMPTS[cname]) if set_prompts else "prompt-free"
        if cache.exists():
            old = pd.read_csv(cache)
            if len(old) and old.prompts.iloc[0] == key and set(grp.path) <= set(old.path):
                parts.append(old[old.path.isin(grp.path)]); continue
        if model is None:
            model = load()
        if set_prompts:
            set_prompts(model, PROMPTS[cname])
        out = []
        for path, r in zip(grp.path, model.predict(list(grp.path), imgsz=640, conf=conf, device=LABEL_DEVICE,
                                                   verbose=False, stream=True)):
            bb = best_box(r, max_area)
            out.append({"path": path, "prompts": key,
                        **(dict(zip(["x1", "y1", "x2", "y2", "conf"], bb)) if bb else
                           {"x1": np.nan, "y1": np.nan, "x2": np.nan, "y2": np.nan, "conf": 0.0})})
        out = pd.DataFrame(out); out.to_csv(cache, index=False); parts.append(out)
    cols = ["path", "prompts", "x1", "y1", "x2", "y2", "conf"]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)

def as_box(row):
    return None if pd.isna(row.x1) else (row.x1, row.y1, row.x2, row.y2)

def step1_labels():
    out_csv = M2_DIR / f"box_labels_{SUBSET}.csv"
    photos = load_photos()
    if out_csv.exists():
        labels = pd.read_csv(out_csv)
        if set(labels.path) == set(photos.path):
            print(f"[1] labels already made: {out_csv.name} (delete it to redo the decisions)")
            return labels

    from ultralytics import YOLO, YOLOE
    def world_prompts(m, names): m.set_classes(names)
    def yoloe_prompts(m, names):
        try:
            m.set_classes(names, m.get_text_pe(names))
        except TypeError:
            m.set_classes(names)

    # pass A: both detectors, prompted with the object's name (cached from earlier runs)
    W = run_detector(photos, "yoloworld", lambda: YOLO("yolov8s-worldv2.pt"), world_prompts).set_index("path")
    E = run_detector(photos, "yoloe", lambda: YOLOE("yoloe-11s-seg.pt"), yoloe_prompts).set_index("path")

    dec = []
    for p in photos.path:
        w, e = W.loc[p], E.loc[p]
        wb, eb = as_box(w), as_box(e)
        ov = iou(wb, eb) if (wb and eb) else 0.0
        if wb and eb and ov >= AGREE_IOU:
            box, status = tuple((a + b) / 2 for a, b in zip(wb, eb)), "agree"
        elif wb and w.conf >= SOLO_CONF and (not eb or w.conf >= e.conf):
            box, status = wb, "yoloworld_only"
        elif eb and e.conf >= SOLO_CONF:
            box, status = eb, "yoloe_only"
        else:
            cand = [(c, b) for c, b in ((w.conf, wb), (e.conf, eb)) if b]
            box, status = (max(cand)[1] if cand else None), "unreliable"
        dec.append({"path": p, "box": box, "status": status, "conf_max": max(float(w.conf), float(e.conf)), "iou": ov})
    dec = pd.DataFrame(dec)

    # pass B (fallback) for the unreliable photos: two name-free / low-threshold detectors.
    # Every photo contains exactly one object, so if both point at the same place, that is the object.
    miss = photos[photos.path.isin(dec.path[dec.status == "unreliable"])]
    print(f"[1] pass A: {len(photos) - len(miss)} photos labelled, {len(miss)} unreliable -> fallback pass")
    if len(miss):
        PF = run_detector(miss, "yoloe_promptfree", lambda: YOLOE("yoloe-11s-seg-pf.pt"), None,
                          conf=0.05, max_area=0.90).set_index("path")
        WL = run_detector(miss, "yoloworld_lowconf", lambda: YOLO("yolov8s-worldv2.pt"), world_prompts,
                          conf=0.001, max_area=0.90).set_index("path")
        for i in dec.index[dec.status == "unreliable"]:
            p = dec.at[i, "path"]
            pb, lb = as_box(PF.loc[p]), as_box(WL.loc[p])
            if pb and lb and iou(pb, lb) >= AGREE_IOU:
                dec.at[i, "box"] = tuple((a + b) / 2 for a, b in zip(pb, lb)); dec.at[i, "status"] = "fallback_agree"
                dec.at[i, "conf_max"] = max(dec.at[i, "conf_max"], float(PF.loc[p].conf))
            elif dec.at[i, "box"] is None and (pb or lb):
                dec.at[i, "box"] = pb or lb                          # weak candidate, only used if rescued below

    labels = photos.merge(dec, on="path")
    labels["keep"] = labels.status != "unreliable"

    # TA: leave out no more than 20% of the object photos used -> if more are unreliable, the most
    # confident unreliable boxes are kept (status "kept_for_review") until the limit is met
    budget = int(math.floor(MAX_DROP * len(labels)))
    unrel = labels[(labels.status == "unreliable") & labels.box.notna()].sort_values("conf_max", ascending=False)
    n_left = (~labels.keep).sum()
    if n_left > budget:
        rescue = unrel.index[: n_left - budget]
        labels.loc[rescue, "keep"] = True; labels.loc[rescue, "status"] = "kept_for_review"
    for k, col in enumerate(["x1", "y1", "x2", "y2"]):
        labels[col] = [round(b[k], 1) if isinstance(b, tuple) else np.nan for b in labels.box]
    labels = labels.drop(columns=["box"])
    labels.to_csv(out_csv, index=False)

    # report: photos left out per Object ID and why (required in the report)
    rep = labels.groupby("class_name").agg(photos=("path", "size"), kept=("keep", "sum"),
                                           agree=("status", lambda s: (s == "agree").sum()),
                                           one_model=("status", lambda s: s.isin(["yoloworld_only", "yoloe_only"]).sum()),
                                           fallback=("status", lambda s: (s == "fallback_agree").sum()),
                                           kept_for_review=("status", lambda s: (s == "kept_for_review").sum()))
    rep["left_out"] = rep.photos - rep.kept
    rep["left_out_pct"] = (100 * rep.left_out / rep.photos).round(1)
    rep.insert(0, "object_name", [OBJECT_NAMES[c] for c in rep.index])
    rep["reason_left_out"] = np.where(rep.left_out > 0, "no detector could locate the object reliably "
                                      "(small, far away, cluttered or unusual view)", "")
    rep.to_csv(RESULTS / f"left_out_per_object_{SUBSET}.csv")
    print("[1] label status:\n" + labels.status.value_counts().to_string())
    print(f"[1] kept {labels.keep.sum()} of {len(labels)} photos | left out {(~labels.keep).sum()} "
          f"({100 * (~labels.keep).mean():.1f}%, limit {100 * MAX_DROP:.0f}%) -> {RESULTS.name}/left_out_per_object_{SUBSET}.csv")

    # review sheets: check a sample of the boxes by eye (TA: "check a sample of your boxes")
    for name, sel in [("agree", labels.status == "agree"),
                      ("one_model_or_fallback", labels.status.isin(["yoloworld_only", "yoloe_only", "fallback_agree"])),
                      ("kept_for_review", labels.status == "kept_for_review"),
                      ("left_out", ~labels.keep)]:
        review_sheet(labels[sel], RESULTS / f"review_{name}_{SUBSET}.jpg")
    print(f"[1] review sheets saved in {RESULTS} (red = final box)")
    return labels

def review_sheet(sel, out_path, n=48, cols=8):
    if not len(sel):
        return
    sel = sel.sample(min(n, len(sel)), random_state=SEED)
    S = IMG + 20
    sheet = Image.new("RGB", (cols * IMG, math.ceil(len(sel) / cols) * S), "white")
    for k, (_, r) in enumerate(sel.iterrows()):
        im = Image.open(r.path).convert("RGB")
        if not pd.isna(r.x1):
            ImageDraw.Draw(im).rectangle([r.x1, r.y1, r.x2, r.y2], outline=(255, 0, 0), width=3)
        x, y = (k % cols) * IMG, (k // cols) * S
        sheet.paste(im, (x, y + 20))
        ImageDraw.Draw(sheet).text((x + 3, y + 4), f"{r.class_name} {r.status}", fill="black")
    sheet.save(out_path, quality=90)


# =============================================================================
# STEP 2: grid images (224x224 photos side by side) + YOLO labels
# =============================================================================
def augment_tile(im, box, rng):
    """Light augmentation for training tiles; the box moves with the image."""
    x1, y1, x2, y2 = box
    if rng.random() < 0.5:                                           # horizontal flip
        im = ImageOps.mirror(im); x1, x2 = IMG - x2, IMG - x1
    if rng.random() < 0.8:                                           # lighting / camera differences
        im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.7, 1.3))
        im = ImageEnhance.Contrast(im).enhance(rng.uniform(0.75, 1.25))
        im = ImageEnhance.Color(im).enhance(rng.uniform(0.7, 1.3))
    return im, (x1, y1, x2, y2)

def step2_grids(labels):
    yaml_path = GRID_DIR / "data.yaml"
    classes = sorted(labels.class_name.unique())                     # class ID 0..N-1 -> Object ID
    cid = {c: i for i, c in enumerate(classes)}
    if yaml_path.exists() and (GRID_DIR / "manifest.csv").exists():
        print(f"[2] grid dataset already built: {GRID_DIR} (delete the folder to rebuild)")
        return yaml_path, classes

    usable = labels[labels.keep].copy()
    rng = random.Random(SEED)
    if GRID_DIR.exists():
        shutil.rmtree(GRID_DIR)
    manifest = []
    for split, counts in GRID_COUNTS.items():
        (GRID_DIR / "images" / split).mkdir(parents=True); (GRID_DIR / "labels" / split).mkdir(parents=True)
        pool = usable[usable.split == split].to_dict("records")
        deck = []                                                    # deal photos like cards: all used before any repeats
        def draw():
            if not deck:
                deck.extend(rng.sample(pool, len(pool)))
            return deck.pop()
        for (cols, rows), count in counts.items():
            for g in tqdm(range(count), desc=f"[2] {split} {cols}x{rows}"):
                tiles, ids = [], set()
                while len(tiles) < cols * rows:                      # no Object ID twice in one grid
                    r = draw()
                    if r["class_name"] not in ids:
                        tiles.append(r); ids.add(r["class_name"])
                W, H = cols * IMG, rows * IMG
                canvas, lines = Image.new("RGB", (W, H)), []
                for i, r in enumerate(tiles):
                    ox, oy = (i % cols) * IMG, (i // cols) * IMG
                    im, box = Image.open(r["path"]).convert("RGB"), (r["x1"], r["y1"], r["x2"], r["y2"])
                    if split == "train":
                        im, box = augment_tile(im, box, rng)
                    canvas.paste(im, (ox, oy))
                    lines.append(yolo_line(cid[r["class_name"]], box[0] + ox, box[1] + oy, box[2] + ox, box[3] + oy, W, H))
                name = f"{split}_grid{cols}x{rows}_{g:04d}"
                canvas.save(GRID_DIR / "images" / split / f"{name}.jpg", quality=92)
                (GRID_DIR / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + "\n")
                manifest.append({"image": name, "split": split, "grid": f"{cols}x{rows}", "n_objects": len(tiles),
                                 "sources": "|".join(t["path"] for t in tiles),
                                 "object_ids": " ".join(t["class_name"] for t in tiles)})
    manifest = pd.DataFrame(manifest)
    manifest.to_csv(GRID_DIR / "manifest.csv", index=False)

    # leakage check (TA: no source image may appear in more than one split)
    seen = {}
    for _, r in manifest.iterrows():
        for p in r.sources.split("|"):
            seen.setdefault(p, set()).add(r.split)
    leaks = [p for p, s in seen.items() if len(s) > 1]
    assert not leaks, f"{len(leaks)} source photos appear in more than one split"
    print("[2] leakage check passed: every source photo is used in exactly one split")

    yaml_path.write_text("\n".join([f"path: {GRID_DIR}", "train: images/train", "val: images/val", "test: images/test",
                                    "names:"] + [f"  {i}: {c}" for i, c in enumerate(classes)]) + "\n")
    summary = manifest.groupby(["split", "grid"]).agg(images=("image", "size"), objects=("n_objects", "sum")).unstack(0)
    print("[2] grid images and objects per split:\n" + summary.to_string())
    summary.to_csv(RESULTS / "grid_summary.csv")
    pd.DataFrame({"class_id": range(len(classes)), "object_id": classes,
                  "object_name": [OBJECT_NAMES[c] for c in classes]}).to_csv(RESULTS / "class_ids.csv", index=False)

    # preview: one grid of each size with its labels drawn on
    picks = [manifest[(manifest.split == "train") & (manifest.grid == g)].image.iloc[0] for g in manifest.grid.unique()]
    for name in picks:
        im = Image.open(GRID_DIR / "images" / "train" / f"{name}.jpg").convert("RGB"); d = ImageDraw.Draw(im)
        for c, x1, y1, x2, y2 in read_yolo_labels(GRID_DIR / "labels" / "train" / f"{name}.txt", *im.size):
            d.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3); d.text((x1 + 4, y1 + 3), classes[c], fill=(0, 255, 0))
        im.save(RESULTS / f"preview_labels_{name}.jpg", quality=90)
    print(f"[2] dataset ready: {yaml_path} | label previews in {RESULTS}")
    return yaml_path, classes


# =============================================================================
# STEP 3: train YOLO11s
# =============================================================================
def step3_train(yaml_path):
    from ultralytics import YOLO
    run = RUNS_DIR / "yolo11s"
    best, last, done = run / "weights" / "best.pt", run / "weights" / "last.pt", run / "DONE"
    if done.exists() and best.exists():
        print(f"[3] already trained: {best}")
        return best
    device = pick_device(TRAIN_DEVICE)
    print(f"[3] training {MODEL} | imgsz {IMGSZ} | batch {BATCH} | up to {EPOCHS} epochs | device {device}")
    t0 = time.time()
    if last.exists():
        print("[3] resuming from", last)
        YOLO(str(last)).train(resume=True)
    else:
        YOLO(MODEL).train(data=str(yaml_path), imgsz=IMGSZ, epochs=EPOCHS, patience=PATIENCE, batch=BATCH,
                          device=device, workers=0, seed=SEED, deterministic=True, cos_lr=True,
                          mosaic=0.0,                       # grids are already multi-object; keep tiles whole
                          fliplr=0.5, scale=0.3, translate=0.1, hsv_h=0.015, hsv_s=0.5, hsv_v=0.3,
                          project=str(RUNS_DIR), name="yolo11s", exist_ok=True, plots=True, verbose=True)
    done.write_text(json.dumps({"finished": time.ctime(), "minutes_this_session": round((time.time() - t0) / 60, 1)}))
    print(f"[3] training done in {(time.time() - t0) / 60:.1f} min -> {best}")
    return best


# =============================================================================
# STEP 4: test-set metrics
# =============================================================================
def step4_test(best, yaml_path, classes):
    from ultralytics import YOLO
    model, device = YOLO(str(best)), pick_device(TRAIN_DEVICE)
    m = model.val(data=str(yaml_path), split="test", imgsz=IMGSZ, batch=BATCH, device=device, workers=0,
                  project=str(RUNS_DIR), name="test_eval", exist_ok=True, plots=True, verbose=False)
    overall = {"model": MODEL, "subset": SUBSET, "object_ids": len(classes), "imgsz": IMGSZ,
               "test_mAP50": round(float(m.box.map50), 4), "test_mAP50_95": round(float(m.box.map), 4),
               "test_precision": round(float(m.box.mp), 4), "test_recall": round(float(m.box.mr), 4)}

    per_class = pd.DataFrame({"object_id": [classes[i] for i in m.box.ap_class_index],
                              "AP50": np.round(m.box.ap50, 4), "AP50_95": np.round(m.box.ap, 4)})
    per_class.insert(1, "object_name", per_class.object_id.map(OBJECT_NAMES))
    per_class.sort_values("AP50").to_csv(RESULTS / "per_class_AP_test.csv", index=False)

    # mAP per grid size: do bigger grids (smaller objects) get harder?
    rows = []
    for g in sorted({p.stem.split("_")[1][4:] for p in (GRID_DIR / "images" / "test").glob("*.jpg")}):
        sub = RUNS_DIR / "test_by_grid" / g
        shutil.rmtree(sub, ignore_errors=True); (sub / "images").mkdir(parents=True); (sub / "labels").mkdir(parents=True)
        for p in (GRID_DIR / "images" / "test").glob(f"test_grid{g}_*.jpg"):
            shutil.copy(p, sub / "images" / p.name)
            shutil.copy(GRID_DIR / "labels" / "test" / f"{p.stem}.txt", sub / "labels" / f"{p.stem}.txt")
        (sub / "data.yaml").write_text(f"path: {sub}\ntrain: images\nval: images\nnames:\n" +
                                       "".join(f"  {i}: {c}\n" for i, c in enumerate(classes)))
        mg = model.val(data=str(sub / "data.yaml"), split="val", imgsz=IMGSZ, batch=BATCH, device=device, workers=0,
                       plots=False, verbose=False, project=str(RUNS_DIR), name=f"test_grid_{g}", exist_ok=True)
        rows.append({"grid": g, "mAP50": round(float(mg.box.map50), 4), "mAP50_95": round(float(mg.box.map), 4),
                     "precision": round(float(mg.box.mp), 4), "recall": round(float(mg.box.mr), 4)})
    by_grid = pd.DataFrame(rows); by_grid.to_csv(RESULTS / "test_mAP_by_grid.csv", index=False)

    (RESULTS / "test_metrics.json").write_text(json.dumps(overall, indent=2))
    print("\n[4] ===== TEST RESULTS =====")
    for k, v in overall.items():
        print(f"    {k:15s} {v}")
    print("[4] by grid size:\n" + by_grid.to_string(index=False))
    print("[4] weakest Object IDs (AP@0.5):\n" + per_class.sort_values("AP50").head(8).to_string(index=False))
    return overall


# =============================================================================
# STEP 5: example detections -> Object ID + location of every object
# =============================================================================
def detect_objects(model, image_path, classes):
    """The required output: Object ID and location (box in pixels) of every object in an image."""
    r = model.predict(str(image_path), imgsz=IMGSZ, conf=CONF, verbose=False, device=pick_device(TRAIN_DEVICE))[0]
    dets = [{"object_id": classes[int(c)], "object_name": OBJECT_NAMES[classes[int(c)]], "confidence": round(float(s), 3),
             "x1": round(float(b[0])), "y1": round(float(b[1])), "x2": round(float(b[2])), "y2": round(float(b[3]))}
            for c, s, b in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy)]
    return pd.DataFrame(dets).sort_values(["y1", "x1"]) if dets else pd.DataFrame(), r

def step5_demo(best, classes, per_grid=1):
    from ultralytics import YOLO
    model = YOLO(str(best))
    rng = random.Random(SEED)
    test_imgs = sorted((GRID_DIR / "images" / "test").glob("*.jpg"))
    picks = []
    for g in sorted({p.stem.split("_")[1] for p in test_imgs}):
        picks += rng.sample([p for p in test_imgs if f"_{g}_" in p.name], per_grid)
    for p in picks:
        dets, r = detect_objects(model, p, classes)
        truth = read_yolo_labels(GRID_DIR / "labels" / "test" / f"{p.stem}.txt", *Image.open(p).size)
        pred_img = Image.fromarray(r.plot()[:, :, ::-1])                      # model prediction
        true_img = Image.open(p).convert("RGB"); d = ImageDraw.Draw(true_img)  # true labels
        for c, x1, y1, x2, y2 in truth:
            d.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3); d.text((x1 + 4, y1 + 3), classes[c], fill=(0, 255, 0))
        both = Image.new("RGB", (pred_img.width * 2 + 20, pred_img.height), "white")
        both.paste(pred_img, (0, 0)); both.paste(true_img, (pred_img.width + 20, 0))
        both.save(RESULTS / f"detection_{p.stem}.jpg", quality=90)
        dets.to_csv(RESULTS / f"detection_{p.stem}.csv", index=False)
        found = set(dets.object_id) if len(dets) else set()
        true_ids = {classes[t[0]] for t in truth}
        print(f"\n[5] {p.name}: {len(dets)} objects detected | {len(true_ids & found)}/{len(true_ids)} true Object IDs found")
        if len(dets):
            print(dets.to_string(index=False))
    print(f"\n[5] example detections saved in {RESULTS} (left = prediction, right = true labels)")


# =============================================================================
if __name__ == "__main__":
    t_start = time.time()
    random.seed(SEED); np.random.seed(SEED)
    labels = step1_labels()
    yaml_path, classes = step2_grids(labels)
    best = step3_train(yaml_path)
    step4_test(best, yaml_path, classes)
    step5_demo(best, classes)
    print(f"\nALL DONE in {(time.time() - t_start) / 60:.1f} min. Results for the report are in {RESULTS}")