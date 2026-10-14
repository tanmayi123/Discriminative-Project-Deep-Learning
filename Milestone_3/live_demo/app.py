"""
The EYE, live upload edition: the Milestone 3 detector (yolo26s_boost) on any image you upload.

Same inference and post-processing as The EYE v3 (m2_eye_app_v3.py) for uploads:
  * the image is scaled by 1280/1120 and, if it fits, pasted on a grey 1280-px canvas (as in training)
  * grids (width and height multiples of 224 px): threshold 0.10, one box per 224-px cell
  * scenes (anything else): threshold 0.35, then class-agnostic duplicate removal at IoU 0.5
Both thresholds were chosen on the validation set (selection.json); nothing here was tuned on test data.
Extra, clearly labelled mode "One object per photo": the photo is resized to 224 x 224 exactly like the
dataset photos and treated as a 1x1 grid (the "singles" test set: recall 0.951).

Runs on a Hugging Face ZeroGPU Space, or locally:  pip install -r requirements.txt  then  python app.py
"""
import json, time
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

try:
    import spaces                                   # present on Hugging Face Spaces (ZeroGPU)
    gpu = spaces.GPU(duration=30)
except Exception:                                   # running locally: no-op decorator
    def gpu(fn): return fn

HERE = Path(__file__).resolve().parent
WEIGHTS = HERE / "best.pt"                          # yolo26s_boost weights (M2 extra/boost/runs/boost/weights/best.pt)
NAMES = json.loads((HERE / "object_names.json").read_text(encoding="utf-8"))

TILE, NOMINAL, IMGSZ = 224, 1120, 1280               # photo cell, nominal grid scale, the selected model's input size
CONF_FLOOR = 0.01
T_GRID, T_SCENE, SCENE_NMS = 0.10, 0.35, 0.50         # validated (selected_model.json)
MAX_SIDE = 3000                                     # uploads larger than this are scaled down first (as in The EYE)

MODES = ["Auto (as in The EYE)", "Grid of 224-px photos", "Scene (several objects, any layout)", "One object per photo"]
LOOKALIKE = {"watches": ["OBJ025", "OBJ046", "OBJ065"], "wallets": ["OBJ061", "OBJ067"], "power banks": ["OBJ048", "OBJ057"],
             "rackets": ["OBJ030", "OBJ034", "OBJ035", "OBJ049"], "sunglasses / goggles": ["OBJ014", "OBJ045", "OBJ043"],
             "brush / comb": ["OBJ050", "OBJ059"], "bottles": ["OBJ002", "OBJ015", "OBJ063"], "projectors": ["OBJ064", "OBJ070"],
             "plush toys": ["OBJ021", "OBJ024", "OBJ071"], "notebook / keyboard": ["OBJ003", "OBJ012"]}
GROUP_OF = {o: g for g, ids in LOOKALIKE.items() for o in ids}

_model = None
def model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO(str(WEIGHTS))
    return _model
if WEIGHTS.exists(): model()                        # load once at start-up (ZeroGPU recommends module-level loading)


# ------------------------------------------------------------------------------------------ geometry (as in The EYE v3)
def iou(a, b):
    iw, ih = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0: return 0.0
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - iw * ih
    return iw * ih / u if u > 0 else 0.0

def cell_of(b): return (int(((b[0] + b[2]) / 2) // TILE), int(((b[1] + b[3]) / 2) // TILE))

def one_per_cell(preds):
    best = {}
    for p in sorted(preds, key=lambda q: -q[1]): best.setdefault(cell_of(p[2]), p)
    return list(best.values())

def agnostic_nms(preds, thr):
    keep = []
    for p in sorted(preds, key=lambda q: -q[1]):
        if all(iou(p[2], k[2]) < thr for k in keep): keep.append(p)
    return keep

def is_grid(img): return img.width % TILE == 0 and img.height % TILE == 0

def postprocess(raw, kind, conf):
    kept = [p for p in raw if p[1] >= conf]
    return one_per_cell(kept) if kind == "grid" else agnostic_nms(kept, SCENE_NMS)


# ------------------------------------------------------------------------------------------ inference
@gpu
def run_model(img):
    """All boxes down to CONF_FLOOR, in the image's own pixels (The EYE v3 run_model, native mode, imgsz 1280)."""
    import torch
    W0, H0 = img.size; s = IMGSZ / NOMINAL
    big = img.resize((max(1, round(W0 * s)), max(1, round(H0 * s))), Image.Resampling.BICUBIC)
    W, H = big.size; padded = W <= IMGSZ and H <= IMGSZ
    inp = big
    if padded:
        inp = Image.new("RGB", (IMGSZ, IMGSZ), (128, 128, 128)); inp.paste(big, (0, 0))
    m = model(); dev = 0 if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()
    r = m.predict(inp, imgsz=IMGSZ, conf=CONF_FLOOR, max_det=100, device=dev, verbose=False)[0]
    ms = (time.perf_counter() - t0) * 1000
    out = []
    for c, p, b in zip(r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.xyxy.cpu().numpy()):
        x1, y1, x2, y2 = map(float, b)
        if padded:
            if (x1 + x2) / 2 > W or (y1 + y2) / 2 > H: continue          # a box centred on the grey padding
            x2, y2 = min(x2, W), min(y2, H)
        out.append((str(m.names[int(c)]), float(p), (x1 / s, y1 / s, x2 / s, y2 / s)))
    return out, ms


# ------------------------------------------------------------------------------------------ drawing
PALETTE = [(57, 255, 136), (34, 228, 255), (255, 176, 32), (255, 79, 216), (255, 93, 93), (160, 140, 255), (255, 240, 90)]
def font(size):
    try: return ImageFont.load_default(size=size)
    except TypeError: return ImageFont.load_default()

def draw(img, preds):
    out = img.copy(); d = ImageDraw.Draw(out); lw = max(2, round(max(img.size) / 400)); f = font(max(12, round(max(img.size) / 55)))
    for k, (o, p, b) in enumerate(preds):
        col = PALETTE[int(o[3:]) % len(PALETTE)] if o[3:].isdigit() else PALETTE[k % len(PALETTE)]
        d.rectangle(b, outline=col, width=lw)
        label = f"{o} {NAMES.get(o, '')} {p:.2f}".strip()
        tb = d.textbbox((0, 0), label, font=f); tw, th = tb[2] - tb[0], tb[3] - tb[1]
        x, y = b[0], max(0, b[1] - th - 6)
        d.rectangle((x, y, x + tw + 8, y + th + 6), fill=col)
        d.text((x + 4, y + 2 - tb[1]), label, fill=(2, 19, 10), font=f)
    return out


# ------------------------------------------------------------------------------------------ the request
def detect(image, mode, threshold_choice, custom):
    if image is None: raise gr.Error("Upload an image first.")
    img = ImageOps.exif_transpose(image).convert("RGB")
    if max(img.size) > MAX_SIDE: img.thumbnail((MAX_SIDE, MAX_SIDE))
    orig = img; single = mode == MODES[3]
    if single:                                     # exactly how every dataset photo was prepared
        img = img.resize((TILE, TILE), Image.Resampling.BICUBIC); kind = "grid"
    elif mode == MODES[1]: kind = "grid"
    elif mode == MODES[2]: kind = "scene"
    else: kind = "grid" if is_grid(img) else "scene"
    if kind == "grid" and not is_grid(img):
        raise gr.Error(f"A grid must be a multiple of 224 px in both directions; this image is {img.width} x {img.height}. Choose Scene or One object per photo.")
    conf = (T_GRID if kind == "grid" else T_SCENE) if threshold_choice == "Validated" else float(custom)

    raw, ms = run_model(img)
    preds = postprocess(raw, kind, conf)
    if single:                                     # boxes back onto the uploaded photo
        sx, sy = orig.width / TILE, orig.height / TILE
        preds = [(o, p, (b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy)) for o, p, b in preds]
    preds.sort(key=lambda t: (int(t[2][1] // TILE) if kind == "grid" and not single else 0, t[2][0]))

    rows = [[o, NAMES.get(o, ""), round(p, 3), GROUP_OF.get(o, ""), *[round(v) for v in b]] for o, p, b in preds]
    how = {"grid": f"grid rule: threshold {conf:.2f}, one box per 224-px cell",
           "scene": f"scene rule: threshold {conf:.2f}, overlapping boxes of any name reduced to the most confident (IoU {SCENE_NMS})"}[kind]
    if single: how = f"one object per photo: resized to 224 x 224 like the dataset photos, then the {how}"
    summary = (f"**{len(preds)} object(s) found** in {ms:.0f} ms · {how}"
               + ("" if threshold_choice == "Validated" else " · threshold set by hand"))
    if not preds:
        summary += "\n\nNothing passed the threshold. Try *One object per photo* for a single close-up, or a lower threshold."
    return draw(orig, preds), summary, rows


INTRO = """# The EYE · live detector
Upload any image and the Milestone 3 detector (**yolo26s_boost**, Group 7, IE7615) finds the course's **73 objects** in it.
**Auto** treats images whose sides are multiples of 224 px as photo grids and everything else as a scene, exactly as The EYE does.
For an ordinary photo of a single object, **One object per photo** prepares it the way every dataset photo was prepared.
Your image is processed only to answer this request; it is not stored."""

with gr.Blocks(title="The EYE · live detector") as demo:
    gr.Markdown(INTRO)
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(type="pil", label="Image", sources=["upload", "clipboard", "webcam"])
            mode = gr.Radio(MODES, value=MODES[0], label="What kind of image is it?")
            thr = gr.Radio(["Validated", "Custom"], value="Validated", label="Threshold",
                           info=f"Validated: {T_GRID} on grids, {T_SCENE} on scenes (chosen on validation)")
            custom = gr.Slider(0.05, 0.90, value=0.25, step=0.05, label="Custom threshold (used only with Custom)")
            btn = gr.Button("Detect", variant="primary")
        with gr.Column(scale=2):
            out_img = gr.Image(label="Detections", type="pil")
            out_md = gr.Markdown()
    table = gr.Dataframe(headers=["Object ID", "Name", "Confidence", "Look-alike group", "x1", "y1", "x2", "y2"],
                         label="Every box (pixels of the uploaded image)", wrap=True)
    btn.click(detect, [inp, mode, thr, custom], [out_img, out_md, table])
    gr.Markdown("Thresholds and post-processing were fixed on the validation set before any test result (see the Milestone 3 report). "
                "Uploads have no true boxes, so the app shows what the model found but cannot mark it right or wrong.")

if __name__ == "__main__":
    demo.queue(max_size=20).launch()
