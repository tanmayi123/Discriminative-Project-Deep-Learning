# =============================================================================
# Discriminative Project - Milestone 1 (Mukul)
# 6 CNN models not used by other group members:
#   G AlexNet | H ResNet-101 | I GoogLeNet | J MNASNet-1.0 | K ShuffleNetV2 x1.5 | L SqueezeNet 1.1
# Final merged dataset: 73 Object IDs, stratified 70/15/15 split (seed 42).
# =============================================================================
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"   # if an operation isn't supported on the Mac GPU, use the CPU for it

import re, hashlib, random, time, json, math
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from PIL import Image, ImageOps
from tqdm.auto import tqdm
import torch, torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from torchvision import transforms as T, models
from torch.utils.data import Dataset, DataLoader


# ===================== PART 1: settings + data =====================
PROJECT = Path("/Applications/Documents/Sem 4/Deep Learning/Project")

hits = sorted(PROJECT.rglob("images_OBJ001"))
if not hits:
    raise FileNotFoundError("No images_OBJ001 folder found inside Project. Unzip the dataset zip first.")
DATA_DIR = hits[0].parent
print("Dataset folder:", DATA_DIR)

OUT_DIR = PROJECT / "mukul_outputs"; OUT_DIR.mkdir(exist_ok=True)
SEED, IMG_SIZE = 42, 224

def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
set_seed()

device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu")
print("PyTorch", torch.__version__, "| device:", device)

# ---- index: one folder per Object ID; the label comes from the folder name ----
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
class_dirs  = sorted(d for d in DATA_DIR.iterdir() if d.is_dir() and not d.name.startswith((".", "__")))
class_names = [re.search(r"OBJ\d+", d.name).group(0) for d in class_dirs]   # "images_OBJ046" -> "OBJ046"
NUM_CLASSES = len(class_names)

rows = []
for label, (name, folder) in enumerate(zip(class_names, class_dirs)):
    for f in sorted(folder.iterdir()):
        if f.suffix.lower() in IMG_EXTS and not f.name.startswith("."):     # skips .DS_Store etc.
            rows.append({"path": str(f), "fname": f.name, "class_name": name, "label": label})
df = pd.DataFrame(rows)
counts = df.class_name.value_counts()
print(f"{NUM_CLASSES} classes | {len(df)} images | per class: min {counts.min()}, max {counts.max()}")

# ---- load every image once: fix rotation, convert to RGB, resize to 224x224 (cached) ----
Image.MAX_IMAGE_PIXELS = None                                  # some course photos are huge
BICUBIC = getattr(Image, "Resampling", Image).BICUBIC

def load_image(path):
    with Image.open(path) as im:
        im.draft("RGB", (IMG_SIZE * 2, IMG_SIZE * 2))          # fast decode for huge JPEGs
        im = ImageOps.exif_transpose(im).convert("RGB")        # fix phone rotation, force RGB
        arr = np.asarray(im.resize((IMG_SIZE, IMG_SIZE), BICUBIC), dtype=np.uint8)
    md5 = hashlib.md5(Path(path).read_bytes()).hexdigest()     # fingerprint for the duplicate check
    return arr, md5

img_cache, md5_cache = OUT_DIR / "images_224.npy", OUT_DIR / "md5.csv"
if img_cache.exists() and md5_cache.exists():
    IMAGES = np.load(img_cache)
    df["md5"] = pd.read_csv(md5_cache).md5.values
    print("Loaded images from cache")
else:
    results = [load_image(p) for p in tqdm(df.path, desc="loading")]
    IMAGES  = np.stack([r[0] for r in results])
    df["md5"] = [r[1] for r in results]
    np.save(img_cache, IMAGES); df[["md5"]].to_csv(md5_cache, index=False)
    print("Images loaded and cached")
print("IMAGES array:", IMAGES.shape, f"({IMAGES.nbytes / 1e9:.2f} GB)")

# ---- remove background-only photos and exact duplicates (before the split) ----
BG_PATTERN = re.compile(r"_bg_|bg\d+", re.IGNORECASE)        # matches _bg_001 and BG001 names
df["is_bg"] = df.fname.apply(lambda f: bool(BG_PATTERN.search(f)))

df["is_dup"] = False                                          # keep the first copy of each duplicate
for _, group in df[~df.is_bg & df.duplicated("md5", keep=False)].groupby("md5"):
    df.loc[group.sort_values("fname").index[1:], "is_dup"] = True

keep  = (~df.is_bg & ~df.is_dup).values
clean = df[keep].reset_index(drop=True)
X     = IMAGES[keep]

df[df.is_bg][["fname", "class_name"]].to_csv(OUT_DIR / "excluded_background.csv", index=False)
df[df.is_dup][["fname", "class_name", "md5"]].to_csv(OUT_DIR / "removed_duplicates.csv", index=False)
print(f"background removed: {df.is_bg.sum()} | duplicates removed: {df.is_dup.sum()} | kept: {len(clean)}")

# ---- stratified 70/15/15 split ----
idx = np.arange(len(clean))
train_idx, tmp_idx = train_test_split(idx, test_size=0.30, stratify=clean.label, random_state=SEED)
val_idx, test_idx  = train_test_split(tmp_idx, test_size=0.50, stratify=clean.label.values[tmp_idx],
                                      random_state=SEED)
clean["split"] = ""
clean.loc[train_idx, "split"] = "train"
clean.loc[val_idx, "split"]   = "val"
clean.loc[test_idx, "split"]  = "test"
clean[["fname", "class_name", "label", "split"]].to_csv(OUT_DIR / "split.csv", index=False)
print(f"train {len(train_idx)} | val {len(val_idx)} | test {len(test_idx)}")


# ===================== PART 2: augmentation + data loaders =====================
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]     # ImageNet statistics (pretrained models expect these)

class RandomLowRes:
    """Simulate a low-quality photo: shrink the image, then scale it back up (loses detail)."""
    def __init__(self, p=0.3, scale=(0.25, 0.6)):
        self.p, self.scale = p, scale
    def __call__(self, img):
        if random.random() >= self.p:
            return img
        s = random.uniform(*self.scale)
        w, h = img.size
        small = img.resize((max(8, int(w * s)), max(8, int(h * s))), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)

# augmentation targeted at student photos: cut-off views, lighting, low resolution, blur, occlusion
train_tf = T.Compose([
    T.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3),
    RandomLowRes(p=0.3),
    T.RandomApply([T.GaussianBlur(5, sigma=(0.5, 2.0))], p=0.3),
    T.ToTensor(), T.Normalize(MEAN, STD),
    T.RandomErasing(p=0.25),
])
eval_tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])  # val/test: no randomness

class ImageArrayDataset(Dataset):
    def __init__(self, images, labels, tf):
        self.images, self.labels, self.tf = images, labels, tf
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, i):
        return self.tf(Image.fromarray(self.images[i])), int(self.labels[i])

y = clean.label.values
BATCH = 32
train_loader = DataLoader(ImageArrayDataset(X[train_idx], y[train_idx], train_tf), batch_size=BATCH, shuffle=True)
val_loader   = DataLoader(ImageArrayDataset(X[val_idx],   y[val_idx],   eval_tf),  batch_size=BATCH)
test_loader  = DataLoader(ImageArrayDataset(X[test_idx],  y[test_idx],  eval_tf),  batch_size=BATCH)
print("batches per epoch:", len(train_loader))


# ===================== PART 3: the 6 models =====================
def build_model(name):
    """ImageNet-pretrained model with its final layer replaced by 73 outputs.
    Returns (model, name of the new final layer)."""
    n = NUM_CLASSES
    if name == "G_alexnet":
        m = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        m.classifier[6] = nn.Linear(m.classifier[6].in_features, n); return m, "classifier.6"
    if name == "H_resnet101":
        m = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
        m.fc = nn.Linear(m.fc.in_features, n); return m, "fc"
    if name == "I_googlenet":
        m = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, n); return m, "fc"
    if name == "J_mnasnet":
        m = models.mnasnet1_0(weights=models.MNASNet1_0_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n); return m, "classifier.1"
    if name == "K_shufflenetv2":
        m = models.shufflenet_v2_x1_5(weights=models.ShuffleNet_V2_X1_5_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, n); return m, "fc"
    if name == "L_squeezenet":
        m = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Conv2d(512, n, kernel_size=1)    # SqueezeNet's final layer is a 1x1 convolution
        return m, "classifier.1"
    raise ValueError(name)

MODEL_NAMES = ["G_alexnet", "H_resnet101", "I_googlenet", "J_mnasnet", "K_shufflenetv2", "L_squeezenet"]

# training settings: all fully fine-tuned; small/efficient models get a higher backbone learning rate
CONFIG = {
    "G_alexnet":      dict(epochs=15, patience=5, lr_head=1e-3, lr_backbone=1e-4, wd=0.01),
    "H_resnet101":    dict(epochs=15, patience=5, lr_head=1e-3, lr_backbone=1e-4, wd=0.01),
    "I_googlenet":    dict(epochs=15, patience=5, lr_head=1e-3, lr_backbone=3e-4, wd=0.01),
    "J_mnasnet":      dict(epochs=15, patience=5, lr_head=1e-3, lr_backbone=3e-4, wd=0.01),
    "K_shufflenetv2": dict(epochs=15, patience=5, lr_head=1e-3, lr_backbone=3e-4, wd=0.01),
    "L_squeezenet":   dict(epochs=15, patience=5, lr_head=1e-3, lr_backbone=3e-4, wd=0.01),
}


# ===================== PART 4: training engine =====================
EXP_DIR = OUT_DIR / "experiments"; EXP_DIR.mkdir(exist_ok=True)

# class-weighted loss: classes with fewer training images count a bit more (no images dropped to balance)
train_counts = np.bincount(y[train_idx], minlength=NUM_CLASSES)
class_w   = torch.tensor(train_counts.sum() / (NUM_CLASSES * train_counts), dtype=torch.float32, device=device)
criterion = nn.CrossEntropyLoss(weight=class_w, label_smoothing=0.1)

@torch.no_grad()
def predict(model, loader):
    """Softmax probabilities for every image in the loader."""
    model.eval(); probs = []
    for xb_, _ in loader:
        probs.append(model(xb_.to(device)).softmax(1).cpu())
    return torch.cat(probs).numpy()

def make_optimizer(model, head, cfg):
    """The new final layer learns faster than the pretrained layers (discriminative learning rates)."""
    head_p = [p for n_, p in model.named_parameters() if p.requires_grad and n_.startswith(head)]
    body_p = [p for n_, p in model.named_parameters() if p.requires_grad and not n_.startswith(head)]
    return torch.optim.AdamW([{"params": head_p, "lr": cfg["lr_head"]},
                              {"params": body_p, "lr": cfg["lr_backbone"]}], weight_decay=cfg["wd"])

def train_one(name):
    """Train one model with early stopping; keep the best epoch by validation macro-F1; test it once.
    Restart-safe: finished models are skipped and interrupted ones resume from their last epoch."""
    cfg, d = CONFIG[name], EXP_DIR / name
    d.mkdir(exist_ok=True)
    if (d / "results.json").exists():
        res = json.loads((d / "results.json").read_text())
        print(f"✓ {name} already finished, skipped (test acc {res['test_acc']:.4f})")
        return res

    set_seed()
    model, head = build_model(name)
    model = model.to(device)
    opt = make_optimizer(model, head, cfg)
    steps, warm = cfg["epochs"] * len(train_loader), len(train_loader)          # 1 warm-up epoch, then cosine decay
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm
                                              else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))

    start_ep, best_f1, wait, history, train_sec = 1, -1.0, 0, [], 0.0
    ckpt = d / "checkpoint.pt"
    if ckpt.exists():                                                           # resume after an interruption
        c = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(c["model"]); opt.load_state_dict(c["opt"]); sched.load_state_dict(c["sched"])
        start_ep, best_f1, wait, history, train_sec = c["epoch"] + 1, c["best_f1"], c["wait"], c["history"], c["train_sec"]
        print(f"↻ {name}: resuming at epoch {start_ep}")

    for ep in range(start_ep, cfg["epochs"] + 1):
        if wait >= cfg["patience"]:
            print(f"  early stop: no improvement for {cfg['patience']} epochs"); break
        model.train(); t0 = time.time(); tot = correct = n = 0
        for xb_, yb_ in train_loader:
            xb_, yb_ = xb_.to(device), yb_.to(device)
            out = model(xb_); loss = criterion(out, yb_)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)                  # keeps training stable
            opt.step(); sched.step()
            tot += loss.item() * len(yb_); correct += (out.argmax(1) == yb_).sum().item(); n += len(yb_)

        pv = predict(model, val_loader); yv = y[val_idx]
        v_acc = float(accuracy_score(yv, pv.argmax(1)))
        v_f1  = float(f1_score(yv, pv.argmax(1), average="macro"))
        sec = time.time() - t0; train_sec += sec
        history.append({"epoch": ep, "train_loss": tot / n, "train_acc": correct / n,
                        "val_acc": v_acc, "val_f1": v_f1, "sec": sec})
        flag = ""
        if v_f1 > best_f1:
            best_f1, wait, flag = v_f1, 0, "  <- best"
            torch.save(model.state_dict(), d / "best.pt")
        else:
            wait += 1
        print(f"[{name}] ep {ep:2d}/{cfg['epochs']} | loss {tot/n:.3f} | train {correct/n:.3f} | "
              f"val acc {v_acc:.3f} F1 {v_f1:.3f} | {sec/60:.1f} min{flag}")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": ep, "best_f1": best_f1, "wait": wait, "history": history, "train_sec": train_sec}, ckpt)

    # test the BEST epoch's model, once
    model.load_state_dict(torch.load(d / "best.pt", map_location="cpu"))
    pt = predict(model, test_loader); yt = y[test_idx]
    res = {"model": name, "best_val_f1": best_f1,
           "test_acc": float(accuracy_score(yt, pt.argmax(1))),
           "test_macro_f1": float(f1_score(yt, pt.argmax(1), average="macro")),
           "test_errors": int((pt.argmax(1) != yt).sum()),
           "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
           "epochs_run": len(history), "train_min": round(train_sec / 60, 1)}
    np.save(d / "test_probs.npy", pt)
    pd.DataFrame(history).to_csv(d / "history.csv", index=False)
    (d / "results.json").write_text(json.dumps(res, indent=2))
    ckpt.unlink()
    print(f"→ {name}: test acc {res['test_acc']:.4f} | macro-F1 {res['test_macro_f1']:.4f} | "
          f"{res['test_errors']} errors | {res['train_min']} min")
    del model, opt
    if device.type == "mps": torch.mps.empty_cache()
    return res


# ===================== PART 5: train all 6 models + results table =====================
all_results = []
for name in MODEL_NAMES:
    print(f"\n========== {name} ==========", flush=True)
    all_results.append(train_one(name))

summary = pd.DataFrame(all_results)

# training time from the per-epoch log, ignoring any epoch interrupted by the laptop sleeping
# (an epoch more than 10x the model's median epoch time is treated as an interruption)
def corrected_minutes(name):
    sec = pd.read_csv(EXP_DIR / name / "history.csv").sec
    return round(sec[sec < 10 * sec.median()].sum() / 60, 1)
summary["train_min"] = [corrected_minutes(n) for n in summary.model]

summary = summary.sort_values("best_val_f1", ascending=False)
summary.to_csv(OUT_DIR / "my_6_models.csv", index=False)
print("\n===== FINAL RESULTS: my 6 models (sorted by validation macro-F1) =====")
print(summary.to_string(index=False))