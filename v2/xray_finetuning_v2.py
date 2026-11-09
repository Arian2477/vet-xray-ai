# %% [markdown]
# <div style="background:#1B2A49;padding:34px 38px;border-radius:10px;color:#ffffff;">
# <div style="font-size:13px;letter-spacing:2px;color:#7FD3CF;text-transform:uppercase;">Veterinary AI · Thoracic Radiology · Study-Level Multi-Label Fine-Tuning</div>
# <h1 style="color:#ffffff;margin:10px 0 6px 0;font-size:32px;">VetXRay v2: Study-Aware Fine-Tuning for 11 Thoracic Findings</h1>
# <div style="font-size:15px;color:#D6E4F0;max-width:900px;line-height:1.55;">
# Same images and same tags as v1, no relabelling. The training objective, evaluation unit, resolution, regularisation and
# input handling change. Every change is accepted or rejected on validation animals only; the locked test animals are scored once.
# </div></div>
#
# ## 1 · What changes compared with v1, and why
#
# | # | Finding in v1 | Evidence | Fix in v2 |
# |---|---|---|---|
# | 1 | Other views of a positive study are trained as negatives | Only 4–60 % of multi-view positive studies have the tag on every view | Those labels are masked in the loss; tags stay untouched |
# | 2 | One film is scored alone, although a study has several views | Fusing stored v1 predictions lifts validation macro AP from 0.467 to 0.547 without training | Evaluation and inference work per study |
# | 3 | Thresholds fitted on one validation split with 1–4 positives | Rare findings: `hernia` 0/5 detected on test | Out-of-fold thresholds, 5 unsupported outputs removed, hernias pooled in `any_hernia` |
# | 4 | Findings with no reliable signal are reported as "validated" | `megaesophagus` 0/18 detected on test | Reliability comes from measured out-of-fold performance |
# | 5 | Inference accepts anything | Noise and gradients produce findings; a negative image drops macro AP from 0.49 to 0.18 | Gate model fixes polarity and rotation; out-of-distribution inputs are rejected |
# | 6 | Inference loader ignores EXIF rotation and transparency | Code review | Loader handles EXIF, alpha, 16-bit and uniform borders |
# | 7 | Optuna tuned 6-epoch trials, so the tuned model lost to the untuned one (0.455 vs 0.467) | v1 selection file | Greedy ladder with the full training budget |
# | 8 | Augmentation never saw compressed or annotated films | Annotated web-like films drop macro AP to 0.41 | JPEG, resize, margins, marks, crops, photo look; lateral films may flip |
# | 9 | 384 px from a 512 px cache, native films are 1,000–3,200 px | Small findings get a few pixels | 512 and 768 px are tested |
#
# **Checked and clean in v1:** no near-duplicate images across animals or partitions, no animal crossing partitions, DICOM polarity handling.
#
# **Honest limits.** Abdominal hernia is not in this dataset, so no setting here can teach it. `any_hernia` has only 41 images and stays experimental.
# The test animals were scored for v1 models, so they are not perfectly fresh, and an external clinic is still needed for a real-world number.

# %% [markdown]
# ## 2 · Configuration
#
# `VXR_MODE=SMOKE` proves every code path on a tiny subset and writes to its own folder. `VXR_MODE=FULL` runs the study. Finished runs are
# reloaded and an interrupted run resumes from its last epoch, which also fits Kaggle session limits. `VXR_STOP_BEFORE_TEST=1` stops before the test cell.

# %%
import copy, ctypes, gc, hashlib, io, json, math, os, random, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

warnings.filterwarnings("ignore", category=UserWarning)

MODE = os.environ.get("VXR_MODE", "SMOKE").upper()
STOP_BEFORE_TEST = os.environ.get("VXR_STOP_BEFORE_TEST", "0") == "1"
N_FOLDS = int(os.environ.get("VXR_FOLDS", "5"))
SEED = 42
KAGGLE = Path("/kaggle/input").exists()


def _find_kaggle(marker):
    for p in Path("/kaggle/input").rglob(marker):
        return p.parent
    return None


if KAGGLE:
    META_DIR = _find_kaggle("vetxray_metadata.csv")
    CACHE_DIRS = [p for p in [_find_kaggle("images_512.npy"), _find_kaggle("images_768.npy")] if p]
    WORK = Path("/kaggle/working/vxr_v2")
    V1_DIR = _find_kaggle("test_probs_full.npz")
    PRETRAINED_DIR = WORK / "pretrained"
else:
    META_DIR = Path(r"G:\VetXRay_cache\kaggle_dataset")
    CACHE_DIRS = [META_DIR, Path(r"G:\Vet_XRay_AI\v2\work\cache")]
    WORK = Path(r"G:\Vet_XRay_AI\v2\work")
    V1_DIR = Path(r"G:\VetXRay_cache\runs_full\hieradet_small")
    PRETRAINED_DIR = Path(r"G:\VetXRay_cache\pretrained")
RUN_ROOT = WORK / ("runs_full" if MODE == "FULL" else "runs_smoke")
FIG_DIR = RUN_ROOT / "figures"
for d in (WORK, RUN_ROOT, FIG_DIR, WORK / "hf_cache", WORK / "cwd"):
    d.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(WORK / "hf_cache"))
os.chdir(WORK / "cwd")   # birder creates ./models at import time
EXPORT = (Path("/kaggle/working") if KAGGLE else Path(r"G:\Vet_XRay_AI\v2")) / "checkpoints" / ("petai_xray_v2.pt" if MODE == "FULL" else "petai_xray_v2_smoke.pt")
EXPORT.parent.mkdir(parents=True, exist_ok=True)

NAVY, TEAL, BLUE, AMBER, MAGENTA, GREY = "#1B2A49", "#1A9E9A", "#2E5FA8", "#C9821A", "#B5446E", "#A3ADB8"
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "font.size": 10.5, "axes.titlesize": 13, "axes.titleweight": "bold",
                     "axes.titlecolor": NAVY, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#E6EAEE", "legend.frameon": False, "figure.facecolor": "white"})

if os.name == "nt":   # a long run must survive idle sleep; a closed lid still sleeps
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


def seed_all(seed=SEED):
    random.seed(seed); np.random.seed(seed); os.environ["PYTHONHASHSEED"] = str(seed)
    import torch
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def show(name):
    plt.tight_layout(); plt.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight")
    plt.show() if "ipykernel" in sys.modules else None
    plt.close()


def table(df, title=None, digits=3):
    if title:
        print(f"\n{title}")
    if "ipykernel" in sys.modules:
        from IPython.display import display
        display(df.round(digits))
    else:
        print(df.round(digits).to_string())


print(f"MODE={MODE} | KAGGLE={KAGGLE} | WORK={WORK} | folds={N_FOLDS} | stop before test={STOP_BEFORE_TEST}")

# %% [markdown]
# ## 3 · Data: same images, same tags, study-aware tables
#
# Outputs are the 10 findings with at least 70 positive images plus `any_hernia`, the union of the two hernia tags. Images of the five dropped
# findings stay in training. A **study** is one animal on one date. A study is positive for a finding when any of its views carries the tag.

# %%
CORE = ["alveolar_pattern", "bronchial_pattern", "cardiomegaly", "foreign_body", "interstitial_pattern", "mass", "megaesophagus",
        "pleural_effusion", "pleural_mineralization", "pneumothorax"]
LABELS = CORE + ["any_hernia"]
HERNIA_TAGS = {"hernia", "diaphragmatic_hernia"}
HERNIA = LABELS.index("any_hernia")
CORE_IDX = [LABELS.index(l) for l in CORE]


def tagset(s):
    return {t.strip() for t in str(s).split("|") if t.strip()}


df = pd.read_csv(META_DIR / "vetxray_metadata.csv", dtype={"StudyDate": str})
idx = pd.read_csv(next(d / "cache_index.csv" for d in CACHE_DIRS if (d / "cache_index.csv").exists()))
df = df.merge(idx, on="FileName", validate="one_to_one")
assert df.groupby("group")["split"].nunique().max() == 1, "an animal crosses partitions"
df["tagset"] = df["tags"].apply(tagset)
df["study"] = df["group"].astype(str) + "_" + df["StudyDate"].fillna("NA")
Y = np.array([[l in t for l in CORE] + [bool(t & HERNIA_TAGS)] for t in df["tagset"]], dtype=np.float32)


def stratified_group_folds(groups, strat, k, seed=SEED):
    """Group-level iterative stratification (Sechidis et al., 2011), as in v1, for k equal folds."""
    rng = np.random.default_rng(seed)
    ids, inv = np.unique(groups, return_inverse=True)
    G = np.zeros((len(ids), strat.shape[1])); np.add.at(G, inv, strat)
    size = np.bincount(inv).astype(float)
    need, need_n = np.outer(np.full(k, 1 / k), G.sum(0)), np.full(k, size.sum() / k)
    free, assign = np.ones(len(ids), bool), np.full(len(ids), -1)
    perm = rng.permutation(len(ids))

    def place(g, s):
        assign[g] = s; need[s] -= G[g]; need_n[s] -= size[g]; free[g] = False

    while free.any():
        rem = G[free].sum(0)
        if not (rem > 0).any():
            for g in perm[free[perm]]:
                place(g, int(np.argmax(need_n + 1e-9 * rng.random(k))))
            break
        l = int(np.argmin(np.where(rem > 0, rem, np.inf)))
        for g in perm[free[perm] & (G[perm, l] > 0)]:
            best = np.flatnonzero(np.isclose(need[:, l], need[:, l].max()))
            place(g, int(best[np.argmax(need_n[best])]))
    return assign[inv]


pool = (df["split"] != "test").values
strat = np.concatenate([Y, (Y.sum(1, keepdims=True) == 0).astype(float), pd.get_dummies(df["species"]).values.astype(float),
                        pd.get_dummies(df["scanner"]).values.astype(float)], axis=1)
df["fold"] = -1
df.loc[pool, "fold"] = stratified_group_folds(df.loc[pool, "group"].values, strat[pool], 5)
assert df[pool].groupby("group")["fold"].nunique().max() == 1

if MODE == "SMOKE":   # a tiny subset that still holds positives; numbers are technical checks only
    rng = np.random.default_rng(SEED); keep = []
    for f in [0, 1, 2, 3, 4, -1]:
        ids = np.flatnonzero(df["fold"].values == f); pos = ids[Y[ids].sum(1) > 0]
        keep += list(rng.choice(pos, 14, replace=False)) + list(rng.choice(ids, 14, replace=False))
    keep = np.unique(keep); df, Y = df.iloc[keep].reset_index(drop=True), Y[keep]
    pool = (df["split"] != "test").values

# study-aware mask: a negative view inside a positive study is "unknown" for the loss, its tag is not changed
s_pos = pd.DataFrame(Y).groupby(df["study"].values).transform("max").values
MASK_STUDY = 1.0 - ((Y == 0) & (s_pos == 1)).astype(np.float32)

counts = pd.DataFrame({"images": Y.sum(0).astype(int), "studies+": [df.loc[Y[:, j] > 0, "study"].nunique() for j in range(len(LABELS))],
                       "animals+": [df.loc[Y[:, j] > 0, "group"].nunique() for j in range(len(LABELS))],
                       "masked negatives": ((1 - MASK_STUDY).sum(0)).astype(int)}, index=LABELS)
for s in ["train", "val", "test"]:
    counts[s] = [int(Y[(df["split"] == s).values, j].sum()) for j in range(len(LABELS))]
table(counts, f"{len(df):,} images, {df['study'].nunique():,} studies, {df['group'].nunique():,} animals")
folds_t = pd.DataFrame({f"fold {f}": Y[df["fold"].values == f].sum(0).astype(int) for f in range(5)}, index=LABELS)
folds_t.loc["images"] = [int((df["fold"] == f).sum()) for f in range(5)]
table(folds_t, "Positive images per fold (train + val animals)")

agree = []
for j, l in enumerate(LABELS):
    g = pd.DataFrame({"s": df["study"], "y": Y[:, j]}).groupby("s")["y"].agg(["sum", "size"])
    m = g[(g["size"] > 1) & (g["sum"] > 0)]
    agree.append((m["sum"] == m["size"]).mean() if len(m) else np.nan)
fig, ax = plt.subplots(figsize=(10, 4))
ax.barh(LABELS[::-1], np.array(agree)[::-1] * 100, color=TEAL)
ax.set(title="Positive multi-view studies with the tag on every view", xlabel="% of studies", xlim=(0, 100))
show("view_consistency")

# %% [markdown]
# ## 4 · Shared preprocessing and augmentation
#
# The first four functions are the same code as in `petai_xray_inference_v2.py`. `robust_aug` passes each augmented film through that same pipeline,
# so training sees what deployment will see.

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, brier_score_loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = DEVICE.type == "cuda"
torch.backends.cudnn.benchmark = True
PCT_LOW, PCT_HIGH = 0.5, 99.5


def window_percentile(arr):
    """Clip to [P0.5, P99.5] and rescale to [0, 1]."""
    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(arr[::4, ::4], [PCT_LOW, PCT_HIGH])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-6:
        lo, hi = float(arr.min()), float(arr.max()) + 1e-6
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def letterbox_uint8(img, size):
    """Resize the long side to `size` (Lanczos), pad to a square with zeros, quantise to uint8."""
    h, w = img.shape
    scale = size / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    small = np.asarray(Image.fromarray(img.astype(np.float32)).resize((nw, nh), Image.Resampling.LANCZOS))
    out = np.zeros((size, size), dtype=np.float32)
    top, left = (size - nh) // 2, (size - nw) // 2
    out[top:top + nh, left:left + nw] = np.clip(small, 0.0, 1.0)
    return np.round(out * 255.0).astype(np.uint8)


def content(u8):
    """The film without the letterbox padding."""
    ys = np.flatnonzero(u8.max(1)); xs = np.flatnonzero(u8.max(0))
    return u8[ys[0]:ys[-1] + 1, xs[0]:xs[-1] + 1] if len(ys) and len(xs) else u8


def to_model_uint8(a, size):
    return letterbox_uint8(window_percentile(a), size)


def web_resize_jpeg(a, side=None, q=None):
    side, q = side or random.randint(350, 900), q or random.randint(35, 90)
    im = Image.fromarray(a); s = min(1.0, side / max(im.size))
    im = im.resize((max(8, round(im.width * s)), max(8, round(im.height * s))), Image.Resampling.BILINEAR)
    b = io.BytesIO(); im.convert("L").save(b, "JPEG", quality=q)
    return np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("L"))


def add_margin(a, frac=None, value=255):
    return np.pad(a, max(2, int(max(a.shape) * (frac or random.uniform(0.02, 0.12)))), constant_values=value)


def add_marks(a):
    im = Image.fromarray(a).convert("L"); d = ImageDraw.Draw(im); h, w = a.shape
    r = lambda lo, hi: random.uniform(lo, hi)
    for _ in range(random.randint(1, 3)):
        x0, y0, x1, y1 = w * r(.1, .9), h * r(.1, .9), w * r(.1, .9), h * r(.1, .9)
        kind = random.random()
        if kind < .4:
            d.line([(x0, y0), (x1, y1)], fill=255, width=max(1, w // random.randint(80, 200)))
        elif kind < .7:
            d.ellipse([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)], outline=255, width=max(1, w // 150))
        else:
            d.rectangle([x0, y0, x0 + w * r(.05, .3), y0 + h * r(.02, .07)], fill=random.choice([0, 255]))
    return np.asarray(im)


def tight_crop(a, frac=None):
    f = frac if frac is not None else random.uniform(0.03, 0.14)
    h, w = a.shape; dy, dx = int(h * f), int(w * f)
    return a[dy:h - dy, dx:w - dx] if h - 2 * dy > 8 and w - 2 * dx > 8 else a


def photo_look(a, g=None, c=None, o=None, v=None):
    g, c, o, v = g or random.uniform(.8, 1.5), c or random.uniform(.65, 1.0), o if o is not None else random.uniform(0, .15), v or random.uniform(.1, .4)
    x = (a / 255.0) ** g * c + o
    h, w = a.shape; yy, xx = np.mgrid[0:h, 0:w]
    x = x * (1 - v * (((yy - h / 2) / (h / 2)) ** 2 + ((xx - w / 2) / (w / 2)) ** 2) / 2)
    return np.clip(x * 255, 0, 255).astype(np.uint8)


def robust_aug(u8, size):
    """Deployment-like damage applied to half of the training films, through the real preprocessing."""
    a = content(u8)
    if random.random() < .7: a = web_resize_jpeg(a)
    if random.random() < .15: a = photo_look(a)
    if random.random() < .2: a = tight_crop(a)
    if random.random() < .2: a = add_marks(a)
    if random.random() < .2: a = add_margin(a)
    if random.random() < .06: a = np.rot90(a, random.randint(1, 3)).copy()
    return to_model_uint8(a, size)


def gpu_augment(x, strong):
    """x: (B,1,H,W) float in [0,1]. v1 ranges when strong is False."""
    B, dev = x.shape[0], x.device
    u = lambda: torch.rand(B, device=dev) * 2 - 1
    ang, scl = u() * math.radians(10 if strong else 7), 1 + u() * (.1 if strong else .07)
    tx, ty = u() * (.1 if strong else .08), u() * (.1 if strong else .08)
    cos, sin = torch.cos(ang) / scl, torch.sin(ang) / scl
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
    x = F.grid_sample(x, F.affine_grid(theta, list(x.shape), align_corners=False), mode="bilinear", padding_mode="zeros", align_corners=False)
    v = lambda: torch.rand(B, 1, 1, 1, device=dev)
    c, b = 1 + (v() * 2 - 1) * (.2 if strong else .1), (v() * 2 - 1) * (.1 if strong else .1)
    g = (.75 + v() * .55) if strong else (.9 + v() * .2)
    x = (x.clamp(0, 1) ** g) * c + b
    if strong:
        x = x + torch.randn_like(x) * (v() * .02)
    return x.clamp(0, 1)


fig, axes = plt.subplots(1, 6, figsize=(16, 3))
_demo = None
for d in CACHE_DIRS:
    if (d / "images_512.npy").exists():
        _demo = np.asarray(np.load(d / "images_512.npy", mmap_mode="r")[int(df["row"].iloc[0])])
if _demo is not None:
    random.seed(1)
    axes[0].imshow(_demo, cmap="gray"); axes[0].set_title("original", fontsize=10)
    for k in range(1, 6):
        axes[k].imshow(robust_aug(_demo, 512), cmap="gray"); axes[k].set_title(f"augmented {k}", fontsize=10)
for a in axes: a.axis("off")
show("augmentation_preview")

# %% [markdown]
# ## 5 · Model, loss and training loop

# %%
_IMG = {}


def get_images(size):
    """Whole cache in RAM. G: is a USB disk, and random reads would dominate the epoch time. 384 px is resized from the 512 px cache, as in v1."""
    size = max(size, 512)
    if size not in _IMG:
        _IMG.clear(); gc.collect()
        path = next((d / f"images_{size}.npy" for d in CACHE_DIRS if (d / f"images_{size}.npy").exists()), None)
        if path is None:
            return None
        m = np.load(path, mmap_mode="r")
        _IMG[size] = m if MODE == "SMOKE" else np.array(m)
    return _IMG[size]


def has_cache(size):
    return any((d / f"images_{max(size, 512)}.npy").exists() for d in CACHE_DIRS)


class Films(Dataset):
    def __init__(self, ids, cfg, train):
        self.ids, self.cfg, self.train = np.asarray(ids), cfg, train
        self.imgs = get_images(cfg["size"])
        self.rows = df["row"].values[self.ids]
        self.ll = (df["projection"].values[self.ids] == "LL")
        self.y = torch.from_numpy(Y[self.ids])
        self.mask = torch.from_numpy(MASK_STUDY[self.ids]) if cfg["mask"] and train else torch.ones_like(self.y)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        a = np.asarray(self.imgs[self.rows[i]])
        if self.train and self.cfg["aug"] == "robust":
            if random.random() < .5:
                a = robust_aug(a, a.shape[0])
            if self.ll[i] and random.random() < .5:   # a lateral film has no left-right anatomy to break; DV/VD films do
                a = a[:, ::-1]
        return torch.from_numpy(a.copy())[None], self.y[i], self.mask[i]


HIERA = "hieradet_d_small_dino-v2-inat21"
ARCHS = {"hieradet_small": dict(mean=(0.5191, 0.5306, 0.4877), std=(0.2316, 0.2304, 0.2588)),
         "convnext_tiny": dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))}


class Classifier(nn.Module):
    def __init__(self, backbone, dim, n_out, drop, arch):
        super().__init__()
        self.backbone, self.arch = backbone, arch
        self.drop = nn.Dropout(drop)
        self.head = nn.Linear(dim, n_out)

    def features(self, x):
        return self.backbone.embedding(x) if self.arch == "hieradet_small" else self.backbone(x)

    def forward(self, x):
        return self.head(self.drop(self.features(x)))


def build_model(cfg, pretrained=True):
    arch = cfg["arch"]
    if arch == "hieradet_small":
        import birder
        from birder.model_registry import registry
        if pretrained:
            PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
            net, _ = birder.load_pretrained_model(HIERA, dst=PRETRAINED_DIR / f"{HIERA}.pt", inference=False)
            net.reset_classifier(0)
        else:
            net = registry.net_factory("hieradet_d_small", 0, 3, size=(cfg["size"], cfg["size"]))
        net.set_grad_checkpointing(True)
        blocks = [[net.stem, net.pos_embed, net.pos_embed_win]] + [[c] for c in net.body.children()] + [[net.features]]
        model = Classifier(net, net.embedding_size, len(LABELS), cfg["drop"], arch)
    else:
        import timm
        net = timm.create_model("convnext_tiny.fb_in22k_ft_in1k", pretrained=pretrained, num_classes=0)
        net.set_grad_checkpointing(True)
        blocks = [[net.stem]] + [[s] for s in net.stages] + [[net.head]]
        model = Classifier(net, net.num_features, len(LABELS), cfg["drop"], arch)
    model.blocks = blocks
    return model


def block_params(block):
    for item in block:
        yield from ([item] if isinstance(item, nn.Parameter) else item.parameters())


def make_optimizer(model, cfg, stage):
    """Stage 1: head only. Stage 2: layer-wise decayed lr; no weight decay on norms, biases and tables."""
    def split(params, lr):
        p = [q for q in params if q.requires_grad and id(q) not in seen]
        seen.update(id(q) for q in p)
        return [{"params": [q for q in p if q.ndim > 1], "lr": lr, "weight_decay": cfg["wd"]},
                {"params": [q for q in p if q.ndim <= 1], "lr": lr, "weight_decay": 0.0}]
    seen, groups = set(), []
    groups += split(model.head.parameters(), cfg["lr_head"] * (3 if stage == 1 else 1))
    if stage == 2:
        base = cfg["lr_head"] * cfg["lr_ratio"]
        for k, block in enumerate(reversed(model.blocks)):
            groups += split(block_params(block), base * cfg["llrd"] ** k)
        groups += split(model.parameters(), base * cfg["llrd"] ** len(model.blocks))
    return torch.optim.AdamW([g for g in groups if g["params"]])


def set_stage(model, stage):
    for p in model.parameters():
        p.requires_grad_(stage == 2)
    for p in model.head.parameters():
        p.requires_grad_(True)


def bn_eval_frozen(model):
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm) and not any(p.requires_grad for p in m.parameters()):
            m.eval()


def loss_fn(logits, y, mask, kind):
    logits = logits.float()
    if kind == "bce":
        l = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    else:   # asymmetric loss (Ridnik et al., 2021): easy negatives fade, positives keep their full weight
        p = torch.sigmoid(logits)
        pn = (p - 0.02).clamp(min=0)
        l = -(y * torch.log(p.clamp(min=1e-6)) + (1 - y) * pn.pow(2) * torch.log((1 - pn).clamp(min=1e-6)))
    return (l * mask).sum() / mask.sum().clamp(min=1)


def prepare(x_u8, cfg, train=False):
    x = x_u8.to(DEVICE, non_blocking=True).float() / 255.0
    if train:
        x = gpu_augment(x, cfg["aug"] == "robust")
    if x.shape[-1] != cfg["size"]:
        x = F.interpolate(x, size=(cfg["size"], cfg["size"]), mode="bilinear", align_corners=False, antialias=True)
    x = x.expand(-1, 3, -1, -1)
    s = ARCHS[cfg["arch"]]
    return (x - torch.tensor(s["mean"], device=DEVICE).view(1, 3, 1, 1)) / torch.tensor(s["std"], device=DEVICE).view(1, 3, 1, 1)


def batch_size(cfg):
    return {384: 16, 512: 12, 768: 6}[cfg["size"]]


@torch.no_grad()
def predict_ids(model, ids, cfg, max_batches=None):
    """Logits for film ids, in the order of ids."""
    model.eval(); out = []
    loader = DataLoader(Films(ids, cfg, False), batch_size=batch_size(cfg) * 2, shuffle=False, num_workers=0)
    for b, (x, _, _) in enumerate(loader):
        if max_batches and b >= max_batches:
            out.append(torch.zeros((len(ids) - sum(len(o) for o in out), len(LABELS))).numpy()); break
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=AMP):
            out.append(model(prepare(x, cfg)).float().cpu().numpy())
    return np.concatenate(out)


def study_table(p, ids, agg=0.5):
    """Film probabilities -> per-study probabilities: agg*max + (1-agg)*mean. Labels: any view tagged."""
    d = pd.DataFrame(p); s = df["study"].values[ids]
    g = d.groupby(s)
    sp = (agg * g.max() + (1 - agg) * g.mean()).values
    sy = pd.DataFrame(Y[ids]).groupby(s).max().values
    sg = pd.Series(df["group"].values[ids]).groupby(s).first().values
    return sp, sy, sg


def macro_ap(y, p, idx=CORE_IDX):
    v = [average_precision_score(y[:, j], p[:, j]) for j in idx if 0 < y[:, j].sum() < len(y)]
    return float(np.mean(v)) if v else float("nan")


def safe_auc(y, p):
    return roc_auc_score(y, p) if 0 < y.sum() < len(y) else float("nan")


BUDGET = dict(e1=1, e2=16, patience=4, warmup=1, max_batches=None) if MODE == "FULL" else dict(e1=1, e2=1, patience=1, warmup=0, max_batches=2)


def run_name(cfg, fold):
    h = hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
    return f"{cfg['arch']}_{cfg['size']}_{h}_f{fold}"


def train_fold(cfg, fold, label=""):
    """Train on the other folds, validate on `fold`. Resumes from last.pt; a finished run is loaded."""
    rd = RUN_ROOT / run_name(cfg, fold)
    rd.mkdir(parents=True, exist_ok=True)
    if (rd / "summary.json").exists():
        s = json.loads((rd / "summary.json").read_text())
        print(f"[{rd.name}] {label} finished earlier: val macro AP(study) {s['best_score']:.4f}")
        return s
    (rd / "config.json").write_text(json.dumps(cfg, indent=1))
    tr, va = np.flatnonzero(pool & (df["fold"].values != fold)), np.flatnonzero(pool & (df["fold"].values == fold))
    seed_all(SEED + fold)
    model = build_model(cfg).to(DEVICE)
    if cfg["bias_init"]:
        prior = np.clip(Y[tr].mean(0), 1e-4, 1 - 1e-4)
        model.head.bias.data = torch.tensor(np.log(prior / (1 - prior)), dtype=torch.float32, device=DEVICE)
    bs = batch_size(cfg); accum = max(1, round(32 / bs))
    scaler = torch.amp.GradScaler("cuda", enabled=AMP)
    ema, hist, best, bad, start = None, [], -np.inf, 0, (1, 0)
    last = rd / "last.pt"
    resume = torch.load(last, map_location="cpu", weights_only=False) if last.exists() else None
    if resume:
        hist, best, bad, start = resume["hist"], resume["best"], resume["bad"], (resume["stage"], resume["ep"] + 1)
        print(f"[{rd.name}] resuming at stage {start[0]} epoch {start[1]}")
        model.load_state_dict(resume["model"])
    t0 = time.time() - (hist[-1]["elapsed_s"] if hist else 0)
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    stop = False
    for stage, n_ep in [(1, BUDGET["e1"]), (2, BUDGET["e2"])]:
        if stage < start[0] or stop:
            continue
        set_stage(model, stage)
        opt = make_optimizer(model, cfg, stage)
        base_steps = len(tr) // bs + (len(tr) % bs > 0)
        steps = math.ceil(min(base_steps, BUDGET["max_batches"] or 10**9) / accum)
        warm, total = steps * (BUDGET["warmup"] if stage == 2 else 0), steps * n_ep
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / max(1, warm) if s < warm else
                                                  0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total - warm)))))
        if cfg["ema"] and stage == 2 and ema is None:
            ema = copy.deepcopy(model).eval()
            for p in ema.parameters():
                p.requires_grad_(False)
        first = start[1] if stage == start[0] else 0
        if resume and stage == resume["stage"]:
            opt.load_state_dict(resume["opt"]); sched.load_state_dict(resume["sched"]); scaler.load_state_dict(resume["scaler"])
            if resume.get("ema") is not None and ema is not None:
                ema.load_state_dict(resume["ema"])
        for ep in range(first, n_ep):
            seed_all(SEED + fold + 1000 * stage + ep)
            model.train(); bn_eval_frozen(model)
            loader = DataLoader(Films(tr, cfg, True), batch_size=bs, shuffle=True, num_workers=0, pin_memory=AMP,
                                generator=torch.Generator().manual_seed(SEED + ep))
            tot, n = 0.0, 0
            opt.zero_grad(set_to_none=True)
            for b, (x, y, m) in enumerate(loader):
                if BUDGET["max_batches"] and b >= BUDGET["max_batches"]:
                    break
                y, m = y.to(DEVICE, non_blocking=True), m.to(DEVICE, non_blocking=True)
                with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=AMP):
                    logits = model(prepare(x, cfg, train=True))
                    loss = loss_fn(logits, y, m, cfg["loss"])
                scaler.scale(loss / accum).backward()
                if (b + 1) % accum == 0 or b + 1 == len(loader):
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
                    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
                    if ema is not None:
                        d = 0.998
                        with torch.no_grad():
                            for pe, pm in zip(ema.state_dict().values(), model.state_dict().values()):
                                if pe.dtype.is_floating_point:
                                    pe.mul_(d).add_(pm.detach(), alpha=1 - d)
                                else:
                                    pe.copy_(pm)
                tot += loss.item() * len(x); n += len(x)
            ev = ema if ema is not None else model
            lg = predict_ids(ev, va, cfg, BUDGET["max_batches"])
            pv = 1 / (1 + np.exp(-lg))
            sp, sy, _ = study_table(pv, va)
            score, img_score = macro_ap(sy, sp), macro_ap(Y[va], pv)
            score = -np.inf if not np.isfinite(score) else score
            rec = dict(stage=stage, ep=ep, train_loss=tot / max(n, 1), val_study_ap=score, val_image_ap=img_score,
                       val_hernia_auc=safe_auc(sy[:, HERNIA], sp[:, HERNIA]), elapsed_s=time.time() - t0)
            hist.append(rec)
            print(f"[{rd.name}] {label} s{stage} ep{ep:02d} loss {rec['train_loss']:.4f} | study AP {score:.4f} | image AP {img_score:.4f} | "
                  f"hernia AUC {rec['val_hernia_auc']:.3f} | {rec['elapsed_s'] / 60:.1f} min", flush=True)
            if score > best:
                best, bad = score, 0
                torch.save(ev.state_dict(), rd / "best.pt"); np.save(rd / "val_logits.npy", lg)
            else:
                bad += 1
                if stage == 2 and bad >= BUDGET["patience"]:
                    stop = True
            torch.save(dict(model=model.state_dict(), ema=ema.state_dict() if ema is not None else None, opt=opt.state_dict(), sched=sched.state_dict(),
                            scaler=scaler.state_dict(), hist=hist, best=best, bad=bad, stage=stage, ep=ep), last)
            if stop:
                break
    s = dict(run=rd.name, label=label, cfg=cfg, fold=fold, best_score=best, best_image_ap=max(h["val_image_ap"] for h in hist if np.isfinite(h["val_image_ap"])) if hist else float("nan"),
             minutes=hist[-1]["elapsed_s"] / 60, epochs=len(hist), peak_vram_gb=torch.cuda.max_memory_allocated() / 1e9 if AMP else None, val_ids=va.tolist())
    (rd / "history.json").write_text(json.dumps(hist, indent=1))
    (rd / "summary.json").write_text(json.dumps(s, indent=1, default=float))
    last.unlink(missing_ok=True)
    del model, ema
    gc.collect(); torch.cuda.empty_cache()
    return s

# %% [markdown]
# ## 6 · Screening ladder on fold 0
#
# Each step adds one idea on top of the best configuration so far. The score is validation macro AP over the 10 core findings per study.
# A step is kept when it gains at least 0.005. Robust augmentation is kept unless it costs more than 0.01, because its benefit shows on damaged films, which this score does not contain.
# One fold cannot resolve small differences, so treat the ladder as a practical choice rather than a proof.

# %%
BASE = dict(arch="hieradet_small", size=384, loss="bce", mask=False, aug="light", ema=False, llrd=1.0, bias_init=False, lr_head=1e-3, lr_ratio=0.05, wd=0.05, drop=0.2)
LADDER = [("A · study-aware label mask", dict(mask=True), 0.005),
          ("B · robust aug + EMA + layer-wise lr + bias init", dict(aug="robust", ema=True, llrd=0.75, bias_init=True), -0.01),
          ("C · 512 px", dict(size=512), 0.005),
          ("D · 768 px", dict(size=768), 0.005),
          ("E · asymmetric loss", dict(loss="asl"), 0.005),
          ("F · ConvNeXt-Tiny", dict(arch="convnext_tiny", lr_ratio=0.1), 0.005)]

best_cfg = dict(BASE)
ladder = [dict(step="v1 protocol on fold 0", **train_fold(best_cfg, 0, "base"), kept=True)]
best_score = ladder[0]["best_score"]
for step, change, min_gain in LADDER:
    trial = {**best_cfg, **change}
    if not has_cache(trial["size"]):
        print(f"skip {step}: no {trial['size']} px cache in this environment")
        continue
    s = train_fold(trial, 0, step.split(" ")[0])
    keep = s["best_score"] - best_score >= min_gain
    ladder.append(dict(step=step, **s, kept=keep))
    if keep:
        best_cfg, best_score = trial, s["best_score"]
lad = pd.DataFrame(ladder)[["step", "best_score", "best_image_ap", "minutes", "epochs", "peak_vram_gb", "kept"]]
table(lad, "Screening ladder (fold 0)")
lad.to_csv(RUN_ROOT / "ladder.csv", index=False)
fig, ax = plt.subplots(figsize=(10, 4))
ax.barh(lad["step"][::-1], lad["best_score"][::-1], color=[TEAL if k else GREY for k in lad["kept"][::-1]])
ax.set(title="Validation macro AP per study, fold 0 (teal = kept)", xlabel="macro AP")
show("ladder")
(RUN_ROOT / "selected_config.json").write_text(json.dumps(best_cfg, indent=1))
print("selected:", best_cfg)

# %% [markdown]
# ## 7 · Cross-validation of the selected configuration
#
# Every train + val animal is predicted by a model that never saw it. Those out-of-fold predictions give the thresholds, the calibration and the reliability grade.

# %%
fold_list = list(range(N_FOLDS))
cv = [train_fold(best_cfg, f, f"cv{f}") for f in fold_list]
table(pd.DataFrame(cv)[["fold", "best_score", "best_image_ap", "minutes", "epochs"]], "Per-fold validation")
oof_ids = np.concatenate([np.array(s["val_ids"]) for s in cv])
oof_logits = np.concatenate([np.load(RUN_ROOT / s["run"] / "val_logits.npy") for s in cv])

from scipy.optimize import minimize_scalar


def fit_temperature(y, lg):
    def nll(t):
        return F.binary_cross_entropy_with_logits(torch.tensor(lg / t), torch.tensor(y, dtype=torch.float32 if lg.dtype == np.float32 else torch.float64)).item()
    return float(minimize_scalar(nll, bounds=(0.3, 5.0), method="bounded").x)


TEMPS = np.array([fit_temperature(Y[oof_ids, j], oof_logits[:, j]) for j in range(len(LABELS))])
sig = lambda x: 1 / (1 + np.exp(-x))
oof_p = sig(oof_logits / TEMPS)
AGGS = [0.0, 0.25, 0.5, 0.75, 1.0]
agg_scores = {a: macro_ap(*study_table(oof_p, oof_ids, a)[1::-1]) for a in AGGS}
AGG = max(agg_scores, key=agg_scores.get)
print("study fusion (share of max vs mean) -> OOF macro AP:", {k: round(v, 4) for k, v in agg_scores.items()}, "| chosen:", AGG)
op, oy, og = study_table(oof_p, oof_ids, AGG)


def threshold_f1(y, p):
    pr, rc, th = precision_recall_curve(y, p)
    return float(th[np.nanargmax(2 * pr[:-1] * rc[:-1] / np.maximum(pr[:-1] + rc[:-1], 1e-9))]) if y.sum() else 0.5


def threshold_sens(y, p, target=0.90):
    if not y.sum():
        return 0.5
    pos = np.sort(p[y == 1])
    return float(pos[max(0, int(math.floor((1 - target) * len(pos))))])


THR_F1 = np.array([threshold_f1(oy[:, j], op[:, j]) for j in range(len(LABELS))])
THR_S90 = np.array([threshold_sens(oy[:, j], op[:, j]) for j in range(len(LABELS))])


def per_label(y, p, thr):
    rows = []
    for j, l in enumerate(LABELS):
        n = int(y[:, j].sum()); pred = p[:, j] >= thr[j]
        tp, fp = int((pred & (y[:, j] == 1)).sum()), int((pred & (y[:, j] == 0)).sum())
        rows.append({"finding": l, "n+": n, "AP": average_precision_score(y[:, j], p[:, j]) if n else np.nan, "AUC": safe_auc(y[:, j], p[:, j]),
                     "sens": tp / n if n else np.nan, "spec": 1 - fp / max(1, (y[:, j] == 0).sum()), "ppv": tp / (tp + fp) if tp + fp else np.nan, "thr": thr[j]})
    return pd.DataFrame(rows).set_index("finding")


oof_table = per_label(oy, op, THR_F1)
oof_s90 = per_label(oy, op, THR_S90)
oof_table["spec@sens90"] = oof_s90["spec"]; oof_table["ppv@sens90"] = oof_s90["ppv"]


def grade(r):
    if r["n+"] < 70: return "experimental"
    return "strong" if r["AUC"] >= 0.92 else "moderate" if r["AUC"] >= 0.80 else "weak"


oof_table["reliability"] = oof_table.apply(grade, axis=1)
table(oof_table, f"Out-of-fold per study: macro AP core {macro_ap(oy, op):.3f}, macro AUC core {np.nanmean([safe_auc(oy[:, j], op[:, j]) for j in CORE_IDX]):.3f}")
oof_table.to_csv(RUN_ROOT / "oof_per_label.csv")
np.savez(RUN_ROOT / "oof.npz", ids=oof_ids, logits=oof_logits, temps=TEMPS, thr_f1=THR_F1, thr_s90=THR_S90, agg=AGG)
fig, ax = plt.subplots(figsize=(10, 4.4))
o = oof_table.sort_values("AUC")
ax.barh(o.index, o["AUC"], color=TEAL); ax.axvline(0.95, color=MAGENTA, ls="--", lw=1, label="0.95")
ax.set(title="Out-of-fold ROC-AUC per study", xlim=(0.5, 1)); ax.legend(loc="lower right")
show("oof_auc")

# %% [markdown]
# ## 8 · Input gate
#
# A small independent network learns the polarity and rotation of a film from self-generated labels, so no tag is needed. Its features also give a Mahalanobis
# distance that rejects inputs unlike any training film. This dataset holds no abdominal or non-radiograph images, so the rejection can be measured on synthetic inputs only.

# %%
GATE_SIZE = 224
g_rows = df["row"].values
_src = get_images(512) if has_cache(512) else get_images(best_cfg["size"])
GATE_IMG = np.stack([np.asarray(Image.fromarray(np.asarray(_src[int(r)])).resize((GATE_SIZE, GATE_SIZE), Image.Resampling.BILINEAR)) for r in g_rows])
GATE_MEAN, GATE_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def gate_apply(u8, inv, k):
    a = u8.copy()
    if inv:
        c = content(a); ys = np.flatnonzero(a.max(1)); xs = np.flatnonzero(a.max(0))
        if len(ys):
            a[ys[0]:ys[-1] + 1, xs[0]:xs[-1] + 1] = 255 - c
    return np.rot90(a, k).copy() if k else a


class GateData(Dataset):
    def __init__(self, ids, train):
        self.ids, self.train = ids, train

    def __len__(self):
        return len(self.ids) * (1 if self.train else 8)

    def __getitem__(self, i):
        if self.train:
            a, c = GATE_IMG[self.ids[i]], random.randrange(8)
            if random.random() < .5:
                a = np.asarray(Image.fromarray(to_model_uint8(web_resize_jpeg(content(a)), GATE_SIZE)))
        else:
            a, c = GATE_IMG[self.ids[i // 8]], i % 8
        return torch.from_numpy(gate_apply(a, c >= 4, c % 4))[None], c


def gate_x(x):
    x = x.to(DEVICE).float().div(255).expand(-1, 3, -1, -1)
    return (x - torch.tensor(GATE_MEAN, device=DEVICE).view(1, 3, 1, 1)) / torch.tensor(GATE_STD, device=DEVICE).view(1, 3, 1, 1)


def build_gate(pretrained=True):
    import timm
    return timm.create_model("resnet18.a1_in1k", pretrained=pretrained, num_classes=8)


gate_path = RUN_ROOT / "gate.pt"
g_tr = np.flatnonzero(pool & (df["fold"].values != 4)); g_ho = np.flatnonzero(pool & (df["fold"].values == 4))
gate = build_gate(not gate_path.exists()).to(DEVICE)
if not gate_path.exists():
    seed_all(); ep_n = 6 if MODE == "FULL" else 1
    opt = torch.optim.AdamW(gate.parameters(), lr=1e-3, weight_decay=0.05)
    dl = DataLoader(GateData(g_tr, True), batch_size=64, shuffle=True, num_workers=0)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=ep_n * len(dl))
    for e in range(ep_n):
        gate.train(); tot = 0
        for b, (x, c) in enumerate(dl):
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=AMP):
                l = F.cross_entropy(gate(gate_x(x)), c.to(DEVICE))
            opt.zero_grad(); l.backward(); opt.step(); sch.step(); tot += l.item()
        print(f"gate epoch {e}: loss {tot / len(dl):.4f}", flush=True)
    torch.save(gate.state_dict(), gate_path)
gate.load_state_dict(torch.load(gate_path, map_location=DEVICE, weights_only=True)); gate.eval()


@torch.no_grad()
def gate_eval(ids):
    ok, n, cm = 0, 0, np.zeros((8, 8), int)
    for x, c in DataLoader(GateData(ids, False), batch_size=128):
        p = gate(gate_x(x)).argmax(1).cpu().numpy(); np.add.at(cm, (c.numpy(), p), 1)
    return cm


@torch.no_grad()
def gate_feats(u8_batch):
    x = gate_x(torch.from_numpy(u8_batch)[:, None])
    f = gate.forward_features(x)
    return gate.global_pool(f).flatten(1).float().cpu().numpy()


cm = gate_eval(g_ho)
print(f"gate accuracy on held-out fold 4: {np.trace(cm) / cm.sum():.4f} | polarity {cm[:4, :4].sum() / cm[:4].sum() * .5 + cm[4:, 4:].sum() / cm[4:].sum() * .5:.4f} | clean film left untouched {cm[0, 0] / cm[0].sum():.4f}")
F_tr = np.concatenate([gate_feats(GATE_IMG[g_tr[i:i + 256]]) for i in range(0, len(g_tr), 256)])
F_mu = F_tr.mean(0); F_cov = np.cov(F_tr, rowvar=False) + 1e-3 * np.eye(F_tr.shape[1]); F_inv = np.linalg.inv(F_cov)


def maha(F_):
    d = F_ - F_mu
    return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", d, F_inv, d), 0))


def web_variant(a):
    return to_model_uint8(add_marks(web_resize_jpeg(content(a))), GATE_SIZE)


F_ho = np.concatenate([gate_feats(GATE_IMG[g_ho[i:i + 256]]) for i in range(0, len(g_ho), 256)])
web_ho = np.stack([web_variant(GATE_IMG[i]) for i in g_ho[:600]])
F_web = gate_feats(web_ho)
OOD_T = float(max(np.percentile(maha(F_ho), 99.9), np.percentile(maha(F_web), 99.5)))
rs = np.random.default_rng(0)
_disc = Image.new("L", (GATE_SIZE, GATE_SIZE)); ImageDraw.Draw(_disc).ellipse([50, 50, 170, 170], fill=255)
FAKES = {"random noise": rs.integers(0, 255, (GATE_SIZE, GATE_SIZE), dtype=np.uint8), "gradient": np.tile(np.linspace(0, 255, GATE_SIZE).astype(np.uint8), (GATE_SIZE, 1)),
         "white disc": np.asarray(_disc), "blank page": np.full((GATE_SIZE, GATE_SIZE), 240, np.uint8), "checkerboard": ((np.indices((GATE_SIZE, GATE_SIZE)).sum(0) // 16) % 2 * 255).astype(np.uint8)}
FAKE_D = {k: float(maha(gate_feats(v[None]))[0]) for k, v in FAKES.items()}
print(f"OOD threshold {OOD_T:.1f} | clean held-out rejected {(maha(F_ho) > OOD_T).mean():.3%} | web-like rejected {(maha(F_web) > OOD_T).mean():.3%}")
print("synthetic inputs:", {k: f"{v:.0f} ({'rejected' if v > OOD_T else 'ACCEPTED'})" for k, v in FAKE_D.items()})

# %% [markdown]
# ## 9 · Export the ensemble
#
# The folds, temperatures, thresholds, reliability grades and gate go into one checkpoint. The checkpoint is written before the test set is scored, so test results cannot influence it.

# %%
state = dict(format_version=2, labels=LABELS, core_labels=CORE, config=best_cfg, input_size=best_cfg["size"], mean=ARCHS[best_cfg["arch"]]["mean"], std=ARCHS[best_cfg["arch"]]["std"],
             folds=[torch.load(RUN_ROOT / s["run"] / "best.pt", map_location="cpu", weights_only=True) for s in cv],
             temperatures=TEMPS.tolist(), study_agg=float(AGG), thresholds_f1=THR_F1.tolist(), thresholds_sens90=THR_S90.tolist(),
             reliability=oof_table["reliability"].to_dict(), oof_auc=oof_table["AUC"].to_dict(),
             gate_state=gate.state_dict(), gate_size=GATE_SIZE, gate_mean=GATE_MEAN, gate_std=GATE_STD, ood_mu=F_mu.tolist(), ood_inv=F_inv.tolist(), ood_threshold=OOD_T,
             preprocess="vxr-pre-2.0", model_version=f"petai-xray-v2-{time.strftime('%Y%m%d')}", dataset_version="VetXRay v2, Zenodo 10.5281/zenodo.19051776 (CC BY 4.0)", mode=MODE)
torch.save(state, EXPORT)
print(f"exported {EXPORT} ({EXPORT.stat().st_size / 1e6:.0f} MB, {len(cv)} folds)")

# %% [markdown]
# ## 10 · Locked test set
#
# Everything above is frozen. Test animals are scored once with the out-of-fold thresholds. Intervals resample whole animals, and v1 is scored on the same studies with the same fusion, so the comparison is paired.

# %%
RUN_TEST = not STOP_BEFORE_TEST
if RUN_TEST:
    te = np.flatnonzero(df["split"].values == "test")
    tf = RUN_ROOT / "test_logits.npy"
    if tf.exists():
        te_lg = np.load(tf)
    else:
        fold_logits = []
        for s in cv:
            m = build_model(best_cfg, pretrained=False).to(DEVICE)
            m.load_state_dict(torch.load(RUN_ROOT / s["run"] / "best.pt", map_location=DEVICE, weights_only=True))
            fold_logits.append(predict_ids(m, te, best_cfg)); del m
        te_lg = np.mean(fold_logits, 0); np.save(tf, te_lg)
    te_p = sig(te_lg / TEMPS)
    tp, ty, tg = study_table(te_p, te, AGG)

    def boot(fn, n=2000 if MODE == "FULL" else 20):
        rng = np.random.default_rng(SEED); ug = np.unique(tg); where = {g: np.flatnonzero(tg == g) for g in ug}; out = []
        for _ in range(n):
            ii = np.concatenate([where[g] for g in rng.choice(ug, len(ug))]); out.append(fn(ii))
        return np.nanpercentile(out, [2.5, 97.5])

    res_f1, res_s90 = per_label(ty, tp, THR_F1), per_label(ty, tp, THR_S90)
    for name, j in [(l, i) for i, l in enumerate(LABELS)]:
        lo, hi = boot(lambda ii: safe_auc(ty[ii, j], tp[ii, j]), 500 if MODE == "FULL" else 10)
        res_f1.loc[name, "AUC lo"], res_f1.loc[name, "AUC hi"] = lo, hi
    res_f1["sens@90"] = res_s90["sens"]; res_f1["spec@90"] = res_s90["spec"]; res_f1["ppv@90"] = res_s90["ppv"]
    res_f1["reliability"] = oof_table["reliability"]
    table(res_f1, "v2 on locked test, per study")
    res_f1.to_csv(RUN_ROOT / "test_per_label.csv")
    v2_ap = macro_ap(ty, tp); v2_auc = float(np.nanmean([safe_auc(ty[:, j], tp[:, j]) for j in CORE_IDX]))
    img_ap = macro_ap(Y[te], te_p)
    ci = boot(lambda ii: macro_ap(ty[ii], tp[ii]))
    rows = [{"model": "v2 ensemble", "macro AP (study)": v2_ap, "95% CI": f"{ci[0]:.3f}-{ci[1]:.3f}", "macro AUC (study)": v2_auc, "macro AP (single film)": img_ap,
             "any_hernia AUC (study)": safe_auc(ty[:, HERNIA], tp[:, HERNIA]), "any_hernia AP (study)": average_precision_score(ty[:, HERNIA], tp[:, HERNIA]) if ty[:, HERNIA].sum() else np.nan}]
    v1f = (V1_DIR / "test_probs_full.npz") if V1_DIR else None
    if v1f is not None and v1f.exists() and MODE == "FULL":
        z = np.load(v1f); V1L = ["alveolar_pattern", "bronchial_pattern", "cardiomegaly", "costal_fracture", "diaphragmatic_hernia", "foreign_body", "fracture", "hernia", "interstitial_pattern",
                                 "mass", "megaesophagus", "pleural_effusion", "pleural_mineralization", "pneumoderma", "pneumomediastinum", "pneumothorax", "tracheal_collapse"]
        p1 = np.stack([z["p"][:, V1L.index(l)] for l in CORE] + [z["p"][:, [V1L.index("hernia"), V1L.index("diaphragmatic_hernia")]].max(1)], 1)
        vp, vy, _ = study_table(p1, te, 1.0)
        assert (vy == ty).all(), "v1 and v2 test studies differ"
        cv1 = boot(lambda ii: macro_ap(vy[ii], vp[ii]))
        rows.append({"model": "v1 HieraDet (best fusion: max)", "macro AP (study)": macro_ap(vy, vp), "95% CI": f"{cv1[0]:.3f}-{cv1[1]:.3f}",
                     "macro AUC (study)": float(np.nanmean([safe_auc(vy[:, j], vp[:, j]) for j in CORE_IDX])), "macro AP (single film)": macro_ap(Y[te], p1),
                     "any_hernia AUC (study)": safe_auc(vy[:, HERNIA], vp[:, HERNIA]), "any_hernia AP (study)": average_precision_score(vy[:, HERNIA], vp[:, HERNIA])})
        dlt = boot(lambda ii: macro_ap(ty[ii], tp[ii]) - macro_ap(vy[ii], vp[ii]))
        print(f"paired difference v2 - v1, macro AP per study: {v2_ap - macro_ap(vy, vp):+.3f} (95% CI {dlt[0]:+.3f} to {dlt[1]:+.3f})")
        v1_per = per_label(vy, vp, np.full(len(LABELS), .5))
        fig, ax = plt.subplots(figsize=(10, 5)); yy = np.arange(len(LABELS))
        ax.barh(yy - .2, v1_per["AUC"], .38, color=GREY, label="v1"); ax.barh(yy + .2, res_f1["AUC"], .38, color=TEAL, label="v2")
        ax.axvline(.95, color=MAGENTA, ls="--", lw=1); ax.set_yticks(yy, LABELS); ax.invert_yaxis(); ax.set(title="Test ROC-AUC per study", xlim=(.5, 1)); ax.legend()
        show("v1_vs_v2_auc")
    summary = pd.DataFrame(rows).set_index("model"); table(summary, "Locked test summary")
    summary.to_csv(RUN_ROOT / "test_summary.csv")
    sub = []
    sdf = df.iloc[te].groupby("study").first()   # sorted by study, like study_table
    for col in ("species", "scanner"):
        for v in sdf[col].unique():
            k = (sdf[col] == v).values
            sub.append({"factor": col, "level": v, "studies": int(k.sum()), "macro AP": macro_ap(ty[k], tp[k])})
    table(pd.DataFrame(sub).set_index(["factor", "level"]), "Subgroups (test, per study)")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for a, l in zip(axes, ["cardiomegaly", "pleural_effusion", "alveolar_pattern"]):
        j = LABELS.index(l); b = np.clip(np.digitize(tp[:, j], np.linspace(0, 1, 11)) - 1, 0, 9)
        conf, frac = [tp[b == i, j].mean() if (b == i).sum() >= 5 else np.nan for i in range(10)], [ty[b == i, j].mean() if (b == i).sum() >= 5 else np.nan for i in range(10)]
        a.plot([0, 1], [0, 1], ":", color=GREY); a.plot(conf, frac, "o-", color=TEAL); a.set(title=f"{l} · Brier {brier_score_loss(ty[:, j], tp[:, j]):.3f}", xlabel="predicted", ylabel="observed", xlim=(0, 1), ylim=(0, 1))
    show("calibration")

# %% [markdown]
# ## 11 · Robustness on damaged films
#
# Same simulated changes as the v1 analysis, single-film macro AP, with fixed severity. The v1 column holds the values measured for the v1 checkpoint on the same test films.
# These are simulations, not real Google images.

# %%
V1_SHIFT = {"clean": .491, "web: 500px + JPEG q60": .478, "web + white margin": .480, "web + arrows/circle/text box": .408, "web + tight crop": .447,
            "web + phone photo of film": .454, "inverted (negative)": .181, "rotated 90 deg": .381}
if RUN_TEST:
    random.seed(SEED)
    web = lambda a: web_resize_jpeg(a, 500, 60)
    VARIANTS = {"clean": lambda a: a, "web: 500px + JPEG q60": web, "web + white margin": lambda a: add_margin(web(a), .1), "web + arrows/circle/text box": lambda a: add_marks(web(a)),
                "web + tight crop": lambda a: tight_crop(web(a), 1 / 8), "web + phone photo of film": lambda a: web(photo_look(a, 1.4, .7, .12, .35)),
                "inverted (negative)": lambda a: 255 - a, "rotated 90 deg": lambda a: np.rot90(a).copy()}
    S = best_cfg["size"]; imgs_full = get_images(S)
    fold_models = []
    for s in (cv if MODE == "FULL" else cv[:1]):
        m = build_model(best_cfg, pretrained=False).to(DEVICE)
        m.load_state_dict(torch.load(RUN_ROOT / s["run"] / "best.pt", map_location=DEVICE, weights_only=True)); fold_models.append(m.eval())

    @torch.no_grad()
    def pipeline(a_list):
        """Mirror of petai_xray_inference_v2: gate correction on the small view, then the ensemble."""
        big = [to_model_uint8(a, S) for a in a_list]
        small = np.stack([np.asarray(Image.fromarray(b).resize((GATE_SIZE, GATE_SIZE), Image.Resampling.BILINEAR)) for b in big])
        pr = torch.softmax(gate(gate_x(torch.from_numpy(small)[:, None])), 1).cpu().numpy()
        for i, (c, conf) in enumerate(zip(pr.argmax(1), pr.max(1))):
            if conf < .8 or c == 0:
                continue
            a = a_list[i]
            if c >= 4:
                big[i] = to_model_uint8(255 - a, S)
            if c % 4:
                big[i] = np.rot90(big[i], -(c % 4)).copy()
        x = torch.from_numpy(np.stack(big))[:, None]
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=AMP):
            return np.mean([m(prepare(x, best_cfg)).float().cpu().numpy() for m in fold_models], 0)

    shift = []
    for name, fn in VARIANTS.items():
        lg = np.concatenate([pipeline([fn(content(np.asarray(imgs_full[int(r)]))) for r in df["row"].values[te[i:i + 8]]]) for i in range(0, len(te), 8)])
        p = sig(lg / TEMPS)
        shift.append({"variant": name, "v1 (single film)": V1_SHIFT[name] if MODE == "FULL" else np.nan, "v2 (single film)": macro_ap(Y[te], p), "v2 any_hernia AUC": safe_auc(Y[te, HERNIA], p[:, HERNIA])})
        print(shift[-1], flush=True)
    shift = pd.DataFrame(shift).set_index("variant"); table(shift, "Robustness"); shift.to_csv(RUN_ROOT / "robustness.csv")
    fig, ax = plt.subplots(figsize=(10, 4.4)); yy = np.arange(len(shift))
    ax.barh(yy - .2, shift["v1 (single film)"], .38, color=GREY, label="v1"); ax.barh(yy + .2, shift["v2 (single film)"], .38, color=TEAL, label="v2")
    ax.set_yticks(yy, shift.index); ax.invert_yaxis(); ax.set(title="Macro AP on damaged test films (simulated)"); ax.legend()
    show("robustness")

# %% [markdown]
# ## 12 · What this model can and cannot do
#
# * **Scope:** dog and cat thoracic radiographs. It does not diagnose abdominal hernia, and the gate cannot be proven to reject abdominal films because none exist here.
# * **Per study:** give every view of one visit at once. A single film works, with lower expected performance.
# * **Reliability:** each output carries the out-of-fold grade. `experimental` means too few cases to trust.
# * **Use:** research and decision support experiments, not a validated diagnostic device.
print("notebook complete")
