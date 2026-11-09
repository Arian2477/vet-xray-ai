"""PetAI thoracic radiograph inference v2 (VetXRay, 10 findings + experimental any_hernia, study level).

Give all views of ONE study (one animal, one visit) together; their results are fused. A single film also works.

    from petai_xray_inference_v2 import predict
    result = predict(["lateral.dcm", "dorsoventral.dcm"])

Command line:

    python petai_xray_inference_v2.py lateral.dcm dv.dcm --out result.json
    python petai_xray_inference_v2.py film.jpg --device cpu --compact

Set-up: `pip install torch timm birder pydicom numpy pillow`, then copy this file and `checkpoints/petai_xray_v2.pt`.
Inputs: DICOM (.dcm) or image (.png, .jpg, .jpeg, .bmp, .tif, .tiff, .webp).
Scope: dog and cat THORACIC radiographs. It does not diagnose abdominal hernia. Research use only: not a validated diagnostic device.
Exit code 0 = study analysed, 1 = study failed or every film was rejected.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "checkpoints" / "petai_xray_v2.pt"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
PCT_LOW, PCT_HIGH = 0.5, 99.5
GATE_MIN_CONF = 0.8
_CACHE: dict = {}


# ---- shared preprocessing (same code as the notebook) ----
def window_percentile(arr):
    """Clip to [P0.5, P99.5] and rescale to [0, 1]."""
    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(arr[::4, ::4], [PCT_LOW, PCT_HIGH])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-6:
        lo, hi = float(arr.min()), float(arr.max()) + 1e-6
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def letterbox_uint8(img, size):
    """Resize the long side to `size` (Lanczos), pad to a square with zeros, quantise to uint8."""
    from PIL import Image

    h, w = img.shape
    scale = size / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    small = np.asarray(Image.fromarray(img.astype(np.float32)).resize((nw, nh), Image.Resampling.LANCZOS))
    out = np.zeros((size, size), dtype=np.float32)
    top, left = (size - nh) // 2, (size - nw) // 2
    out[top:top + nh, left:left + nw] = np.clip(small, 0.0, 1.0)
    return np.round(out * 255.0).astype(np.uint8)


def trim_borders(a, tol=3.0):
    """Remove near-uniform outer rows and columns, such as screenshot or page margins."""
    h, w = a.shape
    rows = np.flatnonzero(a.std(1) > tol)
    cols = np.flatnonzero(a.std(0) > tol)
    if len(rows) < h * 0.3 or len(cols) < w * 0.3:
        return a
    return a[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def read_film(path: Path) -> np.ndarray:
    """File -> 2-D float32 array where bright = radiodense, as far as the file can tell."""
    if path.suffix.lower() in IMAGE_EXTS:
        from PIL import Image, ImageOps

        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            if im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info:
                rgba = im.convert("RGBA")
                bg = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
                im = Image.alpha_composite(bg, rgba).convert("L")
            elif im.mode.startswith("I") or im.mode == "F":
                im = im.convert("F")
            else:
                im = im.convert("L")
            arr = np.asarray(im, dtype=np.float32)
        return trim_borders(arr, tol=3.0 * (arr.max() / 255.0 if arr.max() > 255 else 1.0))
    import pydicom
    from pydicom.pixels import apply_modality_lut

    try:
        ds = pydicom.dcmread(str(path), force=True)
        arr = ds.pixel_array
    except Exception as e:
        raise ValueError(f"Not a readable DICOM or supported image: {path.name} ({type(e).__name__}: {e})") from e
    if arr.ndim == 3:
        arr = arr[0] if arr.shape[-1] not in (3, 4) else arr[..., :3].mean(-1)
    arr = apply_modality_lut(arr, ds).astype(np.float32)
    if str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")).upper() == "MONOCHROME1":
        arr = arr.max() - arr
    return arr


# ---- model reconstruction: all weights come from the checkpoint ----
class _Classifier(nn.Module):
    def __init__(self, backbone: nn.Module, dim: int, n_out: int, drop: float, arch: str):
        super().__init__()
        self.backbone, self.arch = backbone, arch
        self.drop = nn.Dropout(drop)
        self.head = nn.Linear(dim, n_out)

    def forward(self, x):
        f = self.backbone.embedding(x) if self.arch == "hieradet_small" else self.backbone(x)
        return self.head(self.drop(f))


def _build(cfg: dict, n_out: int) -> nn.Module:
    if cfg["arch"] == "hieradet_small":
        import birder  # noqa: F401  (registers the networks)
        from birder.model_registry import registry

        net = registry.net_factory("hieradet_d_small", 0, 3, size=(cfg["size"], cfg["size"]))
        return _Classifier(net, net.embedding_size, n_out, cfg["drop"], cfg["arch"])
    import timm

    net = timm.create_model("convnext_tiny.fb_in22k_ft_in1k", pretrained=False, num_classes=0)
    return _Classifier(net, net.num_features, n_out, cfg["drop"], cfg["arch"])


def _load(checkpoint, device):
    path = Path(checkpoint) if checkpoint else DEFAULT_CHECKPOINT
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    key = (str(path.resolve()), str(dev))
    if key not in _CACHE:
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        ck = torch.load(path, map_location="cpu", weights_only=True)
        missing = {"folds", "labels", "config", "temperatures", "thresholds_f1", "thresholds_sens90", "gate_state", "ood_mu", "ood_inv", "ood_threshold"} - set(ck)
        if missing:
            raise ValueError(f"Invalid checkpoint, missing keys: {sorted(missing)}")
        models = []
        for sd in ck["folds"]:
            m = _build(ck["config"], len(ck["labels"]))
            m.load_state_dict(sd, strict=True)
            models.append(m.to(dev).eval())
        import timm

        gate = timm.create_model("resnet18.a1_in1k", pretrained=False, num_classes=8)
        gate.load_state_dict(ck["gate_state"], strict=True)
        _CACHE[key] = (models, gate.to(dev).eval(), ck, dev)
    return _CACHE[key]


def _norm(u8: torch.Tensor, mean, std, dev) -> torch.Tensor:
    x = u8.to(dev).float().div(255).expand(-1, 3, -1, -1)
    return (x - torch.tensor(mean, device=dev).view(1, 3, 1, 1)) / torch.tensor(std, device=dev).view(1, 3, 1, 1)


def _resize_u8(a: np.ndarray, size: int) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.fromarray(a).resize((size, size), Image.Resampling.BILINEAR))


@torch.no_grad()
def _analyse_film(arr: np.ndarray, models, gate, ck, dev) -> dict:
    size = ck["input_size"]
    big = letterbox_uint8(window_percentile(arr), size)
    x = _norm(torch.from_numpy(_resize_u8(big, ck["gate_size"]))[None, None], ck["gate_mean"], ck["gate_std"], dev)
    feat = gate.global_pool(gate.forward_features(x)).flatten(1).float().cpu().numpy()
    probs = torch.softmax(gate(x), 1)[0].cpu().numpy()
    d = feat[0] - np.asarray(ck["ood_mu"])
    distance = float(np.sqrt(max(d @ np.asarray(ck["ood_inv"]) @ d, 0.0)))
    if distance > ck["ood_threshold"]:
        return {"accepted": False, "distance": distance, "corrected": []}
    cls, conf, corrected = int(probs.argmax()), float(probs.max()), []
    if cls and conf >= GATE_MIN_CONF:
        if cls >= 4:
            big = letterbox_uint8(window_percentile(arr.max() - arr), size)
            corrected.append("polarity inverted")
        if cls % 4:
            big = np.rot90(big, -(cls % 4)).copy()
            corrected.append(f"rotated {(cls % 4) * 90} degrees back")
    xm = _norm(torch.from_numpy(big.copy())[None, None], ck["mean"], ck["std"], dev)
    with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
        logits = torch.stack([m(xm).float() for m in models]).mean(0)[0].cpu().numpy()
    return {"accepted": True, "distance": distance, "corrected": corrected, "probs": 1 / (1 + np.exp(-logits / np.asarray(ck["temperatures"])))}


def predict(image_paths, checkpoint=None, device=None, operating_point: str = "balanced") -> dict:
    """Analyse one study. operating_point: "balanced" (best F1) or "high_sensitivity" (about 90 % sensitivity, more false alarms)."""
    if operating_point not in ("balanced", "high_sensitivity"):
        raise ValueError("operating_point must be 'balanced' or 'high_sensitivity'")
    paths = [Path(p) for p in ([image_paths] if isinstance(image_paths, (str, Path)) else image_paths)]
    if not paths:
        raise ValueError("No input files")
    models, gate, ck, dev = _load(checkpoint, device)
    films, kept = [], []
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(f"Input not found: {p}")
        if p.suffix.lower() not in IMAGE_EXTS | {".dcm", ".dicom", ""}:
            raise ValueError(f"Unsupported input type '{p.suffix}'. Use .dcm or one of {sorted(IMAGE_EXTS)}")
        r = _analyse_film(read_film(p), models, gate, ck, dev)
        films.append({"file": p.name, "accepted": r["accepted"], "input_distance": round(r["distance"], 1), "corrections": r["corrected"]})
        if r["accepted"]:
            kept.append(r["probs"])
    base = {"model_version": ck.get("model_version"), "dataset_version": ck.get("dataset_version"), "films": films,
            "scope": "Dog and cat thoracic radiographs only. Abdominal hernia is not covered.",
            "disclaimer": "Research model for decision support experiments; not a validated diagnostic device."}
    if not kept:
        return {"status": "rejected", "reason": "No input looks like the training radiographs. No finding is reported.", **base}
    P = np.stack(kept)
    a = ck["study_agg"]
    study = a * P.max(0) + (1 - a) * P.mean(0)
    thr = ck["thresholds_f1"] if operating_point == "balanced" else ck["thresholds_sens90"]
    findings = {l: {"score": round(float(s), 4), "threshold": round(float(t), 4), "positive": bool(s >= t),
                    "reliability": ck["reliability"][l], "out_of_fold_auc": round(float(ck["oof_auc"][l]), 3)}
                for l, s, t in zip(ck["labels"], study, thr)}
    return {"status": "ok", "operating_point": operating_point, "n_films_used": len(kept), "labels": list(ck["labels"]), "findings": findings,
            "positive_findings": [l for l in ck["labels"] if findings[l]["positive"]], **base}


def _main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="PetAI thoracic radiograph findings, one study (research use only).")
    ap.add_argument("inputs", nargs="+", help="all views of one study: DICOM or image files")
    ap.add_argument("--checkpoint", help=f"model file (default: {DEFAULT_CHECKPOINT})")
    ap.add_argument("--device", choices=["cpu", "cuda"])
    ap.add_argument("--operating-point", default="balanced", choices=["balanced", "high_sensitivity"])
    ap.add_argument("--out", help="write the JSON to this file instead of stdout")
    ap.add_argument("--compact", action="store_true", help="single-line JSON")
    args = ap.parse_args(argv)
    try:
        result = predict(args.inputs, args.checkpoint, args.device, args.operating_point)
    except Exception as e:
        result = {"status": "error", "error": f"{type(e).__name__}: {e}"}
    text = json.dumps(result, indent=None if args.compact else 2)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(_main())
