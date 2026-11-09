"""PetAI thoracic radiograph inference (VetXRay, 17 findings, multi-label).

Runs ONE fine-tuned model (the candidate selected on validation in xray_finetuning_kaggle.ipynb:
CLIP ViT-B/32, HieraDet-D Small or EfficientNet-B3; whichever is recorded in the checkpoint).
No API, no UI, no database: call `predict(path)` and receive a JSON-serialisable dict.

    from petai_xray_inference import predict
    result = predict("sample_xray.dcm")

Command line (e.g. on a VPS; CPU is used automatically when there is no GPU):

    python petai_xray_inference.py xray1.dcm xray2.png            # JSON on stdout
    python petai_xray_inference.py xray1.dcm --out result.json    # JSON in a file
    python petai_xray_inference.py xray1.dcm --device cpu --compact

Set-up: Python >= 3.11 and `pip install torch pydicom numpy pillow birder` (CPU wheels are enough), then copy this
file and `checkpoints/petai_xray_best.pt` (same relative folder) to the server. The first call loads the model
(slow once); several files given in one command reuse it. Exit code 0 = all inputs fine, 1 = at least one failed.

Inputs: DICOM (.dcm) or a grayscale/RGB image (.png, .jpg, .jpeg, .bmp, .tif, .tiff).
Research use only: not a validated veterinary diagnostic device.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "checkpoints" / "petai_xray_best.pt"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
_CACHE: dict = {}

# ---- shared preprocessing (verbatim copy of the notebook functions) ----
PREPROCESS_VERSION = "vxr-pre-1.0"
CACHE_SIZE = 512            # side of the square, letterboxed image stored in the cache
PCT_LOW, PCT_HIGH = 0.5, 99.5  # robust intensity window (percentiles of the body of the image)


def dicom_to_unit_float(ds):
    """DICOM dataset -> float32 image in [0, 1] where *bright = radiodense* (MONOCHROME2 convention).

    Steps: Modality LUT (slope/intercept) -> robust percentile window -> MONOCHROME1 inversion.
    The VOI window stored in VetXRay headers spans the full bit range (e.g. WC 8192 / WW 16384),
    so it carries no clinical windowing information; a data-driven percentile window is used instead.
    """
    from pydicom.pixels import apply_modality_lut

    arr = ds.pixel_array
    if arr.ndim == 3:  # colour or multi-frame: keep the first frame / luminance
        arr = arr[0] if arr.shape[-1] not in (3, 4) else arr[..., :3].mean(-1)
    arr = apply_modality_lut(arr, ds).astype(np.float32)
    img = window_percentile(arr)
    if str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")).upper() == "MONOCHROME1":
        img = 1.0 - img
    return img


def window_percentile(arr):
    """Clip to [P0.5, P99.5] (estimated on a 4x-subsampled grid for speed) and rescale to [0, 1]."""
    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    sample = arr[::4, ::4]
    lo, hi = np.percentile(sample, [PCT_LOW, PCT_HIGH])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-6:
        lo, hi = float(arr.min()), float(arr.max()) + 1e-6
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def letterbox_uint8(img, size=CACHE_SIZE):
    """Resize the long side to `size` (aspect ratio preserved, Lanczos), pad to a square, quantise to uint8."""
    from PIL import Image

    h, w = img.shape
    scale = size / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    small = np.asarray(Image.fromarray(img.astype(np.float32)).resize((nw, nh), Image.Resampling.LANCZOS))
    out = np.zeros((size, size), dtype=np.float32)
    top, left = (size - nh) // 2, (size - nw) // 2
    out[top:top + nh, left:left + nw] = np.clip(small, 0.0, 1.0)
    return np.round(out * 255.0).astype(np.uint8)


# ---- model reconstruction (no network access: weights come from the checkpoint) ----
class _Classifier(nn.Module):
    def __init__(self, backbone: nn.Module, feat_dim: int, n_out: int, drop: float, family: str):
        super().__init__()
        self.backbone, self.family = backbone, family
        self.drop = nn.Dropout(drop)
        self.head = nn.Linear(feat_dim, n_out)

    def features(self, x):
        if self.family == "clip_vit_b32":
            return self.backbone(pixel_values=x, interpolate_pos_encoding=True).pooler_output
        if self.family == "hieradet_small":
            return self.backbone.embedding(x)
        return self.backbone(x)

    def forward(self, x):
        return self.head(self.drop(self.features(x)))


def _build(ckpt: dict) -> nn.Module:
    arch, cfg, n = ckpt["architecture"], ckpt["arch_config"], len(ckpt["labels"])
    if arch == "clip_vit_b32":
        from transformers import CLIPVisionConfig, CLIPVisionModel
        bb = CLIPVisionModel(CLIPVisionConfig(**json.loads(cfg["config_json"])))
        dim = bb.config.hidden_size
    elif arch == "hieradet_small":
        import birder  # noqa: F401  (registers the networks)
        from birder.model_registry import registry
        size = (ckpt["input_size"], ckpt["input_size"])
        bb = registry.net_factory(cfg["birder_net"], 0, 3, size=size)
        dim = bb.embedding_size
    elif arch == "efficientnet_b3":
        import timm
        bb = timm.create_model(cfg["timm_id"], pretrained=False, num_classes=0)
        dim = bb.num_features
    else:
        raise ValueError(f"Unsupported architecture in checkpoint: {arch!r} (PetAI accepts only the three candidates)")
    model = _Classifier(bb, dim, n, ckpt.get("dropout", 0.0), arch)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model.eval()


def _load(checkpoint: str | Path | None, device: str | None):
    path = Path(checkpoint) if checkpoint else DEFAULT_CHECKPOINT
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    key = (str(path.resolve()), str(dev))
    if key not in _CACHE:
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}. Run the export section of the notebook first.")
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=True)   # safe: tensors and plain types only
        except Exception as e:
            raise RuntimeError(f"Checkpoint could not be loaded safely ({type(e).__name__}): {e}") from e
        required = {"architecture", "state_dict", "labels", "thresholds", "input_size", "mean", "std", "in_chans"}
        missing = required - set(ckpt)
        if missing:
            raise ValueError(f"Invalid checkpoint, missing keys: {sorted(missing)}")
        _CACHE[key] = (_build(ckpt).to(dev), ckpt, dev)
    return _CACHE[key]


def _read_image(path: Path) -> np.ndarray:
    """File -> uint8 512x512 letterboxed image, exactly as stored in the training cache."""
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTS:
        from PIL import Image
        arr = np.asarray(Image.open(path).convert("F"), dtype=np.float32)   # luminance; MONOCHROME2 convention assumed
        return letterbox_uint8(window_percentile(arr))
    import pydicom
    try:
        ds = pydicom.dcmread(str(path), force=True)
        _ = ds.pixel_array
    except Exception as e:
        raise ValueError(f"Not a readable DICOM or supported image: {path.name} ({type(e).__name__}: {e})") from e
    return letterbox_uint8(dicom_to_unit_float(ds))


def _to_tensor(img_u8: np.ndarray, ckpt: dict, dev: torch.device) -> torch.Tensor:
    x = torch.from_numpy(img_u8)[None, None].to(dev).float() / 255.0
    s = ckpt["input_size"]
    x = F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False, antialias=True)
    if ckpt["in_chans"] == 3:
        x = x.expand(-1, 3, -1, -1)
    mean = torch.tensor(ckpt["mean"], device=dev).view(1, -1, 1, 1)
    std = torch.tensor(ckpt["std"], device=dev).view(1, -1, 1, 1)
    return (x - mean) / std


@torch.no_grad()
def predict(image_path: str | Path, checkpoint: str | Path | None = None, device: str | None = None) -> dict:
    """Run the PetAI model on one radiograph and return a JSON-serialisable result."""
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Input not found: {path}")
    if path.suffix.lower() not in IMAGE_EXTS | {".dcm", ".dicom", ""}:
        raise ValueError(f"Unsupported input type '{path.suffix}'. Use .dcm or one of {sorted(IMAGE_EXTS)}")
    model, ckpt, dev = _load(checkpoint, device)
    img = _read_image(path)
    probs = torch.sigmoid(model(_to_tensor(img, ckpt, dev)).float())[0].cpu().numpy()
    labels, thr = ckpt["labels"], ckpt["thresholds"]
    core = set(ckpt.get("core_labels", labels))
    findings = {lab: {"score": round(float(p), 6), "threshold": round(float(t), 6), "positive": bool(p >= t),
                      "reliability": "validated" if lab in core else "insufficient_data"}
                for lab, p, t in zip(labels, probs, thr)}
    return {
        "status": "ok",
        "model_name": ckpt.get("family", ckpt["architecture"]),
        "model_version": ckpt.get("model_version", "unknown"),
        "dataset_version": ckpt.get("dataset_version", "unknown"),
        "num_labels": len(labels),
        "labels": list(labels),
        "findings": findings,
        "positive_findings": [lab for lab in labels if findings[lab]["positive"]],
        "input": {"file": path.name, "type": "dicom" if path.suffix.lower() not in IMAGE_EXTS else "image",
                  "preprocessing": PREPROCESS_VERSION, "model_input_size": ckpt["input_size"]},
        "device": str(dev),
        "disclaimer": "Research model for decision support experiments; not a validated diagnostic device.",
    }


def _main(argv: list[str] | None = None) -> int:
    """Command line: one or more radiographs in, JSON out (stdout or --out). Exit code 1 if any input failed."""
    import argparse

    ap = argparse.ArgumentParser(description="PetAI thoracic radiograph findings (research use only).")
    ap.add_argument("inputs", nargs="+", help="DICOM (.dcm) or image (.png/.jpg/.bmp/.tif) file(s)")
    ap.add_argument("--checkpoint", help=f"model file (default: {DEFAULT_CHECKPOINT})")
    ap.add_argument("--device", choices=["cpu", "cuda"], help="default: cuda if available, else cpu")
    ap.add_argument("--out", help="write the JSON to this file instead of stdout")
    ap.add_argument("--compact", action="store_true", help="single-line JSON")
    args = ap.parse_args(argv)

    results, failed = [], False
    for item in args.inputs:           # the model is loaded once and reused for every file
        try:
            results.append(predict(item, args.checkpoint, args.device))
        except Exception as e:         # one bad file must not stop the batch
            failed = True
            results.append({"status": "error", "input": Path(item).name, "error": f"{type(e).__name__}: {e}"})
    text = json.dumps(results[0] if len(results) == 1 else results, indent=None if args.compact else 2)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
