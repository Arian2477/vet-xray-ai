"""Technical checks of petai_xray_inference_v2 with a given checkpoint: file types, rejection, output format, parity."""
import json, sys, tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import petai_xray_inference_v2 as inf

ck = sys.argv[1]
parity = Path(r"G:\VetXRay_cache\parity_tmp")
dcm = sorted(parity.glob("*.dcm"))[:2]
tmp = Path(tempfile.mkdtemp())
ds = inf.read_film(dcm[0])
u8 = (inf.window_percentile(ds) * 255).astype(np.uint8)
Image.fromarray(u8).save(tmp / "film.jpg", quality=70)
Image.fromarray(u8).save(tmp / "film.png")
Image.fromarray((inf.window_percentile(ds) * 65535).astype(np.uint16)).save(tmp / "film16.png")
Image.fromarray(np.dstack([u8] * 3 + [np.full_like(u8, 255)])).save(tmp / "film_rgba.png")
Image.fromarray(np.pad(u8, 150, constant_values=255)).save(tmp / "film_margin.png")
exif = Image.Exif(); exif[0x0112] = 6
Image.fromarray(np.rot90(u8)).save(tmp / "film_exif.jpg", exif=exif, quality=90)
Image.fromarray(np.random.default_rng(0).integers(0, 255, (512, 512), dtype=np.uint8)).save(tmp / "noise.png")

ref = inf.predict([dcm[0]], ck)
rows = []
for name in ("film.jpg", "film.png", "film16.png", "film_rgba.png", "film_margin.png", "film_exif.jpg"):
    r = inf.predict([tmp / name], ck)
    if r["status"] != "ok":
        rows.append((name, r["status"], None)); continue
    d = max(abs(r["findings"][l]["score"] - ref["findings"][l]["score"]) for l in r["labels"])
    rows.append((name, r["status"], round(d, 3), r["films"][0]["corrections"]))
print("single DICOM:", ref["status"], ref["films"], "positives", ref["positive_findings"])
for row in rows:
    print(row)
print("two views:", inf.predict(dcm, ck)["n_films_used"], "| noise:", inf.predict([tmp / "noise.png"], ck)["status"])
hs = inf.predict([dcm[0]], ck, operating_point="high_sensitivity")
print("high sensitivity:", hs["operating_point"], "| JSON ok:", bool(json.dumps(hs)))
try:
    inf.predict([tmp / "missing.png"], ck)
except FileNotFoundError as e:
    print("missing file handled:", type(e).__name__)
print("CHECKS DONE")
