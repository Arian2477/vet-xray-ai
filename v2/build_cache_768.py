"""Build images_768.npy from the original DICOM zips, with the same preprocessing and row order as images_512.npy."""
import io, json, sys, threading, time, zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

sys.path.insert(0, r"G:\Vet_XRay_AI")
from petai_xray_inference import dicom_to_unit_float, letterbox_uint8

SIZE = 768
DATA = Path(r"G:\VetXRay_2026")
OUT = Path(r"G:\Vet_XRay_AI\v2\work\cache")
OUT.mkdir(parents=True, exist_ok=True)
idx = pd.read_csv(r"G:\VetXRay_cache\kaggle_dataset\cache_index.csv").sort_values("row")
assert (idx["row"].values == np.arange(len(idx))).all()
names = idx["FileName"].tolist()

where = {}
for zp in sorted(DATA.glob("*.zip")):
    with zipfile.ZipFile(zp) as z:
        for m in z.namelist():
            if m.lower().endswith(".dcm") and "__MACOSX" not in m:
                where.setdefault(Path(m).name, (str(zp), m))
missing = [n for n in names if n not in where]
assert not missing, missing[:5]

local = threading.local()


def read(name):
    zp, m = where[name]
    zs = getattr(local, "zs", None)
    if zs is None:
        zs = local.zs = {}
    if zp not in zs:
        zs[zp] = zipfile.ZipFile(zp)
    return pydicom.dcmread(io.BytesIO(zs[zp].read(m)), force=True)


npy = OUT / f"images_{SIZE}.npy"
arr = np.lib.format.open_memmap(npy, mode="w+", dtype=np.uint8, shape=(len(names), SIZE, SIZE))


def work(i):
    arr[i] = letterbox_uint8(dicom_to_unit_float(read(names[i])), SIZE)


t0 = time.time()
with ThreadPoolExecutor(max_workers=6) as ex:
    for k, _ in enumerate(ex.map(work, range(len(names))), 1):
        if k % 500 == 0 or k == len(names):
            print(f"cached {k}/{len(names)} ({time.time() - t0:.0f}s)", flush=True)
arr.flush()
del arr
idx.to_csv(OUT / "cache_index_768.csv", index=False)
(OUT / "cache_768.done.json").write_text(json.dumps({"version": "vxr-pre-1.0", "n": len(names), "size": SIZE}))

# parity: downsized 768 rows must look like the existing 512 rows
from PIL import Image
a768 = np.load(npy, mmap_mode="r")
a512 = np.load(r"G:\VetXRay_cache\kaggle_dataset\images_512.npy", mmap_mode="r")
errs = [np.abs(np.asarray(Image.fromarray(np.asarray(a768[i])).resize((512, 512), Image.Resampling.BILINEAR), dtype=np.float32) - a512[i]).mean()
        for i in np.linspace(0, len(names) - 1, 50).astype(int)]
print(f"DONE parity mean abs diff vs 512 cache: mean {np.mean(errs):.2f}, max {np.max(errs):.2f} (out of 255)", flush=True)
