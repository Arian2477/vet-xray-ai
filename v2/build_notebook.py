"""Turn xray_finetuning_v2.py (# %% cells) into xray_finetuning_v2_kaggle.ipynb. The .py file stays the only source."""
from pathlib import Path

import nbformat as nbf

here = Path(__file__).resolve().parent
cells, kind, buf = [], None, []


def flush():
    text = "\n".join(buf).strip("\n")
    if not text:
        return
    if kind == "markdown":
        cells.append(nbf.v4.new_markdown_cell("\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in text.split("\n"))))
    else:
        cells.append(nbf.v4.new_code_cell(text))


for line in (here / "xray_finetuning_v2.py").read_text(encoding="utf-8").split("\n"):
    if line.startswith("# %%"):
        flush()
        kind, buf = ("markdown" if "[markdown]" in line else "code"), []
    else:
        buf.append(line)
flush()
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                                "language_info": {"name": "python"}})
nbf.validate(nb)
nbf.write(nb, here / "xray_finetuning_v2_kaggle.ipynb")
print(f"{len(cells)} cells written")
