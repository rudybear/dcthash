"""Context numbers (NOT a codec): PSNR of *unquantised* float coefficients, to show how much of the
gap is basis vs quantisation.  K = number of (channel, basis-function) coefficients in trained order.

    python scripts/ceiling.py   ->  results/ceiling.md
"""
import os, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from hash2d.common import aspect_class
from hash2d.codecs.transform import _model, _analyse, _synthesise
from hash2d.metrics import to8, psnr
from scripts.run_eval import load_test_images

imgs = load_test_images()
Ks = [8, 16, 24, 36, 48, 64, 96, 160, 320]
lines = ["| basis | " + " | ".join(f"K={k}" for k in Ks) + " |", "|---|" + "---|" * len(Ks)]
for basis in ("dct", "pca", "legendre", "haar"):
    row = []
    for K in Ks:
        ps = []
        for _, im in imgs:
            H, W = im.shape[:2]
            m = _model(aspect_class(W, H), basis)
            c = _analyse(im, m)
            ps.append(psnr(to8(_synthesise(c[:K], m, W, H)), im))
        row.append(f"{np.mean(ps):.2f}")
    lines.append(f"| {basis} | " + " | ".join(row) + " |")
# resolution ceiling of the 32x32 canonical grid itself
from hash2d.common import resize
ps = [psnr(to8(np.clip(resize(resize(im, 32, 32, "area"), im.shape[0], im.shape[1], "cubic"), 0, 1)), im) for _, im in imgs]
lines.append(f"\n32x32 canonical grid itself (all 3072 values, unquantised): {np.mean(ps):.2f} dB")
open(os.path.join(ROOT, "results", "ceiling.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
