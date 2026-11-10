"""README figures: original vs BlurHash vs DCTHash at equal string length.

    python scripts/make_readme_figures.py

Writes docs/comparison_28.png (six Kodak photos, 28 characters, standard 4x3 BlurHash)
and docs/lengths.png (one photo at 20/28/40/60/80 characters, best-grid BlurHash).
Captions: PSNR against the original, and "colour" = Hasler-Suesstrunk colourfulness of the placeholder
divided by that of the original blurred to placeholder scale (1.00 = as colourful as it should be).
"""
import os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from hash2d import dcthash as dh, metrics
from hash2d.common import resize, fit_size
from hash2d.codecs import blurhash_ref as bh
from scripts.eval_frozen import blurhash_best

W, H, PAD, LBL = 384, 256, 8, 26


def font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def load(name):
    im = Image.open(os.path.join(ROOT, "data", "kodak", name)).convert("RGB")
    if im.height > im.width:
        im = im.rotate(90, expand=True)
    return np.asarray(im.resize((W, H), Image.LANCZOS))


def to8(x):
    return np.clip(np.round(x * 255), 0, 255).astype(np.uint8)


def codecs(im8, n, standard):
    img = im8.astype(np.float32) / 255
    if standard:
        s = bh.encode_str(resize(img, *fit_size(H, W, 128)), 4, 3)
    else:
        s = blurhash_best(img, n)
    b = bh.decode_str(s, W, H)
    d = dh.decode(dh.encode(im8, n), W, H)[..., :3] / 255.0
    cap = lambda x: f"{metrics.psnr(metrics.to8(x), img):.1f} dB, colour {metrics.colour_metrics(x, img)['cf_ratio']:.2f}"
    return (to8(b), cap(b)), (to8(d), cap(d))


def sheet(cells, row_labels, path, col_labels=None):
    rows, cols = len(cells), len(cells[0])
    left = 150
    top = LBL if col_labels else 0
    out = Image.new("RGB", (left + cols * (W + PAD) + PAD, top + rows * (H + LBL + PAD) + PAD), "white")
    dr = ImageDraw.Draw(out)
    f, fs = font(20), font(16)
    for c, t in enumerate(col_labels or []):
        dr.text((left + PAD + c * (W + PAD) + W // 2, 4), t, fill="black", font=f, anchor="ma")
    for r in range(rows):
        y = top + PAD + r * (H + LBL + PAD)
        dr.text((10, y + H // 2), row_labels[r], fill="black", font=f, anchor="lm")
        for c in range(cols):
            im, cap = cells[r][c]
            x = left + PAD + c * (W + PAD)
            out.paste(Image.fromarray(im), (x, y))
            if cap:
                dr.text((x + W // 2, y + H + 3), cap, fill=(70, 70, 70), font=fs, anchor="ma")
    out.save(path, optimize=True)
    print("wrote", path, out.size)


def main():
    os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
    names = ["kodim23.png", "kodim07.png", "kodim15.png", "kodim03.png", "kodim21.png", "kodim12.png"]
    r0, r1, r2 = [], [], []
    for n in names:
        im8 = load(n)
        (b, pb), (d, pd) = codecs(im8, 28, True)
        r0.append((im8, "")); r1.append((b, pb)); r2.append((d, pd))
    sheet([r0, r1, r2], ["Original", "BlurHash\n4x3, 28 ch", "DCTHash\n28 ch"], os.path.join(ROOT, "docs", "comparison_28.png"))

    im8 = load("kodim23.png")
    r1, r2 = [], []
    lengths = [20, 28, 40, 60, 80]
    for n in lengths:
        (b, pb), (d, pd) = codecs(im8, n, False)
        r1.append((b, pb)); r2.append((d, pd))
    sheet([r1, r2], ["BlurHash\nbest grid", "DCTHash"], os.path.join(ROOT, "docs", "lengths.png"),
          col_labels=[f"{n} characters" for n in lengths])


if __name__ == "__main__":
    main()
