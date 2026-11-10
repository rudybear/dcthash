"""Quality check of the FROZEN DCTHash format against BlurHash at equal STRING LENGTH (characters).

    python scripts/eval_frozen.py [--tables format/tables.json] [--lengths 20 28 30 40 60 80]

BlurHash gets its best component grid per image among all grids whose string fits in N characters
(`blurhash`), plus the conventional 4x3 / 3x4 grid at 28 characters (`blurhash_4x3`).
DCTHash strings are exactly N characters of a 64-symbol alphabet (6 bits each; BlurHash's base83
carries 6.375 bits per character, so DCTHash is given slightly FEWER bits at equal length).
Also reports the colour metrics of hash2d.metrics.colour_metrics (OKLab errors against the original blurred with
sigma = 8 px, Hasler-Suesstrunk colourfulness ratio, saturation kept in the most saturated tenth of the picture),
and `dcthash_initial`: the same format with the first published, RGB-MSE-trained tables (format/tables_v0_initial.json).
Writes results/frozen_eval.csv and results/frozen_tables.md.
"""
import argparse, csv, os, sys, time
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from hash2d import dcthash as dh, metrics
from hash2d.common import resize, fit_size
from hash2d.codecs import blurhash_ref as bh
from scripts.run_eval import load_test_images


PREV = os.path.join(ROOT, "format", "tables_v0_initial.json")


def blurhash_best(img, n_chars):
    n = (n_chars - 4) // 2
    c = [(nx, ny) for nx in range(1, 10) for ny in range(1, 10) if nx * ny <= n]
    c = [(a, b) for a, b in c if not any((a2 >= a and b2 >= b and (a2, b2) != (a, b)) for a2, b2 in c)]
    small = resize(img, *fit_size(img.shape[0], img.shape[1], 128))
    best = None
    for nx, ny in c:
        s = bh.encode_str(small, nx, ny)
        err = np.mean((bh.decode_str(s, small.shape[1], small.shape[0]) - small) ** 2)
        if best is None or err < best[0]:
            best = (err, s)
    return best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default=None)
    ap.add_argument("--lengths", type=int, nargs="+", default=[20, 28, 30, 40, 60, 80])
    ap.add_argument("--tag", default="")
    ap.add_argument("--quick", action="store_true", help="DCTHash only, print mean PSNR")
    a = ap.parse_args()
    if a.tables:
        dh.load_tables = (lambda f, p: (lambda path=None: f(p)))(dh.load_tables, a.tables)   # experiment with other table files
    imgs = [(n, np.clip(np.round(im * 255), 0, 255).astype(np.uint8)) for n, im in load_test_images()]
    rows = []
    for name, im8 in imgs:
        img = im8.astype(np.float32) / 255
        H, W = img.shape[:2]
        ref_blur = metrics.blur(img)
        for N in a.lengths:
            cands = {}
            t = time.perf_counter(); s = dh.encode(im8, N); te = time.perf_counter() - t
            assert len(s) == N and dh.is_valid(s)
            cands["dcthash"] = (s, dh.decode(s, W, H)[..., :3] / 255.0)
            if os.path.exists(PREV) and not a.quick:      # the tables first published (RGB-MSE trained), for before/after
                s = dh.encode(im8, N, _tables=PREV)
                cands["dcthash_initial"] = (s, dh.decode(s, W, H, _tables=PREV)[..., :3] / 255.0)
            if not a.quick:
                s = blurhash_best(img, N); assert len(s) <= N
                cands["blurhash"] = (s, bh.decode_str(s, W, H))
                if N == 28:
                    nx, ny = (4, 3) if W >= H else (3, 4)
                    s = bh.encode_str(resize(img, *fit_size(H, W, 128)), nx, ny)
                    cands["blurhash_4x3"] = (s, bh.decode_str(s, W, H))
            for m, (s, rec) in cands.items():
                r = dict(image=name, method=m, length=N, chars=len(s), string=s)
                r.update(metrics.all_metrics(rec, img, ref_blur, deep=False))
                r.update(metrics.colour_metrics(rec, img))
                rows.append(r)
    key = lambda m, N, k: np.array([r[k] for r in rows if r["method"] == m and r["length"] == N])
    if a.quick:
        print(a.tag, " ".join(f"{N}:{key('dcthash', N, 'psnr').mean():.3f}" for N in a.lengths))
        return
    with open(os.path.join(ROOT, "results", "frozen_eval.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    out = ["| chars | method | PSNR | dPSNR vs BlurHash best (± s.e.m.) | wins | SSIM | PSNR vs blurred | OKLab dE | OKLab chroma dE | colourfulness ratio | top-saturation ratio |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for N in a.lengths:
        for m in ("blurhash_4x3", "blurhash", "dcthash_initial", "dcthash"):
            p = key(m, N, "psnr")
            if not len(p):
                continue
            d = p - key("blurhash", N, "psnr")
            mean = lambda k: key(m, N, k).mean()
            out.append(f"| {N} | {m} | {p.mean():.2f} | {d.mean():+.2f} ± {d.std(ddof=1) / np.sqrt(len(d)):.2f} | {int((d > 0).sum())}/{len(d)} | "
                       f"{mean('ssim'):.3f} | {mean('psnr_blur'):.2f} | {mean('de_ok'):.2f} | {mean('dc_ok'):.2f} | {mean('cf_ratio'):.2f} | {mean('sat_top_ratio'):.2f} |")
    open(os.path.join(ROOT, "results", "frozen_tables.md"), "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
