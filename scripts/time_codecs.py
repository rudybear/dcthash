"""Clean single-process timing pass (the main evaluation runs several workers in parallel, which
inflates wall-clock times).  Writes results/timing.md.

    python scripts/time_codecs.py [--images 4] [--budget 32]
"""
import argparse, os, sys, time
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from hash2d import registry
from scripts.run_eval import load_test_images


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=4)
    ap.add_argument("--budget", type=int, default=32)
    a = ap.parse_args()
    imgs = load_test_images(a.images + 1)
    codecs = registry.build()
    lines = [f"Single-process timings at {a.budget} bytes, median over {a.images} Kodak images (384x256), "
             "numpy decoders on one CPU core set, fitted encoders on GPU.\n",
             "| method | encode | decode 384x256 | decode 32px | decoder tables | decoder cost | JS/mobile practical? |",
             "|---|---|---|---|---|---|---|"]
    for c in codecs:
        c.encode(imgs[0][1], a.budget)          # warm-up (table builds, CUDA init) on an image not timed
        enc, dec, dec32 = [], [], []
        for _, im in imgs[1:]:
            H, W = im.shape[:2]
            t = time.perf_counter(); p = c.encode(im, a.budget); enc.append(time.perf_counter() - t)
            c.decode(p, W, H)
            t = time.perf_counter(); c.decode(p, W, H); dec.append(time.perf_counter() - t)
            t = time.perf_counter(); c.decode(p, 32, 32 * H // W if W >= H else 32); dec32.append(time.perf_counter() - t)
        e = np.median(enc)
        tb = c.decoder_tables_bytes
        lines.append(f"| {c.name} | {(f'{e * 1000:.0f} ms' if e < 1 else f'{e:.1f} s')} | {np.median(dec) * 1000:.1f} ms | "
                     f"{np.median(dec32) * 1000:.2f} ms | {(f'{tb / 1024:.0f} KiB' if tb else 'none')} | {c.decoder_cost} | {c.js_practical} |")
        print(lines[-1], flush=True)
    open(os.path.join(ROOT, "results", "timing.md"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
