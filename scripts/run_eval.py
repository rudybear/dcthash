"""Evaluate every codec on the test images at equal byte budgets.

    python scripts/run_eval.py [--budgets 16 24 32 48 64] [--workers 4] [--methods a b c] [--limit N]

Every result goes through real bytes: payload = codec.encode(img, B); assert len(payload) <= B;
rec = codec.decode(payload, w, h).  Payloads are cached in results/payloads/ (hex, one JSON per
image) so re-runs only recompute metrics; delete that directory to re-encode.
Output: results/results.csv
"""
import argparse, csv, glob, json, os, sys, time
import numpy as np
from PIL import Image
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from hash2d.common import resize
from hash2d import registry, metrics


def load_test_images(limit=None):
    files = sorted(glob.glob(os.path.join(ROOT, "data", "kodak", "*.png")))
    if not files:   # offline fallback
        from skimage import data
        return [(n, resize(getattr(data, n)().astype(np.float32)[..., :3] / 255, 256, 384))
                for n in ("astronaut", "coffee", "chelsea", "rocket")]
    out = []
    for f in files[:limit]:
        im = np.asarray(Image.open(f).convert("RGB"), dtype=np.float32) / 255
        out.append((os.path.splitext(os.path.basename(f))[0], resize(im, im.shape[0] // 2, im.shape[1] // 2, "area")))
    return out


def run_image(args):
    name, img, budgets, methods = args
    H, W = img.shape[:2]
    cache_path = os.path.join(ROOT, "results", "payloads", name + ".json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}

    def get_payload(codec, image, budget):
        key = f"{codec.name}|{budget}"
        if key not in cache:
            t = time.perf_counter()
            p = codec.encode(image, budget)
            cache[key] = {"hex": p.hex(), "enc_s": time.perf_counter() - t}
        return bytes.fromhex(cache[key]["hex"])

    codecs = [c for c in registry.build(inner_cache=get_payload) if methods is None or c.name in methods]
    ref_blur = metrics.blur(img)
    rows = []
    for B in budgets:
        for c in codecs:
            payload = get_payload(c, img, B)
            assert len(payload) <= B, (c.name, B, len(payload))
            t = time.perf_counter()
            rec = c.decode(payload, W, H)
            t1 = time.perf_counter() - t
            t = time.perf_counter()
            c.decode(payload, 32, 32 * H // W if W >= H else 32)       # thumbnail-size decode (what a client would do)
            t2 = time.perf_counter() - t
            assert rec.shape == img.shape
            enc_s = cache[f"{c.name}|{B}"]["enc_s"]
            if hasattr(c, "codecs"):     # best-fit: its cost is the sum of all candidate encoders (they may have been cached)
                enc_s = sum(cache[f"{k.name}|{B - 1}"]["enc_s"] for k in c.codecs if f"{k.name}|{B - 1}" in cache)
            row = dict(image=name, method=c.name, budget=B, bytes=len(payload), enc_s=enc_s,
                       dec_s=t1, dec32_s=t2, chosen=c.chosen(payload) if hasattr(c, "chosen") else "")
            row.update(metrics.all_metrics(rec, img, ref_blur))
            rows.append(row)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        json.dump(cache, open(cache_path, "w"))
    print(name, "done", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", type=int, nargs="+", default=[16, 24, 32, 48, 64])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "results.csv"))
    a = ap.parse_args()
    imgs = load_test_images(a.limit)
    jobs = [(n, im, a.budgets, a.methods) for n, im in imgs]
    if a.workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(a.workers) as pool:
            res = pool.map(run_image, jobs, chunksize=1)
    else:
        res = [run_image(j) for j in jobs]
    rows = [r for rr in res for r in rr]
    keys = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print("wrote", a.out, len(rows), "rows")


if __name__ == "__main__":
    main()
