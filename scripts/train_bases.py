"""Train the decoder-side tables for the transform codecs on the Picsum training set
(DISJOINT from the Kodak test set):

  * mean image, per aspect class (landscape / portrait / square) on the canonical 32x32 grid
  * PCA (KLT) bases per channel (Y, Cb, Cr)
  * for every basis (dct / legendre / haar / pca): coefficient ordering by variance,
    Laplacian scale per coefficient (for the arithmetic-coded mode) and
    clip-range + distortion tables per bit depth (for the fixed-length mode)

    python scripts/train_bases.py   ->  models/transform_tables.npz
"""
import os, sys, glob
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from hash2d.common import rgb_to_ycc, resize
from hash2d.bases import ANALYTIC, G

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
N_KEEP = 320
MAXB = 8
CROPS = {"land": (3, 2), "port": (2, 3), "sq": (1, 1)}


def crops_for(img, cls, rng, n=4):
    H, W = img.shape[:2]
    aw, ah = CROPS[cls]
    out = []
    for _ in range(n):
        s = rng.uniform(0.7, 1.0)
        cw = int(min(W, H * aw / ah) * s); ch = int(cw * ah / aw)
        x0 = rng.integers(0, W - cw + 1); y0 = rng.integers(0, H - ch + 1)
        c = resize(img[y0:y0 + ch, x0:x0 + cw], G, G, "area")
        out.append(c)
        out.append(c[:, ::-1])
    return out


def fixed_tables(C):
    """C: (n, K) coefficients.  For b = 1..MAXB find the clip range L minimising the MSE of a
    2^b-level uniform mid-rise quantiser.  Returns L (K, MAXB+1), D (K, MAXB+1); D[:,0] = E[c^2]."""
    n, K = C.shape
    sd = C.std(0) + 1e-9
    L = np.zeros((K, MAXB + 1)); D = np.zeros((K, MAXB + 1))
    D[:, 0] = (C ** 2).mean(0)
    grid = np.geomspace(0.4, 8.0, 28)
    for b in range(1, MAXB + 1):
        nl = 1 << b
        best = np.full(K, np.inf)
        for g in grid:
            Lg = g * sd
            q = np.clip(np.floor((C + Lg) / (2 * Lg) * nl), 0, nl - 1)
            rec = (q + 0.5) / nl * 2 * Lg - Lg
            d = ((rec - C) ** 2).mean(0)
            upd = d < best
            best[upd] = d[upd]; L[upd, b] = Lg[upd]
        D[:, b] = best
    return L, D


def main():
    rng = np.random.default_rng(0)
    files = sorted(glob.glob(os.path.join(ROOT, "data", "picsum", "*.jpg")))
    print(len(files), "training images (Lorem Picsum; disjoint from Kodak)")
    imgs = [np.asarray(Image.open(f).convert("RGB"), dtype=np.float32) / 255 for f in files]
    out = {}
    for cls in CROPS:
        X = np.stack([rgb_to_ycc(c) for im in imgs for c in crops_for(im, cls, rng)])  # (n, G, G, 3)
        X = X.transpose(0, 3, 1, 2).reshape(len(X), 3, G * G)
        mean = X.mean(0)
        X = X - mean
        out[f"{cls}/mean"] = mean.astype(np.float32)
        print(cls, X.shape)
        bases = {name: [fn()] * 3 for name, fn in ANALYTIC.items()}
        pca = []
        for c in range(3):
            w, v = np.linalg.eigh(X[:, c].T @ X[:, c] / len(X))
            pca.append(v[:, ::-1][:, :160].T.copy())
        bases["pca"] = pca
        for name, Bs in bases.items():
            C = np.concatenate([X[:, c] @ Bs[c].T for c in range(3)], axis=1)       # (n, sum K_c)
            chan = np.concatenate([np.full(len(Bs[c]), c) for c in range(3)])
            idx = np.concatenate([np.arange(len(Bs[c])) for c in range(3)])
            order = np.argsort(-(C ** 2).mean(0))[:N_KEEP]
            C = C[:, order]
            L, D = fixed_tables(C)
            p = f"{cls}/{name}/"
            out[p + "chan"] = chan[order].astype(np.int16)
            out[p + "idx"] = idx[order].astype(np.int16)
            out[p + "lap"] = np.abs(C).mean(0).astype(np.float32)
            out[p + "L"] = L.astype(np.float32)
            out[p + "D"] = D.astype(np.float32)
            if name == "pca":
                out[p + "F"] = np.stack([Bs[c][i] for c, i in zip(chan[order], idx[order])]).astype(np.float16)
            ev = (C ** 2).mean(0)
            tot = (X ** 2).sum((1, 2)).mean()
            print(f"  {name:9s} energy captured by top 16/32/64/128 coefs: " +
                  " ".join(f"{ev[:k].sum() / tot:.4f}" for k in (16, 32, 64, 128)),
                  "| chroma coefs in top 32:", int((chan[order][:32] > 0).sum()))
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    dst = os.path.join(ROOT, "models", "transform_tables.npz")
    np.savez_compressed(dst, **out)
    print("saved", dst, os.path.getsize(dst) // 1024, "KiB")


if __name__ == "__main__":
    main()
