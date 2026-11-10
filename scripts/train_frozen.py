"""Train and FREEZE the integer tables of the DCTHash format -> format/tables.json

Two table sets ("profiles"):

  photo   trained on Lorem Picsum photos (disjoint from the Kodak test set): coefficient order, means M,
          scales S, global loading factors g, and the bit sequence.  One set serves every aspect ratio
          (crops of nine aspect ratios are pooled; separate landscape / portrait / square tables were
          tried and gave +0.00 to +0.02 dB on Kodak).
  smooth  for gradients, flat colours and other non-photographic content whose coefficients fall far
          outside photo statistics: every coefficient uses a full-range companded quantiser, and only
          the coefficient order and bit sequence are learned, from procedurally generated gradients.

Everything is measured through the exact integer analysis front-end of hash2d/dcthash.py.

    python scripts/train_frozen.py [--out format/tables.json]
"""
import argparse, glob, json, os, sys
import numpy as np
from PIL import Image
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from hash2d import dcthash as dh

ASPECTS = {"land": [(4, 3), (3, 2), (16, 9)], "port": [(3, 4), (2, 3), (9, 16)], "sq": [(1, 1), (5, 4), (4, 5)]}
N_POOL = 320
SEQ_LEN = 6 * dh.MAX_LENGTH - 7
CHAN, VV, UU = [g.ravel() for g in np.meshgrid(np.arange(3), np.arange(16), np.arange(16), indexing="ij")]
WT = np.array(dh.CWEIGHT)[CHAN] * np.where(UU > 0, 2, 1) * np.where(VV > 0, 2, 1)
ISDC = (UU == 0) & (VV == 0)
# Photo-profile training objective ("look" weights).  Plain RGB-MSE weights (WT) give the highest PSNR but a grey,
# plaid-looking placeholder: only 9 of 64 coded coefficients were chroma at 28 characters, and luma bits went to
# long runs of pure-horizontal / pure-vertical frequencies.  Two changes, both only in how tables are trained:
#   CHROMA_WEIGHT  chroma errors count 8x  -> about half of the bits go to low-frequency chroma
#   radial prior   weight / (1 + (u^2 + v^2) / R0^2)^GAMMA  -> a compact, roughly isotropic low-pass set
# CHROMA_WEIGHT, R0 and GAMMA were chosen by eye on Kodak comparison sheets (see README, "Colour revision").
CHROMA_WEIGHT, R0, GAMMA = 8.0, 3.0, 1.5
WT_LOOK = WT * np.where(CHAN > 0, CHROMA_WEIGHT, 1.0) * (1.0 + (UU ** 2 + VV ** 2) / R0 ** 2) ** (-GAMMA)


def crops(img, cls, rng, n=4):
    H, W = img.shape[:2]
    out = []
    for k in range(n):
        aw, ah = ASPECTS[cls][k % len(ASPECTS[cls])]
        s = rng.uniform(0.7, 1.0)
        cw = int(min(W, H * aw / ah) * s); ch = int(cw * ah / aw)
        x0 = rng.integers(0, W - cw + 1); y0 = rng.integers(0, H - ch + 1)
        c = img[y0:y0 + ch, x0:x0 + cw]
        out += [c, c[:, ::-1]]
    return out


def synth(rng):
    """One procedurally generated smooth image (uint8 RGB): gradients, colour meshes, soft blobs, solid fills."""
    h, w = [(48, 48), (32, 48), (48, 32), (27, 48), (48, 27)][rng.integers(5)]
    y, x = np.mgrid[0:h, 0:w]
    x = (x + 0.5) / w; y = (y + 0.5) / h
    col = lambda: rng.uniform(0, 255, 3) if rng.random() < 0.7 else np.full(3, rng.uniform(0, 255)) + rng.normal(0, 12, 3)
    kind = rng.integers(5)
    if kind == 0:                                                  # linear gradient, 2-4 stops
        a = rng.uniform(0, 2 * np.pi)
        t = (x - 0.5) * np.cos(a) + (y - 0.5) * np.sin(a)
        t = (t - t.min()) / (t.max() - t.min())
        stops = np.sort(np.concatenate([[0, 1], rng.uniform(0.15, 0.85, rng.integers(0, 3))]))
        cols = np.stack([col() for _ in stops])
        img = np.stack([np.interp(t, stops, cols[:, c]) for c in range(3)], -1)
    elif kind == 1:                                                # radial gradient
        cx, cy, r = rng.uniform(-0.2, 1.2), rng.uniform(-0.2, 1.2), rng.uniform(0.4, 1.3)
        t = np.clip(np.hypot(x - cx, (y - cy) * h / w) / r, 0, 1)
        if rng.random() < 0.5:
            t = t * t * (3 - 2 * t)
        c0, c1 = col(), col()
        img = c0 + (c1 - c0) * t[..., None]
    elif kind == 2:                                                # four-corner colour mesh
        c = [col() for _ in range(4)]
        img = (c[0] * ((1 - x) * (1 - y))[..., None] + c[1] * (x * (1 - y))[..., None] +
               c[2] * ((1 - x) * y)[..., None] + c[3] * (x * y)[..., None])
    elif kind == 3:                                                # soft blobs on a base colour
        img = np.zeros((h, w, 3)) + col()
        for _ in range(rng.integers(1, 4)):
            cx, cy, s = rng.uniform(0, 1), rng.uniform(0, 1), rng.uniform(0.15, 0.5)
            g = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s * s))
            img = img + (col() - img) * g[..., None] * rng.uniform(0.4, 1.0)
    else:                                                          # (nearly) solid fill
        img = np.zeros((h, w, 3)) + col() + rng.normal(0, rng.uniform(0, 2), (h, w, 3))
    if rng.random() < 0.3:                                         # low-contrast variants
        img = img.mean((0, 1)) + (img - img.mean((0, 1))) * rng.uniform(0.05, 0.5)
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


def uniform_mse(x, M, L, b):
    """MSE per coefficient of the uniform quantiser (levels M - L + (2q+1) L / 2^b)."""
    L = np.asarray(L, dtype=np.float64)
    q = np.clip(np.floor((x - M + L) * (1 << b) / (2 * L)), 0, (1 << b) - 1)
    return ((M - L + (2 * q + 1) * L / (1 << b) - x) ** 2).mean(0)


def compand_mse(x, b):
    """MSE of the companded quantiser: levels FULL * n|n| / 4^(b-1), n = -(2^(b-1) - 1) .. 2^(b-1); nearest level."""
    n = np.arange(1 << b) - ((1 << (b - 1)) - 1)
    lev = dh.FULL * n * np.abs(n) / 4.0 ** (b - 1)
    q = np.searchsorted((lev[:-1] + lev[1:]) / 2, x, side="right")
    return ((lev[q] - x) ** 2).mean(0)


def bit_sequence(D):
    """Greedy bit allocation as a nested sequence.  D: (K, MAX_BITS+1) weighted distortion at 0..8 bits.
    The gain of a coefficient is its best average drop per bit over the next 1..k bits (lower convex hull),
    so a quantiser whose first bit alone does not pay still gets started."""
    K = len(D)
    bits = np.zeros(K, dtype=int)

    def gain_of(i):
        b = bits[i]
        if b >= dh.MAX_BITS:
            return -np.inf
        return max((D[i, b] - D[i, b + k]) / k for k in range(1, dh.MAX_BITS - b + 1))
    gain = np.array([gain_of(i) for i in range(K)])
    seq = []
    for _ in range(SEQ_LEN):
        i = int(np.argmax(gain))
        seq.append(i); bits[i] += 1
        gain[i] = gain_of(i)
    return seq


def pack(order, seq, extra=None):
    used = np.array(sorted(set(seq)))                              # keep only coefficients the sequence ever touches
    remap = {int(o): k for k, o in enumerate(used)}
    out = {"coefs": [[int(CHAN[order[i]]), int(UU[order[i]]), int(VV[order[i]])] for i in used]}
    for k, v in (extra or {}).items():
        out[k] = [int(v[i]) for i in used]
    out["seq"] = [remap[i] for i in seq]
    return out


def describe(name, p):
    for L in (6, 12, 28, 60, 96):
        b = np.bincount(p["seq"][:dh.payload_bits(L, name == "smooth")], minlength=len(p["coefs"]))
        dc = [int(b[i]) for i, c in enumerate(p["coefs"]) if c[1] == 0 and c[2] == 0]
        print(f"  {name} @ {L:2d} chars: {int((b > 0).sum()):3d} coefficients coded, max {int(b.max())} bits, DC bits {dc}, "
              f"chroma coefficients {sum(1 for i, c in enumerate(p['coefs']) if c[0] and b[i])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "format", "tables.json"))
    a = ap.parse_args()
    cos = np.array(dh.cos_table(), dtype=np.int64)

    # ---------------- photo profile
    cache = os.path.join(ROOT, "data", "train_coefs.npy")           # exact-integer analysis of all crops (slow: ~5 min)
    if os.path.exists(cache):
        allx = np.load(cache)
    else:
        rng = np.random.default_rng(0)
        files = sorted(glob.glob(os.path.join(ROOT, "data", "picsum", "*.jpg")))
        imgs = [np.asarray(Image.open(f).convert("RGB")) for f in files]
        print(len(imgs), "training images (Lorem Picsum; disjoint from Kodak)")
        allx = np.concatenate([np.stack([dh.analyse(c, cos).reshape(-1) for im in imgs for c in crops(im, cls, rng)]) for cls in ASPECTS])
        np.save(cache, allx)
    X = allx / 64.0                                                 # T units
    print("photo profile:", len(X), "crops")
    pool = np.argsort(-X.var(0) * WT_LOOK)[:N_POOL]
    order = pool[np.argsort(-X[:, pool].var(0) * WT_LOOK[pool])]
    x = X[:, order]
    wl = WT_LOOK[order]
    M = np.round(x.mean(0)).astype(np.int64)
    S = np.clip(np.round(x.std(0)), 1, 65535).astype(np.int64)
    # DC terms get fixed full-gamut quantisers so that pure black / white / saturated images are representable
    # (an MSE-trained range of a few sigma would clip them): luma DC uniform over 0..255, chroma DC companded.
    dc, ch = ISDC[order], CHAN[order]
    M[dc & (ch == 0)] = dh.FULL
    M[dc & (ch > 0)] = 0
    S[dc] = dh.FULL
    g = []                                                          # loading factors x256 per bit depth: [luma, chroma] non-DC
    for grp in (0, 1):
        sel = ~dc & ((ch > 0) == bool(grp))
        g.append([])
        for b in range(1, dh.MAX_BITS + 1):
            cand = [int(round(v * 256)) for v in np.geomspace(0.5, 24.0, 85)]
            tot = [float((uniform_mse(x[:, sel], M[sel], np.maximum(1, (S[sel] * G + 128) >> 8), b) * wl[sel] / S[sel] ** 2).sum()) for G in cand]
            g[grp].append(cand[int(np.argmin(tot))])
    print("  loading factors x256 [luma, chroma]:", g)
    D = np.zeros((len(order), dh.MAX_BITS + 1))
    D[:, 0] = ((x - M) ** 2).mean(0)
    for b in range(1, dh.MAX_BITS + 1):
        gb = np.where(ch > 0, g[1][b - 1], g[0][b - 1])
        L = np.where(dc, dh.FULL, np.maximum(1, (S * gb + 128) >> 8))
        D[:, b] = np.where(dc & (ch > 0), compand_mse(x, b), uniform_mse(x, M, L, b))
    photo = pack(order, bit_sequence(D * wl[:, None]), {"M": M, "S": S})
    describe("photo", photo)

    # ---------------- smooth profile
    rng = np.random.default_rng(1)
    Xs = np.stack([dh.analyse(synth(rng), cos).reshape(-1) for _ in range(6000)]) / 64.0
    print("smooth profile:", len(Xs), "synthetic images")
    order = np.argsort(-(Xs ** 2).mean(0) * WT)[:N_POOL]
    x = Xs[:, order]
    ydc = ISDC[order] & (CHAN[order] == 0)
    D = np.zeros((len(order), dh.MAX_BITS + 1))
    D[:, 0] = np.where(ydc, ((x - dh.FULL) ** 2).mean(0), (x ** 2).mean(0))
    for b in range(1, dh.MAX_BITS + 1):
        D[:, b] = np.where(ydc, uniform_mse(x, dh.FULL, dh.FULL, b), compand_mse(x, b))
    smooth = pack(order, bit_sequence(D * WT[order][:, None]))
    describe("smooth", smooth)

    out = {"format": "dcthash", "version": 0, "alphabet": dh.ALPHABET, "aspects": [list(p) for p in dh.ASPECTS],
           "scales": list(dh.SCALES), "cos": dh.cos_table(), "g": g, "photo": photo, "smooth": smooth}
    json.dump(out, open(a.out, "w"), separators=(",", ":"))
    print("wrote", a.out, os.path.getsize(a.out), "bytes")


if __name__ == "__main__":
    main()
