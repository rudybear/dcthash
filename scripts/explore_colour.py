"""Exploration harness for the colour-vibrancy iteration (float simulation of format variants).

Not part of the format.  It trains tables for a variant on the cached Picsum coefficients
(data/train_coefs.npy, written by scripts/train_frozen.py), simulates encode/decode on Kodak and
reports PSNR/SSIM plus the colour metrics of hash2d.metrics.colour_metrics, next to BlurHash and
the shipped DCTHash.  The chosen variant is then implemented exactly in hash2d/dcthash.py.

    python scripts/explore_colour.py baseline       # BlurHash and shipped DCTHash on the colour metrics
    python scripts/explore_colour.py sweep          # variants table
    python scripts/explore_colour.py sheets NAME... # side-by-side sheets for chosen variants
"""
import os, sys, json, itertools
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from hash2d import dcthash as dh, metrics
from hash2d.common import resize, fit_size
from hash2d.codecs import blurhash_ref as bh
from scripts.run_eval import load_test_images

CHAN, VV, UU = [g.ravel() for g in np.meshgrid(np.arange(3), np.arange(16), np.arange(16), indexing="ij")]
ALPHA = np.where(UU > 0, 2, 1) * np.where(VV > 0, 2, 1)
ISDC = (UU == 0) & (VV == 0)
FULL = dh.FULL
MAXB = 8


# ------------------------------------------------------------------ quantiser families (float simulation)
def q_uniform(x, M, L, b):
    q = np.clip(np.floor((x - M + L) * (1 << b) / (2 * L)), 0, (1 << b) - 1)
    return M - L + (2 * q + 1) * L / (1 << b)


def q_zero(x, L, b, p):
    """companded quantiser with an exact zero: levels L * sign(n) |n|^p / (2^(b-1))^p, n = -(2^(b-1)-1) .. 2^(b-1)"""
    n = np.arange(1 << b) - ((1 << (b - 1)) - 1)
    lev = np.sign(n) * np.abs(n) ** p / float(1 << (b - 1)) ** p          # normalised levels, max 1
    t = x / L
    q = np.searchsorted((lev[:-1] + lev[1:]) / 2, t, side="right")
    return lev[q] * L


class Variant:
    """cfg keys:
       cw        chroma weight multiplier in the training / selection objective (1 = RGB-MSE weights)
       ac        'midrise' (shipped) | 'zero'   quantiser family of non-DC coefficients
       p         companding power for 'zero'
       min_bits  hull start is automatic; nothing to set
       lscale / cscale   lists of x4 multipliers selectable per image for luma / chroma non-DC ranges
       rad       exponent of the radial-frequency prior on the allocation weights (0 = none)
       taper     decoder shaping strength (0 = none): K *= 1 / (1 + taper * (f/16)^2) style roll-off
    """
    def __init__(self, name, X, **cfg):
        self.name, self.cfg = name, dict(cw=1.0, ac="midrise", p=2.0, lscale=[2, 3, 4, 6], cscale=None, taper=0.0, pool=320)
        self.cfg.update(cfg)
        c = self.cfg
        self.wt = np.array(dh.CWEIGHT, dtype=float)[CHAN] * ALPHA * np.where(CHAN > 0, c["cw"], 1.0)
        if c.get("rad"):          # isotropy prior: favour low radial frequency over long runs along one axis
            self.wt = self.wt * (1.0 + (UU ** 2 + VV ** 2) / float(c.get("r0", 1.0)) ** 2) ** (-c["rad"])
        self.mse_wt = np.array(dh.CWEIGHT, dtype=float)[CHAN] * ALPHA
        order = np.argsort(-X.var(0) * self.wt)[:c["pool"]]
        order = np.array(sorted(set(order.tolist()) | {0, 256, 512}))
        order = order[np.argsort(-X[:, order].var(0) * self.wt[order])]
        x = X[:, order]
        self.order = order
        self.ch, self.dc = CHAN[order], ISDC[order]
        self.M = np.round(x.mean(0)); self.S = np.clip(np.round(x.std(0)), 1, 65535)
        self.M[self.dc & (self.ch == 0)] = FULL; self.M[self.dc & (self.ch > 0)] = 0; self.S[self.dc] = FULL
        if c["ac"] == "zero":
            self.M[~self.dc] = 0
        ac = ~self.dc
        w = self.wt[order]
        # loading factors per bit depth, separately for luma and chroma non-DC coefficients
        self.g = np.ones((2, MAXB + 1))
        for grp in (0, 1):
            sel = ac & ((self.ch > 0) == bool(grp))
            for b in range(1, MAXB + 1):
                cand = np.geomspace(0.5, 24.0, 29)
                tot = [float((((self._quant_ac(x[:, sel], self.M[sel], self.S[sel] * gg, b) - x[:, sel]) ** 2).mean(0) * w[sel] / self.S[sel] ** 2).sum()) for gg in cand]
                self.g[grp, b] = cand[int(np.argmin(tot))]
        D = np.zeros((len(order), MAXB + 1))
        D[:, 0] = ((x - self.M) ** 2).mean(0)
        for b in range(1, MAXB + 1):
            rec = self._quant_all(x, np.full(len(order), b), 1.0, 1.0)
            D[:, b] = ((rec - x) ** 2).mean(0)
        D *= w[:, None]
        bits = np.zeros(len(order), dtype=int)
        gain_of = lambda i: -np.inf if bits[i] >= MAXB else max((D[i, bits[i]] - D[i, bits[i] + k]) / k for k in range(1, MAXB - bits[i] + 1))
        gain = np.array([gain_of(i) for i in range(len(order))])
        self.seq = []
        for _ in range(6 * 96):
            i = int(np.argmax(gain)); self.seq.append(i); bits[i] += 1; gain[i] = gain_of(i)
        self.seq = np.array(self.seq)

    def _quant_ac(self, x, M, L, b):
        if self.cfg["ac"] == "midrise":
            return q_uniform(x, M, L, b)
        return q_zero(x, L, b, self.cfg["p"])

    def _quant_all(self, x, bits, ls, cs):
        """x: (..., K) -> reconstruction, for per-coefficient bit depths `bits` and range multipliers ls / cs."""
        rec = np.broadcast_to(self.M, x.shape).astype(float).copy()
        for b in np.unique(bits):
            if b == 0:
                continue
            sel = bits == b
            ydc, cdc, ac = sel & self.dc & (self.ch == 0), sel & self.dc & (self.ch > 0), sel & ~self.dc
            if ydc.any(): rec[..., ydc] = q_uniform(x[..., ydc], FULL, FULL, b)
            if cdc.any(): rec[..., cdc] = q_zero(x[..., cdc], FULL, b, 2.0)
            for grp, mult in ((0, ls), (1, cs)):
                s2 = ac & ((self.ch > 0) == bool(grp))
                if s2.any():
                    L = np.maximum(1, self.S[s2] * self.g[grp, b] * mult)
                    rec[..., s2] = self._quant_ac(x[..., s2], self.M[s2], L, b)
        return rec

    def header_bits(self):
        c = self.cfg
        return 7 + int(np.ceil(np.log2(len(c["lscale"])))) + (int(np.ceil(np.log2(len(c["cscale"])))) if c["cscale"] else 0)

    def bits_for(self, length):
        return np.bincount(self.seq[:6 * length - self.header_bits()], minlength=len(self.order))

    def code(self, x768, length):
        """one image: 768 coefficients (T units) -> reconstructed 768 coefficients."""
        c = self.cfg
        x = x768[self.order]
        bits = self.bits_for(length)
        w = self.wt[self.order]
        best = None
        lum = self.ch == 0
        recs_l = [self._quant_all(x, bits, s / 4, 1.0) for s in c["lscale"]]
        el = [float((w * (r - x) ** 2)[lum].sum()) for r in recs_l]
        rec = recs_l[int(np.argmin(el))].copy()
        if c["cscale"]:
            recs_c = [self._quant_all(x, bits, 1.0, s / 4) for s in c["cscale"]]
            ec = [float((w * (r - x) ** 2)[~lum].sum()) for r in recs_c]
            rec[~lum] = recs_c[int(np.argmin(ec))][~lum]
        else:                       # one shared scale: choose on the total error
            et = [float((w * (r - x) ** 2).sum()) for r in recs_l]
            rec = recs_l[int(np.argmin(et))]
        out = np.zeros(768); out[self.order] = rec
        return out

    def describe(self, length):
        b = self.bits_for(length)
        cc = (self.ch > 0)
        return (f"{int((b > 0).sum())} coefs ({int(((b > 0) & cc).sum())} chroma, {int(b[cc].sum())} of {int(b.sum())} bits chroma; "
                f"DC bits {[int(v) for v in b[self.dc]]})")


def render(rec768, W, H, taper=0.0):
    K = (rec768 * ALPHA / 256.0).reshape(3, 16, 16)
    if taper:
        f = np.arange(16) / 16.0
        K = K / (1 + taper * (f[None, :, None] ** 2 + f[None, None, :] ** 2))
    cx = np.cos(np.pi * np.arange(16)[:, None] * (np.arange(W)[None] + .5) / W)
    cy = np.cos(np.pi * np.arange(16)[:, None] * (np.arange(H)[None] + .5) / H)
    return dh._to_rgb(np.einsum("vy,cvu,ux->yxc", cy, K, cx))


# ------------------------------------------------------------------ data
def kodak():
    out = []
    for name, im in load_test_images():
        im8 = np.clip(np.round(im * 255), 0, 255).astype(np.uint8)
        out.append((name, im8, dh.analyse(im8).reshape(-1) / 64.0))
    return out


def all_metrics(rec8, im8):
    img = im8.astype(np.float64) / 255; rec = rec8.astype(np.float64) / 255
    m = metrics.all_metrics(rec, img, deep=False)
    m.update(metrics.colour_metrics(rec, img))
    return m


KEYS = ["psnr", "ssim", "psnr_blur", "de_ok", "dc_ok", "cf_ratio", "chroma_ratio", "sat_top_ratio"]


def blurhash_img(im8, length, standard=False):
    img = im8.astype(np.float32) / 255
    H, W = img.shape[:2]
    if standard:
        nx, ny = (4, 3) if W >= H else (3, 4)
        s = bh.encode_str(resize(img, *fit_size(H, W, 128)), nx, ny)
    else:
        from scripts.eval_frozen import blurhash_best
        s = blurhash_best(img, length)
    return np.clip(np.round(bh.decode_str(s, W, H) * 255), 0, 255).astype(np.uint8)


def row(label, ms):
    return f"| {label} | " + " | ".join(f"{np.mean([m[k] for m in ms]):.3f}" if k not in ("psnr", "psnr_blur", "de_ok", "dc_ok") else f"{np.mean([m[k] for m in ms]):.2f}" for k in KEYS) + " |"


HEAD = "| method | " + " | ".join(KEYS) + " |\n|---|" + "---|" * len(KEYS)


# ------------------------------------------------------------------ side-by-side sheets
def sheets(columns, length, prefix, per_file=8, cell=(264, 176)):
    """columns: list of (label, fn(name, im8, x768, length) -> rgb uint8).  One row per Kodak image, `per_file`
    rows per PNG.  Portrait images are shown rotated.  Caption: PSNR / colourfulness ratio / top-saturation ratio."""
    from PIL import Image, ImageDraw, ImageFont
    fp = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    f = ImageFont.truetype(fp, 15) if os.path.exists(fp) else ImageFont.load_default()
    fs = ImageFont.truetype(fp, 12) if os.path.exists(fp) else ImageFont.load_default()
    K = kodak()
    cw, chh = cell
    pad, cap, top = 6, 16, 24
    paths = []
    for start in range(0, len(K), per_file):
        chunk = K[start:start + per_file]
        out = Image.new("RGB", (pad + (len(columns) + 1) * (cw + pad), top + len(chunk) * (chh + cap + pad) + pad), "white")
        dr = ImageDraw.Draw(out)
        for c, lab in enumerate(["original"] + [c[0] for c in columns]):
            dr.text((pad + c * (cw + pad) + cw // 2, 4), lab, fill="black", font=f, anchor="ma")
        for r, (name, im8, x) in enumerate(chunk):
            y = top + r * (chh + cap + pad)
            cells = [(im8, name)]
            for lab, fn in columns:
                rec = fn(name, im8, x, length)
                m = all_metrics(rec, im8)
                cells.append((rec, f"{m['psnr']:.1f} dB  cf {m['cf_ratio']:.2f}  sat {m['sat_top_ratio']:.2f}"))
            for c, (img, text) in enumerate(cells):
                pil = Image.fromarray(img)
                if pil.height > pil.width:
                    pil = pil.rotate(90, expand=True)
                out.paste(pil.resize((cw, chh), Image.LANCZOS), (pad + c * (cw + pad), y))
                dr.text((pad + c * (cw + pad) + cw // 2, y + chh + 1), text, fill=(60, 60, 60), font=fs, anchor="ma")
        p = f"{prefix}_{length}ch_{start // per_file + 1}.png"
        out.save(p); paths.append(p)
    return paths


COL_BLURHASH = ("BlurHash 4x3 (28 ch)", lambda n, im, x, L: blurhash_img(im, 28, True))
COL_BLURHASH_BEST = ("BlurHash best grid", lambda n, im, x, L: blurhash_img(im, L))
COL_CURRENT = ("DCTHash (reference code)", lambda n, im, x, L: dh.decode(dh.encode(im, L), im.shape[1], im.shape[0])[..., :3])


def col_variant(v, label=None):
    return (label or v.name, lambda n, im, x, L: render(v.code(x, L), im.shape[1], im.shape[0], v.cfg["taper"]))


# ------------------------------------------------------------------ command line
PREV = os.path.join(ROOT, "format", "tables_v0_initial.json")
COL_INITIAL = ("DCTHash, initial tables", lambda n, im, x, L: dh.decode(dh.encode(im, L, _tables=PREV), im.shape[1], im.shape[0], _tables=PREV)[..., :3])
COL_NEW = ("DCTHash, colour revision", lambda n, im, x, L: dh.decode(dh.encode(im, L), im.shape[1], im.shape[0])[..., :3])


def holdout_images():
    """Photos that are neither in the training set (Picsum) nor in Kodak: scikit-image samples."""
    from skimage import data
    out = []
    for n in ("astronaut", "coffee", "chelsea", "rocket", "retina", "hubble_deep_field", "immunohistochemistry", "cat"):
        try:
            im = np.asarray(getattr(data, n)())[..., :3]
        except Exception:
            continue
        from PIL import Image
        pil = Image.fromarray(im); s = 384 / max(pil.size)
        out.append((n, np.asarray(pil.resize((max(1, round(pil.width * s)), max(1, round(pil.height * s))), Image.LANCZOS))))
    return out


def synthetic_images():
    yy, xx = np.mgrid[0:129, 0:257]
    syn = {"rgb ramp + sawtooth": np.stack([xx * 255 // 256, yy * 255 // 128, (xx + yy) % 256], -1).astype(np.uint8)}
    g2 = np.zeros((128, 192, 3), np.uint8); g2[..., 0] = np.linspace(255, 20, 192)[None]; g2[..., 2] = np.linspace(40, 255, 128)[:, None]; g2[..., 1] = 60
    syn["two-axis gradient"] = g2
    sh = np.full((128, 192, 3), 245, np.uint8); sh[20:90, 20:80] = (230, 20, 30); sh[50:120, 100:180] = (20, 60, 220)
    syn["flat shapes"] = sh
    syn["low-contrast wave"] = (128 + 6 * np.sin(xx[:128, :192] / 20.0)[..., None] + np.zeros(3)).astype(np.uint8)
    syn["solid teal"] = np.full((64, 96, 3), (20, 150, 140), np.uint8)
    syn["solid white"] = np.full((64, 96, 3), 255, np.uint8)
    syn["solid black"] = np.zeros((64, 96, 3), np.uint8)
    syn["solid red"] = np.full((64, 96, 3), (255, 0, 0), np.uint8)
    syn["mid grey"] = np.full((64, 96, 3), 128, np.uint8)
    return syn


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "sheets"
    out_dir = os.path.join(ROOT, "results", "colour")
    os.makedirs(out_dir, exist_ok=True)
    psnr8 = lambda a, b: -10 * np.log10(max(np.mean((a.astype(float) - b) ** 2), 1e-9) / 255 ** 2)
    if cmd == "sheets":           # original / BlurHash / initial tables / colour revision, all 24 Kodak images
        for L in (28, 40):
            cols = [COL_BLURHASH if L == 28 else COL_BLURHASH_BEST, COL_INITIAL, COL_NEW]
            print("\n".join(sheets(cols, L, os.path.join(out_dir, "kodak"), cell=(330, 220))))
    elif cmd == "holdout":        # images outside both Picsum (training) and Kodak (where the weights were picked by eye)
        H = holdout_images()
        lines = [HEAD.replace("| method |", "| chars / method |")]
        for L in (28, 40):
            for lab, fn in ((f"{L} BlurHash best grid", COL_BLURHASH_BEST[1]), (f"{L} DCTHash initial tables", COL_INITIAL[1]), (f"{L} DCTHash colour revision", COL_NEW[1])):
                lines.append(row(lab, [all_metrics(fn(n, im, None, L), im) for n, im in H]))
        lines.append(f"\n{len(H)} scikit-image sample photos: " + ", ".join(n for n, _ in H))
        open(os.path.join(out_dir, "holdout.md"), "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
        from PIL import Image
        cells = []
        for n, im in H:
            row_ = [im] + [fn(n, im, None, 28) for fn in (COL_BLURHASH[1], COL_INITIAL[1], COL_NEW[1])]
            cells.append(np.concatenate([np.asarray(Image.fromarray(c).resize((300, 200), Image.LANCZOS)) for c in row_], 1))
        Image.fromarray(np.concatenate(cells, 0)).save(os.path.join(out_dir, "holdout_28ch.png"))
        print("columns of holdout_28ch.png: original | BlurHash 4x3 | DCTHash initial | DCTHash colour revision")
    elif cmd == "synthetic":      # regression check for the solid-colour and gradient fixes
        lines = ["| image | chars | BlurHash 4x3 PSNR | DCTHash PSNR | profile | average colour from the hash |", "|---|---|---|---|---|---|"]
        for n, im in synthetic_images().items():
            for L in (20, 28, 52):
                s = dh.encode(im, L)
                d = dh.decode(s, im.shape[1], im.shape[0])[..., :3]
                nx, ny = {20: (4, 2), 28: (4, 3), 52: (6, 4)}[L]
                b = np.round(bh.decode_str(bh.encode_str(im / 255.0, nx, ny), im.shape[1], im.shape[0]) * 255)
                lines.append(f"| {n} | {L} | {psnr8(b, im):.1f} | {psnr8(d, im):.1f} | {dh.profile(s)} | {dh.average_color(s)} |")
        open(os.path.join(out_dir, "synthetic.md"), "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
