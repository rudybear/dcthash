"""Experiment: LPIPS as a loss term for DCTHash (README, "LPIPS as a loss").

    python scripts/explore_lpips.py formulations     # which LPIPS formulation ranks placeholders sensibly?
    python scripts/explore_lpips.py search           # encoder-side per-image search (format unchanged)
    python scripts/explore_lpips.py tables           # table training: gradient-tuned M / S / g with an LPIPS loss
    python scripts/explore_lpips.py hyper            # LPIPS-selected training hyperparameters (chroma weight, prior)
    python scripts/explore_lpips.py sheets           # visual sheets -> results/lpips/

The shipped format, tables and decoder are never modified here.  Everything that is trained uses Picsum only.
Judging uses metrics that are NOT the loss (DISTS, MS-SSIM, OKLab chroma error, colourfulness, PSNR) and the sheets.
"""
import functools, json, os, sys, time, warnings
import numpy as np
import torch
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from hash2d import dcthash as dh, metrics
from scripts.explore_colour import kodak, holdout_images, blurhash_img, PREV

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = os.path.join(ROOT, "results", "lpips")
os.makedirs(OUT, exist_ok=True)
BLUR = metrics.COLOUR_BLUR_SIGMA          # 8 px at 384x256: placeholder scale


# ------------------------------------------------------------------ perceptual networks
@functools.lru_cache(maxsize=None)
def _lpips(net):
    import lpips
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return lpips.LPIPS(net=net, verbose=False).to(DEV).eval().requires_grad_(False)


@functools.lru_cache(maxsize=None)
def _dists():
    from DISTS_pytorch import DISTS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return DISTS().to(DEV).eval().requires_grad_(False)


def t_img(a):
    """uint8 or float (H, W, 3) [or (B, H, W, 3)] -> float tensor (B, 3, H, W) in 0..1"""
    a = np.asarray(a)
    a = a.astype(np.float32) / 255 if a.dtype == np.uint8 else a.astype(np.float32)
    t = torch.tensor(a, device=DEV)
    return (t[None] if t.ndim == 3 else t).permute(0, 3, 1, 2)


def gblur(t, sigma):
    k = int(4 * sigma) | 1
    x = torch.arange(k, device=t.device, dtype=t.dtype) - k // 2
    g = torch.exp(-x ** 2 / (2 * sigma ** 2)); g = g / g.sum()
    C = t.shape[1]
    t = torch.nn.functional.pad(t, (k // 2, k // 2, k // 2, k // 2), mode="replicate")
    t = torch.nn.functional.conv2d(t, g.view(1, 1, 1, k).repeat(C, 1, 1, 1), groups=C)
    return torch.nn.functional.conv2d(t, g.view(1, 1, k, 1).repeat(C, 1, 1, 1), groups=C)


def lp(net, a, b):
    """LPIPS between float tensors in 0..1 -> (B,)"""
    return _lpips(net)(a * 2 - 1, b * 2 - 1).flatten()


def small(t, side=96):
    s = side / max(t.shape[-2:])
    return torch.nn.functional.interpolate(t, size=(max(16, round(t.shape[-2] * s)), max(16, round(t.shape[-1] * s))), mode="area")


FORMULATIONS = {       # name -> loss(rec, ref) on float tensors (B, 3, H, W); these are the candidate "LPIPS losses"
    "alex_sharp": lambda r, x: lp("alex", r, x),
    "vgg_sharp": lambda r, x: lp("vgg", r, x),
    "alex_blur": lambda r, x: lp("alex", r, gblur(x, BLUR)),
    "vgg_blur": lambda r, x: lp("vgg", r, gblur(x, BLUR)),
    "alex_small": lambda r, x: lp("alex", small(r), small(x)),
    "vgg_small": lambda r, x: lp("vgg", small(r), small(x)),
}


def judge(rec8, im8):
    """All metrics for one image.  rec8, im8: uint8 (H, W, 3)."""
    with torch.no_grad():
        r, x = t_img(rec8), t_img(im8)
        m = {"lpips_" + k: float(f(r, x)) for k, f in FORMULATIONS.items()}
        m["dists"] = float(_dists()(r, x))
        m["dists_blur"] = float(_dists()(r, gblur(x, BLUR)))
        from pytorch_msssim import ms_ssim
        m["ms_ssim"] = float(ms_ssim(r, x, data_range=1.0)) if min(r.shape[-2:]) > 160 else float("nan")
    a, b = rec8.astype(np.float64) / 255, im8.astype(np.float64) / 255
    m["psnr"] = metrics.psnr(a, b)
    m["psnr_blur"] = metrics.psnr(a, metrics.blur(b))
    m.update(metrics.colour_metrics(a, b))
    return m


COLS = ["psnr", "psnr_blur", "ms_ssim", "dists", "dists_blur", "dc_ok", "cf_ratio", "sat_top_ratio",
        "lpips_alex_sharp", "lpips_vgg_sharp", "lpips_alex_blur", "lpips_vgg_blur", "lpips_alex_small", "lpips_vgg_small"]
HEAD = "| method | " + " | ".join(COLS) + " |\n|---|" + "---|" * len(COLS)


def row(label, ms):
    return f"| {label} | " + " | ".join(f"{np.nanmean([m[k] for m in ms]):.2f}" if k.startswith("psnr") or k == "dc_ok" else f"{np.nanmean([m[k] for m in ms]):.3f}" for k in COLS) + " |"


def shipped(im8, L):
    return dh.decode(dh.encode(im8, L), im8.shape[1], im8.shape[0])[..., :3]


def initial(im8, L):
    return dh.decode(dh.encode(im8, L, _tables=PREV), im8.shape[1], im8.shape[0], _tables=PREV)[..., :3]


def datasets():
    return {"kodak": [(n, im) for n, im, _ in kodak()], "holdout": holdout_images()}


def cmd_formulations():
    lines = []
    for dname, imgs in datasets().items():
        lines += [f"\n### {dname} ({len(imgs)} images)\n", HEAD]
        for L in (20, 28, 40, 60, 80):
            if L == 28:
                lines.append(row("28 BlurHash 4x3", [judge(blurhash_img(im, 28, True), im) for _, im in imgs]))
            lines.append(row(f"{L} BlurHash best grid", [judge(blurhash_img(im, L), im) for _, im in imgs]))
            lines.append(row(f"{L} DCTHash MSE-trained tables", [judge(initial(im, L), im) for _, im in imgs]))
            lines.append(row(f"{L} DCTHash shipped", [judge(shipped(im, L), im) for _, im in imgs]))
            print("\n".join(lines[-4:]), flush=True)
    # sanity anchors: what does each formulation say about a plain Gaussian blur of the original, and a flat mean colour?
    imgs = datasets()["kodak"]
    lines += ["\n### anchors on Kodak: the original itself blurred, and its mean colour\n", HEAD]
    for s in (4, 8, 16, 32):
        lines.append(row(f"original blurred sigma={s}", [judge(np.clip(np.round(metrics.blur(im / 255.0, s) * 255), 0, 255).astype(np.uint8), im) for _, im in imgs]))
    lines.append(row("flat mean colour", [judge(np.broadcast_to(im.reshape(-1, 3).mean(0).round().astype(np.uint8), im.shape).copy(), im) for _, im in imgs]))
    open(os.path.join(OUT, "formulations.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[-7:]))


# ------------------------------------------------------------------ shared: torch decode of the shipped format
def cos_mats(W, H):
    f = torch.arange(16, device=DEV, dtype=torch.float32)[:, None]
    cx = torch.cos(np.pi * f * (torch.arange(W, device=DEV) + 0.5) / W)       # [u][x]
    cy = torch.cos(np.pi * f * (torch.arange(H, device=DEV) + 0.5) / H)       # [v][y]
    return cx, cy


YCC2RGB = torch.tensor([[1.0, 0.0, 1.402], [1.0, -0.344136, -0.714136], [1.0, 1.772, 0.0]])


def render_K(K, W, H):
    """K: (B, 3, 16, 16) [c][v][u] in 8-bit levels -> RGB float (B, 3, H, W) in 0..1 (clamped, differentiable)."""
    cx, cy = cos_mats(W, H)
    ycc = torch.einsum("vy,bcvu,ux->bcyx", cy, K, cx)
    rgb = torch.einsum("rc,bcyx->bryx", YCC2RGB.to(K), ycc)
    return (rgb / 255.0).clamp(0, 1)


def pack(a, smooth, sc, qs, bits, length):
    v, n = (a << 1) | int(smooth), 7
    if not smooth:
        v, n = (v << 2) | sc, 9
    for q, b in zip(qs, bits):
        if b:
            v = (v << b) | int(q); n += b
    assert n == 6 * length
    return "".join(dh.ALPHABET[(v >> (6 * (length - 1 - k))) & 63] for k in range(length))


# ------------------------------------------------------------------ 2. encoder-side per-image search
def search_encode(im8, length, loss_name, mse_weight=0.0, max_iter=60):
    """Start from the shipped encoder's choice and greedily change one quantiser index by +-1 at a time while the
    perceptual loss of the DECODED placeholder goes down.  The result is an ordinary DCTHash string."""
    t = dh.load_tables()
    X = dh.analyse(im8, t["cos"])
    H, W = im8.shape[:2]
    best = None
    for smooth, sc in [(False, 0), (False, 1), (False, 2), (False, 3), (True, 0)]:
        qs, bits, E = dh._candidate(t, X, length, smooth, sc)
        if best is None or E < best[0]:
            best = (E, smooth, sc, qs, bits)
    _, smooth, sc, qs, bits = best
    prof = t["smooth" if smooth else "photo"]
    coefs = prof["coefs"].tolist()

    def value(i, q):                       # reconstruction in 8-bit-level K units
        ch, u, v = coefs[i]
        b = bits[i]
        M = dh._mean(prof, smooth, i, ch, u, v)
        if b == 0:
            r = M << 16
        elif dh._kind(smooth, ch, u, v) == "c":
            r = dh._compand_r(q, b)
        else:
            L = dh.FULL if (smooth or (u == 0 and v == 0)) else dh._range(t, i, b, sc)
            r = dh._uniform_r(q, M, L, b)
        return (r / 65536.0) * (2 if u else 1) * (2 if v else 1) / 256.0

    ref = t_img(im8)
    f = FORMULATIONS[loss_name]
    cx, cy = cos_mats(W, H)
    loss_of = lambda imgs: f(imgs, ref.expand(len(imgs), -1, -1, -1)) + mse_weight * ((imgs - ref) ** 2).mean((1, 2, 3))
    qs = list(qs)
    K = torch.zeros(1, 3, 16, 16, device=DEV)
    for i, (ch, u, v) in enumerate(coefs):
        K[0, ch, v, u] = value(i, qs[i])
    with torch.no_grad():
        cur = float(loss_of(render_K(K, W, H)))
        start = cur
        for it in range(max_iter):
            moves, Ks = [], []
            for i, (ch, u, v) in enumerate(coefs):
                if not bits[i]:
                    continue
                for d in (-1, 1):
                    q = qs[i] + d
                    if 0 <= q < (1 << bits[i]):
                        k2 = K.clone(); k2[0, ch, v, u] = value(i, q)
                        moves.append((i, q)); Ks.append(k2)
            ls = torch.cat([loss_of(render_K(torch.cat(Ks[j:j + 64]), W, H)) for j in range(0, len(Ks), 64)])
            j = int(torch.argmin(ls))
            if float(ls[j]) >= cur - 1e-6:
                break
            cur = float(ls[j]); K = Ks[j]; qs[moves[j][0]] = moves[j][1]
    return pack(dh.aspect_code(W, H), smooth, sc, qs, bits, length), dict(iters=it, loss_start=start, loss_end=cur)


SEARCH_LOSSES = [("alex_sharp", 0.0), ("alex_blur", 0.0), ("vgg_blur", 0.0), ("alex_small", 0.0), ("vgg_small", 0.0), ("vgg_blur", 10.0)]


def cmd_search():
    path = os.path.join(OUT, "search_strings.json")
    store = json.load(open(path)) if os.path.exists(path) else {}
    lines = []
    for dname, imgs in datasets().items():
        for L in (28, 40):
            lines += [f"\n### {dname}, {L} chars: encoder-side search (decoder, tables and format unchanged)\n", HEAD.replace("| method |", "| loss minimised by the encoder |")]
            lines.append(row("none (shipped encoder)", [judge(shipped(im, L), im) for _, im in imgs]))
            for name, mw in SEARCH_LOSSES:
                tag = name + (f"+{mw:g}mse" if mw else "")
                ms, ts, its = [], [], []
                for n, im in imgs:
                    key = f"{dname}/{n}/{L}/{tag}"
                    if key not in store:
                        t0 = time.perf_counter()
                        s, info = search_encode(im, L, name, mw)
                        store[key] = dict(hash=s, seconds=time.perf_counter() - t0, **info)
                        json.dump(store, open(path, "w"), indent=0)
                    assert dh.is_valid(store[key]["hash"]) and len(store[key]["hash"]) == L
                    ms.append(judge(dh.decode(store[key]["hash"], im.shape[1], im.shape[0])[..., :3], im))
                    ts.append(store[key]["seconds"]); its.append(store[key]["iters"])
                lines.append(row(f"{tag} ({np.mean(its):.0f} index changes, {np.mean(ts):.1f} s/image)", ms))
                print(lines[-1], flush=True)
    open(os.path.join(OUT, "search.md"), "w").write("\n".join(lines) + "\n")


# ------------------------------------------------------------------ training data (Picsum only)
def picsum_crops(n=None, size=(192, 128), offset=0):
    """Centre 3:2 crops of Picsum photos -> (uint8 (N, H, W, 3), coefficients (N, 768) in T units)."""
    import glob
    from PIL import Image
    cache = os.path.join(ROOT, "data", f"lpips_picsum_{size[0]}x{size[1]}.npz")
    if not os.path.exists(cache):
        ims, xs = [], []
        for f in sorted(glob.glob(os.path.join(ROOT, "data", "picsum", "*.jpg"))):
            im = Image.open(f).convert("RGB")
            w, h = im.size
            ch = min(h, w * 2 // 3); cw = ch * 3 // 2
            im = im.crop(((w - cw) // 2, (h - ch) // 2, (w - cw) // 2 + cw, (h - ch) // 2 + ch)).resize(size, Image.LANCZOS)
            a = np.asarray(im); ims.append(a); xs.append(dh.analyse(a).reshape(-1) / 64.0)
        np.savez_compressed(cache, ims=np.stack(ims), xs=np.stack(xs))
    d = np.load(cache)
    sl = slice(offset, None if n is None else offset + n)
    return d["ims"][sl], d["xs"][sl]


# ------------------------------------------------------------------ 1a. gradient-tuned tables (M, S, g) with an LPIPS loss
class PhotoQuant(torch.nn.Module):
    """Differentiable model of the photo profile's de-quantisation.  Coefficient order and bit sequence are fixed
    (shipped); trainable: centres M and scales S of the non-DC coefficients and the 2x8 loading factors g.  The
    quantiser index is computed with the current parameters (as a real encoder would) and treated as a constant
    in the backward pass, so gradients reach M, S and g through the reconstruction points."""
    def __init__(self, tables_path=None):
        super().__init__()
        t = json.load(open(tables_path or dh._TABLES))
        self.raw = t
        p = t["photo"]
        c = np.array(p["coefs"])
        self.ch, self.u, self.v = [torch.tensor(c[:, k], device=DEV) for k in range(3)]
        self.idx = self.ch * 256 + self.v * 16 + self.u
        self.dc = (self.u == 0) & (self.v == 0)
        self.seq = np.array(p["seq"])
        self.M0 = torch.tensor(p["M"], dtype=torch.float32, device=DEV)
        self.S0 = torch.tensor(p["S"], dtype=torch.float32, device=DEV)
        self.dM = torch.nn.Parameter(torch.zeros(len(c), device=DEV))            # M = M0 + dM * S0
        self.lS = torch.nn.Parameter(torch.zeros(len(c), device=DEV))            # S = S0 * exp(lS)
        self.lg = torch.nn.Parameter(torch.log(torch.tensor(t["g"], dtype=torch.float32, device=DEV)))
        self.alpha = torch.where(self.u > 0, 2.0, 1.0) * torch.where(self.v > 0, 2.0, 1.0)
        self.w = torch.tensor(dh.CWEIGHT, dtype=torch.float32, device=DEV)[self.ch] * self.alpha

    def params(self):
        M = torch.where(self.dc, self.M0, self.M0 + self.dM * self.S0)
        S = torch.where(self.dc, self.S0, self.S0 * torch.exp(self.lS))
        return M, S, torch.exp(self.lg)

    def forward(self, x768, length):
        """x768: (B, 768) T units -> K (B, 3, 16, 16)"""
        bits = torch.tensor(np.bincount(self.seq[:6 * length - 9], minlength=len(self.idx)), device=DEV)
        x = x768[:, self.idx]
        M, S, g = self.params()
        bc = bits.clamp(min=1)
        gb = g[(self.ch > 0).long(), bc - 1]
        L0 = (S * gb / 256).clamp(min=1.0)
        lev = 2.0 ** bc
        best, best_e = None, None
        for sc in dh.SCALES:
            L = torch.where(self.dc, torch.full_like(L0, float(dh.FULL)), (L0 * sc / 4).clamp(min=1.0))
            q = torch.floor((x - M + L) * lev / (2 * L)).detach().clamp(min=0)
            q = torch.minimum(q, lev - 1)
            rec = M - L + (2 * q + 1) * L / lev
            cdc = self.dc & (self.ch > 0)                    # chroma DC: fixed companded quantiser, nearest level
            if cdc.any():
                for i in torch.nonzero(cdc).flatten().tolist():
                    b = int(bits[i])
                    if b:
                        n = torch.arange(1 << b, device=DEV) - ((1 << (b - 1)) - 1)
                        levs = dh.FULL * n * n.abs() / 4.0 ** (b - 1)
                        rec[:, i] = levs[torch.argmin((x[:, i:i + 1] - levs[None]).abs(), 1)]
            rec = torch.where(bits > 0, rec, M.expand_as(rec))
            e = (self.w * (rec - x) ** 2).sum(1).detach()
            if best is None:
                best, best_e = rec, e
            else:
                take = e < best_e
                best = torch.where(take[:, None], rec, best); best_e = torch.minimum(e, best_e)
        K = torch.zeros(x.shape[0], 768, device=DEV)
        K[:, self.idx] = best * self.alpha / 256.0
        return K.view(-1, 3, 16, 16)

    def export(self, path):
        M, S, g = [v.detach().cpu().numpy() for v in self.params()]
        t = json.loads(json.dumps(self.raw))
        t["photo"]["M"] = [int(round(float(v))) for v in M]
        t["photo"]["S"] = [int(min(65535, max(1, round(float(v))))) for v in S]
        t["g"] = [[int(max(1, round(float(v)))) for v in r] for r in g]
        json.dump(t, open(path, "w"), separators=(",", ":"))


TABLE_LOSSES = [("alex_blur", 0.0), ("vgg_blur", 0.0), ("alex_small", 0.0), ("vgg_small", 0.0), ("alex_small", 10.0)]


def tuned_path(tag):
    return os.path.join(OUT, f"tables_tuned_{tag}.json")


def cmd_tables(steps=1500, batch=32):
    ims, xs = picsum_crops()
    ims_t = torch.tensor(ims, device=DEV).permute(0, 3, 1, 2).float() / 255
    xs_t = torch.tensor(xs, dtype=torch.float32, device=DEV)
    H, W = ims.shape[1:3]
    print(len(ims), "Picsum crops", (W, H))
    lines = []
    for name, mw in TABLE_LOSSES:
        tag = name + (f"+{mw:g}mse" if mw else "")
        if not os.path.exists(tuned_path(tag)):
            torch.manual_seed(0)
            m = PhotoQuant()
            opt = torch.optim.Adam(m.parameters(), lr=0.01)
            f = FORMULATIONS[name]
            hist = []
            for it in range(steps):
                b = torch.randint(0, len(ims), (batch,), device=DEV)
                if torch.rand(()) < 0.5:                         # horizontal flip augmentation: mirror the image, negate odd-u coefficients
                    ref = ims_t[b].flip(-1)
                    sign = torch.where((torch.arange(768, device=DEV) % 16) % 2 == 1, -1.0, 1.0)
                    x = xs_t[b] * sign
                else:
                    ref, x = ims_t[b], xs_t[b]
                L = int(np.random.choice([20, 28, 40, 60]))
                rec = render_K(m(x, L), W, H)
                loss = (f(rec, ref) + mw * ((rec - ref) ** 2).mean((1, 2, 3))).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                hist.append(float(loss))
                if it % 250 == 0 or it == steps - 1:
                    print(f"  {tag} step {it}: loss {np.mean(hist[-100:]):.4f}", flush=True)
            m.export(tuned_path(tag))
            M, S, g = m.params()
            print(f"  {tag}: median S change x{float(torch.exp(m.lS[~m.dc]).median()):.2f}, g luma {[int(v) for v in g[0]]}, g chroma {[int(v) for v in g[1]]}")
    for dname, imgs in datasets().items():
        for L in (28, 40):
            lines += [f"\n### {dname}, {L} chars: tables with M, S, g tuned by gradient descent on an LPIPS loss (real integer codec)\n", HEAD.replace("| method |", "| training loss |")]
            lines.append(row("shipped tables", [judge(shipped(im, L), im) for _, im in imgs]))
            for name, mw in TABLE_LOSSES:
                tag = name + (f"+{mw:g}mse" if mw else "")
                tp = tuned_path(tag)
                hs = [dh.encode(im, L, _tables=tp) for _, im in imgs]
                n_smooth = sum(dh.profile(h) == "smooth" for h in hs)
                if n_smooth > len(imgs) // 4:      # the encoder's coefficient-MSE decision abandons the tuned photo tables: force them
                    hs = [dh.encode(im, L, _tables=tp, _photo_only=True) for _, im in imgs]
                    tag += f" (photo profile forced; otherwise {n_smooth}/{len(imgs)} images fall back to the smooth profile)"
                lines.append(row(tag, [judge(dh.decode(h, im.shape[1], im.shape[0], _tables=tp)[..., :3], im) for h, (_, im) in zip(hs, imgs)]))
                print(lines[-1], flush=True)
    open(os.path.join(OUT, "tables.md"), "w").write("\n".join(lines) + "\n")


# ------------------------------------------------------------------ 1b. LPIPS-selected training hyperparameters
HYPER = [("MSE (chroma x1, no prior)", dict(cw=1)), ("x2 r3 g0.5", dict(cw=2, rad=0.5, r0=3)), ("x4 r3 g1", dict(cw=4, rad=1.0, r0=3)),
         ("x8 r3 g1.5 (shipped setting)", dict(cw=8, rad=1.5, r0=3)), ("x8 r3 g2.5", dict(cw=8, rad=2.5, r0=3)), ("x16 r3 g1.5", dict(cw=16, rad=1.5, r0=3)),
         ("x8 r2 g2", dict(cw=8, rad=2.0, r0=2)), ("x8 r4 g1.5", dict(cw=8, rad=1.5, r0=4)), ("x1 r3 g1.5", dict(cw=1, rad=1.5, r0=3)),
         ("x8 no prior", dict(cw=8)), ("x2 r3 g1.5", dict(cw=2, rad=1.5, r0=3)), ("x4 r3 g2.5", dict(cw=4, rad=2.5, r0=3))]


def cmd_hyper(n_val=96, length=28):
    """Which (chroma weight, prior) would each LPIPS formulation have picked?  Selection data: Picsum crops."""
    from scripts.explore_colour import Variant, render
    X = np.load(os.path.join(ROOT, "data", "train_coefs.npy"))[::4] / 64.0
    ims, xs = picsum_crops(None, size=(384, 256))
    keep = [i for i, im in enumerate(ims) if metrics.colourfulness(metrics.blur(im / 255.0, BLUR)) > 8][:n_val]   # skip black-and-white photos
    ims, xs = ims[keep], xs[keep]
    lines = [f"Selection set: {len(ims)} Picsum photos (384x256), {length} chars, float simulation of the format.\n", HEAD.replace("| method |", "| training weights |")]
    res = {}
    for name, cfg in HYPER:
        v = Variant(name, X, **cfg)
        ms = [judge(render(v.code(x, length), im.shape[1], im.shape[0]), im) for im, x in zip(ims, xs)]
        res[name] = {k: float(np.nanmean([m[k] for m in ms])) for k in COLS}
        lines.append(row(name, ms)); print(lines[-1], flush=True)
    lines.append("\nSetting each criterion would pick (lowest is best for LPIPS, DISTS, chroma error; highest for PSNR, MS-SSIM):\n")
    for k in COLS:
        if k in ("cf_ratio", "sat_top_ratio"):
            continue
        pick = (max if k in ("psnr", "psnr_blur", "ms_ssim") else min)(res, key=lambda n: res[n][k])
        lines.append(f"- {k}: {pick}")
    open(os.path.join(OUT, "hyper.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[-13:]))


# ------------------------------------------------------------------ all lengths, for the least-bad candidates
def cmd_lengths():
    path = os.path.join(OUT, "search_strings.json")
    store = json.load(open(path))
    imgs = datasets()["kodak"]
    lines = ["Kodak, 24 images.  BlurHash with the reference encoder; `search` = encoder-side search minimising the named loss;",
             "`tables` = M, S, g tuned on the named loss.  LPIPS columns are losses for some rows: do not judge by them.\n", HEAD]
    tp = tuned_path("alex_blur")
    for L in (20, 28, 40, 60, 80):
        if L == 28:
            lines.append(row("28 BlurHash 4x3", [judge(blurhash_img(im, 28, True), im) for _, im in imgs]))
        lines.append(row(f"{L} BlurHash best grid", [judge(blurhash_img(im, L), im) for _, im in imgs]))
        lines.append(row(f"{L} DCTHash shipped", [judge(shipped(im, L), im) for _, im in imgs]))
        for name, mw in (("alex_blur", 0.0), ("vgg_blur", 10.0)):
            tag = name + (f"+{mw:g}mse" if mw else "")
            ms = []
            for n, im in imgs:
                key = f"kodak/{n}/{L}/{tag}"
                if key not in store:
                    t0 = time.perf_counter()
                    h, info = search_encode(im, L, name, mw)
                    store[key] = dict(hash=h, seconds=time.perf_counter() - t0, **info)
                    json.dump(store, open(path, "w"), indent=0)
                ms.append(judge(dh.decode(store[key]["hash"], im.shape[1], im.shape[0])[..., :3], im))
            lines.append(row(f"{L} search: {tag}", ms))
        lines.append(row(f"{L} tables: alex_blur", [judge(dh.decode(dh.encode(im, L, _tables=tp), im.shape[1], im.shape[0], _tables=tp)[..., :3], im) for _, im in imgs]))
        print("\n".join(lines[-5:]), flush=True)
    open(os.path.join(OUT, "lengths.md"), "w").write("\n".join(lines) + "\n")


# ------------------------------------------------------------------ sheets
def cmd_sheets():
    """original / BlurHash (reference encoder) / shipped DCTHash / LPIPS variants.  Captions: PSNR, colourfulness
    ratio, top-saturation ratio (none of them a training loss)."""
    from scripts import explore_colour as ec
    from PIL import Image
    store = json.load(open(os.path.join(OUT, "search_strings.json")))

    def searched(dname, tag):
        return (f"search: {tag}", lambda n, im, x, L: dh.decode(store[f"{dname}/{n}/{L}/{tag}"]["hash"], im.shape[1], im.shape[0])[..., :3])

    def tuned(tag):
        tp = tuned_path(tag)
        return (f"tables: {tag}", lambda n, im, x, L: dh.decode(dh.encode(im, L, _tables=tp, _photo_only=True), im.shape[1], im.shape[0], _tables=tp)[..., :3])

    for L in (28, 40):
        bhc = ("BlurHash 4x3 (28 ch)", lambda n, im, x, L: blurhash_img(im, 28, True)) if L == 28 else ("BlurHash best grid", lambda n, im, x, L: blurhash_img(im, L))
        cols = [bhc, ("DCTHash shipped", lambda n, im, x, L: shipped(im, L)), searched("kodak", "alex_sharp"), searched("kodak", "alex_blur"),
                searched("kodak", "vgg_small"), tuned("alex_small"), tuned("vgg_blur")]
        print("\n".join(ec.sheets(cols, L, os.path.join(OUT, "kodak"), cell=(246, 164))))
        # hold-out photos: one sheet per length
        H = holdout_images()
        cols[2:5] = [searched("holdout", "alex_sharp"), searched("holdout", "alex_blur"), searched("holdout", "vgg_small")]
        rows = []
        for n, im in H:
            cells = [im] + [fn(n, im, None, L) for _, fn in cols]
            rows.append(np.concatenate([np.asarray(Image.fromarray(c).resize((246, 164), Image.LANCZOS)) for c in cells], 1))
        Image.fromarray(np.concatenate(rows, 0)).save(os.path.join(OUT, f"holdout_{L}ch.png"))
        print(os.path.join(OUT, f"holdout_{L}ch.png"), "columns: original |", " | ".join(c[0] for c in cols))


if __name__ == "__main__":
    globals()["cmd_" + sys.argv[1]]()
