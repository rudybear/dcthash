"""Adaptive-primitive codecs fitted by gradient descent (torch, GPU if available) with
quantisation-aware training; the decoders are plain numpy.

modes
  'add'     anisotropic Gaussian splats, additive over a background colour (signed YCbCr amplitudes)
            per splat 32 bits: x5 y5 | sigma_major3 sigma_minor3 theta3 | Y5 Cb4 Cr4
  'norm'    anisotropic Gaussians with normalised (weighted-average) blending -> can form soft edges
            same 32-bit layout, colours absolute
  'voronoi' soft Voronoi cells + palette: sites x5 y5 + colour Y5 Cb4 Cr4 = 23 bits, one global
            sharpness (softmax temperature) in the header

All parameters live in [0,1] (sigmoid of a free variable), are quantised to their bit depth with a
straight-through estimator in the second half of the fit, and stored as integers.
"""
import numpy as np
from ..common import Codec, BitWriter, BitReader, rgb_to_ycc, ycc_to_rgb, resize, fit_size, _YCC_INV

SIG_MIN, SIG_MAX = 0.025, 0.7
LAYOUT = {
    "add": dict(bits=[5, 5, 3, 3, 3, 5, 4, 4], header=16),
    "norm": dict(bits=[5, 5, 3, 3, 3, 5, 4, 4], header=16),
    "voronoi": dict(bits=[5, 5, 5, 4, 4], header=8),
    "resid": dict(bits=[5, 5, 3, 3, 3, 5, 4, 4], header=0),      # used by the hybrid codec (no background)
}
BG_BITS = [6, 5, 5]
W0 = 0.03                                                        # background weight in 'norm' mode


def n_prims(mode, budget_bytes):
    L = LAYOUT[mode]
    return max(0, (budget_bytes * 8 - L["header"]) // sum(L["bits"]))


def _grid(w, h, xp):
    s = max(w, h)
    x = (xp.arange(w) + 0.5) / s
    y = (xp.arange(h) + 0.5) / s
    return x, y, w / s, h / s


def render(mode, U, bg, w, h, xp=np, extra=None):
    """U: (..., n, F) parameters in [0,1]; bg: (..., 3) in [0,1] (YCbCr-coded) or None.
    Works for numpy (decoder) and torch (fitting); leading batch dims allowed.  Returns RGB (..., h, w, 3)."""
    x, y, ax, ay = _grid(w, h, xp)
    if xp is not np:
        x, y = x.to(U), y.to(U)
    px = ((U[..., 0] * 1.2 - 0.1) * ax)[..., None, None]         # (..., n, 1, 1)
    py = ((U[..., 1] * 1.2 - 0.1) * ay)[..., None, None]
    dx = x[None, :] - px                                         # (..., n, 1, w)
    dy = y[:, None] - py                                         # (..., n, h, 1)
    M = xp.asarray(_YCC_INV.T) if xp is np else xp.tensor(_YCC_INV.T, dtype=U.dtype, device=U.device)
    if mode == "voronoi":
        col = xp.stack([U[..., 2], U[..., 3] * 0.8 - 0.4, U[..., 4] * 0.8 - 0.4], -1)   # (..., n, 3)
        beta = 10.0 * 60.0 ** extra                              # sharpness, extra in [0,1], shape (...,)
        logit = -(dx ** 2 + dy ** 2) * beta[..., None, None, None]
        logit = logit - logit.max(-3, keepdims=True) if xp is np else logit - logit.amax(-3, keepdim=True)
        wgt = xp.exp(logit)
        wgt = wgt / wgt.sum(-3, keepdims=True) if xp is np else wgt / wgt.sum(-3, keepdim=True)
        ycc = xp.einsum("...nhw,...nc->...hwc", wgt, col)
        return ycc @ M
    ratio = np.log(SIG_MAX / SIG_MIN)
    s1 = SIG_MIN * xp.exp(U[..., 2] * ratio)[..., None, None]
    s2 = SIG_MIN * xp.exp(U[..., 3] * ratio)[..., None, None]
    th = (U[..., 4] * np.pi * 7 / 8)[..., None, None]            # 8 levels cover [0, pi)
    c, s = xp.cos(th), xp.sin(th)
    a = (c * dx + s * dy) / s1
    b = (-s * dx + c * dy) / s2
    wgt = xp.exp(-0.5 * (a * a + b * b))                         # (..., n, h, w)
    if mode in ("add", "resid"):
        t = U[..., 5:8] * 2 - 1
        amp = t * xp.abs(t) * (xp.asarray([1.0, 0.5, 0.5]) if xp is np else xp.tensor([1.0, 0.5, 0.5]).to(U))
        ycc = xp.einsum("...nhw,...nc->...hwc", wgt, amp)
        if bg is not None:
            ycc = ycc + _bg_ycc(bg, xp)[..., None, None, :]
        return ycc @ M
    col = xp.stack([U[..., 5], U[..., 6] * 0.8 - 0.4, U[..., 7] * 0.8 - 0.4], -1)
    num = xp.einsum("...nhw,...nc->...hwc", wgt, col) + W0 * _bg_ycc(bg, xp)[..., None, None, :]
    den = wgt.sum(-3)[..., None] + W0
    return (num / den) @ M


def _bg_ycc(bg, xp):
    return xp.stack([bg[..., 0], bg[..., 1] * 0.8 - 0.4, bg[..., 2] * 0.8 - 0.4], -1)


def fit(mode, target, n, restarts=32, steps=(500, 400), seed=0, lr=0.05, base=None):
    """Fit n primitives to target (h, w, 3) RGB numpy (optionally on top of a fixed `base` image).  Returns integer codes (n, F), bg codes (3,) or
    None, extra code (int) or None, and the quantised-render MSE."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator().manual_seed(seed)
    h, w = target.shape[:2]
    tgt = torch.tensor(target, dtype=torch.float32, device=dev)
    base_t = torch.tensor(base, dtype=torch.float32, device=dev) if base is not None else 0.0
    bits = torch.tensor(LAYOUT[mode]["bits"], dtype=torch.float32, device=dev)
    lv = 2 ** bits - 1
    F = len(bits)
    R = restarts
    init = torch.rand(R, n, F, generator=g) * 0.8 + 0.1
    ycc = rgb_to_ycc(target)
    ci = 2 if mode == "voronoi" else 5
    if mode in ("norm", "voronoi"):                              # colours initialised from the image under each site
        xi = (init[..., 0].numpy() * 1.2 - 0.1).clip(0, 0.999) * w
        yi = (init[..., 1].numpy() * 1.2 - 0.1).clip(0, 0.999) * h
        c = ycc[yi.astype(int), xi.astype(int)]
        init[..., ci] = torch.tensor(c[..., 0]).float().clamp(0.02, 0.98)
        init[..., ci + 1:ci + 3] = torch.tensor((c[..., 1:] + 0.4) / 0.8).float().clamp(0.02, 0.98)
    else:
        init[..., 5:8] = 0.5 + 0.05 * torch.randn(R, n, 3, generator=g)
    if mode != "voronoi":
        init[..., 2:4] = 0.35 + 0.3 * torch.rand(R, n, 2, generator=g)
    P = torch.logit(init.clamp(1e-3, 1 - 1e-3)).to(dev).requires_grad_(True)
    params = [P]
    bgP = exP = None
    bglv = torch.tensor([2.0 ** b - 1 for b in BG_BITS], device=dev)
    if mode in ("add", "norm"):
        m = ycc.reshape(-1, 3).mean(0)
        b0 = torch.tensor([m[0], (m[1] + 0.4) / 0.8, (m[2] + 0.4) / 0.8], dtype=torch.float32).clamp(0.02, 0.98)
        bgP = torch.logit(b0).repeat(R, 1).to(dev).requires_grad_(True)
        params.append(bgP)
    if mode == "voronoi":
        exP = torch.zeros(R, device=dev).requires_grad_(True)
        params.append(exP)
    opt = torch.optim.Adam(params, lr=lr)

    def ste(u, levels, on):
        if not on:
            return u
        return u + (torch.round(u * levels) / levels - u).detach()

    def forward(quant):
        U = ste(torch.sigmoid(P), lv, quant)
        bg = ste(torch.sigmoid(bgP), bglv, quant) if bgP is not None else None
        ex = ste(torch.sigmoid(exP), 15.0, quant) if exP is not None else None
        out = render(mode, U, bg, w, h, xp=torch, extra=ex) + base_t
        return ((out.clamp(0, 1) - tgt) ** 2).mean((1, 2, 3)) + 0.1 * ((out - out.clamp(0, 1)) ** 2).mean((1, 2, 3))

    total = sum(steps)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, total, eta_min=lr * 0.05)
    for it in range(total):
        opt.zero_grad()
        forward(it >= steps[0]).sum().backward()
        opt.step(); sched.step()
    with torch.no_grad():
        loss = forward(True)
        r = int(torch.argmin(loss))
        codes = torch.round(torch.sigmoid(P[r]) * lv).long().cpu().numpy()
        bgc = torch.round(torch.sigmoid(bgP[r]) * bglv).long().cpu().numpy() if bgP is not None else None
        exc = int(torch.round(torch.sigmoid(exP[r]) * 15)) if exP is not None else None
    return codes, bgc, exc, float(loss[r])


def pack_prims(bw, mode, codes):
    for row in codes:
        for v, b in zip(row, LAYOUT[mode]["bits"]):
            bw.write(int(v), b)


def unpack_prims(br, mode, n):
    bits = LAYOUT[mode]["bits"]
    codes = np.array([[br.read(b) for b in bits] for _ in range(n)], dtype=np.float64).reshape(n, len(bits))
    return codes / (2.0 ** np.array(bits) - 1)


class SplatCodec(Codec):
    def __init__(self, mode="add", restarts=32):
        self.mode, self.restarts = mode, restarts
        self.name = {"add": "gauss_add", "norm": "gauss_norm", "voronoi": "soft_voronoi"}[mode]
        self.description = {
            "add": "Fitted anisotropic Gaussian splats, additive over background (32 bits/splat)",
            "norm": "Fitted anisotropic Gaussians, normalised blending (32 bits/splat)",
            "voronoi": "Fitted soft-Voronoi sites + colours (23 bits/site), global sharpness"}[mode]
        self.decoder_cost = "n primitives x 1 exp per pixel (not separable); ~40 lines" + \
            ("; needs softmax over sites" if mode == "voronoi" else "")
        self.js_practical = "decoder yes (n<=21 exps/pixel on a 32x32 canvas, then upscale); ENCODER needs iterative optimisation"

    def encode(self, image, budget_bytes):
        H, W = image.shape[:2]
        small = resize(image, *fit_size(H, W, 96))
        n = n_prims(self.mode, budget_bytes)
        assert n >= 1
        codes, bg, ex, _ = fit(self.mode, small, n, restarts=self.restarts)
        bw = BitWriter()
        if bg is not None:
            for v, b in zip(bg, BG_BITS): bw.write(int(v), b)
        if ex is not None:
            bw.write(ex, 8)
        pack_prims(bw, self.mode, codes)
        out = bw.tobytes()
        assert len(out) <= budget_bytes
        return out

    def decode(self, payload, w, h):
        br = BitReader(payload)
        bg = ex = None
        if self.mode in ("add", "norm"):
            bg = np.array([br.read(b) / (2.0 ** b - 1) for b in BG_BITS])
        else:
            ex = np.array(br.read(8) / 15.0)
        n = n_prims(self.mode, len(payload))
        U = unpack_prims(br, self.mode, n)
        return np.clip(render(self.mode, U, bg, w, h, xp=np, extra=ex), 0, 1)
