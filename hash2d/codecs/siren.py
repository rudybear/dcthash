"""Tiny implicit neural representation (SIREN / CPPN-style): an MLP (x, y) -> RGB with sine
activations whose *weights are the payload*, quantised to 6 bits each (quantisation-aware fit).

With one hidden layer this is literally a sum of H plane waves with free direction, frequency
and phase - an adaptive (non-harmonic) Fourier basis.  Two hidden layers add composition.
Header: 1 byte architecture id.
"""
import numpy as np
from ..common import Codec, BitWriter, BitReader, resize, fit_size

WBITS = 6
ARCHS = [(1, H) for H in range(1, 33)] + [(2, H) for H in range(2, 17)]
R_IN, R_B, R_HID, R_OUT = 14.0, np.pi, 4.0, 1.5
OUT_B = (-0.5, 1.5)


def n_params(arch):
    L, H = arch
    return 6 * H + 3 if L == 1 else H * H + 7 * H + 3


def _shapes(arch):
    L, H = arch
    sh = [("w", (H, 2), R_IN), ("b", (H,), R_B)]
    if L == 2:
        sh += [("w", (H, H), R_HID), ("b", (H,), R_B)]
    sh += [("w", (3, H), R_OUT), ("ob", (3,), None)]
    return sh


def _unflatten(u, arch, xp=np):
    """u: (..., P) in [0,1] -> list of real-valued tensors."""
    out, o = [], 0
    for kind, shape, rng in _shapes(arch):
        n = int(np.prod(shape))
        v = u[..., o:o + n].reshape(tuple(u.shape[:-1]) + shape)
        out.append(v * (OUT_B[1] - OUT_B[0]) + OUT_B[0] if kind == "ob" else (v * 2 - 1) * rng)
        o += n
    return out


def _coords(w, h, xp=np):
    s = max(w, h)
    x = ((xp.arange(w) + 0.5) / s) * 2 - w / s
    y = ((xp.arange(h) + 0.5) / s) * 2 - h / s
    if xp is np:
        X, Y = np.meshgrid(x, y)
    else:
        Y, X = xp.meshgrid(y, x, indexing="ij")
    return xp.stack([X.reshape(-1), Y.reshape(-1)], -1)          # (N, 2)


def forward(u, arch, w, h, xp=np):
    T = _unflatten(u, arch, xp)
    X = _coords(w, h, xp)
    if xp is not np:
        X = X.to(u)
    z = xp.sin(xp.einsum("nd,...hd->...nh", X, T[0]) + T[1][..., None, :])
    i = 2
    if arch[0] == 2:
        z = xp.sin(xp.einsum("...nh,...kh->...nk", z, T[2]) + T[3][..., None, :])
        i = 4
    out = xp.einsum("...nh,...ch->...nc", z, T[i]) + T[i + 1][..., None, :]
    return out.reshape(tuple(out.shape[:-2]) + (h, w, 3))


def fit(target, arch, restarts=32, steps=(700, 500), lr=0.03, seed=0):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator().manual_seed(seed)
    h, w = target.shape[:2]
    tgt = torch.tensor(target, dtype=torch.float32, device=dev)
    Pn = n_params(arch)
    init = 0.5 + (torch.rand(restarts, Pn, generator=g) - 0.5) * 0.5
    init[:, -3:] = torch.tensor((target.reshape(-1, 3).mean(0) - OUT_B[0]) / (OUT_B[1] - OUT_B[0])).float()
    P = torch.logit(init.clamp(1e-3, 1 - 1e-3)).to(dev).requires_grad_(True)
    opt = torch.optim.Adam([P], lr=lr)
    lv = 2.0 ** WBITS - 1
    total = sum(steps)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, total, eta_min=lr * 0.05)

    def loss_fn(quant):
        u = torch.sigmoid(P)
        if quant:
            u = u + (torch.round(u * lv) / lv - u).detach()
        out = forward(u, arch, w, h, xp=torch)
        return ((out.clamp(0, 1) - tgt) ** 2).mean((1, 2, 3)) + 0.1 * ((out - out.clamp(0, 1)) ** 2).mean((1, 2, 3))

    for it in range(total):
        opt.zero_grad()
        loss_fn(it >= steps[0]).sum().backward()
        opt.step(); sched.step()
    with torch.no_grad():
        loss = loss_fn(True)
        r = int(torch.argmin(loss))
        return torch.round(torch.sigmoid(P[r]) * lv).long().cpu().numpy(), float(loss[r])


class Siren(Codec):
    name = "siren"
    description = "Tiny sine-activation MLP (x,y)->RGB, weights quantised to 6 bits (QAT), 1-2 hidden layers"
    decoder_cost = "one tiny MLP evaluation per pixel (H sines, or H + H^2 mults for 2 layers); ~30 lines"
    js_practical = "decoder yes (evaluate at 32x32 and upscale); ENCODER needs iterative optimisation with restarts"

    def __init__(self, restarts=32):
        self.restarts = restarts

    def encode(self, image, budget_bytes):
        H, W = image.shape[:2]
        small = resize(image, *fit_size(H, W, 64))
        cap = (budget_bytes - 1) * 8 // WBITS
        best = None
        for L in (1, 2):
            fits = [a for a in ARCHS if a[0] == L and n_params(a) <= cap]
            if not fits:
                continue
            arch = fits[-1]
            codes, loss = fit(small, arch, restarts=self.restarts)
            if best is None or loss < best[0]:
                best = (loss, arch, codes)
        _, arch, codes = best
        bw = BitWriter()
        bw.write(ARCHS.index(arch), 8)
        for v in codes: bw.write(int(v), WBITS)
        out = bw.tobytes()
        assert len(out) <= budget_bytes
        return out

    def decode(self, payload, w, h):
        br = BitReader(payload)
        arch = ARCHS[br.read(8)]
        u = np.array([br.read(WBITS) for _ in range(n_params(arch))]) / (2.0 ** WBITS - 1)
        return np.clip(forward(u, arch, w, h), 0, 1)
