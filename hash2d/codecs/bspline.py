"""Tiny thumbnail + smooth upscaling, done properly: the payload is a low-res grid of *control
points* of a clamped cubic B-spline surface (luma grid gw x gh at 5 bits, chroma grid at lower
resolution and 4 bits), least-squares fitted to the image rather than naively down-sampled.

This also stands in for regular gradient meshes / bicubic patches: a regular mesh spends no
bits on vertex positions.

Header: 1 byte (gw-1, gh-1; 4 bits each).  The chroma grid is the largest proportional grid
that fits in the remaining bits, so the encoder's search over (gw, gh) also explores the
luma/chroma split.
"""
import functools
import numpy as np
from scipy.interpolate import BSpline
from ..common import Codec, BitWriter, BitReader, rgb_to_ycc, ycc_to_rgb, resize, fit_size

BY, BC = 5, 4
Y_RANGE = (-0.15, 1.15)
C_RANGE = (-0.4, 0.4)
FRACS = (1.0, 0.8, 0.67, 0.5, 0.4, 0.33, 0.25, 0.2, 0.125, 0.0)


@functools.lru_cache(maxsize=512)
def bspline_matrix(n_pix, g):
    """(n_pix, g) design matrix of a clamped uniform B-spline (degree min(3, g-1)) at pixel centres."""
    if g == 1:
        return np.ones((n_pix, 1))
    k = min(3, g - 1)
    t = np.concatenate([np.zeros(k), np.linspace(0, 1, g - k + 1), np.ones(k)])
    x = (np.arange(n_pix) + 0.5) / n_pix
    return BSpline.design_matrix(x, t, k).toarray()


def chroma_grid(gw, gh, budget_bytes):
    left = (budget_bytes - 1) * 8 - gw * gh * BY
    for f in FRACS:
        cw, ch = max(1, round(gw * f)), max(1, round(gh * f))
        if 2 * cw * ch * BC <= left:
            return cw, ch
    return None


def _q(v, lo, hi, bits):
    n = (1 << bits) - 1
    return np.clip(np.round((v - lo) / (hi - lo) * n), 0, n).astype(int)


def _dq(q, lo, hi, bits):
    return q / ((1 << bits) - 1) * (hi - lo) + lo


def _render(gy, gc, w, h):
    y = bspline_matrix(h, gy.shape[0]) @ gy @ bspline_matrix(w, gy.shape[1]).T
    c = np.einsum("yj,jic,xi->yxc", bspline_matrix(h, gc.shape[0]), gc, bspline_matrix(w, gc.shape[1]))
    return ycc_to_rgb(np.concatenate([y[..., None], c], axis=2))


class BSplineGrid(Codec):
    name = "bspline_grid"
    description = "Low-res control grid (5-bit luma, 4-bit sub-sampled chroma), cubic B-spline upscaling, LSQ fit"
    decoder_tables_bytes = 0
    decoder_cost = "separable cubic B-spline evaluation (16 taps/pixel) or just a GPU/bilinear-ish upscale of a tiny texture; ~50 lines"
    js_practical = "yes (simplest of all)"

    def encode(self, image, budget_bytes):
        H, W = image.shape[:2]
        small = resize(image, *fit_size(H, W, 96))
        h, w = small.shape[:2]
        ycc = rgb_to_ycc(small)
        best = None
        for gw in range(1, 17):
            for gh in range(1, 17):
                cg = chroma_grid(gw, gh, budget_bytes)
                if cg is None:
                    continue
                if not (0.3 <= (gw / gh) / (W / H) <= 3.4):
                    continue
                cw, ch = cg
                gy = np.linalg.pinv(bspline_matrix(h, gh)) @ ycc[..., 0] @ np.linalg.pinv(bspline_matrix(w, gw)).T
                gc = np.einsum("jy,yxc,ix->jic", np.linalg.pinv(bspline_matrix(h, ch)), ycc[..., 1:],
                               np.linalg.pinv(bspline_matrix(w, cw)))
                qy, qc = _q(gy, *Y_RANGE, BY), _q(gc, *C_RANGE, BC)
                rec = _render(_dq(qy, *Y_RANGE, BY), _dq(qc, *C_RANGE, BC), w, h)
                err = np.mean((np.clip(rec, 0, 1) - small) ** 2)
                if best is None or err < best[0]:
                    best = (err, gw, gh, qy, qc)
        assert best is not None, "budget too small"
        _, gw, gh, qy, qc = best
        bw = BitWriter()
        bw.write(gw - 1, 4); bw.write(gh - 1, 4)
        for v in qy.ravel(): bw.write(v, BY)
        for v in qc.ravel(): bw.write(v, BC)
        out = bw.tobytes()
        out = out + bytes(budget_bytes - len(out))      # decoder derives the chroma grid from the length
        assert len(out) == budget_bytes
        return out

    def decode(self, payload, w, h):
        br = BitReader(payload)
        gw, gh = br.read(4) + 1, br.read(4) + 1
        cw, ch = chroma_grid(gw, gh, len(payload))
        qy = np.array([br.read(BY) for _ in range(gw * gh)]).reshape(gh, gw)
        qc = np.array([br.read(BC) for _ in range(cw * ch * 2)]).reshape(ch, cw, 2)
        return np.clip(_render(_dq(qy, *Y_RANGE, BY), _dq(qc, *C_RANGE, BC), w, h), 0, 1)
