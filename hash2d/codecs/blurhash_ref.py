"""Faithful BlurHash (https://github.com/woltapp/blurhash) re-implementation, vectorised.

Algorithm identical to the reference: linear-light DCT factors, DC stored as 24-bit sRGB,
AC quantised to 19 levels/channel with a sqrt companding curve and a shared max, base83.
tests/test_blurhash.py checks string-equality against the `blurhash` pip package.

Byte accounting: a base83 string of N chars carries N*log2(83) bits, so we serialise the
string as the big integer it represents in ceil(N*log2(83)/8) bytes (round-trips exactly).
For a byte budget the encoder tries every (nx, ny) grid that fits and keeps the one with
the lowest sRGB MSE - i.e. this is BlurHash at its best, not at its default.
"""
import math
import numpy as np
from ..common import Codec, srgb_to_linear, linear_to_srgb, resize, fit_size

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz#$%*+,-.:;=?@[]^_{|}~"
_LOG83 = math.log2(83)


def chars_to_nbytes(n_chars):
    return math.ceil(n_chars * _LOG83 / 8)


def _sign_pow(v, e):
    return np.sign(v) * np.abs(v) ** e


def _basis(n, size):
    return np.cos(np.pi * np.arange(n)[:, None] * np.arange(size)[None, :] / size)  # (n, size)


def encode_str(img, nx, ny, linear=True):
    """img: (H, W, 3) sRGB float in [0,1] (as if from 8-bit).  Returns the BlurHash string."""
    H, W = img.shape[:2]
    lin = np.round(img * 255) / 255
    if linear:
        lin = srgb_to_linear(lin)
    f = np.einsum("jy,yxc,ix->jic", _basis(ny, H), lin, _basis(nx, W), optimize=True) / (W * H)
    f[:] *= 2
    f[0, 0] /= 2
    f = f.reshape(ny * nx, 3)
    dc, ac = f[0], f[1:]
    out = [(nx - 1) + (ny - 1) * 9]
    if len(ac):
        qmax = int(max(0, min(82, math.floor(np.abs(ac).max() * 166 - 0.5))))
        amax = (qmax + 1) / 166
    else:
        qmax, amax = 0, 1.0
    digits = _enc83(out[0], 1) + _enc83(qmax, 1)
    dc8 = np.round((linear_to_srgb(dc) if linear else np.clip(dc, 0, 1)) * 255).astype(int)  # reference: (int)(x*255+0.5)
    digits += _enc83(int((dc8[0] << 16) + (dc8[1] << 8) + dc8[2]), 4)
    if len(ac):
        q = np.clip(np.floor(_sign_pow(ac / amax, 0.5) * 9 + 9.5), 0, 18).astype(int)
        for r, g, b in q:
            digits += _enc83(int(r * 361 + g * 19 + b), 2)
    return digits


def _enc83(v, n):
    s = ""
    for i in range(1, n + 1):
        s += ALPHABET[(v // 83 ** (n - i)) % 83]
    return s


def _dec83(s):
    v = 0
    for ch in s:
        v = v * 83 + ALPHABET.index(ch)
    return v


def decode_str(s, w, h, punch=1.0, linear=True):
    size = _dec83(s[0])
    ny, nx = size // 9 + 1, size % 9 + 1
    assert len(s) == 4 + 2 * nx * ny
    amax = (_dec83(s[1]) + 1) / 166 * punch
    dcv = _dec83(s[2:6])
    cols = np.zeros((nx * ny, 3))
    cols[0] = np.array([dcv >> 16, (dcv >> 8) & 255, dcv & 255]) / 255
    if linear:
        cols[0] = srgb_to_linear(cols[0])
    for k in range(1, nx * ny):
        v = _dec83(s[4 + 2 * k:6 + 2 * k])
        q = np.array([v // 361, (v // 19) % 19, v % 19])
        cols[k] = _sign_pow((q - 9) / 9, 2.0) * amax
    cols = cols.reshape(ny, nx, 3)
    lin = np.einsum("jy,jic,ix->yxc", _basis(ny, h), cols, _basis(nx, w), optimize=True)
    return linear_to_srgb(lin) if linear else np.clip(lin, 0, 1)


def str_to_bytes(s):
    return _dec83(s).to_bytes(chars_to_nbytes(len(s)), "big")


def bytes_to_str(b):
    # string length is recoverable from the byte length: N = 4 + 2*n, nbytes strictly increasing in n
    n_chars = next(N for N in range(6, 4 + 2 * 81 + 1, 2) if chars_to_nbytes(N) == len(b))
    return _enc83(int.from_bytes(b, "big"), n_chars)


class BlurHash(Codec):
    name = "blurhash"
    description = "BlurHash reference (linear-light DCT, 19-level AC), best (nx,ny) per image"
    decoder_tables_bytes = 0
    decoder_cost = "nx*ny cosine products per pixel (separable); ~60 lines"
    js_practical = "yes (reference)"

    def __init__(self, search=True, linear=True):
        self.search, self.linear = search, linear
        if not linear:
            self.name = "blurhash_gamma"
            self.description = "Ablation: BlurHash with the DCT taken in gamma (sRGB) space instead of linear light"
        if not search:
            self.name += "_default"
            self.description = "BlurHash, grid shape picked from the aspect ratio only (no per-image search)"

    @staticmethod
    def max_components(budget_bytes):
        n = 0
        while chars_to_nbytes(4 + 2 * (n + 1)) <= budget_bytes and n < 81:
            n += 1
        return n

    def candidates(self, budget_bytes, w, h):
        n = self.max_components(budget_bytes)
        assert n >= 1, "budget too small for BlurHash"
        c = [(nx, ny) for nx in range(1, 10) for ny in range(1, 10) if nx * ny <= n]
        # keep only maximal grids (no other candidate dominates in both axes)
        c = [(a, b) for a, b in c if not any((a2 >= a and b2 >= b and (a2, b2) != (a, b)) for a2, b2 in c)]
        if not self.search:  # "default-like": the maximal grid whose shape is closest to the image aspect
            c = [min(c, key=lambda t: (abs(math.log((t[0] / t[1]) / (w / h))), -t[0] * t[1]))]
        return c

    def encode(self, image, budget_bytes):
        H, W = image.shape[:2]
        small = resize(image, *fit_size(H, W, 128))  # BlurHash encoders work on thumbnails in practice
        best = None
        for nx, ny in self.candidates(budget_bytes, W, H):
            s = encode_str(small, nx, ny, self.linear)
            err = np.mean((decode_str(s, small.shape[1], small.shape[0], linear=self.linear) - small) ** 2)
            if best is None or err < best[0]:
                best = (err, s)
        b = str_to_bytes(best[1])
        assert len(b) <= budget_bytes
        return b

    def decode(self, payload, w, h):
        return decode_str(bytes_to_str(payload), w, h, linear=self.linear)
