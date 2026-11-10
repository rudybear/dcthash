"""Shared helpers: colour transforms, resampling matrices, bit packing, codec base class.

Images are float32 numpy arrays of shape (H, W, 3), sRGB gamma-encoded, in [0, 1].
All *decoders* in this package are numpy-only on purpose (a placeholder decoder has to
be portable); only some *encoders* use torch for fitting.
"""
import functools
import numpy as np

# ---------------------------------------------------------------- colour
_YCC = np.array([[0.299, 0.587, 0.114],
                 [-0.168736, -0.331264, 0.5],
                 [0.5, -0.418688, -0.081312]], dtype=np.float64)
_YCC_INV = np.linalg.inv(_YCC)


def rgb_to_ycc(img):
    return img @ _YCC.T


def ycc_to_rgb(img):
    return img @ _YCC_INV.T


def srgb_to_linear(v):
    v = np.asarray(v, dtype=np.float64)
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(v):
    v = np.clip(v, 0.0, 1.0)
    return np.where(v <= 0.0031308, v * 12.92, 1.055 * v ** (1 / 2.4) - 0.055)


# ---------------------------------------------------------------- resampling
@functools.lru_cache(maxsize=256)
def area_matrix(n_in, n_out):
    """(n_out, n_in) box-filter (area) resampling matrix; anti-aliased downsampling."""
    A = np.zeros((n_out, n_in))
    s = n_in / n_out
    for j in range(n_out):
        lo, hi = j * s, (j + 1) * s
        for i in range(int(np.floor(lo)), min(n_in, int(np.ceil(hi)))):
            A[j, i] = max(0.0, min(hi, i + 1) - max(lo, i))
    return A / A.sum(1, keepdims=True)


@functools.lru_cache(maxsize=256)
def cubic_matrix(n_in, n_out):
    """(n_out, n_in) Catmull-Rom interpolation matrix (pixel-centre aligned, edge clamped)."""
    A = np.zeros((n_out, n_in))
    pos = (np.arange(n_out) + 0.5) * n_in / n_out - 0.5
    i0 = np.floor(pos).astype(int)
    t = pos - i0
    w = [(-t ** 3 + 2 * t ** 2 - t) / 2, (3 * t ** 3 - 5 * t ** 2 + 2) / 2,
         (-3 * t ** 3 + 4 * t ** 2 + t) / 2, (t ** 3 - t ** 2) / 2]
    for k in range(4):
        idx = np.clip(i0 + k - 1, 0, n_in - 1)
        np.add.at(A, (np.arange(n_out), idx), w[k])
    return A


def resize(img, h, w, kind="area"):
    """Separable resize of (H, W, C) or (H, W) array with a cached matrix per axis."""
    H, W = img.shape[:2]
    mk = area_matrix if kind == "area" else cubic_matrix
    Ah = mk(H, h) if (kind != "area" or H >= h) else cubic_matrix(H, h)
    Aw = mk(W, w) if (kind != "area" or W >= w) else cubic_matrix(W, w)
    return np.einsum("yh,hw...,xw->yx...", Ah, img, Aw, optimize=True)


def fit_size(h, w, max_side):
    s = max_side / max(h, w)
    if s >= 1:
        return h, w
    return max(1, round(h * s)), max(1, round(w * s))


def aspect_class(w, h):
    r = w / h
    return "land" if r > 1.15 else ("port" if r < 0.87 else "sq")


# ---------------------------------------------------------------- bit packing
class BitWriter:
    def __init__(self):
        self.v, self.n = 0, 0

    def write(self, value, bits):
        value = int(value)
        assert 0 <= value < (1 << bits), (value, bits)
        self.v = (self.v << bits) | value
        self.n += bits

    def tobytes(self):
        pad = (-self.n) % 8
        return ((self.v << pad)).to_bytes((self.n + pad) // 8, "big")


class BitReader:
    def __init__(self, data):
        self.v = int.from_bytes(data, "big")
        self.n = len(data) * 8

    def read(self, bits):
        assert bits <= self.n, "payload too short"
        self.n -= bits
        return (self.v >> self.n) & ((1 << bits) - 1)


# ---------------------------------------------------------------- codec base
class Codec:
    """Common interface.  encode() must return at most `budget_bytes` bytes;
    decode() must work from those bytes plus the output size only."""
    name = "base"
    description = ""
    decoder_tables_bytes = 0      # constant tables the decoder must ship
    decoder_cost = ""             # human-readable complexity note
    js_practical = "yes"

    def encode(self, image, budget_bytes) -> bytes:
        raise NotImplementedError

    def decode(self, payload: bytes, w: int, h: int) -> np.ndarray:
        raise NotImplementedError
