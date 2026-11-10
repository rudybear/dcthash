"""Linear-transform codecs sharing one quantisation/packing framework, so that the comparison
isolates (a) the basis and (b) the coefficient coding.

Pipeline (all in gamma-space YCbCr on a canonical 32x32 grid of the normalised image square):

    image --area--> 32x32 YCbCr --minus trained mean image--> project on basis functions
    (taken in the trained variance order, luma and chroma interleaved) --> quantise --> pack

basis   : 'dct' | 'legendre' | 'haar' (analytic) | 'pca' (KLT learned on the training set)
coding  : 'fixed' - fixed-length: trained bit allocation (greedy on trained distortion tables),
                    uniform mid-rise quantisers with trained clip ranges.  No header at all.
          'ac'    - 1 header byte (6 bits: global step q on a log scale; 2 bits: how many
                    coefficients are coded, 40/80/160/320) + static arithmetic coding of uniformly
                    quantised coefficients under per-coefficient Laplacian priors (scales trained).
                    The encoder searches the finest q that fits the budget (and a dead-zone offset).

Decoder: de-quantise, sum K<=192 basis functions on a 32x32 grid, Catmull-Rom upsample.
The trained tables (models/transform_tables.npz) come from Picsum photos, disjoint from the test set.
"""
import functools, os
import numpy as np
from ..common import Codec, BitWriter, BitReader, rgb_to_ycc, ycc_to_rgb, resize, aspect_class
from ..bases import ANALYTIC, G

_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models", "transform_tables.npz")
T_BITS = 16
T = 1 << T_BITS
Q_LEVELS = 64
Q_MIN, Q_MAX = 0.03, 6.0
N_CLASSES = (40, 80, 160, 320)
K_MAX = 2000


@functools.lru_cache(maxsize=None)
def _tables():
    return dict(np.load(_MODEL))


@functools.lru_cache(maxsize=None)
def _model(cls, basis):
    t = _tables()
    p = f"{cls}/{basis}/"
    chan, idx = t[p + "chan"].astype(int), t[p + "idx"].astype(int)
    if basis == "pca":
        F = t[p + "F"].astype(np.float64)
    else:
        F = ANALYTIC[basis]()[idx]
    return dict(mean=t[f"{cls}/mean"].astype(np.float64), chan=chan, F=F, lap=t[p + "lap"].astype(np.float64),
                L=t[p + "L"].astype(np.float64), D=t[p + "D"].astype(np.float64))


def _analyse(image, m):
    H, W = image.shape[:2]
    x = rgb_to_ycc(resize(image, G, G, "area")).transpose(2, 0, 1).reshape(3, G * G) - m["mean"]
    return np.einsum("kp,kp->k", m["F"], x[m["chan"]])


def _synthesise(coef, m, w, h):
    x = m["mean"].copy()
    np.add.at(x, m["chan"][:len(coef)], coef[:, None] * m["F"][:len(coef)])
    img = ycc_to_rgb(x.reshape(3, G, G).transpose(1, 2, 0))
    return np.clip(resize(img, h, w, "cubic"), 0, 1)


# ------------------------------------------------------------------ fixed-length mode
@functools.lru_cache(maxsize=None)
def _allocation(cls, basis, nbits):
    D = _model(cls, basis)["D"]
    K, MB = D.shape
    b = np.zeros(K, dtype=int)
    gain = D[:, 0] - D[:, 1]
    for _ in range(nbits):
        i = int(np.argmax(gain))
        b[i] += 1
        gain[i] = D[i, b[i]] - D[i, b[i] + 1] if b[i] + 1 < MB else -1
    return b


# ------------------------------------------------------------------ arithmetic-coded mode
def _q_of(qi):
    return Q_MIN * (Q_MAX / Q_MIN) ** (qi / (Q_LEVELS - 1))


@functools.lru_cache(maxsize=4096)
def _ac_tables(cls, basis, qi):
    """Per coefficient: (K, cumulative-frequency array of length 2K+2, reconstruction offset)."""
    lap = _model(cls, basis)["lap"]
    q = _q_of(qi)
    out = []
    for b in lap:
        K = int(min(K_MAX, np.ceil(10 * b / q) + 1))
        m = np.arange(1, K + 1)
        p_side = 0.5 * np.exp(-(m - 0.5) * q / b) * (1 - np.exp(-q / b))
        p_side[-1] = 0.5 * np.exp(-(K - 0.5) * q / b)           # tail mass in the last bin
        p = np.concatenate([p_side[::-1], [1 - np.exp(-q / (2 * b))], p_side])
        f = np.floor(p * (T - len(p))).astype(np.int64) + 1
        f[K] += T - f.sum()
        cum = np.concatenate([[0], np.cumsum(f)])
        delta = 0.5 - b / q + 1.0 / np.expm1(q / b)             # centroid shift towards zero, in steps
        out.append((K, [int(v) for v in cum], float(delta)))
    return out


def _ac_encode(ks, tabs):
    low, rng = 0, 1
    for k, (K, cum, _) in zip(ks, tabs):
        s = int(k) + K
        low = low * T + rng * cum[s]
        rng = rng * (cum[s + 1] - cum[s])
    nb = T_BITS * len(ks)
    L = max(0, (nb - rng.bit_length()) // 8)
    while True:                                                 # shortest byte string inside [low, low+rng)
        sh = nb - 8 * L
        v = -((-low) >> sh) if sh >= 0 else low << -sh
        if (v << sh if sh >= 0 else v >> -sh) < low + rng and (sh >= 0 or True):
            if sh >= 0 or (v >> -sh) << -sh == v:
                return v.to_bytes(L, "big") if L else b""
        L += 1


def _ac_decode(data, tabs):
    n = len(tabs)
    nb = T_BITS * n
    sh = nb - 8 * len(data)
    V = int.from_bytes(data, "big")
    V = V << sh if sh >= 0 else V >> -sh
    low, rng = 0, 1
    ks = []
    for i, (K, cum, _) in enumerate(tabs):
        t = V >> (T_BITS * (n - i - 1))
        target = (t - low * T) // rng
        s = int(np.searchsorted(cum, target, side="right")) - 1
        s = max(0, min(2 * K, s))
        ks.append(s - K)
        low = low * T + rng * cum[s]
        rng = rng * (cum[s + 1] - cum[s])
    return ks


class TransformCodec(Codec):
    def __init__(self, basis="dct", coding="ac"):
        self.basis, self.coding = basis, coding
        self.name = f"{basis}_{coding}"
        tbl = {"dct": "cosines (analytic)", "legendre": "polynomials (analytic)",
               "haar": "Haar boxes (analytic)", "pca": "learned KLT basis, 320x1024 table per aspect class (~330 KiB at 8 bit)"}[basis]
        self.description = f"{basis.upper()} basis in YCbCr, trained ordering, " + \
            ("fixed-length trained bit allocation" if coding == "fixed" else "arithmetic-coded Laplacian-prior coefficients")
        self.decoder_tables_bytes = (320 * 1024 if basis == "pca" else 0) + 320 * (2 if coding == "ac" else 18) + 3 * 1024
        self.decoder_cost = f"sum of <=320 basis fns on 32x32 + bicubic upsample; basis = {tbl}" + \
            ("; + ~40-line range decoder" if coding == "ac" else "")
        self.js_practical = "yes" if basis != "pca" else "yes, but ships a basis table"

    # -- fixed
    def _enc_fixed(self, coef, cls, budget):
        m = _model(cls, self.basis)
        bits = _allocation(cls, self.basis, budget * 8)
        bw = BitWriter()
        for i in np.nonzero(bits)[0]:
            nl, L = 1 << bits[i], m["L"][i, bits[i]]
            bw.write(int(np.clip(np.floor((coef[i] + L) / (2 * L) * nl), 0, nl - 1)), int(bits[i]))
        return bw.tobytes()

    def _dec_fixed(self, payload, cls):
        m = _model(cls, self.basis)
        bits = _allocation(cls, self.basis, len(payload) * 8)
        br = BitReader(payload)
        coef = np.zeros(len(bits))
        for i in np.nonzero(bits)[0]:
            nl, L = 1 << bits[i], m["L"][i, bits[i]]
            coef[i] = (br.read(int(bits[i])) + 0.5) / nl * 2 * L - L
        return coef

    # -- ac
    def _quant(self, coef, cls, qi, rho, n):
        tabs = _ac_tables(cls, self.basis, qi)[:n]
        q = _q_of(qi)
        a = np.abs(coef[:n]) / q
        k = np.floor(a + 0.5 - rho).clip(0)
        k = np.minimum(k, [t[0] for t in tabs]) * np.sign(coef[:n])
        return k.astype(int), tabs

    def _dequant(self, ks, tabs, qi):
        q = _q_of(qi)
        ks = np.asarray(ks, dtype=float)
        d = np.array([t[2] for t in tabs])
        return np.sign(ks) * np.maximum(np.abs(ks) - d, 0) * q * (ks != 0)

    def _enc_ac(self, coef, cls, budget):
        best = None
        total = np.sum(coef ** 2)
        for ni, n in enumerate(N_CLASSES):
            for rho in (0.0, 0.15, 0.3):
                lo, hi = 0, Q_LEVELS - 1                        # finest step whose code fits
                while lo < hi:
                    mid = (lo + hi) // 2
                    ks, tabs = self._quant(coef, cls, mid, rho, n)
                    if len(_ac_encode(ks, tabs)) <= budget - 1: hi = mid
                    else: lo = mid + 1
                ks, tabs = self._quant(coef, cls, lo, rho, n)
                data = _ac_encode(ks, tabs)
                if len(data) > budget - 1:
                    continue
                err = total - np.sum(coef[:n] ** 2) + np.sum((self._dequant(ks, tabs, lo) - coef[:n]) ** 2)
                if best is None or err < best[0]:
                    best = (err, bytes([(lo << 2) | ni]) + data)
        assert best is not None, "budget too small"
        return best[1]

    def _dec_ac(self, payload, cls):
        qi, n = payload[0] >> 2, N_CLASSES[payload[0] & 3]
        tabs = _ac_tables(cls, self.basis, qi)[:n]
        return self._dequant(_ac_decode(payload[1:], tabs), tabs, qi)

    def encode(self, image, budget_bytes):
        H, W = image.shape[:2]
        cls = aspect_class(W, H)
        coef = _analyse(image, _model(cls, self.basis))
        out = self._enc_fixed(coef, cls, budget_bytes) if self.coding == "fixed" else self._enc_ac(coef, cls, budget_bytes)
        assert len(out) <= budget_bytes
        return out

    def decode(self, payload, w, h):
        cls = aspect_class(w, h)
        coef = self._dec_fixed(payload, cls) if self.coding == "fixed" else self._dec_ac(payload, cls)
        return _synthesise(coef, _model(cls, self.basis), w, h)
