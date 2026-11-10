"""Hybrids and the per-image "best fit" meta-codec.

Hybrid  : arithmetic-coded DCT base at a reduced budget + k additive anisotropic Gaussian splats
          fitted to the residual (4 bytes each).  Header byte = k.  The encoder tries a few k
          (including k = 0) and keeps the best, so it can only lose 1 byte against the pure base.

BestFit : runs a list of codecs at (budget - 1) bytes, keeps the lowest-MSE result and prepends
          a 1-byte method tag.  The tag is paid for inside the budget.
"""
import numpy as np
from ..common import Codec, BitWriter, BitReader, resize, fit_size
from . import splat


class HybridDctSplat(Codec):
    name = "dct_ac+gauss"
    description = "Hybrid: arithmetic-coded DCT base + k residual Gaussian splats (k chosen per image)"
    decoder_cost = "DCT decoder + k exps per pixel"
    js_practical = "decoder yes; ENCODER needs iterative optimisation"

    def __init__(self, base):
        self.base = base

    def encode(self, image, budget_bytes):
        H, W = image.shape[:2]
        small = resize(image, *fit_size(H, W, 96))
        h, w = small.shape[:2]
        best = None
        for k in sorted({0, max(1, budget_bytes // 16), budget_bytes // 8}):
            bb = budget_bytes - 1 - 4 * k
            if bb < 6:
                continue
            bp = self.base.encode(image, bb)
            rec = self.base.decode(bp, w, h)
            if k:
                codes, _, _, _ = splat.fit("resid", small, k, base=rec)
                bw = BitWriter(); splat.pack_prims(bw, "resid", codes)
                sp = bw.tobytes()
                rec = rec + splat.render("resid", splat.unpack_prims(BitReader(sp), "resid", k), None, w, h)
            else:
                sp = b""
            err = np.mean((np.clip(rec, 0, 1) - small) ** 2)
            if best is None or err < best[0]:
                best = (err, bytes([k]) + sp + bp)
        assert len(best[1]) <= budget_bytes
        return best[1]

    def decode(self, payload, w, h):
        k = payload[0]
        rec = self.base.decode(payload[1 + 4 * k:], w, h)
        if k:
            U = splat.unpack_prims(BitReader(payload[1:1 + 4 * k]), "resid", k)
            rec = rec + splat.render("resid", U, None, w, h)
        return np.clip(rec, 0, 1)


class BestFit(Codec):
    description = "Per-image best method (1-byte tag inside the budget)"
    decoder_cost = "union of the candidate decoders"

    def __init__(self, codecs, name="bestfit", inner_cache=None):
        self.codecs, self.name = list(codecs), name
        self.inner_cache = inner_cache          # optional callable(codec, image, budget) -> payload
        self.js_practical = "as the worst candidate"

    def encode(self, image, budget_bytes):
        H, W = image.shape[:2]
        small = resize(image, *fit_size(H, W, 128))
        h, w = small.shape[:2]
        best = None
        for tag, c in enumerate(self.codecs):
            try:
                p = self.inner_cache(c, image, budget_bytes - 1) if self.inner_cache else c.encode(image, budget_bytes - 1)
            except AssertionError:
                continue
            err = np.mean((c.decode(p, w, h) - small) ** 2)
            if best is None or err < best[0]:
                best = (err, bytes([tag]) + p)
        assert len(best[1]) <= budget_bytes
        return best[1]

    def decode(self, payload, w, h):
        return self.codecs[payload[0]].decode(payload[1:], w, h)

    def chosen(self, payload):
        return self.codecs[payload[0]].name
