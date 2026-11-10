"""The shipped DCTHash format (hash2d/dcthash.py, frozen tables) wrapped in the experiment's
codec interface, so it appears in the phase-1 tables, plots and grids next to the candidates.

Byte accounting: a string of N base64url characters carries 6N bits. For a byte budget B the
string length is floor(8B/6) (so it never uses more bits than the budget), packed 6 bits per
character into ceil(6N/8) bytes; the length is recoverable from the byte count.
"""
import numpy as np
from ..common import Codec
from .. import dcthash as dh


class DCTHashCodec(Codec):
    name = "dcthash"
    description = "Shipped DCTHash format (colour-weighted tables, header included), as an experiment codec"
    decoder_tables_bytes = 3500
    decoder_cost = "fixed-length bit reader + sum of cosines; see ts/"
    js_practical = "yes (shipped)"

    def encode(self, image, budget_bytes):
        n = min(dh.MAX_LENGTH, budget_bytes * 8 // 6)
        s = dh.encode(np.clip(np.round(image * 255), 0, 255).astype(np.uint8), n)
        v = 0
        for ch in s:
            v = (v << 6) | dh.ALPHABET.index(ch)
        nbytes = (6 * n + 7) // 8
        return (v << (8 * nbytes - 6 * n)).to_bytes(nbytes, "big")

    def decode(self, payload, w, h):
        n = len(payload) * 8 // 6
        v = int.from_bytes(payload, "big") >> (8 * len(payload) - 6 * n)
        s = "".join(dh.ALPHABET[(v >> (6 * (n - 1 - k))) & 63] for k in range(n))
        return dh.decode(s, w, h)[..., :3].astype(np.float32) / 255.0
