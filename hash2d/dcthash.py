"""DCTHash v0 - Python reference implementation of the frozen format (see SPEC.md).

Standalone: needs only numpy and format/tables.json.  The ENCODER is integer-only, so any
conforming implementation produces the identical string.  The decoder evaluates cosines in
floating point; outputs agree across implementations to within 1/255 per channel.

    s   = encode(rgba_uint8, length=28)        # rgba: (h, w, 4) or (h, w, 3) uint8
    img = decode(s, 384, 256)                  # (h, w, 4) uint8
"""
import functools, json, math, os
import numpy as np

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
_INDEX = {c: i for i, c in enumerate(ALPHABET)}
GRID = 32            # analysis grid
NFREQ = 16           # frequencies 0..15 per axis
MIN_LENGTH, MAX_LENGTH, DEFAULT_LENGTH = 6, 96, 28
MAX_BITS = 8         # per coefficient
FULL = 32640         # 127.5 grey levels in T units: half the 8-bit range of a DC term
SCALES = (2, 3, 4, 6)  # photo profile: contrast scale code -> multiplier x4 of the non-DC quantiser ranges
CWEIGHT = (12, 13, 10)  # integer RGB-error weights of Y, Cb, Cr used by the encoder's profile decision
MAX_SIDE, MAX_AREA = 4096, 1 << 22          # decoder output limits
MAX_ENCODE_SIDE = 16384
ASPECTS = [(1, 5), (1, 4), (1, 3), (5, 12), (3, 7), (4, 9), (1, 2), (9, 16), (3, 5), (5, 8), (2, 3), (5, 7), (3, 4), (4, 5),
           (5, 6), (1, 1), (6, 5), (5, 4), (4, 3), (7, 5), (3, 2), (8, 5), (5, 3), (16, 9), (2, 1), (9, 4), (7, 3), (12, 5),
           (3, 1), (4, 1), (5, 1)]          # width:height for aspect codes 0..30; code 31 = unspecified
ASPECT_UNSPECIFIED = 31
_TABLES = os.environ.get("DCTHASH_TABLES") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "format", "tables.json")


class DCTHashError(ValueError):
    pass


def cos_table():
    """COS[u][x] = round(2^14 cos(pi u (2x+1) / 64)); frozen in tables.json."""
    return [[int(math.floor(16384 * math.cos(math.pi * u * (2 * x + 1) / 64) + 0.5)) for x in range(GRID)] for u in range(NFREQ)]


@functools.lru_cache(maxsize=4)
def load_tables(path=None):
    t = json.load(open(path or _TABLES))
    arr = lambda v: np.array(v, dtype=np.int64)
    g = t["g"] if isinstance(t["g"][0], list) else [t["g"], t["g"]]     # [luma, chroma]; one shared list in the first tables
    return dict(cos=arr(t["cos"]), g=[[0] + g[0], [0] + g[1]],
                photo=dict(coefs=arr(t["photo"]["coefs"]), M=arr(t["photo"]["M"]), S=arr(t["photo"]["S"]), seq=arr(t["photo"]["seq"])),
                smooth=dict(coefs=arr(t["smooth"]["coefs"]), seq=arr(t["smooth"]["seq"])))


# ------------------------------------------------------------------ pieces shared by encoder and decoder
def aspect_code(w, h):
    """Index of the ASPECTS entry nearest to w/h on a log scale, in exact integer arithmetic:
    the number of k with  w^2 q_k q_(k+1) >= h^2 p_k p_(k+1)  (the geometric-mean boundaries)."""
    return sum(1 for (p0, q0), (p1, q1) in zip(ASPECTS, ASPECTS[1:]) if w * w * q0 * q1 >= h * h * p0 * p1)


def payload_bits(length, smooth):
    """header: version(1) aspect(5) profile(1) [+ contrast scale(2) in the photo profile]"""
    return 6 * length - (7 if smooth else 9)


def allocation(profile, nbits):
    """bits per coefficient: histogram of the first `nbits` entries of the profile's bit sequence."""
    return np.bincount(profile["seq"][:nbits], minlength=len(profile["coefs"])).tolist()


def _kind(profile_is_smooth, ch, u, v):
    """quantiser of a coefficient: 'u' uniform or 'c' companded."""
    if u == 0 and v == 0:
        return "u" if ch == 0 else "c"
    return "c" if profile_is_smooth else "u"


def _mean(prof, smooth, i, ch, u, v):
    """centre M (T units) of a coefficient: table value in the photo profile; in the smooth profile FULL for
    luma DC (mid grey) and 0 for everything else."""
    if smooth:
        return FULL if (ch == 0 and u == 0 and v == 0) else 0
    return int(prof["M"][i])


def _range(t, i, b, scale_code):
    """photo profile: half-range L (T units) of the uniform quantiser of coefficient i at depth b."""
    p = t["photo"]
    ch, u, v = (int(x) for x in p["coefs"][i])
    if u == 0 and v == 0:
        return FULL
    L = max(1, (int(p["S"][i]) * t["g"][1 if ch else 0][b] + 128) >> 8)
    return max(1, (L * SCALES[scale_code]) >> 2)


def _uniform_q(X, M, L, b):
    return min(max((((X - 64 * M + 64 * L) << b) // (128 * L)), 0), (1 << b) - 1)


def _uniform_r(q, M, L, b):
    """reconstruction x 2^16 (T units)"""
    return ((M - L) << 16) + (2 * q + 1) * L * (1 << (16 - b))


def _compand_n(q, b):
    """companded quantiser: index q in 0..2^b-1 -> signed step n in -(2^(b-1) - 1) .. 2^(b-1); n = 0 is exact zero."""
    return q - ((1 << (b - 1)) - 1)


def _compand_q(X, b):
    """nearest level FULL * n|n| / 4^(b-1): count the midpoints between consecutive levels that are <= X / 64."""
    m = [_compand_n(k, b) * abs(_compand_n(k, b)) for k in range(1 << b)]
    lhs = 2 * X * (1 << (2 * (b - 1)))
    return sum(1 for k in range(1, 1 << b) if lhs >= 64 * FULL * (m[k - 1] + m[k]))


def _compand_r(q, b):
    """reconstruction x 2^16 (T units)"""
    n = _compand_n(q, b)
    return FULL * n * abs(n) * (1 << (18 - 2 * b))


# ------------------------------------------------------------------ encoder (integer only)
def _overlap(n):
    """(32, n) integer overlaps of source pixel i = [32 i, 32 i + 32) with cell j = [n j, n j + n)."""
    i = np.arange(n, dtype=np.int64)[None, :] * GRID
    j = np.arange(GRID, dtype=np.int64)[:, None] * n
    return np.clip(np.minimum(i + GRID, j + n) - np.maximum(i, j), 0, None)


def analyse(rgba, cos=None):
    """uint8 (h, w, 3|4) -> int64 X[channel, v, u] = 16 * (unnormalised DCT-II coefficient of the 32x32 YCbCr grid)."""
    rgba = np.asarray(rgba)
    if rgba.dtype != np.uint8 or rgba.ndim != 3 or rgba.shape[2] not in (3, 4):
        raise DCTHashError("expected uint8 array of shape (h, w, 3) or (h, w, 4)")
    h, w = rgba.shape[:2]
    if not (1 <= w <= MAX_ENCODE_SIDE and 1 <= h <= MAX_ENCODE_SIDE):
        raise DCTHashError(f"width and height must be in 1..{MAX_ENCODE_SIDE}")
    px = rgba[..., :3].astype(np.int64)
    if rgba.shape[2] == 4:                                          # composite over white
        a = rgba[..., 3:4].astype(np.int64)
        px = (px * a + 255 * (255 - a) + 127) // 255
    S = np.tensordot(_overlap(h), np.tensordot(px, _overlap(w), axes=([1], [1])), axes=([1], [0]))   # (32, 3, 32)
    S = S.transpose(0, 2, 1)                                         # (32, 32, 3), each cell has area w*h
    V = (S * 16 + (w * h) // 2) // (w * h)                           # 1/16 of an 8-bit level, 0..4080
    R, Gc, B = V[..., 0], V[..., 1], V[..., 2]
    P = np.stack([(19595 * R + 38470 * Gc + 7471 * B + 32768) >> 16,
                  (-11059 * R - 21709 * Gc + 32768 * B + 32768) >> 16,
                  (32768 * R - 27439 * Gc - 5329 * B + 32768) >> 16])  # (3, 32, 32) [c][y][x]
    C = np.array(cos_table(), dtype=np.int64) if cos is None else cos
    c = np.matmul(np.matmul(C, P), C.T)                              # [c][v][u], |c| < 2^51
    return (c + (1 << 27)) >> 28


def _candidate(t, X, length, smooth, scale_code):
    """quantise under one candidate -> (indices q per coefficient, bits, weighted integer error E)."""
    prof = t["smooth" if smooth else "photo"]
    bits = allocation(prof, payload_bits(length, smooth))
    R = np.zeros((3, NFREQ, NFREQ), dtype=object)                   # reconstruction x 2^16 of every (c, v, u)
    qs = []
    for i, (ch, u, v) in enumerate(prof["coefs"].tolist()):
        b, x = bits[i], int(X[ch, v, u])
        M = _mean(prof, smooth, i, ch, u, v)
        if b == 0:
            q, r = 0, M << 16
        elif _kind(smooth, ch, u, v) == "c":
            q = _compand_q(x, b); r = _compand_r(q, b)
        else:
            L = FULL if (smooth or (u == 0 and v == 0)) else _range(t, i, b, scale_code)
            q = _uniform_q(x, M, L, b); r = _uniform_r(q, M, L, b)
        qs.append(q); R[ch, v, u] = r
    E = 0
    for ch in range(3):
        for v in range(NFREQ):
            for u in range(NFREQ):
                e = (int(X[ch, v, u]) * 1024 - int(R[ch, v, u])) >> 15
                E += CWEIGHT[ch] * (2 if u else 1) * (2 if v else 1) * e * e
    return qs, bits, E


def encode(rgba, length=DEFAULT_LENGTH, aspect=True, _tables=None, _photo_only=False):
    """`_tables` and `_photo_only` are experiment hooks (alternative table file; skip the smooth candidate)."""
    if not isinstance(length, int) or isinstance(length, bool) or not (MIN_LENGTH <= length <= MAX_LENGTH):
        raise DCTHashError(f"length must be an integer in {MIN_LENGTH}..{MAX_LENGTH}")
    t = load_tables(_tables)
    X = analyse(rgba, t["cos"])
    h, w = np.asarray(rgba).shape[:2]
    a = aspect_code(w, h) if aspect else ASPECT_UNSPECIFIED
    best = None                                                     # candidates in order: photo scale 0..3, then smooth
    for smooth, sc in [(False, 0), (False, 1), (False, 2), (False, 3), (True, 0)][:4 if _photo_only else 5]:
        qs, bits, E = _candidate(t, X, length, smooth, sc)
        if best is None or E < best[0]:
            best = (E, smooth, sc, qs, bits)
    _, smooth, sc, qs, bits = best
    v, n = (a << 1) | int(smooth), 7                                # version bit 0, aspect, profile
    if not smooth:
        v, n = (v << 2) | sc, 9
    for q, b in zip(qs, bits):
        if b:
            v = (v << b) | q
            n += b
    assert n == 6 * length
    return "".join(ALPHABET[(v >> (6 * (length - 1 - k))) & 63] for k in range(length))


# ------------------------------------------------------------------ decoder
def _parse(s):
    """-> (bit string as int, total bits, aspect code)"""
    if not isinstance(s, str):
        raise DCTHashError("hash must be a string")
    if not (MIN_LENGTH <= len(s) <= MAX_LENGTH):
        raise DCTHashError(f"hash length must be {MIN_LENGTH}..{MAX_LENGTH} characters")
    v = 0
    for ch in s:
        if ch not in _INDEX:
            raise DCTHashError(f"invalid character {ch!r}")
        v = (v << 6) | _INDEX[ch]
    n = 6 * len(s)
    head = v >> (n - 6)
    if head & 32:
        raise DCTHashError("unsupported version")
    return v, n, head & 31


def _coefficients(s, _tables=None):
    """-> K[channel][v][u] float64, scaled so that pixel = sum K cos cos (8-bit levels, YCbCr)."""
    v, n, _ = _parse(s)
    t = load_tables(_tables)
    pos = n - 7
    smooth = bool((v >> pos) & 1)
    sc = 0
    if not smooth:
        pos -= 2
        sc = (v >> pos) & 3
    prof = t["smooth" if smooth else "photo"]
    bits = allocation(prof, payload_bits(len(s), smooth))
    K = np.zeros((3, NFREQ, NFREQ))
    for i, (ch, u, vv) in enumerate(prof["coefs"].tolist()):
        b = bits[i]
        M = _mean(prof, smooth, i, ch, u, vv)
        if b == 0:
            r = M << 16
        else:
            pos -= b
            q = (v >> pos) & ((1 << b) - 1)
            if _kind(smooth, ch, u, vv) == "c":
                r = _compand_r(q, b)
            else:
                L = FULL if (smooth or (u == 0 and vv == 0)) else _range(t, i, b, sc)
                r = _uniform_r(q, M, L, b)
        K[ch, vv, u] = (r / 65536.0) * (2 if u else 1) * (2 if vv else 1) / 256.0
    assert pos == 0
    return K


def _to_rgb(ycc):
    y, cb, cr = ycc[..., 0], ycc[..., 1], ycc[..., 2]
    rgb = np.stack([y + 1.402 * cr, y - 0.344136 * cb - 0.714136 * cr, y + 1.772 * cb], -1)
    return np.clip(np.floor(rgb + 0.5), 0, 255).astype(np.uint8)


def decode(s, width, height, _tables=None):
    """-> uint8 (height, width, 4), alpha = 255."""
    for d in (width, height):
        if not isinstance(d, int) or isinstance(d, bool) or d < 1 or d > MAX_SIDE:
            raise DCTHashError(f"width/height must be integers in 1..{MAX_SIDE}")
    if width * height > MAX_AREA:
        raise DCTHashError("output too large")
    K = _coefficients(s, _tables)
    cx = np.cos(np.pi * np.arange(NFREQ)[:, None] * (np.arange(width)[None, :] + 0.5) / width)     # [u][x]
    cy = np.cos(np.pi * np.arange(NFREQ)[:, None] * (np.arange(height)[None, :] + 0.5) / height)   # [v][y]
    ycc = np.einsum("vy,cvu,ux->yxc", cy, K, cx)
    out = np.empty((height, width, 4), dtype=np.uint8)
    out[..., :3] = _to_rgb(ycc)
    out[..., 3] = 255
    return out


def average_color(s):
    """(r, g, b) mean colour of the placeholder, without decoding pixels."""
    K = _coefficients(s)
    return tuple(int(x) for x in _to_rgb(K[:, 0, 0][None, None, :])[0, 0])


def aspect_ratio(s):
    """approximate width / height stored in the hash (one of 31 common ratios), or None if unspecified."""
    a = _parse(s)[2]
    return None if a == ASPECT_UNSPECIFIED else ASPECTS[a][0] / ASPECTS[a][1]


def profile(s):
    """'photo' or 'smooth' - which table set the encoder chose (informational)."""
    v, n, _ = _parse(s)
    return "smooth" if (v >> (n - 7)) & 1 else "photo"


def is_valid(s):
    try:
        _parse(s)
        return True
    except DCTHashError:
        return False
