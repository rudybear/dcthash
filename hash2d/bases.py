"""Analytic orthonormal bases on the canonical G x G grid (G = 32).  Each returns (K, G*G)."""
import functools
import numpy as np

G = 32


def _orthonormal_rows(M):
    q, _ = np.linalg.qr(M.T)
    return q.T


@functools.lru_cache(maxsize=None)
def dct_basis(n=16):
    x = (np.arange(G) + 0.5) / G
    c = np.stack([np.cos(np.pi * k * x) for k in range(n)])
    c /= np.linalg.norm(c, axis=1, keepdims=True)
    return np.einsum("jy,ix->jiyx", c, c).reshape(n * n, G * G)


@functools.lru_cache(maxsize=None)
def legendre_basis(n=12):
    x = (np.arange(G) + 0.5) / G * 2 - 1
    p = _orthonormal_rows(np.polynomial.legendre.legvander(x, n - 1).T)  # discrete-orthonormal polynomials
    return np.einsum("jy,ix->jiyx", p, p).reshape(n * n, G * G)


@functools.lru_cache(maxsize=None)
def haar_basis(n=16):
    """Separable-per-level (non-standard) 2D Haar wavelets down to (G/n)-pixel support."""
    def haar1d(m):  # m x G rows: scaling function + wavelets, coarse to fine
        rows = [np.ones(G)]
        size = G
        while size > G // m:
            for s in range(0, G, size):
                r = np.zeros(G); r[s:s + size // 2] = 1; r[s + size // 2:s + size] = -1
                rows.append(r)
            size //= 2
        return rows
    # non-standard 2D construction: at each scale, LH / HL / HH of equal support
    funcs = [np.ones((G, G))]
    size = G
    while size > G // n:
        h = size // 2
        for sy in range(0, G, size):
            for sx in range(0, G, size):
                for kind in range(3):
                    f = np.zeros((G, G))
                    blk = np.ones((size, size))
                    if kind in (0, 2): blk[:, h:] *= -1
                    if kind in (1, 2): blk[h:, :] *= -1
                    f[sy:sy + size, sx:sx + size] = blk
                    funcs.append(f)
        size = h
    F = np.stack(funcs).reshape(len(funcs), -1)
    return F / np.linalg.norm(F, axis=1, keepdims=True)


ANALYTIC = {"dct": dct_basis, "legendre": legendre_basis, "haar": haar_basis}
