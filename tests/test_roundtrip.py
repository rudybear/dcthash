"""Every closed-form codec must (a) respect the byte budget, (b) decode from bytes alone, deterministically,
for landscape / portrait / square / odd sizes.  Run: python -m tests.test_roundtrip"""
import numpy as np
from hash2d import registry

rng = np.random.default_rng(1)
codecs = [c for c in registry.build() if c.name not in registry.NEEDS_FIT]
for (h, w) in [(64, 96), (96, 64), (80, 80), (37, 111)]:
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.stack([0.5 + 0.4 * np.sin(xx / 9.0), 0.5 + 0.4 * np.cos(yy / 7.0), (xx + yy) / (h + w)], -1)
    img = np.clip(img + 0.05 * rng.standard_normal(img.shape), 0, 1).astype(np.float32)
    for c in codecs:
        for B in (16, 23, 32, 64):
            p = c.encode(img, B)
            assert isinstance(p, bytes) and len(p) <= B, (c.name, B, len(p))
            a = registry.build()[[k.name for k in registry.build()].index(c.name)].decode(p, w, h)   # fresh codec object
            assert a.shape == (h, w, 3) and np.isfinite(a).all()
            assert np.array_equal(a, c.decode(p, w, h))
            assert np.mean((a - img) ** 2) < np.var(img) * 1.05, (c.name, B, (h, w))
print("round-trip OK for", [c.name for c in codecs])
