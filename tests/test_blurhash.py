"""Our BlurHash must be string-identical to the `blurhash` pip package.  Run: python -m tests.test_blurhash"""
import numpy as np, blurhash
from hash2d.codecs.blurhash_ref import encode_str, decode_str, str_to_bytes, bytes_to_str

rng = np.random.default_rng(0)
base = rng.random((6, 8, 3))
img = (np.kron(base, np.ones((8, 8, 1))) * 255).astype(np.uint8)
for nx, ny in [(1, 1), (4, 3), (3, 4), (6, 4), (9, 9)]:
    ref = blurhash.encode(img, components_x=nx, components_y=ny)
    mine = encode_str(img / 255.0, nx, ny)
    assert ref == mine, (nx, ny, ref, mine)
    assert bytes_to_str(str_to_bytes(mine)) == mine
    d = np.clip(np.round(decode_str(mine, 40, 30) * 255), 0, 255)
    assert np.abs(d - np.array(blurhash.decode(ref, 40, 30))).max() <= 1
print("blurhash reimplementation matches pip package")
