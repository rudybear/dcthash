"""Generate cross-implementation test vectors from the Python reference -> format/vectors.json

    encode : image (RGBA bytes, base64) + length -> exact string
    decode : string + (w, h) -> RGBA bytes (implementations must match within 1 per channel)
    helpers: average colour, aspect ratio
    invalid: strings that every decoder must reject

    python scripts/make_vectors.py
"""
import base64, glob, json, os, sys
import numpy as np
from PIL import Image
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from hash2d import dcthash as dh

b64 = lambda a: base64.b64encode(np.ascontiguousarray(a, dtype=np.uint8).tobytes()).decode()
rng = np.random.default_rng(12345)
images = []


def add(name, arr):
    arr = np.asarray(arr, dtype=np.uint8)
    if arr.shape[2] == 3:
        arr = np.concatenate([arr, np.full(arr.shape[:2] + (1,), 255, np.uint8)], 2)
    images.append((name, arr))


kodak = sorted(glob.glob(os.path.join(ROOT, "data", "kodak", "*.png")))
for f in kodak[::3]:
    im = Image.open(f).convert("RGB")
    add(os.path.basename(f)[:-4] + "_thumb", np.asarray(im.resize((im.width // 8, im.height // 8), Image.LANCZOS)))
if kodak:
    add("kodim23_half", np.asarray(Image.open(kodak[22]).convert("RGB").resize((384, 256), Image.LANCZOS)))
    add("kodim19_portrait_third", np.asarray(Image.open(kodak[18]).convert("RGB").resize((171, 256), Image.LANCZOS)))
yy, xx = np.mgrid[0:129, 0:257]
add("gradient_257x129", np.stack([xx * 255 // 256, yy * 255 // 128, (xx + yy) % 256], -1))
duo = np.zeros((128, 192, 3)); duo[..., 0] = np.linspace(255, 20, 192)[None]; duo[..., 2] = np.linspace(40, 255, 128)[:, None]; duo[..., 1] = 60
add("duotone_192x128", duo)
add("solid_teal_96x64", np.full((64, 96, 3), (20, 150, 140)))
add("soft_grey_90x160", 128 + 20 * np.sin(np.mgrid[0:160, 0:90][0] / 40.0)[..., None] + np.zeros(3))
add("pixel_1x1", [[[200, 30, 90, 255]]])
add("tiny_2x3", rng.integers(0, 256, (3, 2, 3)))
add("noise_31x33", rng.integers(0, 256, (33, 31, 3)))
add("strip_300x7", rng.integers(0, 256, (7, 300, 3)))
add("strip_5x200", rng.integers(0, 256, (200, 5, 3)))
add("black_16x16", np.zeros((16, 16, 3)))
add("white_16x16", np.full((16, 16, 3), 255))
add("alpha_noise_64x48", rng.integers(0, 256, (48, 64, 4)))
al = np.zeros((40, 60, 4), np.uint8); al[..., 0] = 255; al[..., 3] = np.linspace(0, 255, 60).astype(np.uint8)[None, :]
add("alpha_ramp_60x40", al)
add("transparent_black_20x20", np.zeros((20, 20, 4)))

LENGTHS = [6, 7, 12, 20, 28, 29, 40, 64, 95, 96]
enc, dec, helpers = [], [], []
for name, arr in images:
    h, w = arr.shape[:2]
    strings = {str(L): dh.encode(arr, L) for L in LENGTHS}
    enc.append({"name": name, "width": w, "height": h, "rgba": b64(arr), "hashes": strings,
                "no_aspect_28": dh.encode(arr, 28, aspect=False)})
    for L in (6, 28, 96):
        s = strings[str(L)]
        for (dw, dh_) in ((32, 32), (48, 32), (7, 5), (1, 1), (100, 37)):
            if name.startswith("kodim0") or (dw, dh_) == (32, 32):
                dec.append({"hash": s, "width": dw, "height": dh_, "rgba": b64(dh.decode(s, dw, dh_))})
        helpers.append({"hash": s, "average": list(dh.average_color(s)), "aspect": dh.aspect_ratio(s)})
s0 = dh.encode(images[0][1], 28, aspect=False)
helpers.append({"hash": s0, "average": list(dh.average_color(s0)), "aspect": None})
for _ in range(40):                                  # arbitrary valid strings (every bit pattern must decode)
    L = int(rng.integers(dh.MIN_LENGTH, dh.MAX_LENGTH + 1))
    s = dh.ALPHABET[int(rng.integers(0, 32))] + "".join(dh.ALPHABET[int(i)] for i in rng.integers(0, 64, L - 1))
    dec.append({"hash": s, "width": 24, "height": 16, "rgba": b64(dh.decode(s, 24, 16))})
    helpers.append({"hash": s, "average": list(dh.average_color(s)), "aspect": dh.aspect_ratio(s)})
good = dh.encode(images[0][1], 28)
invalid = ["", "A", "AAAAA", "A" * 97, "A" * 1000, good[:-1] + "+", good[:-1] + "/", good[:-1] + "=", good[:-1] + " ",
           good[:-1] + "#", good[:-1] + "é", good[:-1] + "\U0001F600", " " + good[1:], "g" + good[1:], "_" + good[1:],
           good + "\n", "LEHV6nWB2yk8pyo0adR*.7kCMdnj"]            # the last one is a BlurHash
for s in invalid:
    assert not dh.is_valid(s), s
json.dump({"format": "dcthash", "version": 0, "decode_tolerance": 1, "encode": enc, "decode": dec, "helpers": helpers,
           "invalid": invalid}, open(os.path.join(ROOT, "format", "vectors.json"), "w"))
import collections
print("profiles chosen:", dict(collections.Counter(dh.profile(h) for e in enc for h in e["hashes"].values())))
print(len(enc), "images x", len(LENGTHS), "lengths;", len(dec), "decode vectors;", len(helpers), "helper vectors;", len(invalid),
      "invalid;", os.path.getsize(os.path.join(ROOT, "format", "vectors.json")) // 1024, "KiB")
