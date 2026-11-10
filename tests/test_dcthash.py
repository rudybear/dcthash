"""Python reference vs the committed vectors, plus malformed-input behaviour.  Run: python -m tests.test_dcthash"""
import base64, json, os
import numpy as np
from hash2d import dcthash as dh

V = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "format", "vectors.json")))
n = 0
for e in V["encode"]:
    img = np.frombuffer(base64.b64decode(e["rgba"]), dtype=np.uint8).reshape(e["height"], e["width"], 4)
    for L, s in e["hashes"].items():
        assert dh.encode(img, int(L)) == s, (e["name"], L)
        n += 1
for d in V["decode"]:
    want = np.frombuffer(base64.b64decode(d["rgba"]), dtype=np.uint8).reshape(d["height"], d["width"], 4)
    assert np.array_equal(dh.decode(d["hash"], d["width"], d["height"]), want)
for h in V["helpers"]:
    assert list(dh.average_color(h["hash"])) == h["average"] and dh.aspect_ratio(h["hash"]) == h["aspect"]
for s in V["invalid"] + [None, 5, b"AAAAAAAA"]:
    assert not dh.is_valid(s)
    for fn in (lambda: dh.decode(s, 8, 8), lambda: dh.average_color(s), lambda: dh.aspect_ratio(s)):
        try:
            fn(); raise AssertionError(f"accepted {s!r}")
        except dh.DCTHashError:
            pass
good = V["encode"][0]["hashes"]["28"]
for w, h in [(0, 8), (8, 0), (-1, 8), (1.5, 8), (4097, 1), (4096, 4096), ("8", 8), (True, 8)]:
    try:
        dh.decode(good, w, h); raise AssertionError((w, h))
    except dh.DCTHashError:
        pass
for bad in (5, 97, 28.0, "28", None):
    try:
        dh.encode(np.zeros((4, 4, 3), np.uint8), bad); raise AssertionError(bad)
    except dh.DCTHashError:
        pass
rng = np.random.default_rng(0)
for _ in range(300):        # every well-formed string decodes
    L = int(rng.integers(dh.MIN_LENGTH, dh.MAX_LENGTH + 1))
    s = dh.ALPHABET[int(rng.integers(0, 32))] + "".join(dh.ALPHABET[int(i)] for i in rng.integers(0, 64, L - 1))
    assert dh.decode(s, 5, 4).shape == (4, 5, 4)
t = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "format", "tables.json")))
assert t["cos"] == dh.cos_table() and t["aspects"] == [list(p) for p in dh.ASPECTS] and t["scales"] == list(dh.SCALES)
assert t["alphabet"] == dh.ALPHABET
for p in ("photo", "smooth"):
    assert len(t[p]["seq"]) >= 6 * dh.MAX_LENGTH - 7 and max(np.bincount(t[p]["seq"])) <= dh.MAX_BITS
print(f"dcthash reference: {n} encode vectors, {len(V['decode'])} decode vectors, {len(V['helpers'])} helper vectors, "
      f"{len(V['invalid'])} invalid strings - all OK")
