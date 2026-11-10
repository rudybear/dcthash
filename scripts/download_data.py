"""Download the test set (Kodak, 24 photos) and the *disjoint* training set
(Lorem Picsum / Unsplash photos, ~1000 images) used only to train decoder-side
tables (PCA bases, coefficient statistics).

    python scripts/download_data.py
"""
import os, sys, urllib.request, concurrent.futures as cf

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
KODAK = [(f"https://r0k.us/graphics/kodak/kodak/kodim{i:02d}.png", f"kodak/kodim{i:02d}.png") for i in range(1, 25)]
PICSUM = [(f"https://picsum.photos/id/{i}/480/480.jpg", f"picsum/{i:04d}.jpg") for i in range(0, 1085)]


def fetch(job):
    url, rel = job
    dst = os.path.join(ROOT, rel)
    if os.path.exists(dst) and os.path.getsize(dst) > 1000:
        return True
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    for _ in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 hash2d-experiment"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read()
            if len(data) > 1000:
                with open(dst + ".tmp", "wb") as f:
                    f.write(data)
                os.replace(dst + ".tmp", dst)
                return True
        except Exception as e:  # some picsum ids do not exist (404) - that is fine
            err = e
    return False


if __name__ == "__main__":
    for name, jobs in (("kodak", KODAK), ("picsum", PICSUM)):
        with cf.ThreadPoolExecutor(16) as ex:
            ok = sum(ex.map(fetch, jobs))
        print(f"{name}: {ok}/{len(jobs)} files present")
