"""Quality metrics.  All take float images (H, W, 3) in [0, 1]; the decoded image is rounded to
8 bits first, as a real client would display it."""
import functools
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity

BLUR_SIGMA = 4.0   # px at the 384x256 evaluation size (~1% of the long side)


def to8(img):
    return np.round(np.clip(img, 0, 1) * 255) / 255


def psnr(a, b):
    return float(-10 * np.log10(max(np.mean((a - b) ** 2), 1e-12)))


def blur(img, sigma=BLUR_SIGMA):
    return gaussian_filter(img, sigma=(sigma, sigma, 0), mode="nearest")


@functools.lru_cache(maxsize=1)
def _torch_metrics():
    import torch, warnings
    out = {"torch": torch, "dev": "cuda" if torch.cuda.is_available() else "cpu"}
    try:
        from pytorch_msssim import ms_ssim
        out["ms_ssim"] = ms_ssim
    except Exception as e:      # optional
        print("MS-SSIM unavailable:", e)
    try:
        import lpips
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out["lpips"] = lpips.LPIPS(net="alex", verbose=False).to(out["dev"]).eval()
    except Exception as e:      # optional (needs a weight download)
        print("LPIPS unavailable:", e)
    return out


def all_metrics(rec, ref, ref_blur=None, deep=True):
    rec = to8(rec)
    ref_blur = blur(ref) if ref_blur is None else ref_blur
    m = {
        "psnr": psnr(rec, ref),
        "ssim": float(structural_similarity(rec, ref, channel_axis=2, data_range=1.0, gaussian_weights=True, sigma=1.5,
                                            use_sample_covariance=False)),
        "psnr_blur": psnr(rec, ref_blur),
        "ssim_blur": float(structural_similarity(rec, ref_blur, channel_axis=2, data_range=1.0, gaussian_weights=True,
                                                 sigma=1.5, use_sample_covariance=False)),
    }
    if deep:
        tm = _torch_metrics()
        torch = tm["torch"]
        with torch.no_grad():
            a = torch.tensor(rec, dtype=torch.float32, device=tm["dev"]).permute(2, 0, 1)[None]
            b = torch.tensor(ref, dtype=torch.float32, device=tm["dev"]).permute(2, 0, 1)[None]
            if "ms_ssim" in tm:
                m["ms_ssim"] = float(tm["ms_ssim"](a, b, data_range=1.0))
            if "lpips" in tm:
                m["lpips"] = float(tm["lpips"](a * 2 - 1, b * 2 - 1))
                bb = torch.tensor(ref_blur, dtype=torch.float32, device=tm["dev"]).permute(2, 0, 1)[None]
                m["lpips_blur"] = float(tm["lpips"](a * 2 - 1, bb * 2 - 1))
    return m


# ---------------------------------------------------------------- colour-aware metrics
COLOUR_BLUR_SIGMA = 8.0   # px at 384x256 (~2% of the long side): the scale a placeholder can be expected to hold


def srgb_to_oklab(img):
    """float sRGB in [0, 1], (..., 3) -> OKLab (L in 0..1, a/b roughly -0.4..0.4)."""
    v = np.clip(np.asarray(img, dtype=np.float64), 0, 1)
    lin = np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)
    lms = lin @ np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                          [0.2119034982, 0.6806995451, 0.1073969566],
                          [0.0883024619, 0.2817188376, 0.6299787005]]).T
    return np.cbrt(lms) @ np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                                    [1.9779984951, -2.4285922050, 0.4505937099],
                                    [0.0259040371, 0.7827717662, -0.8086757660]]).T


def colourfulness(img):
    """Hasler-Suesstrunk colourfulness on 0..255 sRGB values."""
    r, g, b = [img[..., i].astype(np.float64) * 255 for i in range(3)]
    rg, yb = r - g, 0.5 * (r + g) - b
    return float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))


def colour_metrics(rec, ref):
    """Colour fidelity of a placeholder `rec` against the original `ref`, judged at placeholder scale:
    the reference is the original blurred with sigma = COLOUR_BLUR_SIGMA.

      de_ok       mean OKLab distance x100 (lightness + chroma)                      lower is better
      dc_ok       mean OKLab chroma-plane distance x100 (hue/saturation error only)  lower is better
      cf_ratio    Hasler-Suesstrunk colourfulness, decoded / reference               1 = as colourful
      chroma_ratio  mean OKLab chroma, decoded / reference                           1 = as saturated
      sat_top_ratio mean OKLab chroma of the decoded image over the reference's 10% most saturated pixels,
                    divided by the reference's own value there                       1 = saturated objects kept
    """
    rec = to8(rec)
    refb = blur(ref, COLOUR_BLUR_SIGMA)
    a, b = srgb_to_oklab(rec), srgb_to_oklab(refb)
    ca, cb = np.hypot(a[..., 1], a[..., 2]), np.hypot(b[..., 1], b[..., 2])
    top = cb >= np.quantile(cb, 0.9)
    return {
        "de_ok": float(100 * np.sqrt(((a - b) ** 2).sum(-1)).mean()),
        "dc_ok": float(100 * np.hypot(a[..., 1] - b[..., 1], a[..., 2] - b[..., 2]).mean()),
        "cf_ratio": colourfulness(rec) / max(colourfulness(refb), 1e-6),
        "chroma_ratio": float(ca.mean() / max(cb.mean(), 1e-9)),
        "sat_top_ratio": float(ca[top].mean() / max(cb[top].mean(), 1e-9)),
    }
