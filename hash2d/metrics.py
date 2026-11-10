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
