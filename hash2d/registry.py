"""All codecs under test, in report order."""
from .codecs.blurhash_ref import BlurHash
from .codecs.transform import TransformCodec
from .codecs.bspline import BSplineGrid
from .codecs.splat import SplatCodec
from .codecs.siren import Siren
from .codecs.hybrid import HybridDctSplat, BestFit


def build(inner_cache=None):
    dct_ac = TransformCodec("dct", "ac")
    base = [
        BlurHash(),
        BlurHash(search=False),
        BlurHash(linear=False),
        TransformCodec("dct", "fixed"), dct_ac,
        TransformCodec("pca", "fixed"), TransformCodec("pca", "ac"),
        TransformCodec("legendre", "ac"),
        TransformCodec("haar", "ac"),
        BSplineGrid(),
        SplatCodec("add"), SplatCodec("norm"), SplatCodec("voronoi"),
        Siren(),
        HybridDctSplat(dct_ac),
    ]
    by = {c.name: c for c in base}
    fast = [by[n] for n in ("dct_ac", "pca_ac", "haar_ac", "legendre_ac", "bspline_grid", "blurhash")]
    slow = [by[n] for n in ("gauss_add", "gauss_norm", "soft_voronoi", "siren", "dct_ac+gauss")]
    meta = [BestFit(fast, "bestfit_fast", inner_cache), BestFit(fast + slow, "bestfit_all", inner_cache)]
    return base + meta


NEEDS_FIT = {"gauss_add", "gauss_norm", "soft_voronoi", "siren", "dct_ac+gauss", "bestfit_all"}
