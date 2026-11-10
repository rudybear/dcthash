"""Turn results/results.csv into tables (results/tables.md), rate-distortion plots and visual grids.

    python scripts/make_report.py
"""
import csv, collections, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from hash2d import registry
from scripts.run_eval import load_test_images

RES = os.path.join(ROOT, "results")
MAIN_BUDGETS = [16, 24, 32, 48, 64]
METRICS = [("psnr", "PSNR dB", 2), ("ssim", "SSIM", 3), ("ms_ssim", "MS-SSIM", 3), ("lpips", "LPIPS (lower=better)", 3),
           ("psnr_blur", "PSNR vs blurred ref", 2), ("ssim_blur", "SSIM vs blurred ref", 3), ("lpips_blur", "LPIPS vs blurred ref", 3)]
# categorical palette (fixed order, <= 8 series per chart), light surface
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a63", "#e6e4dc", "#ffffff"


def load():
    rows = list(csv.DictReader(open(os.path.join(RES, "results.csv"))))
    small = os.path.join(RES, "results_small.csv")      # optional extra run at very small budgets (fast codecs only)
    if os.path.exists(small):
        rows += list(csv.DictReader(open(small)))
    for r in rows:
        for k, v in r.items():
            if k not in ("image", "method", "chosen"):
                r[k] = float(v) if v != "" else float("nan")
        r["budget"] = int(r["budget"])
    return rows


def tables(rows):
    by = collections.defaultdict(dict)
    for r in rows:
        by[(r["method"], r["budget"])][r["image"]] = r
    methods = list(dict.fromkeys(r["method"] for r in rows))
    out = []
    summary = {}
    for B in MAIN_BUDGETS:
        ref = by[("blurhash", B)]
        out.append(f"\n### Budget {B} bytes  (= {B * 8 / np.log2(83):.1f} base83 chars)\n")
        out.append("| method | bytes used | PSNR | dPSNR vs BlurHash (± s.e.m.) | wins vs BlurHash | SSIM | MS-SSIM | LPIPS↓ | PSNR-blur | SSIM-blur | LPIPS-blur↓ | encode | decode 384x256 | decode 32px |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        recs = []
        for m in methods:
            d = by.get((m, B))
            if not d:
                continue
            imgs = sorted(d)
            mean = lambda k: float(np.mean([d[i][k] for i in imgs]))
            dp = np.array([d[i]["psnr"] - ref[i]["psnr"] for i in imgs])
            recs.append((mean("psnr"), m, d, imgs, dp))
            summary[(m, B)] = {k: mean(k) for k, _, _ in METRICS}
            summary[(m, B)].update(dpsnr=float(dp.mean()), enc_s=float(np.median([d[i]["enc_s"] for i in imgs])),
                                   dec_s=float(np.median([d[i]["dec_s"] for i in imgs])))
        for p, m, d, imgs, dp in sorted(recs, key=lambda t: -t[0]):
            mean = lambda k: float(np.mean([d[i][k] for i in imgs]))
            med = lambda k: float(np.median([d[i][k] for i in imgs]))
            enc = med("enc_s")
            out.append(f"| {m} | {mean('bytes'):.1f} | {p:.2f} | {dp.mean():+.2f} ± {dp.std(ddof=1) / np.sqrt(len(dp)):.2f} | "
                       f"{int((dp > 0).sum())}/{len(dp)} | {mean('ssim'):.3f} | {mean('ms_ssim'):.3f} | {mean('lpips'):.3f} | "
                       f"{mean('psnr_blur'):.2f} | {mean('ssim_blur'):.3f} | {mean('lpips_blur'):.3f} | "
                       f"{(f'{enc * 1000:.0f} ms' if enc < 1 else f'{enc:.1f} s')} | {med('dec_s') * 1000:.1f} ms | {med('dec32_s') * 1000:.2f} ms |")
    # compact cross-budget tables
    for key, label, nd in [("psnr", "Mean PSNR (dB)", 2), ("dpsnr", "PSNR gain over BlurHash (dB)", 2), ("ssim", "Mean SSIM", 3),
                           ("ms_ssim", "Mean MS-SSIM", 3), ("psnr_blur", "Mean PSNR vs blurred reference (dB)", 2), ("ssim_blur", "Mean SSIM vs blurred reference", 3), ("lpips", "Mean LPIPS vs original (lower is better)", 3), ("lpips_blur", "Mean LPIPS vs blurred reference (lower is better)", 3)]:
        out.append(f"\n### {label} by budget\n")
        out.append("| method | " + " | ".join(f"{B} B" for B in MAIN_BUDGETS) + " |")
        out.append("|---|" + "---|" * len(MAIN_BUDGETS))
        ms = [m for m in methods if (m, 32) in summary]
        for m in sorted(ms, key=lambda m: -summary[(m, 32)]["psnr"]):
            out.append(f"| {m} | " + " | ".join(f"{summary[(m, B)][key]:{'+' if key == 'dpsnr' else ''}.{nd}f}" if (m, B) in summary else "-" for B in MAIN_BUDGETS) + " |")
    # best-fit choices
    out.append("\n### Which method does best-fit pick? (count over images, per budget)\n")
    for m in ("bestfit_fast", "bestfit_all"):
        for B in MAIN_BUDGETS:
            c = collections.Counter(r["chosen"] for r in by.get((m, B), {}).values())
            out.append(f"- `{m}` @ {B} B: " + ", ".join(f"{k} x{v}" for k, v in c.most_common()))
    # equal-quality byte savings: bytes a method needs to match BlurHash@B PSNR (linear interp on the mean RD curve)
    out.append("\n### Bytes needed to match BlurHash's mean PSNR (interpolated on mean RD curves)\n")
    allb = sorted({r["budget"] for r in rows})
    out.append("| method | " + " | ".join(f"BlurHash@{B}" for B in MAIN_BUDGETS) + " |")
    out.append("|---|" + "---|" * len(MAIN_BUDGETS))
    for m in ("dct_fixed", "dct_ac", "pca_ac", "bspline_grid", "gauss_norm"):
        xs = [b for b in allb if (m, b) in by and len(by[(m, b)]) == len(by[("blurhash", 16)])]
        ys = [np.mean([v["psnr"] for v in by[(m, b)].values()]) for b in xs]
        cells = []
        for B in MAIN_BUDGETS:
            t = summary[("blurhash", B)]["psnr"]
            cells.append(f"< {xs[0]}" if t <= ys[0] else (f"> {xs[-1]}" if t > ys[-1] else f"{np.interp(t, ys, xs):.0f}"))
        out.append(f"| {m} | " + " | ".join(cells) + " |")
    open(os.path.join(RES, "tables.md"), "w").write("\n".join(out) + "\n")
    return by, summary, methods


def rd_plot(by, series, fname, title):
    panels = [("psnr", "PSNR vs original (dB)"), ("ssim", "SSIM vs original"), ("psnr_blur", "PSNR vs blurred reference (dB)"), ("lpips_blur", "LPIPS vs blurred reference (lower is better)")]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9), facecolor=SURFACE)
    budgets = sorted({b for (_, b) in by})
    for ax, (key, lab) in zip(axes.ravel(), panels):
        ends = []
        for (m, label), col in zip(series, PAL):
            xs = [b for b in budgets if b >= 15 and (m, b) in by and len(by[(m, b)]) >= len(by[("blurhash", 16)])]
            ys = [np.mean([v[key] for v in by[(m, b)].values()]) for b in xs]
            ax.plot(xs, ys, color=col, lw=2, marker="o", ms=4, label=label, solid_capstyle="round")
            ends.append((ys[-1], label, col))
        ax.set_title(lab, loc="left", fontsize=11, color=INK)
        ax.set_xlabel("payload budget (bytes)", color=MUTED, fontsize=9)
        ax.set_xticks(MAIN_BUDGETS)
        ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.set_facecolor(SURFACE)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=min(4, len(l)), frameon=False, fontsize=10, labelcolor=INK)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    fig.savefig(os.path.join(RES, fname), dpi=110, facecolor=SURFACE)
    plt.close(fig)


def gain_plot(summary, series, fname):
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=SURFACE)
    for (m, label), col in zip(series, PAL):
        ys = [summary[(m, B)]["dpsnr"] for B in MAIN_BUDGETS]
        ax.plot(MAIN_BUDGETS, ys, color=col, lw=2, marker="o", ms=4, label=label)
        ax.annotate(f"{ys[-1]:+.2f}", (MAIN_BUDGETS[-1], ys[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=MUTED)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_title("PSNR gain over BlurHash at equal payload size (mean over 24 Kodak images)", loc="left", fontsize=11, color=INK)
    ax.set_xlabel("payload budget (bytes)", color=MUTED, fontsize=9); ax.set_ylabel("dB vs BlurHash", color=MUTED, fontsize=9)
    ax.set_xticks(MAIN_BUDGETS); ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.legend(frameon=False, fontsize=9, ncol=2, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, fname), dpi=110, facecolor=SURFACE)
    plt.close(fig)


def grids(by, images, methods, budget, names, fname):
    codecs = {c.name: c for c in registry.build()}
    imgs = dict(images)
    n = len(methods) + 1
    fig, axes = plt.subplots(len(names), n, figsize=(2.3 * n, 1.75 * len(names) * (1.0 if imgs[names[0]].shape[0] < imgs[names[0]].shape[1] else 2.2)),
                             facecolor=SURFACE, squeeze=False)
    for r, name in enumerate(names):
        img = imgs[name]
        H, W = img.shape[:2]
        cache = json.load(open(os.path.join(RES, "payloads", name + ".json")))
        axes[r, 0].imshow(img); axes[r, 0].set_title("original" if r == 0 else "", fontsize=9, color=INK)
        for c, m in enumerate(methods):
            rec = codecs[m].decode(bytes.fromhex(cache[f"{m}|{budget}"]["hex"]), W, H)
            # independent re-decode in this fresh process must reproduce the PSNR recorded by run_eval
            chk = -10 * np.log10(np.mean((np.round(np.clip(rec, 0, 1) * 255) / 255 - img) ** 2))
            assert abs(chk - by[(m, budget)][name]["psnr"]) < 1e-3, (m, name, chk, by[(m, budget)][name]["psnr"])
            ax = axes[r, c + 1]
            ax.imshow(np.clip(rec, 0, 1))
            if r == 0: ax.set_title(m, fontsize=9, color=INK)
            ax.set_xlabel(f"{by[(m, budget)][name]['psnr']:.2f} dB", fontsize=8, color=MUTED, labelpad=2)
        for ax in axes[r]:
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_visible(False)
    fig.suptitle(f"All methods at the same payload budget: {budget} bytes (~{budget * 8 / np.log2(83):.0f} base83 chars).  PSNR vs original under each.",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(RES, fname), dpi=100, facecolor=SURFACE)
    plt.close(fig)


def main():
    rows = load()
    by, summary, methods = tables(rows)
    rd_plot(by, [("blurhash", "BlurHash (best grid)"), ("dct_fixed", "DCT, fixed-length"), ("dct_ac", "DCT, arithmetic-coded"),
                 ("bspline_grid", "B-spline thumbnail"), ("gauss_norm", "Gaussian splats"), ("soft_voronoi", "soft Voronoi"),
                 ("siren", "SIREN MLP"), ("bestfit_all", "best-fit (all)")], "rd_families.png",
            "Rate-distortion by representation family (24 Kodak images, 384x256, every point is a real byte payload)")
    rd_plot(by, [("blurhash", "BlurHash (best grid)"), ("blurhash_gamma", "BlurHash, gamma-space"), ("dct_fixed", "DCT fixed-length"),
                 ("dct_ac", "DCT arithmetic-coded"), ("pca_ac", "PCA/KLT arithmetic-coded"), ("legendre_ac", "Legendre arithmetic-coded"),
                 ("haar_ac", "Haar arithmetic-coded"), ("dct_ac+gauss", "DCT + residual Gaussians")], "rd_transforms.png",
            "Rate-distortion within the linear-transform family (same quantiser/coder, basis varies)")
    gain_plot(summary, [("dct_ac", "DCT arithmetic-coded"), ("pca_ac", "PCA arithmetic-coded"), ("dct_fixed", "DCT fixed-length"),
                        ("bestfit_all", "best-fit (all)"), ("bspline_grid", "B-spline thumbnail"), ("gauss_norm", "Gaussian splats"),
                        ("siren", "SIREN MLP"), ("blurhash_gamma", "BlurHash gamma-space")], "gain_over_blurhash.png")
    images = load_test_images()
    show = ["blurhash", "dct_fixed", "dct_ac", "pca_ac", "haar_ac", "bspline_grid", "gauss_norm", "soft_voronoi", "siren", "dct_ac+gauss"]
    land = [n for n, im in images if im.shape[1] > im.shape[0]]
    port = [n for n, im in images if im.shape[1] < im.shape[0]]
    pick = [n for n in ("kodim03", "kodim05", "kodim07", "kodim15", "kodim20", "kodim21", "kodim23", "kodim24") if n in land] or land[:6]
    for B in (16, 24, 32, 64):
        grids(by, images, show, B, pick, f"grid_{B}B.png")
    if port:
        grids(by, images, show, 32, port[:3], "grid_32B_portrait.png")
    print(open(os.path.join(RES, "tables.md")).read())


if __name__ == "__main__":
    main()
