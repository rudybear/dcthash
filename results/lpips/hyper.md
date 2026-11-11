Selection set: 96 Picsum photos (384x256), 28 chars, float simulation of the format.

| training weights | psnr | psnr_blur | ms_ssim | dists | dists_blur | dc_ok | cf_ratio | sat_top_ratio | lpips_alex_sharp | lpips_vgg_sharp | lpips_alex_blur | lpips_vgg_blur | lpips_alex_small | lpips_vgg_small |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MSE (chroma x1, no prior) | 17.62 | 20.76 | 0.495 | 0.549 | 0.356 | 2.15 | 0.768 | 0.636 | 0.774 | 0.719 | 0.330 | 0.430 | 0.784 | 0.668 |
| x2 r3 g0.5 | 17.50 | 20.57 | 0.476 | 0.560 | 0.379 | 1.81 | 0.826 | 0.708 | 0.764 | 0.700 | 0.340 | 0.420 | 0.768 | 0.674 |
| x4 r3 g1 | 17.24 | 20.12 | 0.456 | 0.571 | 0.402 | 1.62 | 0.857 | 0.752 | 0.757 | 0.683 | 0.356 | 0.423 | 0.746 | 0.688 |
| x8 r3 g1.5 (shipped setting) | 16.96 | 19.64 | 0.437 | 0.584 | 0.424 | 1.51 | 0.880 | 0.779 | 0.754 | 0.666 | 0.372 | 0.425 | 0.730 | 0.696 |
| x8 r3 g2.5 | 16.84 | 19.44 | 0.431 | 0.588 | 0.431 | 1.53 | 0.869 | 0.775 | 0.755 | 0.665 | 0.380 | 0.430 | 0.729 | 0.700 |
| x16 r3 g1.5 | 16.79 | 19.36 | 0.429 | 0.589 | 0.433 | 1.45 | 0.891 | 0.795 | 0.753 | 0.660 | 0.378 | 0.426 | 0.722 | 0.698 |
| x8 r2 g2 | 16.73 | 19.27 | 0.424 | 0.593 | 0.438 | 1.52 | 0.862 | 0.769 | 0.755 | 0.660 | 0.389 | 0.433 | 0.725 | 0.702 |
| x8 r4 g1.5 | 17.04 | 19.78 | 0.444 | 0.579 | 0.415 | 1.51 | 0.882 | 0.781 | 0.754 | 0.672 | 0.364 | 0.422 | 0.732 | 0.691 |
| x1 r3 g1.5 | 17.36 | 20.32 | 0.463 | 0.569 | 0.398 | 1.78 | 0.824 | 0.710 | 0.761 | 0.689 | 0.357 | 0.426 | 0.759 | 0.687 |
| x8 no prior | 17.31 | 20.23 | 0.466 | 0.565 | 0.386 | 1.62 | 0.878 | 0.772 | 0.759 | 0.693 | 0.341 | 0.418 | 0.750 | 0.677 |
| x2 r3 g1.5 | 17.26 | 20.16 | 0.455 | 0.574 | 0.405 | 1.65 | 0.839 | 0.730 | 0.758 | 0.680 | 0.359 | 0.424 | 0.750 | 0.690 |
| x4 r3 g2.5 | 16.98 | 19.67 | 0.438 | 0.585 | 0.425 | 1.55 | 0.858 | 0.760 | 0.755 | 0.667 | 0.374 | 0.427 | 0.735 | 0.698 |

Setting each criterion would pick (lowest is best for LPIPS, DISTS, chroma error; highest for PSNR, MS-SSIM):

- psnr: MSE (chroma x1, no prior)
- psnr_blur: MSE (chroma x1, no prior)
- ms_ssim: MSE (chroma x1, no prior)
- dists: MSE (chroma x1, no prior)
- dists_blur: MSE (chroma x1, no prior)
- dc_ok: x16 r3 g1.5
- lpips_alex_sharp: x16 r3 g1.5
- lpips_vgg_sharp: x8 r2 g2
- lpips_alex_blur: MSE (chroma x1, no prior)
- lpips_vgg_blur: x8 no prior
- lpips_alex_small: x16 r3 g1.5
- lpips_vgg_small: MSE (chroma x1, no prior)
