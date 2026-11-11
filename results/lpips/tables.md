
### kodak, 28 chars: tables with M, S, g tuned by gradient descent on an LPIPS loss (real integer codec)

| training loss | psnr | psnr_blur | ms_ssim | dists | dists_blur | dc_ok | cf_ratio | sat_top_ratio | lpips_alex_sharp | lpips_vgg_sharp | lpips_alex_blur | lpips_vgg_blur | lpips_alex_small | lpips_vgg_small |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| shipped tables | 17.61 | 20.58 | 0.390 | 0.661 | 0.494 | 1.70 | 0.798 | 0.683 | 0.897 | 0.752 | 0.437 | 0.474 | 0.809 | 0.755 |
| alex_blur | 17.47 | 20.31 | 0.381 | 0.659 | 0.488 | 1.76 | 0.814 | 0.672 | 0.908 | 0.764 | 0.434 | 0.477 | 0.812 | 0.743 |
| vgg_blur | 17.46 | 20.32 | 0.380 | 0.676 | 0.518 | 1.70 | 0.759 | 0.656 | 0.896 | 0.738 | 0.458 | 0.481 | 0.813 | 0.762 |
| alex_small (photo profile forced; otherwise 24/24 images fall back to the smooth profile) | 11.92 | 12.64 | 0.205 | 0.599 | 0.419 | 3.39 | 1.200 | 0.501 | 0.898 | 0.840 | 0.574 | 0.649 | 0.625 | 0.769 |
| vgg_small (photo profile forced; otherwise 24/24 images fall back to the smooth profile) | 14.25 | 15.55 | 0.224 | 0.645 | 0.448 | 2.20 | 0.840 | 0.578 | 0.954 | 0.827 | 0.459 | 0.544 | 0.833 | 0.668 |
| alex_small+10mse (photo profile forced; otherwise 24/24 images fall back to the smooth profile) | 16.76 | 19.10 | 0.333 | 0.630 | 0.443 | 2.76 | 1.177 | 0.652 | 0.886 | 0.810 | 0.476 | 0.553 | 0.656 | 0.764 |

### kodak, 40 chars: tables with M, S, g tuned by gradient descent on an LPIPS loss (real integer codec)

| training loss | psnr | psnr_blur | ms_ssim | dists | dists_blur | dc_ok | cf_ratio | sat_top_ratio | lpips_alex_sharp | lpips_vgg_sharp | lpips_alex_blur | lpips_vgg_blur | lpips_alex_small | lpips_vgg_small |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| shipped tables | 17.91 | 21.14 | 0.410 | 0.646 | 0.466 | 1.54 | 0.845 | 0.731 | 0.896 | 0.762 | 0.403 | 0.456 | 0.811 | 0.740 |
| alex_blur | 17.75 | 20.83 | 0.402 | 0.650 | 0.471 | 1.58 | 0.843 | 0.723 | 0.904 | 0.765 | 0.406 | 0.460 | 0.808 | 0.728 |
| vgg_blur | 17.72 | 20.80 | 0.398 | 0.663 | 0.498 | 1.56 | 0.806 | 0.695 | 0.896 | 0.743 | 0.430 | 0.462 | 0.806 | 0.744 |
| alex_small (photo profile forced; otherwise 24/24 images fall back to the smooth profile) | 11.98 | 12.71 | 0.211 | 0.599 | 0.418 | 3.34 | 1.192 | 0.508 | 0.896 | 0.838 | 0.570 | 0.645 | 0.625 | 0.765 |
| vgg_small (photo profile forced; otherwise 24/24 images fall back to the smooth profile) | 14.39 | 15.74 | 0.239 | 0.643 | 0.445 | 2.14 | 0.834 | 0.590 | 0.952 | 0.825 | 0.454 | 0.539 | 0.832 | 0.666 |
| alex_small+10mse (photo profile forced; otherwise 24/24 images fall back to the smooth profile) | 16.99 | 19.46 | 0.356 | 0.627 | 0.437 | 2.69 | 1.159 | 0.655 | 0.885 | 0.807 | 0.465 | 0.543 | 0.661 | 0.757 |

### holdout, 28 chars: tables with M, S, g tuned by gradient descent on an LPIPS loss (real integer codec)

| training loss | psnr | psnr_blur | ms_ssim | dists | dists_blur | dc_ok | cf_ratio | sat_top_ratio | lpips_alex_sharp | lpips_vgg_sharp | lpips_alex_blur | lpips_vgg_blur | lpips_alex_small | lpips_vgg_small |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| shipped tables | 17.59 | 20.31 | 0.392 | 0.654 | 0.493 | 1.72 | 0.778 | 0.748 | 0.835 | 0.639 | 0.431 | 0.440 | 0.750 | 0.711 |
| alex_blur | 17.54 | 20.22 | 0.388 | 0.645 | 0.476 | 1.71 | 0.774 | 0.742 | 0.842 | 0.648 | 0.412 | 0.431 | 0.761 | 0.700 |
| vgg_blur | 17.45 | 20.11 | 0.392 | 0.667 | 0.510 | 1.76 | 0.751 | 0.720 | 0.836 | 0.632 | 0.439 | 0.440 | 0.746 | 0.706 |
| alex_small (photo profile forced; otherwise 8/8 images fall back to the smooth profile) | 11.05 | 11.57 | 0.172 | 0.600 | 0.451 | 3.53 | 1.695 | 0.780 | 0.848 | 0.744 | 0.546 | 0.584 | 0.635 | 0.736 |
| vgg_small (photo profile forced; otherwise 8/8 images fall back to the smooth profile) | 13.19 | 14.07 | 0.170 | 0.647 | 0.490 | 2.56 | 0.997 | 0.761 | 0.897 | 0.749 | 0.477 | 0.534 | 0.849 | 0.704 |
| alex_small+10mse (photo profile forced; otherwise 8/8 images fall back to the smooth profile) | 16.70 | 18.72 | 0.337 | 0.621 | 0.459 | 3.01 | 1.519 | 0.942 | 0.834 | 0.715 | 0.461 | 0.521 | 0.630 | 0.743 |

### holdout, 40 chars: tables with M, S, g tuned by gradient descent on an LPIPS loss (real integer codec)

| training loss | psnr | psnr_blur | ms_ssim | dists | dists_blur | dc_ok | cf_ratio | sat_top_ratio | lpips_alex_sharp | lpips_vgg_sharp | lpips_alex_blur | lpips_vgg_blur | lpips_alex_small | lpips_vgg_small |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| shipped tables | 17.83 | 20.70 | 0.402 | 0.643 | 0.473 | 1.56 | 0.800 | 0.756 | 0.835 | 0.642 | 0.408 | 0.428 | 0.759 | 0.696 |
| alex_blur (photo profile forced; otherwise 3/8 images fall back to the smooth profile) | 17.78 | 20.56 | 0.398 | 0.636 | 0.461 | 1.60 | 0.881 | 0.797 | 0.841 | 0.649 | 0.392 | 0.427 | 0.769 | 0.710 |
| vgg_blur (photo profile forced; otherwise 3/8 images fall back to the smooth profile) | 17.65 | 20.43 | 0.402 | 0.659 | 0.497 | 1.67 | 0.782 | 0.736 | 0.842 | 0.633 | 0.420 | 0.431 | 0.768 | 0.715 |
| alex_small (photo profile forced; otherwise 8/8 images fall back to the smooth profile) | 11.09 | 11.61 | 0.173 | 0.601 | 0.451 | 3.46 | 1.683 | 0.790 | 0.849 | 0.744 | 0.543 | 0.582 | 0.635 | 0.735 |
| vgg_small (photo profile forced; otherwise 8/8 images fall back to the smooth profile) | 13.25 | 14.14 | 0.178 | 0.646 | 0.488 | 2.53 | 0.984 | 0.758 | 0.899 | 0.749 | 0.475 | 0.533 | 0.850 | 0.703 |
| alex_small+10mse (photo profile forced; otherwise 8/8 images fall back to the smooth profile) | 16.94 | 19.09 | 0.352 | 0.618 | 0.450 | 2.87 | 1.570 | 0.956 | 0.829 | 0.713 | 0.449 | 0.513 | 0.631 | 0.739 |
