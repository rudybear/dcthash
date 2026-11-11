
### kodak, 28 chars: encoder-side search (decoder, tables and format unchanged)

| loss minimised by the encoder | psnr | psnr_blur | ms_ssim | dists | dists_blur | dc_ok | cf_ratio | sat_top_ratio | lpips_alex_sharp | lpips_vgg_sharp | lpips_alex_blur | lpips_vgg_blur | lpips_alex_small | lpips_vgg_small |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none (shipped encoder) | 17.61 | 20.58 | 0.390 | 0.661 | 0.494 | 1.70 | 0.798 | 0.683 | 0.897 | 0.752 | 0.437 | 0.474 | 0.809 | 0.755 |
| alex_sharp (56 index changes, 0.7 s/image) | 13.94 | 14.95 | 0.331 | 0.645 | 0.489 | 2.39 | 0.838 | 0.599 | 0.791 | 0.769 | 0.493 | 0.542 | 0.747 | 0.754 |
| alex_blur (42 index changes, 0.6 s/image) | 16.23 | 18.18 | 0.367 | 0.644 | 0.464 | 1.84 | 0.839 | 0.692 | 0.873 | 0.779 | 0.377 | 0.480 | 0.813 | 0.750 |
| vgg_blur (27 index changes, 7.5 s/image) | 16.72 | 18.98 | 0.380 | 0.662 | 0.490 | 1.70 | 0.768 | 0.626 | 0.894 | 0.746 | 0.428 | 0.445 | 0.828 | 0.746 |
| alex_small (58 index changes, 0.5 s/image) | 15.00 | 16.44 | 0.347 | 0.652 | 0.495 | 2.96 | 1.139 | 0.742 | 0.870 | 0.782 | 0.516 | 0.557 | 0.639 | 0.767 |
| vgg_small (42 index changes, 1.1 s/image) | 15.52 | 17.10 | 0.372 | 0.654 | 0.486 | 2.02 | 0.876 | 0.625 | 0.892 | 0.751 | 0.460 | 0.486 | 0.799 | 0.665 |
| vgg_blur+10mse (14 index changes, 4.5 s/image) | 17.49 | 20.35 | 0.389 | 0.661 | 0.491 | 1.69 | 0.780 | 0.639 | 0.894 | 0.745 | 0.428 | 0.451 | 0.818 | 0.740 |

### kodak, 40 chars: encoder-side search (decoder, tables and format unchanged)

| loss minimised by the encoder | psnr | psnr_blur | ms_ssim | dists | dists_blur | dc_ok | cf_ratio | sat_top_ratio | lpips_alex_sharp | lpips_vgg_sharp | lpips_alex_blur | lpips_vgg_blur | lpips_alex_small | lpips_vgg_small |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none (shipped encoder) | 17.91 | 21.14 | 0.410 | 0.646 | 0.466 | 1.54 | 0.845 | 0.731 | 0.896 | 0.762 | 0.403 | 0.456 | 0.811 | 0.740 |
| alex_sharp (59 index changes, 1.7 s/image) | 14.71 | 15.98 | 0.342 | 0.626 | 0.456 | 1.94 | 0.866 | 0.676 | 0.792 | 0.768 | 0.455 | 0.515 | 0.746 | 0.744 |
| alex_blur (51 index changes, 1.4 s/image) | 16.79 | 19.11 | 0.397 | 0.632 | 0.438 | 1.73 | 0.899 | 0.745 | 0.878 | 0.785 | 0.341 | 0.465 | 0.819 | 0.729 |
| vgg_blur (41 index changes, 13.2 s/image) | 17.02 | 19.51 | 0.402 | 0.646 | 0.463 | 1.57 | 0.838 | 0.680 | 0.895 | 0.754 | 0.394 | 0.424 | 0.825 | 0.726 |
| alex_small (59 index changes, 0.4 s/image) | 16.30 | 18.33 | 0.371 | 0.639 | 0.469 | 2.51 | 1.097 | 0.811 | 0.868 | 0.782 | 0.468 | 0.528 | 0.644 | 0.746 |
| vgg_small (54 index changes, 1.2 s/image) | 16.25 | 18.21 | 0.392 | 0.641 | 0.458 | 1.86 | 0.915 | 0.679 | 0.898 | 0.762 | 0.422 | 0.471 | 0.810 | 0.635 |
| vgg_blur+10mse (19 index changes, 6.2 s/image) | 17.80 | 20.92 | 0.409 | 0.644 | 0.459 | 1.53 | 0.840 | 0.689 | 0.895 | 0.756 | 0.393 | 0.433 | 0.814 | 0.727 |

### holdout, 28 chars: encoder-side search (decoder, tables and format unchanged)

| loss minimised by the encoder | psnr | psnr_blur | ms_ssim | dists | dists_blur | dc_ok | cf_ratio | sat_top_ratio | lpips_alex_sharp | lpips_vgg_sharp | lpips_alex_blur | lpips_vgg_blur | lpips_alex_small | lpips_vgg_small |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none (shipped encoder) | 17.59 | 20.31 | 0.392 | 0.654 | 0.493 | 1.72 | 0.778 | 0.748 | 0.835 | 0.639 | 0.431 | 0.440 | 0.750 | 0.711 |
| alex_sharp (53 index changes, 0.9 s/image) | 13.95 | 15.37 | 0.336 | 0.612 | 0.454 | 2.69 | 0.874 | 0.740 | 0.737 | 0.675 | 0.470 | 0.505 | 0.674 | 0.692 |
| alex_blur (38 index changes, 0.7 s/image) | 16.00 | 17.74 | 0.373 | 0.610 | 0.428 | 1.97 | 0.884 | 0.777 | 0.806 | 0.668 | 0.365 | 0.438 | 0.743 | 0.723 |
| vgg_blur (26 index changes, 6.6 s/image) | 16.30 | 18.52 | 0.378 | 0.635 | 0.459 | 1.74 | 0.773 | 0.740 | 0.829 | 0.643 | 0.402 | 0.408 | 0.755 | 0.690 |
| alex_small (55 index changes, 0.2 s/image) | 14.49 | 15.75 | 0.308 | 0.628 | 0.471 | 3.08 | 1.341 | 1.285 | 0.805 | 0.681 | 0.468 | 0.502 | 0.586 | 0.709 |
| vgg_small (37 index changes, 0.6 s/image) | 15.44 | 17.12 | 0.361 | 0.639 | 0.474 | 1.97 | 0.848 | 0.795 | 0.823 | 0.637 | 0.442 | 0.451 | 0.733 | 0.621 |
| vgg_blur+10mse (10 index changes, 2.5 s/image) | 17.50 | 20.17 | 0.390 | 0.647 | 0.480 | 1.70 | 0.784 | 0.742 | 0.839 | 0.642 | 0.414 | 0.420 | 0.761 | 0.702 |

### holdout, 40 chars: encoder-side search (decoder, tables and format unchanged)

| loss minimised by the encoder | psnr | psnr_blur | ms_ssim | dists | dists_blur | dc_ok | cf_ratio | sat_top_ratio | lpips_alex_sharp | lpips_vgg_sharp | lpips_alex_blur | lpips_vgg_blur | lpips_alex_small | lpips_vgg_small |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none (shipped encoder) | 17.83 | 20.70 | 0.402 | 0.643 | 0.473 | 1.56 | 0.800 | 0.756 | 0.835 | 0.642 | 0.408 | 0.428 | 0.759 | 0.696 |
| alex_sharp (57 index changes, 1.2 s/image) | 15.50 | 16.86 | 0.353 | 0.596 | 0.423 | 3.01 | 2.616 | 1.501 | 0.743 | 0.685 | 0.433 | 0.486 | 0.688 | 0.717 |
| alex_blur (56 index changes, 1.3 s/image) | 16.71 | 18.81 | 0.387 | 0.599 | 0.407 | 1.70 | 0.944 | 0.826 | 0.803 | 0.668 | 0.337 | 0.426 | 0.730 | 0.699 |
| vgg_blur (40 index changes, 13.6 s/image) | 16.77 | 19.15 | 0.388 | 0.623 | 0.438 | 1.56 | 0.806 | 0.768 | 0.827 | 0.649 | 0.381 | 0.396 | 0.764 | 0.693 |
| alex_small (59 index changes, 0.3 s/image) | 16.34 | 18.12 | 0.346 | 0.619 | 0.451 | 2.54 | 1.224 | 1.141 | 0.804 | 0.667 | 0.429 | 0.472 | 0.601 | 0.696 |
| vgg_small (54 index changes, 1.1 s/image) | 16.19 | 18.00 | 0.376 | 0.628 | 0.458 | 1.92 | 0.883 | 0.826 | 0.829 | 0.633 | 0.412 | 0.434 | 0.729 | 0.607 |
| vgg_blur+10mse (18 index changes, 6.1 s/image) | 17.79 | 20.60 | 0.405 | 0.630 | 0.453 | 1.55 | 0.807 | 0.768 | 0.830 | 0.646 | 0.388 | 0.404 | 0.755 | 0.692 |
