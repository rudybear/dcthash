Single-process timings at 32 bytes, median over 4 Kodak images (384x256), numpy decoders on one CPU core set, fitted encoders on GPU.

| method | encode | decode 384x256 | decode 32px | decoder tables | decoder cost | JS/mobile practical? |
|---|---|---|---|---|---|---|
| blurhash | 5 ms | 2.4 ms | 0.16 ms | none | nx*ny cosine products per pixel (separable); ~60 lines | yes (reference) |
| blurhash_default | 2 ms | 2.4 ms | 0.14 ms | none | nx*ny cosine products per pixel (separable); ~60 lines | yes (reference) |
| blurhash_gamma | 3 ms | 0.3 ms | 0.11 ms | none | nx*ny cosine products per pixel (separable); ~60 lines | yes (reference) |
| dct_fixed | 1 ms | 1.9 ms | 1.60 ms | 9 KiB | sum of <=320 basis fns on 32x32 + bicubic upsample; basis = cosines (analytic) | yes |
| dct_ac | 20 ms | 2.0 ms | 1.65 ms | 4 KiB | sum of <=320 basis fns on 32x32 + bicubic upsample; basis = cosines (analytic); + ~40-line range decoder | yes |
| pca_fixed | 1 ms | 2.0 ms | 1.64 ms | 329 KiB | sum of <=320 basis fns on 32x32 + bicubic upsample; basis = learned KLT basis, 320x1024 table per aspect class (~330 KiB at 8 bit) | yes, but ships a basis table |
| pca_ac | 23 ms | 2.6 ms | 2.19 ms | 324 KiB | sum of <=320 basis fns on 32x32 + bicubic upsample; basis = learned KLT basis, 320x1024 table per aspect class (~330 KiB at 8 bit); + ~40-line range decoder | yes, but ships a basis table |
| legendre_ac | 27 ms | 1.5 ms | 1.17 ms | 4 KiB | sum of <=320 basis fns on 32x32 + bicubic upsample; basis = polynomials (analytic); + ~40-line range decoder | yes |
| haar_ac | 22 ms | 2.1 ms | 1.66 ms | 4 KiB | sum of <=320 basis fns on 32x32 + bicubic upsample; basis = Haar boxes (analytic); + ~40-line range decoder | yes |
| bspline_grid | 52 ms | 10.0 ms | 0.27 ms | none | separable cubic B-spline evaluation (16 taps/pixel) or just a GPU/bilinear-ish upscale of a tiny texture; ~50 lines | yes (simplest of all) |
| gauss_add | 1.2 s | 7.2 ms | 0.12 ms | none | n primitives x 1 exp per pixel (not separable); ~40 lines | decoder yes (n<=21 exps/pixel on a 32x32 canvas, then upscale); ENCODER needs iterative optimisation |
| gauss_norm | 1.3 s | 8.5 ms | 0.14 ms | none | n primitives x 1 exp per pixel (not separable); ~40 lines | decoder yes (n<=21 exps/pixel on a 32x32 canvas, then upscale); ENCODER needs iterative optimisation |
| soft_voronoi | 1.0 s | 8.7 ms | 0.15 ms | none | n primitives x 1 exp per pixel (not separable); ~40 lines; needs softmax over sites | decoder yes (n<=21 exps/pixel on a 32x32 canvas, then upscale); ENCODER needs iterative optimisation |
| siren | 2.3 s | 5.5 ms | 0.12 ms | none | one tiny MLP evaluation per pixel (H sines, or H + H^2 mults for 2 layers); ~30 lines | decoder yes (evaluate at 32x32 and upscale); ENCODER needs iterative optimisation with restarts |
| dct_ac+gauss | 2.2 s | 3.7 ms | 1.50 ms | none | DCT decoder + k exps per pixel | decoder yes; ENCODER needs iterative optimisation |
| bestfit_fast | 97 ms | 2.1 ms | 1.68 ms | none | union of the candidate decoders | as the worst candidate |
| bestfit_all | 7.9 s | 2.6 ms | 1.65 ms | none | union of the candidate decoders | as the worst candidate |
