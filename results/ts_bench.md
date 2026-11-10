Node v22.23.4, 384x256 source image, single thread, warm.

| string | library | encode 384x256 | decode 32x21 | decode 384x256 | averageColor |
|---|---|---|---|---|---|
| 28 chars | DCTHash | 0.61 ms | 30 us | 2.32 ms | 3 us |
| 28 chars (4x3) | blurhash 2.0.5 | 131.74 ms | 0.13 ms | 18.34 ms | - |
| 52 chars | DCTHash | 0.62 ms | 30 us | 2.29 ms | 3 us |
| 52 chars (6x4) | blurhash 2.0.5 | 268.54 ms | 0.20 ms | 27.17 ms | - |
| 94 chars | DCTHash | 0.64 ms | 32 us | 2.32 ms | 5 us |
| 94 chars (9x5) | blurhash 2.0.5 | 511.68 ms | 0.29 ms | 43.84 ms | - |
