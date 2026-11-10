Node v22.23.4, 384x256 source image, single thread, warm.

| string | library | encode 384x256 | decode 32x21 | decode 384x256 | averageColor |
|---|---|---|---|---|---|
| 28 chars | DCTHash | 0.57 ms | 29 us | 2.08 ms | 3 us |
| 28 chars (4x3) | blurhash 2.0.5 | 127.53 ms | 0.13 ms | 17.79 ms | - |
| 52 chars | DCTHash | 0.58 ms | 29 us | 2.07 ms | 3 us |
| 52 chars (6x4) | blurhash 2.0.5 | 259.95 ms | 0.19 ms | 25.88 ms | - |
| 94 chars | DCTHash | 0.61 ms | 33 us | 2.25 ms | 5 us |
| 94 chars (9x5) | blurhash 2.0.5 | 490.55 ms | 0.29 ms | 42.18 ms | - |
