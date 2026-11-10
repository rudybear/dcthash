# dcthash (TypeScript)

Compact image placeholders: a 6 to 96 character string that decodes to a blurry preview, an
average colour and an approximate aspect ratio. A BlurHash alternative that keeps about 1.3 dB
more detail on photographs at the same string length. Zero runtime dependencies, ESM, works in
browsers and Node. Format: [`../SPEC.md`](../SPEC.md).

```ts
import { encode, decode, averageColor, aspectRatio, isValid } from "dcthash";

// encode: RGBA bytes (e.g. ImageData.data), any size; no need to downscale first
const hash = encode(imageData.data, imageData.width, imageData.height);            // 28 chars
const longer = encode(imageData.data, imageData.width, imageData.height, { length: 40 });

// decode: small is fast (about 30 microseconds at 32x21); let the browser scale it up
const pixels = decode(hash, 32, 21);                                                // Uint8ClampedArray RGBA
ctx.putImageData(new ImageData(pixels, 32, 21), 0, 0);

averageColor(hash);   // { r, g, b } without decoding pixels
aspectRatio(hash);    // e.g. 1.5, or null if the encoder left it unspecified
isValid(hash);        // false for anything decode() would reject; never throws
```

Decoder only (smaller bundle): `import { decode } from "dcthash/decode"`.

## API

| function | notes |
|---|---|
| `encode(pixels, width, height, { length = 28, aspect = true })` | `pixels`: `ArrayLike<number>` of `width * height * 4` RGBA bytes, straight alpha (composited over white). `length` 6..96. Deterministic: the same pixels always give the same string, in every implementation. |
| `decode(hash, width, height)` | returns `Uint8ClampedArray` RGBA with alpha 255. `width`, `height` integers in 1..4096, area at most 2^22. |
| `averageColor(hash)` | `{ r, g, b }` |
| `aspectRatio(hash)` | `number \| null`, nearest of 31 common width/height ratios |
| `isValid(hash)` | `boolean` type guard |
| `DCTHashError` | thrown for malformed hashes and bad arguments; nothing else is thrown |

Every string of valid length and alphabet whose first character is in `A`-`f` decodes; there
are no checksums, so a corrupted but well-formed hash decodes to a different picture.

## Build, test, demo

```bash
cd ts
npm install
npm test            # builds, then runs the cross-validation and malformed-input tests
npm run bench       # timings next to the `blurhash` npm package -> ../results/ts_bench.md
```

`npm run build` writes `dist/` (ES modules + `.d.ts`), minified bundles
(`dist/dcthash.min.js`, `dist/dcthash-decode.min.js`, `dist/dcthash-encode.min.js`) with a size
report in `dist/sizes.json`, and the demo bundle.

Demo: after `npm install && npm run build`, open `ts/demo/index.html` directly in a browser
(double-click it, or `xdg-open ts/demo/index.html`). No server is needed; the page loads the
single classic script `demo/demo.js`. Drop an image to compare the original, BlurHash and
DCTHash at the same string length.

`src/tables.ts` is generated from `../format/tables.json` by `npm run gen` (part of the build).
The tests read `../format/vectors.json`, which is written by the Python reference
(`python scripts/make_vectors.py` in the parent directory).

## Sizes and speed

Measured with Node 22 on the development machine (see `../results/ts_bench.md`):

| bundle | minified | gzip |
|---|---|---|
| decoder + helpers | 7.6 kB | 4.1 kB |
| encoder | 8.6 kB | 4.5 kB |
| everything | 10.8 kB | 5.5 kB |

About half of the decoder bundle is the frozen tables.

| | DCTHash (28 chars) | blurhash 2.0.5 (4x3, 28 chars) |
|---|---|---|
| decode 32x21 | 0.03 ms | 0.13 ms |
| decode 384x256 | 2.1 ms | 17.8 ms |
| encode from 384x256 | 0.6 ms | 128 ms |
| average colour | 0.003 ms | n/a |
