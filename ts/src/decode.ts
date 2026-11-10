// DCTHash decoder and cheap helpers.  No dependencies.
import { DCTHashError, FULL, MAX_DECODE_AREA, MAX_DECODE_SIDE, allocation, aspectOf, companded, profiles, range, symbols } from "./core.js";

export { DCTHashError, MIN_LENGTH, MAX_LENGTH, DEFAULT_LENGTH } from "./core.js";

/**
 * De-quantised coefficients as K[channel * 256 + v * 16 + u], scaled so that
 * pixel(x, y) = sum K * cos(pi u (x + .5) / W) * cos(pi v (y + .5) / H)   (8-bit levels, YCbCr).
 */
function coefficients(hash: string): Float64Array {
  const sym = symbols(hash);
  let pos = 6;
  const read = (n: number): number => {
    let q = 0;
    for (let k = 0; k < n; k++, pos++) q = (q << 1) | ((sym[(pos / 6) | 0] >> (5 - (pos % 6))) & 1);
    return q;
  };
  const p = profiles()[read(1)];
  const scale = p.smooth ? 0 : read(2);
  const bits = allocation(p, sym.length);
  const K = new Float64Array(768);
  for (let i = 0; i < p.n; i++) {
    const b = bits[i];
    let val = p.m[i];
    if (b) {
      const q = read(b);
      if (companded(p, i)) {
        const n = q - ((1 << (b - 1)) - 1);
        val = (FULL * n * Math.abs(n)) / (1 << (2 * b - 2));
      } else {
        const L = range(p, i, b, scale);
        val += ((2 * q + 1) * L) / (1 << b) - L;
      }
    }
    const u = p.u[i];
    const v = p.v[i];
    K[p.ch[i] * 256 + v * 16 + u] = (val * (u ? 2 : 1) * (v ? 2 : 1)) / 256;
  }
  return K;
}

function toByte(x: number): number {
  x = Math.floor(x + 0.5);
  return x < 0 ? 0 : x > 255 ? 255 : x;
}

function cosines(size: number): Float64Array {
  const c = new Float64Array(16 * size);
  for (let f = 0; f < 16; f++) {
    for (let i = 0; i < size; i++) c[f * size + i] = Math.cos((Math.PI * f * (i + 0.5)) / size);
  }
  return c;
}

/**
 * Decode a hash to RGBA pixels (alpha = 255), row-major, `width * height * 4` bytes.
 * The result can be passed straight to `new ImageData(pixels, width, height)`.
 * A small size (e.g. 32 px on the long side) scaled up by the browser is the cheapest way to display it.
 * @throws DCTHashError on a malformed hash or an unsupported size
 */
export function decode(hash: string, width: number, height: number): Uint8ClampedArray {
  if (
    !Number.isInteger(width) || !Number.isInteger(height) ||
    width < 1 || height < 1 || width > MAX_DECODE_SIDE || height > MAX_DECODE_SIDE ||
    width * height > MAX_DECODE_AREA
  ) {
    throw new DCTHashError(`width and height must be integers in 1..${MAX_DECODE_SIDE} with width*height <= ${MAX_DECODE_AREA}`);
  }
  const K = coefficients(hash);
  const cx = cosines(width);
  const cy = cosines(height);
  const n = width * height;
  const plane = new Float64Array(3 * n);
  const row = new Float64Array(16);
  for (let c = 0; c < 3; c++) {
    // highest frequencies actually present in this channel (chroma uses far fewer than luma)
    let nu = 0;
    let nv = 0;
    for (let v = 0; v < 16; v++) {
      for (let u = 0; u < 16; u++) {
        if (K[c * 256 + v * 16 + u] !== 0) {
          if (u >= nu) nu = u + 1;
          if (v >= nv) nv = v + 1;
        }
      }
    }
    for (let y = 0; y < height; y++) {
      for (let u = 0; u < nu; u++) {
        let s = 0;
        for (let v = 0; v < nv; v++) s += K[c * 256 + v * 16 + u] * cy[v * height + y];
        row[u] = s;
      }
      const o = c * n + y * width;
      for (let x = 0; x < width; x++) {
        let s = 0;
        for (let u = 0; u < nu; u++) s += row[u] * cx[u * width + x];
        plane[o + x] = s;
      }
    }
  }
  const out = new Uint8ClampedArray(4 * n);
  for (let i = 0; i < n; i++) {
    const y = plane[i];
    const cb = plane[n + i];
    const cr = plane[2 * n + i];
    out[4 * i] = toByte(y + 1.402 * cr);
    out[4 * i + 1] = toByte(y - 0.344136 * cb - 0.714136 * cr);
    out[4 * i + 2] = toByte(y + 1.772 * cb);
    out[4 * i + 3] = 255;
  }
  return out;
}

/** Mean colour of the placeholder, without decoding any pixels. */
export function averageColor(hash: string): { r: number; g: number; b: number } {
  const K = coefficients(hash);
  const y = K[0];
  const cb = K[256];
  const cr = K[512];
  return { r: toByte(y + 1.402 * cr), g: toByte(y - 0.344136 * cb - 0.714136 * cr), b: toByte(y + 1.772 * cb) };
}

/**
 * Approximate width / height of the original image as stored in the hash:
 * the nearest of 31 common ratios (1:5 ... 5:1), or `null` if the encoder left it unspecified.
 */
export function aspectRatio(hash: string): number | null {
  return aspectOf(symbols(hash)[0] & 31);
}

/** True if `hash` is a well-formed DCTHash string (right alphabet, length and version). Never throws. */
export function isValid(hash: unknown): hash is string {
  try {
    symbols(hash);
    return true;
  } catch {
    return false;
  }
}
