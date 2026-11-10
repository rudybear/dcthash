// DCTHash encoder.  Integer arithmetic only (all values are exact integers below 2^53), so every
// conforming encoder produces the identical string for the same pixels.
import {
  ASPECT_UNSPECIFIED, DCTHashError, DEFAULT_LENGTH, FULL, MAX_LENGTH, MIN_LENGTH, allocation, aspectCode, companded,
  profiles, range,
} from "./core.js";
import type { Profile } from "./core.js";
import { A } from "./tables.js";

export interface EncodeOptions {
  /** Hash length in characters, 6..96.  Default 28 (the length of a 4x3 BlurHash). */
  length?: number;
  /** Store the approximate aspect ratio in the hash (default true). */
  aspect?: boolean;
}

/** Largest input width/height accepted by encode(). */
export const MAX_ENCODE_SIDE = 16384;

let COS: Float64Array | undefined;

/** COS[u * 32 + x] = round(2^14 cos(pi u (2x + 1) / 64)); identical to the frozen table in the spec. */
export function cosTable(): Float64Array {
  if (!COS) {
    COS = new Float64Array(16 * 32);
    for (let u = 0; u < 16; u++) {
      for (let x = 0; x < 32; x++) COS[u * 32 + x] = Math.floor(16384 * Math.cos((Math.PI * u * (2 * x + 1)) / 64) + 0.5);
    }
  }
  return COS;
}

/** For each source index i in 0..n-1: the grid cells it overlaps and the integer overlap lengths. */
function overlaps(n: number): { start: Int32Array; cell: Int32Array; weight: Int32Array } {
  const start = new Int32Array(n + 1);
  const cell: number[] = [];
  const weight: number[] = [];
  for (let i = 0; i < n; i++) {
    start[i] = cell.length;
    const lo = 32 * i;
    const hi = lo + 32;
    for (let j = Math.floor(lo / n); j < 32 && n * j < hi; j++) {
      const w = Math.min(hi, n * j + n) - Math.max(lo, n * j);
      if (w > 0) {
        cell.push(j);
        weight.push(w);
      }
    }
  }
  start[n] = cell.length;
  return { start, cell: Int32Array.from(cell), weight: Int32Array.from(weight) };
}

/**
 * Encode RGBA pixels (row-major, 4 bytes per pixel, straight alpha) to a DCTHash string.
 * Transparent pixels are composited over white.  Any image size works; there is no need to
 * downscale first, although doing so is faster.
 * @throws DCTHashError on bad arguments
 */
export function encode(pixels: ArrayLike<number>, width: number, height: number, options: EncodeOptions = {}): string {
  const length = options.length ?? DEFAULT_LENGTH;
  if (!Number.isInteger(length) || length < MIN_LENGTH || length > MAX_LENGTH) {
    throw new DCTHashError(`length must be an integer in ${MIN_LENGTH}..${MAX_LENGTH}`);
  }
  if (
    !Number.isInteger(width) || !Number.isInteger(height) ||
    width < 1 || height < 1 || width > MAX_ENCODE_SIDE || height > MAX_ENCODE_SIDE
  ) {
    throw new DCTHashError(`width and height must be integers in 1..${MAX_ENCODE_SIDE}`);
  }
  if (!pixels || pixels.length !== width * height * 4) {
    throw new DCTHashError("pixels must hold width * height * 4 RGBA bytes");
  }

  // 1. box-filter the image (composited over white) onto a 32x32 grid; sums are exact integers
  const ox = overlaps(width);
  const oy = overlaps(height);
  const sums = new Float64Array(32 * 32 * 3);
  const rowAcc = new Float64Array(32 * 3);
  for (let y = 0; y < height; y++) {
    rowAcc.fill(0);
    let p = y * width * 4;
    for (let x = 0; x < width; x++, p += 4) {
      const a = pixels[p + 3];
      let r = pixels[p];
      let g = pixels[p + 1];
      let b = pixels[p + 2];
      if (a !== 255) {
        const bg = 255 * (255 - a) + 127;
        r = Math.floor((r * a + bg) / 255);
        g = Math.floor((g * a + bg) / 255);
        b = Math.floor((b * a + bg) / 255);
      }
      for (let k = ox.start[x]; k < ox.start[x + 1]; k++) {
        const w = ox.weight[k];
        const o = ox.cell[k] * 3;
        rowAcc[o] += w * r;
        rowAcc[o + 1] += w * g;
        rowAcc[o + 2] += w * b;
      }
    }
    for (let k = oy.start[y]; k < oy.start[y + 1]; k++) {
      const w = oy.weight[k];
      const o = oy.cell[k] * 96;
      for (let i = 0; i < 96; i++) sums[o + i] += w * rowAcc[i];
    }
  }

  // 2. cell means in 1/16 of an 8-bit level, then integer YCbCr; P[c][y][x]
  const area = width * height;
  const half = Math.floor(area / 2);
  const P = new Float64Array(3 * 1024);
  for (let i = 0; i < 1024; i++) {
    const r = Math.floor((sums[3 * i] * 16 + half) / area);
    const g = Math.floor((sums[3 * i + 1] * 16 + half) / area);
    const b = Math.floor((sums[3 * i + 2] * 16 + half) / area);
    P[i] = Math.floor((19595 * r + 38470 * g + 7471 * b + 32768) / 65536);
    P[1024 + i] = Math.floor((-11059 * r - 21709 * g + 32768 * b + 32768) / 65536);
    P[2048 + i] = Math.floor((32768 * r - 27439 * g - 5329 * b + 32768) / 65536);
  }

  // 3. horizontal DCT pass: H[c][y][u] = sum_x COS[u][x] P[c][y][x]
  const cos = cosTable();
  const H = new Float64Array(3 * 32 * 16);
  for (let cy = 0; cy < 96; cy++) {
    for (let u = 0; u < 16; u++) {
      let s = 0;
      for (let x = 0; x < 32; x++) s += cos[u * 32 + x] * P[cy * 32 + x];
      H[cy * 16 + u] = s;
    }
  }

  // 4. vertical pass: X[c][v][u] = 16 x unnormalised DCT coefficient (exact integer)
  const X = new Float64Array(768);
  for (let c = 0; c < 3; c++) {
    for (let v = 0; v < 16; v++) {
      for (let u = 0; u < 16; u++) {
        let s = 0;
        for (let y = 0; y < 32; y++) s += cos[v * 32 + y] * H[c * 512 + y * 16 + u];
        X[c * 256 + v * 16 + u] = Math.floor((s + 134217728) / 268435456);
      }
    }
  }

  // 5. quantise under each candidate (photo profile with contrast scale 0..3, then the smooth profile)
  //    and keep the one with the smallest weighted squared error; ties go to the earlier candidate
  const [photo, smooth] = profiles();
  let best: Uint8Array | undefined;
  let bestErr = Infinity;
  for (let cand = 0; cand < 5; cand++) {
    const p = cand < 4 ? photo : smooth;
    const out = new Uint8Array(length);
    const err = quantise(p, cand & 3, X, length, out);
    if (err < bestErr) {
      bestErr = err;
      best = out;
    }
  }
  const out = best!;
  out[0] |= options.aspect === false ? ASPECT_UNSPECIFIED : aspectCode(width, height);
  let s = "";
  for (let i = 0; i < length; i++) s += A[out[i]];
  return s;
}

const CWEIGHT = [12, 13, 10];

/** Quantises X under one profile / scale into `out` (6-bit symbols; aspect bits left 0) and returns the error E. */
function quantise(p: Profile, scale: number, X: Float64Array, length: number, out: Uint8Array): number {
  const bits = allocation(p, length);
  const rec = new Float64Array(768); // reconstruction x 2^16, T units
  let pos = 6;
  const write = (q: number, n: number): void => {
    for (let k = n - 1; k >= 0; k--, pos++) out[(pos / 6) | 0] |= ((q >> k) & 1) << (5 - (pos % 6));
  };
  write(p.smooth ? 1 : 0, 1);
  if (!p.smooth) write(scale, 2);
  for (let i = 0; i < p.n; i++) {
    const b = bits[i];
    const k = p.ch[i] * 256 + p.v[i] * 16 + p.u[i];
    const x = X[k];
    const M = p.m[i];
    if (!b) {
      rec[k] = M * 65536;
      continue;
    }
    const levels = 1 << b;
    let q = 0;
    if (companded(p, i)) {
      // nearest level FULL n|n| / 4^(b-1), n = q - (2^(b-1) - 1): count the midpoints at or below x
      const off = levels / 2 - 1;
      const lhs = 2 * x * (levels / 2) * (levels / 2);
      for (let j = 1; j < levels; j++) {
        const n0 = j - 1 - off;
        const n1 = j - off;
        if (lhs >= 64 * FULL * (n0 * Math.abs(n0) + n1 * Math.abs(n1))) q++;
      }
      const n = q - off;
      rec[k] = FULL * n * Math.abs(n) * 2 ** (18 - 2 * b);
    } else {
      const L = range(p, i, b, scale);
      q = Math.floor(((x - 64 * M + 64 * L) * levels) / (128 * L));
      q = q < 0 ? 0 : q >= levels ? levels - 1 : q;
      rec[k] = (M - L) * 65536 + (2 * q + 1) * L * 2 ** (16 - b);
    }
    write(q, b);
  }
  let E = 0;
  for (let k = 0; k < 768; k++) {
    const e = Math.floor((X[k] * 1024 - rec[k]) / 32768);
    E += CWEIGHT[k >> 8] * (k & 15 ? 2 : 1) * (k & 240 ? 2 : 1) * e * e;
  }
  return E;
}
