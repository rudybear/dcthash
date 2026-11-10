// Shared by encoder and decoder: alphabet, frozen tables, bit allocation, quantiser definitions.
import { A, G, PC, PM, PQ, PS, R, SC, SQ } from "./tables.js";

export const MIN_LENGTH = 6;
export const MAX_LENGTH = 96;
export const DEFAULT_LENGTH = 28;
/** Largest width/height accepted by decode(), and largest output area in pixels. */
export const MAX_DECODE_SIDE = 4096;
export const MAX_DECODE_AREA = 1 << 22;
export const ASPECT_UNSPECIFIED = 31;
/** 127.5 grey levels in T units: half the 8-bit range of a DC term. */
export const FULL = 32640;
/** Photo profile: contrast scale code -> multiplier x4 of the non-DC quantiser ranges. */
const SCALES = [2, 3, 4, 6];

export class DCTHashError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "DCTHashError";
  }
}

/** One table set.  Coefficient i is (channel ch[i], horizontal frequency u[i], vertical frequency v[i]). */
export interface Profile {
  smooth: boolean;
  n: number;
  ch: Uint8Array;
  u: Uint8Array;
  v: Uint8Array;
  /** centre M and scale S in T units (S is unused in the smooth profile) */
  m: Int32Array;
  s: Int32Array;
  /** k-th payload bit goes to coefficient seq[k] */
  seq: Uint16Array;
}

const digit = new Int8Array(128).fill(-1);
let cached: [Profile, Profile] | undefined;

function unpack(src: string, chars: number, out: { [i: number]: number }, offset = 0): void {
  for (let i = 0, k = 0; i < src.length; k++) {
    let v = 0;
    for (let j = 0; j < chars; j++) v = v * 64 + digit[src.charCodeAt(i++)];
    out[k] = v - offset;
  }
}

function profile(smooth: boolean, coefs: string, seq: string): Profile {
  const n = coefs.length / 2;
  const packed = new Uint16Array(n);
  unpack(coefs, 2, packed);
  const p: Profile = {
    smooth, n,
    ch: new Uint8Array(n), u: new Uint8Array(n), v: new Uint8Array(n),
    m: new Int32Array(n), s: new Int32Array(n), seq: new Uint16Array(seq.length / 2),
  };
  for (let i = 0; i < n; i++) {
    p.ch[i] = packed[i] >> 8;
    p.u[i] = (packed[i] >> 4) & 15;
    p.v[i] = packed[i] & 15;
    if (smooth && !packed[i]) p.m[i] = FULL; // luma DC is centred on mid grey; everything else on 0
  }
  unpack(seq, 2, p.seq);
  return p;
}

/** [photo, smooth] */
export function profiles(): [Profile, Profile] {
  if (cached) return cached;
  for (let i = 0; i < 64; i++) digit[A.charCodeAt(i)] = i;
  const photo = profile(false, PC, PQ);
  unpack(PM, 3, photo.m, 131072);
  unpack(PS, 3, photo.s);
  return (cached = [photo, profile(true, SC, SQ)]);
}

/** Bits per coefficient for a hash of `length` characters (header: 7 bits, +2 in the photo profile). */
export function allocation(p: Profile, length: number): Uint8Array {
  const bits = new Uint8Array(p.n);
  for (let k = 6 * length - (p.smooth ? 7 : 9) - 1; k >= 0; k--) bits[p.seq[k]]++;
  return bits;
}

/** Companded quantiser (levels FULL n|n| / 4^(b-1)) or uniform? */
export function companded(p: Profile, i: number): boolean {
  return p.u[i] | p.v[i] ? p.smooth : p.ch[i] > 0;
}

/** Half-range L (T units) of the uniform quantiser of coefficient i at bit depth b >= 1. */
export function range(p: Profile, i: number, b: number, scale: number): number {
  if (p.smooth || !(p.u[i] | p.v[i])) return FULL;
  let L = Math.floor((p.s[i] * G[b - 1] + 128) / 256);
  if (L < 1) L = 1;
  L = Math.floor((L * SCALES[scale]) / 4);
  return L < 1 ? 1 : L;
}

/** Validates a hash and returns its 6-bit symbols.  Throws DCTHashError. */
export function symbols(hash: unknown): Uint8Array {
  if (typeof hash !== "string") throw new DCTHashError("hash must be a string");
  const n = hash.length;
  if (n < MIN_LENGTH || n > MAX_LENGTH) {
    throw new DCTHashError(`hash length must be ${MIN_LENGTH}..${MAX_LENGTH} characters, got ${n}`);
  }
  profiles();
  const out = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const c = hash.charCodeAt(i);
    const d = c < 128 ? digit[c] : -1;
    if (d < 0) throw new DCTHashError(`invalid character at position ${i}`);
    out[i] = d;
  }
  if (out[0] & 32) throw new DCTHashError("unsupported DCTHash version");
  return out;
}

export function aspectOf(code: number): number | null {
  return code === ASPECT_UNSPECIFIED ? null : R[2 * code] / R[2 * code + 1];
}

/** Aspect code for a width x height image: nearest table ratio on a log scale, exact integer rule. */
export function aspectCode(width: number, height: number): number {
  let a = 0;
  for (let k = 0; k < 30; k++) {
    if (width * width * R[2 * k + 1] * R[2 * k + 3] >= height * height * R[2 * k] * R[2 * k + 2]) a++;
  }
  return a;
}
