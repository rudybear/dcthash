// BlurHash encoder following the reference C implementation (woltapp/blurhash, C/encode.c).
//
// Why not the `blurhash` npm package's encode()? Version 2.0.5 takes the largest *signed* AC
// coefficient as the quantiser range (`Math.max(...ac.map(c => Math.max(...c)))`) where the C
// reference takes the largest *magnitude* (fabsf). When the strongest component is negative the
// npm encoder clips it, which costs BlurHash quality (about 1.2 dB on the built-in sample).
// The comparison should be against BlurHash at its best, so the demo encodes with this function
// and decodes with the npm package's decode(), which matches the reference.
const B83 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz#$%*+,-.:;=?@[]^_{|}~";

function b83(value: number, length: number): string {
  let s = "";
  for (let i = 1; i <= length; i++) s += B83[Math.floor(value / 83 ** (length - i)) % 83];
  return s;
}

const toLinear = (v: number): number => {
  const e = v / 255;
  return e <= 0.04045 ? e / 12.92 : ((e + 0.055) / 1.055) ** 2.4;
};
const toSrgb = (v: number): number => {
  const e = Math.max(0, Math.min(1, v));
  return e <= 0.0031308 ? Math.trunc(e * 12.92 * 255 + 0.5) : Math.trunc((1.055 * e ** (1 / 2.4) - 0.055) * 255 + 0.5);
};
const signPow = (v: number, e: number): number => Math.sign(v) * Math.abs(v) ** e;

export function encodeBlurHash(pixels: Uint8ClampedArray, width: number, height: number, nx: number, ny: number): string {
  const n = width * height;
  const lin = new Float32Array(n * 3);
  const lut = new Float32Array(256);
  for (let i = 0; i < 256; i++) lut[i] = toLinear(i);
  for (let p = 0; p < n; p++) for (let c = 0; c < 3; c++) lin[p * 3 + c] = lut[pixels[p * 4 + c]];
  const factors: number[][] = [];
  const cx = new Float32Array(width);
  const cy = new Float32Array(height);
  for (let j = 0; j < ny; j++) {
    for (let y = 0; y < height; y++) cy[y] = Math.cos((Math.PI * j * y) / height);
    for (let i = 0; i < nx; i++) {
      for (let x = 0; x < width; x++) cx[x] = Math.cos((Math.PI * i * x) / width);
      let r = 0, g = 0, b = 0;
      for (let y = 0, p = 0; y < height; y++) {
        for (let x = 0; x < width; x++, p += 3) {
          const w = cx[x] * cy[y];
          r += w * lin[p]; g += w * lin[p + 1]; b += w * lin[p + 2];
        }
      }
      const s = (i === 0 && j === 0 ? 1 : 2) / n;
      factors.push([r * s, g * s, b * s]);
    }
  }
  const dc = factors[0];
  const ac = factors.slice(1);
  let out = b83(nx - 1 + (ny - 1) * 9, 1);
  let max = 1;
  if (ac.length > 0) {
    let actual = 0;
    for (const f of ac) for (const v of f) actual = Math.max(actual, Math.abs(v));   // magnitude, as in the C reference
    const q = Math.max(0, Math.min(82, Math.floor(actual * 166 - 0.5)));
    max = (q + 1) / 166;
    out += b83(q, 1);
  } else {
    out += b83(0, 1);
  }
  out += b83((toSrgb(dc[0]) << 16) + (toSrgb(dc[1]) << 8) + toSrgb(dc[2]), 4);
  const quant = (v: number): number => Math.max(0, Math.min(18, Math.floor(signPow(v / max, 0.5) * 9 + 9.5)));
  for (const f of ac) out += b83(quant(f[0]) * 361 + quant(f[1]) * 19 + quant(f[2]), 2);
  return out;
}
