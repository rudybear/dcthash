// Malformed input must fail cleanly (DCTHashError), never crash, hang or return garbage silently.
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { encode, decode, averageColor, aspectRatio, isValid, DCTHashError, MIN_LENGTH, MAX_LENGTH } from "../dist/index.js";

const vectors = JSON.parse(readFileSync(new URL("../../format/vectors.json", import.meta.url), "utf8"));
const good = vectors.encode[0].hashes["28"];
const ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

test("invalid strings from the reference vectors are rejected by every entry point", () => {
  for (const s of vectors.invalid) {
    assert.equal(isValid(s), false, JSON.stringify(s));
    assert.throws(() => decode(s, 8, 8), DCTHashError, JSON.stringify(s));
    assert.throws(() => averageColor(s), DCTHashError);
    assert.throws(() => aspectRatio(s), DCTHashError);
  }
});

test("non-string hashes are rejected", () => {
  for (const s of [undefined, null, 42, {}, [], ["A"], new String(good), Symbol("x"), 10n, () => good]) {
    assert.equal(isValid(s), false);
    assert.throws(() => decode(s, 8, 8), DCTHashError);
  }
});

test("every wrong length and every out-of-alphabet ASCII character is rejected", () => {
  for (const n of [0, 1, MIN_LENGTH - 1, MAX_LENGTH + 1, 200, 100000]) assert.throws(() => decode("A".repeat(n), 4, 4), DCTHashError);
  for (let c = 0; c < 256; c++) {
    const ch = String.fromCharCode(c);
    const s = good.slice(0, 10) + ch + good.slice(11);
    assert.equal(isValid(s), ALPHABET.includes(ch), `char code ${c}`);
  }
  for (const ch of ALPHABET.slice(32)) assert.equal(isValid(ch + good.slice(1)), false, "version bit set");
});

test("bad output sizes are rejected before any allocation", () => {
  for (const [w, h] of [[0, 8], [8, 0], [-1, 8], [1.5, 8], [NaN, 8], [Infinity, 8], [8, "8"], [4097, 1], [1, 4097], [4096, 4096], [undefined, 8]]) {
    assert.throws(() => decode(good, w, h), DCTHashError, `${w}x${h}`);
  }
  assert.equal(decode(good, 4096, 1).length, 4096 * 4);
  assert.equal(decode(good, 1, 1).length, 4);
});

test("bad encoder arguments are rejected", () => {
  const px = new Uint8Array(4 * 4 * 4);
  assert.throws(() => encode(px, 4, 5), DCTHashError);
  assert.throws(() => encode(px, 0, 0), DCTHashError);
  assert.throws(() => encode(px, 4.5, 4), DCTHashError);
  assert.throws(() => encode(null, 4, 4), DCTHashError);
  assert.throws(() => encode(new Uint8Array(4 * 4 * 3), 4, 4), DCTHashError);
  assert.throws(() => encode(px, 20000, 1), DCTHashError);
  for (const length of [0, 5, 97, 28.5, NaN, "28", -1]) assert.throws(() => encode(px, 4, 4, { length }), DCTHashError, String(length));
  assert.equal(encode(px, 4, 4, { length: MIN_LENGTH }).length, MIN_LENGTH);
  assert.equal(encode(px, 4, 4, { length: MAX_LENGTH }).length, MAX_LENGTH);
});

test("fuzz: every well-formed string decodes, quickly, to in-range pixels", () => {
  let seed = 0x9e3779b9;
  const rnd = () => ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 4294967296);
  const t0 = performance.now();
  for (let k = 0; k < 3000; k++) {
    const n = MIN_LENGTH + Math.floor(rnd() * (MAX_LENGTH - MIN_LENGTH + 1));
    let s = ALPHABET[Math.floor(rnd() * 32)];
    for (let i = 1; i < n; i++) s += ALPHABET[Math.floor(rnd() * 64)];
    assert.equal(isValid(s), true);
    const px = decode(s, 9, 7);
    assert.equal(px.length, 9 * 7 * 4);
    for (let i = 3; i < px.length; i += 4) assert.equal(px[i], 255);
    const a = averageColor(s);
    assert.ok(a.r >= 0 && a.r <= 255 && a.g >= 0 && a.g <= 255 && a.b >= 0 && a.b <= 255);
    const r = aspectRatio(s);
    assert.ok(r === null || (r >= 0.2 && r <= 5));
  }
  const ms = performance.now() - t0;
  console.log(`  3000 random hashes decoded in ${ms.toFixed(0)} ms`);
  assert.ok(ms < 5000);
});

test("fuzz: random garbage strings never crash", () => {
  let seed = 12345;
  const rnd = () => ((seed = (Math.imul(seed, 1103515245) + 12345) >>> 0) / 4294967296);
  for (let k = 0; k < 3000; k++) {
    const n = Math.floor(rnd() * 120);
    let s = "";
    for (let i = 0; i < n; i++) s += String.fromCharCode(Math.floor(rnd() * (rnd() < 0.1 ? 65536 : 128)));
    if (isValid(s)) decode(s, 4, 4);
    else assert.throws(() => decode(s, 4, 4), DCTHashError);
  }
});

test("encoder handles extreme shapes and a large image", () => {
  for (const [w, h] of [[1, 1], [1, 500], [500, 1], [3, 1000], [33, 31]]) {
    const px = new Uint8Array(w * h * 4).map((_, i) => (i * 37) & 255);
    assert.equal(isValid(encode(px, w, h)), true, `${w}x${h}`);
  }
  const w = 3000, h = 2000;
  const px = new Uint8Array(w * h * 4);
  for (let i = 0; i < px.length; i++) px[i] = (i * 2654435761) >>> 24;
  const t0 = performance.now();
  const s = encode(px, w, h, { length: 40 });
  console.log(`  3000x2000 encode: ${(performance.now() - t0).toFixed(0)} ms -> ${s}`);
  assert.equal(s.length, 40);
});
