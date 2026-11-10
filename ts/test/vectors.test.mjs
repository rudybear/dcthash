// Cross-validation against the Python reference (format/vectors.json, written by scripts/make_vectors.py).
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { encode, decode, averageColor, aspectRatio, isValid } from "../dist/index.js";
import { cosTable } from "../dist/encode.js";

const load = (name) => JSON.parse(readFileSync(new URL(`../../format/${name}`, import.meta.url), "utf8"));
const vectors = load("vectors.json");
const tables = load("tables.json");
const bytes = (b64) => new Uint8Array(Buffer.from(b64, "base64"));

test("encoder cosine table equals the frozen table", () => {
  assert.deepEqual(Array.from(cosTable()), tables.cos.flat());
});

test("encode: strings identical to the Python reference", () => {
  let n = 0;
  for (const v of vectors.encode) {
    const px = bytes(v.rgba);
    for (const [len, expected] of Object.entries(v.hashes)) {
      assert.equal(encode(px, v.width, v.height, { length: Number(len) }), expected, `${v.name} length ${len}`);
      n++;
    }
    assert.equal(encode(px, v.width, v.height, { aspect: false }), v.no_aspect_28, `${v.name} without aspect`);
    assert.equal(encode(px, v.width, v.height), v.hashes["28"], `${v.name} default length`);
    assert.equal(encode(Array.from(px), v.width, v.height), v.hashes["28"], `${v.name} plain array input`);
    assert.equal(encode(new Uint8ClampedArray(px), v.width, v.height), v.hashes["28"], `${v.name} clamped array input`);
  }
  console.log(`  ${n} strings over ${vectors.encode.length} images matched exactly`);
});

test("decode: pixels match the Python reference within the stated tolerance", () => {
  let maxDiff = 0;
  let differing = 0;
  let total = 0;
  for (const v of vectors.decode) {
    const got = decode(v.hash, v.width, v.height);
    const want = bytes(v.rgba);
    assert.equal(got.length, want.length);
    for (let i = 0; i < got.length; i++) {
      const d = Math.abs(got[i] - want[i]);
      if (d) differing++;
      if (d > maxDiff) maxDiff = d;
    }
    total += got.length;
  }
  console.log(`  ${vectors.decode.length} images, ${total} bytes: max difference ${maxDiff}, ${differing} bytes differ`);
  assert.ok(maxDiff <= vectors.decode_tolerance, `max difference ${maxDiff}`);
});

test("helpers: average colour and aspect ratio match the reference", () => {
  for (const v of vectors.helpers) {
    const c = averageColor(v.hash);
    for (const [k, i] of [["r", 0], ["g", 1], ["b", 2]]) assert.ok(Math.abs(c[k] - v.average[i]) <= 1, `${v.hash} ${k}`);
    assert.equal(aspectRatio(v.hash), v.aspect);
    assert.equal(isValid(v.hash), true);
  }
});

test("average colour agrees with the mean of the decoded pixels", () => {
  for (const v of vectors.encode) {
    const h = v.hashes["28"];
    const px = decode(h, 32, 32);
    const mean = [0, 0, 0];
    for (let i = 0; i < px.length; i += 4) for (let c = 0; c < 3; c++) mean[c] += px[i + c] / 1024;
    const a = averageColor(h);
    // clipping at 0/255 in the decoded image pulls its mean away from the DC term (worst on images that are
    // saturated in one channel), hence the loose bound
    for (const [k, i] of [["r", 0], ["g", 1], ["b", 2]]) assert.ok(Math.abs(a[k] - mean[i]) <= 25, `${v.name} ${k}: ${a[k]} vs ${mean[i]}`);
  }
});

test("round trip keeps the picture (PSNR sanity on a smooth gradient)", () => {
  const v = vectors.encode.find((e) => e.name === "duotone_192x128");
  const src = bytes(v.rgba);
  const out = decode(v.hashes["40"], v.width, v.height);
  let se = 0;
  for (let i = 0; i < src.length; i += 4) for (let c = 0; c < 3; c++) se += (src[i + c] - out[i + c]) ** 2;
  const psnr = 10 * Math.log10((255 * 255) / (se / (v.width * v.height * 3)));
  console.log(`  ${v.name} PSNR at 40 chars: ${psnr.toFixed(1)} dB`);
  assert.ok(psnr > 30);
});
