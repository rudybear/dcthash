// Decode / encode timings for DCTHash and the `blurhash` npm package on the same machine.
import { readFileSync, writeFileSync } from "node:fs";
import * as dct from "../dist/index.js";
import * as bh from "blurhash";

const vectors = JSON.parse(readFileSync(new URL("../../format/vectors.json", import.meta.url), "utf8"));
const v = vectors.encode.find((e) => e.name === "kodim23_half") ?? vectors.encode[0];
const px = new Uint8ClampedArray(Buffer.from(v.rgba, "base64"));
const time = (fn, minMs = 300) => {
  for (let i = 0; i < 20; i++) fn();
  let n = 0;
  const t0 = performance.now();
  while (performance.now() - t0 < minMs) { fn(); n++; }
  return (performance.now() - t0) / n;
};
const fmt = (ms) => (ms < 0.1 ? `${(ms * 1000).toFixed(0)} us` : `${ms.toFixed(2)} ms`);
const rows = [];
const grids = { 28: [4, 3], 52: [6, 4], 94: [9, 5] };
for (const len of [28, 52, 94]) {
  const h = dct.encode(px, v.width, v.height, { length: len });
  const [nx, ny] = grids[len];
  const b = bh.encode(px, v.width, v.height, nx, ny);
  rows.push([`${len} chars`, "DCTHash", fmt(time(() => dct.encode(px, v.width, v.height, { length: len }))),
    fmt(time(() => dct.decode(h, 32, 21))), fmt(time(() => dct.decode(h, 384, 256))), fmt(time(() => dct.averageColor(h)))]);
  rows.push([`${b.length} chars (${nx}x${ny})`, "blurhash 2.0.5", fmt(time(() => bh.encode(px, v.width, v.height, nx, ny))),
    fmt(time(() => bh.decode(b, 32, 21))), fmt(time(() => bh.decode(b, 384, 256))), "-"]);
}
const md = [
  `Node ${process.version}, ${v.width}x${v.height} source image, single thread, warm.`,
  "",
  "| string | library | encode 384x256 | decode 32x21 | decode 384x256 | averageColor |",
  "|---|---|---|---|---|---|",
  ...rows.map((r) => `| ${r.join(" | ")} |`),
].join("\n");
console.log(md);
writeFileSync(new URL("../../results/ts_bench.md", import.meta.url), md + "\n");
