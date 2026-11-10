// Minified bundles + size report, and the demo bundle.  Run after tsc (see `npm run build`).
import { build } from "esbuild";
import { gzipSync } from "node:zlib";
import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const common = { bundle: true, minify: true, target: "es2020", logLevel: "warning" };
const targets = [
  ["dist/index.js", "dist/dcthash.min.js", "full library (encode + decode + helpers)"],
  ["dist/decode.js", "dist/dcthash-decode.min.js", "decoder + helpers only"],
  ["dist/encode.js", "dist/dcthash-encode.min.js", "encoder only"],
];
const sizes = {};
for (const [entry, out, label] of targets) {
  await build({ ...common, entryPoints: [join(root, entry)], outfile: join(root, out), format: "esm" });
  const buf = readFileSync(join(root, out));
  sizes[out] = { label, bytes: buf.length, gzip: gzipSync(buf, { level: 9 }).length };
  console.log(`${out}: ${buf.length} bytes minified, ${sizes[out].gzip} bytes gzip  (${label})`);
}
writeFileSync(join(root, "dist", "sizes.json"), JSON.stringify(sizes, null, 2));
// demo: one classic script so that demo/index.html works from file://
await build({ ...common, entryPoints: [join(root, "demo", "demo.ts")], outfile: join(root, "demo", "demo.js"), format: "iife" });
console.log("demo/demo.js built - open ts/demo/index.html in a browser");
