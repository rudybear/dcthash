// Demo: original vs BlurHash vs DCTHash at the same string length.  Bundled to demo.js (IIFE).
import { encode as bhEncode, decode as bhDecode } from "blurhash";
import { encode, decode, averageColor, aspectRatio } from "../src/index";

const GRIDS: [number, number][] = [[3, 3], [4, 3], [5, 3], [4, 4], [5, 4], [6, 4], [6, 5], [7, 5], [8, 5], [9, 5]];
const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const gridSel = $<HTMLSelectElement>("grid");
const drop = $("drop");
let current: ImageData | undefined;

for (const [nx, ny] of GRIDS) {
  const o = document.createElement("option");
  o.value = `${nx},${ny}`;
  o.textContent = `${nx} x ${ny}  (${4 + 2 * nx * ny} chars)`;
  if (nx === 4 && ny === 3) o.selected = true;
  gridSel.append(o);
}

function psnr(a: Uint8ClampedArray, b: Uint8ClampedArray): number {
  let se = 0;
  for (let i = 0; i < a.length; i += 4) for (let c = 0; c < 3; c++) se += (a[i + c] - b[i + c]) ** 2;
  return 10 * Math.log10((255 * 255) / Math.max(se / ((a.length / 4) * 3), 1e-9));
}

function draw(id: string, pixels: Uint8ClampedArray, w: number, h: number): void {
  const c = $<HTMLCanvasElement>(id);
  c.width = w;
  c.height = h;
  c.getContext("2d")!.putImageData(new ImageData(new Uint8ClampedArray(pixels), w, h), 0, 0);
}

function update(): void {
  if (!current) return;
  const { data, width: w, height: h } = current;
  $("error").textContent = "";
  try {
    let [nx, ny] = gridSel.value.split(",").map(Number);
    if (h > w) [nx, ny] = [ny, nx];
    const length = 4 + 2 * nx * ny;

    let t = performance.now();
    const b = bhEncode(data, w, h, nx, ny);
    const tbEnc = performance.now() - t;
    t = performance.now();
    const bPix = bhDecode(b, w, h);
    const tbDec = performance.now() - t;

    t = performance.now();
    const d = encode(data, w, h, { length });
    const tdEnc = performance.now() - t;
    t = performance.now();
    const dPix = decode(d, w, h);
    const tdDec = performance.now() - t;

    draw("c-orig", data, w, h);
    draw("c-blur", bPix, w, h);
    draw("c-dct", dPix, w, h);
    $("s-blur").textContent = b;
    $("s-dct").textContent = d;
    $("m-blur").textContent = `${b.length} chars, ${psnr(data, bPix).toFixed(2)} dB`;
    $("m-dct").textContent = `${d.length} chars, ${psnr(data, dPix).toFixed(2)} dB`;
    $("m-orig").textContent = `${w} x ${h}`;
    $("t-blur").textContent = `encode ${tbEnc.toFixed(1)} ms, decode ${tbDec.toFixed(1)} ms (${nx} x ${ny} components)`;
    const avg = averageColor(d);
    const ar = aspectRatio(d);
    $("t-dct").textContent = `encode ${tdEnc.toFixed(1)} ms, decode ${tdDec.toFixed(1)} ms; from the string alone: ` +
      `average rgb(${avg.r}, ${avg.g}, ${avg.b}), aspect ${ar === null ? "unspecified" : ar.toFixed(3)}`;
    $("size").textContent = "";
  } catch (e) {
    $("error").textContent = `Could not hash this image: ${(e as Error).message}`;
  }
}

function useCanvasSource(src: CanvasImageSource, sw: number, sh: number): void {
  const s = Math.min(1, 512 / Math.max(sw, sh));
  const w = Math.max(1, Math.round(sw * s));
  const h = Math.max(1, Math.round(sh * s));
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const g = c.getContext("2d", { willReadFrequently: true })!;
  g.fillStyle = "#fff";
  g.fillRect(0, 0, w, h);
  g.drawImage(src, 0, 0, w, h);
  current = g.getImageData(0, 0, w, h);
  update();
}

function loadFile(file: File | undefined | null): void {
  if (!file) return;
  if (!file.type.startsWith("image/")) {
    $("error").textContent = "That file is not an image.";
    return;
  }
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => {
    useCanvasSource(img, img.naturalWidth, img.naturalHeight);
    URL.revokeObjectURL(url);
  };
  img.onerror = () => {
    $("error").textContent = "The browser could not decode that image.";
    URL.revokeObjectURL(url);
  };
  img.src = url;
}

function sample(): void {
  const c = document.createElement("canvas");
  c.width = 480;
  c.height = 320;
  const g = c.getContext("2d")!;
  const sky = g.createLinearGradient(0, 0, 0, 220);
  sky.addColorStop(0, "#2b5fa8");
  sky.addColorStop(0.7, "#f3b27a");
  sky.addColorStop(1, "#f7dca8");
  g.fillStyle = sky;
  g.fillRect(0, 0, 480, 320);
  g.fillStyle = "#fff3c4";
  g.beginPath();
  g.arc(340, 150, 34, 0, 7);
  g.fill();
  g.fillStyle = "#3d5a45";
  g.beginPath();
  g.moveTo(0, 230);
  g.bezierCurveTo(120, 150, 220, 250, 480, 190);
  g.lineTo(480, 320);
  g.lineTo(0, 320);
  g.fill();
  g.fillStyle = "#22352b";
  g.beginPath();
  g.moveTo(0, 290);
  g.bezierCurveTo(160, 220, 300, 300, 480, 250);
  g.lineTo(480, 320);
  g.lineTo(0, 320);
  g.fill();
  g.fillStyle = "#c8432f";
  g.fillRect(90, 215, 46, 34);
  g.fillStyle = "#7a2418";
  g.beginPath();
  g.moveTo(84, 216);
  g.lineTo(113, 192);
  g.lineTo(142, 216);
  g.fill();
  useCanvasSource(c, 480, 320);
}

$<HTMLInputElement>("file").addEventListener("change", (e) => loadFile((e.target as HTMLInputElement).files?.[0]));
gridSel.addEventListener("change", update);
$("sample").addEventListener("click", sample);
for (const ev of ["dragenter", "dragover"]) {
  document.addEventListener(ev, (e) => {
    e.preventDefault();
    drop.classList.add("over");
  });
}
for (const ev of ["dragleave", "drop"]) {
  document.addEventListener(ev, (e) => {
    e.preventDefault();
    drop.classList.remove("over");
  });
}
document.addEventListener("drop", (e) => loadFile((e as DragEvent).dataTransfer?.files?.[0]));
sample();
