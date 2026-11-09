/* Checks that the JavaScript port in demo/model.js agrees with NumPy.
 *
 *   python export_web.py      (writes the weights and the fixture)
 *   node tests/js_parity.js
 */
"use strict";
const fs = require("fs");
const path = require("path");

require(path.join(__dirname, "..", "demo", "weights.js"));
const FableModel = require(path.join(__dirname, "..", "demo", "model.js"));

const fixture = JSON.parse(fs.readFileSync(path.join(__dirname, "js_fixture.json"), "utf8"));
const model = FableModel.load(globalThis.FABLE_WEIGHTS);
const pixels = FableModel.decodeBytes(fixture.pixels);
const expected = FableModel.decodeFloat32(fixture.logits);

let failures = 0;
function check(name, ok, detail) {
  console.log(`${ok ? "ok  " : "FAIL"}  ${name}: ${detail}`);
  if (!ok) failures++;
}

// 1. Same logits as NumPy for every fixture image.
let worst = 0, correct = 0, agree = 0;
for (let n = 0; n < fixture.count; n++) {
  const image = new Float32Array(784);
  for (let i = 0; i < 784; i++) image[i] = pixels[n * 784 + i] / 255;
  const { logits, digit } = model.forward(image);
  let numpyDigit = 0;
  for (let k = 0; k < 10; k++) {
    worst = Math.max(worst, Math.abs(logits[k] - expected[n * 10 + k]));
    if (expected[n * 10 + k] > expected[n * 10 + numpyDigit]) numpyDigit = k;
  }
  if (digit === numpyDigit) agree++;
  if (digit === fixture.labels[n]) correct++;
}
check("logits match NumPy", worst < 1e-3, `largest difference ${worst.toExponential(2)} over ${fixture.count} images`);
check("predictions match NumPy", agree === fixture.count, `${agree}/${fixture.count} identical`);
console.log(`      accuracy on fixture images: ${((100 * correct) / fixture.count).toFixed(2)}%`);

// 2. The drawing pipeline: blow each digit up to a 280x280 "canvas", place it
//    off-centre, and check preprocess() brings it back to something the
//    network still reads correctly.
let recovered = 0;
const SIZE = 280;
for (let n = 0; n < fixture.count; n++) {
  const canvas = new Float32Array(SIZE * SIZE);
  const zoom = 5 + (n % 4);                      // 140 to 224 px tall
  const span = 28 * zoom;
  const left = (n * 37) % (SIZE - span + 1);
  const top = (n * 53) % (SIZE - span + 1);
  for (let y = 0; y < span; y++) {
    for (let x = 0; x < span; x++) {
      canvas[(top + y) * SIZE + left + x] = pixels[n * 784 + Math.floor(y / zoom) * 28 + Math.floor(x / zoom)] / 255;
    }
  }
  const input = FableModel.preprocess(canvas, SIZE, SIZE);
  if (input && model.forward(input).digit === fixture.labels[n]) recovered++;
}
const rate = recovered / fixture.count;
check("drawing pipeline", rate >= 0.97, `${(100 * rate).toFixed(2)}% of rescaled, shifted digits recognised`);
check("blank canvas", FableModel.preprocess(new Float32Array(SIZE * SIZE), SIZE, SIZE) === null, "returns null");

process.exit(failures ? 1 : 0);
