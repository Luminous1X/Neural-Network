/* FableNet inference in plain JavaScript.
 *
 * A straight port of the forward pass in fable/models.py, so the network
 * trained in NumPy can run in a browser (or in Node, for the parity test).
 * Tensors are flat Float32Arrays in the same C order NumPy uses.
 */
(function (root) {
  "use strict";

  function decodeFloat32(base64) {
    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return new Float32Array(bytes.buffer);
  }

  function decodeBytes(base64) {
    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return bytes;
  }

  /* Valid cross-correlation, stride 1, followed by ReLU.
   * input: C x H x W, weight: F x C x K x K, returns F x (H-K+1) x (W-K+1). */
  function convRelu(input, C, H, W, weight, bias, F, K) {
    const OH = H - K + 1, OW = W - K + 1;
    const out = new Float32Array(F * OH * OW);
    for (let f = 0; f < F; f++) {
      for (let y = 0; y < OH; y++) {
        for (let x = 0; x < OW; x++) {
          let sum = bias[f];
          for (let c = 0; c < C; c++) {
            const wBase = ((f * C + c) * K) * K;
            const iBase = c * H * W + y * W + x;
            for (let ky = 0; ky < K; ky++) {
              const wRow = wBase + ky * K, iRow = iBase + ky * W;
              for (let kx = 0; kx < K; kx++) sum += weight[wRow + kx] * input[iRow + kx];
            }
          }
          out[(f * OH + y) * OW + x] = sum > 0 ? sum : 0;
        }
      }
    }
    return out;
  }

  /* 2x2 max pooling. input: C x H x W, returns C x H/2 x W/2. */
  function maxPool2(input, C, H, W) {
    const OH = H >> 1, OW = W >> 1;
    const out = new Float32Array(C * OH * OW);
    for (let c = 0; c < C; c++) {
      for (let y = 0; y < OH; y++) {
        for (let x = 0; x < OW; x++) {
          const i = c * H * W + 2 * y * W + 2 * x;
          out[(c * OH + y) * OW + x] = Math.max(input[i], input[i + 1], input[i + W], input[i + W + 1]);
        }
      }
    }
    return out;
  }

  /* y = W x + b. weight: OUT x IN. */
  function linear(input, weight, bias, OUT, IN, relu) {
    const out = new Float32Array(OUT);
    for (let o = 0; o < OUT; o++) {
      let sum = bias[o];
      const base = o * IN;
      for (let i = 0; i < IN; i++) sum += weight[base + i] * input[i];
      out[o] = relu && sum < 0 ? 0 : sum;
    }
    return out;
  }

  function softmax(logits) {
    let max = -Infinity;
    for (const v of logits) if (v > max) max = v;
    const out = new Float32Array(logits.length);
    let total = 0;
    for (let i = 0; i < logits.length; i++) total += out[i] = Math.exp(logits[i] - max);
    for (let i = 0; i < out.length; i++) out[i] /= total;
    return out;
  }

  function load(weights) {
    const tensors = {};
    for (const name of Object.keys(weights.tensors)) tensors[name] = decodeFloat32(weights.tensors[name].data);
    const meta = weights.meta;
    const c1 = meta.config.c1, c2 = meta.config.c2, hidden = meta.config.hidden;

    /* pixels: 784 values in [0, 1], row-major, ink = 1.
     * Returns the logits, the class probabilities and the activations in between. */
    function forward(pixels) {
      const input = new Float32Array(784);
      for (let i = 0; i < 784; i++) input[i] = (pixels[i] - meta.mean) / meta.std;

      const conv1 = convRelu(input, 1, 28, 28, tensors["conv1.weight"], tensors["conv1.bias"], c1, 5);
      const pool1 = maxPool2(conv1, c1, 24, 24);
      const conv2 = convRelu(pool1, c1, 12, 12, tensors["conv2.weight"], tensors["conv2.bias"], c2, 5);
      const pool2 = maxPool2(conv2, c2, 8, 8);
      const fc1 = linear(pool2, tensors["fc1.weight"], tensors["fc1.bias"], hidden, c2 * 16, true);
      const logits = linear(fc1, tensors["fc2.weight"], tensors["fc2.bias"], 10, hidden, false);
      const probs = softmax(logits);

      let digit = 0;
      for (let i = 1; i < 10; i++) if (probs[i] > probs[digit]) digit = i;
      return { digit, probs, logits, conv1, conv2 };
    }

    return { forward, meta, c1, c2 };
  }

  /* Turn a free-hand drawing into the layout MNIST digits have: the ink's
   * bounding box scaled to fit 20x20, then placed in a 28x28 frame so its
   * centre of mass sits in the middle.
   *
   * ink: width*height values in [0, 1]. Returns 784 values, or null if blank. */
  function preprocess(ink, width, height) {
    const threshold = 0.1;
    let x0 = width, y0 = height, x1 = -1, y1 = -1;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        if (ink[y * width + x] > threshold) {
          if (x < x0) x0 = x;
          if (x > x1) x1 = x;
          if (y < y0) y0 = y;
          if (y > y1) y1 = y;
        }
      }
    }
    if (x1 < 0) return null;

    const boxW = x1 - x0 + 1, boxH = y1 - y0 + 1;
    const scale = 20 / Math.max(boxW, boxH);
    const smallW = Math.max(1, Math.round(boxW * scale));
    const smallH = Math.max(1, Math.round(boxH * scale));

    // Area-average the bounding box down to smallW x smallH.
    const small = new Float32Array(smallW * smallH);
    for (let y = 0; y < smallH; y++) {
      const sy0 = y0 + Math.floor((y * boxH) / smallH);
      const sy1 = Math.max(sy0 + 1, y0 + Math.floor(((y + 1) * boxH) / smallH));
      for (let x = 0; x < smallW; x++) {
        const sx0 = x0 + Math.floor((x * boxW) / smallW);
        const sx1 = Math.max(sx0 + 1, x0 + Math.floor(((x + 1) * boxW) / smallW));
        let sum = 0;
        for (let sy = sy0; sy < sy1; sy++) for (let sx = sx0; sx < sx1; sx++) sum += ink[sy * width + sx];
        small[y * smallW + x] = sum / ((sy1 - sy0) * (sx1 - sx0));
      }
    }

    let mass = 0, mx = 0, my = 0;
    for (let y = 0; y < smallH; y++) {
      for (let x = 0; x < smallW; x++) {
        const v = small[y * smallW + x];
        mass += v;
        mx += v * (x + 0.5);
        my += v * (y + 0.5);
      }
    }
    const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
    const offX = clamp(Math.round(14 - mx / mass), 0, 28 - smallW);
    const offY = clamp(Math.round(14 - my / mass), 0, 28 - smallH);

    const out = new Float32Array(784);
    for (let y = 0; y < smallH; y++) {
      for (let x = 0; x < smallW; x++) out[(y + offY) * 28 + x + offX] = small[y * smallW + x];
    }
    return out;
  }

  const api = { load, preprocess, decodeBytes, decodeFloat32 };
  root.FableModel = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
