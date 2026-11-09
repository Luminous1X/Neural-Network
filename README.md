# Fable's Neural Network

A convolutional neural network built from scratch. The only dependency is NumPy,
which supplies arrays and matrix multiplication. Everything that makes it a
neural network is written here: the automatic differentiation engine, the
convolution and pooling layers and their gradients, the Adam optimiser, the
data pipeline, and the training loop.

Trained on MNIST handwritten digits, it reaches **99.55% test accuracy**
(45 mistakes in 10,000 images) in about five minutes on a CPU.

## Try it

Open `demo/index.html` in a browser and draw a digit. The page runs the trained
network in plain JavaScript and shows the prediction along with what each
convolution layer responds to. No server or install is needed: the trained
weights are included in the repository.

## Run it

Needs Python and NumPy (developed with Python 3.11 and NumPy 2.4). The parity
test also needs Node.

```
pip install -r requirements.txt
python train.py              # downloads MNIST (11 MB), trains, saves checkpoints/fablenet.npz
python evaluate.py           # accuracy, confusion matrix, the most confident mistakes
python export_web.py         # writes the weights for the browser demo
python -m unittest discover -s tests     # 63 tests
node tests/js_parity.js      # checks the JavaScript port against NumPy
```

`python train.py --limit 3000 --epochs 2` is a ten-second smoke test.
`python train.py --help` lists the hyperparameters.

## The network

```
input             1 x 28 x 28
conv 5x5, ReLU   16 x 24 x 24
max-pool 2x2     16 x 12 x 12
conv 5x5, ReLU   32 x  8 x  8
max-pool 2x2     32 x  4 x  4
flatten          512
linear, ReLU     128
dropout 0.3
linear           10
```

80,202 parameters. Trained for 12 epochs with AdamW (peak learning rate 2e-3,
linear warm-up then cosine decay, weight decay 0.01), batch size 64, and random
affine augmentation: each training image is rotated up to 10 degrees, scaled by
up to 10% and shifted up to 2 pixels.

## Results

| epoch | train loss | train accuracy | test loss | test accuracy |
|------:|-----------:|---------------:|----------:|--------------:|
| 1     | 0.6458     | 82.06%         | 0.0637    | 98.03%        |
| 2     | 0.1299     | 96.05%         | 0.0391    | 98.69%        |
| 4     | 0.0757     | 97.77%         | 0.0256    | 99.11%        |
| 6     | 0.0556     | 98.32%         | 0.0232    | 99.32%        |
| 9     | 0.0328     | 98.94%         | 0.0166    | 99.53%        |
| 12    | 0.0260     | 99.19%         | 0.0159    | 99.55%        |

Each epoch takes about 27 seconds on a 12-core desktop CPU. The reported number
is the final epoch, not the best one, so the test set played no part in
choosing the model. Training accuracy is lower than test accuracy because it is
measured on augmented images with dropout active. The full log is in
`checkpoints/fablenet_history.json`.

The hardest digit is 9 (99.11%), mostly confused with 4 and 8.

## How it works

**Autograd (`fable/tensor.py`).** A `Tensor` wraps an array. Every operation
returns a new tensor that remembers its inputs and a function mapping the
gradient of its output to gradients of those inputs. `backward()` sorts the
graph topologically and applies those functions in reverse, which is the chain
rule. Broadcasting is handled by summing gradients back down to each operand's
shape.

**Convolution (`fable/functional.py`).** Each 5x5 window of the input is
unrolled into one row of a matrix (im2col), which turns the convolution into a
single matrix product. The backward pass runs the same idea in reverse and
scatters window gradients back onto the input.

**Everything else.** `fable/nn.py` has the layers, `fable/optim.py` has SGD,
Adam and the learning-rate schedule, `fable/data.py` downloads and parses MNIST
(verifying checksums) and implements the augmentation with bilinear resampling.

## How it is checked

- Every operation's gradient is compared with finite differences in float64
  (`fable/gradcheck.py`), including one check through the whole network from
  the loss back to every weight.
- The fast convolution is compared with a plain-loop implementation of the
  definition, across strides and paddings.
- The JavaScript port produces the same logits as NumPy to within 1e-5 on 300
  test images, and its draw-to-28x28 preprocessing is tested by enlarging and
  displacing test digits and checking they are still recognised.

## Layout

```
fable/              the framework
  tensor.py         autograd engine
  functional.py     conv2d, max_pool2d, log_softmax, cross_entropy, dropout
  nn.py             Module, Linear, Conv2d, Dropout, Sequential, ...
  optim.py          SGD, Adam, WarmupCosine
  data.py           MNIST loading, batching, augmentation
  models.py         FableNet and checkpoint save/load
  gradcheck.py      finite-difference gradient checking
train.py            training script
evaluate.py         test-set report
export_web.py       exports weights for the demo
demo/               browser demo (index.html, model.js, weights.js)
tests/              unit tests and the JavaScript parity test
checkpoints/        the trained model and its training history
data/               MNIST, downloaded on first run (not in the repository)
```

## Using the framework for something else

```python
import numpy as np
import fable
from fable import Tensor, nn, optim
from fable import functional as F

x = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32))
y = np.array([0, 1, 1, 0])  # XOR

model = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 2))
optimizer = optim.Adam(model.parameters(), lr=0.05)
for _ in range(300):
    loss = F.cross_entropy(model(x), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(model(x).data.argmax(axis=1))  # [0 1 1 0]
```
