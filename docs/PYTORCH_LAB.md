# PyTorch & CNN Lab

## Availability

PyTorch is an optional engine. `GET /api/system/capabilities` reports installation, version, CUDA availability and device count. Requesting CUDA without a CUDA runtime falls back to CPU with a visible warning rather than pretending GPU execution.

The current verified environment used PyTorch `2.14.1+cu130`; CUDA was not available, so tests ran on CPU.

## Tabular MLP

The MLP lab runs genuine PyTorch operations:

- configurable one-to-four hidden layers;
- ReLU, tanh, sigmoid or GELU;
- Adam, AdamW, SGD or momentum;
- learning rate, batch size, epochs, dropout, weight decay and seed;
- training/validation loss and accuracy for every epoch;
- parameter count, runtime device and duration;
- best validation epoch and warnings.

The existing NumPy neural lab remains separate for inspecting forward and backward propagation.

## CNN lab

The CNN trains on scikit-learn’s bundled 8×8 digits:

```text
Conv2d(1→8) → ReLU → MaxPool
→ Conv2d(8→16) → ReLU → MaxPool
→ Linear(64→10)
```

It returns real learning curves, confusion matrix, correct and misclassified examples, confidence values and first-convolution feature maps. Feature maps are labelled as activations, not a complete explanation of model reasoning.

Training-only augmentation supports no augmentation, bounded pixel shifts, Gaussian noise, or shift plus noise. The response includes original/transformed preview images and confirms validation images are unchanged.

## Checkpoints

Checkpoints are created only by local training. The `.pt` file contains `state_dict` plus controlled metadata; a JSON sidecar records size/hash/config. The API does **not** load arbitrary uploaded pickle/checkpoint files. This avoids the code-execution risk of untrusted PyTorch serialization.

## Determinism

Python, NumPy and PyTorch seeds are set. Data-loader shuffling uses a seeded generator. Hardware/library kernels can still produce small platform-dependent differences; the UI and docs therefore promise reproducibility metadata, not bit-for-bit equality on every device.

## Limits

This is a bounded educational/prototyping engine, not distributed training. Epochs, architecture width, batches and data size are capped. Training requests are synchronous and intended for a local single-user process.
