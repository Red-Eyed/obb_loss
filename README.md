# obb_loss

Typed PyTorch oriented-box losses for DETR-like detectors. Implementation is in progress.
The primary geometry is `(cx, cy, width, height, cos(theta), sin(theta))`.
Packed records preserve per-image boundaries, including empty images; integration helpers
will accept ordinary per-image tensors and dense DETR predictions.

Use `uv sync` to install the development environment and `just check` to run strict
formatting, lint, type, and test checks. Each check recipe synchronizes its environment.
