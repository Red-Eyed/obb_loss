# obb_loss

Pure PyTorch oriented-box and quality-classification losses. Python 3.11+;
development defaults to Python 3.12.

```python
import torch
import obb_loss as obb

preds = torch.tensor([[0., 0., 2., 1., 1., 0.]], requires_grad=True)
targets = torch.tensor([[0.1, 0., 2., 1., 1., 0.]])
match obb.stable_probiou_loss(preds, targets):
    case obb.Ok(values):
        values.mean().backward()
    case obb.Err(error):
        print(error)
```

The public functions are `stable_probiou_loss`, `stable_probiou`, `box_l1_loss`,
`varifocal_loss`, and `quality_focal_loss`. Each returns `Ok(tensor)` or a typed
`Err(InputError)` for incompatible shapes, non-floating dtypes, or device mismatch.
Validation reads metadata only. Numerical input values and scalar options are the
caller's responsibility. There are no separate checked APIs.

Boxes use `(cx, cy, width, height, cos(theta), sin(theta))`. ProbIoU supports
broadcastable leading dimensions. L1 accepts aligned `[pairs, 6]` tensors and
sums absolute errors over the first four coordinates. Classification losses
accept aligned `[items, classes]` logits and caller-supplied quality targets.
Quality must be in `[0, 1]`; zero denotes a negative class.

All losses return unreduced float64 tensors, disable autocast locally, preserve
autograd, and borrow inputs without mutation. Reduction, weighting, matching,
and detector orchestration belong to the caller. Empty inputs produce empty
outputs; choose an explicit policy before applying a mean.

ProbIoU requires finite boxes, positive sides and nonzero direction vectors.
Relative to the largest side in a pair, sides must be at least `1e-30` and center
offsets at most `1e30`. The implementation uses positive covariance residuals,
common scale normalization, `log1p`, and `expm1` to avoid cancellation. Translation
residuals saturate for distant pairs. Coincident boxes use a zero subgradient.
Float64 internal computation cannot prevent overflow when gradients are cast
back into low-precision leaf tensors. These numerical choices are explained
next to the implementation.

References:

- [Gaussian Bounding Boxes and Probabilistic Intersection-over-Union](https://arxiv.org/abs/2106.06072): Gaussian-box covariance, Bhattacharyya distance, and Hellinger similarity.
- [VarifocalNet](https://arxiv.org/abs/2008.13367): Varifocal loss, equation 2.
- [Generalized Focal Loss](https://arxiv.org/abs/2006.04388): Quality Focal loss, equation 2.
- [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872): box L1 loss.

Run `uv sync` and `just check` for strict Ruff, native Pyrefly shape checking,
and pytest. Configurations are separate: `ruff.toml`, `pyrefly.toml`, and
`pytest.ini`. The type-check runner provisions a locked Python 3.12 shape-checking
environment when the runtime is Python 3.11.

Tests cover metadata contracts, independent paper-equation oracles, gradients,
compilation, mixed precision, and seeded numerical stress simulations. CPU tests
always run; CUDA cases skip when CUDA is unavailable. Simulations provide evidence
for their tested domains, rather than a proof for every floating-point input.
