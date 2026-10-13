# obb_loss

Pure PyTorch oriented-box and quality-classification losses with a declarative
set-prediction criterion. Python 3.11+; development defaults to Python 3.12.

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

The primitive loss functions are `stable_probiou_loss`, `stable_probiou`, `box_l1_loss`,
`varifocal_loss`, and `quality_focal_loss`. Each returns `Ok(tensor)` or a typed
`Err(InputError)` for incompatible shapes, non-floating dtypes, or device mismatch.
Validation reads metadata only. Numerical input values and scalar options are the
caller's responsibility. There are no separate checked APIs.

Boxes use `(cx, cy, width, height, cos(theta), sin(theta))`. ProbIoU supports
broadcastable leading dimensions. L1 accepts aligned `[pairs, 6]` tensors and
sums absolute errors over the first four coordinates. Classification losses
accept aligned `[items, classes]` logits and caller-supplied quality targets.
Quality must be in `[0, 1]`; zero denotes a negative class.

Primitive losses return unreduced float64 tensors, disable autocast locally,
preserve autograd, and borrow inputs without mutation. Primitive reduction and
weighting belong to the caller. The optional criterion composes them and owns
matching; detector orchestration stays outside the library. Empty primitive inputs
produce empty outputs; choose an explicit policy before applying a mean.

ProbIoU requires finite boxes and nonnegative sides. Nonzero direction vectors
are normalized internally, so `(1, 1)` and `(0.5, 0.5)` represent the same angle.
If either box has a zero side or a `(0, 0)` direction, its pair receives loss
`1` and similarity `0`, with zero ProbIoU gradients for both boxes. Safe geometry
is substituted before computation to avoid NaN derivatives. An additional L1
loss can still supply size gradients, unless the decoding derivative is zero.

For nondegenerate pairs, relative to the largest side, sides must be at least
`1e-30` and center offsets at most `1e30`. The implementation uses positive
covariance residuals, common scale normalization, `log1p`, and `expm1` to avoid
cancellation. Translation
residuals saturate for distant pairs. Coincident boxes use a zero subgradient.
Float64 internal computation cannot prevent overflow when gradients are cast
back into low-precision leaf tensors. Use FP32 box-head projection and decoding
for small boxes: FP16 cannot represent some correct gradients even when the
loss is finite. These numerical choices are explained next to the implementation.

## Declarative criterion

One call performs matching and loss evaluation. Declare independent matching
costs, a solver, and loss terms with explicit weights and normalization:

```python
import torch
import obb_loss as obb
from obb_loss.losses.set_terms import (
    class_probability_cost,
    matched_box_inputs,
    one_hot_classification_inputs,
    pairwise_box_l1_cost,
    pairwise_probiou_cost,
    target_count,
)

criterion = obb.Criterion(
    matcher=obb.Matcher(
        solver=obb.hungarian_assignment,
        costs=(
            obb.Cost("classes", class_probability_cost, weight=1.0),
            obb.Cost("l1", pairwise_box_l1_cost, weight=5.0),
            obb.Cost("probiou", pairwise_probiou_cost, weight=2.0),
        ),
    ),
    losses=(
        obb.Loss("classes", obb.quality_focal_loss,
                 one_hot_classification_inputs, weight=1.0, normalize=target_count),
        obb.Loss("l1", obb.box_l1_loss,
                 matched_box_inputs, weight=5.0, normalize=target_count),
        obb.Loss("probiou", obb.stable_probiou_loss,
                 matched_box_inputs, weight=2.0, normalize=target_count),
    ),
)
predictions = obb.Predictions(
    boxes=torch.tensor([[[0., 0., 2., 1., 1., 0.],
                         [5., 0., 2., 1., 1., 0.]]], requires_grad=True),
    logits=torch.zeros(1, 2, 2, requires_grad=True),
)
targets = (obb.Targets(
    boxes=torch.tensor([[0.1, 0., 2., 1., 1., 0.]]),
    labels=torch.tensor([1]),
),)
match obb.criterion_loss(criterion, predictions, targets):
    case obb.Ok(output):
        output.total.backward()
        print({name: value.detach().item() for name, value in output.raw.items()})
    case obb.Err(error):
        print(error)
```

Install the optional assignment backend with `uv pip install 'obb-loss[matching]'`.
For this checkout, `uv sync --extra matching` installs it; `just check` enables it
for tests automatically. Base imports and the tensor core do not require SciPy.
The backend transfers detached costs to CPU, which synchronizes CUDA, and returns
int64 indices on the prediction device. Inject another solver to change that
boundary. Empty assignment problems bypass the backend entirely.

The class matching cost uses negative sigmoid probability. The one-hot projection
sets matched classes to 1 and every other entry to 0, so unmatched queries remain
classification negatives. Match indices and labels must be unique/in-range as
appropriate; only tensor metadata is validated. Returned matches expose unmatched
targets when there are more objects than queries.

Each loss contributes `weight * values.sum() / normalize(context)`.
`target_count` divides by `max(total objects, 1)`; `query_count` and `identity`
are available alternatives. Matching weights are separate from loss weights.
`output.raw` contains normalized unweighted terms; `output.weighted` contains
scaled terms. Zero weights skip projections, functions and normalizers and omit
the term from outputs. All-disabled losses also skip matching and return a zero
connected to predictions. Numerical domains, including finite positive custom
divisors, remain the declared functions' responsibility.

## Extending the objective

Add ordinary functions through declarations. For example, this term penalizes
squared center error without changing the criterion:

```python
from dataclasses import replace
from torch import Tensor
from obb_loss.losses.set_terms import matched_box_inputs, target_count


def squared_center_loss(preds: Tensor, targets: Tensor) -> obb.Result[Tensor, obb.InputError]:
    """Compute squared center distance for core-prepared aligned box pairs."""
    with torch.autocast(preds.device.type, enabled=False):
        return obb.Ok((preds[:, :2].double() - targets[:, :2].double()).square().sum(-1))


criterion = replace(criterion, losses=(*criterion.losses, obb.Loss(
    "centers", squared_center_loss, matched_box_inputs,
    weight=0.5, normalize=target_count,
)))
```

A loss's `inputs(context)` returns `Ok(tuple_of_arguments)` or a metadata error;
`compute(*arguments)` returns unreduced values. Their argument types stay paired
under strict checking. Custom projections can access original predictions,
per-image targets and assignments as well as gathered box pairs. Use
`functools.partial` to bind numerical options or `supplied_quality_inputs` to bind
caller-supplied quality tensors; quality gradients remain connected unless the
caller explicitly detaches them.

The functional core lives in `criterion_core.py`. It constructs costs, prepares
matched context and evaluates losses without host transfers or backend imports.
`criterion_loss` is the thin imperative shell that invokes the declared solver
under `no_grad` and then evaluates losses on original differentiable inputs.
Expected failures return `Err(InputError | CriterionError)`; user-function bugs
propagate.

For auxiliary decoder outputs, repeat the criterion call with the same declaration
and explicit layer weights. To reuse assignments, call `prepare_context` and
`evaluate_losses` from the core with returned matches. Supply distributed divisors
through a custom normalizer; no distributed communication runs inside the library.
The supplied tensor-core preparation/evaluation is tested with `torch.compile`
using AOT eager; custom functions determine their own compilation support. The
CPU assignment backend is outside that graph boundary.

References:

- [Gaussian Bounding Boxes and Probabilistic Intersection-over-Union](https://arxiv.org/abs/2106.06072): Gaussian-box covariance, Bhattacharyya distance, and Hellinger similarity.
- [VarifocalNet](https://arxiv.org/abs/2008.13367): Varifocal loss, equation 2.
- [Generalized Focal Loss](https://arxiv.org/abs/2006.04388): Quality Focal loss, equation 2.
- [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872): bipartite set matching and box L1 loss.

Run `uv sync` and `just check` for strict Ruff, native Pyrefly shape checking,
and pytest. Configurations are separate: `ruff.toml`, `pyrefly.toml`, and
`pytest.ini`. The type-check runner provisions a locked Python 3.12 shape-checking
environment when the runtime is Python 3.11.

Tests cover metadata contracts, independent paper-equation oracles, gradients,
compilation, mixed precision, and seeded numerical stress simulations. CPU tests
always run; CUDA cases skip when CUDA is unavailable. Simulations provide evidence
for their tested domains, rather than a proof for every floating-point input.
