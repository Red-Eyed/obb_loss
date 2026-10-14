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

To return one contribution per image, set `Criterion(..., reduction="none")`
and declare which image owns each first-axis loss row:

```python
from obb_loss.losses.set_terms import matched_sample_indices, query_sample_indices

losses = (
    obb.Loss("l1", obb.box_l1_loss, matched_box_inputs, 5.0, target_count,
             sample_indices=matched_sample_indices),
    obb.Loss("class", obb.quality_focal_loss, one_hot_classification_inputs,
             1.0, target_count, sample_indices=query_sample_indices),
)
criterion = obb.Criterion(matcher=criterion.matcher, losses=losses, reduction="none")
```

`output.total` and every named term then have shape `[batch]`. Trailing value
axes (for example, classes) are summed within each row. Custom `sample_indices`
functions return `Ok(int64_indices)` shaped `[rows]` on the prediction device,
or a typed `Err`; indices must be in `[0, batch)`. Only metadata is checked.
Missing ownership returns a declaration error for active terms in `"none"` mode.

The default `"sum"` preserves the scalar objective. `"mean"` divides that scalar
by batch size. All modes retain the declared **batch** normalizer, so summing
`"none"` contributions reproduces `"sum"`; images are not independently
normalized. Scalar modes skip sample-index functions. Empty batches return an
empty vector for `"none"` and a connected zero for scalar modes.

## Integrating a DETR decoder

[examples/detr_training.py](examples/detr_training.py) is a runnable integration
example with shared prediction heads, oriented box decoding, per-image targets,
an optimizer step, and three supervision policies:

```sh
uv run --extra matching examples/detr_training.py
```

It uses synthetic decoder states shaped `[layers, batch, queries, hidden]` and
prints gradients for each layer. Substitute your decoder's actual states at the
marked line. The example leaves your backbone and Transformer implementation
with your model; only the prediction-head and training-loss boundary changes.

The [original DETR model](https://github.com/facebookresearch/detr/blob/main/models/detr.py)
projects each decoder layer to class logits and boxes. For this library's built-in
sigmoid classification, adapt those heads as follows:

- Make the class head output `classes` foreground logits. Original DETR's extra
  no-object column and softmax classification belong to a different objective;
  replacing the head changes that classification objective. Do not simply pass
  its `classes + 1` logits to the one-hot quality-focal declaration.
- Make the oriented box head output five raw values: center/side parameters and
  an angle. `decode_boxes` applies sigmoid to the first four values and converts
  the angle to `(cos(theta), sin(theta))`. Keep projection and decoding in FP32,
  even when the backbone/decoder uses autocast.
- Return one `Predictions` record per decoder layer. The example's `predict_layers`
  shows that last step of model forward. Feed the final record to
  `criterion_loss` in your training loop and backpropagate `output.total`.
- Convert annotations into `Targets`, one record per image. Use contiguous int64
  foreground labels; retain an empty record for images without objects. Keep all
  tensors on the model's device. Input target boxes are `(cx, cy, width, height)`
  in pixels; angles are radians in the same coordinate convention as predictions.

For rotated boxes, use a common spatial scale: the example divides all four
spatial coordinates by `max(image_height, image_width)`. For a 640 by 480 image,
`(160, 120, 80, 40)` becomes `(0.25, 0.1875, 0.125, 0.0625)`; angle coordinates
are unchanged. Independently normalizing x and y changes the rotated geometry.
If your original DETR annotations are already normalized separately by width and
height, first recover pixels with `(width, height, width, height)`, then divide
by the common scale. Apply the convention consistently after augmentation.

For an axis-aligned detector, append a fixed direction `(1, 0)` to its four decoded
coordinates and targets, rather than adding an angle head. Convert existing
normalized coordinates to common units as above. ProbIoU replaces the geometric
similarity term; it does not reproduce original DETR's GIoU objective. Keeping
original softmax/background classification instead requires a declared custom
classification cost, target projection, and numerical loss.

The example's three modes make auxiliary policy explicit:

| Mode | Matching | Gradient path |
| --- | --- | --- |
| `final_only` | Final layer only | Final predictions and their decoder dependencies |
| `auxiliary_rematch` | Independently for each enabled layer | Each supervised layer |
| `auxiliary_shared` | Once on final predictions, then reuse indices | Each supervised layer |

The final layer has weight 1; the two auxiliary layers each have weight 0.5.
These are example choices, not package defaults. `auxiliary_loss` skips zero-weight
layers. Its shared-match branch calls the pure `prepare_context` and
`evaluate_losses` functions on each layer's own tensors, retaining gradients.
The original DETR criterion rematches auxiliary layers; shared matching is an
explicit alternative. Named final-layer terms are already weighted in
`output.total`; multiply by the layer weight once without applying term weights
again. Match indices are nondifferentiable, while box and classification losses
remain connected to the model.

In your real training loop, keep your optimizer, gradient scaler, distributed
normalization and error handling at the training boundary. The example's
`require_loss` aborts on a typed criterion error; choose your own handling policy
there. Auxiliary assignments reused from the final layer require the same query
ordering and target batch across those layers.

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
