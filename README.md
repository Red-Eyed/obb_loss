# obb_loss

Typed PyTorch oriented-box losses and standalone matching for DETR-like detectors.
The package exposes pure functions, Pydantic-configured `nn.Module` criteria, and a
typed callback composition. Detector-layer orchestration stays in your pipeline.

See [runnable examples](examples/README.md) for complete CPU forward/backward scripts.

The primary geometry is `(cx, cy, width, height, cos(theta), sin(theta))`, with positive
sides and nonzero direction vectors. Coordinate units are explicit and shared by predictions
and targets. Use `boxes_from_angles` to convert radians; no format inference occurs.

## Pack existing tensors

```python
import obb_loss as obb
import torch

# Existing per-sample target tensors, including an empty sample.
targets_result = obb.pack_boxes(
    [torch.tensor([[0., 0., 2., 1., 1., 0.]]), torch.empty(0, 6)],
    device=torch.device("cpu"),
    dtype=torch.float32,
)
# Dense detector output [batch, queries, 6] also has a helper.
predictions = torch.tensor([[[0.1, 0., 2., 1., 1., 0.]],
                            [[0.2, 0., 2., 1., 1., 0.]]], requires_grad=True)
preds_result = obb.pack_dense_boxes(predictions)
```

Helpers preserve gradients and sample boundaries, never mutate or silently transfer inputs,
and return `Ok(PackedBoxes)` or `Err(InputError)`. Device and dtype remain explicit even
for zero-sample input. `PackedBoxes.boxes` is `[total_boxes, 6]`; `cu_seqlens` is integer
`[batch + 1]`, starts at zero, ends at total_boxes, and retains empty samples through repeats.

## Match, then evaluate an aligned box loss

```python
import obb_loss as obb

match (preds_result, targets_result):
    case (obb.Ok(value=preds), obb.Ok(value=targets)):
        matcher = obb.HungarianMatcher(obb.MatcherConfig(chunk_size=4096, max_pairs=1_000_000))
        match matcher.forward(preds, targets):
            case obb.Ok(value=assignment):
                matched_preds, matched_targets = obb.gather_matched(preds, targets, assignment)
                criterion = obb.ProbIoULoss(obb.ProbIoUConfig(reduction=obb.Reduction.MEAN))
                loss = criterion.forward(matched_preds, matched_targets)
                loss.backward()
            case obb.Err(error=error):
                print(error)
    case _:
        print("Packing failed; inspect the typed error outcome")
```

Box criterion signature: `forward(preds: PackedBoxes, targets: PackedBoxes) -> Tensor`.
`forward` trusts geometry, aligned counts, and offsets; `checked` validates fresh inputs
and returns a typed result. Pure `stable_probiou_loss` returns unreduced values and
`stable_probiou` returns similarity. Outputs are float64; autocast is disabled locally
without changing your detector's outer mixed-precision context. Empty mean/sum is a
gradient-connected zero. Mean averages matched pairs across the whole pack, not per-sample means.

The matcher solves each sample independently, assigns the smaller side fully, and returns
detached indices on the input device. Its CPU solver makes an explicit cost transfer from
CUDA. Cost construction is chunked; complete cost storage remains quadratic within each
sample. See [matching contracts](docs/matching.md) for limits and solver references.

## Quality classification

For one complete decoder layer, `obb.SetCriterion` owns matching and weighted
classification, L1, and ProbIoU losses. Use `obb.pack_predictions(boxes, logits)` and
`obb.pack_targets(samples, device=..., dtype=...)`, where each `obb.TargetSample`
contains boxes, class labels, and caller-supplied quality. Match `obb.Ok(value=losses)`
from `criterion.forward(preds, targets)` and backpropagate `losses.total`.
Pass `reduction=obb.Reduction.NONE` for `[batch]` losses, or SUM/MEAN for a scalar.
Normalization is explicit: `normalizer=obb.TargetCount.PER_SAMPLE`, PER_BATCH,
or a positive scalar. The defaults SUM/PER_BATCH preserve batch-wide target normalization.
Pass named callbacks through `SetCriterion(losses=..., matcher=...)`; use
`obb.WeightedLoss(callback, obb.WeightConfig(weight=...))` to configure each term.
Matcher cost callbacks own their weights independently. Results expose normalized
weighted values through `losses.terms[name]` and their sum through `losses.total`.
Extend `obb.default_set_losses()` with your own callback, including query regularizers
over unmatched predictions. See [the custom set loss](examples/custom_set_loss.py),
[the DETR example](examples/detr.py), and
[set criterion contracts](docs/set_criterion.md).

```python
import obb_loss as obb

logits = torch.tensor([[0.3, -0.8]], requires_grad=True)
quality = torch.tensor([[0.7, 0.]])  # Caller-supplied [queries, classes], zero = negative.
vfl = obb.VarifocalLoss(obb.VFLConfig(alpha=0.75, gamma=2., quality_weighted=True))
qfl = obb.QualityFocalLoss(obb.QFLConfig(beta=2.))
classification_loss = vfl.forward(logits, quality)  # Or qfl.forward(logits, quality).
```

Pure `varifocal_loss` and `quality_focal_loss` return unreduced `[queries, classes]` values.
Checked modules reject nonfinite logits and quality outside `[0, 1]`. Targets are not
automatically detached; choose that policy explicitly. Mean averages all class elements;
positive-count or distributed normalization is yours. No auxiliary decoder-layer loop or
classification target synthesis is included. See [loss papers and numerical policy](docs/losses.md).

## Custom functions and typed settings

```python
import obb_loss as obb
from torch import Tensor

def center_loss(context: obb.BoxLossContext) -> obb.Result[Tensor, obb.InputError]:
    """Return one squared center penalty per aligned packed pair."""
    delta = context.preds.boxes[:, :2].double() - context.targets.boxes[:, :2].double()
    return obb.Ok(delta.square().sum(-1))

criterion = obb.CompositeBoxLoss(losses={
    "probiou": obb.box_probiou_loss,
    "center": obb.WeightedLoss(center_loss, obb.WeightConfig(weight=0.1)),
})
# criterion(preds, targets, reduction=obb.Reduction.MEAN) returns Ok(loss) or Err.
config_result = obb.parse_config(obb.ProbIoUConfig, {"weight": 2., "reduction": "sum"})
```

Box and set compositions share typed Result-returning callbacks and independently
weighted terms. Box callbacks receive aligned packed geometry through BoxLossContext;
set callbacks receive SetLossContext, including unmatched predictions. Conversion
helpers pack list/dense inputs at the caller boundary. Reduction is applied once by
the criterion. Callback bugs propagate, while observable output failures return Err.
Module parameters, buffers and hooks are preserved; zero-weight terms are omitted.
JSON configuration never serializes live callbacks.

The former BoxLossBuilder and build_loss/build_matcher factories are removed. Construct
CompositeBoxLoss from a callback mapping and built-in modules directly. Composite box
forward now returns Result and takes reduction at evaluation time. Callback module
state paths are shorter; recreate composition before loading a compatible state dict.

`parse_config` accepts JSON-compatible values and returns typed validation errors for
external settings. Direct Pydantic model construction retains its exception semantics.

## Verification

Use `uv sync` to install the development environment and `just check` to run strict
formatting, lint, type, and test checks. Python development is pinned to 3.12; package
metadata supports Python 3.11+. Tool settings live separately in `ruff.toml`,
`pyrefly.toml`, and `pytest.ini`.

Tensor annotations use native Pyrefly shapes. Shape packages are development-only,
versioned with the checker, and installed on Python 3.12+. Shape imports stay under
`TYPE_CHECKING`, so Python 3.11 runtime users need neither package. `just types` uses
`tools.typecheck` to provision an isolated Python 3.12 checker when invoked from 3.11;
the checker still validates Python 3.11 syntax. CI retains runtime tests for 3.11–3.14.
See [typing contracts](docs/typing.md) for guarantees and experimental stub limitations.

```text
just stress --samples 10000 --chunk-size 512 --dtype FP64
just benchmark --samples 4 --queries 128 --targets 32
just test-cuda
just build
```

Stress and benchmark commands emit JSON on stdout and progress on stderr; use `--quiet`
to suppress progress. `test-cuda` fails if hardware is unavailable, so skipped cases cannot
masquerade as GPU verification. Read [verification evidence](docs/verification.md) for
measured results and limitations. Build creates a portable pure-Python wheel; nothing publishes.
