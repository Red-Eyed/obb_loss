# Per-layer set criterion

`obb.SetCriterion` matches one packed decoder layer, evaluates named callbacks,
normalizes their per-sample values, and reduces the batch. Its interface is
`forward(preds: PackedPredictions, targets: PackedTargets, *, reduction=Reduction.SUM,
normalizer=TargetCount.PER_BATCH) -> Result[SetLosses, InputError]`.
The caller orchestrates final and auxiliary layers.

Loss callbacks own their weighting, optionally through `obb.WeightedLoss`. Matcher
cost callbacks are configured independently:

```python
import obb_loss as obb

criterion = obb.SetCriterion(
    matcher=obb.SetHungarianMatcher(
        obb.MatcherConfig(chunk_size=4096, max_pairs=1_000_000),
        costs={
            "classification": obb.set_classification_cost,
            "l1": obb.WeightedLoss(obb.set_l1_cost, obb.WeightConfig(weight=3)),
        },
    ),
    losses={
        "classification": obb.set_varifocal_loss,
        "l1": obb.WeightedLoss(obb.set_l1_loss, obb.WeightConfig(weight=5)),
        "probiou": obb.WeightedLoss(obb.set_probiou_loss, obb.WeightConfig(weight=2)),
    },
)
```

Omitting either selection creates fresh default callbacks: VFL/L1/ProbIoU losses
and sigmoid-class/L1/ProbIoU costs, each with weights 1/5/2. `default_set_losses()`
and `default_set_costs()` return mappings that callers can extend or replace.
An explicit empty loss mapping returns a connected zero and the assignment. An empty
or entirely disabled cost mapping returns a typed configuration error.

## Callback contracts

`LossFunction[SetLossContext]` accepts one positional context and returns
`Ok(floating [samples] values)` or `Err(InputError)`. Values are weighted raw sums:
callbacks do not apply the criterion's target-count normalization or batch reduction.
The context exposes all predictions, targets, assignment, and matched geometry gathered
once with autograd preserved. Query regularizers can inspect unmatched predictions.

```python
from torch import Tensor

def direction_penalty(context: obb.SetLossContext) -> obb.Result[Tensor, obb.InputError]:
    directions = context.preds.geometry.boxes[:, 4:6].double()
    values = (directions.square().sum(-1) - 1).square()
    return obb.Ok(obb.sum_samples(values, context.preds.geometry.cu_seqlens))

criterion = obb.SetCriterion(
    losses={
        **obb.default_set_losses(),
        "direction": obb.WeightedLoss(direction_penalty, obb.WeightConfig(weight=0.5)),
    },
)
```

This constraint enforces `cos² + sin² = 1` on all queries. Geometry admission still
requires nonzero direction vectors; the regularizer does not relax that contract.
See the runnable [custom set loss example](../examples/custom_set_loss.py).

`LossFunction[SetCostContext]` instead returns `Ok(floating [pairs] values)`.
Its context contains packed predictions/targets and aligned global candidate indices
for a bounded same-sample chunk. Lower combined cost means a preferable assignment.
Default costs ignore target quality; custom costs may explicitly use it.
Direct `set_pairwise_costs` calls retain gradients. The matcher runs callbacks under
`no_grad`, transfers packed costs to the CPU solver, and returns detached indices
on the original device. Full storage is `sum(queries_i * targets_i)` and is bounded
by `MatcherConfig.max_pairs`; temporary candidate storage uses `chunk_size`.

Plain functions, configured `functools.partial` callbacks, and typed `nn.Module`
implementations share these contracts. Modules implement `forward(context)` and are
invoked through `module(context)`, preserving hooks, parameters, buffers, `.to()`,
and `state_dict()`. Callback selections are copied at construction; module state
remains caller-owned and can deliberately be shared. Zero-weight wrappers are omitted
from registration and evaluation. A module already constructed by the caller is not
retroactively unconstructed; omit its creation when selecting a disabled component.

`WeightConfig` is frozen and validates finite nonnegative weights. The wrapper
checks floating output before conversion/weighting. Composition checks vector shape,
device, and finiteness, including overflow after weighting. Callback errors retain
their category/detail with the term name prefixed to the field. Unexpected callback
bugs propagate. Callbacks must not mutate borrowed inputs.

Configure classification math separately from weighting:

```python
from functools import partial

qfl = partial(obb.set_quality_focal_loss, config=obb.QFLParameters(beta=1.5))
criterion = obb.SetCriterion(losses={"quality": obb.WeightedLoss(qfl, obb.WeightConfig(weight=2))})
```

`VFLParameters` controls alpha, gamma, and positive-quality weighting;
`QFLParameters` controls beta. The standalone VFL/QFL modules retain their existing
`VFLConfig`/`QFLConfig` interfaces. JSON configs cannot reconstruct live Python
callbacks: rebuild the same named composition before loading a state dictionary.

## Packed supervision and results

`PackedPredictions` holds geometry and logits `[queries, classes]` without a background
column. `PackedTargets` holds geometry, int64 labels `[targets]`, and caller quality
`[targets]` in `[0, 1]`. Records share device and order; predictions/targets share sample
count and coordinate units. Samples may have different query and reference counts.
Use `pack_predictions` for dense outputs and `pack_targets` for `TargetSample` sequences.
The matcher revalidates current contents, including labels against the class count.

L1 compares `(cx, cy, width, height)` and excludes direction. Its scale follows the
supplied units; swapping width/height changes L1 even when rotated geometry is equivalent.
ProbIoU supervises the Gaussian-box geometry. VFL/QFL scatter caller quality to matched
labeled classes; every other query/class receives zero quality. Quality is never
synthesized or detached. VFL treats zero quality as negative even for a matched target.

`SetLosses.total` is the sum of all normalized weighted terms. `SetLosses.terms` is a
read-only mapping from caller names to normalized weighted tensors; disabled terms
are absent. `SetLosses.assignment` contains detached global indices and sample offsets.
No component names are hardcoded into the result. The previous `SetWeights`,
`SetMatcherConfig`, and `SetCriterionConfig` types are removed; use callback wrappers
and `MatcherConfig` instead of fixed weight slots. Access `losses.terms["l1"]`
instead of the former unweighted `losses.l1` field.

| Argument | Option | Behavior |
| --- | --- | --- |
| `normalizer` | `TargetCount.PER_SAMPLE` | Divide each sample by its target count, clamped to 1 |
| `normalizer` | `TargetCount.PER_BATCH` | Divide every sample by the local batch target count, clamped to 1 |
| `normalizer` | Positive finite scalar | Divide every sample by the supplied value; 1 retains raw weighted sums |
| `reduction` | `Reduction.NONE` | Return `[samples]` for every named term and total |
| `reduction` | `Reduction.SUM` | Sum the normalized sample vector |
| `reduction` | `Reduction.MEAN` | Average the normalized sample vector |

For weighted sample sums `[6, 9]` and target counts `[1, 3]`, PER_SAMPLE yields
`[6, 3]`, while PER_BATCH yields `[1.5, 2.25]`. Reduction then operates on that vector.
Empty samples remain in position and count toward MEAN. Zero samples produce an empty
NONE vector or connected scalar zero. Distributed target-count aggregation belongs
to the caller: pass its positive scalar divisor and use SUM.

Autocast is disabled during callbacks. Built-ins and wrapper weighting use float64;
custom functions choose their own internal precision and must return finite values.
Existing ProbIoU numerical domains and low-precision leaf-gradient limits still apply.

Set assignment follows [Carion et al., DETR, ECCV 2020](https://arxiv.org/abs/2005.12872).
Original DETR uses softmax CE and GIoU; the default oriented callbacks use sigmoid
VFL/QFL and ProbIoU. Loss papers and numerical departures are documented in
[losses.md](losses.md).
