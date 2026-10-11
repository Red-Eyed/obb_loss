# Packed set criterion

`SetCriterion(config)` supervises one prediction layer with matching, classification,
L1, and ProbIoU. Its interface is:

```python
forward(preds: Predictions, targets: Targets, reduction: Reduction = Reduction.SUM)
```

`PackedBatch(values, cu_seqlens)` stores tensor items and sample boundaries.
`Predictions` contains packed box geometry and floating `logits[queries, classes]`
before sigmoid, without a background column. `Targets` contains packed geometry
and int64 `labels[targets]`. Logits and labels share their boxes' order and boundaries.
Geometry-only box losses consume `PackedBatch` directly.
`pack_predictions` converts dense outputs; `pack_targets` converts `TargetSample`
records from existing list-based pipelines. Packing retains empty samples and gradients.
The criterion revalidates borrowed tensors because their contents can change.

```python
import obb_loss as obb

config = obb.SetCriterionConfig(
    classification="qfl", classification_weight=2,
    l1_weight=5, probiou_weight=2,
    matcher=obb.MatcherConfig(classification_weight=1, l1_weight=3, probiou_weight=2),
    normalizer=obb.TargetCount.PER_SAMPLE,
)
```

Matching weights determine assignment. Training weights determine gradient strength.
Zero training weights skip the corresponding formula. Classification can be disabled
with `classification="none"`. VFL parameters are `alpha`, `gamma` and
`quality_weighted`; QFL uses `beta`.

Classification quality belongs to a matched prediction/reference pair, not an annotation.
Override `classification_quality` to provide `[matches]` floating scores in `[0, 1]`.
For example, choose detached localization quality explicitly:

```python
from torch import Tensor
from typing_extensions import override

class DetectorCriterion(obb.SetCriterion):
    @override
    def classification_quality(
        self, preds: obb.Predictions, targets: obb.Targets,
        assignment: obb.Assignment,
    ) -> obb.Result[Tensor, obb.InputError]:
        first, second = obb.gather_matched(preds.boxes, targets.boxes, assignment)
        return obb.Ok(obb.stable_probiou(first.values, second.values).detach())

criterion = DetectorCriterion(config)
```

The base method returns a configuration error when classification is enabled. No quality
policy is inferred. Supplied quality gradients remain connected unless the caller detaches.
Unmatched query/classes receive zero classification targets.

Override `l1_loss` or `probiou_loss` to change matched-box formulas. These receive aligned,
gathered `PackedBatch` with match offsets. Override `classification_loss` to change
classification itself. Override `compute_losses(preds, targets, assignment)` to add or
replace terms; return `Ok` of a mapping from names to weighted raw `[samples]` sums,
or `Err`. This method sees all queries, so regularizers can supervise unmatched boxes.
See [custom_set_loss.py](../examples/custom_set_loss.py) for a direction constraint.
Assign custom parameters/modules/buffers to the subclass using normal PyTorch APIs.

Forward validates custom vectors, normalizes, reduces and reports read-only named terms,
their total, and the assignment. `NONE` returns `[samples]`; `SUM` adds the vector;
`MEAN` averages it, including empty samples. Empty scalar reductions return connected zero.
Normalization is configured once: `PER_SAMPLE` divides by each sample's target count
clamped to one; `PER_BATCH` divides every sample by total target count clamped to one;
a positive finite float supplies an external divisor. Defaults are `SUM/PER_BATCH`.

`SetHungarianMatcher(config)` is independently usable. Its `cost` method can be overridden
for custom pair costs; `pairwise_costs` retains gradients when called directly. Assignment
runs without gradients and transfers cost blocks explicitly to the CPU SciPy solver.
Chunks exclude cross-sample pairs; `max_pairs` limits full cost storage.
Packed storage avoids repeated concatenation but does not remove per-sample solver calls
or CPU transfer. Benchmark your workload rather than assuming packing alone is faster.

This criterion uses sigmoid VFL/QFL and oriented ProbIoU. Original
[DETR](https://arxiv.org/abs/2005.12872) uses softmax classification and GIoU;
[loss references](losses.md) document the adopted equations and numerical departures.
