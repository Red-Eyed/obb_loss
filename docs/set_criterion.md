# Per-layer set criterion

`obb.SetCriterion` matches and supervises one decoder layer. Its interface is
`forward(preds: PackedPredictions, targets: PackedTargets, *, reduction=Reduction.SUM,
normalizer=TargetCount.PER_BATCH)`.
It returns `Ok(SetLosses)` or `Err(InputError)`. Iterate over decoder layers and
aggregate their weighted totals in your pipeline; the criterion has no auxiliary loop.

`PackedPredictions` carries `geometry: PackedBoxes` and floating logits `[queries, classes]`.
`PackedTargets` carries geometry, int64 labels `[targets]`, and floating quality
`[targets]` in `[0,1]`. Fields share their record's device and packed order; predictions
and targets share sample count, device, and coordinate units. There is no background
logit. Samples may have different query and target counts. `obb.pack_predictions`
adapts dense outputs; `obb.pack_targets` adapts a sequence of `obb.TargetSample` records.
Checked factories `obb.make_packed_predictions` and `obb.make_packed_targets` borrow
native packed records. Forward revalidates contents, including labels against class count.

Matching combines negative sigmoid probability for the target class, L1 center/side
distance, and ProbIoU loss. Quality never affects assignment. Costs are packed per
sample, constructed in bounded chunks, and detached before the shared CPU solver.
Full storage is `sum(queries_i * targets_i)`; `matcher.storage.max_pairs` limits it.
Assignment pairs the smaller side fully, so use enough queries to cover targets if
every target must be supervised. Matching ignores all training-loss weights.

L1 compares `(cx, cy, width, height)`, excluding direction. Use shared normalized
coordinates for DETR-style L1 weights. Width/height and direction representations
must be consistent: L1 is not invariant to swapping width and height. ProbIoU
supervises orientation. Matched target quality is scattered to its labeled class;
all other query/classes receive zero quality. VFL/QFL train all query/class pairs.
Quality is never synthesized or detached; detach it before packing if that is your
policy. VFL treats zero quality as negative, even for a matched target.

`SetWeights(classification=1, l1=5, probiou=2)` is used independently by
`SetCriterionConfig.weights` and `.matcher.weights`. Classification formulation
settings select `VFLConfig` or `QFLConfig`; their standalone `weight` and `reduction`
must retain defaults because the set criterion owns both operations. All-zero
training weights are allowed; all-zero matching weights are rejected. Disabled
terms skip numerical evaluation and return gradient-connected zeros.

`SetLosses.classification`, `.l1`, and `.probiou` are normalized, unweighted losses;
`.total` is their weighted sum. `.assignment` contains detached global indices.
The criterion first sums query/class losses and matched box losses within each
sample, then normalizes, then reduces across samples. Both options are explicit:

| Argument | Option | Behavior |
| --- | --- | --- |
| `normalizer` | `TargetCount.PER_SAMPLE` | Divide each sample by its own target count, clamped to 1 |
| `normalizer` | `TargetCount.PER_BATCH` | Divide every sample by the local batch target count, clamped to 1 |
| `normalizer` | Positive finite scalar | Divide every sample by the supplied value; 1 gives raw sums |
| `reduction` | `Reduction.NONE` | Return `[batch]` for every loss component and total |
| `reduction` | `Reduction.SUM` | Sum the normalized sample vector into a scalar |
| `reduction` | `Reduction.MEAN` | Average the normalized sample vector into a scalar |

For raw sample losses `[6, 9]` and target counts `[1, 3]`, `PER_SAMPLE` yields
`[6, 3]`, while `PER_BATCH` yields `[1.5, 2.25]`. MEAN then averages the chosen
vector; it does not change its normalization policy. Empty samples are retained
and count toward MEAN. Zero samples produce an empty NONE vector or a connected
scalar zero for SUM/MEAN.

Defaults are SUM and PER_BATCH, preserving the previous batch-wide normalized
scalar behavior. `TargetCount.LOCAL` is replaced by the explicit PER_BATCH name.
Distributed target-count reduction belongs to the caller: pass its positive scalar
value and use SUM. Empty targets produce zero box losses and negative classification
supervision. The DETR example shows explicit per-sample normalization and caller
aggregation of the returned vector.

Computations use float64 with autocast disabled locally. Inputs remain connected
to autograd, including caller quality. Existing ProbIoU domain and low-precision
input/gradient-cast limitations apply. Costs, component losses, and weighted totals
must remain finite; checked violations return typed errors.

The set loss and independent matching weights follow [Carion et al., DETR, ECCV
2020](https://arxiv.org/abs/2005.12872). Original DETR uses softmax classification,
cross-entropy, L1, and GIoU. This oriented variant uses sigmoid VFL/QFL and ProbIoU;
negative sigmoid class probability is the explicit matching policy. Loss-specific
paper references and numerical deviations are in [losses.md](losses.md).
