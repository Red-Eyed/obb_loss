# Examples

Run from the repository root. Each script runs on CPU and shows a complete backward pass.
`uv run` installs the project's environment as needed.

| Script | Demonstrates | Command |
| --- | --- | --- |
| [matching.py](matching.py) | Dense predictions, list targets, an empty sample, matching, and ProbIoU | `uv run examples/matching.py` |
| [classification.py](classification.py) | VFL and QFL with caller-supplied quality targets | `uv run examples/classification.py` |
| [custom_loss.py](custom_loss.py) | A custom pure function combined with ProbIoU through the composition | `uv run examples/custom_loss.py` |
| [detr.py](detr.py) | Per-layer set criterion, separate matcher/loss weights, L1, ProbIoU, and VFL | `uv run examples/detr.py` |
| [custom_set_loss.py](custom_set_loss.py) | Named weighted callbacks and a direction constraint on matched/unmatched queries | `uv run examples/custom_set_loss.py` |

`just examples` runs all five scripts. They use fixed inputs and typed error handling.

For DETR integration, adapt each layer's model output to `DecoderOutput`, pack your
annotations with `obb.pack_targets`, and call `layer_loss(output, targets, criterion)`.
The script uses synthetic decoder outputs to keep it runnable without a detector dependency.
Boxes use `(cx, cy, width, height, cos(theta), sin(theta))`; logits have shape
`[batch, queries, classes]` with no extra background class. Targets and outputs must
share coordinate units and device. Labels are int64, in `[0, classes)`, and ordered
with the target boxes. For angle-based output, use `obb.boxes_from_angles` first.

The criterion rematches each layer using classification, L1, and ProbIoU costs with
independent matching weights. VFL uses caller-supplied per-target quality (0.8 in
this example); unmatched query/classes have zero quality. Training uses separate
weighted callbacks and returns per-sample values with `Reduction.NONE` and
`TargetCount.PER_SAMPLE`. The caller averages those values, weights auxiliary layers
by 0.5, and sums the layer losses. Empty target samples still train negative classification.
For distributed training, compute the appropriate target-count normalizer in your
pipeline and pass its positive scalar value with `reduction=obb.Reduction.SUM`.
See [all normalization/reduction options](../docs/set_criterion.md).
