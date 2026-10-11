# Examples

Run from the repository root. Each script runs on CPU and shows a complete backward pass.
`uv run` installs the project's environment as needed.

| Script | Demonstrates | Command |
| --- | --- | --- |
| [matching.py](matching.py) | Dense predictions, list targets, an empty sample, matching, and ProbIoU | `uv run examples/matching.py` |
| [classification.py](classification.py) | VFL and QFL with caller-supplied quality targets | `uv run examples/classification.py` |
| [custom_loss.py](custom_loss.py) | A custom pure function combined with ProbIoU through the builder | `uv run examples/custom_loss.py` |
| [detr.py](detr.py) | Final and auxiliary decoder outputs, matching, box loss, quality targets, and VFL | `uv run examples/detr.py` |

`just examples` runs all four scripts. They use fixed inputs and typed error handling.

For DETR integration, adapt your model output to `DecoderOutput` and your per-sample
annotations to `Target`, then call `training_loss(final, targets, auxiliary=layers)`.
The script uses synthetic decoder outputs to keep it runnable without a detector dependency.
Boxes use `(cx, cy, width, height, cos(theta), sin(theta))`; logits have shape
`[batch, queries, classes]` with no extra background class. Targets and outputs must
share coordinate units and device. Labels are int64, in `[0, classes)`, and ordered
with the target boxes. For angle-based output, use `obb.boxes_from_angles` first.

The example rematches each layer using geometry costs, gives matched classes detached
ProbIoU quality scores, and gives unmatched queries zero quality. VFL trains all
query/class pairs. It sums both losses, weights box loss by 2, divides by the number
of targets in the local batch (at least 1), and weights auxiliary layers by 0.5.
These are example training policies. An all-empty target batch still trains negative
classification predictions. Distributed target-count normalization and classification-aware
matching remain choices for your pipeline.
