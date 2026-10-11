# Examples

Run from the repository root. Each script uses fixed CPU inputs and typed error handling.

| Script | Demonstrates | Command |
| --- | --- | --- |
| [matching.py](matching.py) | Packed geometry, an empty sample, matching and ProbIoU | `uv run examples/matching.py` |
| [classification.py](classification.py) | VFL/QFL with caller-supplied quality | `uv run examples/classification.py` |
| [detr.py](detr.py) | Per-layer criterion, independent weights, detached matched quality and auxiliary layer aggregation | `uv run examples/detr.py` |
| [custom_set_loss.py](custom_set_loss.py) | A criterion subclass adding a direction penalty to all queries | `uv run examples/custom_set_loss.py` |

`just examples` runs all four scripts.

For DETR, pack dense outputs with `obb.pack_predictions` and list annotations with
`obb.pack_targets`. Boxes use `(cx, cy, width, height, cos(theta), sin(theta))`;
logits have no background column. Inputs share units/device; labels are zero-based int64.
The criterion operates on one layer; the caller chooses layer weights and aggregation.
Normalization is selected in `SetCriterionConfig`; reduction is passed to forward.
For distributed training, configure the positive external target-count divisor supplied
by your pipeline. See [criterion options](../docs/set_criterion.md).
