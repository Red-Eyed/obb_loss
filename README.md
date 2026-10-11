# obb_loss

PyTorch oriented-box ProbIoU, L1, VFL/QFL, and Hungarian matching with packed tensors.
Supports Python 3.11+, CPU/CUDA, and mixed precision inputs. Development uses Python 3.12.

```python
import obb_loss as obb

criterion = obb.SetCriterion(obb.SetCriterionConfig(classification="none"))
result = criterion(preds, targets, reduction=obb.Reduction.NONE)
match result:
    case obb.Ok(value=losses):
        losses.total.mean().backward()
    case obb.Err(error=error):
        print(error)
```

`PackedBatch(values, cu_seqlens)` stores packed tensors and cumulative sample boundaries.
`Predictions(boxes: PackedBatch, logits)` and `Targets(boxes: PackedBatch, labels)`
keep geometry and classification roles explicit. Logits and labels follow box order
and share its boundaries. Geometry is `(cx, cy, w, h, cos(theta), sin(theta))`.
Pack dense decoder outputs with `obb.pack_predictions`; convert variable-length
annotations with `obb.pack_targets`. Geometry-only callers can use `obb.pack_boxes`
and `obb.pack_dense_boxes`.
All checked helpers return typed `Ok`/`Err` values and preserve autograd.

`SetCriterionConfig` selects losses, numerical parameters, normalization, and training
weights. `MatcherConfig` controls independent assignment weights and chunk limits.
`forward(preds, targets, reduction)` supervises one layer; callers aggregate layers.
Subclass `classification_quality` to supply matched VFL/QFL scores and choose their
gradient policy. Subclass loss methods to replace formulas, or `compute_losses` to
add named per-sample terms. Parameters and buffers use ordinary PyTorch registration.

Pure functions include `stable_probiou`, `stable_probiou_loss`, `box_l1_loss`,
`varifocal_loss`, and `quality_focal_loss`. `ProbIoULoss`, `VarifocalLoss`, and
`QualityFocalLoss` provide configured module interfaces. Stable ProbIoU computes in
float64 with autocast disabled; checked APIs reject invalid numerical domains.

See [set criterion](docs/set_criterion.md), [runnable examples](examples/README.md),
[loss equations and papers](docs/losses.md), [typing](docs/typing.md), and
[verification evidence](docs/verification.md).

Run `just check` for formatting, strict Ruff/Pyrefly checks, tests, and examples.
`just stress` runs reproducible numerical simulations; `just benchmark` separates
loss, validation, pairwise costs, and CPU assignment transfer/solve timing.
