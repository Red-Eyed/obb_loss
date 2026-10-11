# obb_loss

PyTorch oriented-box losses, VFL/QFL, and a standalone matcher with packed layouts.
Box criteria consume PackedBatch(values, cu_seqlens). Predictions contains packed
boxes and aligned logits; Targets contains packed boxes and aligned labels.
SetCriterion(config) owns matching, normalization and reduction for one layer;
forward takes Predictions, Targets and reduction. Subclasses override loss methods
or compute_losses to add named weighted sample vectors. Matcher weights are
independent from training weights. Classification quality is explicitly supplied
by overriding classification_quality after matching; it is not target annotation.
Detector layer orchestration belongs to the caller. Public failures return typed errors.
Set reductions operate on sample vectors: NONE retains samples, SUM/MEAN reduce them.
Target-count normalization is explicitly PER_SAMPLE or PER_BATCH; positive external
divisors are supported. Defaults SUM/PER_BATCH preserve batch-normalized losses.
Follow the user's global instructions and the local ignored implementation plan.

## Checks

Use uv for Python commands. Separate configs live in `ruff.toml`, `pyrefly.toml`, and
`pytest.ini`. Run `just check` for formatting, lint, types, and tests; do not weaken
checks or hide missing imports/Any to pass. Before commits run Ruff check, Ruff format,
reread changed files, and Pyrefly on committed Python files, plus relevant pytest tests.

## Contracts

Use strict Rustic Python: small local Ok/Err unions, precise error records, and match
dispatch. Use NamedTuple tensor records, native Pyrefly shape annotations, einops rearrangement,
and Pydantic configs outside tensor execution. Document every module, class, and function.
Library tensors are borrowed and never mutated. Expected public failures return typed errors.
Keep shape imports and symbols under TYPE_CHECKING with postponed annotations so
Python 3.11 does not import the Python 3.12-only shape package. Run Pyrefly through
`uv run python -m tools.typecheck`, which provisions the locked shape environment.
Development defaults to Python 3.12; package metadata and runtime tests retain 3.11+.

Pack at integration boundaries; preserve empty samples and autograd. Geometry is six
coordinates `(cx, cy, w, h, cos(theta), sin(theta))`. Float64-required criteria disable
autocast locally and convert internally. Cite primary papers and document numerical departures.

Keep optional components in separate files. Disabled terms are not instantiated or evaluated.
Custom module components must register parameters/buffers. Configuration never serializes
live callables. User training policy decisions must be explicit.
Results report normalized weighted values in SetLosses.terms and their total.

## Versioning

On main, bump package/user-visible changes and synchronize the lockfile. Runtime version
comes from installed metadata. Commit verified units separately. Only the user publishes.
