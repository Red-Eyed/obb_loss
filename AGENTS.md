# obb_loss

PyTorch oriented-box losses and full DETR criteria with packed sequence layouts.
Follow the user's global instructions and the local ignored implementation plan.

## Checks

Use uv for Python commands. Separate configs live in `ruff.toml`, `pyrefly.toml`, and
`pytest.ini`. Run `just check` for formatting, lint, types, and tests; do not weaken
checks or hide missing imports/Any to pass. Before commits run Ruff check, Ruff format,
reread changed files, and Pyrefly on committed Python files, plus relevant pytest tests.

## Contracts

Use strict Rustic Python: small local Ok/Err unions, precise error records, and match
dispatch. Use NamedTuple tensor records, jaxtyping shape annotations, einops rearrangement,
and Pydantic configs outside tensor execution. Document every module, class, and function.
Library tensors are borrowed and never mutated. Expected public failures return typed errors.

Pack at integration boundaries; preserve empty images and autograd. Geometry is six
coordinates `(cx, cy, w, h, cos(theta), sin(theta))`. Float64-required criteria disable
autocast locally and convert internally. Cite primary papers and document numerical departures.

Keep optional components in separate files. Disabled terms are not instantiated or evaluated.
Custom module components must register parameters/buffers. Configuration never serializes
live callables. User training policy decisions must be explicit.

## Versioning

On main, bump package/user-visible changes and synchronize the lockfile. Runtime version
comes from installed metadata. Commit verified units separately. Only the user publishes.
