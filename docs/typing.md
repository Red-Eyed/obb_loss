# Native tensor and callback typing

Development defaults to Python 3.12; the library supports Python 3.11+.
`pyrefly.toml` checks against Python 3.11 syntax. Library code uses postponed
annotations and guards shape imports and symbol declarations with `TYPE_CHECKING`:

```python
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shape_extensions import IntVar
    from torch import Tensor

    N = IntVar("N")

def aligned(preds: Tensor[[N, 6]], targets: Tensor[[N, 6]]) -> Tensor[[N]]:
    return (preds[:, :4] - targets[:, :4]).abs().sum(-1)
```

These annotations are static contracts. They do not validate dtype, device,
finiteness, geometry, or mutable tensor contents. Checked entry points retain those
runtime validations. Packed records use unknown integer extents with known ranks
and coordinate width; validators establish offset and cross-field relationships.
ProbIoU accepts independently shaped broadcast inputs with a fixed final width of
six; its output shape currently remains gradual through the numerical helpers.

`just types` runs `uv run python -m tools.typecheck`. The runner uses the active
environment when native shapes are installed, or automatically provisions a locked,
isolated Python 3.12 environment. The packages are pinned together at 1.3.1 in the
`shapes` dependency group. No shape package is a runtime dependency.

The native Torch stubs use `ProxyMethod["forward"]` for `nn.Module.__call__`.
Consequently, `module(...)` carries the typed `forward` signature and preserves
PyTorch hooks. Box callbacks accept the `BoxFunction` protocol, including compatible
modules, without accepting arbitrary modules or casting away their signatures.
Static tests verify successful inference and rejection of incorrect coordinate
widths, aligned row counts, module arguments, and callback signatures.

The upstream stubs are experimental. With strict callable subtyping, their base
`Module.forward(*args, **kwargs)` declaration conflicts with ordinary narrowed
subclass signatures. Local `bad-override` directives cover that specific base
signature mismatch; argument, return, and callback protocol checks remain enabled.
Dtype constants are currently typed as `Any` upstream; the precision-selection
adapter casts only those known Torch constants to `torch.dtype`. Tests use the
tuple form of `slogdet` and set empty tensor gradients after allocation to match
both runtime Torch and the native stubs.

Python 3.12 generic declaration syntax cannot appear in executable library files
while supporting 3.11. Newer syntax in external stub files is parsed by Pyrefly,
not imported by the runtime. Do not evaluate native annotations at runtime without
an explicitly supported annotation-resolution boundary.

Set composition uses `LossFunction[SetLossContext]` for raw sample vectors and
`LossFunction[SetCostContext]` for bounded candidate vectors. Both return typed
`Result[Tensor[[int]], InputError]`. Static tests reject unchecked Tensor returns and
mixing a cost context into a sample-loss selection. Runtime checks establish vector
counts, floating dtype, device, finiteness, and valid record contents.
