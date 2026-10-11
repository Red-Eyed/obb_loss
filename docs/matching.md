# Standalone packed assignment

`HungarianMatcher.forward(preds: PackedBoxes, targets: PackedBoxes)` constructs
per-image ProbIoU costs and returns `Ok(Assignment)` or `Err(InputError)`.
A custom unreduced aligned cost function can replace the built-in geometry cost.
The matcher does not impose classification labels, quality targets, or decoder layers.

The assignment problem follows Harold W. Kuhn, *The Hungarian Method for the Assignment
Problem* (1955), [DOI:10.1002/nav.3800020109](https://doi.org/10.1002/nav.3800020109).
The configured CPU solver is SciPy's modified Jonker–Volgenant implementation, rather
than Kuhn's algorithm: David F. Crouse, *On implementing 2D rectangular assignment
algorithms* (2016), [DOI:10.1109/TAES.2016.140952](https://doi.org/10.1109/TAES.2016.140952).
See [SciPy's solver contract](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html).

Each image gets min(predictions, targets) unique pairs. Unmatched boxes stay in caller
records. Empty images produce repeated assignment offsets. Gather matched boxes with
`gather_matched`, then invoke a box criterion; gather preserves gradients.

Costs use row-major rectangular blocks, never cross-image pairs. Construction bounds
temporary pair geometry with `chunk_size`; full costs require sum(n_i*m_i) float64
elements, bounded by `max_pairs`. Defaults are 65536 temporary pairs and 10000000 total
pairs (80 MB cost storage). SciPy may allocate additional working storage. Metadata
transfers to host once per input; complete costs transfer once before solving. Checked
chunk admission synchronizes GPU work. Matching is outside compilation and autograd.
Tie assignments are solver-specific; tests require optimal cost and valid uniqueness,
not one arbitrary permutation. No GPU throughput claim is established by CPU tests.
