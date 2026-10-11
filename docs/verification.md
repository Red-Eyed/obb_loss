# Verification evidence

Local checks use Python 3.11.15 and 3.12.13, PyTorch 2.14.1, and macOS arm64 CPU.
Both interpreter suites passed 103 tests; 77 CUDA cases skipped on each interpreter.
CUDA remains unverified locally. The portable `py3-none-any` wheel supports Python
3.11+, and runnable examples cover public forward/backward APIs. The 0.10.0 wheel
passed packed box-callback forward/backward in an isolated Python 3.11 environment containing neither
jaxtyping nor shape_extensions.
CI is configured to test Python 3.11–3.14 on Linux; its execution is
unverified until the repository workflow runs. AOT eager fullgraph compilation was checked
for the packed ProbIoU criterion with changing pair counts, values, and backward. No
Inductor, export, MPS, or CUDA throughput guarantee is claimed.

Tests compare ProbIoU to independent matrix-solve/log-determinant Gaussian equations on
well-conditioned inputs, plus analytic center-shift/thin-box values and gradients where
matrix conditioning is unsuitable. VFL/QFL use scalar paper equations and gradcheck.
Matcher optimality is verified against exhaustive rectangular assignments, with empty
segments, sample isolation, uniqueness, bounded callback chunks, and gradient-preserving
gather. Strict checker negative fixtures verify Result narrowing and callback signatures.
Native Pyrefly checks also verify module call inference and reject incorrect tensor
widths, aligned row counts, and module callback signatures. The aligned composition custom
module test verifies that forward hooks execute through `Module.__call__`. Native
typing limitations and localized upstream stub workarounds are recorded in
[typing contracts](typing.md).

Set-criterion tests cover independent matching/training weights, analytic weighted
costs, 16 seeded rectangular assignments against brute-force optima, VFL/QFL parity,
L1 gradcheck, caller-quality gradients, disabled-term skipping, field revalidation,
empty samples, and all four floating input dtypes under outer autocast. The DETR
example verifies final and auxiliary layer backward passes. The high-level matcher
and criterion contain validation and a CPU solver; fullgraph compilation coverage
applies to the plain ProbIoU criterion, not this orchestration API.
Per-sample reduction tests compare sample vectors to independent calls, verify
explicit per-sample/batch normalization and SUM/MEAN equivalence, preserve zero-query
segments, and isolate gradients when selecting one sample's loss.
Callback composition tests verify custom regularizers on unmatched queries, registered
module parameters/buffers and hooks, disabled-term omission, custom bounded matching
costs, weighted named reports, typed failures, and malformed output rejection. Static
negative fixtures reject incorrect callback contexts and plain-Tensor return contracts.
Aligned callback tests cover NONE/SUM/MEAN, registered state/hooks, disabled terms,
and custom numerical domains independent of ProbIoU admission.

## Seeded CPU stress study

Run date: 2026-10-11. Per dtype: 10000 generated pairs, seed 901, chunks of at most 512.
Scale is log-uniform from `1e-30` to `1e30`; aspect ratios up to `1e24`. Cases alternate
exact equality, analytic thin-axis shifts, and general pairs with varying side ratios and
relative directions. Settings and the first unexplained failure index appear in JSON for
reproduction. Sampling depends on chunk size; preserve every setting when reproducing.

| Input dtype | Representable pairs evaluated | Input cast rejection | Forward failures | Backward cast overflows | Unexplained backward failures |
|---|---:|---:|---:|---:|---:|
| float64 | 10000 | 0 | 0 | 0 | 0 |
| float32 | 9724 | 276 | 0 | 142 | 0 |
| bfloat16 | 9367 | 633 | 0 | 28 | 0 |
| float16 | 536 | 9464 | 0 | 52 | 0 |

Float64 analytic-shift maximum absolute error: `5.551115123125783e-17`. Cast overflow is
classified only when float64 gradients of the same quantized input are finite and exceed
the leaf dtype range. That comparison diagnoses casting; it is not an independent mathematical
oracle. Input rejection counts nonfinite coordinates and underflowed nonpositive sides.

These distributions intentionally exceed sensible half-precision detector output ranges.
Internal float64 cannot repair already underflowed or quantized geometry, nor preserve a
gradient outside the leaf dtype range. Use suitable coordinate units and choose box-head
output precision explicitly. No nonfinite result is replaced with a plausible score.
The simulation supplies domain-scoped evidence rather than a proof for all possible inputs.

## Small CPU benchmark

Two samples, 32 predictions and 8 targets per sample, float64 inputs, chunk_size 65536,
two measured repeats after warmup. Complete cost storage: 4096 bytes.

| Stage | Mean wall time |
|---|---:|
| Trusted aligned loss | 0.083 ms |
| Checked aligned loss | 0.149 ms |
| Pairwise cost construction | 0.313 ms |
| CPU solver including transfer boundary | 0.020 ms |
| Complete matcher | 0.332 ms |
| Aligned loss forward and backward | 0.331 ms |

These small-workload timings are characterization, not production performance targets.
CUDA peak allocated memory is reported by the benchmark on CUDA; CPU solver working
memory and CPU peak resident memory are not yet instrumented. Timing stages are separated
so validation/transfer costs remain visible. Custom term math requires its own tests.
