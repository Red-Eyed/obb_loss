# Loss formulations and numerical policy

## ProbIoU

Jeffri Murrugarra-Llerena, Luis Felipe Zeni, Lucas N. Kirsten, and Claudio R. Jung.
*Gaussian Bounding Boxes and Probabilistic Intersection-over-Union for Object Detection*
(2021), [arXiv:2106.06072](https://arxiv.org/abs/2106.06072).
The journal version is *Probabilistic Intersection-Over-Union for Training and Evaluation
of Oriented Object Detectors*, IEEE TIP 33 (2024), 671–681,
[DOI:10.1109/TIP.2023.3348697](https://doi.org/10.1109/TIP.2023.3348697).

The rectangle maps to a Gaussian with center `(cx, cy)` and covariance
`R diag(w²/12, h²/12) Rᵀ`. The loss is Hellinger distance
`sqrt(1 - exp(-BD))`; similarity is one minus loss. It measures Gaussian overlap,
not exact polygon intersection. Square orientation and antipodal directions are equivalent.

The numerical reformulation comes from the local `ml-sandbox/04_stable_probiou` experiment:
positive determinant residuals, detached pairwise scale normalization, float64 work,
`log1p` and `expm1`, finite inactive branches, and a vector norm at equality. Exact equality
has loss zero and the chosen zero subgradient; nearby gradients have no epsilon plateau.
Translation residuals clamp to ±32 after which the relevant exponential is negligible.
The loss is computed directly to preserve small near-equality values.

CPU and CUDA only. Inputs must be finite, sides positive, directions nonzero. Pairwise
normalized sides must be at least `1e-30`, offsets at most `1e30`. Checked paths reject
outside-domain inputs; pure/forward paths trust the caller. Value validation synchronizes
accelerators. All loss/similarity outputs are float64 and autocast is locally disabled.
Internal conversion cannot recover quantized inputs or prevent overflow when gradients
return to lower-precision leaves. No input is mutated or detached.

## Quality classification

Haoyang Zhang, Ying Wang, Feras Dayoub, and Niko Sünderhauf.
*VarifocalNet: An IoU-aware Dense Object Detector*, CVPR 2021,
[arXiv:2008.13367](https://arxiv.org/abs/2008.13367).
Varifocal loss weights positive BCE by quality and negative BCE by `alpha * sigmoid(logit)^gamma`.
The configurable non-quality-weighted positive variant follows the authors' implementation.

Xiang Li et al. *Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes
for Dense Object Detection*, NeurIPS 2020,
[arXiv:2006.04388](https://arxiv.org/abs/2006.04388).
Quality focal loss modulates BCE with `abs(target - sigmoid(logit))^beta`.

Quality targets are supplied explicitly in `[0, 1]`, one value per logit; zero represents
a negative class. The library does not derive assignment, positive masks, or quality labels.
It does not detach targets automatically: callers choose whether quality should carry gradients.
Mean reduction averages all class elements; detector-specific normalization belongs to callers.
