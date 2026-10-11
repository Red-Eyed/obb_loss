"""Evaluate VFL and QFL with quality targets supplied by the training pipeline."""

import torch

from obb_loss import Err, Ok, QFLConfig, QualityFocalLoss, VarifocalLoss, VFLConfig


def main() -> None:
    """Compare two classification choices on the same queries and classes."""
    logits = torch.tensor([[0.3, -0.8], [-0.2, 0.7]], requires_grad=True)
    quality = torch.tensor([[0.7, 0.0], [0.0, 0.9]])
    criteria = (
        VarifocalLoss(VFLConfig(alpha=0.75, gamma=2.0, quality_weighted=True)),
        QualityFocalLoss(QFLConfig(beta=2.0)),
    )
    for criterion in criteria:
        logits.grad = None
        match criterion.checked(logits, quality):
            case Err(error=error):
                print(error)
                return
            case Ok(value=loss):
                loss.backward()
        print(f"{type(criterion).__name__}: {loss.detach().item():.6f}")
        print(f"logit gradients: {logits.grad}")


if __name__ == "__main__":
    main()
