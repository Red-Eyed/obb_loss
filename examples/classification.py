"""Evaluate VFL and QFL with quality targets supplied by the training pipeline."""

import torch

import obb_loss as obb


def main() -> None:
    """Compare two classification choices on the same queries and classes."""
    logits = torch.tensor([[0.3, -0.8], [-0.2, 0.7]], requires_grad=True)
    quality = torch.tensor([[0.7, 0.0], [0.0, 0.9]])
    criteria = (
        obb.VarifocalLoss(obb.VFLConfig(alpha=0.75, gamma=2.0, quality_weighted=True)),
        obb.QualityFocalLoss(obb.QFLConfig(beta=2.0)),
    )
    for criterion in criteria:
        logits.grad = None
        match criterion.checked(logits, quality):
            case obb.Err(error=error):
                print(error)
                return
            case obb.Ok(value=loss):
                loss.backward()
        print(f"{type(criterion).__name__}: {loss.detach().item():.6f}")
        print(f"logit gradients: {logits.grad}")


if __name__ == "__main__":
    main()
