"""Pack a batch, match boxes per sample, and backpropagate the matched box loss."""

import torch

from obb_loss import (
    Err,
    HungarianMatcher,
    MatcherConfig,
    Ok,
    ProbIoUConfig,
    ProbIoULoss,
    gather_matched,
    pack_boxes,
    pack_dense_boxes,
)


def main() -> None:
    """Use dense predictions and variable-length targets, including an empty sample."""
    predictions = torch.tensor(
        [
            [[0.1, 0.0, 2.0, 1.0, 1.0, 0.0], [3.0, 0.0, 2.0, 1.0, 1.0, 0.0]],
            [[0.2, 0.0, 2.0, 1.0, 1.0, 0.0], [4.0, 0.0, 2.0, 1.0, 1.0, 0.0]],
        ],
        requires_grad=True,
    )
    match pack_dense_boxes(predictions):
        case Err(error=error):
            print(error)
            return
        case Ok(value=preds):
            pass
    match pack_boxes(
        [torch.tensor([[0.0, 0.0, 2.0, 1.0, 1.0, 0.0]]), torch.empty(0, 6)],
        device=predictions.device,
        dtype=predictions.dtype,
    ):
        case Err(error=error):
            print(error)
            return
        case Ok(value=targets):
            pass
    matcher = HungarianMatcher(MatcherConfig(chunk_size=16, max_pairs=100))
    match matcher.forward(preds, targets):
        case Err(error=error):
            print(error)
            return
        case Ok(value=assignment):
            pass
    matched_preds, matched_targets = gather_matched(preds, targets, assignment)
    criterion = ProbIoULoss(ProbIoUConfig())
    match criterion.checked(matched_preds, matched_targets):
        case Err(error=error):
            print(error)
            return
        case Ok(value=loss):
            loss.backward()
    print(f"assignment offsets: {assignment.cu_seqlens.tolist()}")
    print(f"matched box loss: {loss.detach().item():.6f}")
    print(f"prediction gradients: {predictions.grad}")


if __name__ == "__main__":
    main()
