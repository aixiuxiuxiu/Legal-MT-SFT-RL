import torch


def token_accuracy(
    preds: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = -100,
) -> float:
    keep = target != ignore_index
    return torch.mean(preds[keep] == target[keep], dtype=torch.float).item()


def classification_accuracy(
    preds: list[str],
    target: list[str],
    ignore_case: bool = False,
) -> float:
    result = []
    for pred, tar in zip(preds, target):
        if ignore_case:
            result.append(pred.lower() == tar.lower())
        else:
            result.append(pred == tar)

    return torch.mean(torch.tensor(result), dtype=torch.float).item()
