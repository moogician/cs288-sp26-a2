import torch
from torch import Tensor

def softmax(x: Tensor, dim: int = -1) -> Tensor:
    x_max = x.max(dim=dim, keepdim=True).values
    x_shifted = x - x_max
    exp_x = torch.exp(x_shifted)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)

def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    x_max = logits.max(dim=-1, keepdim=True).values
    log_sum_exp = x_max.squeeze(-1) + torch.log(
        torch.exp(logits - x_max).sum(dim=-1)
    )
    n = logits.shape[0]
    log_probs_correct = logits[torch.arange(n), targets] - log_sum_exp

    return -log_probs_correct.mean()


def gradient_clipping(parameters, max_norm: float) -> Tensor:
    params_with_grad = [p for p in parameters if p.grad is not None]

    if not params_with_grad:
        return torch.tensor(0.0)
    grad_norms = torch.stack([p.grad.detach().norm(2.0) for p in params_with_grad])
    total_norm = grad_norms.norm(2.0)
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)

    for p in params_with_grad:
        p.grad.detach().mul_(clip_coef_clamped)
    return total_norm

def token_accuracy(logits: Tensor, targets: Tensor, ignore_index: int = -100) -> Tensor:
    valid_mask = targets != ignore_index
    if not valid_mask.any():
        return torch.tensor(0.0)

    predictions = logits.argmax(dim=-1)
    correct = (predictions[valid_mask] == targets[valid_mask]).float()
    return correct.mean()

def perplexity(logits: Tensor, targets: Tensor, ignore_index: int = -100) -> Tensor:
    valid_mask = targets != ignore_index
    if not valid_mask.any():
        return torch.tensor(float('inf'))

    valid_logits = logits[valid_mask]
    valid_targets = targets[valid_mask]

    ce = cross_entropy(valid_logits, valid_targets)
    return torch.exp(ce)
