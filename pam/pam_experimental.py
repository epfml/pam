import torch


def torch_softmax2(x, dim=None, dtype=None):
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html
    if dtype is not None:
        x = x.to(dtype=dtype)
    with torch.no_grad():
        shift = torch.floor(torch.max(x))
    x_adj = x - shift
    expx = torch.exp2(x_adj)
    denom = torch.sum(expx, dim=dim, keepdim=True)
    out = expx / denom
    return out


def torch_logsumexp2(x, dim=None, keepdim=None, dtype=None):
    if dtype is not None:
        x = x.to(dtype=dtype)
    with torch.no_grad():
        shift = torch.floor(torch.max(x))
    x_adj = x - shift
    expx = torch.exp2(x_adj)
    out = torch.log2(torch.sum(expx, dim=dim, keepdim=keepdim)) + shift
    return out


def torch_log_softmax2(x, dim=None, dtype=None):
    if dtype is not None:
        x = x.to(dtype=dtype)
    return x - torch_logsumexp2(x, dim=dim, keepdim=True)


def torch_cross_entropy2(
    x, target, weight=None,
    ignore_index=- 100, reduction='mean', label_smoothing=0.0
):
    x_log_prob = x - torch_logsumexp2(x, dim=-1)
    if ignore_index is not None:
        # Remove invalid labels before creating one-hot vector
        ignore_mask = target.eq(ignore_index)
        target = target.clone()
        target[ignore_mask] = 0
    target_prob = torch.nn.functional.one_hot(target, num_classes=x.shape[-1]).float()
    if label_smoothing:
        target_prob = target_prob * (1 - label_smoothing)
        target_prob = target_prob + (label_smoothing / x.shape[-1]) * torch.ones_like(target_prob)
    loss_terms = -x_log_prob * target_prob
    if ignore_index is not None:
        loss_terms = loss_terms * (~ignore_mask.view(-1, 1))
    if weight is not None:
        loss_terms = loss_terms * weight.view(-1, 1)
    loss = torch.sum(loss_terms, dim=-1) # Per sample
    if reduction == 'mean':
        loss = torch.mean(loss)
    elif reduction == 'sum':
        loss = torch.sum(loss)
    elif reduction is not None:
        raise ValueError("Unknown reduction")
    return loss
