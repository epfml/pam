import math

import torch
from torch.optim import Optimizer

class cross_entropy_autograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, X, Y, weight=None, ignore_index=None,
        reduction='mean', label_smoothing=0.0,
        op_dict=None,
    ):
        # Always returns standard loss value for comparison purposes
        # op_dict only used to compute the gradients
        ctx.save_for_backward(X, Y, weight)
        ctx.label_smoothing = label_smoothing
        ctx.ignore_index = ignore_index
        ctx.reduction = reduction
        assert op_dict is not None, "OP dict must be explicitly specified"
        ctx.op_dict = op_dict
        return torch.nn.functional.cross_entropy(
            input=X, target=Y, weight=weight,
            ignore_index=ignore_index,
            reduction=reduction, label_smoothing=label_smoothing,
        )

    @staticmethod
    def backward(ctx, delta_output):
        X, Y, weight = ctx.saved_tensors
        Y = Y.clone()
        X_prob = ctx.op_dict['softmax'](X, dim=-1)
        if ctx.ignore_index is not None:
            # Remove invalid labels before creating one-hot vector
            ignore_mask = Y.eq(ctx.ignore_index)
            Y[ignore_mask] = 0
        target_prob = torch.nn.functional.one_hot(Y, num_classes=X.shape[-1]).float()
        if ctx.label_smoothing:
            # Assume this is precomputed on CPU
            target_prob = target_prob * (1 - ctx.label_smoothing)
            target_prob = target_prob + (ctx.label_smoothing / X.shape[-1]) * torch.ones_like(target_prob)
        delta_X = X_prob - target_prob
        if weight is not None:
            applied_weights = weight[Y]
            delta_X = ctx.op_dict['mul'](delta_X, applied_weights.view(-1, 1))
        if ctx.ignore_index is not None:
            # Multiplication by 0 or 1 assumed exact
            delta_X = delta_X * (~ignore_mask.view(-1, 1))
        if ctx.reduction == 'mean':
            delta_X = ctx.op_dict['div'](delta_X, X.shape[0])
        delta_X = ctx.op_dict['mul'](delta_X, delta_output.view(-1, 1))
        return delta_X, *[None]*6


def cross_entropy(
    X, Y, weight=None, ignore_index=None,
    reduction='mean', label_smoothing=0.0,
    *, op_dict=None, 
):
    # Note that this function is always approx_bwd
    return cross_entropy_autograd.apply(
        X, Y, weight, ignore_index, reduction, label_smoothing, op_dict
    )

