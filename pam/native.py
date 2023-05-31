import math
import torch

if __package__ is None or __package__ == '':
    # uses current directory visibility (running as script / jupyter notebook)
    import utils
else:
    # uses current package visibility (running as a module)
    from . import utils

@torch.jit.script
def get_exp_mantissa(x):
    '''
    Computes the exponent (floor log2) and mantissa fraction for a number x
    '''
    x = x.abs()
    x_e = torch.floor(torch.log2(x))  # the exponent of x, -inf for 0
    x_zm = 2.0 ** x_e  # x rounded down to a power of 2, 0 for 0
    x_m = (x - x_zm) / torch.where(x_zm > 0, x_zm, 1)  # mantissa frac, 0 for 0
    return x_e, x_m


@torch.jit.script
def pa_log2_fwd(x):
    '''
    Piecewise affine log2 (segments between points x=2**k for integer k),
    ignoring the sign of the input x.
    '''
    x_e, x_m = get_exp_mantissa(x)
    return x_e + x_m


@torch.jit.script
def pa_exp2_fwd(x):
    '''
    Piecewise affine exp2 (segments between int values of x)
    '''
    x_floor = torch.floor(x)
    return (2 ** x_floor) * torch.where(x_floor.isfinite(), 1 + x - x_floor, 1)


def pam_fwd(A, B, *, offset=None):
    '''
    Elementwise Piecewise Affine Multiplication between tensors A and B
    using torch primitives.
    Offset PAMs an additional fixed value with the resulting product.
    '''
    if not isinstance(A, torch.Tensor):
        A = torch.as_tensor(A)
    if not isinstance(B, torch.Tensor):
        B = torch.as_tensor(B)

    # exp(log + log), zeros go to zero
    sign = torch.sign(A) * torch.sign(B)
    out = sign * pa_exp2_fwd(pa_log2_fwd(A) + pa_log2_fwd(B))

    # We can introduce a multiplication offset (e.g. 1.0552) to
    # avoid systematic underestimation of the product
    if offset is not None:
        return pam_fwd(out, torch.full_like(out, offset), offset=None)
    return out


@torch.no_grad()
def pam_bwd(A, B, delta_C, *, offset=None):
    '''
    Compute the exact derivative of C = pam(A, B) w.r.t. A.
    The derivative is given by:
        delta_A = delta_C * 2**(floor(log2(C))-floor(log2(A))) * sign(B)
    and only involves a multiplication by an exact power of 2.
    '''
    if offset is not None:
        B = pam_fwd(B, offset)
    _, A_m = get_exp_mantissa(A)
    B_e, B_m = get_exp_mantissa(B)
    slope_exponent = B_e + 1.0 * ((A_m + B_m) >= 1.0)  # Works for zeros
    return delta_C * torch.sign(B) * 2 ** slope_exponent


class pam_autograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, offset=None, approx_bwd=False):
        ctx.save_for_backward(A, B)
        ctx.approx_bwd = approx_bwd  # Can't save non-tensors with other method
        ctx.offset = offset
        return pam_fwd(A, B, offset=offset)

    @staticmethod
    def backward(ctx, delta_Y):
        A, B = ctx.saved_tensors

        offset = ctx.offset
        approx_bwd = ctx.approx_bwd
        if isinstance(approx_bwd, bool):
            approx_bwd = (approx_bwd, approx_bwd)
        approx_A, approx_B = approx_bwd

        if approx_A:
            delta_A = pam_fwd(delta_Y, B, offset=offset)
        else:
            delta_A = pam_bwd(A, B, delta_Y, offset=offset)

        if approx_B:
            delta_B = pam_fwd(delta_Y, A, offset=offset)
        else:
            delta_B = pam_bwd(B, A, delta_Y, offset=offset)

        return delta_A, delta_B, None, None


def pam(A, B, *, offset=None, approx_bwd=False):
    # Always uses custom bwd (could use default autograd for approx_bwd=False)
    A, B = utils.broadcast_tensors(A, B)
    return pam_autograd.apply(A, B, offset, approx_bwd)


def pam_matmul(A, B, *, offset=None, approx_bwd=False):
    '''
    Computes PAM matmul AB with an optional offset using torch primitives.
    Slow and requires and may require large amounts of memory.
    Assumes shapes A is n by k and B is k by m
    '''
    return torch.sum(
        pam(
            A.view(A.shape[0], A.shape[1], 1),
            B.view(1, B.shape[0], B.shape[1]),
            offset=offset,
            approx_bwd=approx_bwd,
        ),
        axis=1,
    )


def pam_matmul_fwd(A, B, *, offset=None):
    '''
    Computes PAM matmul AB with an optional offset using torch primitives.
    Slow and requires and may require large amounts of memory.
    Assumes shapes A is n by k and B is k by m
    '''
    # Uses autograd through the exact fwd pass (mostly for testing)
    return torch.sum(
        pam_fwd(
            A.view(A.shape[0], A.shape[1], 1),
            B.view(1, B.shape[0], B.shape[1]),
            offset=offset,
        ),
        axis=1,
    )


def pad_fwd(A, B, *, offset=None):
    '''
    Elementwise Piecewise Affine Divsion between tensors A and B
    using torch primitives.
    Offset PAMs an additional fixed value with the resulting product.
    '''
    if not isinstance(A, torch.Tensor):
        A = torch.as_tensor(A)
    if not isinstance(B, torch.Tensor):
        B = torch.as_tensor(B)

    # exp(log - log), zeros go to zero
    sign = torch.sign(A) / torch.sign(B)
    out = sign * pa_exp2_fwd(pa_log2_fwd(A) - pa_log2_fwd(B))

    # We can introduce a multiplication offset (e.g. 1.0552) to
    # avoid systematic underestimation of the product
    if offset is not None:
        return pad_fwd(out, torch.full_like(out, offset), offset=None)
    return out


class pad_autograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, offset=None, approx_bwd=False):
        assert approx_bwd == True  # Use pad_fwd for exact derivative via torch autograd
        ctx.save_for_backward(A, B)
        ctx.offset = offset
        return pad_fwd(A, B, offset=offset)

    @staticmethod
    def backward(ctx, delta_Y):
        A, B = ctx.saved_tensors
        offset = ctx.offset
        delta_A = pad_fwd(delta_Y, B)
        B_square = pam_fwd(B, B, offset=offset)
        delta_B = -pad_fwd(pam_fwd(A, delta_Y, offset=offset), B_square, offset=offset)
        return delta_A, delta_B, None, None


def pad(A, B, *, offset=None, approx_bwd=False):
    A, B = utils.broadcast_tensors(A, B)
    if approx_bwd:
        return pad_autograd.apply(A, B, offset, approx_bwd)
    else:
        return pad_fwd(A, B, offset=offset)


def pa_exp_fwd(x, base=math.e, offset=None):
    '''
    Piecewise exp using pa_exp2
    '''
    e_factor = torch.log2(torch.tensor(base))
    return pa_exp2_fwd(pam_fwd(x, e_factor, offset=offset))


def pa_log_fwd(x, base=math.e, offset=None):
    '''
    Piecewise log using pa_log2
    '''
    base = torch.tensor(base)
    return pad_fwd(pa_log2_fwd(x), torch.log2(base), offset=offset)


class pa_exp_autograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, base, offset=None, approx_bwd=False):
        assert approx_bwd == True  # Use pad_fwd for exact derivative via torch autograd
        ctx.save_for_backward(A)
        ctx.base = base
        ctx.offset = offset
        return pa_exp_fwd(A, base, offset=offset)

    @staticmethod
    def backward(ctx, delta_Y):
        A, = ctx.saved_tensors
        base = ctx.base
        offset = ctx.offset

        Y = pa_exp_fwd(A, base, offset)
        if base != math.e:
            ln_base = torch.full_like(A, math.log(base))
            delta_A = pam_fwd(Y, ln_base, offset=offset)
            delta_A = pam_fwd(delta_A, delta_Y, offset=offset)
        else:
            delta_A = pam_fwd(Y, delta_Y, offset=offset)

        return delta_A, None, None, None


def pa_exp(X, *, base=math.e, offset=None, approx_bwd=False):
    if approx_bwd:
        return pa_exp_autograd.apply(X, base, offset, approx_bwd)
    else:
        return pa_exp_fwd(X, base, offset=offset)


def pa_exp2(X, *, offset=None, approx_bwd=False):
    if approx_bwd:
        return pa_exp_autograd.apply(X, 2.0, offset, approx_bwd)
    else:
        return pa_exp_fwd(X, 2.0, offset=offset)


class pa_log_autograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, base, offset=None, approx_bwd=False):
        assert approx_bwd == True  # Use pad_fwd for exact derivative via torch autograd
        ctx.save_for_backward(A)
        ctx.base = base
        ctx.offset = offset
        return pa_log_fwd(A, base, offset=offset)

    @staticmethod
    def backward(ctx, delta_Y):
        A, = ctx.saved_tensors
        base = ctx.base
        offset = ctx.offset

        delta_A = pad_fwd(delta_Y, A)
        if base != math.e:
            ln_base = torch.full_like(A, math.log(base))
            delta_A = pad_fwd(delta_A, ln_base, offset=offset)

        return delta_A, None, None, None


def pa_log(X, *, base=math.e, offset=None, approx_bwd=False):
    if approx_bwd:
        return pa_log_autograd.apply(X, base, offset, approx_bwd)
    else:
        return pa_log_fwd(X, base, offset=offset)


def pa_log2(X, *, offset=None, approx_bwd=False):
    if approx_bwd:
        return pa_log_autograd.apply(X, 2.0, offset, approx_bwd)
    else:
        return pa_log_fwd(X, 2.0, offset=offset)
