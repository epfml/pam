import logging
import math
import pathlib

import torch
from torch.utils.cpp_extension import load

if __package__ is None or __package__ == '':
    # uses current directory visibility (running as script / jupyter notebook)
    import utils
else:
    # uses current package visibility (running as a module)
    from . import utils

kernels_path = str(pathlib.Path(__file__).parent.resolve() / 'cuda_kernels.cu')
# See more flags https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.load
build_dir = pathlib.Path('/tmp/pam_build')
build_dir.mkdir(exist_ok=True)
cuda_kernels = load(
    name='cuda_kernels',
    sources=[kernels_path],
    verbose=True,
    extra_cuda_cflags=['-std=c++17', '-O3'],
    build_directory=build_dir,
)

class pam_autograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, offset=None, approx_bwd=False):
        # TODO: Broadcast A,B if necessary?
        if offset is None:
            offset = 1.0  # Turns it off, equivalent to multiplying result with 1.0
        ctx.save_for_backward(A, B)
        ctx.approx_bwd = approx_bwd  # Can't save non-tensors with other method
        ctx.offset = offset
        return cuda_kernels.pam_fwd(A, B, offset)

    @staticmethod
    def backward(ctx, delta_Y):
        A, B = ctx.saved_tensors
        offset = ctx.offset
        approx_bwd = ctx.approx_bwd
        if isinstance(approx_bwd, bool):
            approx_bwd = (approx_bwd, approx_bwd)
        approx_A, approx_B = approx_bwd

        if approx_A:
            delta_A = cuda_kernels.pam_fwd(delta_Y.contiguous(), B, offset)
        else:
            delta_A = cuda_kernels.pam_bwd(A, B, delta_Y.contiguous(), offset)

        if approx_B:
            delta_B = cuda_kernels.pam_fwd(delta_Y.contiguous(), A, offset)
        else:
            delta_B = cuda_kernels.pam_bwd(B, A, delta_Y.contiguous(), offset)

        return delta_A, delta_B, None, None


def pam(A, B, *, offset=None, approx_bwd=False):
    # Always uses custom bwd (could use default autograd for approx_bwd=False)
    A, B = utils.broadcast_tensors(A, B)
    return pam_autograd.apply(A, B, offset, approx_bwd)


class pad_autograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, offset=None, approx_bwd=False):
        # TODO: Broadcast A,B if necessary?
        if offset is None:
            offset = 1.0  # Turns it off, equivalent to multiplying result with 1.0
        ctx.save_for_backward(A, B)
        ctx.approx_bwd = approx_bwd  # Can't save non-tensors with other method
        ctx.offset = offset
        return cuda_kernels.pad_fwd(A, B, offset)

    @staticmethod
    def backward(ctx, delta_Y):
        A, B = ctx.saved_tensors
        offset = ctx.offset
        approx_bwd = ctx.approx_bwd
        if isinstance(approx_bwd, bool):
            approx_bwd = (approx_bwd, approx_bwd)
        approx_A, approx_B = approx_bwd

        if approx_A:
            delta_A = cuda_kernels.pam_fwd(cuda_kernels.pad_fwd(torch.ones_like(B), B, offset), delta_Y.contiguous(), offset)
        else:
            delta_A = cuda_kernels.pam_bwd(A, cuda_kernels.pad_fwd(torch.ones_like(B), B, offset), delta_Y.contiguous(), 1.0)

        if approx_B:
            dY_dB = cuda_kernels.pad_fwd(cuda_kernels.pad_fwd(A, B, offset), B, offset)
            delta_B = -cuda_kernels.pam_fwd(dY_dB, delta_Y.contiguous(), offset)
        else:
            delta_B = cuda_kernels.pad_bwd(A, B, delta_Y.contiguous(), offset)

        return delta_A, delta_B, None, None


def pad(A, B, *, offset=None, approx_bwd=False):
    # Always uses custom bwd (could use default autograd for approx_bwd=False)
    A, B = utils.broadcast_tensors(A, B)
    return pad_autograd.apply(A, B, offset, approx_bwd)


def check_row_major(tensor, target_dim):
    if tensor.dim() != target_dim:
        raise ValueError(f"{tensor.dim()=} is not equal to {target_dim=}")

    if tensor.stride()[-1] == 1:
        return tensor, True
    elif tensor.stride()[-2] == 1:
        return tensor, False
    else:
        return tensor.contiguous(), True


class pam_matmul_autograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, offset=None, approx_bwd=False, BMM=False):
        A, A_row_major = check_row_major(A, 3 if BMM else 2)
        B, B_row_major = check_row_major(B, 3 if BMM else 2)

        if offset is None:
            offset = 1.0  # Turns it off, equivalent to multiplying result with 1.0

        ctx.save_for_backward(A, B)
        ctx.approx_bwd = approx_bwd  # Can't save non-tensors with other method
        ctx.offset = offset
        ctx.BMM = BMM

        if BMM:
            return cuda_kernels.pam_bmm_fwd(A, B, A_row_major, B_row_major, offset)
        else:
            return cuda_kernels.pam_matmul_fwd(A, B, A_row_major, B_row_major, offset)

    @staticmethod
    def backward(ctx, delta_Y):
        A, B = ctx.saved_tensors
        offset = ctx.offset
        approx_bwd = ctx.approx_bwd
        if isinstance(approx_bwd, bool):
            approx_bwd = (approx_bwd, approx_bwd)
        approx_A, approx_B = approx_bwd
        BMM = ctx.BMM

        delta_Y, delta_Y_rm = check_row_major(delta_Y, 3 if BMM else 2)
        AT, AT_rm = check_row_major(A.mT, 3 if BMM else 2)
        BT, BT_rm = check_row_major(B.mT, 3 if BMM else 2)

        fwd = cuda_kernels.pam_bmm_fwd if BMM else cuda_kernels.pam_matmul_fwd
        bwd = cuda_kernels.pam_bmm_bwd if BMM else cuda_kernels.pam_matmul_bwd

        if approx_A:
            delta_A = fwd(delta_Y, BT, delta_Y_rm, BT_rm, offset)
        else:
            delta_L = True
            delta_A = bwd(delta_Y, BT, A, delta_Y_rm, BT_rm, delta_L, offset)

        if approx_B:
            delta_B = fwd(AT, delta_Y, AT_rm, delta_Y_rm, offset)
        else:
            delta_L = False
            delta_B = bwd(AT, delta_Y, B, AT_rm, delta_Y_rm, delta_L, offset)

        return delta_A, delta_B, None, None, None


def pam_matmul(A, B, *, offset=None, approx_bwd=False):
    return pam_matmul_autograd.apply(A, B, offset, approx_bwd, False)


def pam_bmm(A, B, *, offset=None, approx_bwd=False):
    return pam_matmul_autograd.apply(A, B, offset, approx_bwd, True)


class standard_matmul_autograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, BMM=False):
        A, A_rm = check_row_major(A, 3 if BMM else 2)
        B, B_rm = check_row_major(B, 3 if BMM else 2)

        ctx.save_for_backward(A, B)
        ctx.BMM = BMM

        fwd = cuda_kernels.standard_bmm if BMM else cuda_kernels.standard_matmul
        return fwd(A, B, A_rm, B_rm, 1.0)  # Offset (1.0) is ignored for standard_matmul


    @staticmethod
    def backward(ctx, delta_Y):
        A, B = ctx.saved_tensors
        BMM = ctx.BMM

        delta_Y, delta_Y_rm = check_row_major(delta_Y, 3 if BMM else 2)
        AT, AT_rm = check_row_major(A.mT, 3 if BMM else 2)
        BT, BT_rm = check_row_major(B.mT, 3 if BMM else 2)

        fwd = cuda_kernels.standard_bmm if BMM else cuda_kernels.standard_matmul
        delta_A = fwd(delta_Y, BT, delta_Y_rm, BT_rm, 1.0)
        delta_B = fwd(AT, delta_Y, AT_rm, delta_Y_rm, 1.0)
        return delta_A, delta_B, None


def standard_matmul(A, B, *, offset=None, approx_bwd=None, BMM=False):
    # Ignores extra unused arguments offset and approx_bwd
    return standard_matmul_autograd.apply(A, B, BMM)


class pa_exp_autograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, base, offset=None, approx_bwd=False):
        if offset is None:
            offset = 1.0  # Turns it off, equivalent to multiplying result with 1.0
        ctx.save_for_backward(X)
        ctx.approx_bwd = approx_bwd  # Can't save non-tensors with other method
        ctx.offset = offset
        ctx.base = base
        return cuda_kernels.pa_exp_fwd(X, base, offset)

    @staticmethod
    def backward(ctx, delta_Y):
        X, = ctx.saved_tensors
        base = ctx.base
        offset = ctx.offset
        approx_bwd = ctx.approx_bwd

        if approx_bwd:
            Y = cuda_kernels.pa_exp_fwd(X, base, offset)
            if base != math.e:
                ln_base = torch.full_like(X, math.log(base))
                delta_X = cuda_kernels.pam_fwd(Y, ln_base, offset)
                delta_X = cuda_kernels.pam_fwd(delta_X, delta_Y, offset)
            else:
                delta_X = cuda_kernels.pam_fwd(Y, delta_Y, offset)
        else:
            delta_X = cuda_kernels.pa_exp_bwd(X, delta_Y, base, offset)

        return delta_X, None, None, None


def pa_exp(X, *, base=math.e, offset=None, approx_bwd=False):
    return pa_exp_autograd.apply(X, base, offset, approx_bwd)


def pa_exp2(X, *, offset=None, approx_bwd=False):
    return pa_exp_autograd.apply(X, 2.0, offset, approx_bwd)


class pa_log_autograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, base, offset=None, approx_bwd=False):
        if offset is None:
            offset = 1.0  # Turns it off, equivalent to multiplying result with 1.0
        ctx.save_for_backward(X)
        ctx.approx_bwd = approx_bwd  # Can't save non-tensors with other method
        ctx.offset = offset
        ctx.base = base
        return cuda_kernels.pa_log_fwd(X, base, offset)

    @staticmethod
    def backward(ctx, delta_Y):
        X, = ctx.saved_tensors
        base = ctx.base
        offset = ctx.offset
        approx_bwd = ctx.approx_bwd

        if approx_bwd:
            delta_X = cuda_kernels.pad_fwd(delta_Y, X, offset)
            if base != math.e:
                ln_base = torch.full_like(X, math.log(base))
                delta_X = cuda_kernels.pad_fwd(delta_X, ln_base, offset)
        else:
            delta_X = cuda_kernels.pa_log_bwd(X, delta_Y, base, offset)

        return delta_X, None, None, None


def pa_log(X, *, base=math.e, offset=None, approx_bwd=False):
    return pa_log_autograd.apply(X, base, offset, approx_bwd)


def pa_log2(X, *, offset=None, approx_bwd=False):
    return pa_log_autograd.apply(X, 2.0, offset, approx_bwd)
