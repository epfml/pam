import math
import torch

if __package__ is None or __package__ == '':
    # uses current directory visibility (running as script / jupyter notebook)
    import native
    import cuda_bindings
    import utils
else:
    # uses current package visibility (running as a module)
    from . import native
    from . import cuda_bindings
    from . import utils


def mul(A, B, *, offset=None, approx_bwd=False, use_kernel=True):
    tA, tB = utils.to_tensor(A, B)
    if tA.is_cuda and use_kernel:
        out = cuda_bindings.pam(tA, tB, offset=offset, approx_bwd=approx_bwd)
    else:
        out = native.pam(tA, tB, offset=offset, approx_bwd=approx_bwd)
    return utils.demote_tensor(out, (A, B))


def div(A, B, *, offset=None, approx_bwd=False, use_kernel=True):
    tA, tB = utils.to_tensor(A, B)
    if tA.is_cuda and use_kernel:
        out = cuda_bindings.pad(tA, tB, offset=offset, approx_bwd=approx_bwd)
    else:
        out = native.pad(tA, tB, offset=offset, approx_bwd=approx_bwd)
    return utils.demote_tensor(out, (A, B))


def exp(A, base=math.e, *, offset=None, approx_bwd=False, use_kernel=True):
    tA, = utils.to_tensor(A)
    if tA.is_cuda and use_kernel:
        out = cuda_bindings.pa_exp(tA, base=base, offset=offset, approx_bwd=approx_bwd)
    else:
        out = native.pa_exp(tA, base=base, offset=offset, approx_bwd=approx_bwd)
    return utils.demote_tensor(out, (A,))


def exp2(A, *, offset=None, approx_bwd=False, use_kernel=True):
    tA, = utils.to_tensor(A)
    if tA.is_cuda and use_kernel:
        out = cuda_bindings.pa_exp2(tA, offset=offset, approx_bwd=approx_bwd)
    else:
        out = native.pa_exp2(tA, offset=offset, approx_bwd=approx_bwd)
    return utils.demote_tensor(out, (A,))


def log(A, base=math.e, *, offset=None, approx_bwd=False, use_kernel=True):
    tA, = utils.to_tensor(A)
    if tA.is_cuda and use_kernel:
        out = cuda_bindings.pa_log(tA, base=base, offset=offset, approx_bwd=approx_bwd)
    else:
        out = native.pa_log(tA, base=base, offset=offset, approx_bwd=approx_bwd)
    return utils.demote_tensor(out, (A,))


def log2(A, *, offset=None, approx_bwd=False, use_kernel=True):
    tA, = utils.to_tensor(A)
    if tA.is_cuda and use_kernel:
        out = cuda_bindings.pa_log2(tA, offset=offset, approx_bwd=approx_bwd)
    else:
        out = native.pa_log2(tA, offset=offset, approx_bwd=approx_bwd)
    return utils.demote_tensor(out, (A,))


def pow(A, B, *, offset=None, approx_bwd=False, use_kernel=True):
    tA, tB = utils.to_tensor(A, B)
    zero_mask = (tA == 0)
    kwargs = dict(offset=offset, approx_bwd=approx_bwd, use_kernel=use_kernel)
    out = torch.where(
        zero_mask,
        0,
        exp2(mul(log2(tA, **kwargs), tB, **kwargs), **kwargs)
    )
    return utils.demote_tensor(out, (A, B))


def matmul(A, B, *, offset=None, approx_bwd=False, use_kernel=True):
    if A.is_cuda and use_kernel:
        return cuda_bindings.pam_matmul(A, B, offset=offset, approx_bwd=approx_bwd)
    else:
        return native.pam_matmul(A, B, offset=offset, approx_bwd=approx_bwd)


def bmm(A, B, *, offset=None, approx_bwd=False, use_kernel=True):
    assert A.dim() == B.dim() == 3
    assert A.shape[0] == B.shape[0]
    assert A.shape[2] == B.shape[1]

    use_kernel = A.is_cuda and use_kernel
    if use_kernel:
        return cuda_bindings.pam_bmm(A, B, offset=offset, approx_bwd=approx_bwd)
    else:
        frags = []
        for A_sub, B_sub in zip(A, B):
            frags.append(matmul(
                A_sub, B_sub,
                offset=offset, approx_bwd=approx_bwd, use_kernel=use_kernel
            ))
        return torch.stack(frags)


def linear(X, W, bias=None, *, offset=None, approx_bwd=False, use_kernel=True):
    # Flatten leading dimensions if any like torch.nn.Linear
    X_shape = X.shape
    X = X.reshape(-1, X_shape[-1])
    out = matmul(X, W.T, offset=offset, approx_bwd=approx_bwd, use_kernel=use_kernel)
    out = out.reshape(*X_shape[:-1], out.shape[-1])
    if bias is not None:
        out = out + bias
    return out


def conv2d(
    X, W, bias=None, stride=1, padding=0, dilation=1, *,
    offset=None, approx_bwd=False, use_kernel=True,
):
    # https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html
    # https://github.com/pytorch/pytorch/issues/47990
    N, _, H_in, W_in = X.shape
    C_out, _, kH, kW = W.shape
    if isinstance(padding, str):
        assert kH % 2 == 1 and kW % 2 == 1
        if padding == 'same':
            padding = ((kH-1)//2, (kW-1)//2)
        elif padding == 'valid':
            padding = 0
        else:
            raise ValueError(f"Unknown {padding=}")
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(stride, int):
        stride = (stride, stride)
    H_out = (H_in + 2 * padding[0] - (kH - 1) - 1) // stride[0] + 1
    W_out = (W_in + 2 * padding[1] - (kW - 1) - 1) // stride[1] + 1

    # N, stacked_channels, spatial_locations
    X_unfolded = torch.nn.functional.unfold(
        X, (kH, kW),
        padding=padding, stride=stride, dilation=dilation
    )
    _, stacked_channels, spatial_locations = X_unfolded.shape

    # N, spatial_locations, stacked_channels
    X_unfolded_T = X_unfolded.transpose(1, 2).contiguous()

    Y_unfolded = linear(
        X_unfolded_T.view(-1, stacked_channels),
        W.view(W.size(0), -1),
        offset=offset, approx_bwd=approx_bwd, use_kernel=use_kernel,
    ).view(N, spatial_locations, C_out).transpose(1, 2)
    Y = Y_unfolded.view(N, C_out, H_out, W_out)

    if bias is not None:
        Y = Y + bias.view((1, -1, 1, 1))
    return Y


def conv2d_group(
    X, W, bias=None, stride=1, padding=0, dilation=1, groups=1, *,
    offset=None, approx_bwd=False, use_kernel=True,
):
    # Also supports groups=1 but may be slower than other implementation
    N, C, H_in, W_in = X.shape
    K, _, R, S = W.shape
    G = groups
    if isinstance(padding, str):
        if padding == 'same':
            assert R % 2 == 1 and S % 2 == 1
            padding = ((R-1)//2, (S-1)//2)
        elif padding == 'valid':
            padding = 0
        else:
            raise ValueError(f"Unknown {padding=}")
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(stride, int):
        stride = (stride, stride)
    H_out = (H_in + 2 * padding[0] - (R - 1) - 1) // stride[0] + 1
    W_out = (W_in + 2 * padding[1] - (S - 1) - 1) // stride[1] + 1

    X_fold = torch.nn.functional.unfold(
        X, kernel_size=(R, S),
        padding=padding, stride=stride, dilation=dilation
    )
    X_fold = X_fold.view(N, G, C//G * R * S, -1)
    X_fold = X_fold.permute(1, 2, 0, 3).reshape(G, C//G * R * S, -1)
    W_fold = W.view(G, K//G, C//G * R * S)
    Y = bmm(W_fold, X_fold, offset=offset, approx_bwd=approx_bwd, use_kernel=use_kernel)
    Y = Y.view(K, N, H_out, W_out).transpose(0, 1)

    if bias is not None:
        Y = Y + bias.view((1, -1, 1, 1))
    return Y


class Conv2d(torch.nn.Conv2d):
    def __init__(self, *args, offset=None, approx_bwd=False, use_kernel=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.offset = offset
        self.approx_bwd = approx_bwd
        self.use_kernel = use_kernel

    def forward(self, input):
        if self.groups != 1:
            return conv2d_group(
                input,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
                offset=self.offset,
                approx_bwd=self.approx_bwd,
                use_kernel=self.use_kernel,
            )
        else:
            return conv2d(
                input,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                offset=self.offset,
                approx_bwd=self.approx_bwd,
                use_kernel=self.use_kernel,
            )

    def _get_name(self):
        return "pam." + self.__class__.__name__

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        if self.offset is not None:
            s += ', offset={offset:0.5f}'
        s += ', approx_bwd={approx_bwd}'
        return s.format(**self.__dict__)


class Linear(torch.nn.Linear):
    def __init__(self, *args, offset=None, approx_bwd=False, use_kernel=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.offset = offset
        self.approx_bwd = approx_bwd
        self.use_kernel = use_kernel

    def forward(self, input):
        return linear(
            input,
            self.weight,
            self.bias,
            offset=self.offset,
            approx_bwd=self.approx_bwd,
            use_kernel=self.use_kernel,
        )

    def _get_name(self):
        return "pam." + self.__class__.__name__

    def extra_repr(self):
        s = 'in_features={in_features}, out_features={out_features}, '
        s += f'bias={self.bias is not None}, '
        if self.offset is not None:
            s += 'offset={offset:0.5f}, '
        s += 'approx_bwd={approx_bwd}'
        return s.format(**self.__dict__)


def mean(
    input, dim=None, keepdim=False, *, dtype=None,
    offset=None, approx_bwd=False, use_kernel=True
):
    if dim is None:
        dim = tuple(range(input.dim()))
    if isinstance(dim, int):
        dim = (dim,)

    num_elements = math.prod([input.shape[d] for d in dim])
    return div(
        torch.sum(input, dim=dim, keepdim=keepdim, dtype=dtype),
        num_elements,
        offset=offset, approx_bwd=approx_bwd, use_kernel=use_kernel,
    )


def softmax(x, dim=None, *, offset=None, approx_bwd=False, use_kernel=True):
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html
    # TODO: Autograd version implementing the bwd formula manually?
    # Visually approx_bwd gives a good approximation of torch derivative
    pam_kwargs = dict(offset=offset, approx_bwd=approx_bwd, use_kernel=use_kernel)
    base_constant = math.log2(math.e)
    x_scaled = torch.where(
        # PAM mul gives NaN for infinite inputs, transformer uses -inf for masking
        x == float('-inf'),
        float('-inf'),
        mul(x, base_constant, **pam_kwargs)
    )
    with torch.no_grad():
        shift = torch.floor(torch.max(x_scaled))
    x_adj = x_scaled - shift
    expx = torch.where(
        # PAM mul gives NaN for infinite inputs, transformer uses -inf for masking
        x == float('-inf'),
        0,
        exp2(x_adj, **pam_kwargs),
    )
    denom = torch.sum(expx, dim=dim, keepdim=True)
    out = div(expx, denom, **pam_kwargs)
    return out


def logsumexp(x, dim, keepdim=False, *, offset=None, approx_bwd=False, use_kernel=True):
    pam_kwargs = dict(offset=offset, approx_bwd=approx_bwd, use_kernel=use_kernel)
    assert x.isfinite().all()
    base_constant = math.log2(math.e)
    x_scaled = mul(x, base_constant, **pam_kwargs)
    with torch.no_grad():
        shift = torch.floor(torch.max(x_scaled))
    x_adj = x_scaled - shift
    expx = exp2(x_adj, **pam_kwargs)
    sum_expx = torch.sum(expx, dim=dim, keepdim=keepdim)
    out = div(shift + log2(sum_expx), base_constant, **pam_kwargs)
    return out


def log_softmax(x, dim=None, *, offset=None, approx_bwd=False, use_kernel=True):
    pam_kwargs = dict(offset=offset, approx_bwd=approx_bwd, use_kernel=use_kernel)
    return x - logsumexp(x, dim=dim, keepdim=True, **pam_kwargs)


def layer_norm(
    x, normalized_shape, weight=None, bias=None, eps=1e-05,
    *, offset=None, approx_bwd=False, use_kernel=True
):
    pam_kwargs = dict(offset=offset, approx_bwd=approx_bwd, use_kernel=use_kernel)
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)
    assert x.shape[-len(normalized_shape):] == normalized_shape
    num_elements = math.prod(normalized_shape)

    x_flat = x.reshape((-1, num_elements))
    mu = mean(x_flat, dim=-1, keepdim=True, **pam_kwargs)
    x_center = x_flat - mu
    var = mean(
        mul(x_center, x_center, **pam_kwargs),
        dim=-1,
        keepdim=True,
        **pam_kwargs
    )
    denom = pow(var + eps, 0.5, **pam_kwargs)
    x_normalized = div(x_center, denom, **pam_kwargs)
    out = x_normalized.reshape_as(x)

    if weight is not None:
        out = mul(out, weight, **pam_kwargs)
    if bias is not None:
        out = out + bias

    return out


class LayerNorm(torch.nn.LayerNorm):
    def __init__(self, *args, offset=None, approx_bwd=False, use_kernel=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.offset = offset
        self.approx_bwd = approx_bwd
        self.use_kernel = use_kernel

    def forward(self, input):
        return layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps,
            offset=self.offset, approx_bwd=self.approx_bwd, use_kernel=self.use_kernel,
        )

    def _get_name(self):
        return "pam." + self.__class__.__name__

    def extra_repr(self):
        s = super().extra_repr()
        if self.offset is not None:
            s += ', offset={offset:0.5f}'
        s += ', approx_bwd={approx_bwd}'
        if self.use_kernel is not None:
            s += ', use_kernel={use_kernel}'
        return s.format(**self.__dict__)
