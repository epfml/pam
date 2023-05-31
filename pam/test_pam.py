import torch

from . import cuda_bindings
from . import native
from . import pam_ops

# Force the use of float32 for comparison tests
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def pam_fwd_test(method):
    # Test PAM against handcrafted tests for normal input numbers
    ABC = torch.tensor([
        [1.0, 1.0, 1.0],  # Multiplication by 1 should be exact
        [1.2345, 1.0, 1.2345],
        [1.2345, 32.0, 32.0*1.2345],  # Multiplication by 32 should be exact
        [-1.2345, 32.0, -32.0*1.2345],
        [1.2345, -32.0, -32.0*1.2345],
        [-1.2345, -32.0, 32.0*1.2345],
        [1.0, 0.0, 0.0],  # Multiplication by 0 should give 0
        [0.0, -1.0, -0.0],
        [1.5, 1.5, 2.0],  # PAM(1.5, 1.5) = 2.0
        [1.3, -1.3, -1.6],  # PAM(1.3, -1.3) = -1.6
        [4.5, 3.0, 8*(1+(4.5-4)/4+(3.0-2)/2)],  # Mantissa formula, no overflow
        [-6.0, 3.0, -16.0],  # PAM(-6, 3.0) = -16.0
        [-3.5, -3.5, 8*(1+(3.5-2)/2+(3.5-2)/2-1)],  # Mantissa formula, with overflow (E++, Mf-=1)
        [1e-30, 1e-30, 0.0],  # Should underflow to 0.0
        [1e-30, -1e-30, -0.0],  # Should underflow to 0.0
    ], device='cuda')
    C_hat = method(ABC[:,0].contiguous(), ABC[:,1].contiguous())
    assert torch.allclose(ABC[:,2], C_hat)


def pam_fwd_test_inf_nan(method):
    # Test PAM against handcrafted tests for Inf/NaN
    AB = torch.tensor([
        [1e30, 1e30],  # Overflow to inf
        [1e30, -1e30],  # Overflow to -inf
        [float('-inf'), 1e-20],  # Should remain inf, not become finite
        [float('nan'), -1e-20],  # Should remain nan, not become finite
        [float('inf'), 0.0],  # Should be nan
        [float('nan'), 0.0],  # Should remain nan
    ], device='cuda')
    C_hat = method(AB[:,0].contiguous(), AB[:,1].contiguous())
    assert (~C_hat.isfinite()).all()


def pam_bwd_exact_test(method):
    # Test the exact bwd
    N = 128
    A_exp = torch.randint(-30, 30, (N,))
    A_mantissa = torch.rand(N)
    A = (2.0 ** A_exp) * (1 + A_mantissa)
    B_exp = torch.randint(-30, 30, (N,))
    B_mantissa = torch.rand(N)
    B = (2.0 ** B_exp) * (1 + B_mantissa)
    
    A1 = A.cuda().requires_grad_()
    B1 = B.cuda().requires_grad_()
    A2 = torch.clone(A1.detach()).requires_grad_()
    B2 = torch.clone(B1.detach()).requires_grad_()

    C1 = method(A1, B1)
    delta_C = torch.arange(C1.numel()).float().reshape_as(C1).cuda()
    C1.backward(delta_C)

    # Built-in bwd using torch autograd
    C2 = native.pam_fwd(A2, B2)
    C2.backward(delta_C)

    assert torch.allclose(C1, C2)
    assert torch.allclose(A1.grad, A2.grad)
    assert torch.allclose(B1.grad, B2.grad)


def test_pam_native_fwd():
    pam_fwd_test(native.pam)
    pam_fwd_test_inf_nan(native.pam)


def test_pam_native_bwd():
    pam_bwd_exact_test(native.pam)


def test_pam_cuda_fwd():
    pam_fwd_test(cuda_bindings.pam)
    pam_fwd_test_inf_nan(cuda_bindings.pam)


def test_pam_cuda_bwd():
    pam_bwd_exact_test(cuda_bindings.pam)


def test_pam_matmul():
    # Compare the native and cuda versions of matmul for fwd and bwd
    A1 = (torch.arange(20)**3 % 7 - 2.1).reshape((5, 4)).cuda().requires_grad_()
    B1 = (torch.arange(12)**2 % 5 - 2.1).reshape((3, 4)).T.cuda().requires_grad_()
    A2 = torch.clone(A1.detach()).requires_grad_()
    B2 = torch.clone(B1.detach()).requires_grad_()
    A3 = torch.clone(A1.detach()).requires_grad_()
    B3 = torch.clone(B1.detach()).requires_grad_()


    C1 = native.pam_matmul(A1, B1)
    C1.sum().backward()
    C2 = cuda_bindings.pam_matmul(A2, B2)
    C2.sum().backward()
    C3 = native.pam_matmul_fwd(A3, B3)
    C3.sum().backward()

    assert torch.allclose(C1, C2)
    assert torch.allclose(A1.grad, A2.grad)
    assert torch.allclose(B1.grad, B2.grad)

    assert torch.allclose(C1, C3)
    assert torch.allclose(A1.grad, A3.grad)
    assert torch.allclose(B1.grad, B3.grad)

    # # Only works if points are far from discontinuities
    # assert torch.autograd.gradcheck(
    #     native.pam_matmul,
    #     (A1.double(), B1.double()),
    #     eps=1e-3, atol=1e-01, rtol=0.1
    # )


def test_pam_matmul_bwd_approx():
    # Test that pam_matmul with approx_bwd=True works as expected
    A1 = (torch.arange(20)**3 % 7 - 2.0).reshape((5, 4)).cuda().requires_grad_()
    B1 = (torch.arange(12)**2 % 5 - 2.0).reshape((3, 4)).T.cuda().requires_grad_()
    A2 = torch.clone(A1.detach()).requires_grad_()
    B2 = torch.clone(B1.detach()).requires_grad_()

    C1 = native.pam_matmul(A1, B1, approx_bwd=True)
    delta_C = torch.arange(C1.numel()).float().reshape_as(C1).cuda()
    C1.backward(delta_C)
    C2 = cuda_bindings.pam_matmul(A2, B2, approx_bwd=True)
    C2.backward(delta_C)

    # Analytical 
    A3_grad = native.pam_matmul(delta_C, B1.T)
    B3_grad = native.pam_matmul(A1.T, delta_C)

    assert torch.allclose(C1, C2)
    assert torch.allclose(A1.grad, A2.grad)
    assert torch.allclose(B1.grad, B2.grad)
    assert torch.allclose(A1.grad, A3_grad)
    assert torch.allclose(B1.grad, B3_grad)


def test_pam_conv2d():
    # Test that a conv with a kernel with all powers of 2 matches standard conv
    N, C, H, W = (11, 13, 15, 17)
    K, C, RH, RW = (19, C, 3, 5)
    # X1 with limited mantissa size to avoid numerical error in comparison
    X1 = (torch.randint(-1024, 1024, (N, C, H, W), device='cuda') / 64.0).requires_grad_()
    # W1 as powers of 2 to have a case where PAM = MUL
    W1 = (2.0**torch.randint(-8, 8, (K, C, RH, RW))).cuda().requires_grad_()

    X2 = torch.clone(X1.detach()).requires_grad_()
    W2 = torch.clone(W1.detach()).requires_grad_()

    Y1 = pam_ops.conv2d(X1, W1, approx_bwd=True)  # W1 all powers of 2, should match exactly
    Y1.sum().backward()  # Deltas are all powers of 2, should match exactly
    Y2 = torch.nn.functional.conv2d(X2, W2)
    Y2.sum().backward()

    assert torch.allclose(Y1, Y2)
    assert torch.allclose(X1.grad, X2.grad)
    assert torch.allclose(W1.grad, W2.grad)


def test_pam_group_conv2d():
    # Test that a conv with a kernel with all powers of 2 matches standard conv
    N, C, H, W = (11, 16, 15, 17)
    K, _, RH, RW = (128, C, 3, 5)
    for G in [1, 8]:
        # X1 with limited mantissa size to avoid numerical error in comparison
        X1 = (torch.randint(-8, 8+1, (N, C, H, W), device='cuda') / 4.0).requires_grad_()
        # W1 as powers of 2 to have a case where PAM = MUL
        W1 = (2.0**torch.randint(-2, 2+1, (K, C // G, RH, RW))).cuda().requires_grad_()

        X2 = torch.clone(X1.detach()).requires_grad_()
        W2 = torch.clone(W1.detach()).requires_grad_()

        Y1 = pam_ops.conv2d_group(X1, W1, groups=G, approx_bwd=True)  # W1 all powers of 2, should match exactly
        Y1.sum().backward()  # Deltas are all powers of 2, should match exactly
        Y2 = torch.nn.functional.conv2d(X2, W2, groups=G)
        Y2.sum().backward()

        assert torch.allclose(Y1, Y2)
        assert torch.allclose(X1.grad, X2.grad)
        assert torch.allclose(W1.grad, W2.grad)


def test_pam_bmm():
    # Test that BMM with a matrices of all powers of 2 matches standard BMM
    D1, D2, D3 = (11, 13, 15)
    D4 = 19

    # X1 with limited mantissa size to avoid numerical error in comparison
    X1 = (torch.randint(-128, 128, (D1, D2, D3), device='cuda') / 64.0).requires_grad_()
    # W1 as powers of 2 to have a case where PAM = MUL
    W1 = (2.0**torch.randint(-8, 8, (D1, D3, D4))).cuda().requires_grad_()

    X2 = torch.clone(X1.detach()).requires_grad_()
    W2 = torch.clone(W1.detach()).requires_grad_()

    Y1 = pam_ops.bmm(X1, W1, approx_bwd=True)  # W1 all powers of 2, should match exactly
    Y1.sum().backward()  # Deltas are all powers of 2, should match exactly
    Y2 = torch.bmm(X2, W2)
    Y2.sum().backward()

    assert torch.allclose(Y1, Y2)
    assert torch.allclose(X1.grad, X2.grad)
    assert torch.allclose(W1.grad, W2.grad)


def test_pam_bmm_bwd():
    # Compare native looped BMM with kernel for approx_bwd=False

    # Test that BMM with a matrices of all powers of 2 matches standard BMM
    D1, D2, D3 = (11, 13, 15)
    D4 = 19

    # X1 and W1 with limited mantissa size to avoid numerical error in comparison
    X1 = (torch.randint(-1024, 1024, (D1, D2, D3), device='cuda') / 32.0).requires_grad_()
    W1 = (torch.randint(-1024, 1024, (D1, D3, D4), device='cuda') / 32.0).requires_grad_()
    X2 = torch.clone(X1.detach()).requires_grad_()
    W2 = torch.clone(W1.detach()).requires_grad_()

    Y1 = pam_ops.bmm(X1, W1, approx_bwd=False, use_kernel=False)
    Y1.sum().backward()
    Y2 = pam_ops.bmm(X2, W2, approx_bwd=False, use_kernel=True)
    Y2.sum().backward()

    assert torch.allclose(Y1, Y2)
    assert torch.allclose(X1.grad, X2.grad)
    assert torch.allclose(W1.grad, W2.grad)
