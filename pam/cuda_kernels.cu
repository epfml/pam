#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <vector>
#include <iostream>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x) //; CHECK_CONTIGUOUS(x)

///////////////////////////////////////////////////////////////////////////////////////////////////
// Scalar PAM functions
///////////////////////////////////////////////////////////////////////////////////////////////////
inline __device__ float pam(const float& a, const float& b, const int& offset){
    // Compute PAM(a, b) with the given offset
    int ai = *(int*)&a;
    int bi = *(int*)&b;
    int ci = ai + bi - (0x3f800000 - offset);

    // Use built-in hardware support for float32 multiplication to handle
    // boundary cases and overflows efficiently on GPUs instead of doing
    // an expensive series of bit checks to cover all situations
    float c_true = a * b;
    c_true = c_true < 0 ? -c_true : c_true;  // abs(c_true)
    if(c_true < 1e-37){
        // Number close to denormal or potential underflow, flush to zero
        return 0.0;
    } else if(c_true < 1e37){
        // Number in normal range, return the PAM result
        return *(float*)&ci;
    } else {
        // Number is close to or has already overflowed to INF or -INF
        // Number could also be NAN (previous two comparisons will be false)
        // Return NAN for all these cases (INF, -INF, NAN)
        return NAN;
    }
}

inline __device__ float pam_bwd(
    const float& a, const float& b, const float& delta_c,
    const int& offset
){
    // Compute the exact derivative of PAM with a given offset w.r.t. a
    int ai = *(int*)&a;
    int bi = *(int*)&b;

    const int MANTISSA_MASK = 0x007FFFFF;

    // Slope without accounting for delta_c as float32 bits
    // Equal to the exponent of b with a potential shift of +1
    int si = ~MANTISSA_MASK & (bi + (MANTISSA_MASK & ai) + offset);
    float s = *(float*)&si;

    // This multiplication could be performed exactly with PAM since s is a power of 2
    // However using a float multiplication handles underflow/overflow/NAN/INF for us efficiently. 
    return delta_c * s;
}

inline __device__ float pad(const float& a, const float& b, const int& offset){
    // Compute PAD(a, b) with the given offset
    int ai = *(int*)&a;
    int bi = *(int*)&b;
    int ci = ai - bi + (0x3f800000 - offset);  // Need to add the exponent bias again

    // Use built-in hardware support for float32 division to handle
    // boundary cases and overflows efficiently on GPUs instead of doing
    // an expensive series of bit checks to cover all situations
    float c_true = a / b;
    c_true = c_true < 0 ? -c_true : c_true;  // abs(c_true)
    if(c_true < 1e-37){
        // Number close to denormal or potential underflow, flush to zero
        return 0.0;
    } else if(c_true < 1e37){
        // Number in normal range, return the PAD result
        return *(float*)&ci;
    } else {
        // Number is close to or has already overflowed to INF or -INF
        // Number could also be NAN (previous two comparisons will be false)
        // Return NAN for all these cases (INF, -INF, NAN)
        return NAN;
    }
}

inline __device__ float pad_bwd(
    const float& a, const float& b, const float& delta_c,
    const int& offset
){
    // Compute the exact derivative of PAM with a given offset w.r.t. b
    int bi = *(int*)&b;

    const int MANTISSA_MASK = 0x007FFFFF;
    const int EXPONENT_MASK = 0x7F800000;

    // Slope without accounting for delta_c as float32 bits
    // Equal to 2^(E_a - 2*E_b) with a potential shift of -1
    float out = pad(a, b, offset);
    int out_masked_i = *(int*)&out & (~MANTISSA_MASK);  // Has sign information
    float out_masked = *(float*)&out_masked_i;

    int b_masked_i = bi & EXPONENT_MASK;
    float b_masked = *(float*)&b_masked_i;  // 2^E_b

    // This multiplication could be performed exactly with PAM
    // out_masked and b_masked are powers of 2
    // Float multiplication handles underflow/overflow/NAN/INF for us in hardware
    return -(out_masked/b_masked) * delta_c;
}

inline __device__ float pa_exp2(const float& x){
    // Multiplication can also be done via PAM exactly (power of 2)
    return pow(2, floor(x)) * (1 + (x - floor(x)));
}

inline __device__ float pa_exp2_bwd(const float& x, const float& dy){
    // Multiplication can also be done via PAM exactly
    float slope = pa_exp2(x);
    const int MANTISSA_MASK = 0x007FFFFF;
    int slope_bits = *(int*)&slope;
    slope_bits = slope_bits & (~MANTISSA_MASK);
    slope = *(float*)&slope_bits;
    return slope * dy; // Slope is a power of 2, PAM exact
}

inline __device__ float pa_log2(const float& x){
    // Division can be done via PAM exactly (power of 2)
    float exponent = floor(log2(x));
    float mantissa = (x - pow(2, exponent)) / pow(2, exponent);
    return exponent + mantissa;
}

inline __device__ float pa_log2_bwd(const float& x, const float& dy){
    // Multiplication can be done via PAM exactly (power of 2)
    float exponent = floor(log2(x));
    return pow(2, -exponent) * dy;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Elementwise PAM kernels
///////////////////////////////////////////////////////////////////////////////////////////////////
template <typename scalar_t>
__global__ void elementwise_pam_kernel(
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> A,
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> C,
    int offset
){
    // Assumes A, B, C are one dimensional tensors of the same size
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < A.size(0)){
        C[idx] = pam(A[idx], B[idx], offset);
    }
}

torch::Tensor elementwise_pam_caller(
    torch::Tensor A,
    torch::Tensor B,
    float offset
){
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    auto Af = A.flatten();
    auto Bf = B.flatten();
    auto Cf = torch::empty_like(Af);
    assert(Af.size(0) == Bf.size(0));

    int offset_int = *(int*)&offset - 0x3f800000;    

    const int block_dim = 256;
    const int grid_dim = (Af.size(0) + block_dim - 1) / block_dim;

    AT_DISPATCH_FLOATING_TYPES(Af.type(), "elementwise_pam_kernel", ([&] {
        elementwise_pam_kernel<scalar_t><<<grid_dim, block_dim>>>(
            Af.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            Bf.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            Cf.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            offset_int
        );
    }));

    return Cf.reshape_as(A);
}

template <typename scalar_t>
__global__ void elementwise_pam_bwd_kernel(
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> A,
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> B,
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> delta_C,
    torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> delta_A,
    int offset
){
    // Assumes A, B, delta_C, delta_A are one dimensional tensors of the same size
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < A.size(0)){
        delta_A[idx] = pam_bwd(A[idx], B[idx], delta_C[idx], offset);
    }
}

torch::Tensor elementwise_pam_bwd_caller(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor delta_C,
    float offset
){
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(delta_C);
    assert(A.is_same_size(B) && A.is_same_size(delta_C));

    auto Af = A.flatten();
    auto Bf = B.flatten();
    auto delta_Cf = delta_C.flatten();
    auto delta_Af = torch::empty_like(Af);

    int offset_int = *(int*)&offset - 0x3f800000;    

    const int block_dim = 256;
    const int grid_dim = (Af.size(0) + block_dim - 1) / block_dim;

    AT_DISPATCH_FLOATING_TYPES(Af.type(), "elementwise_pam_bwd_kernel", ([&] {
        elementwise_pam_bwd_kernel<scalar_t><<<grid_dim, block_dim>>>(
            Af.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            Bf.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            delta_Cf.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            delta_Af.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            offset_int
        );
    }));

    return delta_Af.reshape_as(A);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Elementwise PAD kernels (almost identical to the PAM ones, except diff w.r.t. b)
///////////////////////////////////////////////////////////////////////////////////////////////////
template <typename scalar_t>
__global__ void elementwise_pad_kernel(
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> A,
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> C,
    int offset
){
    // Assumes A, B, C are one dimensional tensors of the same size
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < A.size(0)){
        C[idx] = pad(A[idx], B[idx], offset);
    }
}

torch::Tensor elementwise_pad_caller(
    torch::Tensor A,
    torch::Tensor B,
    float offset
){
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    auto Af = A.flatten();
    auto Bf = B.flatten();
    auto Cf = torch::empty_like(Af);
    assert(Af.size(0) == Bf.size(0));

    int offset_int = *(int*)&offset - 0x3f800000;    

    const int block_dim = 256;
    const int grid_dim = (Af.size(0) + block_dim - 1) / block_dim;

    AT_DISPATCH_FLOATING_TYPES(Af.type(), "elementwise_pad_kernel", ([&] {
        elementwise_pad_kernel<scalar_t><<<grid_dim, block_dim>>>(
            Af.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            Bf.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            Cf.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            offset_int
        );
    }));

    return Cf.reshape_as(A);
}

template <typename scalar_t>
__global__ void elementwise_pad_bwd_kernel(
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> A,
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> B,
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> delta_C,
    torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> delta_B,
    int offset
){
    // Assumes A, B, delta_C, delta_B are one dimensional tensors of the same size
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < B.size(0)){
        delta_B[idx] = pad_bwd(A[idx], B[idx], delta_C[idx], offset);
    }
}

torch::Tensor elementwise_pad_bwd_caller(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor delta_C,
    float offset
){
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(delta_C);
    assert(A.is_same_size(B) && A.is_same_size(delta_C));

    auto Af = A.flatten();
    auto Bf = B.flatten();
    auto delta_Cf = delta_C.flatten();
    auto delta_Bf = torch::empty_like(Af);

    int offset_int = *(int*)&offset - 0x3f800000;    

    const int block_dim = 256;
    const int grid_dim = (Af.size(0) + block_dim - 1) / block_dim;

    AT_DISPATCH_FLOATING_TYPES(Af.type(), "elementwise_pad_bwd_kernel", ([&] {
        elementwise_pad_bwd_kernel<scalar_t><<<grid_dim, block_dim>>>(
            Af.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            Bf.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            delta_Cf.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            delta_Bf.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            offset_int
        );
    }));

    return delta_Bf.reshape_as(B);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// Elementwise PAM exp2 kernels
///////////////////////////////////////////////////////////////////////////////////////////////////
template <typename scalar_t>
__global__ void elementwise_pa_exp_kernel(
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> X,
    torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> Y,
    float base,
    int offset
){
    // Computes Y=base^X via piecewise affine exp2 and log2
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < X.size(0)){
        Y[idx] = pa_exp2(pam(X[idx], log2(base), offset));
    }
}

torch::Tensor elementwise_pa_exp_caller(
    torch::Tensor X,
    float base,
    float offset
){
    CHECK_INPUT(X);

    auto Xf = X.flatten();
    auto Yf = torch::empty_like(Xf);

    int offset_int = *(int*)&offset - 0x3f800000;

    const int block_dim = 256;
    const int grid_dim = (Xf.size(0) + block_dim - 1) / block_dim;

    AT_DISPATCH_FLOATING_TYPES(Xf.type(), "elementwise_pa_exp_kernel", ([&] {
        elementwise_pa_exp_kernel<scalar_t><<<grid_dim, block_dim>>>(
            Xf.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            Yf.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            base,
            offset_int
        );
    }));

    return Yf.reshape_as(X);
}

template <typename scalar_t>
__global__ void elementwise_pa_exp_bwd_kernel(
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> X,
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> dY,
    torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> dX,
    float base,
    int offset
){
    // Computes the derivative of Y=base^X via piecewise affine exp2 and log2
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < X.size(0)){
        dX[idx] = pam_bwd(
            X[idx],
            log2(base),
            pa_exp2_bwd(pam(X[idx], log2(base), offset), dY[idx]),
            offset
        );
    }
}

torch::Tensor elementwise_pa_exp_bwd_caller(
    torch::Tensor X,
    torch::Tensor dY,
    float base,
    float offset
){
    CHECK_INPUT(X);

    auto Xf = X.flatten();
    auto dYf = dY.flatten();
    auto dXf = torch::empty_like(Xf);

    int offset_int = *(int*)&offset - 0x3f800000;

    const int block_dim = 256;
    const int grid_dim = (Xf.size(0) + block_dim - 1) / block_dim;

    AT_DISPATCH_FLOATING_TYPES(Xf.type(), "elementwise_pa_exp_bwd_kernel", ([&] {
        elementwise_pa_exp_bwd_kernel<scalar_t><<<grid_dim, block_dim>>>(
            Xf.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            dYf.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            dXf.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            base,
            offset_int
        );
    }));

    return dXf.reshape_as(X);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// Elementwise PAM log2 kernels
///////////////////////////////////////////////////////////////////////////////////////////////////
template <typename scalar_t>
__global__ void elementwise_pa_log_kernel(
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> X,
    torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> Y,
    float base,
    int offset
){
    // Computes Y=log_base(Y) via piecewise affine log2
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < X.size(0)){
        Y[idx] = pad(pa_log2(X[idx]), log2(base), offset);
    }
}

torch::Tensor elementwise_pa_log_caller(
    torch::Tensor X,
    float base,
    float offset
){
    CHECK_INPUT(X);

    auto Xf = X.flatten();
    auto Yf = torch::empty_like(Xf);

    int offset_int = *(int*)&offset - 0x3f800000;

    const int block_dim = 256;
    const int grid_dim = (Xf.size(0) + block_dim - 1) / block_dim;

    AT_DISPATCH_FLOATING_TYPES(Xf.type(), "elementwise_pa_log_kernel", ([&] {
        elementwise_pa_log_kernel<scalar_t><<<grid_dim, block_dim>>>(
            Xf.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            Yf.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            base,
            offset_int
        );
    }));

    return Yf.reshape_as(X);
}

template <typename scalar_t>
__global__ void elementwise_pa_log_bwd_kernel(
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> X,
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> dY,
    torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> dX,
    float base,
    int offset
){
    // Computes the derivative of Y=log_base(Y) via piecewise affine log2
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < X.size(0)){
        dX[idx] = pa_log2_bwd(
            X[idx],
            pam_bwd(pa_log2(X[idx]), pad(1, log2(base), offset), dY[idx], 0)  // Verify this formula
        );
    }
}

torch::Tensor elementwise_pa_log_bwd_caller(
    torch::Tensor X,
    torch::Tensor dY,
    float base,
    float offset
){
    CHECK_INPUT(X);

    auto Xf = X.flatten();
    auto dYf = dY.flatten();
    auto dXf = torch::empty_like(Xf);

    int offset_int = *(int*)&offset - 0x3f800000;

    const int block_dim = 256;
    const int grid_dim = (Xf.size(0) + block_dim - 1) / block_dim;

    AT_DISPATCH_FLOATING_TYPES(Xf.type(), "elementwise_pa_log_bwd_kernel", ([&] {
        elementwise_pa_log_bwd_kernel<scalar_t><<<grid_dim, block_dim>>>(
            Xf.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            dYf.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            dXf.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            base,
            offset_int
        );
    }));

    return dXf.reshape_as(X);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
// Basic Shared Memory Kernel for matmul (maybe 50% speed for standard mul)
///////////////////////////////////////////////////////////////////////////////////////////////////
inline __device__ float standard_mul(const float& a, const float& b, const int& not_used){
    return a * b;
}

// Load a 32x32 tile, into a 32x33 shared memory tile
template <typename scalar_t, int BLOCK_DIM, bool ROW_MAJOR, int MAT_DIM>
inline __device__ void load_tile(
    float* tile,
    const torch::PackedTensorAccessor32<scalar_t,MAT_DIM,torch::RestrictPtrTraits>& mat,
    int origin_row,
    int origin_col
){
    constexpr bool BMM = (MAT_DIM == 3);  // Batch matrix multiply
    const int NUM_WARPS = BLOCK_DIM * BLOCK_DIM / 32;
    const int steps = 32 / NUM_WARPS;
    int linear_thread_idx = blockDim.x * threadIdx.y + threadIdx.x;
    int warp_idx = linear_thread_idx / 32;
    #pragma unroll
    for(int step=0; step<steps; ++step){
        if(ROW_MAJOR){
            int i1 = warp_idx + step * NUM_WARPS; // row for row major
            int i2 = linear_thread_idx % 32; // colum for row major
            int mat_row = origin_row+i1;
            int mat_col = origin_col+i2;
            if(mat_row < mat.size(0+BMM) && mat_col < mat.size(1+BMM)){
                if constexpr(BMM){
                    tile[i1 * 33 + i2] = mat[blockIdx.z][mat_row][mat_col];
                }else{
                    tile[i1 * 33 + i2] = mat[mat_row][mat_col];
                }
            }else{
                tile[i1 * 33 + i2] = 0.0f;
            }
        } else {
            int i1 = warp_idx + step * NUM_WARPS; // row for row major
            int i2 = linear_thread_idx % 32; // colum for row major
            int mat_row = origin_row+i2;
            int mat_col = origin_col+i1;
            if(mat_row < mat.size(0+BMM) && mat_col < mat.size(1+BMM)){
                if constexpr(BMM){
                    tile[i2 * 33 + i1] = mat[blockIdx.z][mat_row][mat_col];
                }else{
                    tile[i2 * 33 + i1] = mat[mat_row][mat_col];
                }
            }else{
                tile[i2 * 33 + i1] = 0.0f;
            }
        }
    }
}


template <
    typename scalar_t,
    int BLOCK_DIM,
    bool ARM, bool BRM,
    float MUL(const float&, const float&, const int&),
    int MAT_DIM
>
__global__ void matmul_kernel(
    const torch::PackedTensorAccessor32<scalar_t,MAT_DIM,torch::RestrictPtrTraits> A,
    const torch::PackedTensorAccessor32<scalar_t,MAT_DIM,torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor32<scalar_t,MAT_DIM,torch::RestrictPtrTraits> C,
    int offset
){
    // Compute C = A @ B
    // Assume A is DL x DI, B is DI x DR
    // Each block computes an area of 32x32 in C and has BLOCK_DIM**2 threads
    // ARM/BRM mean that A/B is stored as row-major (last dim contiguous)
    // If false we assume column major but the same dimensions for the matrix
    // If BMM, perform batch matrix multiplication treating the first dim as the
    // batch dimension and the last two as before
    constexpr bool BMM = (MAT_DIM == 3); // Batch matrix multiply

    extern __shared__ float s[];
    float* A_tile = &s[0];
    float* B_tile = &s[33*32];

    const int TTD = 32 / BLOCK_DIM; // Thread tile dimensions
    float C_array[TTD*TTD] = {0.0f}; // Requires {0.0f} to be zero initialized
    float A_array[TTD];
    float B_array[TTD];

    const int BASE_OUT_ROW = blockIdx.y * 32;
    const int BASE_OUT_COL = blockIdx.x * 32;

    for(int step_idx=0; step_idx<(A.size(1+BMM) + 32 - 1)/32; ++step_idx){
        load_tile<scalar_t, BLOCK_DIM, ARM, MAT_DIM>(A_tile, A, BASE_OUT_ROW, 32*step_idx);
        load_tile<scalar_t, BLOCK_DIM, BRM, MAT_DIM>(B_tile, B, 32*step_idx, BASE_OUT_COL);
        __syncthreads();
        for(int inner_idx=0; inner_idx<32; ++inner_idx){
            // Load A_array, B_array
            #pragma unroll
            for(int a_idx=0; a_idx<TTD; ++a_idx){
                A_array[a_idx] = A_tile[(threadIdx.y + BLOCK_DIM * a_idx) * 33 + inner_idx];
                B_array[a_idx] = B_tile[inner_idx * 33 + (threadIdx.x + BLOCK_DIM * a_idx)];
            }
            // Compute elements
            #pragma unroll
            for(int a_idx=0; a_idx<TTD; ++a_idx){
                #pragma unroll
                for(int b_idx=0; b_idx<TTD; ++b_idx){
                    C_array[a_idx*TTD + b_idx] += MUL(A_array[a_idx], B_array[b_idx], offset);
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for(int a_idx=0; a_idx<TTD; ++a_idx){
        #pragma unroll
        for(int b_idx=0; b_idx<TTD; ++b_idx){
            int out_row = BASE_OUT_ROW + threadIdx.y + a_idx * BLOCK_DIM;
            int out_col = BASE_OUT_COL + threadIdx.x + b_idx * BLOCK_DIM;
            if(out_row < C.size(0+BMM) && out_col < C.size(1+BMM)){
                if constexpr(BMM){
                    C[blockIdx.z][out_row][out_col] = C_array[a_idx*TTD + b_idx];
                }else{
                    C[out_row][out_col] = C_array[a_idx*TTD + b_idx];
                }
            }
        }
    }
}

template<
    bool A_ROW_MAJOR, bool B_ROW_MAJOR,
    float MUL(const float&, const float&, const int&),
    bool BMM
>
torch::Tensor matmul_caller_body(
    torch::Tensor A,
    torch::Tensor B,
    int offset
){
    CHECK_CUDA(A);
    CHECK_CUDA(B);
    assert(A.size(1+BMM) == B.size(0+BMM));

    auto Y_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor Y;
    if constexpr(BMM){
        assert(A.size(0) == B.size(0));
        Y = torch::empty({A.size(0), A.size(1), B.size(2)}, Y_options);
    }else{
        Y = torch::empty({A.size(0), B.size(1)}, Y_options);
    }

    const int BLOCK_DIM = 16;
    const int OUT_TILE_DIM = 32; // Assumed in kernel
    const dim3 block_dim(BLOCK_DIM, BLOCK_DIM);
    dim3 grid_dim;
    if constexpr(BMM){
        grid_dim = dim3(
            (Y.size(2) + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
            (Y.size(1) + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
            Y.size(0)
        );
    }else{
        grid_dim = dim3(
            (Y.size(1) + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
            (Y.size(0) + OUT_TILE_DIM - 1) / OUT_TILE_DIM
        );
    }
    constexpr int MAT_DIM = 2 + BMM;

    int smem_size = (2 * 32 * 33) * sizeof(float);
    AT_DISPATCH_FLOATING_TYPES(A.type(), "matmul_kernel", ([&] {
        matmul_kernel<scalar_t, BLOCK_DIM, A_ROW_MAJOR, B_ROW_MAJOR, MUL, MAT_DIM>
        <<<grid_dim, block_dim, smem_size>>>(
            A.packed_accessor32<scalar_t,MAT_DIM,torch::RestrictPtrTraits>(),
            B.packed_accessor32<scalar_t,MAT_DIM,torch::RestrictPtrTraits>(),
            Y.packed_accessor32<scalar_t,MAT_DIM,torch::RestrictPtrTraits>(),
            offset
        );
    }));

    return Y;
}

template<float MUL(const float&, const float&, const int&), bool BMM>
torch::Tensor matmul_caller(
    torch::Tensor A,
    torch::Tensor B,
    bool A_ROW_MAJOR,
    bool B_ROW_MAJOR,
    float offset
){
    int offset_int = *(int*)&offset - 0x3f800000;
    switch(A_ROW_MAJOR*2 + B_ROW_MAJOR){
        case 0: return matmul_caller_body<false, false, MUL, BMM>(A, B, offset_int);
        case 1: return matmul_caller_body<false, true, MUL, BMM>(A, B, offset_int);
        case 2: return matmul_caller_body<true, false, MUL, BMM>(A, B, offset_int);
        default: return matmul_caller_body<true, true, MUL, BMM>(A, B, offset_int);
    }
}

template torch::Tensor matmul_caller<&pam, false>(torch::Tensor, torch::Tensor, bool, bool, float);
template torch::Tensor matmul_caller<&pam, true>(torch::Tensor, torch::Tensor, bool, bool, float);
template torch::Tensor matmul_caller<&standard_mul, false>(torch::Tensor, torch::Tensor, bool, bool, float);
template torch::Tensor matmul_caller<&standard_mul, true>(torch::Tensor, torch::Tensor, bool, bool, float);

///////////////////////////////////////////////////////////////////////////////////////////////////
// Backward PAM Matmul Kernel Based on the Matmul Kernel
///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename scalar_t, int BLOCK_DIM, bool ARM, bool BRM, bool DELTA_A, int MAT_DIM>
__global__ void pam_matmul_bwd_kernel(
    const torch::PackedTensorAccessor32<scalar_t,MAT_DIM,torch::RestrictPtrTraits> A,
    const torch::PackedTensorAccessor32<scalar_t,MAT_DIM,torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor32<scalar_t,MAT_DIM,torch::RestrictPtrTraits> C,
    const torch::PackedTensorAccessor32<scalar_t,MAT_DIM,torch::RestrictPtrTraits> D,
    int offset
){
    // Compute C = A @ B with each multiplication modified by D (identical to C in size)
    // The tensor names are kept from the fwd version and don't match the A,B,C used there
    // Assume A is DL x DI, B is DI x DR
    // Each block computes an area of 32x32 in C and has BLOCK_DIM**2 threads
    // ARM/BRM mean that A/B is stored as row-major
    // If false we assume column major but the same dimensions for the matrix

    constexpr bool BMM = (MAT_DIM == 3);
    
    extern __shared__ float s[];
    float* A_tile = &s[0];
    float* B_tile = &s[33*32];

    const int TTD = 32 / BLOCK_DIM; // Thread tile dimensions
    float C_array[TTD*TTD] = {0.0f}; // Requires {0.0f} to be zero initialized
    float D_array[TTD*TTD] = {0.0f}; // NOTE: Differs from standard matmul
    float A_array[TTD];
    float B_array[TTD];
    
    const int BASE_OUT_ROW = blockIdx.y * 32;
    const int BASE_OUT_COL = blockIdx.x * 32;

    // Load in the D elements corresponding to each thread
    // vvvvv NOTE: Differs from standard matmul vvvvv
    #pragma unroll
    for(int a_idx=0; a_idx<TTD; ++a_idx){
        #pragma unroll
        for(int b_idx=0; b_idx<TTD; ++b_idx){
            int out_row = BASE_OUT_ROW + threadIdx.y + a_idx * BLOCK_DIM;
            int out_col = BASE_OUT_COL + threadIdx.x + b_idx * BLOCK_DIM;
            if(out_row < D.size(0+BMM) && out_col < D.size(1+BMM)){
                if constexpr(BMM){
                    D_array[a_idx*TTD + b_idx] = D[blockIdx.z][out_row][out_col];
                }else{
                    D_array[a_idx*TTD + b_idx] = D[out_row][out_col];
                }
            }
        }
    }
    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    for(int step_idx=0; step_idx<(A.size(1+BMM) + 32 - 1)/32; ++step_idx){
        load_tile<scalar_t, BLOCK_DIM, ARM, MAT_DIM>(A_tile, A, BASE_OUT_ROW, 32*step_idx);
        load_tile<scalar_t, BLOCK_DIM, BRM, MAT_DIM>(B_tile, B, 32*step_idx, BASE_OUT_COL);
        __syncthreads();
        for(int inner_idx=0; inner_idx<32; ++inner_idx){
            // Load A_array, B_array
            #pragma unroll
            for(int a_idx=0; a_idx<TTD; ++a_idx){
                A_array[a_idx] = A_tile[(threadIdx.y + BLOCK_DIM * a_idx) * 33 + inner_idx];
                B_array[a_idx] = B_tile[inner_idx * 33 + (threadIdx.x + BLOCK_DIM * a_idx)];
            }
            // Compute elements
            #pragma unroll
            for(int a_idx=0; a_idx<TTD; ++a_idx){
                #pragma unroll
                for(int b_idx=0; b_idx<TTD; ++b_idx){
                    // vvvvv NOTE: Differs from standard matmul vvvvv
                    if(DELTA_A){
                        C_array[a_idx*TTD + b_idx] += pam_bwd(
                            D_array[a_idx*TTD + b_idx],
                            B_array[b_idx],
                            A_array[a_idx], // Deltas go last
                            offset
                        );
                    } else {
                        C_array[a_idx*TTD + b_idx] += pam_bwd(
                            D_array[a_idx*TTD + b_idx],
                            A_array[a_idx],
                            B_array[b_idx], // Deltas go last
                            offset
                        );
                    }
                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for(int a_idx=0; a_idx<TTD; ++a_idx){
        #pragma unroll
        for(int b_idx=0; b_idx<TTD; ++b_idx){
            int out_row = BASE_OUT_ROW + threadIdx.y + a_idx * BLOCK_DIM;
            int out_col = BASE_OUT_COL + threadIdx.x + b_idx * BLOCK_DIM;
            if(out_row < C.size(0+BMM) && out_col < C.size(1+BMM)){
                if constexpr(BMM){
                    C[blockIdx.z][out_row][out_col] = C_array[a_idx*TTD + b_idx];
                }else{
                    C[out_row][out_col] = C_array[a_idx*TTD + b_idx];
                }
            }
        }
    }
}

template<bool L_ROW_MAJOR, bool R_ROW_MAJOR, bool DELTA_L, bool BMM>
torch::Tensor pam_matmul_bwd_caller_body(
    torch::Tensor L,
    torch::Tensor R,
    torch::Tensor M,
    int offset
){
    // Computes a modified version of dM = L @ R where products are modified by M
    // DELTA_L indicates that the derivative is computed w.r.t. L (instead of R) 
    CHECK_CUDA(M);
    CHECK_CUDA(R);
    CHECK_CUDA(L);

    assert(L.size(1+BMM) == R.size(0+BMM));
    assert(M.size(0+BMM) == L.size(0+BMM));
    assert(M.size(1+BMM) == R.size(1+BMM));

    auto dM_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor dM;
    if constexpr(BMM){
        assert(M.size(0) == R.size(0));
        assert(M.size(0) == L.size(0));
        dM = torch::empty({M.size(0), M.size(1), M.size(2)}, dM_options);
    }else{
        dM = torch::empty({M.size(0), M.size(1)}, dM_options);
    }

    const int BLOCK_DIM = 16;
    const int OUT_TILE_DIM = 32; // Assumed in kernel
    const dim3 block_dim(BLOCK_DIM, BLOCK_DIM);
    dim3 grid_dim;
    if constexpr(BMM){
        grid_dim = dim3(
            (dM.size(2) + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
            (dM.size(1) + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
            dM.size(0)
        );
    }else{
        grid_dim = dim3(
            (dM.size(1) + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
            (dM.size(0) + OUT_TILE_DIM - 1) / OUT_TILE_DIM
        );
    }
    constexpr int MAT_DIM = 2 + BMM;

    int smem_size = (2 * 32 * 33) * sizeof(float);
    AT_DISPATCH_FLOATING_TYPES(M.type(), "pam_matmul_bwd_kernel", ([&] {
        pam_matmul_bwd_kernel<scalar_t, BLOCK_DIM, L_ROW_MAJOR, R_ROW_MAJOR, DELTA_L, MAT_DIM>
        <<<grid_dim, block_dim, smem_size>>>(
            L.packed_accessor32<scalar_t,MAT_DIM,torch::RestrictPtrTraits>(),
            R.packed_accessor32<scalar_t,MAT_DIM,torch::RestrictPtrTraits>(),
            dM.packed_accessor32<scalar_t,MAT_DIM,torch::RestrictPtrTraits>(),
            M.packed_accessor32<scalar_t,MAT_DIM,torch::RestrictPtrTraits>(),
            offset
        );
    }));

    return dM;
}

template<bool BMM>
torch::Tensor pam_matmul_bwd_caller(
    torch::Tensor L,
    torch::Tensor R,
    torch::Tensor M,
    bool L_ROW_MAJOR,
    bool R_ROW_MAJOR,
    bool delta_L,
    float offset
){
    // Computes a modified version of dM = L @ R (mul modified by M)
    // PAM_BWD has to distinguish between L and R, it needs to know which one was
    // multiplied by M in the forward pass.
    // delta_L indicates that L is the delta of the output, i.e. wasn't used in fwd
    int offset_int = *(int*)&offset - 0x3f800000;
    if(delta_L){
        switch(L_ROW_MAJOR*2 + R_ROW_MAJOR){
            case 0: return pam_matmul_bwd_caller_body<false, false, true, BMM>(L, R, M, offset_int);
            case 1: return pam_matmul_bwd_caller_body<false, true, true, BMM>(L, R, M, offset_int);
            case 2: return pam_matmul_bwd_caller_body<true, false, true, BMM>(L, R, M, offset_int);
            default: return pam_matmul_bwd_caller_body<true, true, true, BMM>(L, R, M, offset_int);
        }
    }else{
        switch(L_ROW_MAJOR*2 + R_ROW_MAJOR){
            case 0: return pam_matmul_bwd_caller_body<false, false, false, BMM>(L, R, M, offset_int);
            case 1: return pam_matmul_bwd_caller_body<false, true, false, BMM>(L, R, M, offset_int);
            case 2: return pam_matmul_bwd_caller_body<true, false, false, BMM>(L, R, M, offset_int);
            default: return pam_matmul_bwd_caller_body<true, true, false, BMM>(L, R, M, offset_int);
        }
    }
}

template torch::Tensor pam_matmul_bwd_caller<false>(torch::Tensor, torch::Tensor, torch::Tensor, bool, bool, bool, float);
template torch::Tensor pam_matmul_bwd_caller<true>(torch::Tensor, torch::Tensor, torch::Tensor, bool, bool, bool, float);

///////////////////////////////////////////////////////////////////////////////////////////////////
// Python Bindings
///////////////////////////////////////////////////////////////////////////////////////////////////
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pam_fwd", &elementwise_pam_caller, "PAM Fwd");
    m.def("pam_bwd", &elementwise_pam_bwd_caller, "PAM Bwd");
    m.def("pad_fwd", &elementwise_pad_caller, "PAD Fwd");
    m.def("pad_bwd", &elementwise_pad_bwd_caller, "PAD Bwd");
    m.def("pam_matmul_fwd", &matmul_caller<&pam, false>, "PAM Matmul Fwd");
    m.def("pam_bmm_fwd", &matmul_caller<&pam, true>, "PAM BMM Fwd");
    m.def("pam_matmul_bwd", &pam_matmul_bwd_caller<false>, "PAM Matmul Bwd");
    m.def("pam_bmm_bwd", &pam_matmul_bwd_caller<true>, "PAM BMM Bwd");
    m.def("standard_matmul", &matmul_caller<&standard_mul, false>, "Standard Matmul");
    m.def("standard_bmm", &matmul_caller<&standard_mul, true>, "Standard BMM");
    m.def("pa_exp_fwd", &elementwise_pa_exp_caller, "PA Exp Fwd");
    m.def("pa_exp_bwd", &elementwise_pa_exp_bwd_caller, "PA Exp Bwd");
    m.def("pa_log_fwd", &elementwise_pa_log_caller, "PA Log Fwd");
    m.def("pa_log_bwd", &elementwise_pa_log_bwd_caller, "PA Log Bwd");
}
