#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

template <typename scalar_t>
__device__ __forceinline__ void block_copy(
    scalar_t* __restrict__ dst,
    const scalar_t* __restrict__ src,
    int n_elem
) {
    int tid = threadIdx.x;
    int stride = blockDim.x;

    for (int i = tid; i < n_elem; i += stride) {
        dst[i] = src[i];
    }

    __syncthreads(); // 确保所有线程写完 dst
}

template <typename scalar_t>
__global__ void dequantize_kernel(
    const u_int8_t* __restrict__ w_indices,
    const scalar_t* __restrict__ w_codebook,
    scalar_t* __restrict__ w_dq,
    int in, int out, int n_groups, int c, int n
) {
    int width = in / n_groups;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = out * n_groups;
    if(idx >= total) return;

    int codebook_id = blockIdx.x;
    int entry_id = threadIdx.x;

    extern __shared__ char smem[];

    const scalar_t* codebook_pts = w_codebook + c * width * codebook_id;
    const u_int8_t* indices_pts = w_indices + out * codebook_id;

    scalar_t* block_codebook = reinterpret_cast<scalar_t*>(smem);
    uint8_t* block_indices = reinterpret_cast<uint8_t*>(
        smem + sizeof(scalar_t) * c * width
    );
    scalar_t* w_dq_pts = w_dq + out * width * codebook_id;

    block_copy(block_codebook, codebook_pts, c * width);
    block_copy(block_indices, indices_pts, out);
    
    for(int i = 0; i < n; i++) {
    int index = static_cast<u_int8_t>(block_indices[entry_id * n + i]);
        for(int j = 0; j < width; j++) {
            w_dq_pts[entry_id * n + i + j * out] = block_codebook[index + j * c];
        }
    }
}

void dequant_cuda_launcher(
    torch::Tensor w_indices,
    torch::Tensor w_codebook,
    torch::Tensor w_dq,
    int n
) {
    const int in = w_codebook.size(0);
    const int out = w_indices.size(1);
    const int c = w_codebook.size(1);
    const int n_groups = w_indices.size(0);

    const int blocks = n_groups;
    const int threads = out / n;
    size_t shared_mem = w_codebook.element_size() * c * (in / n_groups) + sizeof(u_int8_t) * out;

    AT_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half,
    at::ScalarType::BFloat16,
    w_codebook.scalar_type(), "VectorQuant", ([&] {
        dequantize_kernel<scalar_t><<<blocks, threads, shared_mem>>>(
            w_indices.data_ptr<u_int8_t>(),
            w_codebook.data_ptr<scalar_t>(),
            w_dq.data_ptr<scalar_t>(),
            in, out, n_groups, c, n);
    }));
}

