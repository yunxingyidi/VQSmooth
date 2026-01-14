#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

template <typename scalar_t>
__global__ void dequantize_kernel(
    const int8_t* __restrict__ t_q,
    const scalar_t* __restrict__ delta_base,
    const int8_t* __restrict__ e,
    scalar_t* __restrict__ t_fp,
    int B, int T, int H, int G,
    int residual_bits
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= B * T * G) return;

    int batch = idx / (T * G);
    int token = (idx / G) % T;
    int group = idx % G;
    int residual_group = H / G;

    const int8_t* t_q_ptr = t_q + batch * T * H + token * H + group * residual_group;
    const scalar_t* delta_base_ptr = delta_base + batch * T + token;
    const int8_t* e_ptr = e + batch * T * G + token * G + group;
    scalar_t* t_fp_ptr = t_fp + batch * T * H + token * H + group * residual_group;

    scalar_t delta_b = *delta_base_ptr;
    scalar_t e_g = static_cast<scalar_t>(*e_ptr);
    scalar_t R = delta_b / (1 << residual_bits);
    scalar_t delta_rec = fmaxf(delta_b + (*e_ptr) * R, 1e-5f);
    
    for(int i = 0; i < residual_group; i++) {
        t_fp_ptr[i] = t_q_ptr[i] * delta_rec;
    }
}

void dequant_cuda_launcher(
    torch::Tensor t_q,
    torch::Tensor delta_base,
    torch::Tensor e,
    torch::Tensor t_fp,
    int residual_bits
) {
    const int B = t_fp.size(0);
    const int T = t_fp.size(1);
    const int H = t_fp.size(2);
    const int G = e.size(2);

    const int blocks = B * T;
    const int threads = G;
    const int shared_mem = G * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        t_fp.scalar_type(), "GroupQuant", ([&] {
        dequantize_kernel<scalar_t><<<blocks, threads, shared_mem>>>(
            t_q.data_ptr<int8_t>(),
            delta_base.data_ptr<scalar_t>(),
            e.data_ptr<int8_t>(),
            t_fp.data_ptr<scalar_t>(),
            B, T, H, G,
            residual_bits
        );
    }));
}

