#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

template <typename scalar_t>
__global__ void quantize_kernel(
    const scalar_t* __restrict__ t_fp,
    int8_t* __restrict__ t_q,
    scalar_t* __restrict__ delta_base,
    int8_t* __restrict__ e,
    int B, int T, int H, int G,
    int n_bits, int residual_bits)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * T * G;  // 每个 thread 处理一个 group

    if (idx >= total) return;

    int batch = idx / (T * G);
    int token = (idx / G) % T;
    int group = idx % G;
    int residual_group = H / G;

    const scalar_t* t_ptr = t_fp + batch * T * H + token * H + group * residual_group;
    int8_t* q_ptr = t_q + batch * T * H + token * H + group * residual_group;

    scalar_t absmax = 1e-5f;
    for(int i = 0; i < residual_group; i++) {
        scalar_t v = fabs(t_ptr[i]);
        absmax=max(absmax, v);
    }

    int q_max = (1 << (n_bits - 1)) - 1;
    scalar_t delta = absmax / q_max;

    extern __shared__ float  s_delta_f[];
    scalar_t* s_delta = reinterpret_cast<scalar_t*>(s_delta_f);
    s_delta[group] = delta;
    __syncthreads();

    scalar_t delta_b;

    if (group == 0) {
        delta_b = s_delta[0];
        for (int g = 1; g < G; g++)
            delta_b = fmax(delta_b, s_delta[g]);

        delta_base[batch * T + token] = delta_b;
    }
    __syncthreads();

    delta_b = delta_base[batch * T + token];
    scalar_t R = delta_b / (1 << residual_bits);

    int8_t e_val = (int8_t)roundf((delta - delta_b) / R);

    // clamp 到 [-2^{r-1}, 0]
    int8_t e_min = -(1 << (residual_bits - 1));
    e_val = max(e_val, e_min);
    e_val = min(e_val, (int8_t)0);

    e[(batch * T + token) * G + group] = e_val;

    scalar_t delta_rec = delta_b + e_val * R;
    
    delta_rec = fmax(delta_rec, (scalar_t)1e-5);

    for (int i = 0; i < residual_group; i++) {
        int q = (int)roundf(t_ptr[i] / delta_rec);
        q = max(q, -q_max);
        q = min(q,  q_max);
        q_ptr[i] = (int8_t)q;
    }
}

void quant_cuda_launcher(
    torch::Tensor t_fp,
    torch::Tensor t_q,
    torch::Tensor delta_base,
    torch::Tensor e,
    int n_bits,
    int residual_bits
) {
    const int B = t_fp.size(0);
    const int T = t_fp.size(1);
    const int H = t_fp.size(2);
    const int G = e.size(2);

    const int blocks = B * T;
    const int threads = G;
    const int shared_mem = G * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(t_fp.scalar_type(), "quant_cuda", ([&] {
        quantize_kernel<scalar_t><<<blocks, threads, shared_mem>>>(
            t_fp.data_ptr<scalar_t>(),
            t_q.data_ptr<int8_t>(),
            delta_base.data_ptr<scalar_t>(),
            e.data_ptr<int8_t>(),
            B, T, H, G,
            n_bits, residual_bits
        );
    }));
}

