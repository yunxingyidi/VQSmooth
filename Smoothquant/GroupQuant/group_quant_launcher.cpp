#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>

void quant_cuda_launcher(
    at::Tensor t_fp,
    at::Tensor t_q,
    at::Tensor delta_base,
    at::Tensor e,
    int n_bits,
    int residual_bits
);


void quant_forward(
    torch::Tensor t_fp,
    torch::Tensor t_q,
    torch::Tensor delta_base,
    torch::Tensor e,
    int n_bits,
    int residual_bits
) {
    quant_cuda_launcher(
        t_fp, t_q, delta_base, e, n_bits, residual_bits
    );
}

PYBIND11_MODULE(quant_cuda, m) {
    m.def("forward", &quant_forward, "Quantize CUDA");
}
