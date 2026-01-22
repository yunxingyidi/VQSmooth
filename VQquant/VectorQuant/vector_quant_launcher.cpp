#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>


void dequant_cuda_launcher(
    torch::Tensor w_indices,
    torch::Tensor w_codebook,
    torch::Tensor w_dq,
    int n
);


void dequant_forward(
    torch::Tensor w_indices,
    torch::Tensor w_codebook,
    torch::Tensor w_dq,
    int n
) {
    dequant_cuda_launcher(
        w_indices, w_codebook, w_dq, n
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dequant_forward", &dequant_forward, "Dequantize CUDA");
}
