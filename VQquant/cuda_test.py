import torch
import VectorQuant
import sys
import os
from VQquant.smooth_vq import SmoothVQ

# -------------------------------
# CPU 参考实现
# -------------------------------
def dequant_cpu(w_indices, w_codebook, n_groups):
    # w_codebook: [in_, c]
    # w_indices:  [n_groups, out]

    in_, c = w_codebook.shape
    n_groups_, out = w_indices.shape
    assert n_groups_ == n_groups

    width = in_ // n_groups
    w_dq = torch.empty((in_, out), dtype=w_codebook.dtype)

    for g in range(n_groups):
        for i in range(width):
            row = g * width + i
            for j in range(out):
                idx = int(w_indices[g, j])
                w_dq[row, j] = w_codebook[row, idx]

    return w_dq

class layer:
    def __init__(self, weight):
        self.weight = weight
# -------------------------------
# Python 测试
# -------------------------------
def test_dequant_cuda():
    torch.manual_seed(20)

    # 测试参数
    in_ = 4096
    out = 4096
    n_groups = 1024
    codebook_width = 256
    n = 4  # 每线程处理列数
    width = in_ // n_groups
    # 随机生成 codebook 和索引
    w = torch.rand((out, in_), dtype=torch.float16, device='cuda')
    # w_codebook = torch.rand((in_, codebook_width), dtype=torch.float16, device='cuda')
    # w_indices  = torch.randint(0, codebook_width, (n_groups, out), dtype=torch.uint8, device='cuda')
    w_dq      = torch.zeros((in_, out), dtype=torch.float16, device='cuda')
    # debug = torch.zeros((width, codebook_width), dtype=torch.float16, device='cuda')
    smoothvq = SmoothVQ(w)
    w_indices, w_codebook = smoothvq.fasterquant()
    # 调用 CUDA kernel
    VectorQuant.dequant_forward(w_indices, w_codebook, w_dq, 4)
    # print(w_dq)

    # CPU 对比
    # w_dq_ref = dequant_cpu(w_indices.cpu(), w_codebook.cpu(), n_groups)
    # print(w_dq_ref)
    # 计算最大误差
    max_err = (w_dq.t().cpu() - w.cpu()).abs().max().item()
    print("Max error between CUDA and CPU:", max_err)

    assert max_err < 1e-5
    print("Test passed!")

# -------------------------------
# 调用
# -------------------------------
# 假设你已经通过 cpp_extension 编译了 dequant_cuda_launcher
# 比如：
# from my_extension import dequant_cuda_launcher
# test_dequant_cuda(dequant_cuda_launcher)
test_dequant_cuda()